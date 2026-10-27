"""Export pinned Bekko weights, reference fixtures and a small real retrieval smoke set."""

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL = "hotchpotch/bekko-embedding-v1-a8m"
REV = "c721113d59a1d91b447450324f51c4b3332c924a"
SHA = "1feddfa11f8e232a30b39c4965328a396fa9388dc65bb06cad043045e2dc56ef"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ROOT / ".cache/bekko")
    p.add_argument("--reuse-pack", type=Path)
    p.add_argument("--retrieval", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(3)
    path = Path(
        snapshot_download(
            MODEL, revision=REV, allow_patterns=["*.json", "*.safetensors"]
        )
    )
    dest = args.output / "model.pack"
    if args.reuse_pack:
        import shutil

        shutil.copyfile(args.reuse_pack, dest)
    elif not dest.exists():
        data = bytearray()
        metadata = {}
        for name, tensor in load_file(path / "model.safetensors").items():
            a = tensor.float().numpy()
            offset = len(data)
            if a.ndim == 2 and "tok_embeddings" not in name:
                scale = np.maximum(abs(a).max(1) / 127, 1e-12).astype("<f4")
                data += (
                    np.clip(np.round(a / scale[:, None]), -127, 127)
                    .astype("i1")
                    .tobytes()
                    + scale.tobytes()
                )
                kind = "i8"
            else:
                data += a.astype("<f4").tobytes()
                kind = "f32"
            metadata[name] = {
                "shape": list(a.shape),
                "offset": offset,
                "bytes": len(data) - offset,
                "kind": kind,
            }
        header = json.dumps(metadata, separators=(",", ":")).encode()
        dest.write_bytes(struct.pack("<Q", len(header)) + header + data)
    assert hashlib.sha256(dest.read_bytes()).hexdigest() == SHA, (
        "unexpected pack; remove it and regenerate from the pinned model"
    )
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(
        path, dtype=torch.float32, attn_implementation="sdpa"
    ).eval()
    topics = {
        "ja": "文書を意味で検索するために埋め込みを使います。質問と文章を比較して関連する情報を探します。",
        "en": "The search service stores document embeddings and retrieves relevant passages by semantic similarity. Database backups run nightly.",
        "mixed": "Internet ComputerのcanisterでWebAssemblyを実行します。Query calls have an instruction limit. 文章を検索します。",
    }
    cases = []
    for topic, text in topics.items():
        for n in [2, 32, 64, 128, 256]:
            ids = tok((text + " ") * 100, max_length=n, truncation=True)["input_ids"]
            assert len(ids) == n
            cases.append({"case": f"{topic}-{n}", "ids": ids})
    retrieval = None
    if args.retrieval:
        # Reuse the established dataset selection/chunking rules; source revisions are pinned there.
        sys.path.insert(0, str(ROOT / "experiments/multilingual"))
        from expanded import DATA, chunks, load_dataset

        qs, docs, rel = load_dataset("jaquad", 32, 128)
        pieces, parents = chunks(tok, docs, 256, "")
        texts = [q["text"] for q in qs] + pieces
        inputs = [tok(t)["input_ids"] for t in texts]
        assert all(2 <= len(ids) <= 256 for ids in inputs)
        retrieval = {
            "dataset": DATA["jaquad"],
            "seed": 20261027,
            "query_ids": [q["_id"] for q in qs],
            "document_ids": [d["_id"] for d in docs],
            "qrels": {k: sorted(v) for k, v in rel.items()},
            "parents": parents,
            "inputs": inputs,
        }
    all_cases = cases + ([{"ids": v} for v in retrieval["inputs"]] if retrieval else [])

    def infer():
        vectors = []
        for c in all_cases:
            ids = torch.tensor([c["ids"]])
            with torch.inference_mode():
                v = torch.nn.functional.normalize(
                    model(input_ids=ids).last_hidden_state[0].float().mean(0), dim=0
                )
            vectors.append(v.tolist())
        return vectors

    fp = infer()
    # Same independent W8A8 reference as the comparison experiments.
    sys.path.insert(0, str(ROOT / "experiments/multilingual"))

    class W8A8(torch.nn.Module):
        def __init__(self, original):
            super().__init__()
            assert original.bias is None
            w = original.weight.detach().float()
            scale = w.abs().amax(-1).div(127).clamp_min(1e-12)
            self.register_buffer(
                "weight", (w / scale[:, None]).round().clamp(-127, 127)
            )
            self.register_buffer("scale", scale)

        def forward(self, x):
            scale = x.abs().amax(-1, keepdim=True).div(127).clamp_min(1e-12)
            return (
                torch.nn.functional.linear(
                    (x / scale).round().clamp(-127, 127), self.weight
                )
                * scale
                * self.scale
            )

    for name, m in list(model.named_modules()):
        if isinstance(m, torch.nn.Linear):
            parent, _, leaf = name.rpartition(".")
            setattr(model.get_submodule(parent), leaf, W8A8(m))
    original = torch.nn.functional.scaled_dot_product_attention

    def sdpa(q, k, v, *a, **kw):
        return original(q, k.bfloat16().float(), v.bfloat16().float(), *a, **kw)

    torch.nn.functional.scaled_dot_product_attention = sdpa
    quant = infer()
    torch.nn.functional.scaled_dot_product_attention = original
    for i, c in enumerate(cases):
        c.update(fp32=fp[i], w8a8=quant[i])
    (args.output / "cases.json").write_text(json.dumps(cases))
    if retrieval:
        retrieval["fp32"] = fp[len(cases) :]
        retrieval["w8a8"] = quant[len(cases) :]
        (args.output / "retrieval.json").write_text(json.dumps(retrieval))
    manifest = {
        "model": MODEL,
        "revision": REV,
        "pack_sha256": SHA,
        "pack_bytes": dest.stat().st_size,
        "cases_sha256": hashlib.sha256(
            (args.output / "cases.json").read_bytes()
        ).hexdigest(),
        "cases": len(cases),
    }
    if retrieval:
        manifest.update(
            retrieval_inputs=len(retrieval["inputs"]),
            retrieval_sha256=hashlib.sha256(
                (args.output / "retrieval.json").read_bytes()
            ).hexdigest(),
        )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(manifest)


if __name__ == "__main__":
    main()
