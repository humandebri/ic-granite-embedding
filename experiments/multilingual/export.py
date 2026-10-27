"""Export public pinned ModernBERT weights and exact-length FP32 fixtures."""

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]


def export_model(key):
    spec = json.loads(Path(__file__).with_name("models.json").read_text())[key]
    path = Path(
        snapshot_download(
            spec["id"],
            revision=spec["revision"],
            allow_patterns=["*.json", "*.safetensors"],
        )
    )
    dest = ROOT / ".cache/multilingual" / key
    dest.mkdir(parents=True, exist_ok=True)
    metadata = {}
    data = bytearray()
    for name, tensor in load_file(path / "model.safetensors").items():
        a = tensor.float().numpy()
        offset = len(data)
        if a.ndim == 2 and "tok_embeddings" not in name:
            s = np.maximum(abs(a).max(1) / 127, 1e-12).astype("<f4")
            q = np.clip(np.round(a / s[:, None]), -127, 127).astype("i1")
            data += q.tobytes() + s.tobytes()
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
    pack = struct.pack("<Q", len(header)) + header + data
    (dest / "model.pack").write_bytes(pack)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(
        path, dtype=torch.float32, attn_implementation="sdpa"
    ).eval()
    texts = {
        "ja": "文書を意味で検索するために埋め込みを使います。日本語の文章を小さな段落に分割して保存し、質問に関連する情報を検索します。暗号化されたバックアップから障害時にデータを復元できます。",
        "en": "A search service stores document embeddings and retrieves relevant passages by semantic similarity. Database backups run nightly and remain available for thirty days.",
        "mixed": "Internet ComputerのcanisterはWebAssemblyを実行します。Query calls have an instruction limit. モデルをINT8に量子化して計算資源を節約します。",
    }
    refs = []
    for topic, text in texts.items():
        for n in [128, 256, 304] + ([512] if key == "bekko8" else []):
            ids = tokenizer((text + " ") * 100, max_length=n, truncation=True)[
                "input_ids"
            ]
            assert len(ids) == n
            with torch.inference_mode():
                h = model(input_ids=torch.tensor([ids])).last_hidden_state[0]
                a = h[0] if spec["pooling"] == "cls" else h.mean(0)
                a = torch.nn.functional.normalize(a, dim=0).numpy()
            refs.append({"case": f"{topic}-{n}", "ids": ids, "fp32": a.tolist()})
    # Independent W8A8/BF16-KV reference for separating quantization loss from port errors.
    from evaluate import W8A8

    for name, module in list(model.named_modules()):
        if isinstance(module, torch.nn.Linear):
            parent, _, leaf = name.rpartition(".")
            setattr(model.get_submodule(parent), leaf, W8A8(module))
    original = torch.nn.functional.scaled_dot_product_attention

    def sdpa(q, k, v, *args, **kwargs):
        return original(q, k.bfloat16().float(), v.bfloat16().float(), *args, **kwargs)

    torch.nn.functional.scaled_dot_product_attention = sdpa
    try:
        for case in refs:
            with torch.inference_mode():
                h = model(input_ids=torch.tensor([case["ids"]])).last_hidden_state[0]
                a = h[0] if spec["pooling"] == "cls" else h.mean(0)
                case["w8a8"] = torch.nn.functional.normalize(a, dim=0).tolist()
    finally:
        torch.nn.functional.scaled_dot_product_attention = original
    (dest / "cases.json").write_text(json.dumps(refs))
    (ROOT / f"benchmarks/multilingual/{key}-pack.json").write_text(
        json.dumps(
            {
                "bytes": len(pack),
                "sha256": hashlib.sha256(pack).hexdigest(),
                "weights_sha256": hashlib.sha256(
                    (path / "model.safetensors").read_bytes()
                ).hexdigest(),
                "cases_sha256": hashlib.sha256(
                    (dest / "cases.json").read_bytes()
                ).hexdigest(),
                "model": spec,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model", choices=["granite", "bekko8"])
    args = p.parse_args()
    torch.set_num_threads(4)
    export_model(args.model)
