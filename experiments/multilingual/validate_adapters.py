"""Compare our FP32 pooling/projection adapters with Sentence Transformers."""

import argparse
import json
from pathlib import Path

import torch
from evaluate import encode
from expanded import SPECS, prefixes
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="bge,e5en,minilm,granite107,horizon")
    args = p.parse_args()
    torch.set_num_threads(2)
    rows = []
    for key in args.models.split(","):
        spec = SPECS[key]
        path = Path(
            snapshot_download(
                spec["id"],
                revision=spec["revision"],
                allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"],
            )
        )
        qp, dp = prefixes(spec)
        texts = [
            qp + "How do plants store energy?",
            dp + "Plants use photosynthesis to store energy as sugars.",
            qp + "文書検索では何を比較しますか？",
            dp + "文書検索では質問と文章の意味をベクトルとして比較します。",
        ]
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModel.from_pretrained(
            path, dtype=torch.float32, attn_implementation="sdpa"
        ).eval()
        if spec.get("projection"):
            c = json.loads((path / spec["projection"] / "config.json").read_text())
            dense = torch.nn.Linear(c["in_features"], c["out_features"], bias=c["bias"])
            w = load_file(path / spec["projection"] / "model.safetensors")
            dense.weight.data.copy_(w["linear.weight"])
            model.add_module("sentence_projection", dense.eval())
        ours, _ = encode(model, tokenizer, texts, spec["pooling"], 256)
        del model
        reference = SentenceTransformer(
            str(path),
            device="cpu",
            model_kwargs={"dtype": torch.float32, "attn_implementation": "sdpa"},
        )
        reference.max_seq_length = 256
        theirs = reference.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        assert ours.shape == theirs.shape
        err = float(abs(ours - theirs).max())
        similarity = (ours * theirs).sum(1)
        assert err < 1e-5, (key, err)
        rows.append(
            {
                "model": key,
                "shape": list(ours.shape),
                "max_abs_difference": err,
                "min_cosine": float(similarity.min()),
            }
        )
        print(rows[-1], flush=True)
        del reference
    output = ROOT / "benchmarks/multilingual-expanded/adapters.json"
    existing = (
        {r["model"]: r for r in json.loads(output.read_text())}
        if output.exists()
        else {}
    )
    existing.update({r["model"]: r for r in rows})
    output.write_text(json.dumps(list(existing.values()), indent=2))


if __name__ == "__main__":
    main()
