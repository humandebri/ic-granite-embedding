"""Pinned text-only EmbeddingGemma 2 comparison on the existing EN/JA pools."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from evaluate import W8A8, encode, metrics
from expanded import DATA, chunks, load_dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
REVISION = "21ad7b3a498496f5c3e5de0d9afecd706e172a04"
MODEL = "google/embeddinggemma-2"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--datasets", default="scifact,fiqa,jaquad")
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    output = ROOT / "benchmarks/embeddinggemma2"
    output.mkdir(parents=True, exist_ok=True)
    cache = ROOT / ".cache/embeddinggemma2/vectors"
    cache.mkdir(parents=True, exist_ok=True)
    assert args.snapshot.name == REVISION
    config = AutoConfig.from_pretrained(args.snapshot, local_files_only=True)
    config.vision_config = None
    config.audio_config = None
    model = AutoModel.from_pretrained(
        args.snapshot,
        config=config,
        dtype=torch.float32,
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    linears = [
        {"name": name, "rows": m.out_features, "cols": m.in_features}
        for name, m in model.named_modules()
        if isinstance(m, torch.nn.Linear)
    ]
    info = {
        "model": MODEL,
        "revision": REVISION,
        "text_only": True,
        "parameters": sum(p.numel() for p in model.parameters()),
        "embedding_parameters": model.get_input_embeddings().weight.numel(),
        "linear_parameters": sum(r["rows"] * r["cols"] for r in linears),
        "linears": linears,
        "config": config.to_dict(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "threads": args.threads,
        "pooling": "mean including prompts and special tokens, followed by L2",
        "query_prefix": "task: search result | query: ",
        "document_format": "title: none | text: {title}\\n{text}",
        "note": "Original titles kept in body to preserve the previous chunking protocol; title field fixed to none.",
    }
    (output / "model.json").write_text(json.dumps(info, indent=2))
    print("model", info["parameters"], "linear", info["linear_parameters"], flush=True)
    datasets = {}
    for key in args.datasets.split(","):
        qs, ds, rel = load_dataset(key, 128, 512)
        selection = json.loads(
            (
                ROOT / f"benchmarks/multilingual-expanded/{key}/selection.json"
            ).read_text()
        )
        assert selection["queries"] == [q["_id"] for q in qs]
        assert selection["documents"] == [d["_id"] for d in ds]
        assert selection["qrels"] == {k: sorted(v) for k, v in rel.items()}
        cs, parents = chunks(tokenizer, ds, 256, "title: none | text: ")
        texts = ["task: search result | query: " + q["text"] for q in qs] + cs
        datasets[key] = qs, ds, rel, texts, parents
    original = torch.nn.functional.scaled_dot_product_attention
    previous = {}
    try:
        for mode in ["fp32", "w8a8-bf16kv"]:
            if mode != "fp32":
                for name, module in list(model.named_modules()):
                    if isinstance(module, torch.nn.Linear):
                        parent, _, leaf = name.rpartition(".")
                        setattr(model.get_submodule(parent), leaf, W8A8(module))

                def sdpa(q, k, v, *a, **kw):
                    return original(
                        q, k.bfloat16().float(), v.bfloat16().float(), *a, **kw
                    )

                torch.nn.functional.scaled_dot_product_attention = sdpa
            for key, (qs, ds, rel, texts, parents) in datasets.items():
                saved = cache / f"{key}-{mode}.npz"
                print("encode", key, mode, len(texts), flush=True)
                if saved.exists():
                    data = np.load(saved)
                    assert np.array_equal(data["parents"], parents)
                    vectors = data["vectors"]
                    stats = json.loads(str(data["stats"]))
                else:
                    vectors, stats = encode(
                        model, tokenizer, texts, "mean", 256, batch=4
                    )
                    assert stats["truncated"] == 0, (key, stats)
                    np.savez_compressed(
                        saved, vectors=vectors, parents=parents, stats=json.dumps(stats)
                    )
                assert np.isfinite(vectors).all()
                assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
                score, ranks = metrics(
                    vectors[: len(qs)], vectors[len(qs) :], qs, ds, rel, parents
                )
                result = {
                    "model": MODEL,
                    "revision": REVISION,
                    "dataset": DATA[key],
                    "mode": mode,
                    "queries": len(qs),
                    "documents": len(ds),
                    "chunks": len(parents),
                    "token_limit": 256,
                    "overlap_tokens": 32,
                    "metrics": score,
                    "ranks": ranks,
                    "encoding": stats,
                    "vectors_sha256": hashlib.sha256(saved.read_bytes()).hexdigest(),
                    "selection_sha256": hashlib.sha256(
                        (
                            ROOT
                            / f"benchmarks/multilingual-expanded/{key}/selection.json"
                        ).read_bytes()
                    ).hexdigest(),
                }
                if mode == "fp32":
                    previous[key] = vectors
                else:
                    similarities = (vectors * previous[key]).sum(1)
                    result["cosine_to_fp32"] = {
                        "min": float(similarities.min()),
                        "mean": float(similarities.mean()),
                    }
                (output / f"{key}-{mode}.json").write_text(json.dumps(result, indent=2))
                print(key, mode, score, flush=True)
    finally:
        torch.nn.functional.scaled_dot_product_attention = original


if __name__ == "__main__":
    main()
