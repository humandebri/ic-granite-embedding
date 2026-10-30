"""Aligned six-language retrieval and cross-language retrieval at 128 tokens."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from evaluate import W8A8, metrics
from expanded import SPECS, chunks, prefixes
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".cache/multilingual-global"
OUT = ROOT / "benchmarks/multilingual-global"
LANGUAGES = ["en", "de", "es", "fr", "ar", "ko"]
TASKS = ["NanoSciFact", "NanoFiQA2018"]
DATA_REVISION = "8a4be55eb80b3ed4d2e9a423a5212228c217d426"
MODELS = {k: SPECS[k] for k in ["english", "granite", "bekko8", "bekko25", "e5"]}
MODELS["gemma2"] = {
    "id": "google/embeddinggemma-2",
    "revision": "21ad7b3a498496f5c3e5de0d9afecd706e172a04",
    "pooling": "mean",
    "query_prefix": "task: search result | query: ",
    "document_prefix": "title: none | text: ",
    "text_only": True,
}
MODELS["harrier"] = {
    "id": "microsoft/harrier-oss-v1-270m",
    "revision": "31de22b673913c7d658c0f03f792d77c2dcf8ebd",
    "pooling": "last",
    "query_prefix": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset(task, lang, negatives=128):
    tables = {}
    for split in ["queries", "corpus", "qrels"]:
        path = CACHE / "data" / f"{task}_{split}_{lang}.parquet"
        tables[split] = pq.read_table(path).to_pylist()
    assert len({r["_id"] for r in tables["queries"]}) == len(tables["queries"])
    assert len({r["_id"] for r in tables["corpus"]}) == len(tables["corpus"])
    qs = sorted(tables["queries"], key=lambda r: r["_id"])
    ds = sorted(tables["corpus"], key=lambda r: r["_id"])
    rel = {q["_id"]: set() for q in qs}
    valid = {d["_id"] for d in ds}
    for row in tables["qrels"]:
        assert row["query-id"] in rel and row["corpus-id"] in valid
        rel[row["query-id"]].add(row["corpus-id"])
    assert all(rel.values())
    positives = set().union(*rel.values())
    candidates = [d for d in ds if d["_id"] not in positives]
    rng = np.random.default_rng(20261028)
    keep = positives | {
        candidates[i]["_id"]
        for i in rng.choice(len(candidates), negatives, replace=False)
    }
    return qs, [d for d in ds if d["_id"] in keep], rel


def select():
    result = {}
    OUT.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        canonical = None
        sources = []
        for lang in LANGUAGES:
            qs, ds, rel = dataset(task, lang)
            selection = {
                "queries": [q["_id"] for q in qs],
                "documents": [d["_id"] for d in ds],
                "qrels": {k: sorted(v) for k, v in rel.items()},
            }
            if canonical is None:
                canonical = selection
            else:
                assert canonical == selection, (task, lang, "translations not aligned")
            result[(task, lang)] = qs, ds, rel
            sources.extend(
                {
                    "file": f"{task}_{split}_{lang}.parquet",
                    "sha256": digest(CACHE / "data" / f"{task}_{split}_{lang}.parquet"),
                }
                for split in ["queries", "corpus", "qrels"]
            )
        (OUT / f"{task}-selection.json").write_text(
            json.dumps(
                {
                    "dataset": "LiquidAI/nanobeir-multilingual-extended",
                    "revision": DATA_REVISION,
                    "seed": 20261028,
                    "random_distractors": 128,
                    **canonical,
                    "sources": sources,
                },
                indent=2,
            )
        )
    return result


def model_path(key):
    spec = MODELS[key]
    if key == "gemma2":
        return (
            ROOT
            / f".cache/embeddinggemma2/hf/hub/models--google--embeddinggemma-2/snapshots/{spec['revision']}"
        )
    if key == "harrier":
        return (
            CACHE
            / f"models/models--microsoft--harrier-oss-v1-270m/snapshots/{spec['revision']}"
        )
    return Path(
        snapshot_download(
            spec["id"],
            revision=spec["revision"],
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"],
            local_files_only=True,
        )
    )


def load_model(key):
    path = model_path(key)
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    if key == "gemma2":
        config.vision_config = config.audio_config = None
    model = AutoModel.from_pretrained(
        path,
        config=config,
        dtype=torch.float32,
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.padding_side = "right"
    return model, tokenizer


def pool(hidden, mask, kind):
    if kind == "cls":
        return hidden[:, 0]
    if kind == "last":
        # All experiment tokenizers are explicitly right padded.
        return hidden[torch.arange(len(hidden), device=hidden.device), mask.sum(1) - 1]
    return (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)


def encode(model, tokenizer, texts, pooling, limit, batch):
    vectors = []
    lengths = []
    start = time.perf_counter()
    for at in range(0, len(texts), batch):
        group = texts[at : at + batch]
        lengths.extend(map(len, tokenizer(group, truncation=False)["input_ids"]))
        enc = tokenizer(group, padding=True, truncation=False, return_tensors="pt")
        assert enc["input_ids"].shape[1] <= limit
        enc = {k: v.to(next(model.parameters()).device) for k, v in enc.items()}
        with torch.inference_mode():
            h = model(**enc).last_hidden_state.float()
            v = pool(h, enc["attention_mask"], pooling)
            vectors.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
    return np.concatenate(vectors), {
        "seconds": time.perf_counter() - start,
        "tokens_max": max(lengths),
        "tokens_p50": float(np.median(lengths)),
        "truncated": 0,
        "device": str(next(model.parameters()).device),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--models", default="bekko8,bekko25,granite,e5,gemma2,harrier,english"
    )
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--limit", type=int, default=128)
    p.add_argument("--modes", default="fp32,w8a8-bf16kv")
    p.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    datasets = select()
    for key in args.models.split(","):
        spec = MODELS[key]
        print("load", key, flush=True)
        model, tokenizer = load_model(key)
        model.to(args.device)
        info = {
            "model": spec,
            "parameters": sum(p.numel() for p in model.parameters()),
            "embedding_parameters": model.get_input_embeddings().weight.numel(),
            "config": model.config.to_dict(),
            "environment": {
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "threads": args.threads,
            },
        }
        (OUT / f"{key}-model.json").write_text(json.dumps(info, indent=2))
        texts = {}
        for (task, lang), (qs, ds, rel) in datasets.items():
            if key == "english" and lang != "en":
                continue
            qp, dp = prefixes(spec)
            cs, parents = chunks(tokenizer, ds, args.limit, dp)
            all_texts = [qp + q["text"] for q in qs] + cs
            text_sha = hashlib.sha256(
                json.dumps(all_texts, ensure_ascii=False).encode()
            ).hexdigest()
            texts[(task, lang)] = all_texts, parents, text_sha
        original = torch.nn.functional.scaled_dot_product_attention
        try:
            for mode in args.modes.split(","):
                if mode == "w8a8-bf16kv":
                    for name, module in list(model.named_modules()):
                        if isinstance(module, torch.nn.Linear):
                            parent, _, leaf = name.rpartition(".")
                            setattr(model.get_submodule(parent), leaf, W8A8(module))

                    def sdpa(q, k, v, *a, _original=original, **kw):
                        return _original(
                            q, k.bfloat16().float(), v.bfloat16().float(), *a, **kw
                        )

                    torch.nn.functional.scaled_dot_product_attention = sdpa
                for (task, lang), (all_texts, parents, text_sha) in texts.items():
                    qs, ds, rel = datasets[(task, lang)]
                    stem = f"{task}-{lang}-{key}-{args.limit}-{mode}"
                    cached = CACHE / "vectors" / f"{stem}.npz"
                    cached.parent.mkdir(exist_ok=True)
                    print("encode", stem, len(all_texts), flush=True)
                    prior_report = OUT / f"{stem}.json"
                    cache_valid = False
                    if cached.exists() and prior_report.exists():
                        proof = json.loads(prior_report.read_text())
                        cache_valid = proof.get("model") == spec and proof.get(
                            "vectors_sha256"
                        ) == digest(cached)
                    if cache_valid:
                        archive = np.load(cached)
                        assert str(archive["text_sha256"]) == text_sha
                        assert np.array_equal(archive["parents"], parents)
                        vectors = archive["vectors"]
                        stats = json.loads(str(archive["stats"]))
                        stats.setdefault("device", "cpu")
                    else:
                        vectors, stats = encode(
                            model,
                            tokenizer,
                            all_texts,
                            spec["pooling"],
                            args.limit,
                            batch=32
                            if args.device == "mps"
                            else (4 if key in ["gemma2", "harrier"] else 8),
                        )
                        np.savez_compressed(
                            cached,
                            vectors=vectors,
                            parents=parents,
                            stats=json.dumps(stats),
                            text_sha256=text_sha,
                        )
                    assert np.isfinite(vectors).all()
                    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
                    score, ranks = metrics(
                        vectors[: len(qs)], vectors[len(qs) :], qs, ds, rel, parents
                    )
                    report = {
                        "model": spec,
                        "task": task,
                        "language": lang,
                        "precision": mode,
                        "queries": len(qs),
                        "documents": len(ds),
                        "chunks": len(parents),
                        "token_limit": args.limit,
                        "overlap_tokens": 32,
                        "selection_sha256": digest(OUT / f"{task}-selection.json"),
                        "text_sha256": text_sha,
                        "vectors_sha256": digest(cached),
                        "metrics": score,
                        "ranks": ranks,
                        "encoding": stats,
                    }
                    (OUT / f"{stem}.json").write_text(json.dumps(report, indent=2))
                    print("result", stem, score, flush=True)
        finally:
            torch.nn.functional.scaled_dot_product_attention = original
        del model


if __name__ == "__main__":
    main()
