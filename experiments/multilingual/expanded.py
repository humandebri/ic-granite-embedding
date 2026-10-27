"""Expanded EN/JA comparisons with shared fixed candidate pools and proper model heads."""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from evaluate import W8A8, encode, metrics
from evaluate import dataset as japanese_dataset
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
SPECS = json.loads(Path(__file__).with_name("expanded-models.json").read_text())
DATA = {
    "scifact": {
        "repo": "BeIR/scifact",
        "revision": "b3b5335604bf5ee3c4447671af975ea25143d4f5",
        "qrels": "BeIR/scifact-qrels",
        "qrels_revision": "2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029",
    },
    "fiqa": {
        "repo": "BeIR/fiqa",
        "revision": "979c07a7cb5ccc6ca009792241fa1250b98055dd",
        "qrels": "BeIR/fiqa-qrels",
        "qrels_revision": "252958f2d646e22cab6d0c72dd3f0d5de6d0655a",
    },
    "jaquad": {
        "repo": "mteb/JaQuADRetrieval",
        "revision": "5713cd1b76a22b79c3595827c717c72b6cab8852",
        "filter": "text length >=80 Unicode characters",
    },
}


def load_dataset(key, n, negatives):
    if key == "jaquad":
        return japanese_dataset(n, negatives)
    spec = DATA[key]
    tables = {}
    for name in ["corpus", "queries"]:
        p = hf_hub_download(
            spec["repo"],
            f"{name}/{name}-00000-of-00001.parquet",
            revision=spec["revision"],
            repo_type="dataset",
        )
        tables[name] = pq.read_table(p).to_pylist()
        for row in tables[name]:
            row["_id"] = str(row["_id"])
    p = hf_hub_download(
        spec["qrels"], "test.tsv", revision=spec["qrels_revision"], repo_type="dataset"
    )
    with open(p) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert {int(r["score"]) for r in rows} <= {0, 1}, (
        "metrics currently support binary relevance only"
    )
    valid = {d["_id"] for d in tables["corpus"]}
    rel = {}
    for r in rows:
        if int(r["score"]) > 0 and r["corpus-id"] in valid:
            rel.setdefault(r["query-id"], set()).add(r["corpus-id"])
    available = [q for q in tables["queries"] if q["_id"] in rel]
    rng = np.random.default_rng(20261027)
    queries = [
        available[i]
        for i in sorted(
            rng.choice(len(available), min(n, len(available)), replace=False)
        )
    ]
    rel = {q["_id"]: rel[q["_id"]] for q in queries}
    positives = set().union(*rel.values())
    other = [d for d in tables["corpus"] if d["_id"] not in positives]
    chosen = positives | {
        other[i]["_id"]
        for i in rng.choice(len(other), min(negatives, len(other)), replace=False)
    }
    return queries, [d for d in tables["corpus"] if d["_id"] in chosen], rel


def prefixes(spec):
    return (
        spec.get("query_prefix", "query: " if spec.get("prefix") else ""),
        spec.get("document_prefix", "passage: " if spec.get("prefix") else ""),
    )


def chunks(tokenizer, corpus, limit, prefix):
    budget = (
        limit
        - tokenizer.num_special_tokens_to_add()
        - len(tokenizer.encode(prefix, add_special_tokens=False))
    )
    assert budget > 32
    texts = []
    parents = []
    for parent, doc in enumerate(corpus):
        text = (doc.get("title", "") + "\n" + doc["text"]).strip()
        offsets = tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True
        )["offset_mapping"]
        if not offsets:
            texts.append(prefix)
            parents.append(parent)
            continue
        start = 0
        while start < len(offsets):
            end = min(start + budget, len(offsets))
            piece = text[offsets[start][0] : offsets[end - 1][1]]
            while len(tokenizer.encode(prefix + piece)) > limit:
                end -= 1
                assert end > start
                piece = text[offsets[start][0] : offsets[end - 1][1]]
            texts.append(prefix + piece)
            parents.append(parent)
            if end == len(offsets):
                break
            start = max(start + 1, end - 32)
    return texts, parents


def bm25(queries, corpus, rel):
    docs = [
        re.findall(r"[a-z0-9]+", (d.get("title", "") + " " + d["text"]).lower())
        for d in corpus
    ]
    counts = [Counter(d) for d in docs]
    df = Counter(w for d in docs for w in set(d))
    lengths = np.array(list(map(len, docs)))
    avg = lengths.mean()
    scores = np.zeros((len(queries), len(corpus)))
    for i, q in enumerate(queries):
        for term in set(re.findall(r"[a-z0-9]+", q["text"].lower())):
            freq = np.array([c[term] for c in counts])
            idf = np.log(1 + (len(docs) - df[term] + 0.5) / (df[term] + 0.5))
            scores[i] += idf * freq * 2.5 / (freq + 1.5 * (0.25 + 0.75 * lengths / avg))
    # Feeding the score matrix as query vectors and identity as document vectors reuses rank/metric logic.
    result, ranks = metrics(scores, np.eye(len(corpus)), queries, corpus, rel)
    return {
        "metrics": result,
        "ranks": ranks,
        "tokenization": "ASCII alphanumeric lowercase; whole documents; k1=1.5,b=0.75",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default=",".join(SPECS))
    p.add_argument("--datasets", default="scifact,fiqa,jaquad")
    p.add_argument("--queries", type=int, default=128)
    p.add_argument("--negatives", type=int, default=512)
    p.add_argument("--limit", type=int, default=256)
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--reuse-fp32",
        action="store_true",
        help="Reuse matching saved FP32 vectors after interruption",
    )
    p.add_argument(
        "--output", type=Path, default=ROOT / "benchmarks/multilingual-expanded"
    )
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    datasets = {}
    for key in args.datasets.split(","):
        qs, ds, rel = load_dataset(key, args.queries, args.negatives)
        dest = args.output / key
        dest.mkdir(exist_ok=True)
        selection = {
            "dataset": DATA[key],
            "seed": 20261027,
            "queries": [q["_id"] for q in qs],
            "documents": [d["_id"] for d in ds],
            "qrels": {k: sorted(v) for k, v in rel.items()},
        }
        content = json.dumps(selection, indent=2)
        (dest / "selection.json").write_text(content)
        digest = hashlib.sha256(content.encode()).hexdigest()
        datasets[key] = (qs, ds, rel, digest)
        print("dataset", key, len(qs), len(ds), flush=True)
        if key != "jaquad":
            (dest / "bm25.json").write_text(json.dumps(bm25(qs, ds, rel), indent=2))
    for key in args.models.split(","):
        spec = SPECS[key]
        todo = []
        for data in datasets:
            path = args.output / data / f"{key}-{args.limit}.json"
            if args.resume and path.exists():
                old = json.loads(path.read_text())
                if (
                    old.get("model") == spec
                    and old.get("selection_sha256") == datasets[data][3]
                    and set(old.get("runs", {})) == {"fp32", "w8a8-bf16kv"}
                ):
                    continue
            todo.append(data)
        if not todo:
            continue
        print("loading", key, flush=True)
        path = Path(
            snapshot_download(
                spec["id"],
                revision=spec["revision"],
                allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"],
            )
        )
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModel.from_pretrained(
            path, dtype=torch.float32, attn_implementation="sdpa"
        ).eval()
        if spec.get("projection"):
            cfg = json.loads((path / spec["projection"] / "config.json").read_text())
            assert cfg["activation_function"].endswith(".Identity")
            dense = torch.nn.Linear(
                cfg["in_features"], cfg["out_features"], bias=cfg["bias"]
            )
            weights = load_file(path / spec["projection"] / "model.safetensors")
            dense.weight.data.copy_(weights["linear.weight"])
            if dense.bias is not None:
                dense.bias.data.copy_(weights["linear.bias"])
            model.add_module("sentence_projection", dense.eval())
        texts = {}
        reports = {}
        parents = {}
        previous = {}
        for data in todo:
            qs, ds, rel, digest = datasets[data]
            qp, dp = prefixes(spec)
            cs, ps = chunks(tokenizer, ds, args.limit, dp)
            texts[data] = [qp + q["text"] for q in qs] + cs
            parents[data] = ps
            reports[data] = {
                "model": spec,
                "dataset": DATA[data],
                "selection_sha256": digest,
                "queries": len(qs),
                "documents": len(ds),
                "chunks": len(cs),
                "token_limit": args.limit,
                "overlap_tokens": 32,
                "config": model.config.to_dict(),
                "parameters": sum(v.numel() for v in model.parameters()),
                "embedding_parameters": model.get_input_embeddings().weight.numel(),
                "environment": {
                    "torch": torch.__version__,
                    "transformers": __import__("transformers").__version__,
                    "threads": args.threads,
                },
                "runs": {},
            }
        if args.reuse_fp32:
            for data in todo:
                oldpath = args.output / data / f"{key}-{args.limit}.json"
                cached = (
                    ROOT
                    / f".cache/multilingual-expanded/{data}/{key}-{args.limit}-fp32.npz"
                )
                if not oldpath.exists() or not cached.exists():
                    continue
                old = json.loads(oldpath.read_text())
                if (
                    old.get("model") != spec
                    or old.get("selection_sha256") != datasets[data][3]
                    or "fp32" not in old.get("runs", {})
                ):
                    continue
                archive = np.load(cached)
                vectors = archive["vectors"]
                assert len(vectors) == len(texts[data])
                assert np.array_equal(archive["parents"], parents[data])
                previous[data] = vectors
                reports[data]["runs"]["fp32"] = old["runs"]["fp32"]
                reports[data]["fp32_reuse"] = {
                    "environment": old["environment"],
                    "vectors_sha256": hashlib.sha256(cached.read_bytes()).hexdigest(),
                }
                print("reused FP32", data, key, flush=True)
        original = torch.nn.functional.scaled_dot_product_attention
        try:
            for mode in ["fp32", "w8a8-bf16kv"]:
                if mode != "fp32":
                    for name, module in list(model.named_modules()):
                        if isinstance(module, torch.nn.Linear):
                            parent, _, leaf = name.rpartition(".")
                            setattr(model.get_submodule(parent), leaf, W8A8(module))

                    def sdpa(q, k, v, *a, _original=original, **kw):
                        return _original(
                            q, k.bfloat16().float(), v.bfloat16().float(), *a, **kw
                        )

                    torch.nn.functional.scaled_dot_product_attention = sdpa
                for data in todo:
                    if mode == "fp32" and data in previous:
                        continue
                    qs, ds, rel, digest = datasets[data]
                    vectors, stats = encode(
                        model, tokenizer, texts[data], spec["pooling"], args.limit
                    )
                    assert stats["truncated"] == 0, (data, key, stats)
                    score, ranks = metrics(
                        vectors[: len(qs)],
                        vectors[len(qs) :],
                        qs,
                        ds,
                        rel,
                        parents[data],
                    )
                    report = {
                        "metrics": score,
                        "ranks": ranks,
                        "encoding": stats,
                        "dimensions": vectors.shape[1],
                    }
                    if mode == "fp32":
                        previous[data] = vectors
                    else:
                        report["cosine_to_fp32"] = {
                            "min": float((vectors * previous[data]).sum(1).min()),
                            "mean": float((vectors * previous[data]).sum(1).mean()),
                        }
                    reports[data]["runs"][mode] = report
                    (args.output / data / f"{key}-{args.limit}.json").write_text(
                        json.dumps(reports[data], indent=2)
                    )
                    cache = ROOT / ".cache/multilingual-expanded" / data
                    cache.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        cache / f"{key}-{args.limit}-{mode}.npz",
                        vectors=vectors,
                        parents=np.array(parents[data]),
                    )
                    print(
                        data, key, mode, score, round(stats["seconds"], 2), flush=True
                    )
        finally:
            torch.nn.functional.scaled_dot_product_attention = original
        del model


if __name__ == "__main__":
    main()
