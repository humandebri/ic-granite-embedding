"""Evaluate actual answer-span recovery by chunk length on full JaQuAD validation."""

import argparse
import bisect
import gzip
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from evaluate import W8A8, encode
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
SOURCE_REV = "05600ff310a0970823e70f82f428893b85c71ffe"
MODEL = "hotchpotch/bekko-embedding-v1-a8m"
MODEL_REV = "c721113d59a1d91b447450324f51c4b3332c924a"
BOUNDARY = re.compile(r"[。！？][」』”’]*|\n+")


def load_source(cache):
    manifest = json.loads((cache / "source.json").read_text())
    assert manifest["revision"] == SOURCE_REV
    docs, queries, keys = [], [], {}
    for f in manifest["files"]:
        path = cache / Path(f["path"]).name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == f["sha256"]
        for article in json.loads(path.read_text())["data"]:
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]
                title = article["title"].strip()
                key = (title, context)
                if key not in keys:
                    keys[key] = len(docs)
                    docs.append({"title": title, "context": context})
                parent = keys[key]
                shift = len(title) + 1
                for qa in paragraph["qas"]:
                    spans, sentences = [], []
                    for answer in qa["answers"]:
                        start = answer["answer_start"]
                        end = start + len(answer["text"])
                        assert end > start and context[start:end] == answer["text"], qa[
                            "id"
                        ]
                        spans.append((shift + start, shift + end))
                        left, right = sentence_bounds(context, start, end)
                        sentences.append((shift + left, shift + right))
                    assert spans
                    queries.append(
                        {
                            "id": qa["id"],
                            "text": qa["question"],
                            "type": qa["question_type"],
                            "parent": parent,
                            "spans": spans,
                            "sentences": sentences,
                        }
                    )
    assert len(queries) == len({q["id"] for q in queries}) == 3939
    assert len(docs) == 1431
    return docs, queries, manifest


def sentence_bounds(text, start, end):
    boundaries = [0] + [m.end() for m in BOUNDARY.finditer(text)] + [len(text)]
    left = boundaries[bisect.bisect_right(boundaries, start) - 1]
    right = boundaries[bisect.bisect_left(boundaries, end)]
    return left, right


def make_chunks(tokenizer, docs, limit, strategy):
    chunks = []
    budget = limit - tokenizer.num_special_tokens_to_add()
    assert budget > 64
    for parent, doc in enumerate(docs):
        text = doc["title"] + "\n" + doc["context"]
        offsets = tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True
        )["offset_mapping"]
        ends = [b for a, b in offsets]
        boundaries = [m.end() for m in BOUNDARY.finditer(text)]
        at = 0
        while at < len(offsets):
            stop = min(at + budget, len(offsets))
            begin, finish = offsets[at][0], offsets[stop - 1][1]
            while len(tokenizer.encode(text[begin:finish])) > limit:
                stop -= 1
                assert stop > at
                finish = offsets[stop - 1][1]
            if strategy == "sentence_end" and stop < len(offsets):
                # Prefer a sentence/line ending in the latter half of this window.
                minimum = offsets[at + (stop - at) // 2][1]
                allowed = [b for b in boundaries if minimum <= b <= finish]
                if allowed:
                    finish = allowed[-1]
                    stop = bisect.bisect_right(ends, finish)
            piece = text[begin:finish]
            cost = len(tokenizer.encode(piece))
            assert 2 <= cost <= limit
            assert finish > begin
            chunks.append(
                {
                    "parent": parent,
                    "start": begin,
                    "end": finish,
                    "text": piece,
                    "tokens": cost,
                }
            )
            if stop == len(offsets):
                break
            at = max(at + 1, stop - 32)
    # Every non-whitespace source character must remain covered.
    for parent, doc in enumerate(docs):
        text = doc["title"] + "\n" + doc["context"]
        covered = np.zeros(len(text), dtype=bool)
        for chunk in chunks:
            if chunk["parent"] == parent:
                covered[chunk["start"] : chunk["end"]] = True
        assert all(covered[i] or ch.isspace() for i, ch in enumerate(text)), parent
    return chunks


def contains(chunk, parent, spans):
    return chunk["parent"] == parent and any(
        chunk["start"] <= a and b <= chunk["end"] for a, b in spans
    )


def measure(qvec, cvec, queries, chunks, context_budget=1024):
    similarities = qvec @ cvec.T
    order = np.argsort(-similarities, axis=1, kind="stable")
    parents = np.array([c["parent"] for c in chunks])
    costs = np.array([c["tokens"] for c in chunks])
    rows = []
    for i, query in enumerate(queries):
        ranked = order[i]
        gold = np.flatnonzero(parents == query["parent"])
        answer = {
            int(j) for j in gold if contains(chunks[j], query["parent"], query["spans"])
        }
        sentence = {
            int(j)
            for j in gold
            if contains(chunks[j], query["parent"], query["sentences"])
        }
        assert sentence <= answer
        top5 = ranked[:5]
        total = np.cumsum(costs[ranked])
        selected = ranked[: int(np.searchsorted(total, context_budget, side="right"))]
        distinct = list(dict.fromkeys(int(parents[j]) for j in ranked))[:5]
        parent5 = any(parents[j] == query["parent"] for j in top5)
        answer5 = any(int(j) in answer for j in top5)
        sentence5 = any(int(j) in sentence for j in top5)
        row = {
            "query_id": query["id"],
            "question_type": query["type"],
            "answer@1": int(int(ranked[0]) in answer),
            "answer@5": int(answer5),
            "sentence@1": int(int(ranked[0]) in sentence),
            "sentence@5": int(sentence5),
            "chunk_parent@5": int(parent5),
            "deduplicated_parent@5": int(query["parent"] in distinct),
            f"answer_budget{context_budget}": int(
                any(int(j) in answer for j in selected)
            ),
            f"sentence_budget{context_budget}": int(
                any(int(j) in sentence for j in selected)
            ),
            "oracle_answer": int(bool(answer)),
            "oracle_sentence": int(bool(sentence)),
            "top5_tokens": int(costs[top5].sum()),
            "budget_tokens": int(costs[selected].sum()),
            "budget_chunks": len(selected),
            "parent_miss@5": int(not parent5),
            "answer_missing_with_parent@5": int(parent5 and not answer5),
            "sentence_missing_with_answer@5": int(answer5 and not sentence5),
            "top5_chunks": [int(j) for j in top5],
        }
        assert row["answer@5"] <= row["chunk_parent@5"] <= row["deduplicated_parent@5"]
        assert row["sentence@5"] <= row["answer@5"] <= row["oracle_answer"]
        rows.append(row)
    names = [k for k, v in rows[0].items() if isinstance(v, (int, float))]
    aggregate = {k: float(np.mean([r[k] for r in rows])) for k in names}
    by_type = {
        kind: {
            k: float(np.mean([r[k] for r in rows if r["question_type"] == kind]))
            for k in names
        }
        for kind in sorted({q["type"] for q in queries})
    }
    return {"metrics": aggregate, "by_question_type": by_type, "queries": rows}


def fetch(cache):
    import urllib.request

    cache.mkdir(parents=True, exist_ok=True)
    manifest = {
        "repository": "SkelterLabsInc/JaQuAD",
        "revision": SOURCE_REV,
        "files": [],
    }
    for i in range(4):
        path = f"data/dev/jaquad_dev_{i:04d}.json"
        data = urllib.request.urlopen(
            f"https://huggingface.co/datasets/SkelterLabsInc/JaQuAD/resolve/{SOURCE_REV}/{path}",
            timeout=60,
        ).read()
        (cache / Path(path).name).write_bytes(data)
        manifest["files"].append(
            {
                "path": path,
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
        )
    (cache / "source.json").write_text(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache/answer-spans")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/answer-spans")
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    if args.fetch:
        fetch(args.cache)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    docs, queries, source = load_source(args.cache)
    path = snapshot_download(
        MODEL,
        revision=MODEL_REV,
        local_files_only=True,
        allow_patterns=["*.json", "*.safetensors"],
    )
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModel.from_pretrained(
        path, local_files_only=True, dtype=torch.float32, attn_implementation="sdpa"
    ).eval()
    selection = {
        "source": source,
        "model": MODEL,
        "revision": MODEL_REV,
        "queries": [
            {
                "id": q["id"],
                "parent": q["parent"],
                "spans": q["spans"],
                "sentences": q["sentences"],
            }
            for q in queries
        ],
        "documents": [
            {
                "title_sha256": hashlib.sha256(d["title"].encode()).hexdigest(),
                "context_sha256": hashlib.sha256(d["context"].encode()).hexdigest(),
            }
            for d in docs
        ],
    }
    (args.output / "selection.json").write_text(json.dumps(selection, indent=2))
    configs = {}
    for strategy in ["tokens", "sentence_end"]:
        for limit in [128, 256, 512]:
            cs = make_chunks(tokenizer, docs, limit, strategy)
            key = f"{strategy}-{limit}"
            configs[key] = cs
            (args.output / f"chunks-{key}.json").write_text(
                json.dumps([{k: v for k, v in c.items() if k != "text"} for c in cs])
            )
            print("prepared", key, len(cs), flush=True)
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
            qvec, qs = encode(
                model, tokenizer, [q["text"] for q in queries], "mean", 256, batch=16
            )
            assert qs["truncated"] == 0
            np.save(args.cache / f"queries-{mode}.npy", qvec)
            for key, cs in configs.items():
                limit = int(key.rsplit("-", 1)[1])
                vectors, stats = encode(
                    model, tokenizer, [c["text"] for c in cs], "mean", limit, batch=16
                )
                assert stats["truncated"] == 0
                np.savez_compressed(
                    args.cache / f"vectors-{key}-{mode}.npz", vectors=vectors
                )
                report = measure(qvec, vectors, queries, cs)
                report.update(
                    model=MODEL,
                    revision=MODEL_REV,
                    source_revision=SOURCE_REV,
                    precision=mode,
                    strategy=key,
                    query_count=len(queries),
                    document_count=len(docs),
                    chunk_count=len(cs),
                    query_encoding=qs,
                    chunk_encoding=stats,
                    environment={
                        "torch": torch.__version__,
                        "transformers": __import__("transformers").__version__,
                        "threads": 4,
                    },
                )
                rows = report.pop("queries")
                rows_path = args.output / f"{key}-{mode}-queries.json.gz"
                rows_path.write_bytes(
                    gzip.compress(
                        json.dumps(rows, separators=(",", ":")).encode(), mtime=0
                    )
                )
                report["queries_file"] = rows_path.name
                report["queries_sha256"] = hashlib.sha256(
                    rows_path.read_bytes()
                ).hexdigest()
                (args.output / f"{key}-{mode}.json").write_text(
                    json.dumps(report, indent=2)
                )
                print(mode, key, report["metrics"], flush=True)
    finally:
        torch.nn.functional.scaled_dot_product_attention = original


if __name__ == "__main__":
    main()
