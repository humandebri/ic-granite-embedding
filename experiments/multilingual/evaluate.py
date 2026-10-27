"""Pinned, bounded Japanese retrieval experiment; not a full MTEB score."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
SPECS = json.loads((Path(__file__).with_name("models.json")).read_text())
DATA_REV = "5713cd1b76a22b79c3595827c717c72b6cab8852"


class W8A8(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        w = original.weight.detach().float()
        s = w.abs().amax(-1).div(127).clamp_min(1e-12)
        self.register_buffer("w", (w / s[:, None]).round().clamp(-127, 127))
        self.register_buffer("s", s)
        self.register_buffer(
            "bias",
            original.bias.detach().float() if original.bias is not None else None,
        )

    def forward(self, x):
        s = x.abs().amax(-1, keepdim=True).div(127).clamp_min(1e-12)
        y = (
            torch.nn.functional.linear((x / s).round().clamp(-127, 127), self.w)
            * s
            * self.s
        )
        return y if self.bias is None else y + self.bias


def dataset(n, negatives):
    tables = {}
    for name in ["corpus", "queries", "qrels"]:
        p = hf_hub_download(
            "mteb/JaQuADRetrieval",
            f"{name}/validation-00000-of-00001.parquet",
            repo_type="dataset",
            revision=DATA_REV,
        )
        tables[name] = pq.read_table(p).to_pylist()
    # This source also contains answer-only documents. Keep passage retrieval only.
    tables["corpus"] = [d for d in tables["corpus"] if len(d["text"]) >= 80]
    passage_ids = {d["_id"] for d in tables["corpus"]}
    tables["qrels"] = [r for r in tables["qrels"] if r["corpus-id"] in passage_ids]
    eligible = {r["query-id"] for r in tables["qrels"] if r["score"] > 0}
    tables["queries"] = [q for q in tables["queries"] if q["_id"] in eligible]
    rng = np.random.default_rng(20261027)
    queries = [
        tables["queries"][i]
        for i in sorted(rng.choice(len(tables["queries"]), n, replace=False))
    ]
    qids = {q["_id"] for q in queries}
    rel = {}
    for r in tables["qrels"]:
        if r["query-id"] in qids and r["score"] > 0:
            rel.setdefault(r["query-id"], set()).add(r["corpus-id"])
    positives = set().union(*rel.values())
    other = [d for d in tables["corpus"] if d["_id"] not in positives]
    chosen = {
        other[i]["_id"]
        for i in rng.choice(len(other), min(negatives, len(other)), replace=False)
    } | positives
    corpus = [d for d in tables["corpus"] if d["_id"] in chosen]
    return queries, corpus, rel


def metrics(q, d, queries, corpus, rel, parents=None):
    sims = q @ d.T
    if parents is not None:
        merged = np.full((len(q), len(corpus)), -np.inf)
        for j, parent in enumerate(parents):
            merged[:, parent] = np.maximum(merged[:, parent], sims[:, j])
        sims = merged
    order = np.argsort(-sims, axis=1, kind="stable")
    values = []
    ranks = []
    for i, query in enumerate(queries):
        good = rel[query["_id"]]
        hits = np.array([corpus[j]["_id"] in good for j in order[i]])
        rank = int(np.flatnonzero(hits)[0]) + 1
        dcg = sum(float(hits[j]) / np.log2(j + 2) for j in range(min(10, len(hits))))
        ideal = sum(1 / np.log2(j + 2) for j in range(min(10, len(good))))
        values.append(
            [
                float(hits[0]),
                sum(hits[:5]) / len(good),
                sum(hits[:10]) / len(good),
                1 / rank,
                dcg / ideal,
            ]
        )
        ranks.append(
            {
                "query_id": query["_id"],
                "rank": rank,
                "top1": corpus[order[i, 0]]["_id"],
                "recall@5": float(sum(hits[:5]) / len(good)),
                "ndcg@10": dcg / ideal,
            }
        )
    return dict(
        zip(
            ["hit@1", "recall@5", "recall@10", "mrr", "ndcg@10"],
            np.mean(values, axis=0).tolist(),
        )
    ), ranks


SMOKE = [
    (
        "パスワードを忘れた場合の対処方法は？",
        "To recover access, request a password reset link from the sign-in page.",
    ),
    (
        "データベースのバックアップはいつ行いますか？",
        "Database backups run every night at two in the morning and are retained for thirty days.",
    ),
    (
        "画像をアップロードできる最大サイズは？",
        "Uploaded images must be smaller than ten megabytes; larger files are rejected.",
    ),
    (
        "有料プランを解約するとデータは消えますか？",
        "Cancelling a paid subscription keeps your documents but disables premium features.",
    ),
    (
        "検索結果が古いままなのはなぜ？",
        "The search index refreshes hourly, so newly edited pages may take an hour to appear.",
    ),
    (
        "APIの呼び出し回数制限はいくつ？",
        "Each API key allows sixty requests per minute; excess requests receive status 429.",
    ),
    (
        "削除したファイルを復元したい",
        "Deleted files remain in the recycle bin for fourteen days before permanent removal.",
    ),
    (
        "ログインに二段階認証を設定するには？",
        "Enable two-factor authentication in account security and scan the QR code with an authenticator.",
    ),
    (
        "複数人で同時に文書を編集できますか？",
        "Shared documents support simultaneous editing with changes visible to all collaborators.",
    ),
    (
        "請求書の宛名を変更したい",
        "Update the billing name and address under payment settings before the next invoice is issued.",
    ),
    (
        "オフラインでもメモを読めますか？",
        "Previously synchronized notes can be read offline; edits synchronize after connectivity returns.",
    ),
    (
        "保存されたデータは暗号化されますか？",
        "Stored data is encrypted at rest and all network traffic is protected by TLS.",
    ),
]


def encode(model, tokenizer, texts, pooling, limit, batch=8):
    result = []
    lengths = []
    chars = []
    started = time.perf_counter()
    for at in range(0, len(texts), batch):
        group = texts[at : at + batch]
        raw = tokenizer(group, add_special_tokens=True, truncation=False)["input_ids"]
        lengths.extend(map(len, raw))
        enc = tokenizer(
            group,
            padding=True,
            truncation=True,
            max_length=limit,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offsets = enc.pop("offset_mapping")
        chars.extend(int(x[:, 1].max()) for x in offsets)
        with torch.inference_mode():
            h = model(**enc).last_hidden_state.float()
            if pooling == "cls":
                v = h[:, 0]
            else:
                mask = enc["attention_mask"].unsqueeze(-1)
                v = (h * mask).sum(1) / mask.sum(1)
            if hasattr(model, "sentence_projection"):
                v = model.sentence_projection(v)
            result.append(torch.nn.functional.normalize(v, dim=-1).numpy())
    return np.concatenate(result), {
        "seconds": time.perf_counter() - started,
        "tokens_p50": float(np.median(lengths)),
        "tokens_p95": float(np.percentile(lengths, 95)),
        "truncated": sum(x > limit for x in lengths),
        "retained_characters_p50": float(np.median(chars)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default=",".join(SPECS))
    p.add_argument("--queries", type=int, default=128)
    p.add_argument("--negatives", type=int, default=512)
    p.add_argument("--limit", type=int, default=256)
    p.add_argument("--output", type=Path, default=ROOT / "benchmarks/multilingual")
    p.add_argument("--export", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    queries, corpus, rel = dataset(args.queries, args.negatives)
    fixture = {
        "dataset": "mteb/JaQuADRetrieval",
        "revision": DATA_REV,
        "seed": 20261027,
        "passage_filter": "text length >= 80 Unicode characters; qrels restricted to these passages",
        "queries": [q["_id"] for q in queries],
        "documents": [d["_id"] for d in corpus],
        "qrels": {k: sorted(v) for k, v in rel.items()},
    }
    (args.output / "selection.json").write_text(json.dumps(fixture, indent=2))
    for key in args.models.split(","):
        spec = SPECS[key]
        print("loading", key, flush=True)
        path = snapshot_download(
            spec["id"],
            revision=spec["revision"],
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"],
        )
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModel.from_pretrained(
            path, dtype=torch.float32, attn_implementation="sdpa"
        ).eval()
        info = {
            "model": spec,
            "config": model.config.to_dict(),
            "parameters": sum(x.numel() for x in model.parameters()),
            "environment": {
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "platform": platform.platform(),
                "threads": 4,
            },
            "max_tokens": args.limit,
            "query_count": len(queries),
            "corpus_count": len(corpus),
            "dataset_revision": DATA_REV,
            "selection_sha256": hashlib.sha256(
                (args.output / "selection.json").read_bytes()
            ).hexdigest(),
            "runs": {},
        }
        qtexts = [q["text"] for q in queries]
        dtexts = [(d.get("title", "") + "\n" + d["text"]).strip() for d in corpus]
        chunks = []
        parents = []
        retained = []
        raw_lengths = []
        # Split every passage, retaining all text. Compare document ranks using max chunk score.
        budget = (
            args.limit
            - 2
            - (
                len(tokenizer.encode("passage: ", add_special_tokens=False))
                if spec.get("prefix")
                else 0
            )
        )
        if budget <= 32:
            raise ValueError("token budget must exceed the 32-token overlap")
        for j, text in enumerate(dtexts):
            enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            offsets = enc["offset_mapping"]
            raw_lengths.append(len(offsets) + 2)
            retained.append(offsets[min(budget, len(offsets)) - 1][1])
            start = 0
            while start < len(offsets):
                end = min(start + budget, len(offsets))
                chunk = text[offsets[start][0] : offsets[end - 1][1]]
                prefix = "passage: " if spec.get("prefix") else ""
                # Text slicing can split a subword; recheck the encoded budget.
                while len(tokenizer.encode(prefix + chunk)) > args.limit:
                    end -= 1
                    assert end > start
                    chunk = text[offsets[start][0] : offsets[end - 1][1]]
                chunks.append(chunk)
                parents.append(j)
                if end == len(offsets):
                    break
                start = max(start + 1, end - 32)
        texts = qtexts + chunks
        if spec.get("prefix"):
            texts = [
                "query: " + s if i < len(queries) else "passage: " + s
                for i, s in enumerate(texts)
            ]
        info["chunking"] = {
            "overlap_tokens": 32,
            "chunks": len(chunks),
            "aggregation": "maximum chunk cosine per document",
            "original_document_tokens_p50": float(np.median(raw_lengths)),
            "first_chunk_characters_p50": float(np.median(retained)),
        }
        previous = None
        for mode in ["fp32", "w8a8-bf16kv"]:
            if mode != "fp32":
                for name, module in list(model.named_modules()):
                    if isinstance(module, torch.nn.Linear):
                        parent, _, leaf = name.rpartition(".")
                        setattr(model.get_submodule(parent), leaf, W8A8(module))
                original = torch.nn.functional.scaled_dot_product_attention

                def sdpa(q, k, v, *a, _original=original, **kw):
                    return _original(
                        q, k.bfloat16().float(), v.bfloat16().float(), *a, **kw
                    )

                torch.nn.functional.scaled_dot_product_attention = sdpa
            vectors, stats = encode(
                model, tokenizer, texts, spec["pooling"], args.limit
            )
            scores, ranks = metrics(
                vectors[: len(queries)],
                vectors[len(queries) :],
                queries,
                corpus,
                rel,
                parents,
            )
            smokes, _ = encode(
                model,
                tokenizer,
                [("query: " if spec.get("prefix") else "") + x[0] for x in SMOKE]
                + [("passage: " if spec.get("prefix") else "") + x[1] for x in SMOKE],
                spec["pooling"],
                args.limit,
            )
            cross = smokes[: len(SMOKE)] @ smokes[len(SMOKE) :].T
            run = {
                "metrics": scores,
                "encoding": stats,
                "crosslingual_handwritten_hit@1": float(
                    np.mean(cross.argmax(1) == np.arange(len(SMOKE)))
                ),
                "ranks": ranks,
            }
            if previous is not None:
                run["cosine_to_fp32"] = {
                    "min": float((vectors * previous).sum(1).min()),
                    "mean": float((vectors * previous).sum(1).mean()),
                }
            else:
                previous = vectors
            info["runs"][mode] = run
            (args.output / f"{key}-{args.limit}.json").write_text(
                json.dumps(info, indent=2, ensure_ascii=False)
            )
            print(key, mode, scores, stats, flush=True)
            if mode != "fp32":
                torch.nn.functional.scaled_dot_product_attention = original
        if args.export and key in ["granite", "bekko8"]:
            from export import export_model

            export_model(key)
        del model


if __name__ == "__main__":
    main()
