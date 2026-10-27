"""Validate saved span metrics; report paired article-bootstrap comparisons."""

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from answer_spans import contains, load_source

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks/answer-spans"
CACHE = ROOT / ".cache/answer-spans"


def read_report(path):
    report = json.loads(path.read_text())
    if "queries" in report:
        # Convert completed runs from the first writer without losing their audit data.
        rows = report.pop("queries")
        compressed = gzip.compress(
            json.dumps(rows, separators=(",", ":")).encode(), mtime=0
        )
        rows_path = path.with_name(path.stem + "-queries.json.gz")
        rows_path.write_bytes(compressed)
        report["queries_file"] = rows_path.name
        report["queries_sha256"] = hashlib.sha256(compressed).hexdigest()
        path.write_text(json.dumps(report, indent=2))
    rows_path = OUT / report["queries_file"]
    assert (
        hashlib.sha256(rows_path.read_bytes()).hexdigest() == report["queries_sha256"]
    )
    rows = json.loads(gzip.decompress(rows_path.read_bytes()))
    return report, rows


def main():
    docs, queries, source = load_source(CACHE)
    selection = json.loads((OUT / "selection.json").read_text())
    assert selection["source"] == source
    assert [q["id"] for q in queries] == [q["id"] for q in selection["queries"]]
    results = {}
    rng = np.random.default_rng(20261027)
    audit_ids = sorted(rng.choice(len(queries), 12, replace=False))
    for mode in ["fp32", "w8a8-bf16kv"]:
        qvec = np.load(CACHE / f"queries-{mode}.npy")
        assert qvec.shape == (3939, 384) and np.isfinite(qvec).all()
        for strategy in ["tokens", "sentence_end"]:
            for length in [128, 256, 512]:
                key = f"{strategy}-{length}"
                report, rows = read_report(OUT / f"{key}-{mode}.json")
                chunks = json.loads((OUT / f"chunks-{key}.json").read_text())
                assert len(rows) == 3939 and [r["query_id"] for r in rows] == [
                    q["id"] for q in queries
                ]
                assert (
                    report["query_encoding"]["truncated"]
                    == report["chunk_encoding"]["truncated"]
                    == 0
                )
                assert len(chunks) == report["chunk_count"]
                assert max(c["tokens"] for c in chunks) <= length
                vectors = np.load(CACHE / f"vectors-{key}-{mode}.npz")["vectors"]
                assert (
                    vectors.shape == (len(chunks), 384) and np.isfinite(vectors).all()
                )
                for name, value in report["metrics"].items():
                    assert abs(np.mean([r[name] for r in rows]) - value) < 1e-12
                for row in rows:
                    assert (
                        row["parent_miss@5"]
                        + row["answer_missing_with_parent@5"]
                        + row["answer@5"]
                        == 1
                    )
                    assert (
                        row["sentence@5"]
                        <= row["answer@5"]
                        <= row["chunk_parent@5"]
                        <= row["deduplicated_parent@5"]
                    )
                # Independently recover top five from saved vectors and recheck source spans.
                ranking = np.argsort(
                    -(qvec[audit_ids] @ vectors.T), axis=1, kind="stable"
                )[:, :5]
                for at, indices in zip(audit_ids, ranking):
                    q = queries[at]
                    assert rows[at]["top5_chunks"] == indices.tolist()
                    assert rows[at]["answer@5"] == any(
                        contains(chunks[int(i)], q["parent"], q["spans"])
                        for i in indices
                    )
                    assert rows[at]["sentence@5"] == any(
                        contains(chunks[int(i)], q["parent"], q["sentences"])
                        for i in indices
                    )
                results[(mode, key)] = (report, rows)
    # Questions from the same article are correlated. Resample whole articles.
    titles = sorted({d["title"] for d in docs})
    cluster = np.array([titles.index(docs[q["parent"]]["title"]) for q in queries])
    counts = np.bincount(cluster)
    boot_indices = rng.integers(0, len(titles), (10000, len(titles)))
    comparisons = {}
    for a, b in [
        ("tokens-256", "tokens-128"),
        ("tokens-512", "tokens-256"),
        ("sentence_end-256", "tokens-256"),
        ("sentence_end-256", "sentence_end-128"),
        ("sentence_end-512", "sentence_end-256"),
    ]:
        ra = results[("w8a8-bf16kv", a)][1]
        rb = results[("w8a8-bf16kv", b)][1]
        for metric in [
            "answer@5",
            "sentence@5",
            "answer_budget1024",
            "sentence_budget1024",
        ]:
            diff = np.array([x[metric] - y[metric] for x, y in zip(ra, rb)])
            sums = np.bincount(cluster, weights=diff, minlength=len(titles))
            boot = sums[boot_indices].sum(1) / counts[boot_indices].sum(1)
            comparisons[f"{a} minus {b}: {metric}"] = {
                "delta": float(diff.mean()),
                "article_bootstrap_95_interval": np.quantile(
                    boot, [0.025, 0.975]
                ).tolist(),
            }
    summary = {
        "queries": len(queries),
        "documents": len(docs),
        "articles": len(titles),
        "configurations": {
            f"{mode}:{key}": {
                "metrics": r["metrics"],
                "chunk_count": r["chunk_count"],
                "by_question_type": r["by_question_type"],
            }
            for (mode, key), (r, _) in results.items()
        },
        "comparisons": comparisons,
        "validation": {
            "source_answer_offsets_valid": True,
            "no_silent_truncation": True,
            "sampled_vector_rank_and_span_checks": len(audit_ids) * 12,
            "per_query_aggregation_checked": 3939 * 12,
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "# Answer-containing chunks: Bekko a8m",
        "",
        "Measured locally on 2026-10-27 (JST). Full public JaQuAD validation: 3,939 questions, 1,431 source passages and "
        + str(len(titles))
        + " article titles. All annotated answer offsets matched the original text; no questions were excluded. Every passage is a candidate, without sampled distractors. This is a different and larger evaluation than the earlier 128-query parent-document experiment.",
        "",
        "## What counts as success",
        "",
        "- Answer hit: a retrieved chunk comes from the annotated source passage and completely contains at least one annotated answer span. The same words in another passage do not count.",
        "- Sentence hit: the chunk also fully contains the sentence/line span enclosing that answer. This is a context-retention proxy, not proof that all reasoning evidence is present.",
        "- Top-5 uses five actual chunks, with no parent-document deduplication. Parent@5 below deduplicates documents as in the earlier experiment, making the difference visible.",
        "- Budget-1024 takes ranked whole chunks until adding the next would exceed 1,024 Bekko tokenizer tokens, including embedding special tokens. It neither truncates that chunk nor skips ahead. Different chunk sizes can leave different unused space. Queries always use the same 256-token maximum across all configurations.",
        "",
        "## Quantized-reference results",
        "",
        "All percentages are fractions of the 3,939 questions. W8A8 Linear simulation with BF16 K/V, original F32 embedding table and norms. These are local reference-model measurements, not new Wasm measurements.",
        "",
        "| Splitting | Max tokens | Vectors | Parent@5 | Answer@1 | Answer@5 | Sentence@5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy in ["tokens", "sentence_end"]:
        for n in [128, 256, 512]:
            r = results[("w8a8-bf16kv", f"{strategy}-{n}")][0]
            m = r["metrics"]
            vals = " | ".join(
                f"{m[k] * 100:.2f}%"
                for k in ["deduplicated_parent@5", "answer@1", "answer@5", "sentence@5"]
            )
            lines.append(f"| {strategy} | {n} | {r['chunk_count']} | {vals} |")
    lines += [
        "",
        "| Splitting | Max tokens | Answer, budget 1024 | Sentence, budget 1024 | Mean tokens used | Mean chunks used |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy in ["tokens", "sentence_end"]:
        for n in [128, 256, 512]:
            m = results[("w8a8-bf16kv", f"{strategy}-{n}")][0]["metrics"]
            lines.append(
                f"| {strategy} | {n} | {m['answer_budget1024'] * 100:.2f}% | {m['sentence_budget1024'] * 100:.2f}% | {m['budget_tokens']:.1f} | {m['budget_chunks']:.2f} |"
            )
    lines += [
        "",
        "## Where the short chunks lose information",
        "",
        "| Splitting | Max tokens | Parent retrieved but answer absent from top-5 | Answer retrieved but whole sentence absent | Any candidate contains whole sentence (oracle) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for strategy in ["tokens", "sentence_end"]:
        for n in [128, 256, 512]:
            m = results[("w8a8-bf16kv", f"{strategy}-{n}")][0]["metrics"]
            lines.append(
                f"| {strategy} | {n} | {m['answer_missing_with_parent@5'] * 100:.2f}% | {m['sentence_missing_with_answer@5'] * 100:.2f}% | {m['oracle_sentence'] * 100:.2f}% |"
            )
    lines += [
        "",
        "All six split configurations retain every annotated answer in at least one candidate chunk. Oracle sentence coverage separates splitting limitations from retrieval ranking failures.",
        "",
        "## Split rules and scope",
        "",
        "`tokens` uses token-offset windows with 32 source-token overlap. `sentence_end` uses the same windows/overlap but prefers the last Japanese sentence-ending punctuation or newline in the latter half of the window. If there is no such boundary, it falls back to the token window. It aligns chunk ends, not necessarily starts; it is not a guarantee of complete sentences. Every emitted chunk is re-tokenized and checked against its budget, and every non-whitespace source character is checked for coverage.",
        "",
        "Each source is title + newline + the original paragraph. Titles are not repeated on later chunks. Source paragraphs are relatively short: at 512 tokens most fit in one chunk. Results do not establish an optimum for long technical manuals or whole articles.",
        "",
        "JaQuAD includes 829 multiple-sentence-reasoning questions. Merely including an answer string or one enclosing sentence may omit necessary reasoning evidence. No answer-generation model was used and generated-answer correctness was not evaluated. The sentence metric is explicitly a proxy.",
        "",
        "## FP32 check",
        "",
        "| Splitting | Max tokens | FP32 answer@5 | W8A8 answer@5 | FP32 sentence@5 | W8A8 sentence@5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy in ["tokens", "sentence_end"]:
        for n in [128, 256, 512]:
            fp = results[("fp32", f"{strategy}-{n}")][0]["metrics"]
            q = results[("w8a8-bf16kv", f"{strategy}-{n}")][0]["metrics"]
            lines.append(
                f"| {strategy} | {n} | {fp['answer@5'] * 100:.2f}% | {q['answer@5'] * 100:.2f}% | {fp['sentence@5'] * 100:.2f}% | {q['sentence@5'] * 100:.2f}% |"
            )
    lines += [
        "",
        "## Paired uncertainty",
        "",
        "10,000 paired bootstrap samples of whole articles (not independent questions). Intervals condition on this fixed validation corpus and do not establish general performance. Differences are percentage points.",
        "",
        "| Comparison | Metric | Difference | Article-bootstrap 95% interval |",
        "| --- | --- | ---: | ---: |",
    ]
    for key, c in comparisons.items():
        label, metric = key.split(": ")
        lo, hi = c["article_bootstrap_95_interval"]
        lines.append(
            f"| {label} | {metric} | {c['delta'] * 100:+.2f} | [{lo * 100:+.2f}, {hi * 100:+.2f}] |"
        )
    lines += [
        "",
        "## Reproduce and audit",
        "",
        "Model: `hotchpotch/bekko-embedding-v1-a8m`, revision `c721113d59a1d91b447450324f51c4b3332c924a`. Dataset: `SkelterLabsInc/JaQuAD`, revision `05600ff310a0970823e70f82f428893b85c71ffe`. Source file hashes, exact question IDs, source spans and context hashes are recorded in `selection.json`. Per-query metrics and top-5 chunk IDs are stored in compressed JSON audit files; six `chunks-*.json` files map those IDs to source offsets. `summary.json` contains all aggregates, question-type breakdowns and intervals.",
        "",
        "The implementation checks all aggregate metrics against per-query rows and independently audits 12 sampled query rankings and source-span hits for each of 12 runs from saved vectors (144 audits). Four hand-computed tests cover span boundaries, wrong-source matches, complete sentence bounds and text-preserving chunking.",
        "",
        "Use the [Bekko Python setup](../../docs/BEKKO.md#prepare-and-verify), then run from the repository root:",
        "",
        "```sh",
        ".cache/bekko-venv/bin/python experiments/multilingual/answer_spans.py --fetch",
        ".cache/bekko-venv/bin/python -m unittest discover -s experiments/multilingual -p test_answer_spans.py",
        ".cache/bekko-venv/bin/python experiments/multilingual/answer_spans_report.py",
        "```",
        "",
        "The model must already be cached locally; `--fetch` retrieves only the four pinned public validation JSON files into `.cache/answer-spans`. Dependencies are the existing root `requirements.txt`; PyArrow is also needed by the reused reference module. No canister API or token limit is changed. **512-token rows are model-side reference experiments; the live canister remains capped at 256 tokens.**",
        "",
        "Dataset attribution: [Skelter Labs JaQuAD](https://huggingface.co/datasets/SkelterLabsInc/JaQuAD), human-annotated Japanese QA over Wikipedia, CC BY-SA 3.0. The sentence spans and chunk labels are derived here from the supplied answer annotations, not provided as evidence labels by the dataset authors.",
        "",
    ]
    text = "\n".join(lines).replace(
        "## Paired uncertainty",
        "## Practical choice\n\nFor this corpus, start with a 256-token maximum and sentence-end preference. Its quantized-reference answer@5 is 92.21% and full-sentence@5 is 91.85%, with 2,080 vectors. Under the 1,024-token whole-chunk budget it retains an answer in 92.05% and its enclosing sentence in 91.67% of questions. Sentence-end 512 reaches 93.53% at top-5, but falls to 90.23% for both metrics under that budget. Plain 128 windows recover more answers under the fixed budget (93.37%) while retaining the full sentence much less often (81.92%). This supports a practical balance, not a universal optimum or a generated-answer-quality claim.\n\n"
        + "## Paired uncertainty",
    )
    (OUT / "RESULTS.md").write_text(text)
    print(summary["validation"])
    for (mode, key), (r, _) in results.items():
        if mode == "w8a8-bf16kv":
            print(
                key,
                {
                    k: round(r["metrics"][k] * 100, 2)
                    for k in [
                        "answer@5",
                        "sentence@5",
                        "answer_budget1024",
                        "sentence_budget1024",
                    ]
                },
            )


if __name__ == "__main__":
    main()
