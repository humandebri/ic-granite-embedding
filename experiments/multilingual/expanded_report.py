"""Validate expanded measurements against cached vectors and report all languages."""

import hashlib
import json
from pathlib import Path

import numpy as np
from evaluate import metrics
from expanded import DATA, SPECS

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks/multilingual-expanded"
LABELS = {
    "english": "Granite Small English R2",
    "granite": "Granite 97M R2 multilingual",
    "bekko8": "Bekko a8m",
    "bekko25": "Bekko a25m",
    "e5": "multilingual-e5-small",
    "bge": "bge-small-en-v1.5",
    "e5en": "e5-small-v2",
    "minilm": "all-MiniLM-L6-v2",
    "granite107": "Granite 107M multilingual",
    "horizon": "Horizon multilingual-small",
    "granite311": "Granite 311M R2 multilingual",
}


def load_result(task, key, selection):
    path = OUT / task / f"{key}-256.json"
    if path.exists():
        x = json.loads(path.read_text())
        assert (
            x["model"]["id"] == SPECS[key]["id"]
            and x["model"]["revision"] == SPECS[key]["revision"]
        )
        assert (
            x["selection_sha256"]
            == hashlib.sha256((OUT / task / "selection.json").read_bytes()).hexdigest()
        )
        for mode, r in x["runs"].items():
            assert r["encoding"]["truncated"] == 0
            cached = np.load(
                ROOT / f".cache/multilingual-expanded/{task}/{key}-256-{mode}.npz"
            )
            vec = cached["vectors"]
            parents = cached["parents"]
            assert np.isfinite(vec).all()
            assert np.max(abs(np.linalg.norm(vec, axis=1) - 1)) < 1e-5
            nq = len(selection["queries"])
            assert len(vec) == nq + x["chunks"]
            assert len(parents) == x["chunks"]
            rel = {k: set(v) for k, v in selection["qrels"].items()}
            recomputed, ranks = metrics(
                vec[:nq],
                vec[nq:],
                [{"_id": q} for q in selection["queries"]],
                [{"_id": d} for d in selection["documents"]],
                rel,
                parents,
            )
            for name, score in recomputed.items():
                assert abs(score - r["metrics"][name]) < 1e-12, (task, key, name)
            assert [v["query_id"] for v in ranks] == selection["queries"]
            assert [v["rank"] for v in ranks] == [v["rank"] for v in r["ranks"]]
        assert set(x["runs"]) == {"fp32", "w8a8-bf16kv"}, (task, key, "incomplete")
        return x, False
    assert task == "jaquad" and key in [
        "english",
        "granite",
        "bekko8",
        "bekko25",
        "e5",
    ], path
    oldselection = json.loads(
        (ROOT / "benchmarks/multilingual/selection.json").read_text()
    )
    for f in ["queries", "documents", "qrels"]:
        assert oldselection[f] == selection[f]
    x = json.loads((ROOT / f"benchmarks/multilingual/{key}-256.json").read_text())
    return x, True


def per_query(result, metric):
    ranks = result["runs"]["w8a8-bf16kv"]["ranks"]
    if metric in ranks[0]:
        return np.array([r[metric] for r in ranks])
    # Reused Japanese results have exactly one relevant passage per query.
    assert metric in ["recall@5", "ndcg@10"]
    return np.array(
        [
            float(r["rank"] <= 5)
            if metric == "recall@5"
            else (1 / np.log2(r["rank"] + 1) if r["rank"] <= 10 else 0)
            for r in ranks
        ]
    )


def main():
    results = {}
    summary = {
        "datasets": {},
        "models": {},
        "comparisons": {},
        "validation": {"vector_recomputed": 0, "reused_earlier_japanese": 0},
    }
    for task in DATA:
        selection = json.loads((OUT / task / "selection.json").read_text())
        assert len(selection["queries"]) == 128
        summary["datasets"][task] = {
            "queries": 128,
            "documents": len(selection["documents"]),
            "source": DATA[task],
        }
        results[task] = {}
        if task == "jaquad":
            assert all(len(r) == 1 for r in selection["qrels"].values())
        for key in SPECS:
            x, reused = load_result(task, key, selection)
            results[task][key] = x
            summary["validation"][
                "reused_earlier_japanese" if reused else "vector_recomputed"
            ] += 1
            summary["models"].setdefault(
                key,
                {
                    "id": SPECS[key]["id"],
                    "revision": SPECS[key]["revision"],
                    "tasks": {},
                },
            )
            summary["models"][key]["tasks"][task] = {
                mode: r["metrics"] for mode, r in x["runs"].items()
            }
            if "embedding_parameters" in x:
                summary["models"][key].update(
                    parameters=x["parameters"],
                    active_parameters=x["parameters"] - x["embedding_parameters"],
                    dimensions=x["runs"]["fp32"]["dimensions"],
                )
    rng = np.random.default_rng(20261027)
    for task in DATA:
        for other in ["english", "granite", "bekko25", "e5", "granite311"]:
            for metric in ["recall@5", "ndcg@10"]:
                d = per_query(results[task]["bekko8"], metric) - per_query(
                    results[task][other], metric
                )
                boot = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
                summary["comparisons"][f"{task}:bekko8-minus-{other}:{metric}"] = {
                    "delta": float(d.mean()),
                    "query_bootstrap_95_interval": np.quantile(
                        boot, [0.025, 0.975]
                    ).tolist(),
                    "queries_better": int((d > 0).sum()),
                    "queries_worse": int((d < 0).sum()),
                }
    adapters = json.loads((OUT / "adapters.json").read_text())
    assert {r["model"] for r in adapters} == set(SPECS)
    assert all(r["max_abs_difference"] < 1e-5 for r in adapters)
    summary["validation"]["sentence_transformers_adapter_checks"] = len(adapters)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "# Expanded English and Japanese embedding evaluation",
        "",
        "Measured locally on 2026-10-27 (JST). Eleven models, three fixed candidate pools, 256-token chunks, FP32 and W8A8/BF16-KV. These are bounded retrieval experiments, not full BEIR, MTEB, JMTEB or a canister performance comparison.",
        "",
        "## Data and method",
        "",
        "| Task | Language/domain | Queries | Candidate documents |",
        "| --- | --- | ---: | ---: |",
    ]
    for task, label in [
        ("scifact", "English scientific claims / abstracts"),
        ("fiqa", "English finance questions / answers"),
        ("jaquad", "Japanese questions / passages"),
    ]:
        lines.append(
            f"| {task} | {label} | 128 | {summary['datasets'][task]['documents']} |"
        )
    lines += [
        "",
        "Every pool includes all positive documents for the selected queries plus 512 seeded random distractors (seed 20261027). The English datasets use test qrels. Japanese IDs/qrels match the earlier filtered JaQuAD validation experiment; five earlier Japanese results are reused after checking exact query/document/qrel equality. Each model has the same candidate pool per task. Model training/evaluation overlap was not audited. Random distractors are often easier than a full corpus or mined hard negatives.",
        "",
        "Each model uses its own tokenizer, max 256 tokens including prefixes/special tokens, 32-token source overlap, and maximum chunk cosine per document with document deduplication. This measures parent-document retrieval, not whether the returned chunk contains an answer. E5 gets query/passage prefixes. BGE gets its recommended retrieval instruction on queries only. Horizon includes its learned 384→1024 projection; it is not evaluated as the raw backbone. Sentence Transformers checks verify all eleven FP32 pooling/projection adapters on four short English/Japanese texts, with max absolute difference <1e-5.",
        "",
        "W8A8 simulates per-channel INT8 Linear weights and per-token INT8 activations, plus BF16 K/V; embeddings/norms remain F32. Horizon’s output Linear projection is also quantized. Float simulations are not integer-speed measurements. Vectors are saved locally and every newly measured result is recomputed from saved vectors to validate ranking and aggregation.",
        "",
        "## Retrieval quality",
        "",
        "nDCG@10 rewards placing relevant documents near the top. Values below are ×100. All three columns use the same metric.",
        "",
        "| Model | English SciFact FP32 / W8A8 | English FiQA FP32 / W8A8 | Japanese JaQuAD FP32 / W8A8 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in SPECS:
        vals = []
        for task in DATA:
            x = results[task][key]
            vals.append(
                " / ".join(
                    f"{x['runs'][m]['metrics']['ndcg@10'] * 100:.2f}"
                    for m in ["fp32", "w8a8-bf16kv"]
                )
            )
        lines.append(f"| {LABELS[key]} | {' | '.join(vals)} |")
    lines += [
        "",
        "Recall@5, W8A8/BF16-KV: mean fraction of the query’s known relevant documents found in the first five deduplicated documents. A query can have multiple relevant documents, especially in FiQA.",
        "",
        "| Model | English SciFact | English FiQA | Japanese JaQuAD |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in SPECS:
        vals = [
            f"{results[t][key]['runs']['w8a8-bf16kv']['metrics']['recall@5'] * 100:.2f}%"
            for t in DATA
        ]
        lines.append(f"| {LABELS[key]} | {' | '.join(vals)} |")
    lines += [
        "",
        "ASCII BM25 baseline over whole original documents (k1=1.5, b=0.75, deduplicated query terms; not a tuned baseline):",
        "",
        "| Task | nDCG@10 | Recall@5 |",
        "| --- | ---: | ---: |",
    ]
    for task in ["scifact", "fiqa"]:
        x = json.loads((OUT / task / "bm25.json").read_text())["metrics"]
        lines.append(
            f"| {task} | {x['ndcg@10'] * 100:.2f} | {x['recall@5'] * 100:.2f}% |"
        )
    lines += [
        "",
        "BGE-small-en, E5-small-v2, MiniLM-L6-v2 and Granite Small English R2 are English-oriented baselines. Their Japanese results are diagnostic, not claims of official Japanese support. No aggregate score across these tasks is used to declare a universal winner.",
        "",
        "## Why Bekko a8m is efficient",
        "",
        "The name counts about 7.67M non-embedding parameters, not all weights: total size is about 106M. The model retains mmBERT-small layers [0,1,2,18] from a pretrained 22-layer encoder, then receives approximately 1.1 billion multilingual contrastive pairs and hard-negative fine-tuning. This preserves a large multilingual vocabulary and selected pretrained layers while cutting repeated transformer compute. The embedding training stages do not use teacher distillation. This explains a plausible route to strong quality per unit of compute; it does not prove the cause of a particular benchmark result. [Bekko paper](https://arxiv.org/html/2607.25180v1)",
        "",
        "The earlier 96.1% Japanese Recall@5 was a score on 128 queries against 635 candidates, not a universal accuracy rate. a8m versus a25m differed by only a net three queries at top-5 (four wins, one loss). The paired query-bootstrap interval below includes zero for that comparison. The developer’s broader MMTEB retrieval comparison instead reports 56.2 for a8m, 57.5 for a25m and 60.3 for Granite 97M R2. These are different tasks/metrics and are not directly comparable with our Recall percentages. [Published comparison](https://huggingface.co/hotchpotch/bekko-embedding-v1-a8m)",
        "",
        "## Paired comparisons",
        "",
        "Difference in W8A8 Recall@5, a8m minus the comparator, in percentage points. Intervals resample the same 128 queries 10,000 times. They quantify query sampling within this fixed pool only; they do not cover new corpora, annotation gaps, model training overlap or benchmark selection.",
        "",
        "| Task | Comparator | Difference | Query bootstrap 95% interval |",
        "| --- | --- | ---: | ---: |",
    ]
    for task in DATA:
        for key in ["english", "granite", "bekko25", "e5", "granite311"]:
            x = summary["comparisons"][f"{task}:bekko8-minus-{key}:recall@5"]
            lo, hi = x["query_bootstrap_95_interval"]
            lines.append(
                f"| {task} | {LABELS[key]} | {x['delta'] * 100:+.2f} | [{lo * 100:+.2f}, {hi * 100:+.2f}] |"
            )
    lines += [
        "",
        "## Model size and execution scope",
        "",
        "| Model | Total parameters M | Non-embedding parameters M | Output dimensions |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in SPECS:
        x = summary["models"][key]
        lines.append(
            f"| {LABELS[key]} | {x['parameters'] / 1e6:.2f} | {x['active_parameters'] / 1e6:.2f} | {x['dimensions']} |"
        )
    lines += [
        "",
        "Non-embedding parameters are a rough compute indicator, not a query-instruction measurement. Large vocabularies consume memory even though lookup costs less than dense transformer work. Output dimensionality also changes index storage and similarity costs.",
        "",
        "This expansion adds retrieval experiments, not six new canister ports. The earlier local Wasm measurements remain: Bekko a8m about 1.94B instructions at 256 tokens, Granite 97M about 4.61B, both one query on three test inputs. There are residual numerical differences between those Wasm ports and the Torch simulation; no full-corpus Wasm retrieval equivalence is claimed. [Earlier Wasm results](../multilingual/RESULTS.md)",
        "",
        "EmbeddingGemma-300M was considered but its weight/config access was gated for the available Hugging Face session, so it was not measured. No score was imputed.",
        "",
        "## Evidence and reproduction",
        "",
        "- `summary.json`: validated metrics, parameter counts and paired comparisons.",
        "- `<task>/selection.json`: exact source revisions, query/document IDs and qrels.",
        "- `<task>/<model>-256.json`: model/config, per-query ranks, both arithmetic modes and truncation diagnostics.",
        "- `adapters.json`: comparison with Sentence Transformers for all eleven models.",
        "- Local `.cache/multilingual-expanded/`: vectors and parent indices for recomputation.",
        "- [Commands and limitations](../../experiments/multilingual/README.md#expanded-english-and-japanese-comparison).",
        "",
        "Sources: [BEIR SciFact](https://huggingface.co/datasets/BeIR/scifact), [SciFact test qrels](https://huggingface.co/datasets/BeIR/scifact-qrels), [BEIR FiQA](https://huggingface.co/datasets/BeIR/fiqa), [FiQA test qrels](https://huggingface.co/datasets/BeIR/fiqa-qrels), [JaQuAD](https://huggingface.co/datasets/mteb/JaQuADRetrieval). Model revisions are pinned in [expanded-models.json](../../experiments/multilingual/expanded-models.json).",
        "",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(lines))
    print(summary["validation"])
    for task in DATA:
        print(
            task,
            sorted(
                [
                    (
                        LABELS[k],
                        round(
                            results[task][k]["runs"]["w8a8-bf16kv"]["metrics"][
                                "ndcg@10"
                            ]
                            * 100,
                            2,
                        ),
                    )
                    for k in SPECS
                ],
                key=lambda x: -x[1],
            ),
        )


if __name__ == "__main__":
    main()
