"""Verify all saved vectors; evaluate aligned cross-language rankings and paired differences."""

import argparse
import json

import numpy as np
from evaluate import metrics
from global_eval import CACHE, LANGUAGES, MODELS, OUT, ROOT, TASKS, dataset, digest

LABELS = {
    "english": "Granite English",
    "granite": "Granite 97M multi",
    "bekko8": "Bekko a8m",
    "bekko25": "Bekko a25m",
    "e5": "multilingual E5 small",
    "gemma2": "EmbeddingGemma 2",
    "harrier": "Harrier 270M",
}


def paired_ci(a, b):
    # Two domains, each with 50 aligned query IDs; translations are not independent samples.
    diff = np.asarray(a) - np.asarray(b)
    assert diff.shape == (2, 50)
    rng = np.random.default_rng(20261028)
    sampled = [diff[t][rng.integers(0, 50, (5000, 50))].mean(1) for t in range(2)]
    low, high = np.quantile(np.mean(sampled, axis=0), [0.025, 0.975])
    return {
        "difference_pp": float(diff.mean() * 100),
        "ci95_pp": [float(low * 100), float(high * 100)],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--partial", action="store_true")
    args = p.parse_args()
    measured = {}
    vectors = {}
    checks = []
    missing = []
    for model in MODELS:
        langs = ["en"] if model == "english" else LANGUAGES
        for mode in ["fp32", "w8a8-bf16kv"]:
            for task in TASKS:
                for lang in langs:
                    stem = f"{task}-{lang}-{model}-128-{mode}"
                    path = OUT / f"{stem}.json"
                    if not path.exists():
                        missing.append(stem)
                        continue
                    report = json.loads(path.read_text())
                    assert report["model"] == MODELS[model]
                    qs, ds, rel = dataset(task, lang)
                    archive_path = CACHE / "vectors" / f"{stem}.npz"
                    archive = np.load(archive_path)
                    v = archive["vectors"]
                    parents = archive["parents"]
                    assert len(v) == len(qs) + len(parents)
                    assert set(parents.tolist()) == set(range(len(ds)))
                    assert report["vectors_sha256"] == digest(archive_path)
                    assert report["selection_sha256"] == digest(
                        OUT / f"{task}-selection.json"
                    )
                    assert str(archive["text_sha256"]) == report["text_sha256"]
                    assert np.isfinite(v).all() and np.allclose(
                        np.linalg.norm(v, axis=1), 1, atol=1e-5
                    )
                    score, ranks = metrics(
                        v[: len(qs)], v[len(qs) :], qs, ds, rel, parents
                    )
                    assert ranks == report["ranks"]
                    for metric in score:
                        assert abs(score[metric] - report["metrics"][metric]) < 1e-12
                    assert report["encoding"]["truncated"] == 0
                    measured[(model, mode, task, lang)] = report
                    vectors[(model, mode, task, lang)] = (v, parents)
                    checks.append({"run": stem, "queries": len(qs), "vectors": len(v)})
    if missing and not args.partial:
        raise RuntimeError(f"{len(missing)} incomplete runs: {missing[:5]}")
    summary = {
        "complete": not missing,
        "missing": missing,
        "vector_checks": checks,
        "models": {},
        "paired_comparisons": {},
        "cross_language": {},
        "encoding_devices": {},
        "quantization": {},
    }
    for result in measured.values():
        device = result["encoding"].get("device", "cpu")
        summary["encoding_devices"][device] = (
            summary["encoding_devices"].get(device, 0) + 1
        )
    for model in MODELS:
        langs = ["en"] if model == "english" else LANGUAGES
        table = {}
        for mode in ["fp32", "w8a8-bf16kv"]:
            table[mode] = {}
            for lang in langs:
                found = [measured.get((model, mode, task, lang)) for task in TASKS]
                if all(found):
                    table[mode][lang] = {
                        metric: float(np.mean([r["metrics"][metric] for r in found]))
                        for metric in found[0]["metrics"]
                    }
        summary["models"][model] = table
        cosines = []
        for task in TASKS:
            for lang in langs:
                fp = vectors.get((model, "fp32", task, lang))
                w8 = vectors.get((model, "w8a8-bf16kv", task, lang))
                if fp is not None and w8 is not None:
                    assert np.array_equal(fp[1], w8[1])
                    cosines.extend((fp[0] * w8[0]).sum(1).tolist())
        if cosines:
            changes = {
                lang: (
                    table["w8a8-bf16kv"][lang]["recall@5"]
                    - table["fp32"][lang]["recall@5"]
                )
                * 100
                for lang in langs
                if lang in table["fp32"] and lang in table["w8a8-bf16kv"]
            }
            summary["quantization"][model] = {
                "vector_cosine_min": min(cosines),
                "vector_cosine_mean": float(np.mean(cosines)),
                "recall_change_pp_by_language": changes,
            }
        if model == "english":
            continue
        for mode in ["fp32", "w8a8-bf16kv"]:
            for lang in LANGUAGES[1:]:
                for qlang, dlang in [("en", lang), (lang, "en")]:
                    reports = []
                    for task in TASKS:
                        if (model, mode, task, qlang) not in vectors or (
                            model,
                            mode,
                            task,
                            dlang,
                        ) not in vectors:
                            continue
                        qs, ds, rel = dataset(task, qlang)
                        qv, _ = vectors[(model, mode, task, qlang)]
                        dv, parents = vectors[(model, mode, task, dlang)]
                        score, ranks = metrics(
                            qv[: len(qs)], dv[len(qs) :], qs, ds, rel, parents
                        )
                        reports.append({"task": task, "metrics": score, "ranks": ranks})
                    if len(reports) == 2:
                        summary["cross_language"][
                            f"{model}/{mode}/{qlang}->{dlang}"
                        ] = {
                            "metrics": {
                                m: float(np.mean([r["metrics"][m] for r in reports]))
                                for m in reports[0]["metrics"]
                            },
                            "tasks": reports,
                        }
    for model in MODELS:
        if model in ["english", "bekko8"]:
            continue
        for scope in ["en", "non_english"]:
            langs = ["en"] if scope == "en" else LANGUAGES[1:]
            values = []
            for key in [model, "bekko8"]:
                tasks = []
                for task in TASKS:
                    runs = [
                        measured.get((key, "w8a8-bf16kv", task, lang)) for lang in langs
                    ]
                    if not all(runs):
                        break
                    tasks.append(
                        np.mean(
                            [[r["recall@5"] for r in run["ranks"]] for run in runs],
                            axis=0,
                        )
                    )
                if len(tasks) == 2:
                    values.append(tasks)
            if len(values) == 2:
                summary["paired_comparisons"][f"{model}-bekko8/{scope}"] = paired_ci(
                    *values
                )
    for model in ["gemma2", "harrier"]:
        for scope in ["en", "non_english", "en_to_ar"]:
            arrays = []
            for key in [model, "bekko25"]:
                tasks = []
                for task in TASKS:
                    if scope == "en_to_ar":
                        cross = summary["cross_language"].get(
                            f"{key}/w8a8-bf16kv/en->ar"
                        )
                        if cross is None:
                            break
                        run = next(r for r in cross["tasks"] if r["task"] == task)
                        tasks.append([r["recall@5"] for r in run["ranks"]])
                    else:
                        langs = ["en"] if scope == "en" else LANGUAGES[1:]
                        runs = [
                            measured.get((key, "w8a8-bf16kv", task, lang))
                            for lang in langs
                        ]
                        if not all(runs):
                            break
                        tasks.append(
                            np.mean(
                                [[r["recall@5"] for r in run["ranks"]] for run in runs],
                                axis=0,
                            )
                        )
                if len(tasks) == 2:
                    arrays.append(tasks)
            if len(arrays) == 2:
                summary["paired_comparisons"][f"{model}-bekko25/{scope}"] = paired_ci(
                    *arrays
                )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    if args.partial:
        print("completed", len(checks), "missing", len(missing))
        return
    rows = []
    for key, model in summary["models"].items():
        cells = [
            f"{model['w8a8-bf16kv'][lang]['recall@5'] * 100:.2f}%"
            if lang in model["w8a8-bf16kv"]
            else "—"
            for lang in LANGUAGES
        ]
        rows.append("| " + LABELS[key] + " | " + " | ".join(cells) + " |")
    cross_rows = []
    quant_rows = []
    for key, row in summary["quantization"].items():
        changes = row["recall_change_pp_by_language"]
        foreign = [v for lang, v in changes.items() if lang != "en"]
        foreign_cell = f"{np.mean(foreign):+.2f}" if foreign else "—"
        quant_rows.append(
            f"| {LABELS[key]} | {changes['en']:+.2f} | {foreign_cell} | {row['vector_cosine_min']:.6f} |"
        )
    for key in MODELS:
        if key == "english":
            continue
        cells = [
            f"{summary['cross_language'][f'{key}/w8a8-bf16kv/en->{lang}']['metrics']['recall@5'] * 100:.2f}%"
            for lang in LANGUAGES[1:]
        ]
        cross_rows.append("| " + LABELS[key] + " | " + " | ".join(cells) + " |")
    deltas = []
    for pair, stats in summary["paired_comparisons"].items():
        lo, hi = stats["ci95_pp"]
        deltas.append(
            f"| {pair} | {stats['difference_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |"
        )
    for file in [
        "gemma2-split.json",
        "gemma2-split-extra.json",
        "bekko25-canister.json",
    ]:
        assert (OUT / file).exists(), file
    split = (
        json.loads((OUT / "gemma2-split.json").read_text())["results"]
        + json.loads((OUT / "gemma2-split-extra.json").read_text())["results"]
    )
    wasm_hash = digest(
        ROOT / ".cache/gemma2-target/wasm32-unknown-unknown/release/gemma2_probe.wasm"
    )
    for name in ["gemma2-split.json", "gemma2-split-extra.json"]:
        assert json.loads((OUT / name).read_text())["wasm_sha256"] == wasm_hash
    harrier_validation = json.loads((OUT / "harrier-validation.json").read_text())
    assert not harrier_validation["accepted"], (
        "Update report interpretation after a successful Harrier port"
    )
    runtime_rows = []
    assert {r["case"] for r in split if r["tokens"] == 128} == {
        "en-128",
        "ja-128",
        "ar-128",
        "ko-128",
    }
    for row in split:
        if row["tokens"] != 128:
            continue
        assert "error" not in row
        assert sum(c["layers"] for c in row["calls"]) == 24
        assert [c["start_layer"] for c in row["calls"]] == list(range(0, 24, 2))
        assert row["native_split_exact"] and row["successful_queries"] == 12
        assert max(c["instructions"] for c in row["calls"]) < 5_000_000_000
        runtime_rows.append(
            f"| {row['case']} | {row['successful_queries']} | {row['max_instructions_per_query']:,} | {row['total_instructions']:,} | {row['transfer_bytes']:,} | {row['cosine_to_w8a8']:.8f} |"
        )
    bekko25 = json.loads((OUT / "bekko25-canister.json").read_text())
    b25_refs = {
        r["name"]: r for r in json.loads((CACHE / "bekko25/cases.json").read_text())
    }
    assert {
        r["case"] for r in bekko25["results"] if r["tokens"] == 128 and "error" not in r
    } == {"en-128", "ar-128", "ko-128"}
    b25_rows = []
    for r in bekko25["results"]:
        if "instructions" in r:
            assert r["instructions"] < 5_000_000_000
            a = np.array(r["embedding"], dtype=np.float64)
            b = np.array(b25_refs[r["case"]]["w8a8"], dtype=np.float64)
            assert a.shape == (384,) and np.isfinite(a).all()
            assert abs(np.linalg.norm(a) - 1) < 1e-5
            cosine = float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
            assert abs(cosine - r["cosine_to_w8a8"]) < 1e-12
        else:
            assert "CanisterInstructionLimitExceeded" in r["error"]
        cost = f"{r['instructions']:,}" if "instructions" in r else "limit exceeded"
        cosine_cell = f"{r['cosine_to_w8a8']:.8f}" if "cosine_to_w8a8" in r else "—"
        b25_rows.append(f"| {r['case']} | {cost} | {cosine_cell} |")
    sources = {}
    for file in [
        ROOT / "experiments/multilingual/global_eval.py",
        ROOT / "experiments/multilingual/global_report.py",
        ROOT / "experiments/multilingual/bekko25_probe.py",
        ROOT / "experiments/multilingual/export_harrier.py",
        ROOT / "experiments/multilingual/expanded-models.json",
        ROOT / "experiments/multilingual/expanded.py",
        ROOT / "experiments/multilingual/evaluate.py",
        *sorted((ROOT / "experiments/embeddinggemma2/src").rglob("*.rs")),
    ]:
        sources[str(file.relative_to(ROOT))] = digest(file)
    sources["gemma2_wasm"] = digest(
        ROOT / ".cache/gemma2-target/wasm32-unknown-unknown/release/gemma2_probe.wasm"
    )
    sources["bekko25_wasm"] = digest(
        ROOT
        / ".cache/global-port-target/wasm32-unknown-unknown/release/bekko25_probe.wasm"
    )
    (OUT / "verification.json").write_text(
        json.dumps(
            {
                "vector_recomputed_runs": len(checks),
                "ranked_queries": sum(c["queries"] for c in checks),
                "source_hashes": sources,
                "gemma_128_cases": len(runtime_rows),
                "harrier_numeric_gate_passed": harrier_validation["accepted"],
            },
            indent=2,
        )
    )
    report = f"""# English-first multilingual RAG comparison

Measured locally on 2026-10-28 JST. These are bounded retrieval experiments and disposable local canister probes; no mainnet deployment or change to the Bekko service.

## Retrieval at 128 tokens

Each cell is the equal-domain mean of NanoSciFact and NanoFiQA2018 **parent-document Recall@5**: 50 queries per domain, all their relevant documents plus 128 seeded random distractors. Same query IDs, document IDs and qrels across all six languages; translations were aligned by ID, never row order. Document score is its maximum chunk similarity, then parent IDs are deduplicated. These are not answer-containing chunk recall or generated-answer accuracy.

| W8A8/BF16 KV reference | English | German | Spanish | French | Arabic | Korean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

The English-only Granite baseline is intentionally evaluated only in English. All other models use their pinned revisions from `global_eval.py`. Each input fits 128 tokens including published query/document prefixes and specials, with 32 source-token overlap. Mean/CLS pooling follows the model; Harrier uses last non-padding token and its published web-search instruction. Its adapter matched Sentence Transformers exactly on four EN/AR/KO texts (adapter check (`harrier-adapter.json`, generated locally)). Gemma 2 loads text encoders only. Every vector is L2 normalized. For the original 2026-10-28 evaluation, FP32 and W8A8 results, individual domain scores and per-query ranks are retained in the [historical per-run records](https://github.com/humandebri/ic-granite-embedding/tree/3aaaea994534105e422cfaba6136fe984a5ad0d8/benchmarks/multilingual-global) and [summary.json](https://github.com/humandebri/ic-granite-embedding/blob/3aaaea994534105e422cfaba6136fe984a5ad0d8/benchmarks/multilingual-global/summary.json). Per-run JSON files from later evaluations are local and ignored by Git; pinned selections and model metadata remain tracked.

Completed CPU calculations were retained, and the remaining reference calculations ran on local Apple MPS in F32. A two-text EN/AR CPU/MPS check for each of Gemma 2 and Harrier found maximum FP32 differences below 2.1e-7 (device check (`device-check.json`, generated locally)). This is a limited numerical check, not a proof of bit equivalence; W8 rounding can amplify backend differences. Each encoding record states its device (legacy cached records without a device field were CPU). Timings are diagnostic and are not controlled speed comparisons. Wasm instruction measurements use PocketIC independently of these reference devices.

The public dataset contains translated versions of English retrieval sets, not independently collected native-language questions. Candidate pools are smaller than full NanoBEIR, and there are only 50 shared query IDs per domain; translations do not constitute six independent samples. No training-overlap audit was performed. Results do not establish a universal language ranking. The candidate pool/length differ from the older 256-token comparison, so do not interpret changes against its absolute percentages as regressions.

## Quantization effect

W8A8/BF16 KV minus FP32, in Recall@5 percentage points. Non-English is an equal-language/domain mean. Vector cosine is the minimum across all query/chunk vectors available for the model; high cosine alone does not guarantee identical rankings.

| Model | English change (pp) | Non-English change (pp) | Minimum vector cosine |
| --- | ---: | ---: | ---: |
{chr(10).join(quant_rows)}

## English query → foreign-language documents

The same saved query/document vectors and identical relevance IDs allow cross-language scoring with no translation at inference time. Equal-domain mean Recall@5:

| W8A8/BF16 KV reference | → German | → Spanish | → French | → Arabic | → Korean |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(cross_rows)}

The reverse directions (foreign queries → English documents), both precision modes and per-task ranks are also stored in `summary.json`. These tests rank against one target language at a time; a mixed-language index with duplicate translations was not tested.

## Paired differences against Bekko baselines

Recall@5 differences in percentage points, W8A8 references. English uses the two sets of 50 queries. Non-English averages the five translations per query before resampling; stratified paired bootstrap resamples the 50 query IDs within each of two domains, 5,000 samples, seed 20261028. Intervals are exploratory and not adjusted for multiple comparisons.

| Comparison/scope | Difference (pp) | Paired 95% interval |
| --- | ---: | --- |
{chr(10).join(deltas)}

## Actual Gemma 2 layer continuation

Full-token hidden states are carried by the client across **12 query calls**, two complete layers per call. Each call recomputes only the required projection-only PLE slice from the original token embeddings, which is mathematically equivalent to the original full projection under the per-row/per-token quantization scheme. No input chunk pooling replaces full attention. All 24 layers execute. Native split/full inference is bit-identical on the exact-length 32/64/128/256 fixtures in Japanese and English, plus the added Arabic/Korean 128 fixtures.

PocketIC 11, Application subnet / Production defaults, actual 5B/query ceiling:

| Fixture | Queries | Max instructions/query | Total instructions | Candid request+reply bytes | Cosine to Torch W8 reference |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(runtime_rows)}

These fixtures substitute exported scaled input embeddings for the vocabulary lookup only. All model computation after lookup executes at runtime. Full-vocabulary loading and a production token-ID API were not implemented. Rust/Wasm and Torch remain numerically approximate: dynamic quantization thresholds amplify small floating-point differences, and the original [numerical limitations](../embeddinggemma2/RESULTS.md) still apply. A full Wasm retrieval evaluation was not performed; the quality tables above measure the Torch references.

At 256 tokens, one layer per query passes the first five layers but the first wide global-Attention layer still exceeds 5B. Its completion is not claimed. At 128 tokens, the measured ~9.2 MB aggregate transfer and 12 sequential calls are material costs; mainnet latency was not measured. The client-provided intermediate state is unauthenticated and this API is only an execution prototype.

## Actual Bekko a25m query probe

The isolated port uses the a8m kernels with all 13 layers, the a25m weights and the matching RoPE/Attention settings. Default Production query limit, controller-only local uploads, no changes to the a8m service. The vocabulary lookup is fully loaded for this probe.

| Fixture | Instructions/query | Cosine to Torch W8 reference |
| --- | ---: | ---: |
{chr(10).join(b25_rows)}

Existing measured 128-token references: Bekko a8m about 0.811B/query ([dedicated validation](../bekko/RESULTS.md)); Granite 97M multilingual about 2.393B/query ([isolated validation](../multilingual/RESULTS.md)). E5 and Harrier have no measured canister implementation in this comparison; their reference accuracy does not establish a one-query budget.

## Harrier port validation did not pass

An initial independent 18-layer Harrier prototype was implemented with the same generic W8 SIMD matrix kernel, causal grouped Attention, Gemma 3 RMSNorm, RoPE, GELU and last-token pooling. Native full/split calculation is bit-identical on its fixtures, but agreement with the independent Torch W8 reference failed even the coarse cosine floor of 0.99. The first layer of an English 64-token trace closely agrees, then differences grow through later layers. The discrepancy was narrowed but not resolved, so **no canister execution costs or Wasm retrieval-quality claims are reported for Harrier**. This does not invalidate its separate Torch accuracy results. Evidence: numerical gate and layer trace (`harrier-validation.json`, generated locally), [fixture/model/pack hashes](harrier-fixtures.json).

The initial failed prototype remains under `experiments/embeddinggemma2/src/harrier.rs` for reproducible diagnosis. Its harness exits with code 2 on this gate failure and deliberately skips canister measurements. It is not a deployable Harrier service.

## Interpretation

For same-language retrieval under a 128-token single-query constraint, Bekko a25m is the strongest measured candidate here: its reference English and five-language means exceed a8m, while actual EN/AR/KO queries complete at about 2.608B instructions. These small quality gains have paired intervals that include zero, so they are promising rather than conclusive, and a25m Wasm retrieval remains unmeasured.

Gemma 2 has the highest English reference point estimate, but it takes 12 queries in this prototype and does not beat a25m on the five-language same-language mean. English-query → Arabic-document retrieval is a distinct case where Gemma's reference point estimate is higher; see the cross-language table and paired interval. A choice should therefore distinguish same-language search from cross-language search, rather than infer one winner from the English or Japanese score alone.

## Reproduction and evidence

Pinned public data: `LiquidAI/nanobeir-multilingual-extended` revision `8a4be55eb80b3ed4d2e9a423a5212228c217d426`. Download `{{NanoSciFact,NanoFiQA2018}}_{{queries,corpus,qrels}}_{{en,de,es,fr,ar,ko}}.parquet` to `.cache/multilingual-global/data`. Source file hashes and selected IDs/qrels are in `*-selection.json`. Data attribution follows the [dataset card](https://huggingface.co/datasets/LiquidAI/nanobeir-multilingual-extended) and its original NanoBEIR/SciFact/FiQA sources; redistribution of source texts is avoided here.

Use the existing Python evaluation environment (Torch 2.14.1, Transformers 5.20.2, Sentence Transformers 6.2.0). Download/cache models at the revisions in `global_eval.py`; Gemma 2 and Harrier paths are explicit in `model_path`. Run from the repository root:

```sh
HF_HUB_OFFLINE=1 "$PY" experiments/multilingual/global_eval.py
HF_HUB_OFFLINE=1 "$PY" experiments/multilingual/bekko25_probe.py
CARGO_TARGET_DIR=.cache/global-port-target cargo build --manifest-path .cache/multilingual-global/bekko25/port/Cargo.toml --release -p bekko25-probe --target wasm32-unknown-unknown --locked
CARGO_TARGET_DIR=.cache/gemma2-target cargo build --manifest-path experiments/embeddinggemma2/Cargo.toml --release --lib --target wasm32-unknown-unknown --locked
CARGO_TARGET_DIR=.cache/gemma2-target cargo build --manifest-path experiments/embeddinggemma2/Cargo.toml --release --bins --locked
POCKET_IC_BIN=/path/to/pocket-ic-11 .cache/gemma2-target/release/bekko25
POCKET_IC_BIN=/path/to/pocket-ic-11 .cache/gemma2-target/release/split
# Extra Arabic/Korean Gemma fixtures:
HF_HUB_OFFLINE=1 "$PY" experiments/multilingual/export_gemma2.py --snapshot "$GEMMA_SNAPSHOT" --cache .cache/multilingual-global/gemma-extra --report benchmarks/multilingual-global/gemma-extra-fixtures.json --languages ar,ko --lengths 128
GEMMA_CASES=.cache/multilingual-global/gemma-extra/cases.json GEMMA_OUTPUT=benchmarks/multilingual-global/gemma2-split-extra.json POCKET_IC_BIN=/path/to/pocket-ic-11 .cache/gemma2-target/release/split
# Harrier validation generates an input for global_report.py;
# the recorded failed port exits with code 2 after writing that result.
HF_HUB_OFFLINE=1 "$PY" experiments/multilingual/export_harrier.py
.cache/gemma2-target/release/harrier
"$PY" experiments/multilingual/global_report.py
```

Initial JA/EN Gemma fixtures and the non-vocabulary pack are generated by the earlier `export_gemma2.py` default command. The report generator rederives all measured rankings from cached vectors, checks hashes, selections, finite unit vectors and zero truncation, computes both cross-language directions, and checks the successful 128-token query plans. Cached model/vector/pack files and regenerated per-run results are local and ignored. The report, fixture/model pins, selections and source/build hashes remain tracked; historical detailed results are linked above. See verification.json (`verification.json`, generated locally), Gemma split calls (`gemma2-split.json`, generated locally), Arabic/Korean split calls (`gemma2-split-extra.json`, generated locally), Bekko a25m calls (`bekko25-canister.json`, generated locally), [Bekko source/pack hashes](bekko25-fixtures.json).
"""
    (OUT / "RESULTS.md").write_text(report)
    print(
        "verified",
        len(checks),
        "runs",
        sum(c["queries"] for c in checks),
        "query rankings",
    )


if __name__ == "__main__":
    main()
