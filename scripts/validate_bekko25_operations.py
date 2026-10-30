"""Independently verify actual Wasm retrieval, lifecycle and instruction records."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from bekko25_provenance import digest, validate_report_inputs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/multilingual"))
from evaluate import metrics
from global_eval import dataset


def normalize(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--without-reference",
        action="store_true",
        help="verify actual Wasm rankings without the optional Torch reference cache",
    )
    args = parser.parse_args()
    out = ROOT / "benchmarks/bekko25-operations"
    cache = ROOT / ".cache/bekko25"
    target = ROOT / os.environ.get("CARGO_TARGET_DIR", ".cache/bekko25-prod-target")
    report, _ = validate_report_inputs(ROOT, out, target)
    validation_sha256 = digest(out / "validation.json")
    evidence_sha256 = {
        name: digest(out / name)
        for name in [
            "sweep.json",
            "stress.json",
            "live.json",
            "live-sweep.json",
            "recipe-artifact.json",
        ]
    }
    assert len(report["final_checks"]) == 2
    assert report["model"]["layers"] == 13 and report["model"]["max_tokens"] == 256
    assert report["model"]["weights_verified"] and report["model"]["active"]
    wasm = target / "wasm32-unknown-unknown/release/bekko25_canister.wasm"
    assert hashlib.sha256(wasm.read_bytes()).hexdigest() == report["wasm_sha256"]
    sweep = json.loads((out / "sweep.json").read_text())
    stress = json.loads((out / "stress.json").read_text())
    assert [r["tokens"] for r in sweep] == list(range(2, 257))
    assert len(stress) == 102 and len(report["cases"]) == 78
    costs = [r["instructions"] for r in sweep + stress + report["cases"]]
    live = json.loads((out / "live.json").read_text())
    live_sweep = json.loads((out / "live-sweep.json").read_text())
    recipe = json.loads((out / "recipe-artifact.json").read_text())
    assert (
        live["status"]["module_hash"]
        == live_sweep["status"]["module_hash"]
        == "0x" + recipe["deployed_sha256"]
    )
    assert recipe["raw_cargo_sha256"] == report["wasm_sha256"]
    assert recipe["metadata_transform_reproduces_deployed_artifact"]
    assert [r["tokens"] for r in live_sweep["results"]] == list(range(2, 257))
    assert len(live["cases"]) == 9 and all(r["golden_bit_exact"] for r in live["cases"])
    assert len(live["a8_regression_against_original_service"]) == 8 and all(
        r["bit_exact_old_service_vs_new_runtime"]
        for r in live["a8_regression_against_original_service"]
    )
    costs.extend(r["instructions"] for r in live_sweep["results"] + live["cases"])
    for case in report["cases"]:
        v = np.array(case["embedding"])
        assert v.shape == (384,) and np.isfinite(v).all()
        assert abs(np.linalg.norm(v) - 1) < 1e-5
    inputs = json.loads((cache / "retrieval.json").read_text())
    assert len(report["retrieval"]) == len(inputs) == 12
    rows = []
    for selection, saved in zip(inputs, report["retrieval"]):
        task, lang = selection["task"], selection["language"]
        assert saved["task"] == task and saved["language"] == lang
        qs, ds, rel = dataset(task, lang)
        assert selection["query_ids"] == [q["_id"] for q in qs]
        assert selection["document_ids"] == [d["_id"] for d in ds]
        assert selection["qrels"] == {q: sorted(r) for q, r in rel.items()}
        path = cache / saved["vectors_file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == saved["vectors_sha256"]
        v = np.fromfile(path, dtype="<f4").reshape(-1, 384)
        assert len(v) == len(selection["ids"]) == saved["inputs"]
        assert np.isfinite(v).all() and np.allclose(
            np.linalg.norm(v, axis=1), 1, atol=1e-5
        )
        nq = len(qs)
        actual = normalize(v)
        score, ranks = metrics(
            actual[:nq], actual[nq:], qs, ds, rel, selection["parents"]
        )
        for key in ["recall@5", "ndcg@10"]:
            assert abs(score[key] - saved["metrics"][key]) < 1e-12, (task, lang, key)
        assert [r["rank"] for r in ranks] == [
            r["rank"] for r in saved["metrics"]["ranks"]
        ]
        modes = {}
        references = (
            []
            if args.without_reference
            else [("w8a8-bf16kv", ""), ("w8a8-qk13-bf16v", "-masked-v2")]
        )
        for mode, suffix in references:
            ref_path = (
                ROOT
                / f".cache/bekko25-runtime/quality/{task}-{lang}-{mode}{suffix}.npz"
            )
            with np.load(ref_path) as reference:
                assert np.array_equal(reference["parents"], selection["parents"])
                ref = normalize(reference["vectors"])
            reference_hashes = json.loads(
                (ROOT / "benchmarks/bekko25-runtime/reference-sha256.json").read_text()
            )
            expected = reference_hashes[f"{task}/{lang}/{mode}"]
            assert hashlib.sha256(ref_path.read_bytes()).hexdigest() == expected
            rs, rr = metrics(ref[:nq], ref[nq:], qs, ds, rel, selection["parents"])
            modes[mode] = {
                "metrics": rs,
                "ranks": rr,
                "cache_sha256": hashlib.sha256(ref_path.read_bytes()).hexdigest(),
                "cosine_to_wasm_min": float(np.sum(actual * ref, axis=1).min()),
            }
        result = {
            "task": task,
            "language": lang,
            "wasm": {"metrics": score, "ranks": ranks},
            "references": modes,
            "vectors_sha256": saved["vectors_sha256"],
        }
        (out / f"{task}-{lang}-verified.json").write_text(json.dumps(result, indent=2))
        rows.append(result)
        costs.append(saved["maximum_query_instructions"])
    maximum = max(costs)
    # This release reserves at least 2% of the actual 5B query ceiling.
    assert maximum < 4_900_000_000, maximum
    languages = {}
    for lang in ["en", "de", "es", "fr", "ar", "ko"]:
        selected = [r for r in rows if r["language"] == lang]
        languages[lang] = {
            "wasm_recall5": float(
                np.mean([r["wasm"]["metrics"]["recall@5"] for r in selected])
            ),
            "reference_recall5": None
            if args.without_reference
            else float(
                np.mean(
                    [
                        r["references"]["w8a8-bf16kv"]["metrics"]["recall@5"]
                        for r in selected
                    ]
                )
            ),
            "qk13_reference_recall5": None
            if args.without_reference
            else float(
                np.mean(
                    [
                        r["references"]["w8a8-qk13-bf16v"]["metrics"]["recall@5"]
                        for r in selected
                    ]
                )
            ),
        }
    result = {
        "validation_sha256": validation_sha256,
        "build_record_sha256": report["build_record_sha256"],
        "evidence_sha256": evidence_sha256,
        "wasm_sha256": report["wasm_sha256"],
        "retrieval_input_sha256": hashlib.sha256(
            (cache / "retrieval.json").read_bytes()
        ).hexdigest(),
        "maximum_instructions": maximum,
        "query_limit": 5_000_000_000,
        "margin_percent": (1 - maximum / 5_000_000_000) * 100,
        "lengths_checked": len(sweep),
        "deployed_artifact_lengths_checked": len(live_sweep["results"]),
        "deployed_sha256": recipe["deployed_sha256"],
        "stress_cases": len(stress),
        "natural_cases": len(report["cases"]),
        "retrieval_queries": 600,
        "retrieval_embeddings": sum(r["inputs"] for r in report["retrieval"]),
        "languages": languages,
        "torch_references_checked": not args.without_reference,
    }
    # Recheck before publishing: a long independent verification must not bind
    # a changed source tree or another validation run to the computed metrics.
    current, _ = validate_report_inputs(ROOT, out, target)
    assert current == report, "validation changed during verification"
    assert digest(out / "validation.json") == validation_sha256
    assert {name: digest(out / name) for name in evidence_sha256} == evidence_sha256, (
        "evidence changed during verification"
    )
    (out / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
