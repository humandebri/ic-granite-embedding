"""Summarize the pinned Japanese retrieval fixture from the Wasm comparison."""
import hashlib
import json
import math
import os
from pathlib import Path
from contextlib import contextmanager
from bekko25_provenance import COMPACT_INPUTS, digest, validate_build, validate_compact_recipe

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / ".cache/bekko-memory-study/retrieval-ja-hard.json"
RESULT = ROOT / ".cache/bekko25/compact-verification.json"
FIXTURE_SHA256 = "60a37fa51fab4a2ce0280dd24d6c7f2c8db987b76e7774067ebd46c6c77c98db"
PACK_SHA256 = "71b8c47a0a508dd5cd3e704d83934e900c8e634d2c52e7a02a85e882fa5be9c4"
BASELINE_PACK_SHA256 = "8fc88d56d68f550d1a920bab7d1ec42af9449240048393bd6daabbfcfa861724"


def normalized(values):
    if len(values) != 384 or not all(math.isfinite(v) for v in values):
        raise ValueError("invalid embedding")
    norm = math.sqrt(sum(v * v for v in values))
    if norm == 0:
        raise ValueError("zero embedding")
    return [v / norm for v in values]


@contextmanager
def result_lock():
    path = ROOT / ".cache/bekko25/compact-verification.lock"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as e:
        raise ValueError("compact verification is still running or its lock needs inspection") from e
    try:
        os.close(fd)
        yield
    finally:
        path.unlink(missing_ok=True)


def report():
    with result_lock():
        return _report_locked()


def _report_locked():
    data = FIXTURE.read_bytes()
    if hashlib.sha256(data).hexdigest() != FIXTURE_SHA256:
        raise ValueError("Japanese fixture checksum mismatch")
    fixture = json.loads(data)[0]
    run = json.loads(RESULT.read_text())
    target = Path(os.environ.get("CARGO_TARGET_DIR", ".cache/bekko25-prod-target"))
    if not target.is_absolute():
        target = ROOT / target
    provenance = run.get("build_provenance")
    if not provenance:
        raise ValueError("verification has no build provenance; rerun compact verification")
    record = provenance["build_record"]
    validate_build(ROOT, target, record)
    if digest(target / "build-record.json") != provenance["build_record_sha256"]:
        raise ValueError("build record changed after verification")
    recipe, recipe_record_sha = validate_compact_recipe(ROOT, target, record)
    if provenance.get("compact_recipe") != recipe or provenance.get("compact_recipe_record_sha256") != recipe_record_sha:
        raise ValueError("compact recipe record changed after verification")
    if run["int8_wasm_sha256"] != recipe["recipe_wasm_sha256"]:
        raise ValueError("verification tested another compact recipe Wasm")
    inputs = provenance.get("input_sha256", {})
    if set(inputs) != COMPACT_INPUTS:
        raise ValueError("compact verification input record is incomplete")
    for name, expected in inputs.items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"validation input changed: {name}")
    if run["int8_pack_sha256"] != PACK_SHA256 or run["original_pack_sha256"] != BASELINE_PACK_SHA256:
        raise ValueError("pack checksum mismatch")
    rows = run["comparisons"]
    nq = len(fixture["query_ids"])
    nd = len(fixture["document_ids"])
    if nq != 50 or nd != 50 or len(rows) != 475:
        raise ValueError("incomplete retrieval comparisons")
    ranks = {}
    for mode in ("before", "after"):
        vectors = [normalized(row[mode]) for row in rows[:nq + nd]]
        ranks[mode] = []
        for query in vectors[:nq]:
            scores = [sum(a * b for a, b in zip(query, doc)) for doc in vectors[nq:]]
            ranks[mode].append(sorted(range(nd), key=lambda i: -scores[i]))
    out = {}
    for mode, ranking in ranks.items():
        positions = []
        for qid, order in zip(fixture["query_ids"], ranking):
            positions.append(next(i for i, doc in enumerate(order, 1)
                                  if fixture["document_ids"][doc] in fixture["qrels"][qid]))
        out[mode] = {f"hits_at_{k}": sum(i <= k for i in positions) for k in (1, 3, 5)}
        out[mode]["mrr"] = sum(1 / i for i in positions) / nq
        out[mode]["positions"] = positions
    old, new = out["before"]["positions"], out["after"]["positions"]
    old_metrics, new_metrics = out["before"], out["after"]
    if (new_metrics["hits_at_5"] < old_metrics["hits_at_5"]
            or new_metrics["hits_at_1"] + 1 < old_metrics["hits_at_1"]
            or new_metrics["mrr"] + 0.02 < old_metrics["mrr"]):
        raise ValueError("compact retrieval quality fell below the regression gate")
    for row in rows:
        a, b = normalized(row["before"]), normalized(row["after"])
        if sum(x * y for x, y in zip(a, b)) < 0.97:
            raise ValueError("compact embedding similarity fell below the regression gate")
        if row["after_instructions"] >= 4_900_000_000:
            raise ValueError("compact instructions exceeded the regression gate")
    summary = {
        "queries": nq,
        "documents": nd,
        "old": {k: v for k, v in out["before"].items() if k != "positions"},
        "compact": {k: v for k, v in out["after"].items() if k != "positions"},
        "correct_rank_improved": sum(a > b for a, b in zip(old, new)),
        "correct_rank_worsened": sum(a < b for a, b in zip(old, new)),
        "top5_set_changed": sum(set(a[:5]) != set(b[:5]) for a, b in zip(ranks["before"], ranks["after"])),
        "sampled_cases": len(rows),
        "max_compact_instructions": max(row["after_instructions"] for row in rows),
        "baseline_wasm_sha256": run["baseline_wasm_sha256"],
        "compact_wasm_sha256": run["int8_wasm_sha256"],
        "raw_compact_wasm_sha256": recipe["raw_cargo_sha256"],
        "icp_yaml_sha256": recipe["icp_yaml_sha256"],
        "fixture_sha256": FIXTURE_SHA256,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    report()
