"""Build-time source fingerprints and validation/report identity checks."""

import argparse
import hashlib
import json
from pathlib import Path

CONFIG = "scripts/bekko25-build-inputs.json"
COMPACT_RAW = "wasm32-unknown-unknown/release/bekko25_compact_canister.wasm"
COMPACT_RECIPE = "compact-recipe/bekko25_compact.wasm"
COMPACT_RECIPE_RECORD = "compact-recipe/record.json"
COMPACT_INPUTS = {".cache/bekko25/cases.json", ".cache/bekko-memory-study/retrieval-ja-hard.json"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(root, kind):
    config = json.loads((root / CONFIG).read_text())
    paths = set(config[f"{kind}_files"])
    for folder in config[f"{kind}_roots"]:
        base = root / folder
        if not base.is_dir():
            raise ValueError(f"missing build source directory: {folder}")
        paths.update(
            str(p.relative_to(root))
            for p in base.rglob("*")
            if p.suffix in {".rs", ".toml", ".did"} and p.is_file()
        )
    return {name: digest(root / name) for name in sorted(paths)}


def require_equal(actual, expected, message):
    if actual != expected:
        raise ValueError(message)


def validate_build(root, target, record):
    require_equal(record.get("schema"), 1, "unsupported build record")
    for kind in ["runtime", "validation"]:
        require_equal(
            snapshot(root, kind),
            record[f"{kind}_source_sha256"],
            f"{kind} sources changed since build; run scripts/build_bekko25.py",
        )
    required = {
        "wasm32-unknown-unknown/release/bekko25_canister.wasm",
        "wasm32-unknown-unknown/release/bekko25_compact_canister.wasm",
        "wasm32-unknown-unknown/release/bekko_canister.wasm",
        "release/bekko25-bench",
        "release/bekko25-load",
        "release/bekko25-live",
        "release/bekko25-compact-verify",
    }
    require_equal(
        set(record["artifacts"]), required, "incomplete build artifact record"
    )
    for name, expected in record["artifacts"].items():
        require_equal(
            digest(target / name), expected, f"build artifact changed: {name}"
        )


def validate_compact_recipe(root, target, record):
    """Bind the ICP recipe artifact to the recorded raw build and configuration."""
    path = target / COMPACT_RECIPE_RECORD
    recipe = json.loads(path.read_text())
    require_equal(recipe.get("schema"), 1, "unsupported compact recipe record")
    require_equal(recipe["build_record_sha256"], digest(target / "build-record.json"), "compact recipe belongs to another build")
    require_equal(recipe["icp_yaml_sha256"], digest(root / "icp.yaml"), "ICP configuration changed after recipe build")
    require_equal(recipe["raw_cargo_sha256"], record["artifacts"][COMPACT_RAW], "compact recipe/raw Wasm mismatch")
    require_equal(recipe["recipe_wasm_sha256"], digest(target / COMPACT_RECIPE), "compact recipe Wasm changed")
    return recipe, digest(path)


def preflight(root, target, executable, retrieval=False, compact=False):
    # Check inexpensive prerequisites before reading the 418MB pack or starting
    # PocketIC. The a8m artifact is used by the final wrong-model upgrade test.
    required = [
        target / "wasm32-unknown-unknown/release/bekko25_canister.wasm",
        target / "wasm32-unknown-unknown/release/bekko_canister.wasm",
        root / ".cache/bekko25/model.pack",
        root / ".cache/bekko25/cases.json",
        target / "build-record.json",
    ]
    if retrieval:
        required.append(root / ".cache/bekko25/retrieval.json")
    for path in required:
        if not path.is_file():
            raise ValueError(f"missing validation prerequisite: {path}")
    record = json.loads((target / "build-record.json").read_text())
    validate_build(root, target, record)
    binary = "release/bekko25-compact-verify" if compact else "release/bekko25-bench"
    require_equal(
        digest(executable),
        record["artifacts"][binary],
        "running harness differs from recorded build",
    )
    inputs = [".cache/bekko25/cases.json"] + (
        [".cache/bekko25/retrieval.json"] if retrieval else []
    )
    if compact:
        inputs.append(".cache/bekko-memory-study/retrieval-ja-hard.json")
    result = {
        "build_record": record,
        "build_record_sha256": digest(target / "build-record.json"),
        "input_sha256": {name: digest(root / name) for name in inputs},
    }
    if compact:
        result["compact_recipe"], result["compact_recipe_record_sha256"] = validate_compact_recipe(root, target, record)
    return result


def validate_report_inputs(root, out, target):
    """Reject stale inputs before publishing any release file."""
    report_path = out / "validation.json"
    report = json.loads(report_path.read_text())
    record = report.get("build_record")
    if not record:
        raise ValueError(
            "validation has no build provenance; rerun the updated harness"
        )
    validate_build(root, target, record)
    require_equal(
        digest(target / "build-record.json"),
        report["build_record_sha256"],
        "build record changed after validation",
    )
    require_equal(
        json.loads((target / "build-record.json").read_text()),
        record,
        "validation/build record mismatch",
    )
    raw = record["artifacts"]["wasm32-unknown-unknown/release/bekko25_canister.wasm"]
    require_equal(report["wasm_sha256"], raw, "validation tested another Wasm")
    for name, expected in report["input_sha256"].items():
        require_equal(
            digest(root / name), expected, f"validation input changed: {name}"
        )
    for name, expected in report["evidence_sha256"].items():
        require_equal(
            digest(out / name), expected, f"validation evidence changed: {name}"
        )
    return report, record


def validate_release_inputs(root, out, target):
    report, record = validate_report_inputs(root, out, target)
    verified = json.loads((out / "verification.json").read_text())
    require_equal(
        verified.get("validation_sha256"),
        digest(out / "validation.json"),
        "verification belongs to another validation run",
    )
    require_equal(
        verified.get("build_record_sha256"),
        report["build_record_sha256"],
        "verification/build record mismatch",
    )
    require_equal(
        verified["wasm_sha256"], report["wasm_sha256"], "verification/Wasm mismatch"
    )
    for name, expected in verified["evidence_sha256"].items():
        require_equal(
            digest(out / name), expected, f"verification evidence changed: {name}"
        )
    recipe = json.loads((out / "recipe-artifact.json").read_text())
    require_equal(
        recipe["raw_cargo_sha256"], report["wasm_sha256"], "recipe/raw Wasm mismatch"
    )
    require_equal(
        recipe["deployed_sha256"],
        verified["deployed_sha256"],
        "recipe/deployed Wasm mismatch",
    )
    return report, record, verified


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check benchmark inputs before starting PocketIC"
    )
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--retrieval", action="store_true")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(preflight(root, args.target_dir, args.executable, args.retrieval, args.compact)))
