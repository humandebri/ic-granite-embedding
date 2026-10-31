"""Build and pin the ICP recipe Wasm for the already recorded compact build."""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from bekko25_provenance import (
    COMPACT_RAW, COMPACT_RECIPE, COMPACT_RECIPE_RECORD, digest, validate_build,
)
from report_bekko25_compact import result_lock

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", type=Path,
                        default=Path(os.environ.get("CARGO_TARGET_DIR", ".cache/bekko25-prod-target")))
    args = parser.parse_args()
    target = (ROOT / args.target_dir).resolve()
    (ROOT / ".cache/bekko25").mkdir(parents=True, exist_ok=True)
    # Coordinate artifact replacement with verification, reporting and the
    # deployment status check. Failed/interrupting builds leave no attestation.
    with result_lock():
        build(target)


def build(target):
    record_path = target / "build-record.json"
    stamp = target / COMPACT_RECIPE_RECORD
    # Do not leave an old recipe attestation usable if this build fails.
    stamp.unlink(missing_ok=True)
    record = json.loads(record_path.read_text())
    validate_build(ROOT, target, record)
    build_hash = digest(record_path)
    icp_yaml_hash = digest(ROOT / "icp.yaml")
    for tool in ("rustc", "cargo"):
        actual = subprocess.check_output([tool, "--version"], cwd=ROOT).decode().strip()
        if actual != record[tool]:
            raise ValueError(f"{tool} version differs from the recorded build")
    env = dict(os.environ, CARGO_TARGET_DIR=str(target))
    for name, value in record["rustflags"].items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    icp_version = subprocess.check_output(["icp", "--version"], cwd=ROOT, env=env).decode().strip()
    subprocess.run(["icp", "build", "bekko25_compact", "-e", "local"], cwd=ROOT, env=env, check=True)
    validate_build(ROOT, target, record)
    if digest(record_path) != build_hash or digest(ROOT / "icp.yaml") != icp_yaml_hash:
        raise ValueError("build or ICP configuration changed during recipe build")
    produced = ROOT / ".icp/cache/artifacts/bekko25_compact"
    expected = digest(produced)
    output = target / COMPACT_RECIPE
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix="compact-recipe-", suffix=".tmp", dir=output.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(produced, temporary)
        if digest(temporary) != expected or digest(produced) != expected:
            raise ValueError("ICP recipe artifact changed during capture")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    validate_build(ROOT, target, record)
    if digest(record_path) != build_hash or digest(ROOT / "icp.yaml") != icp_yaml_hash:
        raise ValueError("build or ICP configuration changed during artifact capture")
    result = {
        "schema": 1,
        "build_record_sha256": build_hash,
        "icp_yaml_sha256": icp_yaml_hash,
        "raw_cargo_sha256": record["artifacts"][COMPACT_RAW],
        "recipe_wasm_sha256": expected,
        "icp_version": icp_version,
    }
    pending = stamp.with_suffix(".tmp")
    pending.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(pending, stamp)
    print(f"Recorded recipe artifact: {output} SHA-256 {expected}")


if __name__ == "__main__":
    main()
