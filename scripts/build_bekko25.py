"""Build both canisters and host tools while recording their exact source inputs."""

import argparse
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from bekko25_provenance import digest, snapshot

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path(os.environ.get("CARGO_TARGET_DIR", ".cache/bekko25-prod-target")),
    )
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    target = (ROOT / args.target_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    stamp = target / "build-record.json"
    # A failed/interrupted build must not leave an apparently fresh attestation.
    stamp.unlink(missing_ok=True)
    before = {
        f"{kind}_source_sha256": snapshot(ROOT, kind)
        for kind in ["runtime", "validation"]
    }
    env = dict(os.environ, CARGO_TARGET_DIR=str(target))
    base = ["cargo", "build", "--release", "--locked"] + (
        ["--offline"] if args.offline else []
    )
    subprocess.run(
        base
        + [
            "-p",
            "bekko25-canister",
            "-p",
            "bekko25-compact-canister",
            "-p",
            "bekko-canister",
            "--target",
            "wasm32-unknown-unknown",
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    subprocess.run(
        base + ["-p", "bekko25-bench", "--bins"], cwd=ROOT, env=env, check=True
    )
    after = {
        f"{kind}_source_sha256": snapshot(ROOT, kind)
        for kind in ["runtime", "validation"]
    }
    if before != after:
        raise ValueError("sources changed during build; rerun from a stable checkout")
    names = [
        "wasm32-unknown-unknown/release/bekko25_canister.wasm",
        "wasm32-unknown-unknown/release/bekko25_compact_canister.wasm",
        "wasm32-unknown-unknown/release/bekko_canister.wasm",
        "release/bekko25-bench",
        "release/bekko25-load",
        "release/bekko25-live",
        "release/bekko25-compact-verify",
    ]
    record = {
        "schema": 1,
        "built_at_utc": datetime.now(UTC).isoformat(),
        **before,
        "artifacts": {name: digest(target / name) for name in names},
        "rustc": subprocess.check_output(["rustc", "--version"]).decode().strip(),
        "cargo": subprocess.check_output(["cargo", "--version"]).decode().strip(),
        "rustflags": {
            key: os.environ.get(key) for key in ["RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS"]
        },
    }
    pending = stamp.with_suffix(".tmp")
    pending.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    pending.replace(stamp)
    print(f"Recorded build provenance: {stamp}")


if __name__ == "__main__":
    main()
