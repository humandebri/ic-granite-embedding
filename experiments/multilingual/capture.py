"""Record exact materialized experiment sources and artifacts after a build."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for name in ["port", "bekko-port"]:
    base = ROOT / ".cache/multilingual" / name
    paths = [
        base / "Cargo.toml",
        base / "Cargo.lock",
        base / "rust-toolchain.toml",
        base / ".cargo/config.toml",
    ]
    paths += sorted(p for p in (base / "crates").rglob("*") if p.is_file())
    manifest = {
        str(p.relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }
    (ROOT / f"benchmarks/multilingual/{name}-materialized.json").write_text(
        json.dumps(manifest, indent=2)
    )
