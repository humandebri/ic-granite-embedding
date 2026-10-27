"""Build an isolated experimental multilingual port from the current checkout.
The production core/canister source files are never modified.
"""

import hashlib
import json
import shutil
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / ".cache/multilingual/port"
DEST.mkdir(parents=True, exist_ok=True)
provenance = {}
workspace = (ROOT / "Cargo.toml").read_text()
# Preserve the complete workspace so path dependencies and Cargo.lock stay valid.
for member in tomllib.loads(workspace)["workspace"]["members"]:
    target = DEST / member
    shutil.copytree(ROOT / member, target, dirs_exist_ok=True)
    for f in sorted((ROOT / member).rglob("*")):
        if f.is_file():
            provenance[str(f.relative_to(ROOT))] = hashlib.sha256(
                f.read_bytes()
            ).hexdigest()
for f in ["Cargo.lock", "rust-toolchain.toml"]:
    shutil.copyfile(ROOT / f, DEST / f)
shutil.copytree(ROOT / ".cargo", DEST / ".cargo", dirs_exist_ok=True)
(DEST / "Cargo.toml").write_text(workspace)
# Keep the same optimized GEMM/attention implementation; change only model specifics.
p = DEST / "crates/core/src/lib.rs"
s = p.read_text()
for a, b in [
    ("Some(&50281)", "Some(&179934)"),
    ("Some(&50282)", "Some(&179938)"),
    ("ids.contains(&50283)", "ids.contains(&179935)"),
    ("*id >= 50368", "*id >= 180000"),
    ("&[50368, H]", "&[180000, H]"),
    ("[80000f32, 10000f32]", "[150000f32, 160000f32]"),
]:
    assert a in s, a
    s = s.replace(a, b)
a = s.index("fn gelu_gates(")
b = s.index("\nfn max_abs", a)
s = (
    s[:a]
    + """fn gelu_gates(wide: &[f32]) -> Vec<f32> {
    assert_eq!(wide.len() % (2 * I), 0);
    let mut gates = Vec::with_capacity(wide.len() / 2);
    for row in wide.chunks_exact(2 * I) {
        for i in 0..I {
            let v = row[i];
            gates.push(v / (1. + (-v).exp()) * row[I + i]);
        }
    }
    gates
}
"""
    + s[b:]
)
p.write_text(s)
p = DEST / "crates/canister/src/lib.rs"
s = p.read_text().replace("160_000_000", "400_000_000")
p.write_text(s)
(DEST / "crates/bench/src").mkdir(parents=True, exist_ok=True)
shutil.copyfile(ROOT / "crates/bench/Cargo.toml", DEST / "crates/bench/Cargo.toml")
shutil.copyfile(
    Path(__file__).with_name("pocket.rs"), DEST / "crates/bench/src/main.rs"
)
(ROOT / "benchmarks/multilingual/port-source.json").write_text(
    json.dumps(provenance, indent=2)
)
print(DEST)
