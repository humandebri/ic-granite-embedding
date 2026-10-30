//! Fail fast before starting PocketIC; Python uses only the standard library.
use serde_json::Value;
use std::{path::Path, process::Command};

pub fn preflight(root: &Path, target: &Path, retrieval: bool) -> Value {
    let mut command =
        Command::new(std::env::var_os("BEKKO25_PYTHON").unwrap_or_else(|| "python3".into()));
    command
        .arg(root.join("scripts/bekko25_provenance.py"))
        .arg("--target-dir")
        .arg(target)
        .arg("--executable")
        .arg(std::env::current_exe().expect("running benchmark path"));
    if retrieval {
        command.arg("--retrieval");
    }
    let output = command
        .output()
        .expect("preflight requires Python 3 (BEKKO25_PYTHON)");
    assert!(
        output.status.success(),
        "validation preflight failed before PocketIC:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("preflight provenance record")
}
