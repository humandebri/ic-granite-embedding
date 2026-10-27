//! Upload the pinned weight pack to the local icp-cli project.
use candid::Encode;
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf, process::Command};
fn call(method: &str, args: &str, file: Option<&PathBuf>) {
    let mut command = Command::new("icp");
    command.args([
        "canister",
        "call",
        "bekko",
        method,
        "-e",
        "local",
        "--candid",
        "crates/bekko-canister/bekko.did",
        "--output",
        "candid",
    ]);
    if let Some(file) = file {
        command
            .args(["--args-format", "bin", "--args-file"])
            .arg(file);
    } else {
        command.arg(args);
    }
    let result = command.output().expect("run icp");
    let text = String::from_utf8_lossy(&result.stdout);
    assert!(
        result.status.success() && text.contains("Ok"),
        "{method}: {}{}",
        text,
        String::from_utf8_lossy(&result.stderr)
    );
}
fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    std::env::set_current_dir(&root).unwrap();
    if std::env::args().any(|v| v == "--activate-only") {
        call("activate", "()", None);
        println!("activated from durable weights");
        return;
    }
    let pack = fs::read(root.join(".cache/bekko/model.pack")).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&pack)),
        bekko_core::PACK_SHA256
    );
    call("beginUpload", "()", None);
    let path = root.join(".cache/bekko/upload-args.bin");
    for (i, chunk) in pack.chunks(1_800_000).enumerate() {
        fs::write(
            &path,
            Encode!(
                &((i * 1_800_000) as u64),
                &serde_bytes::ByteBuf::from(chunk.to_vec())
            )
            .unwrap(),
        )
        .unwrap();
        call("upload", "", Some(&path));
        if i % 50 == 0 {
            println!("uploaded {}/{}", (i * 1_800_000 + chunk.len()), pack.len());
        }
    }
    call("activate", "()", None);
    fs::remove_file(path).unwrap();
    println!("Bekko is active on the local network");
}
