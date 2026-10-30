//! Check the actual local CLI endpoint against the saved QK13 golden vectors.
use candid::{CandidType, Decode, Deserialize, Encode};
use serde::Deserialize as JsonDeserialize;
use serde_json::json;
use std::{fs, path::PathBuf, process::Command};
#[derive(CandidType, Deserialize)]
struct Embedding {
    embedding: Vec<f32>,
    embedding_space: String,
    tokens: u32,
    instructions: u64,
}
#[derive(JsonDeserialize)]
struct Case {
    name: String,
    ids: Vec<u32>,
}
fn call_at(canister: &str, did: &str, method: &str, args: Vec<u8>) -> Vec<u8> {
    let path = PathBuf::from(format!(
        ".cache/bekko25/live-args-{}.bin",
        std::process::id()
    ));
    fs::write(&path, args).unwrap();
    let out = Command::new("icp")
        .args([
            "canister",
            "call",
            canister,
            method,
            "-e",
            "local",
            "--query",
            "--candid",
            did,
            "--output",
            "hex",
            "--args-format",
            "bin",
            "--args-file",
        ])
        .arg(&path)
        .output()
        .unwrap();
    fs::remove_file(path).unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let hex = String::from_utf8(out.stdout).unwrap();
    let hex = hex.trim().trim_start_matches("0x");
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        .collect()
}
fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    std::env::set_current_dir(root).unwrap();
    let cases: Vec<Case> =
        serde_json::from_slice(&fs::read(".cache/multilingual-global/bekko25/cases.json").unwrap())
            .unwrap();
    let baseline: serde_json::Value =
        serde_json::from_slice(&fs::read("benchmarks/bekko25-runtime/iteration-8.json").unwrap())
            .unwrap();
    let mut results = Vec::new();
    for case in cases {
        let response = call_at(
            "bekko25",
            "crates/bekko25-canister/bekko25.did",
            "embed",
            Encode!(&case.ids).unwrap(),
        );
        let got = Decode!(&response,Result<Embedding,String>)
            .unwrap()
            .unwrap();
        let expected: Vec<f32> = serde_json::from_value(
            baseline["results"]
                .as_array()
                .unwrap()
                .iter()
                .find(|r| r["case"] == case.name)
                .unwrap()["embedding"]
                .clone(),
        )
        .unwrap();
        assert_eq!(got.embedding_space, bekko25_core::EMBEDDING_SPACE);
        assert_eq!(got.tokens as usize, case.ids.len());
        assert!(got.instructions < 4_900_000_000);
        assert_eq!(
            got.embedding
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "{}",
            case.name
        );
        results.push(
            json!({"case":case.name,"instructions":got.instructions,"golden_bit_exact":true}),
        );
    }
    #[derive(CandidType, Deserialize)]
    struct A8Embedding {
        embedding: Vec<f32>,
        tokens: u32,
        instructions: u64,
    }
    let data: serde_json::Value =
        serde_json::from_slice(&fs::read(".cache/bekko/retrieval.json").unwrap()).unwrap();
    let vectors = fs::read(".cache/bekko/wasm-retrieval.f32").unwrap();
    let mut a8_results = Vec::new();
    for index in [0usize, 1, 31, 63, 127, 191, 250, 251] {
        let ids: Vec<u32> = serde_json::from_value(data["inputs"][index].clone()).unwrap();
        let reply = call_at(
            "bekko",
            "crates/bekko-canister/bekko.did",
            "embed",
            Encode!(&ids).unwrap(),
        );
        let old = Decode!(&reply,Result<A8Embedding,String>).unwrap().unwrap();
        let expected: Vec<u32> = vectors[index * 384 * 4..(index + 1) * 384 * 4]
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(
            old.embedding
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            expected,
            "a8m regression at {index}"
        );
        a8_results.push(json!({"input_index":index,"tokens":old.tokens,"instructions_old_service":old.instructions,"bit_exact_old_service_vs_new_runtime":true}));
    }
    let status = Command::new("icp")
        .args(["canister", "status", "bekko25", "-e", "local", "--json"])
        .output()
        .unwrap();
    assert!(status.status.success());
    let status: serde_json::Value = serde_json::from_slice(&status.stdout).unwrap();
    let network = Command::new("icp")
        .args(["network", "status", "--json"])
        .output()
        .unwrap();
    assert!(network.status.success());
    let network: serde_json::Value = serde_json::from_slice(&network.stdout).unwrap();
    if std::env::args().any(|arg| arg == "--sweep") {
        let mut seed = 0x29102026u32;
        let mut rows = Vec::new();
        for n in 2..=256 {
            let mut ids = vec![2];
            for _ in 0..n - 2 {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                ids.push(3 + seed % (bekko25_core::VOCAB - 3));
            }
            ids.push(1);
            let reply = call_at(
                "bekko25",
                "crates/bekko25-canister/bekko25.did",
                "embed",
                Encode!(&ids).unwrap(),
            );
            let got = Decode!(&reply,Result<Embedding,String>).unwrap().unwrap();
            assert_eq!(got.embedding_space, bekko25_core::EMBEDDING_SPACE);
            assert!(
                got.instructions < 4_900_000_000 && got.embedding.iter().all(|v| v.is_finite())
            );
            assert!((got.embedding.iter().map(|v| v * v).sum::<f32>().sqrt() - 1.).abs() < 1e-5);
            rows.push(json!({"tokens":n,"instructions":got.instructions}));
            if n % 32 == 0 {
                println!("live sweep {n}/256");
            }
        }
        fs::write(
            "benchmarks/bekko25-operations/live-sweep.json",
            serde_json::to_vec_pretty(&json!({"status":status,"results":rows})).unwrap(),
        )
        .unwrap();
    }
    fs::write("benchmarks/bekko25-operations/live.json",serde_json::to_vec_pretty(&json!({"network":network,"status":status,"embedding_space":bekko25_core::EMBEDDING_SPACE,"cases":results,"a8_regression_against_original_service":a8_results})).unwrap()).unwrap();
    println!("Local CLI: 9 golden vectors match bit-for-bit");
}
