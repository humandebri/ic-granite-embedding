//! Compare pinned old and new release Wasm on one disposable PocketIC subnet.
use candid::{CandidType, Decode, Deserialize, Encode, Principal};
use pocket_ic::PocketIcBuilder;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf, time::Instant};

const MIN_HEAP_REDUCTION: u64 = 390_000_000;
const CASES_SHA256: &str = "a81b5c4d75c4b5ebe30717d36b152c13ec4f6702cb649a619ce1966fbeafa1ff";
const JAPANESE_SHA256: &str = "60a37fa51fab4a2ce0280dd24d6c7f2c8db987b76e7774067ebd46c6c77c98db";

#[derive(Debug, CandidType, Deserialize)]
struct Embedding {
    embedding_space: String,
    embedding: Vec<f32>,
    tokens: u32,
    instructions: u64,
}
#[derive(CandidType, Deserialize)]
struct Info {
    active: bool,
    weights_verified: bool,
    uploaded_bytes: u64,
    embedding_space: String,
    runtime_id: String,
}
fn info(pic: &pocket_ic::PocketIc, id: Principal) -> Info {
    let b = pic
        .query_call(id, Principal::anonymous(), "modelInfo", Encode!().unwrap())
        .unwrap();
    Decode!(&b, Info).unwrap()
}
fn query(pic: &pocket_ic::PocketIc, id: Principal, ids: &[u32]) -> Embedding {
    let b = pic
        .query_call(
            id,
            Principal::anonymous(),
            "embed",
            Encode!(&ids.to_vec()).unwrap(),
        )
        .unwrap();
    Decode!(&b, Result<Embedding, String>).unwrap().unwrap()
}
fn try_query(pic: &pocket_ic::PocketIc, id: Principal, ids: &[u32]) -> Result<Embedding, String> {
    let b = pic
        .query_call(
            id,
            Principal::anonymous(),
            "embed",
            Encode!(&ids.to_vec()).unwrap(),
        )
        .map_err(|e| format!("{e:?}"))?;
    Decode!(&b, Result<Embedding, String>).map_err(|e| format!("{e:?}"))?
}

fn compare(old: &Embedding, new: Result<&Embedding, &str>) -> Result<(), String> {
    let new = new.map_err(str::to_string)?;
    if old.tokens != new.tokens || old.embedding_space != new.embedding_space {
        return Err("embedding metadata differs".into());
    }
    if !old
        .embedding
        .iter()
        .map(|x| x.to_bits())
        .eq(new.embedding.iter().map(|x| x.to_bits()))
    {
        return Err("embedding differs at the bit level".into());
    }
    Ok(())
}

fn check_heap_reduction(before: u64, after: u64) -> Result<(), String> {
    match before.checked_sub(after) {
        Some(reduction) if reduction >= MIN_HEAP_REDUCTION => Ok(()),
        _ => Err(format!(
            "heap reduction is below {MIN_HEAP_REDUCTION} bytes: {before} -> {after}"
        )),
    }
}

fn memory_bytes(size: &candid::Nat) -> u64 {
    u64::try_from(&size.0).expect("memory size exceeds u64")
}

fn check_fixture(data: &[u8], expected: &str) -> Result<(), String> {
    let actual = format!("{:x}", Sha256::digest(data));
    if actual != expected {
        return Err(format!(
            "fixture SHA-256 mismatch: expected {expected}, got {actual}"
        ));
    }
    Ok(())
}

fn read_fixture(path: &str, expected: &str) -> Vec<u8> {
    let data = fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    check_fixture(&data, expected).unwrap_or_else(|e| panic!("{path}: {e}"));
    data
}

fn main() {
    // The same lifecycle and bit-exact sweep also verifies the subsequent
    // INT8-resident linear-weight optimization against runtime 2.
    let verify_linear_i8 = std::env::var("BEKKO25_VERIFY_LINEAR_I8").as_deref() == Ok("1");
    let old = fs::read(std::env::var("BEKKO25_BASELINE_WASM").expect("baseline Wasm path"))
        .expect("read baseline Wasm");
    let new =
        fs::read(std::env::var("BEKKO25_NEW_WASM").expect("new Wasm path")).expect("read new Wasm");
    // Reject stale fixtures before starting PocketIC or uploading the model.
    let cases_bytes = read_fixture(".cache/bekko25/cases.json", CASES_SHA256);
    let japanese_bytes = read_fixture(
        ".cache/bekko-memory-study/retrieval-ja-hard.json",
        JAPANESE_SHA256,
    );
    let pack = fs::read(".cache/bekko25/model.pack").unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&pack)),
        bekko25_core::PACK_SHA256
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(&old)),
        std::env::var("BEKKO25_BASELINE_SHA256").expect("expected baseline SHA-256"),
        "unexpected baseline Wasm"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(&new)),
        std::env::var("BEKKO25_NEW_SHA256").expect("expected new SHA-256"),
        "unexpected new Wasm"
    );
    assert_ne!(
        Sha256::digest(&old),
        Sha256::digest(&new),
        "baseline and new Wasm must differ"
    );
    let binary = PathBuf::from(std::env::var("POCKET_IC_BIN").expect("PocketIC binary"));
    let pic = PocketIcBuilder::new()
        .with_application_subnet()
        .with_server_binary(binary)
        .with_max_request_time_ms(Some(300_000))
        .build();
    let id = pic.create_canister();
    pic.add_cycles(id, 1_000_000_000_000_000);
    pic.install_canister(id, old.clone(), Encode!().unwrap(), None);
    let upload_start = Instant::now();
    for (i, chunk) in pack.chunks(1_800_000).enumerate() {
        let b = pic
            .update_call(
                id,
                Principal::anonymous(),
                "upload",
                Encode!(
                    &(i as u64 * 1_800_000),
                    &serde_bytes::ByteBuf::from(chunk.to_vec())
                )
                .unwrap(),
            )
            .unwrap();
        Decode!(&b, Result<u64, String>).unwrap().unwrap();
        if i % 50 == 0 {
            eprintln!("upload chunk {i}");
        }
    }
    let upload_seconds = upload_start.elapsed().as_secs_f64();
    let initial = info(&pic, id);
    assert!(initial.weights_verified);
    assert_eq!(initial.embedding_space, bekko25_core::EMBEDDING_SPACE);
    assert_eq!(
        initial.runtime_id,
        if verify_linear_i8 {
            "bekko25-runtime-2"
        } else {
            "bekko25-runtime-1"
        }
    );
    let start = Instant::now();
    let b = pic
        .update_call(id, Principal::anonymous(), "activate", Encode!().unwrap())
        .unwrap();
    Decode!(&b, Result<(), String>).unwrap().unwrap();
    let old_activation = start.elapsed().as_secs_f64();
    let before = pic.canister_status(id, None).unwrap();
    let mut inputs = Vec::new();
    let mut seed = 0x29102026u32;
    for n in 2..=256 {
        let mut ids = vec![2];
        for _ in 0..n - 2 {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            ids.push(3 + seed % (bekko25_core::VOCAB - 3));
        }
        ids.push(1);
        inputs.push(ids);
    }
    let cases: serde_json::Value = serde_json::from_slice(&cases_bytes).unwrap();
    for c in cases
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| [64, 128, 256].contains(&c["ids"].as_array().unwrap().len()))
    {
        inputs.push(
            c["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect(),
        );
    }
    let ja: serde_json::Value = serde_json::from_slice(&japanese_bytes).unwrap();
    let japanese = ja[0]["ids"].as_array().unwrap();
    for c in japanese
        .iter()
        .take(20)
        .chain(japanese.iter().skip(50).take(20))
    {
        inputs.push(
            c.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect(),
        );
    }
    let mut repeated = vec![3; 256];
    repeated[0] = 2;
    repeated[255] = 1;
    inputs.push(repeated);
    for token in [1, 2, 3, 127, 65535, 255999] {
        for n in [
            2, 3, 7, 8, 31, 32, 63, 64, 65, 127, 128, 129, 191, 192, 193, 255, 256,
        ] {
            let mut ids = vec![token; n];
            ids[0] = 2;
            ids[n - 1] = 1;
            inputs.push(ids);
        }
    }
    assert_eq!(inputs.len(), 416, "unexpected input fixture count");
    let mut oversized = vec![3; 257];
    oversized[0] = 2;
    oversized[256] = 1;
    let invalid = vec![
        vec![],
        vec![2],
        vec![2, 0, 1],
        vec![2, bekko25_core::VOCAB, 1],
        oversized,
    ];
    let old_errors: Vec<_> = invalid
        .iter()
        .map(|ids| {
            try_query(&pic, id, ids)
                .err()
                .expect("baseline accepted invalid input")
        })
        .collect();
    assert_eq!(
        old_errors,
        [
            "supported input: 2..=256 tokens including BOS/EOS",
            "supported input: 2..=256 tokens including BOS/EOS",
            "invalid token ID or padding",
            "invalid token ID or padding",
            "supported input: 2..=256 tokens including BOS/EOS",
        ]
    );
    let old_outputs: Vec<_> = inputs
        .iter()
        .enumerate()
        .map(|(i, x)| {
            if i % 50 == 0 {
                eprintln!("old query {i}/{}", inputs.len());
            }
            query(&pic, id, x)
        })
        .collect();
    pic.advance_time(std::time::Duration::from_secs(3600));
    for _ in 0..500 {
        pic.tick();
    }
    pic.upgrade_canister(id, new.clone(), Encode!().unwrap(), None)
        .unwrap();
    let after_upgrade = info(&pic, id);
    assert!(
        !after_upgrade.active
            && after_upgrade.weights_verified
            && after_upgrade.uploaded_bytes == pack.len() as u64
    );
    assert_eq!(after_upgrade.embedding_space, initial.embedding_space);
    assert_eq!(after_upgrade.runtime_id, bekko25_core::RUNTIME_ID);
    let stranger = Principal::self_authenticating(b"stream-test-not-controller");
    for method in ["activate", "beginUpload"] {
        let denied = pic
            .update_call(id, stranger, method, Encode!().unwrap())
            .unwrap();
        assert_eq!(
            Decode!(&denied, Result<(), String>).unwrap(),
            Err("controller only".into()),
            "unauthorized {method} returned the wrong error"
        );
    }
    let denied = pic
        .update_call(
            id,
            stranger,
            "upload",
            Encode!(&0u64, &serde_bytes::ByteBuf::from(vec![1])).unwrap(),
        )
        .unwrap();
    assert_eq!(
        Decode!(&denied, Result<u64, String>).unwrap(),
        Err("controller only".into()),
        "unauthorized upload returned the wrong error"
    );
    let start = Instant::now();
    let b = pic
        .update_call(id, Principal::anonymous(), "activate", Encode!().unwrap())
        .unwrap();
    Decode!(&b, Result<(), String>).unwrap().unwrap();
    let new_activation = start.elapsed().as_secs_f64();
    let after = pic.canister_status(id, None).unwrap();
    assert_eq!(
        before.memory_metrics.stable_memory_size,
        after.memory_metrics.stable_memory_size
    );
    let before_heap = memory_bytes(&before.memory_metrics.wasm_memory_size);
    let after_heap = memory_bytes(&after.memory_metrics.wasm_memory_size);
    if verify_linear_i8 {
        assert!(
            before_heap
                .checked_sub(after_heap)
                .is_some_and(|n| n >= 10_000_000),
            "linear weights did not reduce heap sufficiently"
        );
    } else {
        check_heap_reduction(before_heap, after_heap)
            .expect("new Wasm did not reduce heap as expected");
    }
    for (ids, expected) in invalid.iter().zip(&old_errors) {
        assert_eq!(
            &try_query(&pic, id, ids)
                .err()
                .expect("accepted invalid input"),
            expected
        );
    }
    let mut comparisons = Vec::new();
    let mut failures = 0;
    for (i, (ids, old)) in inputs.iter().zip(&old_outputs).enumerate() {
        if i % 50 == 0 {
            eprintln!("new query {i}/{}", inputs.len());
        }
        match try_query(&pic, id, ids) {
            Ok(new) => {
                let equal = compare(old, Ok(&new));
                let bit_equal = equal.is_ok();
                let result = equal.and_then(|()| {
                    if new.instructions >= 4_900_000_000 {
                        Err("query instruction count exceeded release gate".into())
                    } else {
                        Ok(())
                    }
                });
                failures += usize::from(result.is_err());
                comparisons.push(json!({"case":i,"tokens":ids.len(),"before":old.instructions,"after":new.instructions,"bit_equal":bit_equal,"error":result.err()}));
            }
            Err(error) => {
                failures += 1;
                comparisons.push(
                    json!({"case":i,"tokens":ids.len(),"before":old.instructions,"error":error}),
                );
            }
        }
    }
    let reset = pic
        .update_call(
            id,
            Principal::anonymous(),
            "beginUpload",
            Encode!().unwrap(),
        )
        .unwrap();
    Decode!(&reset, Result<(), String>).unwrap().unwrap();
    let state = info(&pic, id);
    assert!(!state.active && !state.weights_verified && state.uploaded_bytes == 0);
    assert!(
        try_query(&pic, id, &[2, 1])
            .unwrap_err()
            .contains("not active")
    );
    let first = &pack[..1_800_000];
    let args = Encode!(&0u64, &serde_bytes::ByteBuf::from(first.to_vec())).unwrap();
    for _ in 0..2 {
        let result = pic
            .update_call(id, Principal::anonymous(), "upload", args.clone())
            .unwrap();
        assert_eq!(
            Decode!(&result, Result<u64, String>).unwrap().unwrap(),
            first.len() as u64
        );
    }
    println!("{}", serde_json::to_string_pretty(&json!({"old_wasm_sha256":format!("{:x}",Sha256::digest(&old)),"new_wasm_sha256":format!("{:x}",Sha256::digest(&new)),"pack_sha256":bekko25_core::PACK_SHA256,"cases_sha256":CASES_SHA256,"japanese_sha256":JAPANESE_SHA256,"before_heap_bytes":before.memory_metrics.wasm_memory_size.to_string(),"after_heap_bytes":after.memory_metrics.wasm_memory_size.to_string(),"before_stable_bytes":before.memory_metrics.stable_memory_size.to_string(),"after_stable_bytes":after.memory_metrics.stable_memory_size.to_string(),"old_activation_seconds":old_activation,"new_activation_seconds":new_activation,"upload_seconds":upload_seconds,"comparisons":comparisons})).unwrap());
    if failures != 0 {
        eprintln!("{failures} comparison(s) failed");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Embedding {
        Embedding {
            embedding_space: "space".into(),
            embedding: vec![0.0, 1.0],
            tokens: 2,
            instructions: 1,
        }
    }

    #[test]
    fn comparison_rejects_errors_metadata_and_bit_differences() {
        let old = sample();
        assert!(compare(&old, Ok(&sample())).is_ok());
        assert!(compare(&old, Err("query trapped")).is_err());
        let mut changed = sample();
        changed.embedding[0] = -0.0;
        assert!(compare(&old, Ok(&changed)).is_err());
        changed = sample();
        changed.tokens += 1;
        assert!(compare(&old, Ok(&changed)).is_err());
        changed = sample();
        changed.embedding.pop();
        assert!(compare(&old, Ok(&changed)).is_err());
    }

    #[test]
    fn heap_check_rejects_insufficient_savings_or_growth() {
        assert_eq!(
            memory_bytes(&candid::Nat::from(568_000_512u64)),
            568_000_512
        );
        assert!(check_heap_reduction(600_000_000, 200_000_000).is_ok());
        assert!(check_heap_reduction(600_000_000, 300_000_000).is_err());
        assert!(check_heap_reduction(200_000_000, 600_000_000).is_err());
    }

    #[test]
    fn fixture_check_rejects_modified_inputs() {
        let digest = format!("{:x}", Sha256::digest(b"fixture"));
        assert!(check_fixture(b"fixture", &digest).is_ok());
        assert!(check_fixture(b"changed", &digest).is_err());
    }
}
