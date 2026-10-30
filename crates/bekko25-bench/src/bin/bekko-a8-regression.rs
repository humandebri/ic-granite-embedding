//! Check a8m output across the shared-core change on a disposable subnet.
use candid::{CandidType, Decode, Deserialize, Encode, Principal};
use pocket_ic::PocketIcBuilder;
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf};
const CASES_SHA256: &str = "bd26ef17aad08d45136169a42c9044885819991987595a397901d78e267a4c00";

#[derive(Debug, CandidType, Deserialize)]
struct Embedding {
    embedding: Vec<f32>,
    tokens: u32,
    instructions: u64,
}

#[derive(CandidType, Deserialize)]
struct Info {
    active: bool,
    weights_verified: bool,
    uploaded_bytes: u64,
}

fn info(pic: &pocket_ic::PocketIc, id: Principal) -> Info {
    let bytes = pic
        .query_call(id, Principal::anonymous(), "modelInfo", Encode!().unwrap())
        .unwrap();
    Decode!(&bytes, Info).unwrap()
}

fn query(pic: &pocket_ic::PocketIc, id: Principal, ids: &[u32]) -> Embedding {
    let bytes = pic
        .query_call(
            id,
            Principal::anonymous(),
            "embed",
            Encode!(&ids.to_vec()).unwrap(),
        )
        .unwrap();
    Decode!(&bytes, Result<Embedding, String>).unwrap().unwrap()
}

fn main() {
    let old = fs::read(std::env::var("BEKKO8_BASELINE_WASM").expect("baseline Wasm path")).unwrap();
    let new = fs::read(std::env::var("BEKKO8_NEW_WASM").expect("new Wasm path")).unwrap();
    for (bytes, var) in [
        (&old, "BEKKO8_BASELINE_SHA256"),
        (&new, "BEKKO8_NEW_SHA256"),
    ] {
        assert_eq!(
            format!("{:x}", Sha256::digest(bytes)),
            std::env::var(var).expect("expected Wasm hash")
        );
    }
    assert_ne!(
        Sha256::digest(&old),
        Sha256::digest(&new),
        "baseline and new Wasm must differ"
    );
    let pack = fs::read(".cache/bekko/model.pack").unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&pack)),
        "1feddfa11f8e232a30b39c4965328a396fa9388dc65bb06cad043045e2dc56ef"
    );
    let cases_bytes = fs::read(".cache/bekko/cases.json").unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&cases_bytes)),
        CASES_SHA256,
        "a8m fixture changed"
    );
    let cases: serde_json::Value = serde_json::from_slice(&cases_bytes).unwrap();
    let inputs: Vec<Vec<u32>> = cases
        .as_array()
        .unwrap()
        .iter()
        .map(|c| serde_json::from_value(c["ids"].clone()).unwrap())
        .collect();
    assert_eq!(inputs.len(), 15);
    let pic = PocketIcBuilder::new()
        .with_application_subnet()
        .with_server_binary(PathBuf::from(
            std::env::var("POCKET_IC_BIN").expect("PocketIC binary"),
        ))
        .with_max_request_time_ms(Some(300_000))
        .build();
    let id = pic.create_canister();
    pic.add_cycles(id, 1_000_000_000_000_000);
    pic.install_canister(id, old, Encode!().unwrap(), None);
    for (i, chunk) in pack.chunks(1_800_000).enumerate() {
        let bytes = pic
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
        Decode!(&bytes, Result<u64, String>).unwrap().unwrap();
        if i % 50 == 0 {
            eprintln!("a8m upload chunk {i}");
        }
    }
    let bytes = pic
        .update_call(id, Principal::anonymous(), "activate", Encode!().unwrap())
        .unwrap();
    Decode!(&bytes, Result<(), String>).unwrap().unwrap();
    let before: Vec<_> = inputs.iter().map(|ids| query(&pic, id, ids)).collect();
    pic.advance_time(std::time::Duration::from_secs(3600));
    for _ in 0..500 {
        pic.tick();
    }
    pic.upgrade_canister(id, new, Encode!().unwrap(), None)
        .unwrap();
    let state = info(&pic, id);
    assert!(!state.active && state.weights_verified && state.uploaded_bytes == pack.len() as u64);
    let bytes = pic
        .update_call(id, Principal::anonymous(), "activate", Encode!().unwrap())
        .unwrap();
    Decode!(&bytes, Result<(), String>).unwrap().unwrap();
    for (ids, old) in inputs.iter().zip(before.iter()) {
        let new = query(&pic, id, ids);
        assert_eq!(old.tokens, new.tokens);
        assert!(new.instructions < 5_000_000_000);
        assert!(
            old.embedding
                .iter()
                .map(|x| x.to_bits())
                .eq(new.embedding.iter().map(|x| x.to_bits()))
        );
    }
    println!("a8m: {} cases bit-identical after upgrade", inputs.len());
}
