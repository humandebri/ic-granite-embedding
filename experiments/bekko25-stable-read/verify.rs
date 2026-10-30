//! Compare an old canister and an upgrade on one disposable PocketIC subnet.
use candid::{CandidType, Decode, Deserialize, Encode, Principal};
use pocket_ic::PocketIcBuilder;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf, time::Instant};

const PROTOTYPE_SHA256: &str = "0ce0de4b515a543a16a47d6f9ac34cc057261ebf317066bbc7010e83c4a15ef9";
const MIN_HEAP_REDUCTION: u64 = 390_000_000;

#[derive(CandidType, Deserialize)]
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

fn main() {
    let old =
        fs::read(".cache/bekko25-prod-target/wasm32-unknown-unknown/release/bekko25_canister.wasm")
            .unwrap();
    let new =
        fs::read(".cache/bekko-stable-target/wasm32-unknown-unknown/release/bekko25_canister.wasm")
            .unwrap();
    let pack = fs::read(".cache/bekko25/model.pack").unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&pack)),
        bekko25_core::PACK_SHA256
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(&old)),
        "c198d9ade07132daf4c9a27151744b9381bb7ebfbbf149c6829bf7f70046cede"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(&new)),
        PROTOTYPE_SHA256,
        "unexpected prototype Wasm"
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
    assert!(info(&pic, id).weights_verified);
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
    let cases: serde_json::Value =
        serde_json::from_slice(&fs::read(".cache/bekko25/cases.json").unwrap()).unwrap();
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
    let ja: serde_json::Value = serde_json::from_slice(
        &fs::read(".cache/bekko-memory-study/retrieval-ja-hard.json").unwrap(),
    )
    .unwrap();
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
    assert_eq!(inputs.len(), 314, "unexpected input fixture count");
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
    let stranger = Principal::self_authenticating(b"stream-test-not-controller");
    let denied = pic
        .update_call(id, stranger, "activate", Encode!().unwrap())
        .unwrap();
    assert!(Decode!(&denied, Result<(), String>).unwrap().is_err());
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
    check_heap_reduction(
        memory_bytes(&before.memory_metrics.wasm_memory_size),
        memory_bytes(&after.memory_metrics.wasm_memory_size),
    )
    .expect("prototype did not reduce heap as expected");
    let mut comparisons = Vec::new();
    let mut failures = 0;
    for (i, (ids, old)) in inputs.iter().zip(&old_outputs).enumerate() {
        if i % 50 == 0 {
            eprintln!("new query {i}/{}", inputs.len());
        }
        match try_query(&pic, id, ids) {
            Ok(new) => {
                let result = compare(old, Ok(&new));
                failures += usize::from(result.is_err());
                comparisons.push(json!({"case":i,"tokens":ids.len(),"before":old.instructions,"after":new.instructions,"bit_equal":result.is_ok(),"error":result.err()}));
            }
            Err(error) => {
                failures += 1;
                comparisons.push(
                    json!({"case":i,"tokens":ids.len(),"before":old.instructions,"error":error}),
                );
            }
        }
    }
    println!("{}", serde_json::to_string_pretty(&json!({"old_wasm_sha256":format!("{:x}",Sha256::digest(&old)),"new_wasm_sha256":format!("{:x}",Sha256::digest(&new)),"pack_sha256":bekko25_core::PACK_SHA256,"before_heap_bytes":before.memory_metrics.wasm_memory_size.to_string(),"after_heap_bytes":after.memory_metrics.wasm_memory_size.to_string(),"before_stable_bytes":before.memory_metrics.stable_memory_size.to_string(),"after_stable_bytes":after.memory_metrics.stable_memory_size.to_string(),"old_activation_seconds":old_activation,"new_activation_seconds":new_activation,"upload_seconds":upload_seconds,"comparisons":comparisons})).unwrap());
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
}
