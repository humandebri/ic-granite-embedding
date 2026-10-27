use candid::{CandidType, Decode, Deserialize, Encode, Principal};
use pocket_ic::PocketIcBuilder;
use serde::Deserialize as JsonDeserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf, time::Instant};
#[derive(CandidType, Deserialize)]
struct Step {
    next_layer: u32,
    tokens: u32,
    state: serde_bytes::ByteBuf,
    embedding: Vec<f32>,
    instructions: u64,
}
#[derive(JsonDeserialize)]
struct Case {
    case: String,
    ids: Vec<u32>,
    fp32: Vec<f32>,
}
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(a, b)| *a as f64 * *b as f64).sum();
    let aa: f64 = a.iter().map(|a| (*a as f64).powi(2)).sum();
    let bb: f64 = b.iter().map(|b| (*b as f64).powi(2)).sum();
    dot / (aa * bb).sqrt()
}
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let root = PathBuf::from(&args[1]);
    let pack = fs::read(root.join(".cache/multilingual/granite/model.pack")).unwrap();
    let cases: Vec<Case> = serde_json::from_slice(
        &fs::read(root.join(".cache/multilingual/granite/cases.json")).unwrap(),
    )
    .unwrap();
    let wasm = fs::read(root.join(
        ".cache/multilingual/port/target/wasm32-unknown-unknown/release/granite_canister.wasm",
    ))
    .unwrap();
    let wasm_hash = format!("{:x}", Sha256::digest(&wasm));
    let pack_hash = format!("{:x}", Sha256::digest(&pack));
    let pic = PocketIcBuilder::new()
        .with_application_subnet()
        .with_server_binary(PathBuf::from(std::env::var("POCKET_IC_BIN").unwrap()))
        .with_max_request_time_ms(Some(300_000))
        .build();
    let id = pic.create_canister();
    pic.add_cycles(id, 1_000_000_000_000_000);
    pic.install_canister(id, wasm, Encode!().unwrap(), None);
    for (i, chunk) in pack.chunks(1_800_000).enumerate() {
        pic.update_call(
            id,
            Principal::anonymous(),
            "upload",
            Encode!(
                &((i * 1_800_000) as u64),
                &serde_bytes::ByteBuf::from(chunk.to_vec())
            )
            .unwrap(),
        )
        .unwrap();
    }
    let t = Instant::now();
    let b = pic
        .update_call(id, Principal::anonymous(), "activate", Encode!().unwrap())
        .unwrap();
    Decode!(&b,Result<(),String>).unwrap().unwrap();
    let activation = t.elapsed().as_secs_f64();
    drop(pack);
    let status = pic.canister_status(id, None).unwrap();
    let mut results = Vec::new();
    for c in cases {
        let t = Instant::now();
        let reply = pic.query_call(
            id,
            Principal::anonymous(),
            "startEmbedding",
            Encode!(&c.ids).unwrap(),
        );
        let item = match reply {
            Err(e) => json!({"case":c.case,"tokens":c.ids.len(),"error":format!("{e:?}")}),
            Ok(b) => match Decode!(&b,Result<Step,String>).unwrap() {
                Err(e) => json!({"case":c.case,"error":e}),
                Ok(s) => {
                    assert_eq!(s.next_layer, 12);
                    assert_eq!(s.tokens as usize, c.ids.len());
                    assert!(s.state.is_empty());
                    assert_eq!(s.embedding.len(), 384);
                    assert!(s.embedding.iter().all(|v| v.is_finite()));
                    let norm = s.embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
                    assert!((norm - 1.).abs() < 1e-5);
                    json!({"case":c.case,"tokens":c.ids.len(),"instructions":s.instructions,"seconds":t.elapsed().as_secs_f64(),"cosine_to_fp32":cosine(&s.embedding,&c.fp32),"embedding":s.embedding})
                }
            },
        };
        println!(
            "{}",
            json!({"case":item["case"],"instructions":item["instructions"],"cosine":item["cosine_to_fp32"],"error":item["error"]})
        );
        results.push(item);
    }
    fs::write(root.join("benchmarks/multilingual/pocket-granite.json"),serde_json::to_vec_pretty(&json!({"wasm_sha256":wasm_hash,"pack_sha256":pack_hash,"subnet":"application/production","pocket_ic_server":std::env::var("POCKET_IC_BIN").unwrap(),"activation_seconds":activation,"wasm_memory_bytes":status.memory_metrics.wasm_memory_size.to_string(),"cases":results})).unwrap()).unwrap();
}
