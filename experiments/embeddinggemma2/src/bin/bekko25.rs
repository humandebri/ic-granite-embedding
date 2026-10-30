use candid::{Decode, Encode, Principal};
use pocket_ic::PocketIcBuilder;
use serde::Deserialize;
use serde_json::json;
use std::{fs, path::PathBuf};
#[derive(Deserialize)]
struct Case {
    name: String,
    ids: Vec<u32>,
    fp32: Vec<f32>,
    w8a8: Vec<f32>,
}
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum::<f64>()
        / a.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt()
        / b.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt()
}
fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let pack = fs::read(root.join(".cache/multilingual-global/bekko25/model.pack")).unwrap();
    let cases: Vec<Case> = serde_json::from_slice(
        &fs::read(root.join(".cache/multilingual-global/bekko25/cases.json")).unwrap(),
    )
    .unwrap();
    let wasm = fs::read(
        root.join(".cache/global-port-target/wasm32-unknown-unknown/release/bekko25_probe.wasm"),
    )
    .unwrap();
    let pic = PocketIcBuilder::new()
        .with_application_subnet()
        .with_server_binary(PathBuf::from(std::env::var("POCKET_IC_BIN").unwrap()))
        .with_max_request_time_ms(Some(300_000))
        .build();
    let id = pic.create_canister();
    pic.add_cycles(id, 1_000_000_000_000_000);
    pic.install_canister(id, wasm, Encode!().unwrap(), None);
    let mut offset = 0;
    for chunk in pack.chunks(1_700_000) {
        pic.update_call(
            id,
            Principal::anonymous(),
            "upload",
            Encode!(
                &(offset as u64),
                &serde_bytes::ByteBuf::from(chunk.to_vec())
            )
            .unwrap(),
        )
        .unwrap();
        offset += chunk.len();
    }
    pic.update_call(id, Principal::anonymous(), "activate", Encode!().unwrap())
        .unwrap();
    let mut results = Vec::new();
    for c in cases {
        let row = match pic.query_call(
            id,
            Principal::anonymous(),
            "embed",
            Encode!(&c.ids).unwrap(),
        ) {
            Ok(reply) => {
                let (v, instructions) = Decode!(&reply, Vec<f32>, u64).unwrap();
                assert_eq!(v.len(), 384);
                assert!(v.iter().all(|x| x.is_finite()));
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                assert!((norm - 1.).abs() < 1e-5);
                let similarity = cosine(&v, &c.w8a8);
                assert!(similarity > 0.99, "Gross reference divergence");
                json!({"case":c.name,"tokens":c.ids.len(),"instructions":instructions,"cosine_to_fp32":cosine(&v,&c.fp32),"cosine_to_w8a8":similarity,"embedding":v})
            }
            Err(e) => json!({"case":c.name,"tokens":c.ids.len(),"error":format!("{e:?}")}),
        };
        println!(
            "{} tokens {} instructions {} error {}",
            row["case"], row["tokens"], row["instructions"], row["error"]
        );
        results.push(row);
        fs::write(root.join("benchmarks/multilingual-global/bekko25-canister.json"),serde_json::to_vec_pretty(&json!({"runtime":"PocketIC 11 Application Production","query_limit":5_000_000_000u64,"results":results})).unwrap()).unwrap();
    }
}
