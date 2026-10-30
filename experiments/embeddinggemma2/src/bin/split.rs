//! End-to-end layer continuation with full-token state, under production query limits.
use candid::{Decode, Encode, Principal};
use gemma2_probe::Model;
use pocket_ic::PocketIcBuilder;
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf};

#[derive(Deserialize)]
struct Case {
    name: String,
    input: Vec<f32>,
    w8a8: Vec<f32>,
    fp32: Vec<f32>,
}
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot = a
        .iter()
        .zip(b)
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum::<f64>();
    dot / a.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt()
        / b.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt()
}
fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cache = root.join(".cache/embeddinggemma2");
    let pack = fs::read(cache.join("text.pack")).unwrap();
    let case_path = std::env::var_os("GEMMA_CASES")
        .map(PathBuf::from)
        .unwrap_or_else(|| cache.join("cases.json"));
    let cases: Vec<Case> = serde_json::from_slice(&fs::read(case_path).unwrap()).unwrap();
    let model = Model::load(&pack);
    let mut native = std::collections::BTreeMap::new();
    for c in &cases {
        if !(c.name.ends_with("-32")
            || c.name.ends_with("-64")
            || c.name.ends_with("-128")
            || c.name.ends_with("-256"))
        {
            continue;
        }
        let full = model.forward(&c.input);
        let mut hidden = c.input.clone();
        let mut output = Vec::new();
        for start in (0..24).step_by(2) {
            (hidden, output) = model.step(&c.input, &hidden, start, 2);
        }
        assert_eq!(
            full, output,
            "Native continuation must be bit-identical to whole inference"
        );
        println!("native split {} exact", c.name);
        native.insert(c.name.clone(), output);
    }
    drop(model);
    let pic = PocketIcBuilder::new()
        .with_application_subnet()
        .with_server_binary(PathBuf::from(std::env::var("POCKET_IC_BIN").unwrap()))
        .with_max_request_time_ms(Some(300_000))
        .build();
    let id = pic.create_canister();
    pic.add_cycles(id, 1_000_000_000_000_000);
    let wasm = fs::read(
        root.join(".cache/gemma2-target/wasm32-unknown-unknown/release/gemma2_probe.wasm"),
    )
    .unwrap();
    let wasm_sha = format!("{:x}", Sha256::digest(&wasm));
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
    drop(pack);
    let mut results = Vec::new();
    for c in &cases {
        let Some(reference) = native.get(&c.name) else {
            continue;
        };
        let n = c.input.len() / 512;
        let count = if n <= 128 { 2 } else { 1 };
        let mut hidden = c.input.clone();
        let mut output = Vec::new();
        let mut calls = Vec::new();
        let mut total_bytes = 0usize;
        let mut error = None;
        for start in (0..24).step_by(count) {
            let args = Encode!(&c.input, &hidden, &(start as u32), &(count as u32)).unwrap();
            total_bytes += args.len();
            match pic.query_call(id, Principal::anonymous(), "step", args) {
                Ok(reply) => {
                    total_bytes += reply.len();
                    let (state, embedding, instructions) =
                        Decode!(&reply, Vec<f32>, Vec<f32>, u64).unwrap();
                    assert!(instructions < 5_000_000_000);
                    if start + count == 24 {
                        assert!(state.is_empty());
                        assert_eq!(embedding.len(), 768);
                    } else {
                        assert_eq!(state.len(), c.input.len());
                        assert!(embedding.is_empty());
                    }
                    calls.push(
                        json!({"start_layer":start,"layers":count,"instructions":instructions}),
                    );
                    hidden = state;
                    output = embedding;
                }
                Err(e) => {
                    error = Some(format!("{e:?}"));
                    break;
                }
            }
        }
        let mut row = json!({"case":c.name,"tokens":n,"query_limit":5_000_000_000u64,"successful_queries":calls.len(),"calls":calls,"transfer_bytes":total_bytes,"native_split_exact":true});
        if let Some(e) = error {
            row["error"] = json!(e);
        } else {
            assert!(output.iter().all(|x| x.is_finite()));
            row["embedding"] = json!(output);
            row["cosine_to_native_full"] = json!(cosine(&output, reference));
            row["cosine_to_w8a8"] = json!(cosine(&output, &c.w8a8));
            row["cosine_to_fp32"] = json!(cosine(&output, &c.fp32));
            row["total_instructions"] = json!(
                calls
                    .iter()
                    .map(|x| x["instructions"].as_u64().unwrap())
                    .sum::<u64>()
            );
            row["max_instructions_per_query"] = json!(
                calls
                    .iter()
                    .map(|x| x["instructions"].as_u64().unwrap())
                    .max()
                    .unwrap()
            );
        }
        println!(
            "split {} {} queries max {} error {}",
            c.name,
            calls.len(),
            row["max_instructions_per_query"],
            row["error"]
        );
        results.push(row);
        let out = std::env::var_os("GEMMA_OUTPUT")
            .map(PathBuf::from)
            .unwrap_or_else(|| root.join("benchmarks/multilingual-global/gemma2-split.json"));
        fs::create_dir_all(out.parent().unwrap()).unwrap();
        fs::write(
            out,
            serde_json::to_vec_pretty(
                &json!({"runtime":"PocketIC 11 Application Production","wasm_sha256":wasm_sha,"results":results}),
            )
            .unwrap(),
        )
        .unwrap();
    }
}
