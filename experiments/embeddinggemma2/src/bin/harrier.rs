use candid::{Decode, Encode, Principal};
use gemma2_probe::harrier::Harrier;
use pocket_ic::PocketIcBuilder;
use serde::Deserialize;
use serde_json::json;
use std::{fs, path::PathBuf};
#[derive(Deserialize)]
struct Case {
    name: String,
    input: Vec<f32>,
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
    let cache = root.join(".cache/multilingual-global/harrier");
    let pack = fs::read(cache.join("text.pack")).unwrap();
    let cases: Vec<Case> =
        serde_json::from_slice(&fs::read(cache.join("cases.json")).unwrap()).unwrap();
    let model = Harrier::load(&pack);
    let mut refs = Vec::new();
    let mut validation = Vec::new();
    let mut trace_report = Vec::new();
    for c in &cases {
        if c.name == "en-64" {
            let mut trace = Vec::new();
            model.forward_trace(&c.input, Some(&mut trace));
            let reference: Vec<Vec<f32>> =
                serde_json::from_slice(&fs::read(cache.join("trace.json")).unwrap()).unwrap();
            for (i, (a, b)) in trace.iter().zip(&reference).enumerate() {
                trace_report.push(json!({"layer":i,"cosine":cosine(a,b),"max_abs_difference":a.iter().zip(b).map(|(a,b)|(a-b).abs()).fold(0f32,f32::max)}));
                println!(
                    "layer {i} cosine {} maxdiff {}",
                    cosine(a, b),
                    a.iter()
                        .zip(b)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max)
                );
            }
        }
        let full = model.forward(&c.input);
        let cos = cosine(&full, &c.w8a8);
        println!("native {} cosine {}", c.name, cos);
        let mut state = c.input.clone();
        let mut out = Vec::new();
        for start in (0..18).step_by(2) {
            (state, out) = model.step(&state, start, 2);
        }
        assert_eq!(full, out, "Native full/split must be identical");
        validation.push(json!({"case":c.name,"cosine_to_w8a8":cos,"native_split_exact":true,"accepted":cos>0.99}));
        refs.push(full);
    }
    let accepted = validation.iter().all(|r| r["accepted"].as_bool().unwrap());
    fs::write(root.join("benchmarks/multilingual-global/harrier-validation.json"),serde_json::to_vec_pretty(&json!({"accepted":accepted,"cosine_floor":0.99,"results":validation,"en_64_layer_trace":trace_report,"note":"Gate failed: no canister execution results should be inferred from this prototype."})).unwrap()).unwrap();
    if !accepted {
        eprintln!("Harrier numerical gate failed; canister measurements skipped");
        std::process::exit(2);
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
    pic.update_call(
        id,
        Principal::anonymous(),
        "activateHarrier",
        Encode!().unwrap(),
    )
    .unwrap();
    let mut results = Vec::new();
    for (c, native) in cases.iter().zip(&refs) {
        let mut row = json!({"case":c.name,"tokens":c.input.len()/640,"native_split_exact":true,"native_cosine_to_w8a8":cosine(native,&c.w8a8)});
        match pic.query_call(
            id,
            Principal::anonymous(),
            "harrierEmbed",
            Encode!(&c.input).unwrap(),
        ) {
            Ok(reply) => {
                let (v, cost) = Decode!(&reply, Vec<f32>, u64).unwrap();
                assert_eq!(v.len(), 640);
                row["single_query"] =
                    json!({"instructions":cost,"cosine_to_w8a8":cosine(&v,&c.w8a8)});
            }
            Err(e) => {
                row["single_query"] = json!({"error":format!("{e:?}")});
            }
        }
        let mut state = c.input.clone();
        let mut output = Vec::new();
        let mut calls = Vec::new();
        let mut bytes = 0usize;
        let mut error = None;
        for start in (0..18).step_by(2) {
            let args = Encode!(&state, &(start as u32), &2u32).unwrap();
            bytes += args.len();
            match pic.query_call(id, Principal::anonymous(), "harrierStep", args) {
                Ok(reply) => {
                    bytes += reply.len();
                    let (s, v, cost) = Decode!(&reply, Vec<f32>, Vec<f32>, u64).unwrap();
                    assert!(cost < 5_000_000_000);
                    calls.push(json!({"start_layer":start,"layers":2,"instructions":cost}));
                    state = s;
                    output = v;
                }
                Err(e) => {
                    error = Some(format!("{e:?}"));
                    break;
                }
            }
        }
        row["successful_queries"] = json!(calls.len());
        row["calls"] = json!(calls);
        row["transfer_bytes"] = json!(bytes);
        if let Some(e) = error {
            row["error"] = json!(e);
        } else {
            assert_eq!(output.len(), 640);
            assert!(output.iter().all(|x| x.is_finite()));
            row["total_instructions"] = json!(
                calls
                    .iter()
                    .map(|r| r["instructions"].as_u64().unwrap())
                    .sum::<u64>()
            );
            row["max_instructions_per_query"] = json!(
                calls
                    .iter()
                    .map(|r| r["instructions"].as_u64().unwrap())
                    .max()
                    .unwrap()
            );
            row["cosine_to_native_full"] = json!(cosine(&output, native));
            row["cosine_to_w8a8"] = json!(cosine(&output, &c.w8a8));
            row["cosine_to_fp32"] = json!(cosine(&output, &c.fp32));
            row["embedding"] = json!(output);
        }
        println!(
            "{} queries {} max {} error {}",
            c.name, row["successful_queries"], row["max_instructions_per_query"], row["error"]
        );
        results.push(row);
        fs::write(root.join("benchmarks/multilingual-global/harrier-canister.json"),serde_json::to_vec_pretty(&json!({"runtime":"PocketIC 11 Application Production","query_limit":5_000_000_000u64,"results":results})).unwrap()).unwrap();
    }
}
