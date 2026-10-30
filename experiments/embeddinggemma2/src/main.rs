use candid::{Decode, Encode, Principal};
use gemma2_probe::Model;
use pocket_ic::PocketIcBuilder;
use serde::Deserialize;
use serde_json::json;
use std::{fs, path::PathBuf};

#[derive(Deserialize)]
struct Case {
    name: String,
    input: Vec<f32>,
    w8a8: Vec<f32>,
    fp32: Vec<f32>,
}
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(a, b)| *a as f64 * *b as f64).sum();
    dot / a.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt()
        / b.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt()
}
fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let pack = fs::read(root.join(".cache/embeddinggemma2/text.pack")).unwrap();
    let cases: Vec<Case> =
        serde_json::from_slice(&fs::read(root.join(".cache/embeddinggemma2/cases.json")).unwrap())
            .unwrap();
    if std::env::args().any(|a| a == "--native") {
        let model = Model::load(&pack);
        let mut trace = Vec::new();
        model.forward_trace(&cases[0].input, Some(&mut trace));
        let mut trace_report = Vec::new();
        if let Ok(data) = fs::read(root.join(".cache/embeddinggemma2/trace.json")) {
            let reference: Vec<Vec<f32>> = serde_json::from_slice(&data).unwrap();
            for (i, (a, b)) in trace.iter().zip(reference).enumerate() {
                trace_report.push(json!({"layer":i,"cosine":cosine(a,&b),"max_abs_difference":a.iter().zip(&b).map(|(a,b)|(a-b).abs()).fold(0f32,f32::max)}));
                println!(
                    "layer {i} cosine {} maxdiff {}",
                    cosine(a, &b),
                    a.iter()
                        .zip(&b)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max)
                );
            }
        }
        let mut rows = Vec::new();
        for case in &cases {
            let out = model.forward(&case.input);
            let c = cosine(&out, &case.w8a8);
            println!("native {} cosine {}", case.name, c);
            assert!(out.iter().all(|x| x.is_finite()));
            if case.name.ends_with("-2") || case.name.ends_with("-16") {
                assert!(c > 0.999, "native reference mismatch");
            }
            rows.push(json!({"case":case.name,"cosine_to_w8a8":c,"embedding":out}));
        }
        fs::write(
            root.join("benchmarks/embeddinggemma2/native.json"),
            serde_json::to_vec_pretty(&json!({"ja_2_layer_trace":trace_report,"results":rows}))
                .unwrap(),
        )
        .unwrap();
        return;
    }
    let pic = PocketIcBuilder::new()
        .with_application_subnet()
        .with_server_binary(PathBuf::from(
            std::env::var("POCKET_IC_BIN").expect("PocketIC 11 required"),
        ))
        .with_max_request_time_ms(Some(300_000))
        .build();
    let id = pic.create_canister();
    pic.add_cycles(id, 1_000_000_000_000_000);
    let wasm = fs::read(
        root.join(".cache/gemma2-target/wasm32-unknown-unknown/release/gemma2_probe.wasm"),
    )
    .unwrap();
    pic.install_canister(id, wasm, Encode!().unwrap(), None);
    let mut offset = 0u64;
    for chunk in pack.chunks(1_700_000) {
        pic.update_call(
            id,
            Principal::anonymous(),
            "upload",
            Encode!(&offset, &serde_bytes::ByteBuf::from(chunk.to_vec())).unwrap(),
        )
        .unwrap();
        offset += chunk.len() as u64;
    }
    pic.update_call(id, Principal::anonymous(), "activate", Encode!().unwrap())
        .unwrap();
    let mut results = Vec::new();
    for case in cases {
        println!("query {}", case.name);
        let result = pic.query_call(
            id,
            Principal::anonymous(),
            "embed",
            Encode!(&case.input).unwrap(),
        );
        let row = match result {
            Ok(bytes) => {
                let (out, instructions) = Decode!(&bytes, Vec<f32>, u64).unwrap();
                assert_eq!(out.len(), 768);
                assert!(out.iter().all(|x| x.is_finite()));
                let c = cosine(&out, &case.w8a8);
                assert!(c > 0.999, "Wasm reference mismatch {}: {}", case.name, c);
                json!({"case":case.name,"tokens":case.input.len()/512,"instructions":instructions,"cosine_to_w8a8":c,"cosine_to_fp32":cosine(&out,&case.fp32),"embedding":out})
            }
            Err(e) => {
                json!({"case":case.name,"tokens":case.input.len()/512,"error":format!("{e:?}")})
            }
        };
        println!(
            "{}",
            json!({"case":row["case"],"instructions":row["instructions"],"error":row["error"],"cosine":row["cosine_to_w8a8"]})
        );
        let query_failed = row.get("error").is_some();
        results.push(row);
        if query_failed && case.name.starts_with("ja-") {
            println!("update {}", case.name);
            let result = pic.update_call(
                id,
                Principal::anonymous(),
                "embedUpdate",
                Encode!(&case.input).unwrap(),
            );
            let row = match result {
                Ok(bytes) => {
                    let (out, instructions) = Decode!(&bytes, Vec<f32>, u64).unwrap();
                    let c = cosine(&out, &case.w8a8);
                    assert!(out.iter().all(|x| x.is_finite()));
                    json!({"case":case.name,"call":"update","tokens":case.input.len()/512,"instructions":instructions,"cosine_to_w8a8":c,"cosine_to_fp32":cosine(&out,&case.fp32),"embedding":out})
                }
                Err(e) => {
                    json!({"case":case.name,"call":"update","tokens":case.input.len()/512,"error":format!("{e:?}")})
                }
            };
            println!(
                "{}",
                json!({"case":row["case"],"call":"update","instructions":row["instructions"],"error":row["error"],"cosine":row["cosine_to_w8a8"]})
            );
            results.push(row);
        }
        fs::write(root.join("benchmarks/embeddinggemma2/canister.json"),serde_json::to_vec_pretty(&json!({"runtime":"PocketIC 11.0.0 Application Production", "query_limit":5_000_000_000u64,"update_limit":40_000_000_000u64,"fixture_only":true,"results":results})).unwrap()).unwrap();
    }
}
