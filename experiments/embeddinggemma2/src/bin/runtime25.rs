//! Persistent disposable PocketIC instance: upload once, test successive Wasm builds.
use candid::{Decode, Encode, Principal};
use pocket_ic::PocketIcBuilder;
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::PathBuf, time::Duration};
#[derive(Deserialize)]
struct Case {
    name: String,
    ids: Vec<u32>,
    w8a8: Vec<f32>,
}
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum::<f64>()
        / a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt()
        / b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt()
}
fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let out = root.join("benchmarks/bekko25-runtime");
    fs::create_dir_all(&out).unwrap();
    let cases: Vec<Case> = serde_json::from_slice(
        &fs::read(root.join(".cache/multilingual-global/bekko25/cases.json")).unwrap(),
    )
    .unwrap();
    let pack = fs::read(root.join(".cache/multilingual-global/bekko25/model.pack")).unwrap();
    let pic = PocketIcBuilder::new()
        .with_application_subnet()
        .with_server_binary(PathBuf::from(std::env::var("POCKET_IC_BIN").unwrap()))
        .with_max_request_time_ms(Some(300_000))
        .build();
    let id = pic.create_canister();
    pic.add_cycles(id, 1_000_000_000_000_000);
    let initial = std::env::args().nth(1).expect("initial Wasm path");
    let mut path = PathBuf::from(initial);
    let mut baseline = BTreeMap::<String, Vec<f32>>::new();
    if let Ok(path) = std::env::var("RUNTIME_BASELINE") {
        let data: serde_json::Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        for row in data["results"].as_array().unwrap() {
            baseline.insert(
                row["case"].as_str().unwrap().into(),
                serde_json::from_value(row["embedding"].clone()).unwrap(),
            );
        }
    }
    let mut first = true;
    let mut iteration = std::env::var("RUNTIME_ITERATION")
        .ok()
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or(0);
    let mut pending = std::env::args().skip(2);
    loop {
        let is_baseline = baseline.is_empty();
        let wasm = fs::read(&path).unwrap();
        let sha = format!("{:x}", Sha256::digest(&wasm));
        if first {
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
        } else {
            pic.advance_time(Duration::from_secs(3600));
            for _ in 0..500 {
                pic.tick();
            }
            pic.upgrade_canister(id, wasm, Encode!().unwrap(), None)
                .unwrap();
        }
        pic.update_call(id, Principal::anonymous(), "activate", Encode!().unwrap())
            .unwrap();
        let mut results = Vec::new();
        for c in &cases {
            let reply = pic.query_call(
                id,
                Principal::anonymous(),
                "embed",
                Encode!(&c.ids).unwrap(),
            );
            let mut row = json!({"case":c.name,"tokens":c.ids.len()});
            let mut vector = None;
            match reply {
                Ok(b) => {
                    let (v, cost) = Decode!(&b, Vec<f32>, u64).unwrap();
                    assert!(cost < 5_000_000_000);
                    row["instructions"] = json!(cost);
                    vector = Some(v);
                }
                Err(e) => {
                    row["error"] = json!(format!("{e:?}"));
                }
            }
            if c.ids.len() == 256 && (is_baseline || c.name == "en-256") {
                let b = pic
                    .query_call(
                        id,
                        Principal::anonymous(),
                        "start",
                        Encode!(&c.ids).unwrap(),
                    )
                    .unwrap();
                let mut state = Decode!(&b, Vec<f32>).unwrap();
                let mut phases = BTreeMap::<String, u64>::new();
                let mut total = 0;
                for layer in 0..13u32 {
                    let b = pic
                        .query_call(
                            id,
                            Principal::anonymous(),
                            "profileLayer",
                            Encode!(&layer, &state).unwrap(),
                        )
                        .unwrap();
                    let (s, p, cost) = Decode!(&b, Vec<f32>, Vec<(String, u64)>, u64).unwrap();
                    state = s;
                    total += cost;
                    for (name, n) in p {
                        *phases.entry(name).or_default() += n;
                    }
                }
                let b = pic
                    .query_call(
                        id,
                        Principal::anonymous(),
                        "finalize",
                        Encode!(&state).unwrap(),
                    )
                    .unwrap();
                let v = Decode!(&b, Vec<f32>).unwrap();
                if let Some(ref full) = vector {
                    assert_eq!(v, *full, "profile/full outputs differ");
                } else {
                    vector = Some(v);
                }
                println!("PROFILE {} total {} {:?}", c.name, total, phases);
                row["layer_query_instructions"] = json!(total);
                row["phases"] = json!(phases);
            }
            if let Some(v) = vector {
                assert_eq!(v.len(), 384);
                assert!(v.iter().all(|x| x.is_finite()));
                assert!((v.iter().map(|x| x * x).sum::<f32>().sqrt() - 1.).abs() < 1e-5);
                row["cosine_to_w8a8"] = json!(cosine(&v, &c.w8a8));
                if is_baseline {
                    baseline.insert(c.name.clone(), v.clone());
                } else {
                    let equal = v
                        .iter()
                        .zip(&baseline[&c.name])
                        .all(|(a, b)| a.to_bits() == b.to_bits());
                    row["baseline_bit_exact"] = json!(equal);
                    row["cosine_to_baseline"] = json!(cosine(&v, &baseline[&c.name]));
                    if std::env::var_os("RUNTIME_APPROX").is_none() {
                        assert!(equal, "optimization changed {}", c.name);
                    }
                }
                row["embedding"] = json!(v);
            }
            println!(
                "CASE {} instructions {} error {}",
                c.name, row["instructions"], row["error"]
            );
            results.push(row);
        }
        fs::write(out.join(format!("iteration-{iteration}.json")),serde_json::to_vec_pretty(&json!({"wasm":path,"wasm_sha256":sha,"query_limit":5_000_000_000u64,"results":results})).unwrap()).unwrap();
        first = false;
        iteration += 1;
        if std::env::var_os("RUNTIME_ONCE").is_some() {
            break;
        }
        if let Some(next) = pending.next() {
            path = PathBuf::from(next);
        } else {
            break;
        }
    }
}
