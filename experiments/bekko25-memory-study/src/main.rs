use std::{collections::BTreeSet, fs, time::Instant};
use bekko25_core::Model;
use serde_json::{Value, json};

fn main() {
    let cases: Vec<Value> = serde_json::from_slice(&fs::read(".cache/bekko25/cases.json").unwrap()).unwrap();
    let cases: Vec<_> = cases.iter().filter(|c| c["ids"].as_array().unwrap().len() == 256).collect();
    let ids: Vec<Vec<u32>> = cases.iter().map(|c| c["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()).collect();
    let names: Vec<_> = cases.iter().map(|c| c["case"].as_str().unwrap()).collect();
    let used: BTreeSet<u32> = ids.iter().flatten().copied().collect();
    let mut pack = fs::read(".cache/bekko25/model.pack").unwrap();
    let n = u64::from_le_bytes(pack[..8].try_into().unwrap()) as usize;
    let desc: Value = serde_json::from_slice(&pack[8..8+n]).unwrap();
    let offset = 8+n+desc["embeddings.tok_embeddings.weight"]["offset"].as_u64().unwrap() as usize;
    let mut results = Vec::new();
    let mut reference: Vec<Vec<f32>> = Vec::new();
    for mode in ["original", "fp16", "bf16"] {
        if mode != "original" {
            pack = fs::read(".cache/bekko25/model.pack").unwrap();
            for id in &used {
                for i in 0..384 {
                    let pos = offset + (*id as usize * 384 + i)*4;
                    let x = f32::from_le_bytes(pack[pos..pos+4].try_into().unwrap());
                    let y = if mode == "fp16" { half::f16::from_f32(x).to_f32() } else { half::bf16::from_f32(x).to_f32() };
                    pack[pos..pos+4].copy_from_slice(&y.to_le_bytes());
                }
            }
        }
        let model = Model::load(&pack).unwrap();
        for (idx, input) in ids.iter().enumerate() {
            let start = Instant::now();
            let out = model.full(input).unwrap();
            let seconds = start.elapsed().as_secs_f64();
            if mode == "original" { reference.push(out); }
            else {
                let r = &reference[idx];
                let dot: f64 = out.iter().zip(r).map(|(a,b)| *a as f64 * *b as f64).sum();
                let norm = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
                let max_abs = out.iter().zip(r).map(|(a,b)| (a-b).abs()).fold(0_f32, f32::max);
                results.push(json!({"case":names[idx],"mode":mode,"cosine":dot/(norm(&out)*norm(r)),"max_abs":max_abs,"seconds":seconds}));
            }
            eprintln!("{mode} {} {seconds:.3}s", names[idx]);
        }
    }
    println!("{}",serde_json::to_string_pretty(&json!({"cases":names,"unique_tokens":used.len(),"results":results,"note":"native core, quantized only rows used in these inputs; not Wasm or retrieval ranking"})).unwrap());
}
