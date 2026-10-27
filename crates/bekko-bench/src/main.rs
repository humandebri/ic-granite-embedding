use candid::{CandidType, Decode, Deserialize, Encode, Principal};
use pocket_ic::PocketIcBuilder;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf, time::Instant};
#[derive(Debug, CandidType, Deserialize)]
struct Embedding {
    embedding: Vec<f32>,
    tokens: u32,
    instructions: u64,
}
#[derive(CandidType, Deserialize, serde::Serialize)]
struct Info {
    model_id: String,
    revision: String,
    dimensions: u32,
    max_tokens: u32,
    weight_pack_sha256: String,
    uploaded_bytes: u64,
    weights_verified: bool,
    active: bool,
    last_error: Option<String>,
}
#[derive(serde::Deserialize)]
struct Case {
    case: String,
    ids: Vec<u32>,
    fp32: Vec<f32>,
    w8a8: Vec<f32>,
}
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let d: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let aa: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum();
    let bb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum();
    d / (aa * bb).sqrt()
}
fn query(pic: &pocket_ic::PocketIc, id: Principal, ids: &[u32]) -> Result<Embedding, String> {
    let b = pic
        .query_call(
            id,
            Principal::anonymous(),
            "embed",
            Encode!(&ids.to_vec()).unwrap(),
        )
        .unwrap();
    Decode!(&b,Result<Embedding,String>).unwrap()
}
fn info(pic: &pocket_ic::PocketIc, id: Principal) -> Info {
    let b = pic
        .query_call(id, Principal::anonymous(), "modelInfo", Encode!().unwrap())
        .unwrap();
    Decode!(&b, Info).unwrap()
}
fn metrics(v: &[Vec<f32>], data: &Value) -> Value {
    let nq = data["query_ids"].as_array().unwrap().len();
    let nd = data["document_ids"].as_array().unwrap().len();
    let parents = data["parents"].as_array().unwrap();
    let mut recall = 0.;
    let mut ndcg = 0.;
    let mut ranks = Vec::new();
    for i in 0..nq {
        let mut scores = vec![f64::NEG_INFINITY; nd];
        for (j, parent) in parents.iter().enumerate() {
            let p = parent.as_u64().unwrap() as usize;
            scores[p] = scores[p].max(cosine(&v[i], &v[nq + j]));
        }
        let mut order: Vec<_> = (0..nd).collect();
        order.sort_by(|a, b| scores[*b].total_cmp(&scores[*a]).then(a.cmp(b)));
        let qid = data["query_ids"][i].as_str().unwrap();
        let good = data["qrels"][qid].as_array().unwrap();
        let hits: Vec<_> = order
            .iter()
            .map(|j| good.contains(&data["document_ids"][*j]))
            .collect();
        let r = hits.iter().take(5).filter(|h| **h).count() as f64 / good.len() as f64;
        let dcg: f64 = hits
            .iter()
            .take(10)
            .enumerate()
            .map(|(k, h)| if *h { 1. / ((k + 2) as f64).log2() } else { 0. })
            .sum();
        let ideal: f64 = (0..good.len().min(10))
            .map(|k| 1. / ((k + 2) as f64).log2())
            .sum();
        recall += r;
        ndcg += dcg / ideal;
        ranks.push(
            json!({"query_id":qid,"rank":hits.iter().position(|h|*h).unwrap()+1,"recall@5":r}),
        );
    }
    json!({"recall@5":recall/nq as f64,"ndcg@10":ndcg/nq as f64,"ranks":ranks})
}
fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cache = root.join(".cache/bekko");
    let target = std::env::var("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| root.join(".cache/bekko-target"));
    let wasm = fs::read(target.join("wasm32-unknown-unknown/release/bekko_canister.wasm")).unwrap();
    let wasm_sha = format!("{:x}", Sha256::digest(&wasm));
    let pack = fs::read(cache.join("model.pack")).unwrap();
    assert_eq!(pack.len(), bekko_core::PACK_BYTES);
    assert_eq!(
        format!("{:x}", Sha256::digest(&pack)),
        bekko_core::PACK_SHA256
    );
    let cases: Vec<Case> =
        serde_json::from_slice(&fs::read(cache.join("cases.json")).unwrap()).unwrap();
    let pic = PocketIcBuilder::new()
        .with_application_subnet()
        .with_server_binary(PathBuf::from(
            std::env::var("POCKET_IC_BIN").expect("PocketIC 11.0.0 binary required"),
        ))
        .with_max_request_time_ms(Some(300_000))
        .build();
    let id = pic.create_canister();
    pic.add_cycles(id, 1_000_000_000_000_000);
    pic.install_canister(id, wasm.clone(), Encode!().unwrap(), None);
    let stranger = Principal::self_authenticating(b"bekko-test-non-controller");
    let mut checks = Vec::new();
    assert!(!info(&pic, id).active);
    assert!(query(&pic, id, &[2, 1]).unwrap_err().contains("not active"));
    for method in ["beginUpload", "activate"] {
        let b = pic
            .update_call(id, stranger, method, Encode!().unwrap())
            .unwrap();
        assert_eq!(
            Decode!(&b,Result<(),String>).unwrap(),
            Err("controller only".into())
        );
    }
    let b = pic
        .update_call(
            id,
            stranger,
            "upload",
            Encode!(&0u64, &serde_bytes::ByteBuf::from(vec![1])).unwrap(),
        )
        .unwrap();
    assert_eq!(
        Decode!(&b,Result<u64,String>).unwrap(),
        Err("controller only".into())
    );
    checks.push("controller-only weight mutation");
    let b = pic
        .update_call(
            id,
            Principal::anonymous(),
            "upload",
            Encode!(&1u64, &serde_bytes::ByteBuf::from(vec![1])).unwrap(),
        )
        .unwrap();
    assert!(Decode!(&b,Result<u64,String>).unwrap().is_err());
    let b = pic
        .update_call(id, Principal::anonymous(), "activate", Encode!().unwrap())
        .unwrap();
    assert!(Decode!(&b,Result<(),String>).unwrap().is_err());
    checks.push("offset and incomplete activation rejected");
    // Interrupted-upload upgrade resets transient digest/progress without inventing verification.
    let b = pic
        .update_call(
            id,
            Principal::anonymous(),
            "upload",
            Encode!(&0u64, &serde_bytes::ByteBuf::from(vec![1])).unwrap(),
        )
        .unwrap();
    Decode!(&b,Result<u64,String>).unwrap().unwrap();
    pic.upgrade_canister(id, wasm.clone(), Encode!().unwrap(), None)
        .unwrap();
    assert_eq!(info(&pic, id).uploaded_bytes, 0);
    assert!(!info(&pic, id).weights_verified);
    checks.push("incomplete upload discarded on upgrade");
    let setup = Instant::now();
    let mut offset = 0;
    for (i, chunk) in pack.chunks(1_800_000).enumerate() {
        if offset + chunk.len() == pack.len() {
            let mut bad = chunk.to_vec();
            bad[0] ^= 1;
            let b = pic
                .update_call(
                    id,
                    Principal::anonymous(),
                    "upload",
                    Encode!(&(offset as u64), &serde_bytes::ByteBuf::from(bad)).unwrap(),
                )
                .unwrap();
            assert!(
                Decode!(&b,Result<u64,String>)
                    .unwrap()
                    .unwrap_err()
                    .contains("SHA256 mismatch")
            );
            assert_eq!(info(&pic, id).uploaded_bytes, offset as u64);
            checks.push("wrong weight checksum rejected without advancing progress");
        }
        let b = pic
            .update_call(
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
        assert_eq!(
            Decode!(&b,Result<u64,String>).unwrap().unwrap(),
            (offset + chunk.len()) as u64
        );
        offset += chunk.len();
        if i % 50 == 0 {
            println!("uploaded {offset}/{}", pack.len());
        }
    }
    drop(pack);
    let loaded = info(&pic, id);
    assert!(loaded.weights_verified && !loaded.active);
    let b = pic
        .update_call(id, Principal::anonymous(), "activate", Encode!().unwrap())
        .unwrap();
    Decode!(&b,Result<(),String>).unwrap().unwrap();
    assert!(info(&pic, id).active);
    let setup_seconds = setup.elapsed().as_secs_f64();
    for ids in [
        vec![],
        vec![2],
        vec![1, 2],
        vec![2, 0, 1],
        vec![2, 256000, 1],
        {
            let mut v = vec![3; 257];
            v[0] = 2;
            v[256] = 1;
            v
        },
    ] {
        assert!(query(&pic, id, &ids).is_err());
    }
    checks.push("bad frames, padding, vocabulary and 257-token inputs rejected");
    let mut results = Vec::new();
    let mut saved = None;
    for c in &cases {
        let got = query(&pic, id, &c.ids).unwrap();
        assert_eq!(got.tokens as usize, c.ids.len());
        assert_eq!(got.embedding.len(), 384);
        assert!(got.embedding.iter().all(|v| v.is_finite()));
        assert!((got.embedding.iter().map(|v| v * v).sum::<f32>().sqrt() - 1.).abs() < 1e-5);
        assert!(got.instructions < 5_000_000_000);
        let quant = cosine(&got.embedding, &c.w8a8);
        assert!(quant > 0.99);
        if c.case == "ja-256" {
            saved = Some((c.ids.clone(), got.embedding.clone()));
        }
        println!(
            "{} instructions={} cosine={quant:.8}",
            c.case, got.instructions
        );
        results.push(json!({"case":c.case,"tokens":c.ids.len(),"instructions":got.instructions,"cosine_to_fp32":cosine(&got.embedding,&c.fp32),"cosine_to_w8a8":quant}));
    }
    // Production heap-delta throttling applies after the 400 MB upload.
    // Let ordinary idle rounds replenish the quota before installing another module.
    pic.advance_time(std::time::Duration::from_secs(3600));
    for _ in 0..500 {
        pic.tick();
    }
    pic.upgrade_canister(id, wasm.clone(), Encode!().unwrap(), None)
        .unwrap();
    let after = info(&pic, id);
    assert!(after.weights_verified && !after.active);
    assert_eq!(after.uploaded_bytes, bekko_core::PACK_BYTES as u64);
    assert!(query(&pic, id, &[2, 1]).is_err());
    let b = pic
        .update_call(id, stranger, "activate", Encode!().unwrap())
        .unwrap();
    assert!(Decode!(&b,Result<(),String>).unwrap().is_err());
    let b = pic
        .update_call(id, Principal::anonymous(), "activate", Encode!().unwrap())
        .unwrap();
    Decode!(&b,Result<(),String>).unwrap().unwrap();
    let (ids, expected) = saved.unwrap();
    let restored = query(&pic, id, &ids).unwrap();
    assert_eq!(restored.embedding, expected);
    checks.push(
        "verified weights survive upgrade; controller activation restores identical embedding",
    );
    let retrieval = if cache.join("retrieval.json").exists() {
        let data: Value =
            serde_json::from_slice(&fs::read(cache.join("retrieval.json")).unwrap()).unwrap();
        let inputs = data["inputs"].as_array().unwrap();
        let mut vectors = Vec::new();
        let mut maximum = 0;
        for (i, ids) in inputs.iter().enumerate() {
            let ids: Vec<u32> = serde_json::from_value(ids.clone()).unwrap();
            let output = query(&pic, id, &ids).unwrap();
            maximum = maximum.max(output.instructions);
            assert_eq!(output.embedding.len(), 384);
            vectors.push(output.embedding);
            if i % 50 == 0 {
                println!("retrieval {i}/{}", inputs.len());
            }
        }
        let fp: Vec<Vec<f32>> = serde_json::from_value(data["fp32"].clone()).unwrap();
        let quant: Vec<Vec<f32>> = serde_json::from_value(data["w8a8"].clone()).unwrap();
        let bytes: Vec<u8> = vectors
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        fs::write(cache.join("wasm-retrieval.f32"), &bytes).unwrap();
        json!({"inputs":inputs.len(),"queries":data["query_ids"].as_array().unwrap().len(),"documents":data["document_ids"].as_array().unwrap().len(),"maximum_query_instructions":maximum,"wasm":metrics(&vectors,&data),"fp32":metrics(&fp,&data),"w8a8_reference":metrics(&quant,&data),"vectors_sha256":format!("{:x}",Sha256::digest(&bytes))})
    } else {
        Value::Null
    };
    let status = pic.canister_status(id, None).unwrap();
    let report = json!({"model":info(&pic,id),"canister_id":id.to_text(),"subnet":"Application/Production","wasm_sha256":wasm_sha,"setup_seconds":setup_seconds,"wasm_memory_bytes":status.memory_metrics.wasm_memory_size.to_string(),"stable_memory_bytes":status.memory_metrics.stable_memory_size.to_string(),"checks":checks,"cases":results,"retrieval":retrieval});
    let out = root.join("benchmarks/bekko");
    fs::create_dir_all(&out).unwrap();
    fs::write(
        out.join("validation.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    println!(
        "validated {} cases, {} lifecycle checks",
        cases.len(),
        checks.len()
    );
}
