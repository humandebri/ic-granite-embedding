use candid::{CandidType, Decode, Deserialize, Encode, Principal};
use pocket_ic::{PocketIc, PocketIcBuilder};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

const RESULT_PATH: &str = ".cache/bekko25/compact-verification.json";
const LOCK_PATH: &str = ".cache/bekko25/compact-verification.lock";
const MIN_COSINE: f64 = 0.97;

struct RunGuard {
    lock: PathBuf,
}
impl RunGuard {
    fn start(result: &Path, lock: &Path) -> Result<Self, String> {
        fs::create_dir_all(result.parent().ok_or("result parent")?).map_err(|e| e.to_string())?;
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(lock)
            .map_err(|e| format!("verification lock: {e}"))?;
        let guard = Self { lock: lock.into() };
        if result.exists() {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|e| e.to_string())?
                .as_nanos();
            let backup = result.with_file_name(format!(
                "compact-verification-previous-{}-{nonce}.json",
                std::process::id()
            ));
            fs::rename(result, backup).map_err(|e| format!("archive previous result: {e}"))?;
        }
        Ok(guard)
    }
}
impl Drop for RunGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.lock);
    }
}

fn require_saving(before: u64, after: u64, minimum: u64) -> Result<(), String> {
    if before.checked_sub(after).is_some_and(|n| n >= minimum) {
        Ok(())
    } else {
        Err(format!(
            "memory saving below {minimum}: {before} -> {after}"
        ))
    }
}

fn provenance(target: &Path) -> Result<Value, String> {
    let executable = std::env::current_exe().map_err(|e| e.to_string())?;
    let output = Command::new("python3")
        .args(["scripts/bekko25_provenance.py", "--compact", "--target-dir"])
        .arg(target)
        .arg("--executable")
        .arg(executable)
        .output()
        .map_err(|e| format!("run preflight: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "preflight: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    serde_json::from_slice(&output.stdout).map_err(|e| format!("preflight JSON: {e}"))
}

fn cosine(a: &[f32], b: &[f32]) -> Result<f64, String> {
    if a.len() != 384 || b.len() != 384 {
        return Err("embedding dimensions".into());
    }
    let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
    for (&a, &b) in a.iter().zip(b) {
        let (a, b) = (f64::from(a), f64::from(b));
        if !a.is_finite() || !b.is_finite() {
            return Err("nonfinite embedding".into());
        }
        ab += a * b;
        aa += a * a;
        bb += b * b;
    }
    if aa == 0.0 || bb == 0.0 {
        return Err("zero embedding".into());
    }
    Ok(ab / (aa * bb).sqrt())
}

fn check_retrieval(before: &[Vec<f32>], after: &[Vec<f32>], fixture: &Value) -> Result<(), String> {
    let queries = fixture["query_ids"].as_array().ok_or("query ids")?;
    let documents = fixture["document_ids"].as_array().ok_or("document ids")?;
    let (nq, nd) = (queries.len(), documents.len());
    if nq != 50 || nd != 50 || before.len() < nq + nd || after.len() < nq + nd {
        return Err("retrieval fixture shape".into());
    }
    let scores = |vectors: &[Vec<f32>]| -> Result<(usize, usize, f64), String> {
        let mut hits1 = 0;
        let mut hits5 = 0;
        let mut reciprocal = 0.0;
        for (i, qid) in queries.iter().enumerate() {
            let id = qid.as_str().ok_or("query id")?;
            let relevant = fixture["qrels"][id].as_array().ok_or("qrels")?;
            let mut ranking = (0..nd)
                .map(|j| Ok((j, cosine(&vectors[i], &vectors[nq + j])?)))
                .collect::<Result<Vec<_>, String>>()?;
            ranking.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            let position = ranking
                .iter()
                .position(|(j, _)| relevant.contains(&documents[*j]))
                .ok_or("missing relevant document")?
                + 1;
            hits1 += usize::from(position == 1);
            hits5 += usize::from(position <= 5);
            reciprocal += 1.0 / position as f64;
        }
        Ok((hits1, hits5, reciprocal / nq as f64))
    };
    let (old1, old5, oldmrr) = scores(before)?;
    let (new1, new5, newmrr) = scores(after)?;
    if new5 < old5 || new1 + 1 < old1 || newmrr + 0.02 < oldmrr {
        return Err(format!(
            "retrieval regression: hits@1 {old1}->{new1}, hits@5 {old5}->{new5}, MRR {oldmrr:.6}->{newmrr:.6}"
        ));
    }
    Ok(())
}
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
    embedding_space: String,
    last_error: Option<String>,
}
fn info(p: &PocketIc, id: Principal) -> Info {
    Decode!(
        &p.query_call(id, Principal::anonymous(), "modelInfo", Encode!().unwrap())
            .unwrap(),
        Info
    )
    .unwrap()
}
fn install(p: &PocketIc, wasm: Vec<u8>, pack: &[u8]) -> Principal {
    let id = p.create_canister();
    p.add_cycles(id, 1_000_000_000_000_000);
    p.install_canister(id, wasm, Encode!().unwrap(), None);
    for (i, c) in pack.chunks(1_800_000).enumerate() {
        if i % 50 == 0 {
            eprintln!("{} upload chunk {i}", pack.len());
        }
        let r = p
            .update_call(
                id,
                Principal::anonymous(),
                "upload",
                Encode!(
                    &(i as u64 * 1_800_000),
                    &serde_bytes::ByteBuf::from(c.to_vec())
                )
                .unwrap(),
            )
            .unwrap();
        Decode!(&r,Result<u64,String>).unwrap().unwrap();
    }
    assert!(info(p, id).weights_verified);
    activate(p, id);
    id
}
fn activate(p: &PocketIc, id: Principal) {
    let b = p
        .update_call(id, Principal::anonymous(), "activate", Encode!().unwrap())
        .unwrap();
    Decode!(&b,Result<(),String>).unwrap().unwrap();
    assert!(info(p, id).active);
}
fn run(p: &PocketIc, id: Principal, ids: &[u32]) -> Embedding {
    let r = p
        .query_call(
            id,
            Principal::anonymous(),
            "embed",
            Encode!(&ids.to_vec()).unwrap(),
        )
        .unwrap();
    Decode!(&r,Result<Embedding,String>).unwrap().unwrap()
}
fn digest(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn main() {
    // Invalidate the published result before any preflight failure, while
    // preserving its detailed vectors in an archived local file.
    let _run =
        RunGuard::start(Path::new(RESULT_PATH), Path::new(LOCK_PATH)).expect("start verification");
    let target = PathBuf::from(
        std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| ".cache/bekko25-prod-target".into()),
    );
    let build = provenance(&target).expect("build provenance");
    let base =
        fs::read(std::env::var("BEKKO25_BASELINE_WASM").expect("baseline Wasm path")).unwrap();
    let new = fs::read(std::env::var("BEKKO25_COMPACT_WASM").expect("compact Wasm path")).unwrap();
    assert_eq!(
        digest(&base),
        std::env::var("BEKKO25_BASELINE_SHA256").expect("baseline Wasm hash")
    );
    assert_eq!(
        digest(&new),
        std::env::var("BEKKO25_COMPACT_SHA256").expect("compact Wasm hash")
    );
    assert_eq!(
        digest(&new),
        build["compact_recipe"]["recipe_wasm_sha256"]
            .as_str()
            .expect("recorded compact recipe Wasm"),
        "compact Wasm does not match the recipe artifact"
    );
    assert_ne!(digest(&base), digest(&new));
    let a = fs::read(".cache/bekko25/model.pack").unwrap();
    let b = fs::read(".cache/bekko25/model-int8.pack").unwrap();
    assert_eq!(
        digest(&a),
        "8fc88d56d68f550d1a920bab7d1ec42af9449240048393bd6daabbfcfa861724"
    );
    assert_eq!(
        digest(&b),
        "71b8c47a0a508dd5cd3e704d83934e900c8e634d2c52e7a02a85e882fa5be9c4"
    );
    let raw = fs::read(".cache/bekko-memory-study/retrieval-ja-hard.json").unwrap();
    assert_eq!(
        digest(&raw),
        "60a37fa51fab4a2ce0280dd24d6c7f2c8db987b76e7774067ebd46c6c77c98db"
    );
    let d: Value = serde_json::from_slice(&raw).unwrap();
    let mut inputs: Vec<Vec<u32>> = d[0]["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect()
        })
        .collect();
    assert_eq!(inputs.len(), 100);
    let cases = fs::read(".cache/bekko25/cases.json").unwrap();
    assert_eq!(
        digest(&cases),
        "a81b5c4d75c4b5ebe30717d36b152c13ec4f6702cb649a619ce1966fbeafa1ff"
    );
    let cases: Value = serde_json::from_slice(&cases).unwrap();
    let multilingual: Vec<Vec<u32>> = cases
        .as_array()
        .unwrap()
        .iter()
        .filter(|x| [64, 128, 256].contains(&x["ids"].as_array().unwrap().len()))
        .map(|x| {
            x["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect()
        })
        .collect();
    assert_eq!(multilingual.len(), 18);
    inputs.extend(multilingual);
    let mut seed = 0x29102026u32;
    for n in 2..=256 {
        let mut v = vec![2];
        for _ in 0..n - 2 {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            v.push(3 + seed % 255997)
        }
        v.push(1);
        inputs.push(v);
    }
    for tok in [1, 2, 3, 127, 65535, 255999] {
        for n in [
            2, 3, 7, 8, 31, 32, 63, 64, 65, 127, 128, 129, 191, 192, 193, 255, 256,
        ] {
            let mut v = vec![tok; n];
            v[0] = 2;
            v[n - 1] = 1;
            inputs.push(v);
        }
    }
    let p = PocketIcBuilder::new()
        .with_application_subnet()
        .with_server_binary(PathBuf::from(std::env::var("POCKET_IC_BIN").unwrap()))
        .with_max_request_time_ms(Some(300_000))
        .build();
    let before = install(&p, base.clone(), &a);
    let after = install(&p, new.clone(), &b);
    assert_eq!(
        info(&p, before).embedding_space,
        bekko25_core::EMBEDDING_SPACE
    );
    assert_eq!(
        info(&p, after).embedding_space,
        bekko25_core::COMPACT_EMBEDDING_SPACE
    );
    let m0 = p.canister_status(before, None).unwrap().memory_metrics;
    let installed = p.canister_status(after, None).unwrap();
    assert_eq!(
        installed.module_hash.as_deref(),
        Some(Sha256::digest(&new).as_slice()),
        "installed module differs from verified recipe artifact"
    );
    let m1 = installed.memory_metrics;
    require_saving(
        u64::try_from(&m0.stable_memory_size.0).unwrap(),
        u64::try_from(&m1.stable_memory_size.0).unwrap(),
        290_000_001,
    )
    .expect("stable memory");
    require_saving(
        u64::try_from(&m0.wasm_memory_size.0).unwrap(),
        u64::try_from(&m1.wasm_memory_size.0).unwrap(),
        10_000_001,
    )
    .expect("Wasm heap");
    let stranger = Principal::self_authenticating(b"int8-test-stranger");
    for method in ["activate", "beginUpload"] {
        let denied = p
            .update_call(after, stranger, method, Encode!().unwrap())
            .unwrap();
        assert_eq!(
            Decode!(&denied,Result<(),String>).unwrap(),
            Err("controller only".into())
        );
    }
    let denied = p
        .update_call(
            after,
            stranger,
            "upload",
            Encode!(&0u64, &serde_bytes::ByteBuf::from(vec![1])).unwrap(),
        )
        .unwrap();
    assert_eq!(
        Decode!(&denied,Result<u64,String>).unwrap(),
        Err("controller only".into())
    );
    // An accidental cross-format upgrade must not erase the original pack.
    p.upgrade_canister(before, new.clone(), Encode!().unwrap(), None)
        .unwrap();
    let wrong = info(&p, before);
    assert!(!wrong.active && !wrong.weights_verified);
    assert_eq!(
        wrong.last_error.as_deref(),
        Some("unsupported weight storage version")
    );
    p.upgrade_canister(before, base.clone(), Encode!().unwrap(), None)
        .unwrap();
    assert!(info(&p, before).weights_verified);
    activate(&p, before);
    p.upgrade_canister(after, new.clone(), Encode!().unwrap(), None)
        .unwrap();
    let post = info(&p, after);
    assert!(!post.active && post.weights_verified && post.uploaded_bytes == b.len() as u64);
    assert_eq!(
        p.canister_status(after, None)
            .unwrap()
            .module_hash
            .as_deref(),
        Some(Sha256::digest(&new).as_slice()),
        "upgraded module differs from verified recipe artifact"
    );
    activate(&p, after);
    let mut rows = Vec::new();
    let mut baseline_vectors = Vec::new();
    let mut compact_vectors = Vec::new();
    for (i, ids) in inputs.iter().enumerate() {
        if i % 20 == 0 {
            eprintln!("query {i}/{}", inputs.len())
        }
        let old = run(&p, before, ids);
        let new = run(&p, after, ids);
        assert_eq!(old.tokens, new.tokens);
        assert_eq!(old.embedding_space, bekko25_core::EMBEDDING_SPACE);
        assert_eq!(new.embedding_space, bekko25_core::COMPACT_EMBEDDING_SPACE);
        assert_eq!(old.embedding.len(), 384);
        assert_eq!(new.embedding.len(), 384);
        assert!(new.embedding.iter().all(|x| x.is_finite()));
        let similarity = cosine(&old.embedding, &new.embedding).expect("embedding comparison");
        assert!(
            similarity >= MIN_COSINE,
            "embedding similarity {similarity} below {MIN_COSINE} at case {i}"
        );
        assert!(
            new.instructions < 4_900_000_000,
            "instruction gate at case {i}"
        );
        baseline_vectors.push(old.embedding.clone());
        compact_vectors.push(new.embedding.clone());
        rows.push(json!({"tokens":ids.len(),"before_instructions":old.instructions,"after_instructions":new.instructions,"before":old.embedding,"after":new.embedding}));
    }
    check_retrieval(&baseline_vectors, &compact_vectors, &d[0]).expect("retrieval quality");
    assert_eq!(
        build,
        provenance(&target).expect("final build provenance"),
        "build changed during verification"
    );
    let out = json!({"build_provenance":build,"baseline_wasm_sha256":digest(&base),"int8_wasm_sha256":digest(&new),"original_pack_sha256":digest(&a),"int8_pack_sha256":digest(&b),"baseline_heap":m0.wasm_memory_size.to_string(),"baseline_stable":m0.stable_memory_size.to_string(),"int8_heap":m1.wasm_memory_size.to_string(),"int8_stable":m1.stable_memory_size.to_string(),"comparisons":rows});
    let temporary = format!("{RESULT_PATH}.{}.tmp", std::process::id());
    fs::write(&temporary, serde_json::to_vec(&out).unwrap()).unwrap();
    fs::rename(&temporary, RESULT_PATH).unwrap();
    eprintln!("completed {} comparisons", inputs.len());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn savings_reject_memory_growth_and_insufficient_reduction() {
        assert!(require_saving(1, 2, 10_000_000).is_err());
        assert!(require_saving(100, 95, 10).is_err());
        assert!(require_saving(100, 90, 10).is_ok());
    }

    #[test]
    fn result_is_archived_before_failure_and_concurrent_runs_are_locked() {
        let dir = std::env::temp_dir().join(format!(
            "bekko-compact-verify-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        let result = dir.join("compact-verification.json");
        let lock = dir.join("compact-verification.lock");
        fs::write(&result, b"previous evidence").unwrap();
        let guard = RunGuard::start(&result, &lock).unwrap();
        assert!(!result.exists());
        assert!(RunGuard::start(&result, &lock).is_err());
        let backups: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().is_some_and(|x| x == "json"))
            .collect();
        assert_eq!(backups.len(), 1);
        assert_eq!(fs::read(&backups[0]).unwrap(), b"previous evidence");
        drop(guard);
        assert!(!lock.exists());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn retrieval_gate_rejects_collapsed_embeddings() {
        let mut before = Vec::new();
        for i in 0..50 {
            let mut v = vec![0.0; 384];
            v[i] = 1.0;
            before.push(v);
        }
        before.extend(before.clone());
        let collapsed = vec![before[0].clone(); 100];
        let ids: Vec<_> = (0..50).map(|i| format!("d{i}")).collect();
        let qids: Vec<_> = (0..50).map(|i| format!("q{i}")).collect();
        let qrels: serde_json::Map<String, Value> = (0..50)
            .map(|i| (qids[i].clone(), json!([ids[i]])))
            .collect();
        let fixture = json!({"query_ids":qids,"document_ids":ids,"qrels":qrels});
        assert!(check_retrieval(&before, &before, &fixture).is_ok());
        assert!(check_retrieval(&before, &collapsed, &fixture).is_err());
        assert!(cosine(&before[0], &before[1]).unwrap() < MIN_COSINE);
    }
}
