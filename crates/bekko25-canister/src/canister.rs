// Fixed-model Bekko a25m canister. Durable weights; bounded stateless queries.
use bekko25_core::ExternalModel;
use candid::{CandidType, Deserialize};
use ic_cdk::stable::{stable_grow, stable_read, stable_size, stable_write};
use sha2::{Digest, Sha256};
use std::cell::RefCell;
const HEADER: u64 = 64;
const PAGE: u64 = 65536;
const MAGIC: &[u8; 8] = if COMPACT_VOCABULARY { b"BKI25001" } else { b"BEK25001" };
const PACK_BYTES: usize = if COMPACT_VOCABULARY { bekko25_core::COMPACT_PACK_BYTES } else { bekko25_core::PACK_BYTES };
const PACK_SHA256: &str = if COMPACT_VOCABULARY { bekko25_core::COMPACT_PACK_SHA256 } else { bekko25_core::PACK_SHA256 };
const EMBEDDING_SPACE: &str = if COMPACT_VOCABULARY { bekko25_core::COMPACT_EMBEDDING_SPACE } else { bekko25_core::EMBEDDING_SPACE };
const RUNTIME_ID: &str = if COMPACT_VOCABULARY { bekko25_core::COMPACT_RUNTIME_ID } else { bekko25_core::RUNTIME_ID };
thread_local! {
    static MODEL: RefCell<Option<ExternalModel>> = const { RefCell::new(None) };
    static HASH: RefCell<Option<Sha256>> = const { RefCell::new(None) };
    static LAST_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
}
fn controller() -> Result<(), String> {
    if ic_cdk::api::is_controller(&ic_cdk::api::msg_caller()) {
        Ok(())
    } else {
        Err("controller only".into())
    }
}
fn header() -> Result<(u64, bool), String> {
    if stable_size() == 0 {
        return Err("weight storage is not initialized".into());
    }
    let mut b = [0u8; 17];
    stable_read(0, &mut b);
    if &b[..8] != MAGIC {
        return Err("unsupported weight storage version".into());
    }
    let n = u64::from_le_bytes(b[8..16].try_into().unwrap());
    if n > PACK_BYTES as u64
        || b[16] > 1
        || b[16] == 1 && n != PACK_BYTES as u64
        || HEADER + n > stable_size() * PAGE
    {
        return Err("invalid weight metadata".into());
    }
    Ok((n, b[16] == 1))
}
fn write_header(n: u64, verified: bool) {
    let mut b = [0u8; 17];
    b[..8].copy_from_slice(MAGIC);
    b[8..16].copy_from_slice(&n.to_le_bytes());
    b[16] = verified as u8;
    stable_write(0, &b);
}
fn reset_upload() -> Result<(), String> {
    if stable_size() == 0 {
        stable_grow(1).map_err(|e| format!("stable grow: {e:?}"))?;
    } else {
        // Never reinterpret another model's storage as a25m. Deploy a new ID.
        header()?;
    }
    write_header(0, false);
    HASH.with_borrow_mut(|h| *h = Some(Sha256::new()));
    LAST_ERROR.with_borrow_mut(|e| *e = None);
    MODEL.with_borrow_mut(|m| *m = None);
    Ok(())
}
#[ic_cdk::init]
fn init() {
    reset_upload().expect("initialize model storage");
}
#[ic_cdk::update(name = "beginUpload")]
fn begin_upload() -> Result<(), String> {
    controller()?;
    reset_upload()
}
#[ic_cdk::update]
fn upload(offset: u64, bytes: serde_bytes::ByteBuf) -> Result<u64, String> {
    controller()?;
    if bytes.is_empty() || bytes.len() > 1_800_000 {
        return Err("upload chunk must be 1..=1800000 bytes".into());
    }
    let (n, verified) = header()?;
    let end = offset
        .checked_add(bytes.len() as u64)
        .ok_or("upload length overflow")?;
    if end > PACK_BYTES as u64 {
        return Err("weight pack too large".into());
    }
    if offset < n && end <= n {
        // Recover a lost update reply without hashing the same bytes twice.
        let mut stored = vec![0; bytes.len()];
        stable_read(HEADER + offset, &mut stored);
        return if stored.as_slice() == bytes.as_ref() {
            Ok(n)
        } else {
            Err("retry bytes differ from stored upload".into())
        };
    }
    if verified {
        return Err("weights already verified; beginUpload before replacing them".into());
    }
    if offset != n {
        return Err(format!("expected offset {n}"));
    }
    let mut hash = HASH
        .with_borrow(|h| h.clone())
        .ok_or("upload interrupted; beginUpload required")?;
    hash.update(&bytes);
    let done = end == PACK_BYTES as u64;
    if done && format!("{:x}", hash.clone().finalize()) != PACK_SHA256 {
        return Err("weight SHA256 mismatch; restart with beginUpload".into());
    }
    let pages = (HEADER + end).div_ceil(PAGE);
    if pages > stable_size() {
        stable_grow(pages - stable_size()).map_err(|e| format!("stable grow: {e:?}"))?;
    }
    stable_write(HEADER + n, &bytes);
    write_header(end, done);
    HASH.with_borrow_mut(|h| *h = if done { None } else { Some(hash) });
    Ok(end)
}
fn read_pack(offset: usize, dest: &mut [u8]) -> Result<(), String> {
    let end = offset
        .checked_add(dest.len())
        .ok_or("pack offset overflow")?;
    if end > PACK_BYTES {
        return Err("pack read out of bounds".into());
    }
    let offset = u64::try_from(offset).map_err(|_| "pack offset overflow")?;
    let start = HEADER.checked_add(offset).ok_or("pack offset overflow")?;
    stable_read(start, dest);
    Ok(())
}
fn activate_model() -> Result<(), String> {
    let (n, verified) = header()?;
    if !verified || n != PACK_BYTES as u64 {
        return Err("upload the complete verified weight pack first".into());
    }
    let model = ExternalModel::load_from(PACK_BYTES, read_pack)?;
    MODEL.with_borrow_mut(|m| *m = Some(model));
    LAST_ERROR.with_borrow_mut(|e| *e = None);
    Ok(())
}
#[ic_cdk::update]
fn activate() -> Result<(), String> {
    controller()?;
    if MODEL.with_borrow(|m| m.is_some()) && header()?.1 {
        return Ok(());
    }
    let result = activate_model();
    if let Err(ref e) = result {
        LAST_ERROR.with_borrow_mut(|v| *v = Some(e.clone()));
    }
    result
}
#[ic_cdk::post_upgrade]
fn post_upgrade() {
    let result = match header() {
        Ok((_, true)) => Ok(()),
        Ok(_) => {
            let r = reset_upload();
            LAST_ERROR.with_borrow_mut(|e| {
                *e = Some("incomplete upload discarded during upgrade; restart upload".into())
            });
            r
        }
        Err(e) => Err(e),
    };
    if let Err(e) = result {
        LAST_ERROR.with_borrow_mut(|v| *v = Some(e));
    }
}
#[derive(CandidType, Deserialize)]
struct Embedding {
    embedding_space: String,
    embedding: Vec<f32>,
    tokens: u32,
    instructions: u64,
}
#[ic_cdk::query]
fn embed(token_ids: Vec<u32>) -> Result<Embedding, String> {
    let before = ic_cdk::api::performance_counter(0);
    bekko25_core::validate_tokens(&token_ids)?;
    let embedding = MODEL.with_borrow(|m| {
        m.as_ref()
            .ok_or("model is not active".to_string())?
            .full(&token_ids, read_pack)
    })?;
    if embedding.len() != 384 || embedding.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite embedding".into());
    }
    Ok(Embedding {
        embedding_space: EMBEDDING_SPACE.into(),
        embedding,
        tokens: token_ids.len() as u32,
        instructions: ic_cdk::api::performance_counter(0) - before,
    })
}
#[derive(CandidType, Deserialize)]
struct ModelInfo {
    embedding_space: String,
    runtime_id: String,
    layers: u32,
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
#[ic_cdk::query(name = "modelInfo")]
fn model_info() -> ModelInfo {
    let state = header();
    ModelInfo {
        embedding_space: EMBEDDING_SPACE.into(),
        runtime_id: RUNTIME_ID.into(),
        layers: 13,
        model_id: bekko25_core::MODEL_ID.into(),
        revision: bekko25_core::REVISION.into(),
        dimensions: 384,
        max_tokens: 256,
        weight_pack_sha256: PACK_SHA256.into(),
        uploaded_bytes: state.as_ref().map(|v| v.0).unwrap_or(0),
        weights_verified: state.as_ref().map(|v| v.1).unwrap_or(false),
        active: MODEL.with_borrow(|m| m.is_some()),
        last_error: LAST_ERROR
            .with_borrow(|e| e.clone())
            .or_else(|| state.err()),
    }
}
ic_cdk::export_candid!();
