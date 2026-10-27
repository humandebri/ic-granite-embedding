//! Local proof of concept: controller-only heap weight loading; stateless query inference.
use candid::{CandidType, Deserialize};
use granite_core::{EmbeddingProgress, Model, carry};
use std::cell::RefCell;

thread_local! {
    static PACK: RefCell<Vec<u8>> = const {RefCell::new(Vec::new())};
    static MODEL: RefCell<Option<Model>> = const {RefCell::new(None)};
}

#[derive(CandidType, Deserialize)]
struct Measurement {
    #[serde(with = "serde_bytes")]
    data: Vec<u8>,
    instructions: u64,
}

#[ic_cdk::update]
fn upload(offset: u64, #[allow(unused)] bytes: serde_bytes::ByteBuf) {
    assert!(
        ic_cdk::api::is_controller(&ic_cdk::api::msg_caller()),
        "controller only"
    );
    assert!(bytes.len() <= 1_900_000, "chunk too large");
    PACK.with_borrow_mut(|p| {
        assert_eq!(p.len() as u64, offset);
        assert!(p.len() + bytes.len() <= 160_000_000);
        p.extend_from_slice(&bytes);
    });
}

#[ic_cdk::update]
fn activate() -> Result<(), String> {
    assert!(
        ic_cdk::api::is_controller(&ic_cdk::api::msg_caller()),
        "controller only"
    );
    let model = PACK.with_borrow(|p| Model::load(p))?;
    MODEL.with_borrow_mut(|m| *m = Some(model));
    PACK.with_borrow_mut(|p| *p = Vec::new());
    Ok(())
}

#[ic_cdk::query]
fn step(request: serde_bytes::ByteBuf) -> Result<Measurement, String> {
    let before = ic_cdk::api::performance_counter(0);
    let data =
        MODEL.with_borrow(|m| m.as_ref().ok_or("not activated".to_string())?.run(&request))?;
    Ok(Measurement {
        data,
        instructions: ic_cdk::api::performance_counter(0) - before,
    })
}

#[cfg(feature = "profiling")]
#[derive(CandidType, Deserialize)]
struct ProfileMeasurement {
    #[serde(with = "serde_bytes")]
    data: Vec<u8>,
    instructions: u64,
    phases: Vec<(String, u64, u64)>,
}
#[cfg(feature = "profiling")]
#[ic_cdk::query(name = "profileStep")]
fn profile_step(request: serde_bytes::ByteBuf) -> Result<ProfileMeasurement, String> {
    granite_core::profile::begin();
    let measured = step(request);
    let phases = granite_core::profile::finish()
        .into_iter()
        .map(|p| (p.name, p.instructions, p.calls))
        .collect();
    let measured = measured?;
    Ok(ProfileMeasurement {
        data: measured.data,
        instructions: measured.instructions,
        phases,
    })
}

#[derive(CandidType, Deserialize)]
struct EmbeddingStep {
    next_layer: u32,
    tokens: u32,
    state: serde_bytes::ByteBuf,
    embedding: Vec<f32>,
    instructions: u64,
}

fn progress_reply(progress: EmbeddingProgress, before: u64) -> Result<EmbeddingStep, String> {
    let done = progress.next_layer == 12;
    if progress.values.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite result".into());
    }
    Ok(EmbeddingStep {
        next_layer: progress.next_layer as u32,
        tokens: progress.tokens as u32,
        state: serde_bytes::ByteBuf::from(if done {
            Vec::new()
        } else {
            carry::encode(&progress.values)?
        }),
        embedding: if done { progress.values } else { Vec::new() },
        instructions: ic_cdk::api::performance_counter(0) - before,
    })
}

#[ic_cdk::query(name = "startEmbedding")]
fn start_embedding(token_ids: Vec<u32>) -> Result<EmbeddingStep, String> {
    let before = ic_cdk::api::performance_counter(0);
    let progress = MODEL.with_borrow(|m| {
        m.as_ref()
            .ok_or("not activated".to_string())?
            .start_embedding(&token_ids)
    })?;
    progress_reply(progress, before)
}

#[ic_cdk::query(name = "continueEmbedding")]
fn continue_embedding(
    tokens: u32,
    next_layer: u32,
    state: serde_bytes::ByteBuf,
) -> Result<EmbeddingStep, String> {
    let before = ic_cdk::api::performance_counter(0);
    let progress = MODEL.with_borrow(|m| {
        m.as_ref()
            .ok_or("not activated".to_string())?
            .continue_embedding(tokens as usize, next_layer as usize, &state)
    })?;
    progress_reply(progress, before)
}

ic_cdk::export_candid!();
