//! Pinned a25m configuration; kernels are shared without Cargo model features.
pub use bekko_core::{D, H, I, MAX_TOKENS, VOCAB, validate_tokens};
pub type Model = bekko_core::Encoder<13, true>;
pub const MODEL_ID: &str = "hotchpotch/bekko-embedding-v1-a25m";
pub const REVISION: &str = "44f0b8af0f487acd0ccf1a7cb7ae7a29a6dfc09c";
pub const PACK_BYTES: usize = 418404508;
pub const PACK_SHA256: &str = "8fc88d56d68f550d1a920bab7d1ec42af9449240048393bd6daabbfcfa861724";
/// Change when numerical semantics change; rebuild the document index then.
pub const EMBEDDING_SPACE: &str = "bekko-a25m-44f0b8af-w8a8-bf16kv-qk13-v1";
pub const RUNTIME_ID: &str = "bekko25-runtime-1";
