"""Isolated 4-layer Bekko experiment. Uses F32 attention, not the D=32 integer kernel."""

import shutil
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / ".cache/multilingual/bekko-port"
DEST.mkdir(parents=True, exist_ok=True)
workspace = tomllib.loads((ROOT / "Cargo.toml").read_text())
# Preserve the complete workspace so path dependencies and Cargo.lock stay valid.
for member in workspace["workspace"]["members"]:
    shutil.copytree(ROOT / member, DEST / member, dirs_exist_ok=True)
for f in ["Cargo.toml", "Cargo.lock", "rust-toolchain.toml"]:
    shutil.copyfile(ROOT / f, DEST / f)
shutil.copytree(ROOT / ".cargo", DEST / ".cargo", dirs_exist_ok=True)
p = DEST / "crates/core/src/lib.rs"
s = p.read_text()
s = s.replace("pub const D: usize = 32;", "pub const D: usize = 64;").replace(
    "1536", "1152"
)
s = (
    s.replace("for l in 0..12 {", "for l in 0..4 {")
    .replace("for layer in 0..12 {", "for layer in 0..4 {")
    .replace("for head in 0..12 {", "for head in 0..6 {")
    .replace("for head in (0..12).step_by(3)", "for head in (0..6).step_by(3)")
)
s = s.replace("*id >= 50368", "*id >= 256000").replace(
    "[80000f32, 10000f32]", "[160000f32, 160000f32]"
)
s = s.replace("&[50368, H]", "&[256000, H]")
s = (
    s.replace("[(f32, f32); 16]", "[(f32, f32); 32]")
    .replace("frequencies: [f32; 16]", "frequencies: [f32; 32]")
    .replace("for i in 0..16 {", "for i in 0..32 {")
    .replace("let b = a + 16;", "let b = a + 32;")
)
# Normalize every token before mean pooling, then L2 normalize the mean.
s = s.replace(
    "Ok(self.final_embedding(&x[..H]))",
    """{
            norm(&mut x, &self.final_norm);
            let mut mean = vec![0f32; H];
            for row in x.chunks_exact(H) { for i in 0..H { mean[i] += row[i] / ids.len() as f32; } }
            let scale = mean.iter().map(|v| v*v).sum::<f32>().sqrt().max(1e-12);
            for v in &mut mean { *v /= scale; }
            Ok(mean)
        }""",
    1,
)
# The old unrolled F32 functions cover exactly 32 dimensions. Generate all 64.
a = s.index("fn score_head(query:")
b = s.index(
    '\n#[cfg(not(target_arch = "wasm32"))]\n#[inline(always)]\nfn weighted_add', a
)
s = (
    s[:a]
    + """fn score_head(query: &[f32], k: &[f32], offset: usize, stride: usize, scores: &mut [f32]) -> f32 {
    use core::arch::wasm32::*;
    let mut max = f32::NEG_INFINITY;
    for (j,score) in scores.iter_mut().enumerate() {
        unsafe {
            let mut sum = f32x4_splat(0.);
            for d in (0..D).step_by(4) {
                sum = f32x4_add(sum,f32x4_mul(v128_load(query.as_ptr().add(d).cast()),v128_load(k.as_ptr().add(offset+j*stride+d).cast())));
            }
            *score = (f32x4_extract_lane::<0>(sum)+f32x4_extract_lane::<1>(sum)+f32x4_extract_lane::<2>(sum)+f32x4_extract_lane::<3>(sum)) / (D as f32).sqrt();
        }
        max=max.max(*score);
    }
    max
}
#[cfg(target_arch = "wasm32")]
fn value_head(dest:&mut [f32],v:&[f32],offset:usize,stride:usize,scores:&[f32],inverse:f32) {
    use core::arch::wasm32::*;
    for d in (0..D).step_by(4) {
        unsafe {
            let mut sum=f32x4_splat(0.);
            for (j,p) in scores.iter().enumerate() {
                sum=f32x4_add(sum,f32x4_mul(f32x4_splat(*p*inverse),v128_load(v.as_ptr().add(offset+j*stride+d).cast())));
            }
            v128_store(dest.as_mut_ptr().add(d).cast(),sum);
        }
    }
}
"""
    + s[b:]
)
p.write_text(s)
p = DEST / "crates/core/build.rs"
p.write_text(p.read_text().replace("1536", "1152").replace("768", "576"))
p = DEST / "crates/core/src/strassen.rs"
p.write_text(p.read_text().replace("768", "576"))
p = DEST / "crates/canister/Cargo.toml"
p.write_text(p.read_text().replace('"integer-attention", ', ""))
p = DEST / "crates/canister/src/lib.rs"
p.write_text("""use candid::{CandidType,Deserialize};
use std::cell::RefCell;
thread_local!{static PACK:RefCell<Vec<u8>>=const{RefCell::new(Vec::new())};static MODEL:RefCell<Option<granite_core::Model>>=const{RefCell::new(None)};}
#[ic_cdk::update]
fn upload(offset:u64,bytes:serde_bytes::ByteBuf){assert!(ic_cdk::api::is_controller(&ic_cdk::api::msg_caller()));PACK.with_borrow_mut(|p|{assert_eq!(p.len()as u64,offset);assert!(bytes.len()<=1_900_000);assert!(p.len()+bytes.len()<=450_000_000);p.extend_from_slice(&bytes);});}
#[ic_cdk::update]
fn activate()->Result<(),String>{assert!(ic_cdk::api::is_controller(&ic_cdk::api::msg_caller()));let m=PACK.with_borrow(|p|granite_core::Model::load(p))?;MODEL.with_borrow_mut(|v|*v=Some(m));PACK.with_borrow_mut(|p|*p=Vec::new());Ok(())}
#[derive(CandidType,Deserialize)]
struct Step{next_layer:u32,tokens:u32,state:serde_bytes::ByteBuf,embedding:Vec<f32>,instructions:u64}
#[ic_cdk::query(name="startEmbedding")]
fn embed(ids:Vec<u32>)->Result<Step,String>{let t=ic_cdk::api::performance_counter(0);if ids.len()>512{return Err("experiment supports <=512 tokens".into())}let embedding=MODEL.with_borrow(|m|m.as_ref().ok_or("not active".to_string())?.full(&ids))?;Ok(Step{next_layer:12,tokens:ids.len()as u32,state:serde_bytes::ByteBuf::new(),embedding,instructions:ic_cdk::api::performance_counter(0)-t})}
""")
(DEST / "crates/bench/src").mkdir(parents=True, exist_ok=True)
shutil.copyfile(ROOT / "crates/bench/Cargo.toml", DEST / "crates/bench/Cargo.toml")
s = (
    Path(__file__)
    .with_name("pocket.rs")
    .read_text()
    .replace("multilingual/granite/", "multilingual/bekko8/")
    .replace("multilingual/port/", "multilingual/bekko-port/")
    .replace("pocket-granite.json", "pocket-bekko8.json")
)
(DEST / "crates/bench/src/main.rs").write_text(s)
print(DEST)
