"""Build isolated a25m runtime variants with layer profiling and durable probe weights."""

import argparse
import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / ".cache/multilingual-global/bekko25/port"
DEST = ROOT / ".cache/bekko25-runtime/port"


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--variant",
        default="baseline",
        choices=[
            "baseline",
            "attention",
            "attention-qk",
            "strassen2",
            "qk13",
            "direct",
            "qk13-direct",
            "qk13-full-wo",
        ],
    )
    args = p.parse_args()
    for path in ["Cargo.toml", "Cargo.lock", "rust-toolchain.toml"]:
        DEST.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(BASE / path, DEST / path)
    shutil.copytree(BASE / ".cargo", DEST / ".cargo", dirs_exist_ok=True)
    shutil.copytree(BASE / "crates", DEST / "crates", dirs_exist_ok=True)
    core = DEST / "crates/bekko-core/src/lib.rs"
    s = core.read_text()
    a = s.index("macro_rules! phase {")
    b = s.index("include!(", a)
    s = (
        s[:a]
        + """thread_local! {static PROFILE: std::cell::RefCell<Vec<(String,u64)>> = const {std::cell::RefCell::new(Vec::new())};}
pub fn profile_reset(){PROFILE.with_borrow_mut(|p|p.clear());}
pub fn profile_take()->Vec<(String,u64)>{PROFILE.with_borrow_mut(std::mem::take)}
macro_rules! phase {
    ($name:expr,$body:expr)=>{{
        #[cfg(target_arch="wasm32")] let before=ic_cdk::api::performance_counter(0);
        let result=$body;
        #[cfg(target_arch="wasm32")] $crate::PROFILE.with_borrow_mut(|p|p.push(($name.into(),ic_cdk::api::performance_counter(0)-before)));
        result
    }};
}
"""
        + s[b:]
    )
    a = s.index("        for layer in 0..13 {", s.index("pub fn full("))
    b = s.index("            x = self.finish(layer, &x, &context);", a)
    attention = s[s.index("            let mut context", a) : b]
    helper = (
        """    pub fn profile_layer(&self,layer:usize,x:&[f32])->Vec<f32>{
        assert!(layer<13 && x.len()%H==0 && (2..=256).contains(&(x.len()/H)));
        let tokens=x.len()/H;
        let qkv=phase!("qkv_total",self.qkv(layer,0,x));
        let context=phase!("attention_total",{
"""
        + attention.replace("ids.len()", "tokens").replace(
            "&wire)?;", "&wire).unwrap();"
        )
        + """
            context
        });
        phase!("finish_total",self.finish(layer,x,&context))
    }
    pub fn finalize(&self,mut x:Vec<f32>)->Vec<f32>{
        let tokens=x.len()/H;norm(&mut x,&self.final_norm);
        let mut mean=vec![0f32;H];for row in x.chunks_exact(H){for i in 0..H{mean[i]+=row[i]/tokens as f32;}}
        let scale=mean.iter().map(|v|v*v).sum::<f32>().sqrt().max(1e-12);for v in &mut mean{*v/=scale;}mean
    }
"""
    )
    i = s.index("    pub fn full(")
    s = s[:i] + helper + s[i:]
    if args.variant in [
        "attention",
        "attention-qk",
        "strassen2",
        "qk13",
        "direct",
        "qk13-direct",
        "qk13-full-wo",
    ]:
        # Every accumulator keeps the original key order. Share probability and
        # address work across all 16 SIMD vectors rather than repeating it per d.
        a = s.index("fn value_head(")
        b = s.index('\n#[cfg(not(target_arch = "wasm32"))]', a)
        body = """fn value_head(dest:&mut [f32],v:&[f32],offset:usize,stride:usize,scores:&[f32],inverse:f32){
            use core::arch::wasm32::*;
            unsafe{
"""
        body += "".join(f"let mut s{d}=f32x4_splat(0.);" for d in range(16))
        body += "for (j,p) in scores.iter().enumerate(){let prob=f32x4_splat(*p*inverse);let src=v.as_ptr().add(offset+j*stride);"
        body += "".join(
            f"s{d}=f32x4_add(s{d},f32x4_mul(prob,v128_load(src.add({d * 4}).cast())));"
            for d in range(16)
        )
        body += "}"
        body += "".join(
            f"v128_store(dest.as_mut_ptr().add({d * 4}).cast(),s{d});"
            for d in range(16)
        )
        body += "}}\n"
        s = s[:a] + body + s[b:]
    if args.variant in ["attention-qk", "strassen2", "direct"]:
        a = s.index("fn score_head(")
        b = s.index('\n#[cfg(target_arch = "wasm32")]\nfn value_head', a)
        body = """fn score_head(query:&[f32],k:&[f32],offset:usize,stride:usize,scores:&mut [f32])->f32{
            use core::arch::wasm32::*;let mut max=f32::NEG_INFINITY;unsafe{
"""
        body += "".join(
            f"let q{d}=v128_load(query.as_ptr().add({d * 4}).cast());"
            for d in range(16)
        )
        body += "for (j,score) in scores.iter_mut().enumerate(){let src=k.as_ptr().add(offset+j*stride);let mut sum=f32x4_splat(0.);"
        body += "".join(
            f"sum=f32x4_add(sum,f32x4_mul(q{d},v128_load(src.add({d * 4}).cast())));"
            for d in range(16)
        )
        body += "*score=(f32x4_extract_lane::<0>(sum)+f32x4_extract_lane::<1>(sum)+f32x4_extract_lane::<2>(sum)+f32x4_extract_lane::<3>(sum))/8.;max=max.max(*score);}}max}\n"
        s = s[:a] + body + s[b:]
    if args.variant in ["qk13", "qk13-direct", "qk13-full-wo"]:
        i = s.index("pub fn attention(")
        s = s[:i] + Path(__file__).with_name("qk13_fragment.rs").read_text() + s[i:]
        s = s.replace(
            "    let mut out = vec![0.; rows * width];",
            """    #[cfg(target_arch="wasm32")]
    let (quant_keys,key_scales)={
        let mut quant=vec![0i16;k.len()];let mut scales=vec![0f32;kn*heads];
        for j in 0..kn {for h in 0..heads {let at=j*width+h*D;scales[j*heads+h]=quant13(&k[at..at+D],&mut quant[at..at+D]);}}
        (quant,scales)
    };
    let mut out = vec![0.; rows * width];""",
        )
        s = s.replace(
            "let max = score_head(query, &k, first * width + h * D, width, active);",
            "let max = score13(query, &quant_keys, &key_scales, first * width + h * D, width,first*heads+h,heads,active);",
        )
    if args.variant in ["direct", "qk13-direct", "qk13-full-wo"]:
        a = s.index("    if k.iter().chain(&v).any(")
        s = (
            s[:a]
            + """    attention_values(layer,start,rows,heads,ks,kn,&q,&k,&v)
}
fn attention_values(layer:usize,start:usize,rows:usize,heads:usize,ks:usize,kn:usize,q:&[f32],k:&[f32],v:&[f32])->Result<Vec<f32>,String>{
    let width=heads*D;
"""
            + s[a:]
        )
        i = s.index("pub fn attention(")
        s = (
            s[:i]
            + Path(__file__).with_name("attention_direct_fragment.rs").read_text()
            + s[i:]
        )
        pattern = r"let wire = attention_wire\(&qkv, start, rows, head, 3, kstart, kend\);\s*let result =\s*attention\(layer, start, rows, head, 3, kstart, kend - kstart, &wire\)(\?|\.unwrap\(\));"
        s, n = re.subn(
            pattern,
            lambda m: (
                "let result=attention_direct(layer,start,rows,head,3,kstart,kend,&qkv)"
                + m.group(1)
                + ";"
            ),
            s,
        )
        assert n == 2, n
        s = s.replace(
            "if k.iter().chain(&v).any(|v| !v.is_finite())",
            "if q.iter().chain(k).chain(v).any(|v| !v.is_finite())",
        )
    if args.variant == "qk13-full-wo":
        # Internal full inference has no state-frame size restriction. Compute
        # each key's BF16 conversion and QK quantization once per layer.
        for length, suffix in [("tokens", ".unwrap()"), ("ids.len()", "?")]:
            a = s.index(f"            let mut context = vec![0.; {length} * H];")
            b = s.index("\n            }", s.index("\n                }", a)) + len(
                "\n            }"
            )
            s = (
                s[:a]
                + f"            let context=attention_direct(layer,0,{length},0,6,0,{length},&qkv){suffix};"
                + s[b:]
            )
    core.write_text(s)
    if args.variant in ["strassen2", "qk13-full-wo"]:
        path = DEST / "crates/core/src/strassen.rs"
        v = path.read_text()
        v = v.replace(
            "    products: [Vec<i16>; 7],",
            "    products: [Vec<i16>; 7],\n    children: Option<Box<[Prepared;7]>>,",
        )
        a = v.index("    pub fn new(")
        b = v.index("    fn products(", a)
        v = (
            v[:a]
            + """    pub fn new(w:&[i16],rows:usize,cols:usize)->Self {Self::new_depth(w,rows,cols,2)}
    fn new_depth(w:&[i16],rows:usize,cols:usize,depth:usize)->Self {
        let products:[Vec<i16>;7]=std::array::from_fn(|i|part(w,rows,cols,WEIGHTS[i]));
        let children=if depth>1 {Some(Box::new(std::array::from_fn(|i|Self::new_depth(&products[i],rows/2,cols/2,depth-1))))}else{None};
        Self{rows,cols,products,children}
    }
"""
            + v[b:]
        )
        v = v.replace(
            """            phase!(
                "packed_leaf",
                leaf(""",
            """            if n>=128 && n%128==0 {if let Some(children)=&self.children{return phase!("recursive_product",children[i].multiply_integer(&input,n/2));}}
            phase!(
                "packed_leaf",
                leaf(""",
        )
        a = v.index("    pub fn apply(")
        v = (
            v[:a]
            + Path(__file__).with_name("strassen2_fragment.rs").read_text()
            + v[a:]
        )
        v = v.replace(
            """            let outputs = if cols == 192 && rows % 128 == 0 {""",
            """            if cols==96 || cols==288 {
                let outputs=if cols==96 && rows%64==0 {64}else{32};
                for t in (0..n).step_by(32){for r in (0..rows).step_by(outputs){
                    let xp=q.as_ptr().add(t*cols);let wp=w.as_ptr().add(r*cols);let yp=out.as_mut_ptr().add(t*rows+r);
                    match (cols,outputs){(96,64)=>super::packed_96_64(xp,wp,yp,rows),(96,32)=>super::packed_96_32(xp,wp,yp,rows),(288,32)=>super::packed_288_32(xp,wp,yp,rows),_=>unreachable!()}
                }}out.set_len(n*rows);return out;
            }
            let outputs = if cols == 192 && rows % 128 == 0 {""",
        )
        if args.variant == "qk13-full-wo":
            v = v.replace(
                "Self::new_depth(w,rows,cols,2)",
                "Self::new_depth(w,rows,cols,if cols==super::I {2}else{1})",
            )
        path.write_text(v + Path(__file__).with_name("strassen2_tests.rs").read_text())
        gen = DEST / "crates/kernel-gen/src/lib.rs"
        v = gen.read_text()
        a = v.index("    fs::write(", v.index("pub fn generate("))
        v = (
            v[:a]
            + """    for (cols,outputs) in [(96,64),(96,32),(288,32)] {
        s.push_str(&kernel(cols,32,outputs,true,intermediate).replace(&format!("packed_{cols}("),&format!("packed_{cols}_{outputs}(")));
    }
"""
            + v[a:]
        )
        gen.write_text(v)
    cargo = DEST / "crates/bekko-core/Cargo.toml"
    cargo.write_text(
        cargo.read_text().replace('libm = "0.2"', 'libm = "0.2"\nic-cdk = "=0.20.3"')
    )
    (DEST / "crates/probe/src/lib.rs").write_text("""use std::cell::RefCell;
use ic_cdk::stable::{stable_size,stable_grow,stable_read,stable_write};
thread_local!{static MODEL:RefCell<Option<bekko_core::Model>>=const{RefCell::new(None)};}
fn owner(){assert!(ic_cdk::api::is_controller(&ic_cdk::api::msg_caller()));}
#[ic_cdk::update]
fn upload(offset:u64,bytes:serde_bytes::ByteBuf){owner();assert!(bytes.len()<=1_800_000);assert!(offset+bytes.len()as u64<=430_000_000);let end=8+offset+bytes.len()as u64;let pages=end.div_ceil(65536);if pages>stable_size(){stable_grow(pages-stable_size()).unwrap();}stable_write(8+offset,&bytes);stable_write(0,&(offset+bytes.len()as u64).to_le_bytes());}
#[ic_cdk::update]
fn activate(){owner();let mut b=[0u8;8];stable_read(0,&mut b);let n=u64::from_le_bytes(b)as usize;assert!(n==bekko_core::PACK_BYTES);let mut p=vec![0;n];stable_read(8,&mut p);MODEL.with_borrow_mut(|m|*m=Some(bekko_core::Model::load(&p).unwrap()));}
#[ic_cdk::query]
fn embed(ids:Vec<u32>)->(Vec<f32>,u64){let t=ic_cdk::api::performance_counter(0);let v=MODEL.with_borrow(|m|m.as_ref().unwrap().full(&ids).unwrap());(v,ic_cdk::api::performance_counter(0)-t)}
#[ic_cdk::query]
fn start(ids:Vec<u32>)->Vec<f32>{MODEL.with_borrow(|m|m.as_ref().unwrap().embed(&ids).unwrap())}
#[ic_cdk::query(name="profileLayer")]
fn profile_layer(layer:u32,x:Vec<f32>)->(Vec<f32>,Vec<(String,u64)>,u64){bekko_core::profile_reset();let t=ic_cdk::api::performance_counter(0);let v=MODEL.with_borrow(|m|m.as_ref().unwrap().profile_layer(layer as usize,&x));let cost=ic_cdk::api::performance_counter(0)-t;(v,bekko_core::profile_take(),cost)}
#[ic_cdk::query]
fn finalize(x:Vec<f32>)->Vec<f32>{MODEL.with_borrow(|m|m.as_ref().unwrap().finalize(x))}
""")
    for file in DEST.rglob("*"):
        if file.is_file() and file.suffix in [".rs", ".toml"]:
            os.utime(file, None)
    print(DEST)


if __name__ == "__main__":
    main()
