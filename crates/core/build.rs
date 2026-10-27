use std::{env, fs, path::PathBuf};

// Exact integer products; balanced sums avoid accumulator local.get/local.set
// instructions in the Wasm hot path. No SIMD reassociation of floats.
fn tree(parts: &[String]) -> String {
    if parts.len() == 1 {
        return parts[0].clone();
    }
    let m = parts.len() / 2;
    format!("i32x4_add({}, {})", tree(&parts[..m]), tree(&parts[m..]))
}
// Store four completed output channels immediately. The caller's x/w/y
// allocations are disjoint, so writes need not wait for unrelated weight loads.
fn streamed_body(cols: usize, tokens: usize, outputs: usize, integer: bool) -> String {
    let mut s = String::new();
    s.push_str("#[inline(always)]\nfn sum4(a:v128,b:v128,c:v128,d:v128)->v128 { let ab=i32x4_add(i32x4_shuffle::<0,1,4,5>(a,b),i32x4_shuffle::<2,3,6,7>(a,b));let cd=i32x4_add(i32x4_shuffle::<0,1,4,5>(c,d),i32x4_shuffle::<2,3,6,7>(c,d));i32x4_add(i32x4_shuffle::<0,2,4,6>(ab,cd),i32x4_shuffle::<1,3,5,7>(ab,cd)) }\n");
    for a in 0..tokens {
        for k in 0..cols / 8 {
            s.push_str(&format!(
                "let x{a}_{k}=v128_load(x.add({}).cast());\n",
                a * cols + k * 8
            ));
        }
        if !integer {
            s.push_str(&format!(
                "let sx{a}=f32x4_splat(*sx.add({a}));let y{a}=y.add({a}*stride);\n"
            ));
        }
    }
    // A 16-channel generated body limits code size while retaining shared loads.
    let group_width = 16.min(outputs);
    assert!(group_width >= 4 && group_width % 4 == 0 && outputs % group_width == 0);
    let compact = group_width < outputs;
    if compact {
        s.push_str(&format!(
            "for group in (0..{outputs}).step_by({group_width}) {{let w=w.add(group*{cols});\n"
        ));
        if !integer {
            s.push_str("let sw=sw.add(group);\n");
        }
        for a in 0..tokens {
            if integer {
                s.push_str(&format!("let y{a}=y.add({a}*stride+group);\n"));
            } else {
                s.push_str(&format!("let y{a}=y{a}.add(group);\n"));
            }
        }
    }
    for group in (0..if compact { group_width } else { outputs }).step_by(4) {
        s.push_str("{\n");
        for b in group..group + 4 {
            for k in 0..cols / 8 {
                s.push_str(&format!(
                    "let w{b}_{k}=v128_load(w.add({}).cast());\n",
                    b * cols + k * 8
                ));
            }
        }
        if !integer {
            s.push_str(&format!("let scale=v128_load(sw.add({group}).cast());\n"));
        }
        for a in 0..tokens {
            for b in group..group + 4 {
                let terms: Vec<_> = (0..cols / 8)
                    .map(|k| format!("i32x4_dot_i16x8(x{a}_{k},w{b}_{k})"))
                    .collect();
                s.push_str(&format!("let a{a}_{b}={};\n", tree(&terms)));
            }
            let sum = format!(
                "sum4(a{a}_{group},a{a}_{},a{a}_{},a{a}_{})",
                group + 1,
                group + 2,
                group + 3
            );
            if integer {
                s.push_str(&format!(
                    "v128_store({}.cast(),{sum});\n",
                    if compact {
                        format!("y{a}.add({group})")
                    } else {
                        format!("y.add({a}*stride+{group})")
                    }
                ));
            } else {
                s.push_str(&format!("v128_store(y{a}.add({group}).cast(),f32x4_mul(f32x4_convert_i32x4({sum}),f32x4_mul(sx{a},scale)));\n"));
            }
        }
        s.push_str("}\n");
    }
    if compact {
        s.push_str("}\n");
    }
    s.push_str("}\n");
    s
}
fn kernel(cols: usize, tokens: usize, outputs: usize, integer: bool) -> String {
    let mut s = String::new();
    let declaration = if integer {
        format!("unsafe fn packed_{cols}(x:*const i16,w:*const i16,y:*mut i32,stride:usize)")
    } else {
        format!(
            "unsafe fn tree_{cols}_{tokens}(x:*const i16,w:*const i16,y:*mut f32,stride:usize,sx:*const f32,sw:*const f32)"
        )
    };
    s.push_str(&format!("#[cfg(target_arch=\"wasm32\")]\n#[inline(never)]\n#[allow(unsafe_op_in_unsafe_fn)]\n{declaration} {{\nuse core::arch::wasm32::*;\n"));
    let chunking = env::var_os("CARGO_FEATURE_CHUNKED_GEMM").is_some();
    let width = if integer && cols == 768 {
        192
    } else if chunking && cols == 1536 && tokens == 32 {
        384
    } else {
        cols
    };
    let chunked = width < cols;
    if !chunked && outputs >= 32 {
        s.push_str(&streamed_body(cols, tokens, outputs, integer));
        return s;
    }
    if chunked {
        for a in 0..tokens {
            for b in 0..outputs {
                s.push_str(&format!("let mut a{a}_{b}=i32x4_splat(0);\n"));
            }
        }
        s.push_str(&format!(
            "for block in 0..{} {{let x=x.add(block*{width});let w=w.add(block*{width});\n",
            cols / width
        ));
    }
    for prefix in ["x", "w"] {
        for a in 0..if prefix == "x" { tokens } else { outputs } {
            for k in 0..width / 8 {
                s.push_str(&format!(
                    "let {prefix}{a}_{k}=v128_load({prefix}.add({}).cast());\n",
                    a * cols + k * 8
                ));
            }
        }
    }
    for a in 0..tokens {
        for b in 0..outputs {
            let terms: Vec<_> = (0..width / 8)
                .map(|k| format!("i32x4_dot_i16x8(x{a}_{k},w{b}_{k})"))
                .collect();
            if chunked {
                s.push_str(&format!("a{a}_{b}=i32x4_add(a{a}_{b},{});\n", tree(&terms)));
            } else {
                s.push_str(&format!("let a{a}_{b}={};\n", tree(&terms)));
            }
        }
    }
    if chunked {
        s.push_str("}\n");
    }
    s.push_str("#[inline(always)]\nfn sum4(a:v128,b:v128,c:v128,d:v128)->v128 { let ab=i32x4_add(i32x4_shuffle::<0,1,4,5>(a,b),i32x4_shuffle::<2,3,6,7>(a,b));let cd=i32x4_add(i32x4_shuffle::<0,1,4,5>(c,d),i32x4_shuffle::<2,3,6,7>(c,d));i32x4_add(i32x4_shuffle::<0,2,4,6>(ab,cd),i32x4_shuffle::<1,3,5,7>(ab,cd)) }\n");
    if integer {
        for a in 0..tokens {
            for b in (0..outputs).step_by(4) {
                s.push_str(&format!("v128_store(y.add({a}*stride+{b}).cast(),sum4(a{a}_{b},a{a}_{},a{a}_{},a{a}_{}));\n",b+1,b+2,b+3));
            }
        }
    } else {
        for b in (0..outputs).step_by(4) {
            s.push_str(&format!("let sw{}=v128_load(sw.add({b}).cast());\n", b / 4));
        }
        for a in 0..tokens {
            s.push_str(&format!(
                "let sx{a}=f32x4_splat(*sx.add({a}));let y{a}=y.add({a}*stride);\n"
            ));
            for b in (0..outputs).step_by(4) {
                s.push_str(&format!("v128_store(y{a}.add({b}).cast(),f32x4_mul(f32x4_convert_i32x4(sum4(a{a}_{b},a{a}_{},a{a}_{},a{a}_{})),f32x4_mul(sx{a},sw{})));\n",b+1,b+2,b+3,b/4));
            }
        }
    }
    s.push_str("}\n");
    s
}
// Generate named accumulators so every probability load is shared across D=32.
// Keeping the loop body explicit avoids dynamic indexing into SIMD arrays.
fn weighted_kernel() -> String {
    let mut s = String::from(
        "#[allow(clippy::too_many_arguments)]\nfn weighted_head(p:*const i16,w:*const i16,stride:usize,cols:usize,scales:*const f32,inverse:f32,out:*mut f32) {if cols>512 {weighted_head_wide(p,w,stride,cols,scales,inverse,out);return;}unsafe {\n",
    );
    for i in 0..32 {
        s.push_str(&format!(
            "let mut a{i}=i32x4_splat(0);let w{i}=w.add({i}*stride);\n"
        ));
    }
    s.push_str(
        "let mut i=0;macro_rules! chunk {($off:expr)=>{{let q=v128_load(p.add(i+$off).cast());\n",
    );
    for i in 0..32 {
        s.push_str(&format!(
            "a{i}=i32x4_add(a{i},i32x4_dot_i16x8(q,v128_load(w{i}.add(i+$off).cast())));\n"
        ));
    }
    s.push_str("}};}while i+32<=cols {chunk!(0);chunk!(8);chunk!(16);chunk!(24);i+=32;}while i<cols {chunk!(0);i+=8;}let inverse=f32x4_splat(inverse);\n");
    for i in (0..32).step_by(4) {
        s.push_str(&format!("let v=f32x4_convert_i32x4(sum_four(a{i},a{},a{},a{}));let scale=f32x4_mul(v128_load(scales.add({i}).cast()),inverse);v128_store(out.add({i}).cast(),f32x4_mul(v,scale));\n",i+1,i+2,i+3));
    }
    s.push_str("}}\n");
    s
}
// Raw bounded dot products used only by the long-context path.
fn weighted_block_kernel() -> String {
    let mut s = String::from(
        "fn weighted_block(p:*const i16,w:*const i16,stride:usize,cols:usize,out:*mut i32) {unsafe {\n",
    );
    for i in 0..32 {
        s.push_str(&format!(
            "let mut a{i}=i32x4_splat(0);let w{i}=w.add({i}*stride);\n"
        ));
    }
    s.push_str("for i in (0..cols).step_by(8) {let q=v128_load(p.add(i).cast());\n");
    for i in 0..32 {
        s.push_str(&format!(
            "a{i}=i32x4_add(a{i},i32x4_dot_i16x8(q,v128_load(w{i}.add(i).cast())));\n"
        ));
    }
    s.push_str("}\n");
    for i in (0..32).step_by(4) {
        s.push_str(&format!(
            "v128_store(out.add({i}).cast(),sum_four(a{i},a{},a{},a{}));\n",
            i + 1,
            i + 2,
            i + 3
        ));
    }
    s.push_str("}}\n");
    s
}
// Four query rows reuse each K vector. Integer sums and F32 scale order
// match score_head; only independent rows are computed together.
fn score_four_kernel() -> String {
    let mut s = String::from(
        "#[allow(clippy::too_many_arguments)] pub(super) unsafe fn score_four(q:*const i16, qstride:usize, qs:*const f32, qsstride:usize, k:*const i16, ks:*const f32, kn:usize, scores:*mut f32)->[f32;4] {unsafe {\n",
    );
    for a in 0..4 {
        for d in 0..4 {
            s.push_str(&format!(
                "let q{a}_{d}=v128_load(q.add({a}*qstride+{}).cast());\n",
                d * 8
            ));
        }
        s.push_str(&format!("let s{a}=f32x4_splat(*qs.add({a}*qsstride)*(1./(D as f32).sqrt()));let mut max{a}=f32x4_splat(f32::NEG_INFINITY);\n"));
    }
    s.push_str("let end=kn/4*4;for j in (0..end).step_by(4) {\n");
    for b in 0..4 {
        for d in 0..4 {
            s.push_str(&format!(
                "let k{b}_{d}=v128_load(k.add((j+{b})*D+{}).cast());\n",
                d * 8
            ));
        }
        for a in 0..4 {
            let terms: Vec<_> = (0..4)
                .map(|d| format!("i32x4_dot_i16x8(q{a}_{d},k{b}_{d})"))
                .collect();
            s.push_str(&format!("let a{a}_{b}={};\n", tree(&terms)));
        }
    }
    s.push_str("let sk=v128_load(ks.add(j).cast());\n");
    for a in 0..4 {
        s.push_str(&format!("let sum=sum_four(a{a}_0,a{a}_1,a{a}_2,a{a}_3);let z=f32x4_mul(f32x4_convert_i32x4(sum),f32x4_mul(s{a},sk));v128_store(scores.add({a}*kn+j).cast(),z);max{a}=f32x4_max(max{a},z);\n"));
    }
    s.push_str("}\n");
    for a in 0..4 {
        s.push_str(&format!("let mut m{a}=f32x4_extract_lane::<0>(max{a}).max(f32x4_extract_lane::<1>(max{a})).max(f32x4_extract_lane::<2>(max{a})).max(f32x4_extract_lane::<3>(max{a}));if end<kn {{let tail=std::slice::from_raw_parts_mut(scores.add({a}*kn+end),kn-end);let m=score_head(q.add({a}*qstride),*qs.add({a}*qsstride)*(1./(D as f32).sqrt()),k,ks,end,tail);if m>m{a} {{m{a}=m;}}}}\n"));
    }
    s.push_str("[m0,m1,m2,m3]}}\n");
    s
}

fn main() {
    let mut s = String::new();
    for (cols, tokens, outputs) in [
        (384, 1, 16),
        (1536, 1, 16),
        (384, 4, 32),
        (1536, 4, 32),
        (384, 8, 32),
        (1536, 8, 32),
        (384, 32, 64),
        (1536, 32, 32),
    ] {
        s.push_str(&kernel(cols, tokens, outputs, false));
    }
    if env::var_os("CARGO_FEATURE_STRASSEN").is_some() {
        s.push_str(&kernel(192, 32, 128, true));
        if env::var_os("CARGO_FEATURE_STRASSEN_QKV").is_some()
            || env::var_os("CARGO_FEATURE_STRASSEN_OUT").is_some()
        {
            s.push_str(&kernel(192, 32, 64, true).replace("packed_192(", "packed_192_64("));
        }
        if env::var_os("CARGO_FEATURE_STRASSEN_WO").is_some() {
            s.push_str(&kernel(768, 32, 64, true));
        }
    }
    fs::write(
        PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("tree.rs"),
        s,
    )
    .unwrap();
    fs::write(
        PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("weighted.rs"),
        weighted_kernel() + &weighted_block_kernel(),
    )
    .unwrap();
    fs::write(
        PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("score_four.rs"),
        score_four_kernel(),
    )
    .unwrap();
    println!("cargo:rerun-if-changed=build.rs");
}
