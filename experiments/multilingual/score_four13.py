"""Generate QK13 for four independent queries sharing the same K loads."""


def generate():
    s = r"""
#[cfg(target_arch="wasm32")]
fn score_four13(q:&[f32],qstride:usize,k:&[i16],scales:&[f32],head:usize,heads:usize,kn:usize,first:&[usize;4],last:&[usize;4],count:usize,scores:&mut[f32])->[f32;4] {
 use core::arch::wasm32::*;
 let mut quant=[0i16;256];let mut qs=[1e-12f32;4];let mut maximum=[f32::NEG_INFINITY;4];
 for a in 0..count {qs[a]=quant13(&q[a*qstride..a*qstride+64],&mut quant[a*64..(a+1)*64]);}
 unsafe {
"""
    for a in range(4):
        for d in range(8):
            s += f"let q{a}_{d}=v128_load(quant.as_ptr().add({a * 64 + d * 8}).cast());\n"
    s += "for j in first[0]..last[count-1] {let key=k.as_ptr().add(j*heads*64+head*64);let ks=scales[j*heads+head];\n"
    for d in range(8):
        s += f"let k{d}=v128_load(key.add({d * 8}).cast());\n"
    for a in range(4):
        s += f"let mut sum{a}=i32x4_dot_i16x8(q{a}_0,k0);\n"
        for d in range(1, 8):
            s += f"sum{a}=i32x4_add(sum{a},i32x4_dot_i16x8(q{a}_{d},k{d}));\n"
        s += f"if {a}<count && j>=first[{a}] && j<last[{a}] {{\n"
        s += f"let total=i32x4_extract_lane::<0>(sum{a})as i64+i32x4_extract_lane::<1>(sum{a})as i64+i32x4_extract_lane::<2>(sum{a})as i64+i32x4_extract_lane::<3>(sum{a})as i64;\n"
        s += f"let score=total as f32*(qs[{a}]*ks)*0.125;scores[{a}*kn+j]=score;maximum[{a}]=maximum[{a}].max(score);}}\n"
    s += "}}maximum}\n"
    return s
