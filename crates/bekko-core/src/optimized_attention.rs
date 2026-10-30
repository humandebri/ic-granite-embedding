//! a25m Attention: BF16 KV, QK13 integer dot, four independent AV rows.
#[cfg(not(target_arch = "wasm32"))]
use super::bf16;
#[cfg(not(target_arch = "wasm32"))]
use super::softmax_exp;
#[cfg(target_arch = "wasm32")]
use super::softmax_row;
use super::{D, H, MAX_TOKENS, max_abs};
include!(concat!(env!("OUT_DIR"), "/bekko_av.rs"));
include!(concat!(env!("OUT_DIR"), "/bekko_qk.rs"));

fn finite(values: &[f32]) -> bool {
    #[cfg(target_arch = "wasm32")]
    unsafe {
        use core::arch::wasm32::*;
        let mut chunks = values.chunks_exact(4);
        for row in chunks.by_ref() {
            // SAFETY: each chunk has four initialized elements.
            if !i32x4_all_true(f32x4_le(
                f32x4_abs(v128_load(row.as_ptr().cast())),
                f32x4_splat(f32::MAX),
            )) {
                return false;
            }
        }
        chunks.remainder().iter().all(|v| v.is_finite())
    }
    #[cfg(not(target_arch = "wasm32"))]
    values.iter().all(|v| v.is_finite())
}

pub(super) fn compute(layer: usize, tokens: usize, qkv: &[f32]) -> Result<Vec<f32>, String> {
    if !(2..=MAX_TOKENS).contains(&tokens) || qkv.len() != tokens * 3 * H || !finite(qkv) {
        return Err("invalid internal Attention input".into());
    }
    let mut q = vec![0f32; tokens * H];
    let mut k = vec![0f32; tokens * H];
    let mut v = vec![0f32; tokens * H];
    for t in 0..tokens {
        q[t * H..(t + 1) * H].copy_from_slice(&qkv[t * 3 * H..t * 3 * H + H]);
        for (side, dest) in [(1, &mut k), (2, &mut v)] {
            let src = &qkv[t * 3 * H + side * H..t * 3 * H + (side + 1) * H];
            let dst = &mut dest[t * H..(t + 1) * H];
            #[cfg(target_arch = "wasm32")]
            unsafe {
                use core::arch::wasm32::*;
                // SAFETY: H is a multiple of four; source/destination are
                // disjoint complete rows. Round BF16 exactly as the wire codec.
                for i in (0..H).step_by(4) {
                    let bits = v128_load(src.as_ptr().add(i).cast());
                    let bump = i32x4_add(
                        i32x4_splat(0x7fff),
                        v128_and(u32x4_shr(bits, 16), i32x4_splat(1)),
                    );
                    v128_store(
                        dst.as_mut_ptr().add(i).cast(),
                        v128_and(i32x4_add(bits, bump), i32x4_splat(0xffff0000u32 as i32)),
                    );
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            for (a, b) in dst.iter_mut().zip(src) {
                *a = f32::from_bits((bf16(*b) as u32) << 16);
            }
        }
    }
    if !finite(&k) || !finite(&v) {
        return Err("BF16 Attention overflow".into());
    }
    let mut keys = vec![0i16; k.len()];
    let mut scales = vec![0f32; tokens * 6];
    for j in 0..tokens {
        for h in 0..6 {
            let at = j * H + h * D;
            scales[j * 6 + h] = quant13(&k[at..at + D], &mut keys[at..at + D]);
        }
    }
    let mut out = vec![0f32; tokens * H];
    let mut scores = vec![0f32; 4 * tokens];
    for t in (0..tokens).step_by(4) {
        let count = (tokens - t).min(4);
        let mut first = [0usize; 4];
        let mut last = [0usize; 4];
        for a in 0..count {
            (first[a], last[a]) = if layer % 3 == 0 {
                (0, tokens)
            } else {
                ((t + a).saturating_sub(64), (t + a + 65).min(tokens))
            };
        }
        for h in 0..6 {
            let mut inverse = [0f32; 4];
            let maximum = score_four13(
                &q[t * H + h * D..],
                H,
                &keys,
                &scales,
                h,
                6,
                tokens,
                &first,
                &last,
                count,
                &mut scores,
            );
            for a in 0..count {
                let active = &mut scores[a * tokens + first[a]..a * tokens + last[a]];
                let max = maximum[a];
                #[cfg(target_arch = "wasm32")]
                {
                    inverse[a] = 1. / softmax_row(active, max);
                }
                #[cfg(not(target_arch = "wasm32"))]
                {
                    let mut denom = 0f32;
                    for score in active {
                        *score = softmax_exp(*score - max);
                        denom += *score;
                    }
                    inverse[a] = 1. / denom;
                }
            }
            value_four(
                &v,
                tokens,
                h,
                &scores,
                &inverse,
                &first,
                &last,
                count,
                &mut out[t * H..],
            );
        }
    }
    Ok(out)
}

#[cfg(not(target_arch = "wasm32"))]
fn score_four13(
    q: &[f32],
    qstride: usize,
    k: &[i16],
    scales: &[f32],
    head: usize,
    heads: usize,
    kn: usize,
    first: &[usize; 4],
    last: &[usize; 4],
    count: usize,
    scores: &mut [f32],
) -> [f32; 4] {
    let mut maximum = [f32::NEG_INFINITY; 4];
    for a in 0..count {
        maximum[a] = score13(
            &q[a * qstride..a * qstride + D],
            k,
            scales,
            first[a] * heads * D + head * D,
            heads * D,
            first[a] * heads + head,
            heads,
            &mut scores[a * kn + first[a]..a * kn + last[a]],
        );
    }
    maximum
}

#[cfg(not(target_arch = "wasm32"))]
fn value_four(
    v: &[f32],
    kn: usize,
    h: usize,
    scores: &[f32],
    inverse: &[f32; 4],
    first: &[usize; 4],
    last: &[usize; 4],
    count: usize,
    out: &mut [f32],
) {
    for a in 0..count {
        for j in first[a]..last[a] {
            let p = scores[a * kn + j] * inverse[a];
            for d in 0..D {
                out[a * H + h * D + d] += p * v[j * H + h * D + d];
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn quant13(row: &[f32], out: &mut [i16]) -> f32 {
    assert_eq!(row.len(), D);
    assert_eq!(out.len(), D);
    let scale = (max_abs(row) / 8191.).max(1e-12);
    for (a, b) in out.iter_mut().zip(row) {
        *a = (*b / scale).round_ties_even().clamp(-8191., 8191.) as i16;
    }
    scale
}

#[cfg(not(target_arch = "wasm32"))]
fn score13(
    query: &[f32],
    k: &[i16],
    scales: &[f32],
    offset: usize,
    stride: usize,
    scale_offset: usize,
    heads: usize,
    scores: &mut [f32],
) -> f32 {
    let mut q = [0i16; D];
    let qs = quant13(query, &mut q);
    let mut max = f32::NEG_INFINITY;
    for (j, score) in scores.iter_mut().enumerate() {
        let total: i64 = q
            .iter()
            .zip(&k[offset + j * stride..offset + j * stride + D])
            .map(|(a, b)| *a as i64 * *b as i64)
            .sum();
        *score = total as f32 * (qs * scales[scale_offset + j * heads]) * 0.125;
        max = max.max(*score);
    }
    max
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn integer_dot_widens_beyond_i32_without_overflow() {
        let q = [1f32; D];
        let k = [8191i16; D];
        let mut scores = [0f32; 1];
        score13(&q, &k, &[1. / 8191.], 0, D, 0, 1, &mut scores);
        assert!((scores[0] - 8.).abs() < 2e-6);
        assert_eq!(64i64 * 8191 * 8191, 4_293_918_784);
    }
    #[test]
    fn nonfinite_and_invalid_shapes_are_rejected() {
        assert!(compute(0, 1, &[]).is_err());
        for v in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
            let qkv = vec![v; 2 * 3 * H];
            assert!(compute(0, 2, &qkv).is_err());
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn quant13(row: &[f32], out: &mut [i16]) -> f32 {
    assert_eq!(row.len(), D);
    assert_eq!(out.len(), D);
    use core::arch::wasm32::*;
    let scale = (max_abs(row) / 8191.).max(1e-12);
    unsafe {
        for i in (0..row.len()).step_by(8) {
            let a = f32x4_nearest(f32x4_div(
                v128_load(row.as_ptr().add(i).cast()),
                f32x4_splat(scale),
            ));
            let b = f32x4_nearest(f32x4_div(
                v128_load(row.as_ptr().add(i + 4).cast()),
                f32x4_splat(scale),
            ));
            let a = i32x4_trunc_sat_f32x4(f32x4_min(
                f32x4_splat(8191.),
                f32x4_max(f32x4_splat(-8191.), a),
            ));
            let b = i32x4_trunc_sat_f32x4(f32x4_min(
                f32x4_splat(8191.),
                f32x4_max(f32x4_splat(-8191.), b),
            ));
            v128_store(out.as_mut_ptr().add(i).cast(), i16x8_narrow_i32x4(a, b));
        }
    }
    scale
}
#[cfg(all(target_arch = "wasm32", test))]
fn score13(
    query: &[f32],
    k: &[i16],
    scales: &[f32],
    offset: usize,
    stride: usize,
    scale_offset: usize,
    heads: usize,
    scores: &mut [f32],
) -> f32 {
    use core::arch::wasm32::*;
    let mut q = [0i16; 64];
    let qs = quant13(query, &mut q);
    let mut max = f32::NEG_INFINITY;
    unsafe {
        for (j, score) in scores.iter_mut().enumerate() {
            let mut sum = i32x4_splat(0);
            for d in (0..64).step_by(8) {
                sum = i32x4_add(
                    sum,
                    i32x4_dot_i16x8(
                        v128_load(q.as_ptr().add(d).cast()),
                        v128_load(k.as_ptr().add(offset + j * stride + d).cast()),
                    ),
                );
            }
            // Each lane <=16*8191^2=1,073,479,696. Use i64 across lanes;
            // the complete dot may reach 4,293,918,784, exceeding i32.
            let total = i32x4_extract_lane::<0>(sum) as i64
                + i32x4_extract_lane::<1>(sum) as i64
                + i32x4_extract_lane::<2>(sum) as i64
                + i32x4_extract_lane::<3>(sum) as i64;
            *score = total as f32 * (qs * scales[scale_offset + j * heads]) * 0.125;
            max = max.max(*score);
        }
    }
    max
}
