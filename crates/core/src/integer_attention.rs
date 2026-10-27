//! Experimental high precision integer attention; W8A8 linear weights stay unchanged.
//! QK: +/-8191, dot width32 => max 32*8191^2 = 2,146,959,392 < i32::MAX.
//! AV: bounded 512-key i32 blocks, then i64 accumulation for longer windows.
//! Each block <=512*2047^2=2,145,387,008; full width <=5000 stays inside i64.
#![cfg(target_arch = "wasm32")]
use super::{D, max_abs};
use core::arch::wasm32::*;

#[cfg(feature = "profiling")]
pub(super) fn weighted_test(n: usize, pattern: usize) -> Vec<f32> {
    let stride = n.div_ceil(8) * 8;
    let mut p = vec![0i16; stride];
    let mut w = vec![0i16; 32 * stride];
    for t in 0..n {
        p[t] = if pattern == 3 {
            ((t * 73 + 11) % 2048) as i16
        } else {
            2047
        };
        for c in 0..32 {
            w[c * stride + t] = match pattern {
                0 => 2047,
                1 => -2047,
                2 => {
                    if t / 512 % 2 == 0 {
                        2047
                    } else {
                        -2047
                    }
                }
                _ => ((t * 41 + c * 97 + 3) % 4095) as i16 - 2047,
            };
        }
    }
    let scales = [1f32; 32];
    let mut out = vec![0.; 32];
    weighted_head(
        p.as_ptr(),
        w.as_ptr(),
        stride,
        stride,
        scales.as_ptr(),
        1.,
        out.as_mut_ptr(),
    );
    out
}

include!("attention_rows.rs");

fn pack_heads(x: &[f32], heads: usize, transpose: bool) -> (Vec<i16>, Vec<f32>) {
    let mut q = vec![0; x.len()];
    let mut scales = vec![0.; x.len() / D];
    let tokens = x.len() / (heads * D);
    for (i, row) in x.chunks_exact(D).enumerate() {
        let scale = (max_abs(row) / 8191.).max(1e-12);
        let at = if transpose {
            (i % heads) * tokens + i / heads
        } else {
            i
        };
        scales[at] = scale;
        quantize(row, scale, 8191., &mut q[at * D..(at + 1) * D]);
    }
    (q, scales)
}
/// Horizontally sum four independent integer dot products into four lanes.
/// Integer reassociation is exact: all partial sums remain inside i32 bounds.
#[inline(always)]
fn sum_four(a: v128, b: v128, c: v128, d: v128) -> v128 {
    let ab = i32x4_add(
        i32x4_shuffle::<0, 1, 4, 5>(a, b),
        i32x4_shuffle::<2, 3, 6, 7>(a, b),
    );
    let cd = i32x4_add(
        i32x4_shuffle::<0, 1, 4, 5>(c, d),
        i32x4_shuffle::<2, 3, 6, 7>(c, d),
    );
    i32x4_add(
        i32x4_shuffle::<0, 2, 4, 6>(ab, cd),
        i32x4_shuffle::<1, 3, 5, 7>(ab, cd),
    )
}
#[inline(always)]
unsafe fn qk_dot(q0: v128, q1: v128, q2: v128, q3: v128, k: *const i16) -> v128 {
    unsafe {
        let a = i32x4_dot_i16x8(q0, v128_load(k.cast()));
        let b = i32x4_dot_i16x8(q1, v128_load(k.add(8).cast()));
        let c = i32x4_dot_i16x8(q2, v128_load(k.add(16).cast()));
        let d = i32x4_dot_i16x8(q3, v128_load(k.add(24).cast()));
        i32x4_add(i32x4_add(a, b), i32x4_add(c, d))
    }
}

// Exercise the SIMD kernel against both the former path and scalar i64,
// including the maximum permitted 14-bit inputs and all key-count tails.
#[cfg(feature = "profiling")]
pub(super) fn qk_test(kn: usize, pattern: usize) -> u64 {
    let value = |i: usize| match pattern {
        0 => 8191i16,
        1 => {
            if i % 3 == 0 {
                -8191
            } else {
                8191
            }
        }
        2 => ((i * 73 + 11) % 16383) as i16 - 8191,
        _ => 0,
    };
    let q: Vec<_> = (0..4 * D).map(value).collect();
    let k: Vec<_> = (0..kn * D).map(|i| value(i * 7 + 3)).collect();
    let qs = [0.001f32, 0.003, 0.009, 0.017];
    let ks: Vec<_> = (0..kn)
        .map(|i| 0.0003 + (i % 23) as f32 * 0.00001)
        .collect();
    let mut got = vec![0.; 4 * kn];
    let maxima = unsafe {
        batched::score_four(
            q.as_ptr(),
            D,
            qs.as_ptr(),
            1,
            k.as_ptr(),
            ks.as_ptr(),
            kn,
            got.as_mut_ptr(),
        )
    };
    for t in 0..4 {
        let mut previous = vec![0.; kn];
        let maximum = score_head(
            unsafe { q.as_ptr().add(t * D) },
            qs[t] * (1. / (D as f32).sqrt()),
            k.as_ptr(),
            ks.as_ptr(),
            0,
            &mut previous,
        );
        assert_eq!(maximum.to_bits(), maxima[t].to_bits());
        for j in 0..kn {
            let sum: i64 = (0..D)
                .map(|d| q[t * D + d] as i64 * k[j * D + d] as i64)
                .sum();
            let expected = sum as f32 * ((qs[t] * (1. / (D as f32).sqrt())) * ks[j]);
            assert_eq!(
                got[t * kn + j].to_bits(),
                expected.to_bits(),
                "i64 QK rows={kn} pattern={pattern} q={t} key={j}"
            );
            assert_eq!(got[t * kn + j].to_bits(), previous[j].to_bits());
        }
    }
    got.iter().fold(0xcbf29ce484222325u64, |h, v| {
        (h ^ v.to_bits() as u64).wrapping_mul(0x100000001b3)
    })
}

// The generated head kernel shares each probability load across all 32 V channels.
include!(concat!(env!("OUT_DIR"), "/weighted.rs"));
#[allow(clippy::too_many_arguments)]
pub fn run(
    layer: usize,
    start: usize,
    rows: usize,
    heads: usize,
    ks: usize,
    kn: usize,
    q: &[f32],
    k: &[f32],
    v: &[f32],
) -> Result<Vec<f32>, String> {
    assert_eq!(q.len(), rows * heads * D);
    let (qq, qs) = phase!("quantize_q", pack_heads(q, heads, false));
    run_quantized_q(layer, start, rows, heads, ks, kn, &qq, &qs, k, v)
}

#[allow(clippy::too_many_arguments)]
pub fn run_quantized_q(
    layer: usize,
    start: usize,
    rows: usize,
    heads: usize,
    ks: usize,
    kn: usize,
    qq: &[i16],
    qs: &[f32],
    k: &[f32],
    v: &[f32],
) -> Result<Vec<f32>, String> {
    let width = heads * D;
    let kp = kn.div_ceil(8) * 8;
    assert!(
        kn <= super::MAX_TOKENS
            && qq.len() == rows * width
            && qs.len() == rows * heads
            && k.len() == kn * width
            && v.len() == kn * width
    );
    let (qk, k_scales) = phase!("quantize_k", pack_heads(k, heads, true));
    let (qv, vs) = phase!("transpose_quantize_v", {
        let mut packed = vec![0i16; width * kp];
        let mut scales = Vec::with_capacity(width);
        let mut col = vec![0.; kp];
        for d in 0..width {
            for t in 0..kn {
                col[t] = v[t * width + d];
            }
            let scale = (max_abs(&col) / 2047.).max(1e-12);
            scales.push(scale);
            quantize(&col, scale, 2047., &mut packed[d * kp..(d + 1) * kp]);
        }
        (packed, scales)
    });
    if layer % 3 == 0 && kn <= 512 && rows >= 4 {
        return batched::run(rows, heads, kn, qq, qs, &qk, &k_scales, &qv, &vs);
    }
    // Every token/head writes its full D-channel output below.
    let mut out = Vec::<f32>::with_capacity(rows * width);
    let mut scores = vec![0.; kn];
    let mut p = vec![0i16; kp];
    #[cfg(feature = "profiling")]
    let mut costs = [0u64; 3];
    for t in 0..rows {
        for h in 0..heads {
            #[cfg(feature = "profiling")]
            let qk_start = super::profile::now();
            let (first, last) = if layer % 3 == 0 {
                (0, kn)
            } else {
                (
                    (start + t).saturating_sub(64).saturating_sub(ks),
                    (start + t + 65 - ks).min(kn),
                )
            };
            let max = score_head(
                unsafe { qq.as_ptr().add(t * width + h * D) },
                qs[t * heads + h] * (1. / (D as f32).sqrt()),
                unsafe { qk.as_ptr().add(h * kn * D) },
                unsafe { k_scales.as_ptr().add(h * kn) },
                first,
                &mut scores[first..last],
            );
            if !max.is_finite() {
                return Err("integer attention nonfinite score".into());
            }
            #[cfg(feature = "profiling")]
            let sm_start = {
                let now = super::profile::now();
                costs[0] += now - qk_start;
                now
            };
            let k0 = first / 8 * 8;
            let k1 = last.div_ceil(8) * 8;
            // AV reads only this aligned interval; clear its padding, not all keys.
            p[k0..first].fill(0);
            p[last..k1].fill(0);
            let denom = softmax_quantize(&mut scores[first..last], max, &mut p[first..last]);
            #[cfg(feature = "profiling")]
            let av_start = {
                let now = super::profile::now();
                costs[1] += now - sm_start;
                now
            };

            if denom == 0 {
                return Err("integer attention zero denominator".into());
            }
            let inverse = 1. / denom as f32;
            unsafe {
                weighted_head(
                    p.as_ptr().add(k0),
                    qv.as_ptr().add(h * D * kp + k0),
                    kp,
                    k1 - k0,
                    vs.as_ptr().add(h * D),
                    inverse,
                    out.as_mut_ptr().add(t * width + h * D),
                );
            }
            #[cfg(feature = "profiling")]
            {
                costs[2] += super::profile::now() - av_start;
            }
        }
    }
    #[cfg(feature = "profiling")]
    for (name, cost) in ["qk_scores", "softmax", "av_weighted_sum"]
        .into_iter()
        .zip(costs)
    {
        super::profile::add(name, cost, (rows * heads) as u64);
    }
    // All rows*heads*D entries were written; errors above leave len=0.
    unsafe {
        out.set_len(rows * width);
    }
    Ok(out)
}

// Shared source is compiled in two modules so small-context batching cannot
// change inlining of the established long-context AV and softmax loops.
mod batched {
    use super::*;
    include!("attention_rows.rs");
    include!(concat!(env!("OUT_DIR"), "/weighted.rs"));
    include!(concat!(env!("OUT_DIR"), "/score_four.rs"));
    // Isolate register/code pressure from the existing long-context AV path.
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run(
        rows: usize,
        heads: usize,
        kn: usize,
        qq: &[i16],
        qs: &[f32],
        qk: &[i16],
        k_scales: &[f32],
        qv: &[i16],
        vs: &[f32],
    ) -> Result<Vec<f32>, String> {
        let width = heads * D;
        let kp = kn.div_ceil(8) * 8;
        let layer = 0;
        let start = 0usize;
        let ks = 0usize;
        // Every token/head writes its full D-channel output below.
        let mut out = Vec::<f32>::with_capacity(rows * width);
        let mut scores = vec![0.; kn * if layer % 3 == 0 { 4 } else { 1 }];
        let mut p = vec![0i16; kp];
        #[cfg(feature = "profiling")]
        let mut costs = [0u64; 3];
        let batch = if layer % 3 == 0 { 4 } else { 1 };
        for base in (0..rows).step_by(batch) {
            let count = (rows - base).min(batch);
            for h in 0..heads {
                #[cfg(feature = "profiling")]
                let qk_start = crate::profile::now();
                let mut maxima = [f32::NEG_INFINITY; 4];
                if count == 4 {
                    maxima = unsafe {
                        score_four(
                            qq.as_ptr().add(base * width + h * D),
                            width,
                            qs.as_ptr().add(base * heads + h),
                            heads,
                            qk.as_ptr().add(h * kn * D),
                            k_scales.as_ptr().add(h * kn),
                            kn,
                            scores.as_mut_ptr(),
                        )
                    };
                } else {
                    for (r, max) in maxima.iter_mut().enumerate().take(count) {
                        let t = base + r;
                        let (first, last) = if layer % 3 == 0 {
                            (0, kn)
                        } else {
                            (
                                (start + t).saturating_sub(64).saturating_sub(ks),
                                (start + t + 65 - ks).min(kn),
                            )
                        };
                        *max = score_head(
                            unsafe { qq.as_ptr().add(t * width + h * D) },
                            qs[t * heads + h] * (1. / (D as f32).sqrt()),
                            unsafe { qk.as_ptr().add(h * kn * D) },
                            unsafe { k_scales.as_ptr().add(h * kn) },
                            first,
                            &mut scores[r * kn + first..r * kn + last],
                        );
                    }
                }
                #[cfg(feature = "profiling")]
                {
                    costs[0] += crate::profile::now() - qk_start;
                }
                for (r, &max) in maxima.iter().enumerate().take(count) {
                    let t = base + r;
                    let scores = &mut scores[r * kn..(r + 1) * kn];
                    let (first, last) = if layer % 3 == 0 {
                        (0, kn)
                    } else {
                        (
                            (start + t).saturating_sub(64).saturating_sub(ks),
                            (start + t + 65 - ks).min(kn),
                        )
                    };
                    if !max.is_finite() {
                        return Err("integer attention nonfinite score".into());
                    }
                    #[cfg(feature = "profiling")]
                    let sm_start = { crate::profile::now() };
                    let k0 = first / 8 * 8;
                    let k1 = last.div_ceil(8) * 8;
                    // AV reads only this aligned interval; clear its padding, not all keys.
                    p[k0..first].fill(0);
                    p[last..k1].fill(0);
                    let denom =
                        softmax_quantize(&mut scores[first..last], max, &mut p[first..last]);
                    #[cfg(feature = "profiling")]
                    let av_start = {
                        let now = crate::profile::now();
                        costs[1] += now - sm_start;
                        now
                    };

                    if denom == 0 {
                        return Err("integer attention zero denominator".into());
                    }
                    let inverse = 1. / denom as f32;
                    unsafe {
                        weighted_head(
                            p.as_ptr().add(k0),
                            qv.as_ptr().add(h * D * kp + k0),
                            kp,
                            k1 - k0,
                            vs.as_ptr().add(h * D),
                            inverse,
                            out.as_mut_ptr().add(t * width + h * D),
                        );
                    }
                    #[cfg(feature = "profiling")]
                    {
                        costs[2] += crate::profile::now() - av_start;
                    }
                }
            }
        }
        #[cfg(feature = "profiling")]
        for (name, cost) in ["qk_scores", "softmax", "av_weighted_sum"]
            .into_iter()
            .zip(costs)
        {
            crate::profile::add(name, cost, (rows * heads) as u64);
        }
        // All rows*heads*D entries were written; errors above leave len=0.
        unsafe {
            out.set_len(rows * width);
        }
        Ok(out)
    }
}
