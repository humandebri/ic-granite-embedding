// Shared source, separately compiled for long and batched attention.
// Bounded 512-key dot blocks, joined in i64 before one F32 cast.
fn weighted_head_wide(
    p: *const i16,
    w: *const i16,
    stride: usize,
    cols: usize,
    scales: *const f32,
    inverse: f32,
    out: *mut f32,
) {
    assert!(cols <= crate::MAX_TOKENS.div_ceil(8) * 8 && cols % 8 == 0);
    let mut sums = [0i64; 32];
    unsafe {
        for base in (0..cols).step_by(512) {
            let mut part = [std::mem::MaybeUninit::<i32>::uninit(); 32];
            weighted_block(
                p.add(base),
                w.add(base),
                stride,
                (cols - base).min(512),
                part.as_mut_ptr().cast(),
            );
            for i in 0..32 {
                sums[i] += part[i].assume_init() as i64;
            }
        }
        for i in (0..32).step_by(4) {
            let value = f32x4(
                sums[i] as f32,
                sums[i + 1] as f32,
                sums[i + 2] as f32,
                sums[i + 3] as f32,
            );
            let scale = f32x4_mul(v128_load(scales.add(i).cast()), f32x4_splat(inverse));
            v128_store(out.add(i).cast(), f32x4_mul(value, scale));
        }
    }
}

fn quantize(x: &[f32], scale: f32, limit: f32, out: &mut [i16]) {
    assert_eq!(x.len(), out.len());
    let end = x.len() / 8 * 8;
    unsafe {
        let vs = f32x4_splat(scale);
        let lo = f32x4_splat(-limit);
        let hi = f32x4_splat(limit);
        for i in (0..end).step_by(8) {
            let a = f32x4_nearest(f32x4_div(v128_load(x.as_ptr().add(i).cast()), vs));
            let b = f32x4_nearest(f32x4_div(v128_load(x.as_ptr().add(i + 4).cast()), vs));
            let a = i32x4_trunc_sat_f32x4(f32x4_min(hi, f32x4_max(lo, a)));
            let b = i32x4_trunc_sat_f32x4(f32x4_min(hi, f32x4_max(lo, b)));
            v128_store(out.as_mut_ptr().add(i).cast(), i16x8_narrow_i32x4(a, b));
        }
    }
    for i in end..x.len() {
        out[i] = (x[i] / scale).round_ties_even().clamp(-limit, limit) as i16;
    }
}

// Quantize unnormalized exp scores; AV uses the integer denominator.
fn softmax_quantize(scores: &mut [f32], max: f32, out: &mut [i16]) -> i32 {
    assert_eq!(scores.len(), out.len());
    let end = scores.len() / 8 * 8;
    let mut tail_sum = 0;
    unsafe {
        let vm = f32x4_splat(max);
        let vs = f32x4_splat(1. / 2047.);
        let lo = f32x4_splat(-2047.);
        let hi = f32x4_splat(2047.);
        let mut sum = i32x4_splat(0);
        for i in (0..end).step_by(8) {
            let a = crate::exp4(f32x4_sub(v128_load(scores.as_ptr().add(i).cast()), vm));
            let b = crate::exp4(f32x4_sub(v128_load(scores.as_ptr().add(i + 4).cast()), vm));
            let a = f32x4_nearest(f32x4_div(a, vs));
            let b = f32x4_nearest(f32x4_div(b, vs));
            let a = i32x4_trunc_sat_f32x4(f32x4_min(hi, f32x4_max(lo, a)));
            let b = i32x4_trunc_sat_f32x4(f32x4_min(hi, f32x4_max(lo, b)));
            sum = i32x4_add(sum, i32x4_add(a, b));
            v128_store(out.as_mut_ptr().add(i).cast(), i16x8_narrow_i32x4(a, b));
        }
        tail_sum += i32x4_extract_lane::<0>(sum)
            + i32x4_extract_lane::<1>(sum)
            + i32x4_extract_lane::<2>(sum)
            + i32x4_extract_lane::<3>(sum);
    }
    // Preserve the original four-lane/scalar tail partition exactly.
    crate::softmax_row(&mut scores[end..], max);
    quantize(&scores[end..], 1. / 2047., 2047., &mut out[end..]);
    tail_sum += out[end..].iter().map(|v| *v as i32).sum::<i32>();
    tail_sum
}

fn score_head(
    q: *const i16,
    qscale: f32,
    k: *const i16,
    scales: *const f32,
    first: usize,
    scores: &mut [f32],
) -> f32 {
    unsafe {
        let q0 = v128_load(q.cast());
        let q1 = v128_load(q.add(8).cast());
        let q2 = v128_load(q.add(16).cast());
        let q3 = v128_load(q.add(24).cast());
        let mut max = f32::NEG_INFINITY;
        let mut vmax = f32x4_splat(max);
        let mut j = 0;
        while j + 4 <= scores.len() {
            let ptr = k.add((first + j) * D);
            let sums = sum_four(
                qk_dot(q0, q1, q2, q3, ptr),
                qk_dot(q0, q1, q2, q3, ptr.add(D)),
                qk_dot(q0, q1, q2, q3, ptr.add(2 * D)),
                qk_dot(q0, q1, q2, q3, ptr.add(3 * D)),
            );
            let scale = f32x4_mul(f32x4_splat(qscale), v128_load(scales.add(first + j).cast()));
            let z = f32x4_mul(f32x4_convert_i32x4(sums), scale);
            v128_store(scores.as_mut_ptr().add(j).cast(), z);
            vmax = f32x4_max(vmax, z);
            j += 4;
        }
        max = f32x4_extract_lane::<0>(vmax)
            .max(f32x4_extract_lane::<1>(vmax))
            .max(f32x4_extract_lane::<2>(vmax))
            .max(f32x4_extract_lane::<3>(vmax));
        while j < scores.len() {
            let v = qk_dot(q0, q1, q2, q3, k.add((first + j) * D));
            let sum = i32x4_extract_lane::<0>(v)
                + i32x4_extract_lane::<1>(v)
                + i32x4_extract_lane::<2>(v)
                + i32x4_extract_lane::<3>(v);
            let z = sum as f32 * (qscale * *scales.add(first + j));
            scores[j] = z;
            if z > max {
                max = z;
            }
            j += 1;
        }
        max
    }
}
