#[cfg(target_arch = "wasm32")]
fn quant13(row: &[f32], out: &mut [i16]) -> f32 {
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
#[cfg(target_arch = "wasm32")]
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
