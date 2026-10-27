//! One-level exact packed Strassen products for Wi and optional Wo.
//! Leaf operands are bounded by +/-254 and width <=768.
//! Each leaf is bounded by 49,548,288 and four-term reconstruction by
//! 198,193,152, both within i32 without wrapping.
pub(super) struct Prepared {
    rows: usize,
    cols: usize,
    products: [Vec<i16>; 7],
}

// Each tuple selects a quadrant and optionally adds/subtracts another one.
type Part = (usize, usize, Option<(usize, usize, i16)>);
const WEIGHTS: [Part; 7] = [
    (0, 0, Some((1, 1, 1))),
    (0, 0, None),
    (1, 0, Some((1, 1, -1))),
    (0, 1, Some((0, 0, -1))),
    (1, 1, None),
    (0, 0, Some((1, 0, 1))),
    (0, 1, Some((1, 1, 1))),
];
const INPUTS: [Part; 7] = [
    (0, 0, Some((1, 1, 1))),
    (1, 0, Some((1, 1, 1))),
    (0, 0, None),
    (1, 1, None),
    (0, 0, Some((0, 1, 1))),
    (1, 0, Some((0, 0, -1))),
    (0, 1, Some((1, 1, -1))),
];

fn part(x: &[i16], rows: usize, cols: usize, (a, b, other): Part) -> Vec<i16> {
    assert_eq!(x.len(), rows * cols);
    assert!(rows % 2 == 0 && cols % 16 == 0);
    let (m, k) = (rows / 2, cols / 2);
    let mut out = Vec::<i16>::with_capacity(m * k);
    // SAFETY: each source quadrant is within x, all m*k destinations are written.
    unsafe {
        for t in 0..m {
            let src = x.as_ptr().add((a * m + t) * cols + b * k);
            let dst = out.as_mut_ptr().add(t * k);
            if let Some((c, d, sign)) = other {
                let rhs = x.as_ptr().add((c * m + t) * cols + d * k);
                #[cfg(target_arch = "wasm32")]
                {
                    use core::arch::wasm32::*;
                    for i in (0..k).step_by(8) {
                        let u = v128_load(src.add(i).cast());
                        let v = v128_load(rhs.add(i).cast());
                        let z = if sign == 1 {
                            i16x8_add(u, v)
                        } else {
                            i16x8_sub(u, v)
                        };
                        v128_store(dst.add(i).cast(), z);
                    }
                }
                #[cfg(not(target_arch = "wasm32"))]
                for i in 0..k {
                    dst.add(i).write(*src.add(i) + sign * *rhs.add(i));
                }
            } else {
                std::ptr::copy_nonoverlapping(src, dst, k);
            }
        }
        out.set_len(m * k);
    }
    out
}

impl Prepared {
    pub fn new(w: &[i16], rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            products: std::array::from_fn(|i| part(w, rows, cols, WEIGHTS[i])),
        }
    }
    fn products(&self, q: &[i16], n: usize) -> [Vec<i32>; 7] {
        std::array::from_fn(|i| {
            let input = phase!("prepare_input", part(q, n, self.cols, INPUTS[i]));
            phase!(
                "packed_leaf",
                leaf(
                    &input,
                    &self.products[i],
                    n / 2,
                    self.rows / 2,
                    self.cols / 2
                )
            )
        })
    }
    pub fn apply(
        &self,
        q: &[i16],
        n: usize,
        sx: &[f32],
        sw: &[f32],
        out: &mut [std::mem::MaybeUninit<f32>],
    ) {
        assert!(n % 2 == 0 && q.len() == n * self.cols && sx.len() == n && sw.len() == self.rows);
        assert_eq!(out.len(), n * self.rows);
        let out = out.as_mut_ptr().cast::<f32>();
        let (m, r) = (n / 2, self.rows / 2);
        let products = self.products(q, n);
        #[cfg(feature = "profiling")]
        let _combine = super::profile::scope("combine_and_scale");
        // SAFETY: caller supplies capacity n*rows; the four quadrants cover it.
        unsafe {
            #[cfg(target_arch = "wasm32")]
            {
                use core::arch::wasm32::*;
                for t in 0..m {
                    let s0 = f32x4_splat(sx[t]);
                    let s1 = f32x4_splat(sx[t + m]);
                    for c in (0..r).step_by(4) {
                        let at = t * r + c;
                        let [p0, p1, p2, p3, p4, p5, p6] =
                            std::array::from_fn(|i| v128_load(products[i].as_ptr().add(at).cast()));
                        let a = i32x4_add(i32x4_sub(i32x4_add(p0, p3), p4), p6);
                        let b = i32x4_add(p2, p4);
                        let c0 = i32x4_add(p1, p3);
                        let d = i32x4_add(i32x4_add(i32x4_sub(p0, p1), p2), p5);
                        let w0 = v128_load(sw.as_ptr().add(c).cast());
                        let w1 = v128_load(sw.as_ptr().add(c + r).cast());
                        let dst = out.add(t * self.rows + c);
                        v128_store(
                            dst.cast(),
                            f32x4_mul(f32x4_convert_i32x4(a), f32x4_mul(s0, w0)),
                        );
                        v128_store(
                            dst.add(r).cast(),
                            f32x4_mul(f32x4_convert_i32x4(b), f32x4_mul(s0, w1)),
                        );
                        v128_store(
                            dst.add(m * self.rows).cast(),
                            f32x4_mul(f32x4_convert_i32x4(c0), f32x4_mul(s1, w0)),
                        );
                        v128_store(
                            dst.add(m * self.rows + r).cast(),
                            f32x4_mul(f32x4_convert_i32x4(d), f32x4_mul(s1, w1)),
                        );
                    }
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            for t in 0..m {
                for c in 0..r {
                    let [p0, p1, p2, p3, p4, p5, p6] =
                        std::array::from_fn(|i| products[i][t * r + c]);
                    for (t0, c0, v) in [
                        (t, c, p0 + p3 - p4 + p6),
                        (t, c + r, p2 + p4),
                        (t + m, c, p1 + p3),
                        (t + m, c + r, p0 - p1 + p2 + p5),
                    ] {
                        out.add(t0 * self.rows + c0)
                            .write(v as f32 * (sx[t0] * sw[c0]));
                    }
                }
            }
        }
    }
}
#[cfg(all(target_arch = "wasm32", feature = "strassen-wo"))]
use super::packed_768 as packed_half;

fn leaf(q: &[i16], w: &[i16], n: usize, rows: usize, cols: usize) -> Vec<i32> {
    assert!(q.len() == n * cols && w.len() == rows * cols);
    let mut out = Vec::<i32>::with_capacity(n * rows);
    unsafe {
        #[cfg(target_arch = "wasm32")]
        {
            let tile = 32;
            let outputs = if cols == 192 && rows % 128 == 0 {
                128
            } else {
                64
            };
            assert!([192, super::I / 2].contains(&cols) && n % tile == 0 && rows % outputs == 0);
            for t in (0..n).step_by(tile) {
                for r in (0..rows).step_by(outputs) {
                    let xp = q.as_ptr().add(t * cols);
                    let wp = w.as_ptr().add(r * cols);
                    let yp = out.as_mut_ptr().add(t * rows + r);
                    match cols {
                        192 if outputs == 128 => super::packed_192(xp, wp, yp, rows),
                        #[cfg(any(feature = "strassen-qkv", feature = "strassen-out"))]
                        192 => super::packed_192_64(xp, wp, yp, rows),
                        #[cfg(feature = "strassen-wo")]
                        c if c == super::I / 2 => packed_half(xp, wp, yp, rows),
                        _ => unreachable!(),
                    }
                }
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        for t in 0..n {
            for r in 0..rows {
                let dot = q[t * cols..(t + 1) * cols]
                    .iter()
                    .zip(&w[r * cols..(r + 1) * cols])
                    .map(|(a, b)| *a as i32 * *b as i32)
                    .sum();
                out.as_mut_ptr().add(t * rows + r).write(dot);
            }
        }
        out.set_len(n * rows);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packed_strassen_matches_direct_products_at_int8_extremes() {
        let (n, rows) = (8, 16);
        for (cols, pattern) in [384, 1536]
            .into_iter()
            .flat_map(|cols| (0..3).map(move |p| (cols, p)))
        {
            let value = |i: usize| match pattern {
                0 => {
                    if i % 3 == 0 {
                        -127
                    } else {
                        127
                    }
                }
                1 => ((i * 73 + 11) % 255) as i16 - 127,
                _ => 0,
            };
            let q: Vec<_> = (0..n * cols).map(value).collect();
            let w: Vec<_> = (0..rows * cols).map(|i| value(i * 7 + 3)).collect();
            let sx: Vec<_> = (0..n).map(|i| 0.01 + i as f32 / 100.).collect();
            let sw: Vec<_> = (0..rows).map(|i| 0.02 + i as f32 / 200.).collect();
            let mut got = Vec::<f32>::with_capacity(n * rows);
            Prepared::new(&w, rows, cols).apply(
                &q,
                n,
                &sx,
                &sw,
                &mut got.spare_capacity_mut()[..n * rows],
            );
            unsafe {
                got.set_len(n * rows);
            }
            for t in 0..n {
                for r in 0..rows {
                    let exact: i64 = (0..cols)
                        .map(|k| q[t * cols + k] as i64 * w[r * cols + k] as i64)
                        .sum();
                    assert_eq!(
                        got[t * rows + r].to_bits(),
                        (exact as f32 * (sx[t] * sw[r])).to_bits()
                    );
                }
            }
        }
    }
}
