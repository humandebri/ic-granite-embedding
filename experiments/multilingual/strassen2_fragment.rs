fn multiply_integer(&self, q: &[i16], n: usize) -> Vec<i32> {
    let p = self.products(q, n);
    let (m, r) = (n / 2, self.rows / 2);
    let mut out = vec![0i32; n * self.rows];
    #[cfg(target_arch = "wasm32")]
    unsafe {
        use core::arch::wasm32::*;
        for t in 0..m {
            for col in (0..r).step_by(4) {
                let at = t * r + col;
                let [p0, p1, p2, p3, p4, p5, p6] =
                    std::array::from_fn(|i| v128_load(p[i].as_ptr().add(at).cast()));
                let a = i32x4_add(i32x4_sub(i32x4_add(p0, p3), p4), p6);
                let b = i32x4_add(p2, p4);
                let c = i32x4_add(p1, p3);
                let d = i32x4_add(i32x4_add(i32x4_sub(p0, p1), p2), p5);
                let dst = out.as_mut_ptr().add(t * self.rows + col);
                v128_store(dst.cast(), a);
                v128_store(dst.add(r).cast(), b);
                v128_store(dst.add(m * self.rows).cast(), c);
                v128_store(dst.add(m * self.rows + r).cast(), d);
            }
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    for t in 0..m {
        for col in 0..r {
            let [p0, p1, p2, p3, p4, p5, p6] = std::array::from_fn(|i| p[i][t * r + col]);
            out[t * self.rows + col] = p0 + p3 - p4 + p6;
            out[t * self.rows + col + r] = p2 + p4;
            out[(t + m) * self.rows + col] = p1 + p3;
            out[(t + m) * self.rows + col + r] = p0 - p1 + p2 + p5;
        }
    }
    out
}
