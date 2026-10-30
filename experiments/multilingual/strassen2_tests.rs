#[cfg(test)]
mod recursive_tests {
    use super::*;
    #[test]
    fn second_level_matches_i64_products_at_int8_extremes() {
        for (n, rows, cols) in [
            (128, 384, 384),
            (256, 384, 1152),
            (128, 1152, 384),
            (256, 2304, 384),
        ] {
            for pattern in 0..2 {
                let value = |i: usize| {
                    if pattern == 0 {
                        if i % 3 == 0 { -127i16 } else { 127i16 }
                    } else {
                        ((i * 73 + 11) % 255) as i16 - 127
                    }
                };
                let q: Vec<_> = (0..n * cols).map(value).collect();
                let w: Vec<_> = (0..rows * cols).map(|i| value(i * 7 + 3)).collect();
                // Exercise recursion even when the runtime selects it only
                // for the MLP output projection.
                let prepared = Prepared::new_depth(&w, rows, cols, 2);
                assert!(prepared.children.is_some());
                let mut got = Vec::<f32>::with_capacity(n * rows);
                prepared.apply(
                    &q,
                    n,
                    &vec![1.; n],
                    &vec![1.; rows],
                    &mut got.spare_capacity_mut()[..n * rows],
                );
                unsafe {
                    got.set_len(n * rows);
                }
                for t in [0, 1, n / 2 - 1, n / 2, n - 2, n - 1] {
                    for r in [0, 1, rows / 2 - 1, rows / 2, rows - 2, rows - 1] {
                        let exact: i64 = (0..cols)
                            .map(|k| q[t * cols + k] as i64 * w[r * cols + k] as i64)
                            .sum();
                        assert_eq!(
                            got[t * rows + r].to_bits(),
                            (exact as f32).to_bits(),
                            "n={n} rows={rows} cols={cols} t={t} r={r}"
                        );
                    }
                }
            }
        }
    }
}
