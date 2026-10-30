fn attention_direct(
    layer: usize,
    start: usize,
    rows: usize,
    head: usize,
    heads: usize,
    ks: usize,
    ke: usize,
    qkv: &[f32],
) -> Result<Vec<f32>, String> {
    let width = heads * D;
    let mut q = vec![0f32; rows * width];
    for t in 0..rows {
        q[t * width..(t + 1) * width].copy_from_slice(
            &qkv[(start + t) * 3 * H + head * D..(start + t) * 3 * H + (head + heads) * D],
        );
    }
    let mut kv = [vec![0f32; (ke - ks) * width], vec![0f32; (ke - ks) * width]];
    for (side, dest) in kv.iter_mut().enumerate() {
        for t in ks..ke {
            let src = &qkv[t * 3 * H + (side + 1) * H + head * D
                ..t * 3 * H + (side + 1) * H + (head + heads) * D];
            let out = &mut dest[(t - ks) * width..(t - ks + 1) * width];
            #[cfg(target_arch = "wasm32")]
            unsafe {
                use core::arch::wasm32::*;
                for i in (0..width).step_by(4) {
                    let bits = v128_load(src.as_ptr().add(i).cast());
                    let bump = i32x4_add(
                        i32x4_splat(0x7fff),
                        v128_and(u32x4_shr(bits, 16), i32x4_splat(1)),
                    );
                    let rounded =
                        v128_and(i32x4_add(bits, bump), i32x4_splat(0xffff0000u32 as i32));
                    v128_store(out.as_mut_ptr().add(i).cast(), rounded);
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            for (a, b) in out.iter_mut().zip(src) {
                *a = f32::from_bits((bf16(*b) as u32) << 16);
            }
        }
    }
    attention_values(layer, start, rows, heads, ks, ke - ks, &q, &kv[0], &kv[1])
}

#[cfg(test)]
mod direct_attention_tests {
    use super::*;

    #[test]
    fn direct_buffers_match_wire_at_global_and_local_window_boundaries() {
        let mut qkv = vec![0f32; 256 * 3 * H];
        for (i, v) in qkv.iter_mut().enumerate() {
            *v = ((i * 97 % 8191) as f32 - 4095.) / 1024.;
        }
        for (layer, start, rows, ks, ke) in [
            (0, 0, 128, 0, 256),
            (3, 128, 128, 0, 256),
            (1, 0, 128, 0, 192),
            (2, 128, 128, 64, 256),
        ] {
            for head in [0, 3] {
                let wire = attention_wire(&qkv, start, rows, head, 3, ks, ke);
                let old = attention(layer, start, rows, head, 3, ks, ke - ks, &wire).unwrap();
                let direct = attention_direct(layer, start, rows, head, 3, ks, ke, &qkv).unwrap();
                assert!(
                    old.iter()
                        .zip(direct)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
            }
        }
    }
}
