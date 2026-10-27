//! Client-owned Q:F32, K/V:BF16 rows, retained in their wire representation.
use super::{D, H, MAX_TOKENS, append_float_bytes, max_abs};

#[derive(Default)]
pub struct PackedQkv {
    bytes: Vec<u8>,
}
impl PackedQkv {
    pub fn with_capacity(rows: usize) -> Self {
        assert!(rows <= MAX_TOKENS);
        Self {
            bytes: Vec::with_capacity(rows * H * 8),
        }
    }
    pub fn rows(&self) -> usize {
        self.bytes.len() / (H * 8)
    }
    pub fn extend(&mut self, bytes: &[u8]) -> Result<(), String> {
        if bytes.is_empty()
            || bytes.len() % (H * 8) != 0
            || self.rows() + bytes.len() / (H * 8) > MAX_TOKENS
        {
            return Err("packed qkv length".into());
        }
        for row in bytes.chunks_exact(H * 8) {
            if row[..H * 4]
                .chunks_exact(4)
                .any(|v| !f32::from_le_bytes(v.try_into().unwrap()).is_finite())
                || row[H * 4..]
                    .chunks_exact(2)
                    .any(|v| u16::from_le_bytes(v.try_into().unwrap()) & 0x7f80 == 0x7f80)
            {
                return Err("packed qkv nonfinite".into());
            }
        }
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }
    /// Q rows and selected heads followed by the original BF16 K and V bytes.
    pub fn attention_wire(
        &self,
        start: usize,
        rows: usize,
        head: usize,
        heads: usize,
        ks: usize,
        ke: usize,
        quantized_q: bool,
    ) -> Vec<u8> {
        assert!(rows > 0 && start <= self.rows() && rows <= self.rows() - start);
        assert!(heads > 0 && heads <= 12 && head <= 12 - heads);
        assert!(ks <= start && ke >= start + rows && ke <= self.rows());
        let width = heads * D;
        let qbytes = if quantized_q {
            rows * heads * (D * 2 + 4)
        } else {
            rows * width * 4
        };
        let mut out = Vec::with_capacity(qbytes + (ke - ks) * width * 4);
        if quantized_q {
            append_q14(
                &mut out,
                (start..start + rows).flat_map(|t| {
                    let q =
                        &self.bytes[t * H * 8 + head * D * 4..t * H * 8 + (head + heads) * D * 4];
                    q.chunks_exact(D * 4).map(|q| {
                        std::array::from_fn(|i| {
                            f32::from_le_bytes(q[i * 4..i * 4 + 4].try_into().unwrap())
                        })
                    })
                }),
            );
        } else {
            for t in start..start + rows {
                out.extend_from_slice(
                    &self.bytes[t * H * 8 + head * D * 4..t * H * 8 + (head + heads) * D * 4],
                );
            }
        }
        for side in 0..2 {
            for t in ks..ke {
                let at = t * H * 8 + H * 4 + side * H * 2;
                out.extend_from_slice(&self.bytes[at + head * D * 2..at + (head + heads) * D * 2]);
            }
        }
        out
    }
}

pub(super) fn append_q14(out: &mut Vec<u8>, queries: impl Iterator<Item = [f32; D]>) {
    let mut scales = Vec::new();
    for q in queries {
        let scale = (max_abs(&q) / 8191.).max(1e-12);
        scales.push(scale);
        out.extend(q.iter().flat_map(|v| {
            ((v / scale).round_ties_even().clamp(-8191., 8191.) as i16).to_le_bytes()
        }));
    }
    append_float_bytes(out, &scales);
}

pub(super) fn finite_query(q: &[f32]) -> [f32; D] {
    assert!(q.iter().all(|v| v.is_finite()), "finite Q");
    q.try_into().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[should_panic(expected = "finite Q")]
    fn float_q14_helper_preserves_nonfinite_rejection() {
        let mut qkv = vec![0.; 3 * H];
        qkv[0] = f32::NAN;
        crate::packed_attention_wire(&qkv, 0, 1, 0, 1, 0, 1);
    }
    #[test]
    fn retained_packed_rows_match_both_attention_formats() {
        let qkv: Vec<f32> = (0..33 * 3 * H)
            .map(|i| ((i * 71 % 2049) as f32 - 1024.) / 997.)
            .collect();
        let bytes = crate::pack_qkv(&qkv);
        let mut packed = PackedQkv::default();
        packed.extend(&bytes[..7 * H * 8]).unwrap();
        packed.extend(&bytes[7 * H * 8..]).unwrap();
        assert_eq!(packed.rows(), 33);
        for heads in [1, 3, 12] {
            for head in (0..12).step_by(heads) {
                assert_eq!(
                    packed.attention_wire(1, 17, head, heads, 0, 33, false),
                    crate::attention_wire(&qkv, 1, 17, head, heads, 0, 33)
                );
                assert_eq!(
                    packed.attention_wire(1, 17, head, heads, 0, 33, true),
                    crate::packed_attention_wire(&qkv, 1, 17, head, heads, 0, 33)
                );
            }
        }
    }
    #[test]
    fn malformed_packet_leaves_accumulated_rows_unchanged() {
        let mut packed = PackedQkv::default();
        let valid = vec![0; H * 8];
        packed.extend(&valid).unwrap();
        for at in [0, H * 4, H * 6] {
            let mut bad = valid.clone();
            if at == 0 {
                bad[..4].copy_from_slice(&f32::NAN.to_le_bytes());
            } else {
                bad[at..at + 2].copy_from_slice(&0x7f80u16.to_le_bytes());
            }
            assert!(packed.extend(&bad).is_err());
            assert_eq!(packed.rows(), 1);
        }
        assert!(packed.extend(&valid[..valid.len() - 1]).is_err());
        assert!(packed.extend(&vec![0; MAX_TOKENS * H * 8]).is_err());
        assert_eq!(packed.rows(), 1);
    }
}
