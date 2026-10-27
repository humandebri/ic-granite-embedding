//! Lossless, bounded F32 byte-plane transport. No precision is discarded.
use super::{DEFAULT_MAX_TOKENS, H, float_bytes, floats};
const MAGIC: &[u8; 4] = b"GRC1";

pub fn encode(values: &[f32]) -> Result<Vec<u8>, String> {
    encode_raw(values)
}

fn encode_raw(values: &[f32]) -> Result<Vec<u8>, String> {
    if values.len() > DEFAULT_MAX_TOKENS * H || values.iter().any(|v| !v.is_finite()) {
        return Err("carry values".into());
    }
    let mut out = Vec::with_capacity(12 + values.len() * 4);
    out.extend(MAGIC);
    out.extend(0u32.to_le_bytes());
    out.extend((values.len() as u32).to_le_bytes());
    out.extend(float_bytes(values));
    Ok(out)
}

// Build old-format fixtures only for decoder compatibility tests.
#[cfg(test)]
fn encode_compressed(values: &[f32]) -> Result<Vec<u8>, String> {
    if values.len() > DEFAULT_MAX_TOKENS * H || values.iter().any(|v| !v.is_finite()) {
        return Err("carry values".into());
    }
    let raw = float_bytes(values);
    let n = values.len();
    let mut planes = vec![0u8; raw.len()];
    for (i, bytes) in raw.chunks_exact(4).enumerate() {
        for j in 0..4 {
            planes[j * n + i] = bytes[j];
        }
    }
    let compressed = miniz_oxide::deflate::compress_to_vec(&planes, 1);
    let use_compressed = compressed.len() < raw.len();
    let mut out = Vec::with_capacity(12 + raw.len().min(compressed.len()));
    out.extend(MAGIC);
    out.extend((use_compressed as u32).to_le_bytes());
    out.extend((n as u32).to_le_bytes());
    out.extend(if use_compressed { &compressed } else { &raw });
    Ok(out)
}

pub fn decode(frame: &[u8], expected: usize) -> Result<Vec<f32>, String> {
    if expected > DEFAULT_MAX_TOKENS * H
        || frame.len() < 12
        || frame.len() > 12 + expected * 4
        || &frame[..4] != MAGIC
    {
        return Err("carry frame bounds/version".into());
    }
    let mode = u32::from_le_bytes(frame[4..8].try_into().unwrap());
    if u32::from_le_bytes(frame[8..12].try_into().unwrap()) as usize != expected {
        return Err("carry shape".into());
    }
    let raw = match mode {
        0 => {
            if frame.len() != 12 + expected * 4 {
                return Err("carry raw length".into());
            }
            return floats(&frame[12..]);
        }
        1 => {
            let planes =
                miniz_oxide::inflate::decompress_to_vec_with_limit(&frame[12..], expected * 4)
                    .map_err(|_| "carry deflate".to_string())?;
            if planes.len() != expected * 4 {
                return Err("carry decoded length".into());
            }
            let mut raw = vec![0; planes.len()];
            for (i, bytes) in raw.chunks_exact_mut(4).enumerate() {
                for j in 0..4 {
                    bytes[j] = planes[j * expected + i];
                }
            }
            raw
        }
        _ => return Err("carry codec".into()),
    };
    floats(&raw)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn finite_f32_bits_round_trip_including_signed_zero_and_extremes() {
        let mut x = vec![
            0.,
            -0.,
            f32::from_bits(1),
            -f32::from_bits(1),
            f32::MAX,
            f32::MIN,
            f32::MIN_POSITIVE,
        ];
        for i in 0..8192u32 {
            let bits = i.wrapping_mul(2654435761);
            let f = f32::from_bits(bits);
            if f.is_finite() {
                x.push(f);
            }
        }
        for input in [x, vec![1.25; 8192]] {
            for frame in [
                encode_raw(&input).unwrap(),
                encode_compressed(&input).unwrap(),
            ] {
                let decoded = decode(&frame, input.len()).unwrap();
                assert_eq!(
                    input.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    decoded.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                );
            }
        }
    }
    #[test]
    fn rejects_wrong_shape_truncation_bombs_and_nonfinite_values() {
        let frame = encode_compressed(&vec![1.25; 8192]).unwrap();
        assert!(decode(&frame, 8191).is_err());
        assert!(decode(&frame[..frame.len() / 2], 8192).is_err());
        let mut bomb = frame.clone();
        bomb[8..12].copy_from_slice(&1024u32.to_le_bytes());
        assert!(decode(&bomb, 1024).unwrap_err().contains("deflate"));
        let mut nan = b"GRC1".to_vec();
        nan.extend(0u32.to_le_bytes());
        nan.extend(1u32.to_le_bytes());
        nan.extend(f32::NAN.to_le_bytes());
        assert!(decode(&nan, 1).unwrap_err().contains("nonfinite"));
        assert!(encode(&[f32::INFINITY]).is_err());
    }
}
