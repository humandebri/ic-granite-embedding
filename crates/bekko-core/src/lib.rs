//! Shared fixed-shape Bekko encoder. Model aliases pin layer count and precision.
use serde::Deserialize;
use std::{cell::RefCell, collections::BTreeMap};
macro_rules! phase {
    ($name:expr, $body:expr) => {
        $body
    };
}
include!(concat!(env!("OUT_DIR"), "/tree.rs"));
mod optimized_attention;
#[cfg(feature = "strassen")]
#[allow(unexpected_cfgs)]
mod strassen;
#[cfg(all(target_arch = "wasm32", feature = "strassen-wo"))]
use self::packed_576 as packed_768;
pub const H: usize = 384;
pub const D: usize = 64;
pub const I: usize = 1152;
pub const MAX_TOKENS: usize = 256;
pub const VOCAB: u32 = 256000;
pub const MODEL_ID: &str = "hotchpotch/bekko-embedding-v1-a8m";
pub const REVISION: &str = "c721113d59a1d91b447450324f51c4b3332c924a";
pub const PACK_BYTES: usize = 400967494;
pub const PACK_SHA256: &str = "1feddfa11f8e232a30b39c4965328a396fa9388dc65bb06cad043045e2dc56ef";
pub fn validate_tokens(ids: &[u32]) -> Result<(), String> {
    if !(2..=MAX_TOKENS).contains(&ids.len()) {
        return Err("supported input: 2..=256 tokens including BOS/EOS".into());
    }
    if ids.first() != Some(&2) || ids.last() != Some(&1) {
        return Err("expected BOS (2) ... EOS (1)".into());
    }
    if ids.contains(&0) || ids.iter().any(|id| *id >= VOCAB) {
        return Err("invalid token ID or padding".into());
    }
    Ok(())
}
#[derive(Deserialize)]
struct Descriptor {
    shape: Vec<usize>,
    offset: usize,
    bytes: usize,
    kind: String,
}

fn tensor_range(
    data_len: usize,
    descriptor: &Descriptor,
    shape: &[usize],
    kind: &str,
) -> Result<std::ops::Range<usize>, String> {
    if descriptor.shape != shape || descriptor.kind != kind {
        return Err("tensor shape/kind".into());
    }
    let bytes = match (kind, shape) {
        ("f32", _) => shape.iter().try_fold(4usize, |n, d| n.checked_mul(*d)),
        ("i8", [rows, cols]) => cols.checked_add(4).and_then(|n| rows.checked_mul(n)),
        _ => None,
    }
    .ok_or("tensor size overflow")?;
    if descriptor.bytes != bytes {
        return Err("tensor length".into());
    }
    let end = descriptor.offset.checked_add(bytes).ok_or("tensor range")?;
    if end > data_len {
        return Err("tensor range".into());
    }
    Ok(descriptor.offset..end)
}

fn tensor_bytes<'a>(
    data: &'a [u8],
    descriptor: &Descriptor,
    shape: &[usize],
    kind: &str,
) -> Result<&'a [u8], String> {
    let range = tensor_range(data.len(), descriptor, shape, kind)?;
    Ok(&data[range])
}

pub struct Linear {
    #[cfg(feature = "strassen")]
    prepared: Option<strassen::Prepared>,
    weights: Vec<i16>,
    scales: Vec<f32>,
    rows: usize,
    cols: usize,
    bucketed: bool,
}
pub struct Layer {
    qkv: Linear,
    out: Linear,
    wi: Linear,
    wo: Linear,
    attn_norm: Vec<f32>,
    mlp_norm: Vec<f32>,
}
pub struct Encoder<const LAYERS: usize, const QK13: bool> {
    embeddings: Vec<f32>,
    embedding_norm: Vec<f32>,
    final_norm: Vec<f32>,
    layers: Vec<Layer>,
    rope: [Vec<[(f32, f32); 32]>; 2],
}
/// The existing a8m model retains its four-layer F32 Attention path.
pub type Model = Encoder<4, false>;

fn floats(b: &[u8]) -> Result<Vec<f32>, String> {
    if b.len() % 4 != 0 {
        return Err("unaligned f32".into());
    }
    #[cfg(target_endian = "little")]
    let v = unsafe {
        let mut v = Vec::<f32>::with_capacity(b.len() / 4);
        // F32 accepts every bit pattern. No source alignment is required for u8.
        std::ptr::copy_nonoverlapping(b.as_ptr(), v.as_mut_ptr().cast::<u8>(), b.len());
        v.set_len(b.len() / 4);
        v
    };
    #[cfg(not(target_endian = "little"))]
    let v: Vec<_> = b
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect();
    if v.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite input".into());
    }
    Ok(v)
}
fn append_float_bytes(out: &mut Vec<u8>, x: &[f32]) {
    #[cfg(target_endian = "little")]
    unsafe {
        let bytes = x.len().checked_mul(4).expect("F32 byte length overflow");
        let start = out.len();
        out.reserve(bytes);
        // The disjoint source and fresh capacity cover exactly bytes elements.
        std::ptr::copy_nonoverlapping(x.as_ptr().cast::<u8>(), out.as_mut_ptr().add(start), bytes);
        out.set_len(start + bytes);
    }
    #[cfg(not(target_endian = "little"))]
    for v in x {
        out.extend(v.to_le_bytes());
    }
}
pub fn float_bytes(x: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(x.len() * 4);
    append_float_bytes(&mut out, x);
    out
}

pub fn bf16(x: f32) -> u16 {
    let b = x.to_bits();
    ((b.wrapping_add(0x7fff + ((b >> 16) & 1))) >> 16) as u16
}
pub fn append_bf16(out: &mut Vec<u8>, x: &[f32]) {
    let start = out.len();
    out.reserve(x.len() * 2);
    let mut i = 0;
    #[cfg(target_arch = "wasm32")]
    unsafe {
        use core::arch::wasm32::*;
        while i + 8 <= x.len() {
            let round = |bits: v128| {
                u32x4_shr(
                    i32x4_add(
                        bits,
                        i32x4_add(
                            i32x4_splat(0x7fff),
                            v128_and(u32x4_shr(bits, 16), i32x4_splat(1)),
                        ),
                    ),
                    16,
                )
            };
            let a = round(v128_load(x.as_ptr().add(i).cast()));
            let b = round(v128_load(x.as_ptr().add(i + 4).cast()));
            v128_store(
                out.as_mut_ptr().add(start + i * 2).cast(),
                u16x8_narrow_i32x4(a, b),
            );
            i += 8;
        }
    }
    // Every byte in the new capacity is written before exposing its length.
    unsafe {
        while i < x.len() {
            let bytes = bf16(x[i]).to_le_bytes();
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr().add(start + i * 2), 2);
            i += 1;
        }
        out.set_len(start + x.len() * 2);
    }
}

fn read_bf16(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|v| f32::from_bits((u16::from_le_bytes(v.try_into().unwrap()) as u32) << 16))
        .collect()
}

impl<const LAYERS: usize, const QK13: bool> Encoder<LAYERS, QK13> {
    pub fn load(pack: &[u8]) -> Result<Self, String> {
        if !matches!(LAYERS, 4 | 13) || QK13 && LAYERS != 13 {
            return Err("unsupported fixed Bekko configuration".into());
        }
        let len = usize::try_from(u64::from_le_bytes(
            pack.get(..8).ok_or("pack header")?.try_into().unwrap(),
        ))
        .map_err(|_| "pack metadata length")?;
        let end = 8usize.checked_add(len).ok_or("pack metadata length")?;
        let header: BTreeMap<String, Descriptor> =
            serde_json::from_slice(pack.get(8..end).ok_or("pack metadata")?)
                .map_err(|e| e.to_string())?;
        let data = &pack[end..];
        let get = |name: &str, shape: &[usize], kind: &str| -> Result<&[u8], String> {
            let d = header.get(name).ok_or_else(|| format!("missing {name}"))?;
            tensor_bytes(data, d, shape, kind).map_err(|e| format!("{name}: {e}"))
        };
        let f = |name: &str, shape: &[usize]| -> Result<Vec<f32>, String> {
            floats(get(name, shape, "f32")?)
        };
        let linear = |name: &str, rows: usize, cols: usize| -> Result<Linear, String> {
            let b = get(name, &[rows, cols], "i8")?;
            let weights: Vec<i16> = b[..rows * cols].iter().map(|x| *x as i8 as i16).collect();
            if weights.iter().any(|v| !(-127..=127).contains(v)) {
                return Err("INT8 range".into());
            }
            #[cfg(feature = "strassen")]
            let prepared = (rows == 2 * I && cols == H
                || cfg!(feature = "strassen-wo") && rows == H && cols == I
                || cfg!(feature = "strassen-qkv") && rows == 3 * H && cols == H
                || cfg!(feature = "strassen-out") && rows == H && cols == H)
                .then(|| {
                    strassen::Prepared::new_depth(
                        &weights,
                        rows,
                        cols,
                        if LAYERS == 13 && cols == I { 2 } else { 1 },
                    )
                });
            Ok(Linear {
                #[cfg(feature = "strassen")]
                prepared,
                weights,
                scales: floats(&b[rows * cols..])?,
                rows,
                cols,
                bucketed: LAYERS == 13,
            })
        };
        let mut layers = Vec::new();
        for l in 0..LAYERS {
            let p = format!("layers.{l}");
            layers.push(Layer {
                qkv: linear(&format!("{p}.attn.Wqkv.weight"), 3 * H, H)?,
                out: linear(&format!("{p}.attn.Wo.weight"), H, H)?,
                wi: linear(&format!("{p}.mlp.Wi.weight"), 2 * I, H)?,
                wo: linear(&format!("{p}.mlp.Wo.weight"), H, I)?,
                attn_norm: if l == 0 {
                    Vec::new()
                } else {
                    f(&format!("{p}.attn_norm.weight"), &[H])?
                },
                mlp_norm: f(&format!("{p}.mlp_norm.weight"), &[H])?,
            });
        }
        Ok(Self {
            embeddings: f("embeddings.tok_embeddings.weight", &[256000, H])?,
            embedding_norm: f("embeddings.norm.weight", &[H])?,
            final_norm: f("final_norm.weight", &[H])?,
            layers,
            rope: [160000f32, 160000f32].map(|theta| {
                let frequencies: [f32; 32] =
                    std::array::from_fn(|i| theta.powf(-((2 * i) as f32) / D as f32));
                (0..MAX_TOKENS)
                    .map(|position| {
                        std::array::from_fn(|i| (position as f32 * frequencies[i]).sin_cos())
                    })
                    .collect()
            }),
        })
    }

    /// Loads a verified pack through bounded reads without copying it into the heap.
    /// Keep tensor validation and model construction equivalent to `load`.
    pub fn load_from(
        pack_len: usize,
        read: impl FnMut(usize, &mut [u8]) -> Result<(), String>,
    ) -> Result<Self, String> {
        if !matches!(LAYERS, 4 | 13) || QK13 && LAYERS != 13 {
            return Err("unsupported fixed Bekko configuration".into());
        }
        if pack_len < 8 {
            return Err("pack header".into());
        }
        let read = RefCell::new(read);
        let mut header_len = [0; 8];
        read.borrow_mut()(0, &mut header_len)?;
        let len =
            usize::try_from(u64::from_le_bytes(header_len)).map_err(|_| "pack metadata length")?;
        let end = 8usize.checked_add(len).ok_or("pack metadata length")?;
        if end > pack_len || len > 1_048_576 {
            return Err("pack metadata length".into());
        }
        let mut metadata = vec![0; len];
        read.borrow_mut()(8, &mut metadata)?;
        let header: BTreeMap<String, Descriptor> =
            serde_json::from_slice(&metadata).map_err(|e| e.to_string())?;
        let range =
            |name: &str, shape: &[usize], kind: &str| -> Result<std::ops::Range<usize>, String> {
                let d = header.get(name).ok_or_else(|| format!("missing {name}"))?;
                let r = tensor_range(pack_len - end, d, shape, kind)
                    .map_err(|e| format!("{name}: {e}"))?;
                Ok(end + r.start..end + r.end)
            };
        let f = |name: &str, shape: &[usize]| -> Result<Vec<f32>, String> {
            let r = range(name, shape, "f32")?;
            let mut values = vec![0.0_f32; r.len() / 4];
            #[cfg(target_endian = "little")]
            {
                // f32 accepts every bit pattern; the initialized allocation covers r.len() bytes.
                let bytes = unsafe {
                    std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(), r.len())
                };
                for (i, chunk) in bytes.chunks_mut(1_048_576).enumerate() {
                    read.borrow_mut()(r.start + i * 1_048_576, chunk)?;
                }
            }
            #[cfg(not(target_endian = "little"))]
            for (i, chunk) in values.chunks_mut(16_384).enumerate() {
                let mut bytes = vec![0; chunk.len() * 4];
                read.borrow_mut()(r.start + i * 65_536, &mut bytes)?;
                for (v, b) in chunk.iter_mut().zip(bytes.chunks_exact(4)) {
                    *v = f32::from_le_bytes(b.try_into().unwrap());
                }
            }
            if values.iter().any(|x| !x.is_finite()) {
                return Err("nonfinite input".into());
            }
            Ok(values)
        };
        let linear = |name: &str, rows: usize, cols: usize| -> Result<Linear, String> {
            let r = range(name, &[rows, cols], "i8")?;
            let mut b = vec![0; r.len()];
            read.borrow_mut()(r.start, &mut b)?;
            let weights: Vec<i16> = b[..rows * cols].iter().map(|x| *x as i8 as i16).collect();
            if weights.iter().any(|v| !(-127..=127).contains(v)) {
                return Err("INT8 range".into());
            }
            #[cfg(feature = "strassen")]
            let prepared = (rows == 2 * I && cols == H
                || cfg!(feature = "strassen-wo") && rows == H && cols == I
                || cfg!(feature = "strassen-qkv") && rows == 3 * H && cols == H
                || cfg!(feature = "strassen-out") && rows == H && cols == H)
                .then(|| {
                    strassen::Prepared::new_depth(
                        &weights,
                        rows,
                        cols,
                        if LAYERS == 13 && cols == I { 2 } else { 1 },
                    )
                });
            Ok(Linear {
                #[cfg(feature = "strassen")]
                prepared,
                weights,
                scales: floats(&b[rows * cols..])?,
                rows,
                cols,
                bucketed: LAYERS == 13,
            })
        };
        let mut layers = Vec::new();
        for l in 0..LAYERS {
            let p = format!("layers.{l}");
            layers.push(Layer {
                qkv: linear(&format!("{p}.attn.Wqkv.weight"), 3 * H, H)?,
                out: linear(&format!("{p}.attn.Wo.weight"), H, H)?,
                wi: linear(&format!("{p}.mlp.Wi.weight"), 2 * I, H)?,
                wo: linear(&format!("{p}.mlp.Wo.weight"), H, I)?,
                attn_norm: if l == 0 {
                    Vec::new()
                } else {
                    f(&format!("{p}.attn_norm.weight"), &[H])?
                },
                mlp_norm: f(&format!("{p}.mlp_norm.weight"), &[H])?,
            });
        }
        Ok(Self {
            embeddings: f("embeddings.tok_embeddings.weight", &[256000, H])?,
            embedding_norm: f("embeddings.norm.weight", &[H])?,
            final_norm: f("final_norm.weight", &[H])?,
            layers,
            rope: [160000f32, 160000f32].map(|theta| {
                let frequencies: [f32; 32] =
                    std::array::from_fn(|i| theta.powf(-((2 * i) as f32) / D as f32));
                (0..MAX_TOKENS)
                    .map(|position| {
                        std::array::from_fn(|i| (position as f32 * frequencies[i]).sin_cos())
                    })
                    .collect()
            }),
        })
    }

    pub fn embed(&self, ids: &[u32]) -> Result<Vec<f32>, String> {
        validate_tokens(ids)?;
        let mut x = Vec::with_capacity(ids.len() * H);
        for id in ids {
            x.extend_from_slice(&self.embeddings[*id as usize * H..(*id as usize + 1) * H]);
        }
        norm(&mut x, &self.embedding_norm);
        Ok(x)
    }

    pub fn qkv(&self, layer: usize, start: usize, x: &[f32]) -> Vec<f32> {
        let l = &self.layers[layer];
        let mut n = x.to_vec();
        if layer != 0 {
            norm(&mut n, &l.attn_norm);
        }
        let mut y = phase!("projection", l.qkv.apply(&n));
        assert!(start <= MAX_TOKENS && x.len() / H <= MAX_TOKENS - start);
        let rope = &self.rope[if layer % 3 == 0 { 0 } else { 1 }];
        for (t, row) in y.chunks_exact_mut(3 * H).enumerate() {
            let rotations = rope[start + t];
            for side in 0..2 {
                for head in 0..6 {
                    for i in 0..32 {
                        let (s, c) = rotations[i];
                        let a = side * H + head * D + i;
                        let b = a + 32;
                        let (u, v) = (row[a], row[b]);
                        row[a] = u * c - v * s;
                        row[b] = v * c + u * s;
                    }
                }
            }
        }
        y
    }

    pub fn finish(&self, layer: usize, residual: &[f32], context: &[f32]) -> Vec<f32> {
        let l = &self.layers[layer];
        let x = phase!("out_projection", l.out.apply(context));
        self.finish_projected(layer, residual, x)
    }

    fn finish_projected(&self, layer: usize, residual: &[f32], mut x: Vec<f32>) -> Vec<f32> {
        let l = &self.layers[layer];
        for (a, b) in x.iter_mut().zip(residual) {
            *a += b;
        }
        let mut n = x.clone();
        norm(&mut n, &l.mlp_norm);
        let wide = phase!("mlp_wi", l.wi.apply(&n));
        let gates = gelu_gates(&wide);
        let delta = phase!("mlp_wo", l.wo.apply(&gates));
        for (a, b) in x.iter_mut().zip(delta) {
            *a += b;
        }
        x
    }

    pub fn full(&self, ids: &[u32]) -> Result<Vec<f32>, String> {
        let mut x = self.embed(ids)?;
        for layer in 0..LAYERS {
            let qkv = self.qkv(layer, 0, &x);
            if QK13 {
                let context = optimized_attention::compute(layer, ids.len(), &qkv)?;
                x = self.finish(layer, &x, &context);
                continue;
            }
            let mut context = vec![0.; ids.len() * H];
            for start in (0..ids.len()).step_by(128) {
                let rows = 128.min(ids.len() - start);
                for head in (0..6).step_by(3) {
                    let (kstart, kend) = key_range(layer, start, rows, ids.len());
                    let wire = attention_wire(&qkv, start, rows, head, 3, kstart, kend);
                    let result =
                        attention(layer, start, rows, head, 3, kstart, kend - kstart, &wire)?;
                    for t in 0..rows {
                        context[(start + t) * H + head * D..(start + t) * H + (head + 3) * D]
                            .copy_from_slice(&result[t * 3 * D..(t + 1) * 3 * D]);
                    }
                }
            }
            x = self.finish(layer, &x, &context);
        }
        {
            norm(&mut x, &self.final_norm);
            let mut mean = vec![0f32; H];
            for row in x.chunks_exact(H) {
                for i in 0..H {
                    mean[i] += row[i] / ids.len() as f32;
                }
            }
            let scale = mean.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            for v in &mut mean {
                *v /= scale;
            }
            Ok(mean)
        }
    }
}
fn norm(x: &mut [f32], weight: &[f32]) {
    for row in x.chunks_exact_mut(H) {
        let mean = row.iter().sum::<f32>() / H as f32;
        #[cfg(target_arch = "wasm32")]
        let var = unsafe {
            use core::arch::wasm32::*;
            let vm = f32x4_splat(mean);
            let mut sum = 0.;
            for i in (0..H).step_by(4) {
                let d = f32x4_sub(v128_load(row.as_ptr().add(i).cast()), vm);
                let v = f32x4_mul(d, d);
                // Keep the original sequential floating-point reduction order.
                sum += f32x4_extract_lane::<0>(v);
                sum += f32x4_extract_lane::<1>(v);
                sum += f32x4_extract_lane::<2>(v);
                sum += f32x4_extract_lane::<3>(v);
            }
            sum / H as f32
        };
        #[cfg(not(target_arch = "wasm32"))]
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / H as f32;
        let r = 1. / (var + 1e-5).sqrt();
        #[cfg(target_arch = "wasm32")]
        unsafe {
            use core::arch::wasm32::*;
            assert!(weight.len() >= H);
            let vm = f32x4_splat(mean);
            let vr = f32x4_splat(r);
            for i in (0..H).step_by(4) {
                let v = f32x4_sub(v128_load(row.as_ptr().add(i).cast()), vm);
                let w = v128_load(weight.as_ptr().add(i).cast());
                v128_store(
                    row.as_mut_ptr().add(i).cast(),
                    f32x4_mul(f32x4_mul(v, vr), w),
                );
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        for (v, w) in row.iter_mut().zip(weight) {
            *v = (*v - mean) * r * w;
        }
    }
}

fn gelu_gates(wide: &[f32]) -> Vec<f32> {
    assert_eq!(wide.len() % (2 * I), 0);
    let mut gates = Vec::<f32>::with_capacity(wide.len() / 2);
    #[cfg(all(target_arch = "wasm32", feature = "approx-gelu"))]
    unsafe {
        use core::arch::wasm32::*;
        for (row_index, row) in wide.chunks_exact(2 * I).enumerate() {
            for i in (0..I).step_by(4) {
                let v = v128_load(row.as_ptr().add(i).cast());
                let gate = v128_load(row.as_ptr().add(I + i).cast());
                let x = f32x4_mul(v, f32x4_splat(std::f32::consts::FRAC_1_SQRT_2));
                let ax = f32x4_abs(x);
                let t = f32x4_div(
                    f32x4_splat(1.),
                    f32x4_add(f32x4_splat(1.), f32x4_mul(f32x4_splat(0.3275911), ax)),
                );
                let p = f32x4_add(
                    f32x4_splat(-1.453152027),
                    f32x4_mul(f32x4_splat(1.061405429), t),
                );
                let p = f32x4_add(f32x4_splat(1.421413741), f32x4_mul(p, t));
                let p = f32x4_add(f32x4_splat(-0.284496736), f32x4_mul(p, t));
                let p = f32x4_add(f32x4_splat(0.254829592), f32x4_mul(p, t));
                let p = f32x4_mul(p, t);
                let erf = f32x4_sub(
                    f32x4_splat(1.),
                    f32x4_mul(p, exp4(f32x4_neg(f32x4_mul(ax, ax)))),
                );
                let erf = v128_xor(erf, v128_and(x, i32x4_splat(i32::MIN)));
                let value = f32x4_mul(
                    f32x4_mul(
                        f32x4_mul(f32x4_splat(0.5), v),
                        f32x4_add(f32x4_splat(1.), erf),
                    ),
                    gate,
                );
                v128_store(gates.as_mut_ptr().add(row_index * I + i).cast(), value);
            }
        }
        gates.set_len(wide.len() / 2);
    }
    #[cfg(not(all(target_arch = "wasm32", feature = "approx-gelu")))]
    for row in wide.chunks_exact(2 * I) {
        for i in 0..I {
            let v = row[i];
            gates.push(
                0.5 * v * (1. + libm::erff(v * std::f32::consts::FRAC_1_SQRT_2)) * row[I + i],
            );
        }
    }
    gates
}

fn max_abs(row: &[f32]) -> f32 {
    #[cfg(target_arch = "wasm32")]
    unsafe {
        use core::arch::wasm32::*;
        let mut m = f32x4_splat(0.);
        for i in (0..row.len()).step_by(4) {
            m = f32x4_max(m, f32x4_abs(v128_load(row.as_ptr().add(i).cast())));
        }
        // Preserve Rust max-number behavior even for nonfinite internal values.
        if !v128_any_true(f32x4_ne(m, m)) {
            return f32x4_extract_lane::<0>(m)
                .max(f32x4_extract_lane::<1>(m))
                .max(f32x4_extract_lane::<2>(m))
                .max(f32x4_extract_lane::<3>(m));
        }
    }
    row.iter().fold(0f32, |m, v| m.max(v.abs()))
}
fn quantize_into(row: &[f32], scale: f32, quant: &mut [std::mem::MaybeUninit<i16>]) {
    assert_eq!(row.len(), quant.len());
    #[cfg(target_arch = "wasm32")]
    unsafe {
        use core::arch::wasm32::*;
        let scale = f32x4_splat(scale);
        let lo = f32x4_splat(-127.);
        let hi = f32x4_splat(127.);
        for i in (0..row.len()).step_by(8) {
            let a = f32x4_nearest(f32x4_div(v128_load(row.as_ptr().add(i).cast()), scale));
            let b = f32x4_nearest(f32x4_div(v128_load(row.as_ptr().add(i + 4).cast()), scale));
            let a = i32x4_trunc_sat_f32x4(f32x4_min(hi, f32x4_max(lo, a)));
            let b = i32x4_trunc_sat_f32x4(f32x4_min(hi, f32x4_max(lo, b)));
            v128_store(quant.as_mut_ptr().add(i).cast(), i16x8_narrow_i32x4(a, b));
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    for (q, v) in quant.iter_mut().zip(row) {
        q.write((v / scale).round_ties_even().clamp(-127., 127.) as i16);
    }
}

impl Linear {
    fn apply(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len() % self.cols, 0);
        let n = x.len() / self.cols;
        // All entries are written by the disjoint token/output tiles below.
        // Expose the initialized length only after every tile has finished.
        let mut quant = Vec::<i16>::with_capacity(x.len());
        let mut scales = Vec::with_capacity(n);
        for (t, row) in x.chunks_exact(self.cols).enumerate() {
            let scale = (max_abs(row) / 127.).max(1e-12);
            scales.push(scale);
            quantize_into(
                row,
                scale,
                &mut quant.spare_capacity_mut()[t * self.cols..(t + 1) * self.cols],
            );
        }
        // Every row was initialized through MaybeUninit, without a zero pass.
        unsafe {
            quant.set_len(x.len());
        }
        self.apply_quantized(&quant, &scales)
    }
    fn apply_quantized(&self, quant: &[i16], scales: &[f32]) -> Vec<f32> {
        let n = scales.len();
        assert_eq!(quant.len(), n * self.cols);
        // Quantization is per row. Zero rows cannot change any real token's
        // product, and rounding up avoids expensive scalar/tiny GEMM tails.
        // Padding exists only inside this linear operator, never in Attention
        // or mean pooling. a8m keeps its existing execution path.
        if self.bucketed && n >= 64 && n % 64 != 0 {
            let padded = n.next_multiple_of(64);
            let mut q = quant.to_vec();
            q.resize(padded * self.cols, 0);
            let mut s = scales.to_vec();
            s.resize(padded, 1.);
            let mut out = self.apply_quantized(&q, &s);
            out.truncate(n * self.rows);
            return out;
        }
        let mut y = Vec::<f32>::with_capacity(n * self.rows);
        let mut t = 0;
        #[cfg(feature = "strassen")]
        if n >= 64 {
            if let Some(prepared) = &self.prepared {
                let bulk = n / 64 * 64;
                prepared.apply(
                    &quant[..bulk * self.cols],
                    bulk,
                    &scales[..bulk],
                    &self.scales,
                    &mut y.spare_capacity_mut()[..bulk * self.rows],
                );
                t = bulk;
            }
        }
        #[cfg(target_arch = "wasm32")]
        while t + 32 <= n {
            let output_tile = if self.cols == H { 64 } else { 32 };
            for r in (0..self.rows).step_by(output_tile) {
                unsafe {
                    let xp = quant.as_ptr().add(t * self.cols);
                    let wp = self.weights.as_ptr().add(r * self.cols);
                    let yp = y.as_mut_ptr().add(t * self.rows + r);
                    let sx = scales.as_ptr().add(t);
                    let sw = self.scales.as_ptr().add(r);
                    if self.cols == H {
                        tree_384_32(xp, wp, yp, self.rows, sx, sw);
                    } else {
                        tree_1152_32(xp, wp, yp, self.rows, sx, sw);
                    }
                }
            }
            t += 32;
        }
        #[cfg(target_arch = "wasm32")]
        while t + 8 <= n {
            let outputs = 32;
            for r in (0..self.rows).step_by(outputs) {
                unsafe {
                    let xp = quant.as_ptr().add(t * self.cols);
                    let wp = self.weights.as_ptr().add(r * self.cols);
                    let yp = y.as_mut_ptr().add(t * self.rows + r);
                    let sx = scales.as_ptr().add(t);
                    let sw = self.scales.as_ptr().add(r);
                    if self.cols == H {
                        tree_384_8(xp, wp, yp, self.rows, sx, sw);
                    } else {
                        tree_1152_8(xp, wp, yp, self.rows, sx, sw);
                    }
                }
            }
            t += 8;
        }
        #[cfg(target_arch = "wasm32")]
        for block in [4, 1] {
            while t + block <= n {
                let outputs = if block != 1 { 32 } else { 16 };
                for r in (0..self.rows).step_by(outputs) {
                    unsafe {
                        let xp = quant.as_ptr().add(t * self.cols);
                        let wp = self.weights.as_ptr().add(r * self.cols);
                        let yp = y.as_mut_ptr().add(t * self.rows + r);
                        let sx = scales.as_ptr().add(t);
                        let sw = self.scales.as_ptr().add(r);
                        match (self.cols, block) {
                            (H, 4) => tree_384_4(xp, wp, yp, self.rows, sx, sw),
                            (I, 4) => tree_1152_4(xp, wp, yp, self.rows, sx, sw),
                            (H, 1) => tree_384_1(xp, wp, yp, self.rows, sx, sw),
                            (I, 1) => tree_1152_1(xp, wp, yp, self.rows, sx, sw),
                            _ => unreachable!(),
                        }
                    }
                }
                t += block;
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            while t + 4 <= n {
                for r in (0..self.rows).step_by(4) {
                    let input = &quant[t * self.cols..(t + 4) * self.cols];
                    let weights = &self.weights[r * self.cols..(r + 4) * self.cols];
                    let sums = if self.cols == H {
                        dot_tile_fixed::<H>(input, weights)
                    } else {
                        dot_tile_fixed::<I>(input, weights)
                    };
                    for a in 0..4 {
                        for b in 0..4 {
                            unsafe {
                                y.as_mut_ptr().add((t + a) * self.rows + r + b).write(
                                    sums[a * 4 + b] as f32 * (scales[t + a] * self.scales[r + b]),
                                );
                            }
                        }
                    }
                }
                t += 4;
            }
            while t < n {
                for r in (0..self.rows).step_by(4) {
                    let sums = dot_four(
                        &quant[t * self.cols..(t + 1) * self.cols],
                        &self.weights[r * self.cols..(r + 4) * self.cols],
                    );
                    for j in 0..4 {
                        unsafe {
                            y.as_mut_ptr()
                                .add(t * self.rows + r + j)
                                .write(sums[j] as f32 * (scales[t] * self.scales[r + j]));
                        }
                    }
                }
                t += 1;
            }
        }
        // SAFETY: loader restricts rows to multiples of 32 and cols to H/I.
        // Each loop covers disjoint full output rows, including all tails.
        unsafe {
            y.set_len(n * self.rows);
        }

        y
    }
}

/// Four output channels share activation loads; 32 columns per loop iteration.
/// All model linear dimensions are multiples of 32 and output rows multiples of 4.
#[cfg(any(not(target_arch = "wasm32"), test))]
#[inline(always)]
fn dot_four(x: &[i16], w: &[i16]) -> [i32; 4] {
    let cols = x.len();
    assert!(cols % 8 == 0 && w.len() == 4 * cols);
    #[cfg(target_arch = "wasm32")]
    unsafe {
        use core::arch::wasm32::*;
        let mut a0 = i32x4_splat(0);
        let mut a1 = a0;
        let mut a2 = a0;
        let mut a3 = a0;
        let mut i = 0;
        while i + 32 <= cols {
            macro_rules! chunk {
                ($offset:expr) => {{
                    let v = v128_load(x.as_ptr().add(i + $offset).cast());
                    a0 = i32x4_add(
                        a0,
                        i32x4_dot_i16x8(v, v128_load(w.as_ptr().add(i + $offset).cast())),
                    );
                    a1 = i32x4_add(
                        a1,
                        i32x4_dot_i16x8(v, v128_load(w.as_ptr().add(cols + i + $offset).cast())),
                    );
                    a2 = i32x4_add(
                        a2,
                        i32x4_dot_i16x8(
                            v,
                            v128_load(w.as_ptr().add(2 * cols + i + $offset).cast()),
                        ),
                    );
                    a3 = i32x4_add(
                        a3,
                        i32x4_dot_i16x8(
                            v,
                            v128_load(w.as_ptr().add(3 * cols + i + $offset).cast()),
                        ),
                    );
                }};
            }
            chunk!(0);
            chunk!(8);
            chunk!(16);
            chunk!(24);
            i += 32;
        }
        while i < cols {
            let v = v128_load(x.as_ptr().add(i).cast());
            a0 = i32x4_add(a0, i32x4_dot_i16x8(v, v128_load(w.as_ptr().add(i).cast())));
            a1 = i32x4_add(
                a1,
                i32x4_dot_i16x8(v, v128_load(w.as_ptr().add(cols + i).cast())),
            );
            a2 = i32x4_add(
                a2,
                i32x4_dot_i16x8(v, v128_load(w.as_ptr().add(2 * cols + i).cast())),
            );
            a3 = i32x4_add(
                a3,
                i32x4_dot_i16x8(v, v128_load(w.as_ptr().add(3 * cols + i).cast())),
            );
            i += 8;
        }
        fn sum(v: v128) -> i32 {
            i32x4_extract_lane::<0>(v)
                + i32x4_extract_lane::<1>(v)
                + i32x4_extract_lane::<2>(v)
                + i32x4_extract_lane::<3>(v)
        }
        return [sum(a0), sum(a1), sum(a2), sum(a3)];
    }
    #[cfg(not(target_arch = "wasm32"))]
    std::array::from_fn(|r| dot_i16(x, &w[r * cols..(r + 1) * cols]))
}

/// 4 tokens x 4 output rows, with shared activation and weight loads.
#[inline(always)]
#[cfg(test)]
fn dot_tile(x: &[i16], w: &[i16], cols: usize) -> [i32; 16] {
    match cols {
        H => dot_tile_fixed::<H>(x, w),
        I => dot_tile_fixed::<I>(x, w),
        _ => panic!("unsupported dot width"),
    }
}

#[cfg(any(not(target_arch = "wasm32"), test))]
#[inline(always)]
fn dot_tile_fixed<const COLS: usize>(x: &[i16], w: &[i16]) -> [i32; 16] {
    let cols = COLS;
    assert!(cols % 8 == 0 && x.len() == 4 * cols && w.len() == 4 * cols);
    #[cfg(target_arch = "wasm32")]
    unsafe {
        use core::arch::wasm32::*;
        let z = i32x4_splat(0);
        let (mut a00, mut a01, mut a02, mut a03) = (z, z, z, z);
        let (mut a10, mut a11, mut a12, mut a13) = (z, z, z, z);
        let (mut a20, mut a21, mut a22, mut a23) = (z, z, z, z);
        let (mut a30, mut a31, mut a32, mut a33) = (z, z, z, z);
        let mut i = 0;
        macro_rules! chunk {
            ($off:expr) => {{
                let x0 = v128_load(x.as_ptr().add(i + $off).cast());
                let x1 = v128_load(x.as_ptr().add(cols + i + $off).cast());
                let x2 = v128_load(x.as_ptr().add(2 * cols + i + $off).cast());
                let x3 = v128_load(x.as_ptr().add(3 * cols + i + $off).cast());
                macro_rules! row {
                    ($r:expr,$a:ident,$b:ident,$c:ident,$d:ident) => {{
                        let v = v128_load(w.as_ptr().add($r * cols + i + $off).cast());
                        $a = i32x4_add($a, i32x4_dot_i16x8(x0, v));
                        $b = i32x4_add($b, i32x4_dot_i16x8(x1, v));
                        $c = i32x4_add($c, i32x4_dot_i16x8(x2, v));
                        $d = i32x4_add($d, i32x4_dot_i16x8(x3, v));
                    }};
                }
                row!(0, a00, a10, a20, a30);
                row!(1, a01, a11, a21, a31);
                row!(2, a02, a12, a22, a32);
                row!(3, a03, a13, a23, a33);
            }};
        }
        while i + 32 <= cols {
            chunk!(0);
            chunk!(8);
            chunk!(16);
            chunk!(24);
            i += 32;
        }
        while i < cols {
            chunk!(0);
            i += 8;
        }
        fn sum(v: v128) -> i32 {
            i32x4_extract_lane::<0>(v)
                + i32x4_extract_lane::<1>(v)
                + i32x4_extract_lane::<2>(v)
                + i32x4_extract_lane::<3>(v)
        }
        return [
            sum(a00),
            sum(a01),
            sum(a02),
            sum(a03),
            sum(a10),
            sum(a11),
            sum(a12),
            sum(a13),
            sum(a20),
            sum(a21),
            sum(a22),
            sum(a23),
            sum(a30),
            sum(a31),
            sum(a32),
            sum(a33),
        ];
    }
    #[cfg(not(target_arch = "wasm32"))]
    std::array::from_fn(|i| {
        dot_i16(
            &x[(i / 4) * cols..(i / 4 + 1) * cols],
            &w[(i % 4) * cols..(i % 4 + 1) * cols],
        )
    })
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
unsafe fn exp4(x: core::arch::wasm32::v128) -> core::arch::wasm32::v128 {
    use core::arch::wasm32::*;
    let n = f32x4_nearest(f32x4_mul(x, f32x4_splat(std::f32::consts::LOG2_E)));
    let r = f32x4_sub(
        f32x4_sub(x, f32x4_mul(n, f32x4_splat(0.693145751953125))),
        f32x4_mul(n, f32x4_splat(1.4286068e-6)),
    );
    let p = f32x4_div(r, f32x4_splat(720.));
    let p = f32x4_add(f32x4_splat(1. / 120.), p);
    let p = f32x4_add(f32x4_splat(1. / 24.), f32x4_mul(r, p));
    let p = f32x4_add(f32x4_splat(1. / 6.), f32x4_mul(r, p));
    let p = f32x4_add(f32x4_splat(0.5), f32x4_mul(r, p));
    let p = f32x4_add(f32x4_splat(1.), f32x4_mul(r, p));
    let p = f32x4_add(f32x4_splat(1.), f32x4_mul(r, p));
    let pow = i32x4_shl(i32x4_add(i32x4_trunc_sat_f32x4(n), i32x4_splat(127)), 23);
    let y = v128_and(f32x4_mul(p, pow), f32x4_ge(x, f32x4_splat(-80.)));
    y
}

/// SIMD evaluation of the same polynomial, with the original denominator order.
#[cfg(target_arch = "wasm32")]
fn softmax_row(active: &mut [f32], max: f32) -> f32 {
    use core::arch::wasm32::*;
    unsafe {
        let mut denom = 0.;
        let n4 = active.len() / 4 * 4;
        for i in (0..n4).step_by(4) {
            let ptr = active.as_mut_ptr().add(i);
            let x = f32x4_sub(v128_load(ptr.cast()), f32x4_splat(max));
            let y = exp4(x);
            v128_store(ptr.cast(), y);
            denom += f32x4_extract_lane::<0>(y);
            denom += f32x4_extract_lane::<1>(y);
            denom += f32x4_extract_lane::<2>(y);
            denom += f32x4_extract_lane::<3>(y);
        }
        for score in &mut active[n4..] {
            *score = softmax_exp(*score - max);
            denom += *score;
        }
        denom
    }
}

/// Range-reduced degree-6 exp for nonpositive softmax arguments.
/// Relative error is checked against std exp; values below -80 are negligible.
#[inline(always)]
fn softmax_exp(x: f32) -> f32 {
    if x < -80. {
        return 0.;
    }
    let n = (x * std::f32::consts::LOG2_E).round_ties_even();
    let r = x - n * 0.693145751953125 - n * 1.4286068e-6;
    let p = 1. + r * (1. + r * (0.5 + r * (1. / 6. + r * (1. / 24. + r * (1. / 120. + r / 720.)))));
    p * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

#[inline(always)]
#[cfg(any(not(target_arch = "wasm32"), test))]
fn dot_i16(a: &[i16], b: &[i16]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "wasm32")]
    unsafe {
        use core::arch::wasm32::*;
        let mut sum = i32x4_splat(0);
        let mut i = 0;
        while i < a.len() {
            sum = i32x4_add(
                sum,
                i32x4_dot_i16x8(
                    v128_load(a.as_ptr().add(i).cast()),
                    v128_load(b.as_ptr().add(i).cast()),
                ),
            );
            i += 8;
        }
        return i32x4_extract_lane::<0>(sum)
            + i32x4_extract_lane::<1>(sum)
            + i32x4_extract_lane::<2>(sum)
            + i32x4_extract_lane::<3>(sum);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let mut sum = vdupq_n_s32(0);
        let mut i = 0;
        while i < a.len() {
            let x = vld1q_s16(a.as_ptr().add(i));
            let w = vld1q_s16(b.as_ptr().add(i));
            sum = vmlal_s16(sum, vget_low_s16(x), vget_low_s16(w));
            sum = vmlal_high_s16(sum, x, w);
            i += 8;
        }
        return vaddvq_s32(sum);
    }
    #[cfg(not(any(target_arch = "wasm32", target_arch = "aarch64")))]
    a.iter().zip(b).map(|(a, b)| *a as i32 * *b as i32).sum()
}

#[cfg(not(target_arch = "wasm32"))]
#[inline(always)]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "wasm32")]
    unsafe {
        use core::arch::wasm32::*;
        let mut sum = f32x4_splat(0.);
        for i in (0..D).step_by(4) {
            sum = f32x4_add(
                sum,
                f32x4_mul(
                    v128_load(a.as_ptr().add(i).cast()),
                    v128_load(b.as_ptr().add(i).cast()),
                ),
            );
        }
        return f32x4_extract_lane::<0>(sum)
            + f32x4_extract_lane::<1>(sum)
            + f32x4_extract_lane::<2>(sum)
            + f32x4_extract_lane::<3>(sum);
    }
    #[cfg(not(target_arch = "wasm32"))]
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

#[cfg(target_arch = "wasm32")]
#[inline(never)]
fn score_head(query: &[f32], k: &[f32], offset: usize, stride: usize, scores: &mut [f32]) -> f32 {
    use core::arch::wasm32::*;
    let mut max = f32::NEG_INFINITY;
    for (j, score) in scores.iter_mut().enumerate() {
        unsafe {
            let mut sum = f32x4_splat(0.);
            for d in (0..D).step_by(4) {
                sum = f32x4_add(
                    sum,
                    f32x4_mul(
                        v128_load(query.as_ptr().add(d).cast()),
                        v128_load(k.as_ptr().add(offset + j * stride + d).cast()),
                    ),
                );
            }
            *score = (f32x4_extract_lane::<0>(sum)
                + f32x4_extract_lane::<1>(sum)
                + f32x4_extract_lane::<2>(sum)
                + f32x4_extract_lane::<3>(sum))
                / (D as f32).sqrt();
        }
        max = max.max(*score);
    }
    max
}
#[cfg(target_arch = "wasm32")]
fn value_head(
    dest: &mut [f32],
    v: &[f32],
    offset: usize,
    stride: usize,
    scores: &[f32],
    inverse: f32,
) {
    use core::arch::wasm32::*;
    for d in (0..D).step_by(4) {
        unsafe {
            let mut sum = f32x4_splat(0.);
            for (j, p) in scores.iter().enumerate() {
                sum = f32x4_add(
                    sum,
                    f32x4_mul(
                        f32x4_splat(*p * inverse),
                        v128_load(v.as_ptr().add(offset + j * stride + d).cast()),
                    ),
                );
            }
            v128_store(dest.as_mut_ptr().add(d).cast(), sum);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[inline(always)]
fn weighted_add(out: &mut [f32], v: &[f32], p: f32) {
    #[cfg(target_arch = "wasm32")]
    unsafe {
        use core::arch::wasm32::*;
        let p = f32x4_splat(p);
        for i in (0..D).step_by(4) {
            let ptr = out.as_mut_ptr().add(i);
            v128_store(
                ptr.cast(),
                f32x4_add(
                    v128_load(ptr.cast()),
                    f32x4_mul(p, v128_load(v.as_ptr().add(i).cast())),
                ),
            );
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    for (a, b) in out.iter_mut().zip(v) {
        *a += p * b;
    }
}

pub fn key_range(layer: usize, start: usize, rows: usize, total: usize) -> (usize, usize) {
    if layer % 3 == 0 {
        (0, total)
    } else {
        (start.saturating_sub(64), (start + rows + 64).min(total))
    }
}

/// Residual F32 followed by per-token context scales and signed INT8.
/// This is exactly the dynamic quantization performed by the output linear.
pub fn attention_wire(
    qkv: &[f32],
    start: usize,
    rows: usize,
    head: usize,
    heads: usize,
    ks: usize,
    ke: usize,
) -> Vec<u8> {
    let mut wire = Vec::new();
    for t in start..start + rows {
        wire.extend(float_bytes(
            &qkv[t * 3 * H + head * D..t * 3 * H + (head + heads) * D],
        ));
    }
    for side in 1..=2 {
        for t in ks..ke {
            append_bf16(
                &mut wire,
                &qkv[t * 3 * H + side * H + head * D..t * 3 * H + side * H + (head + heads) * D],
            );
        }
    }
    wire
}

pub fn attention(
    layer: usize,
    start: usize,
    rows: usize,
    _head: usize,
    heads: usize,
    ks: usize,
    kn: usize,
    b: &[u8],
) -> Result<Vec<f32>, String> {
    if heads == 0
        || heads > 12
        || rows == 0
        || rows > MAX_TOKENS
        || kn == 0
        || kn > MAX_TOKENS
        || start < ks
        || start
            .checked_add(rows)
            .is_none_or(|end| end > ks.saturating_add(kn))
    {
        return Err("attention shape bounds".into());
    }
    let width = heads * D;
    let qb = rows * width * 4;
    let kb = kn * width * 2;
    if b.len() != qb + 2 * kb {
        return Err("attention payload length".into());
    }
    let q = floats(&b[..qb])?;
    let k = read_bf16(&b[qb..qb + kb]);
    let v = read_bf16(&b[qb + kb..]);
    if k.iter().chain(&v).any(|v| !v.is_finite()) {
        return Err("attention nonfinite".into());
    }
    let mut out = vec![0.; rows * width];
    let mut scores = vec![0.; kn];
    for t in 0..rows {
        for h in 0..heads {
            let (first, last) = if layer % 3 == 0 {
                (0, kn)
            } else {
                (
                    (start + t).saturating_sub(64).saturating_sub(ks),
                    (start + t + 65 - ks).min(kn),
                )
            };
            let query = &q[t * width + h * D..t * width + (h + 1) * D];
            #[cfg(not(target_arch = "wasm32"))]
            let mut max = f32::NEG_INFINITY;
            let active = &mut scores[first..last];
            #[cfg(target_arch = "wasm32")]
            let max = score_head(query, &k, first * width + h * D, width, active);
            #[cfg(not(target_arch = "wasm32"))]
            for (j, score) in active.iter_mut().enumerate() {
                // Payload shape and first/last/head bounds were checked above.
                let key = unsafe {
                    std::slice::from_raw_parts(k.as_ptr().add((first + j) * width + h * D), D)
                };
                let z = dot_f32(query, key) * (1. / (D as f32).sqrt());
                *score = z;
                max = max.max(z);
            }
            #[cfg(target_arch = "wasm32")]
            let denom = softmax_row(active, max);
            #[cfg(not(target_arch = "wasm32"))]
            let mut denom = 0.;
            #[cfg(not(target_arch = "wasm32"))]
            for score in active.iter_mut() {
                *score = softmax_exp(*score - max);
                denom += *score;
            }
            let dest = &mut out[t * width + h * D..t * width + (h + 1) * D];
            let inverse = 1. / denom;
            #[cfg(target_arch = "wasm32")]
            value_head(dest, &v, first * width + h * D, width, active, inverse);
            #[cfg(not(target_arch = "wasm32"))]
            for (j, score) in active.iter().enumerate() {
                let value = unsafe {
                    std::slice::from_raw_parts(v.as_ptr().add((first + j) * width + h * D), D)
                };
                weighted_add(dest, value, *score * inverse);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_invalid_frames_and_lengths() {
        assert!(validate_tokens(&[2, 1]).is_ok());
        for ids in [
            vec![],
            vec![2],
            vec![1, 2],
            vec![2, 0, 1],
            vec![2, VOCAB, 1],
            vec![2; 257],
        ] {
            assert!(validate_tokens(&ids).is_err());
        }
        let mut ids = vec![37; 256];
        ids[0] = 2;
        ids[255] = 1;
        assert!(validate_tokens(&ids).is_ok());
    }
    #[test]
    fn malformed_pack_returns_error_without_panicking() {
        assert!(Model::load(&[]).is_err());
        assert!(Model::load(&u64::MAX.to_le_bytes()).is_err());
        let d = Descriptor {
            shape: vec![H],
            offset: usize::MAX,
            bytes: H * 4,
            kind: "f32".into(),
        };
        assert!(tensor_range(0, &d, &[H], "f32").is_err());
        let d = Descriptor {
            shape: vec![H + 1],
            offset: 0,
            bytes: H * 4,
            kind: "f32".into(),
        };
        assert!(tensor_range(H * 4, &d, &[H], "f32").is_err());
        assert!(Model::load_from(8, |_, _| Err("read failure".into())).is_err());
        let too_large = (1_048_577_u64).to_le_bytes();
        assert!(
            Model::load_from(2_000_000, |offset, dest| {
                assert_eq!(offset, 0);
                dest.copy_from_slice(&too_large);
                Ok(())
            })
            .is_err()
        );
    }
    #[test]
    fn attention_includes_last_dimension_of_last_head() {
        let total = 130;
        let mut qkv = vec![0.; total * 3 * H];
        for t in 0..total {
            qkv[t * 3 * H + 2 * H + 5 * D + 63] = t as f32;
        }
        for (layer, expected) in [(0, 64.5), (1, 64.)] {
            let (ks, ke) = key_range(layer, 64, 1, total);
            let wire = attention_wire(&qkv, 64, 1, 5, 1, ks, ke);
            let got = attention(layer, 64, 1, 5, 1, ks, ke - ks, &wire).unwrap();
            assert!((got[63] - expected).abs() < 1e-4);
            assert!(got[..63].iter().all(|v| *v == 0.));
        }
    }
    #[test]
    fn linear_1152_accumulation_matches_i64_reference() {
        let m = Linear {
            #[cfg(feature = "strassen")]
            prepared: None,
            weights: vec![127; H * I],
            scales: vec![1.; H],
            rows: H,
            cols: I,
            bucketed: false,
        };
        let q: Vec<i16> = (0..4 * I)
            .map(|i| if i % 3 == 0 { -127 } else { 127 })
            .collect();
        let got = m.apply_quantized(&q, &[1.; 4]);
        assert_eq!(
            m.apply(&vec![1.; 4 * I]),
            m.apply_quantized(&vec![127; 4 * I], &[1. / 127.; 4])
        );
        for t in 0..4 {
            let expected = q[t * I..(t + 1) * I]
                .iter()
                .map(|v| *v as i64 * 127)
                .sum::<i64>() as f32;
            assert!(got[t * H..(t + 1) * H].iter().all(|v| *v == expected));
        }
    }

    #[test]
    fn padded_linear_preserves_every_real_row_at_bucket_boundaries() {
        for (rows, cols) in [(3 * H, H), (H, H), (2 * I, H), (H, I)] {
            let weights: Vec<i16> = (0..rows * cols)
                .map(|i| ((i * 73 + 11) % 255) as i16 - 127)
                .collect();
            let mut linear = Linear {
                #[cfg(feature = "strassen")]
                prepared: Some(strassen::Prepared::new_depth(
                    &weights,
                    rows,
                    cols,
                    if cols == I { 2 } else { 1 },
                )),
                weights,
                scales: vec![0.003; rows],
                rows,
                cols,
                bucketed: false,
            };
            for n in [65, 127, 129, 193, 255] {
                let q: Vec<i16> = (0..n * cols)
                    .map(|i| ((i * 37 + 19) % 255) as i16 - 127)
                    .collect();
                let scales = vec![0.01; n];
                linear.bucketed = false;
                let reference = linear.apply_quantized(&q, &scales);
                linear.bucketed = true;
                let padded = linear.apply_quantized(&q, &scales);
                assert_eq!(padded.len(), n * rows);
                assert!(
                    reference
                        .iter()
                        .zip(padded)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "n={n} rows={rows} cols={cols}"
                );
            }
        }
    }
}
