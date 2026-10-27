//! Experimental ModernBERT W8A8 inference. No persistent request state.
use serde::Deserialize;
use std::collections::BTreeMap;
pub mod carry;
#[cfg(feature = "profiling")]
pub mod profile;
macro_rules! phase {
    ($name:expr, $body:expr) => {{
        #[cfg(feature = "profiling")]
        let _scope = $crate::profile::scope($name);
        $body
    }};
}
include!(concat!(env!("OUT_DIR"), "/tree.rs"));

#[cfg(feature = "integer-attention")]
mod integer_attention;
#[cfg(feature = "strassen")]
mod strassen;

mod packed;
pub use packed::PackedQkv;

pub const H: usize = 384;
pub const D: usize = 32;
pub const I: usize = 1536;
pub const MAX_TOKENS: usize = 5000;
pub const DEFAULT_MAX_TOKENS: usize = 1365;
pub const SINGLE_QUERY_MAX_TOKENS: usize = 304;

/// Exact dependency prefix needed for CLS at the final local-attention layer.
pub fn hidden_rows(tokens: usize, next_layer: usize) -> usize {
    match next_layer {
        10 => tokens.min(129),
        11 => tokens.min(65),
        12 => 1,
        _ => tokens,
    }
}

pub fn query_plan(tokens: usize) -> Result<&'static [usize], String> {
    match tokens {
        2..=SINGLE_QUERY_MAX_TOKENS => Ok(&[12]),
        289..=512 => Ok(&[5, 7]),
        513..=768 => Ok(&[3, 3, 3, 3]),
        769..=1024 => Ok(&[2, 2, 2, 2, 4]),
        1025..=DEFAULT_MAX_TOKENS => Ok(&[1, 2, 1, 2, 1, 2, 3]),
        _ => Err("supported input: 2..=1365 tokens including CLS/SEP".into()),
    }
}

pub fn next_count(tokens: usize, first: usize) -> Result<usize, String> {
    let mut at = 0;
    for count in query_plan(tokens)? {
        if at == first {
            return Ok(*count);
        }
        at += count;
    }
    Err("continuation bounds".into())
}

pub struct EmbeddingProgress {
    pub next_layer: usize,
    pub tokens: usize,
    pub values: Vec<f32>,
}

#[derive(Deserialize)]
struct Descriptor {
    shape: Vec<usize>,
    offset: usize,
    bytes: usize,
    kind: String,
}

pub struct Linear {
    #[cfg(feature = "strassen")]
    prepared: Option<strassen::Prepared>,
    weights: Vec<i16>,
    scales: Vec<f32>,
    rows: usize,
    cols: usize,
}
pub struct Layer {
    qkv: Linear,
    out: Linear,
    wi: Linear,
    wo: Linear,
    attn_norm: Vec<f32>,
    mlp_norm: Vec<f32>,
}
pub struct Model {
    embeddings: Vec<f32>,
    embedding_norm: Vec<f32>,
    final_norm: Vec<f32>,
    layers: Vec<Layer>,
    rope: [Vec<[(f32, f32); 16]>; 2],
}

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

impl Model {
    pub fn start_embedding(&self, ids: &[u32]) -> Result<EmbeddingProgress, String> {
        let count = next_count(ids.len(), 0)?;
        if ids.first() != Some(&50281) || ids.last() != Some(&50282) || ids.contains(&50283) {
            return Err("expected CLS ... SEP without padding".into());
        }
        let values = self.fused_layers(0, count, ids.len(), self.embed(ids)?)?;
        Ok(EmbeddingProgress {
            next_layer: count,
            tokens: ids.len(),
            values,
        })
    }

    pub fn continue_embedding(
        &self,
        tokens: usize,
        first: usize,
        state: &[u8],
    ) -> Result<EmbeddingProgress, String> {
        let count = next_count(tokens, first)?;
        if first == 0 || first >= 12 {
            return Err("continuation bounds".into());
        }
        let values = self.fused_layers(
            first,
            count,
            tokens,
            carry::decode(state, hidden_rows(tokens, first) * H)?,
        )?;
        Ok(EmbeddingProgress {
            next_layer: first + count,
            tokens,
            values,
        })
    }
    pub fn load(pack: &[u8]) -> Result<Self, String> {
        let len =
            u64::from_le_bytes(pack.get(..8).ok_or("pack header")?.try_into().unwrap()) as usize;
        let header: BTreeMap<String, Descriptor> =
            serde_json::from_slice(pack.get(8..8 + len).ok_or("pack metadata")?)
                .map_err(|e| e.to_string())?;
        let data = &pack[8 + len..];
        let get = |name: &str| -> Result<(&Descriptor, &[u8]), String> {
            let d = header.get(name).ok_or_else(|| format!("missing {name}"))?;
            Ok((
                d,
                data.get(d.offset..d.offset + d.bytes)
                    .ok_or("tensor range")?,
            ))
        };
        let f = |name: &str| -> Result<Vec<f32>, String> {
            let (d, b) = get(name)?;
            if d.kind != "f32" {
                return Err("expected f32".into());
            }
            floats(b)
        };
        let linear = |name: &str| -> Result<Linear, String> {
            let (d, b) = get(name)?;
            if d.kind != "i8" || d.shape.len() != 2 {
                return Err("linear format".into());
            }
            let (rows, cols) = (d.shape[0], d.shape[1]);
            if ![H, 3 * H, I, 2 * I].contains(&rows) || ![H, I].contains(&cols) {
                return Err("unsupported linear shape".into());
            }
            if b.len() != rows * cols + rows * 4 {
                return Err("linear length".into());
            }
            let weights: Vec<i16> = b[..rows * cols].iter().map(|x| *x as i8 as i16).collect();
            if weights.iter().any(|v| !(-127..=127).contains(v)) {
                return Err("INT8 range".into());
            }
            #[cfg(feature = "strassen")]
            let prepared = (rows == 2 * I && cols == H
                || cfg!(feature = "strassen-wo") && rows == H && cols == I
                || cfg!(feature = "strassen-qkv") && rows == 3 * H && cols == H
                || cfg!(feature = "strassen-out") && rows == H && cols == H)
                .then(|| strassen::Prepared::new(&weights, rows, cols));
            Ok(Linear {
                #[cfg(feature = "strassen")]
                prepared,
                weights,
                scales: floats(&b[rows * cols..])?,
                rows,
                cols,
            })
        };
        let mut layers = Vec::new();
        for l in 0..12 {
            let p = format!("layers.{l}");
            layers.push(Layer {
                qkv: linear(&format!("{p}.attn.Wqkv.weight"))?,
                out: linear(&format!("{p}.attn.Wo.weight"))?,
                wi: linear(&format!("{p}.mlp.Wi.weight"))?,
                wo: linear(&format!("{p}.mlp.Wo.weight"))?,
                attn_norm: if l == 0 {
                    Vec::new()
                } else {
                    f(&format!("{p}.attn_norm.weight"))?
                },
                mlp_norm: f(&format!("{p}.mlp_norm.weight"))?,
            });
        }
        Ok(Self {
            embeddings: f("embeddings.tok_embeddings.weight")?,
            embedding_norm: f("embeddings.norm.weight")?,
            final_norm: f("final_norm.weight")?,
            layers,
            rope: [80000f32, 10000f32].map(|theta| {
                let frequencies: [f32; 16] =
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
        if ids.is_empty() || ids.len() > MAX_TOKENS || ids.iter().any(|id| *id >= 50368) {
            return Err("token bounds".into());
        }
        let mut x = Vec::with_capacity(ids.len() * H);
        for id in ids {
            x.extend_from_slice(&self.embeddings[*id as usize * H..(*id as usize + 1) * H]);
        }
        norm(&mut x, &self.embedding_norm);
        Ok(x)
    }

    pub fn qkv(&self, layer: usize, start: usize, x: &[f32]) -> Vec<f32> {
        #[cfg(feature = "profiling")]
        let _scope = profile::scope("qkv");
        let l = &self.layers[layer];
        let mut n = x.to_vec();
        if layer != 0 {
            norm(&mut n, &l.attn_norm);
        }
        let mut y = phase!("projection", l.qkv.apply(&n));
        #[cfg(feature = "profiling")]
        let _rope = profile::scope("rope");
        assert!(start <= MAX_TOKENS && x.len() / H <= MAX_TOKENS - start);
        let rope = &self.rope[if layer % 3 == 0 { 0 } else { 1 }];
        for (t, row) in y.chunks_exact_mut(3 * H).enumerate() {
            let rotations = rope[start + t];
            for side in 0..2 {
                for head in 0..12 {
                    for i in 0..16 {
                        let (s, c) = rotations[i];
                        let a = side * H + head * D + i;
                        let b = a + 16;
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
        #[cfg(feature = "profiling")]
        let gelu_scope = profile::scope("gelu");
        let gates = gelu_gates(&wide);
        #[cfg(feature = "profiling")]
        drop(gelu_scope);
        let delta = phase!("mlp_wo", l.wo.apply(&gates));
        for (a, b) in x.iter_mut().zip(delta) {
            *a += b;
        }
        x
    }

    pub fn final_embedding(&self, cls: &[f32]) -> Vec<f32> {
        let mut x = cls.to_vec();
        norm(&mut x, &self.final_norm);
        let scale = x.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        for v in &mut x {
            *v /= scale;
        }
        x
    }

    pub fn full(&self, ids: &[u32]) -> Result<Vec<f32>, String> {
        let mut x = self.embed(ids)?;
        for layer in 0..12 {
            let qkv = self.qkv(layer, 0, &x);
            let mut context = vec![0.; ids.len() * H];
            for start in (0..ids.len()).step_by(128) {
                let rows = 128.min(ids.len() - start);
                for head in (0..12).step_by(3) {
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
        Ok(self.final_embedding(&x[..H]))
    }

    /// Keep all intermediates inside one query. Only layer boundaries cross the wire.
    pub fn fused_layers(
        &self,
        first: usize,
        count: usize,
        total: usize,
        mut x: Vec<f32>,
    ) -> Result<Vec<f32>, String> {
        if count == 0
            || first + count > 12
            || total == 0
            || total > DEFAULT_MAX_TOKENS
            || x.len() != hidden_rows(total, first) * H
        {
            return Err("fused layer bounds".into());
        }
        for layer in first..first + count {
            #[cfg(feature = "profiling")]
            let _layer = profile::scope(&format!("layer{layer}"));
            let qkv = self.qkv(layer, 0, &x);
            let inputs = x.len() / H;
            let outputs = hidden_rows(total, layer + 1);
            let context = attention_inprocess(layer, inputs, outputs, &qkv)?;
            x = phase!("finish", self.finish(layer, &x[..outputs * H], &context));
        }
        if first + count == 12 {
            Ok(self.final_embedding(&x[..H]))
        } else {
            Ok(x)
        }
    }

    fn hidden_and_qkv(&self, layer: usize, start: usize, hidden: &[f32]) -> Vec<u8> {
        let qkv = self.qkv(layer, start, hidden);
        let mut out = Vec::with_capacity(hidden.len() * 12);
        append_float_bytes(&mut out, hidden);
        append_packed_qkv(&mut out, &qkv);
        out
    }

    /// Binary request: 8 little-endian u32 fields, then token IDs or float/BF16 payload.
    pub fn run(&self, request: &[u8]) -> Result<Vec<u8>, String> {
        if request.len() < 32 || request.len() > 2 * 1024 * 1024 - 128 {
            return Err("request size".into());
        }
        let a: Vec<usize> = request[..32]
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize)
            .collect();
        let (op, layer, start, total, head, heads, kstart, rows) =
            (a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
        if total == 0
            || total > MAX_TOKENS
            || layer >= 12
            || rows == 0
            || rows > total
            || start > total
            || rows > total - start
        {
            return Err("shape bounds".into());
        }
        let b = &request[32..];
        #[cfg(all(
            target_arch = "wasm32",
            feature = "profiling",
            feature = "integer-attention"
        ))]
        if op == 10 {
            if rows != total || start != 0 || head > 3 || !b.is_empty() {
                return Err("AV micro bounds".into());
            }
            return Ok(float_bytes(&integer_attention::weighted_test(total, head)));
        }
        #[cfg(all(
            target_arch = "wasm32",
            feature = "profiling",
            feature = "integer-attention"
        ))]
        if op == 14 {
            if rows != total || start != 0 || head > 3 || !b.is_empty() {
                return Err("QK micro bounds".into());
            }
            return Ok(integer_attention::qk_test(total, head)
                .to_le_bytes()
                .to_vec());
        }
        #[cfg(all(target_arch = "wasm32", feature = "profiling"))]
        if op == 18 {
            if rows != total || start != 0 || b.len() != rows * 4 {
                return Err("codec micro bounds".into());
            }
            let x: Vec<f32> = b
                .chunks_exact(4)
                .map(|v| f32::from_bits(u32::from_le_bytes(v.try_into().unwrap())))
                .collect();
            let mut out = Vec::with_capacity(rows * 2);
            append_bf16(&mut out, &x);
            return Ok(out);
        }
        if op == 9 {
            if total > 128 || rows != total || b.len() != rows * 2 * I * 4 {
                return Err("gelu micro bounds".into());
            }
            let wide = floats(b)?;
            return Ok(float_bytes(&phase!("gelu", gelu_gates(&wide))));
        }
        if op == 8 {
            if total > 256 || rows != total || start != 0 || head > 3 {
                return Err("micro bounds".into());
            }
            let l = &self.layers[layer];
            let linear = match head {
                0 => &l.qkv,
                1 => &l.out,
                2 => &l.wi,
                _ => &l.wo,
            };
            if b.len() != rows * linear.cols * 4 {
                return Err("micro length".into());
            }
            let y = phase!("micro", linear.apply(&floats(b)?));
            let hash = y.iter().fold(0xcbf29ce484222325u64, |h, v| {
                (h ^ v.to_bits() as u64).wrapping_mul(0x100000001b3)
            });
            return Ok(hash.to_le_bytes().to_vec());
        }
        let result = match op {
            6 | 7 => {
                if rows != total
                    || start != 0
                    || total > DEFAULT_MAX_TOKENS
                    || head == 0
                    || layer + head > 12
                {
                    return Err("fused request bounds".into());
                }
                let x = if op == 7 {
                    if layer != 0 || b.len() != total * 4 {
                        return Err("fused ids length".into());
                    }
                    let ids: Vec<u32> = b
                        .chunks_exact(4)
                        .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
                        .collect();
                    self.embed(&ids)?
                } else {
                    if b.len() != hidden_rows(total, layer) * H * 4 {
                        return Err("fused hidden length".into());
                    }
                    floats(b)?
                };
                self.fused_layers(layer, head, total, x)?
            }
            0 | 5 => {
                if b.len() != total * 4 || rows != total {
                    return Err("ids length".into());
                }
                let ids: Vec<u32> = b
                    .chunks_exact(4)
                    .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
                    .collect();
                if op == 0 {
                    self.embed(&ids)?
                } else {
                    self.full(&ids)?
                }
            }
            15 => {
                if layer != 0 || rows > 680 || b.len() != rows * 4 {
                    return Err("initial projection bounds".into());
                }
                let ids: Vec<u32> = b
                    .chunks_exact(4)
                    .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                let x = self.embed(&ids)?;
                return Ok(self.hidden_and_qkv(0, start, &x));
            }
            1 | 11 => {
                if op == 11 && rows > 1023 {
                    return Err("packed qkv rows".into());
                }
                if b.len() != rows * H * 4 {
                    return Err("qkv length".into());
                }
                let qkv = self.qkv(layer, start, &floats(b)?);
                if op == 11 {
                    return Ok(pack_qkv(&qkv));
                }
                qkv
            }
            2 => {
                if heads == 0 || heads > 12 || head >= 12 || head + heads > 12 {
                    return Err("heads".into());
                }
                let (expected_start, kend) = key_range(layer, start, rows, total);
                if kstart != expected_start {
                    return Err("key start".into());
                }
                attention(layer, start, rows, head, heads, kstart, kend - kstart, b)?
            }
            #[cfg(all(target_arch = "wasm32", feature = "integer-attention"))]
            13 => {
                if layer % 3 == 0 || heads == 0 || heads > 12 || head >= 12 || head + heads > 12 {
                    return Err("packed local attention heads".into());
                }
                let (ks, ke) = key_range(layer, start, rows, total);
                if kstart != ks {
                    return Err("key start".into());
                }
                let kn = ke - ks;
                let width = heads * D;
                let qb = rows * width * 2;
                let sb = rows * heads * 4;
                let kb = kn * width * 2;
                if b.len() != qb + sb + 2 * kb {
                    return Err("packed attention length".into());
                }
                let qq: Vec<i16> = b[..qb]
                    .chunks_exact(2)
                    .map(|v| i16::from_le_bytes(v.try_into().unwrap()))
                    .collect();
                let qs = floats(&b[qb..qb + sb])?;
                if qq.iter().any(|v| !(-8191..=8191).contains(v)) || qs.iter().any(|v| *v < 1e-12) {
                    return Err("packed attention quantization bounds".into());
                }
                let k = read_bf16(&b[qb + sb..qb + sb + kb]);
                let v = read_bf16(&b[qb + sb + kb..]);
                if k.iter().chain(&v).any(|v| !v.is_finite()) {
                    return Err("packed attention nonfinite".into());
                }
                integer_attention::run_quantized_q(
                    layer, start, rows, heads, ks, kn, &qq, &qs, &k, &v,
                )?
            }
            3 => {
                if b.len() != rows * H * 8 {
                    return Err("finish length".into());
                }
                let v = floats(b)?;
                self.finish(layer, &v[..rows * H], &v[rows * H..])
            }
            12 | 16 => {
                if op == 16 && (rows > 680 || (layer == 11 && (start != 0 || rows != 1))) {
                    return Err("chained finish bounds".into());
                }
                if rows > 1088 || b.len() != rows * (H * 5 + 4) {
                    return Err("packed finish length".into());
                }
                let residual = floats(&b[..rows * H * 4])?;
                let end_scales = rows * (H * 4 + 4);
                let scales = floats(&b[rows * H * 4..end_scales])?;
                if scales.iter().any(|s| *s < 1e-12) || b[end_scales..].contains(&128) {
                    return Err("packed finish quantization bounds".into());
                }
                let quant: Vec<i16> = b[end_scales..].iter().map(|q| *q as i8 as i16).collect();
                let projection = self.layers[layer].out.apply_quantized(&quant, &scales);
                let x = self.finish_projected(layer, &residual, projection);
                if op == 16 {
                    return if layer == 11 {
                        Ok(float_bytes(&self.final_embedding(&x)))
                    } else {
                        Ok(self.hidden_and_qkv(layer + 1, start, &x))
                    };
                }
                x
            }
            4 => {
                if b.len() != H * 4 {
                    return Err("cls length".into());
                }
                self.final_embedding(&floats(b)?)
            }
            _ => return Err("unknown op".into()),
        };
        Ok(float_bytes(&result))
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
        #[cfg(feature = "profiling")]
        let _linear = profile::scope(&format!("linear_{}x{}x{}", n, self.cols, self.rows));
        #[cfg(feature = "profiling")]
        let quant_scope = profile::scope("quantize");
        // All entries are written by the disjoint token/output tiles below.
        // Expose the initialized length only after every tile has finished.
        let mut quant = Vec::<i16>::with_capacity(x.len());
        let mut scales = Vec::with_capacity(n);
        #[cfg(feature = "profiling")]
        let mut quant_costs = [0u64; 2];
        for (t, row) in x.chunks_exact(self.cols).enumerate() {
            #[cfg(feature = "profiling")]
            let began = profile::now();
            let scale = (max_abs(row) / 127.).max(1e-12);
            #[cfg(feature = "profiling")]
            let middle = {
                let now = profile::now();
                quant_costs[0] += now - began;
                now
            };
            scales.push(scale);
            quantize_into(
                row,
                scale,
                &mut quant.spare_capacity_mut()[t * self.cols..(t + 1) * self.cols],
            );
            #[cfg(feature = "profiling")]
            {
                quant_costs[1] += profile::now() - middle;
            }
        }
        // Every row was initialized through MaybeUninit, without a zero pass.
        unsafe {
            quant.set_len(x.len());
        }
        #[cfg(feature = "profiling")]
        for (name, cost) in ["max_abs", "round_and_cast"].into_iter().zip(quant_costs) {
            profile::add(name, cost, n as u64);
        }
        #[cfg(feature = "profiling")]
        drop(quant_scope);
        #[cfg(feature = "profiling")]
        let _gemm = profile::scope("gemm_and_scale");
        self.apply_quantized(&quant, &scales)
    }
    fn apply_quantized(&self, quant: &[i16], scales: &[f32]) -> Vec<f32> {
        let n = scales.len();
        assert_eq!(quant.len(), n * self.cols);
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
                        tree_1536_32(xp, wp, yp, self.rows, sx, sw);
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
                        tree_1536_8(xp, wp, yp, self.rows, sx, sw);
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
                            (I, 4) => tree_1536_4(xp, wp, yp, self.rows, sx, sw),
                            (H, 1) => tree_384_1(xp, wp, yp, self.rows, sx, sw),
                            (I, 1) => tree_1536_1(xp, wp, yp, self.rows, sx, sw),
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
    unsafe {
        let q0 = v128_load(query.as_ptr().add(0).cast());
        let q1 = v128_load(query.as_ptr().add(4).cast());
        let q2 = v128_load(query.as_ptr().add(8).cast());
        let q3 = v128_load(query.as_ptr().add(12).cast());
        let q4 = v128_load(query.as_ptr().add(16).cast());
        let q5 = v128_load(query.as_ptr().add(20).cast());
        let q6 = v128_load(query.as_ptr().add(24).cast());
        let q7 = v128_load(query.as_ptr().add(28).cast());
        let mut max = f32::NEG_INFINITY;
        for (j, score) in scores.iter_mut().enumerate() {
            let ptr = k.as_ptr().add(offset + j * stride);
            let mut sum = f32x4_splat(0.);
            sum = f32x4_add(sum, f32x4_mul(q0, v128_load(ptr.add(0).cast())));
            sum = f32x4_add(sum, f32x4_mul(q1, v128_load(ptr.add(4).cast())));
            sum = f32x4_add(sum, f32x4_mul(q2, v128_load(ptr.add(8).cast())));
            sum = f32x4_add(sum, f32x4_mul(q3, v128_load(ptr.add(12).cast())));
            sum = f32x4_add(sum, f32x4_mul(q4, v128_load(ptr.add(16).cast())));
            sum = f32x4_add(sum, f32x4_mul(q5, v128_load(ptr.add(20).cast())));
            sum = f32x4_add(sum, f32x4_mul(q6, v128_load(ptr.add(24).cast())));
            sum = f32x4_add(sum, f32x4_mul(q7, v128_load(ptr.add(28).cast())));
            let z = (f32x4_extract_lane::<0>(sum)
                + f32x4_extract_lane::<1>(sum)
                + f32x4_extract_lane::<2>(sum)
                + f32x4_extract_lane::<3>(sum))
                * (1. / (D as f32).sqrt());
            *score = z;
            max = max.max(z);
        }
        max
    }
}
#[cfg(target_arch = "wasm32")]
#[inline(never)]
fn value_head(
    dest: &mut [f32],
    v: &[f32],
    offset: usize,
    stride: usize,
    scores: &[f32],
    inverse: f32,
) {
    use core::arch::wasm32::*;
    unsafe {
        let mut a0 = f32x4_splat(0.);
        let mut a1 = f32x4_splat(0.);
        let mut a2 = f32x4_splat(0.);
        let mut a3 = f32x4_splat(0.);
        let mut a4 = f32x4_splat(0.);
        let mut a5 = f32x4_splat(0.);
        let mut a6 = f32x4_splat(0.);
        let mut a7 = f32x4_splat(0.);
        for (j, score) in scores.iter().enumerate() {
            let p = f32x4_splat(*score * inverse);
            let ptr = v.as_ptr().add(offset + j * stride);
            a0 = f32x4_add(a0, f32x4_mul(p, v128_load(ptr.add(0).cast())));
            a1 = f32x4_add(a1, f32x4_mul(p, v128_load(ptr.add(4).cast())));
            a2 = f32x4_add(a2, f32x4_mul(p, v128_load(ptr.add(8).cast())));
            a3 = f32x4_add(a3, f32x4_mul(p, v128_load(ptr.add(12).cast())));
            a4 = f32x4_add(a4, f32x4_mul(p, v128_load(ptr.add(16).cast())));
            a5 = f32x4_add(a5, f32x4_mul(p, v128_load(ptr.add(20).cast())));
            a6 = f32x4_add(a6, f32x4_mul(p, v128_load(ptr.add(24).cast())));
            a7 = f32x4_add(a7, f32x4_mul(p, v128_load(ptr.add(28).cast())));
        }
        v128_store(dest.as_mut_ptr().add(0).cast(), a0);
        v128_store(dest.as_mut_ptr().add(4).cast(), a1);
        v128_store(dest.as_mut_ptr().add(8).cast(), a2);
        v128_store(dest.as_mut_ptr().add(12).cast(), a3);
        v128_store(dest.as_mut_ptr().add(16).cast(), a4);
        v128_store(dest.as_mut_ptr().add(20).cast(), a5);
        v128_store(dest.as_mut_ptr().add(24).cast(), a6);
        v128_store(dest.as_mut_ptr().add(28).cast(), a7);
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

fn attention_inprocess(
    layer: usize,
    inputs: usize,
    outputs: usize,
    qkv: &[f32],
) -> Result<Vec<f32>, String> {
    #[cfg(all(target_arch = "wasm32", feature = "integer-attention"))]
    if inputs <= MAX_TOKENS {
        let (q, k, v) = phase!("prepare_attention", split_qkv(qkv, inputs, outputs));
        if q.iter().chain(&k).chain(&v).any(|x| !x.is_finite()) {
            return Err("attention nonfinite".into());
        }
        return phase!(
            "attention",
            integer_attention::run(layer, 0, outputs, 12, 0, inputs, &q, &k, &v)
        );
    }
    let wire = phase!("wire", attention_wire(qkv, 0, outputs, 0, 12, 0, inputs));
    phase!(
        "attention",
        attention(layer, 0, outputs, 0, 12, 0, inputs, &wire)
    )
}
#[cfg(all(target_arch = "wasm32", feature = "integer-attention"))]
fn split_qkv(qkv: &[f32], inputs: usize, outputs: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    assert!(outputs <= inputs && qkv.len() == inputs * 3 * H);
    let mut q = Vec::<f32>::with_capacity(outputs * H);
    let mut k = Vec::<f32>::with_capacity(inputs * H);
    let mut v = Vec::<f32>::with_capacity(inputs * H);
    unsafe {
        use core::arch::wasm32::*;
        for t in 0..inputs {
            let src = qkv.as_ptr().add(t * 3 * H);
            if t < outputs {
                std::ptr::copy_nonoverlapping(src, q.as_mut_ptr().add(t * H), H);
            }
            for i in (0..H).step_by(4) {
                let kb = v128_load(src.add(H + i).cast());
                let vb = v128_load(src.add(2 * H + i).cast());
                let kb = i32x4_add(
                    kb,
                    i32x4_add(
                        i32x4_splat(0x7fff),
                        v128_and(u32x4_shr(kb, 16), i32x4_splat(1)),
                    ),
                );
                let vb = i32x4_add(
                    vb,
                    i32x4_add(
                        i32x4_splat(0x7fff),
                        v128_and(u32x4_shr(vb, 16), i32x4_splat(1)),
                    ),
                );
                v128_store(
                    k.as_mut_ptr().add(t * H + i).cast(),
                    v128_and(kb, i32x4_splat(0xffff0000u32 as i32)),
                );
                v128_store(
                    v.as_mut_ptr().add(t * H + i).cast(),
                    v128_and(vb, i32x4_splat(0xffff0000u32 as i32)),
                );
            }
        }
        q.set_len(outputs * H);
        k.set_len(inputs * H);
        v.set_len(inputs * H);
    }
    (q, k, v)
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
pub fn pack_finish(residual: &[f32], context: &[f32]) -> Vec<u8> {
    assert_eq!(residual.len(), context.len());
    let mut out = Vec::with_capacity(context.len() / H * (H * 5 + 4));
    append_float_bytes(&mut out, residual);
    append_context(&mut out, context);
    out
}
pub fn pack_finish_bytes(residual: &[u8], context: &[f32]) -> Vec<u8> {
    assert_eq!(residual.len(), context.len() * 4);
    let mut out = Vec::with_capacity(context.len() / H * (H * 5 + 4));
    out.extend_from_slice(residual);
    append_context(&mut out, context);
    out
}
fn append_context(out: &mut Vec<u8>, context: &[f32]) {
    assert_eq!(context.len() % H, 0);
    let scales: Vec<_> = context
        .chunks_exact(H)
        .map(|row| (max_abs(row) / 127.).max(1e-12))
        .collect();
    append_float_bytes(out, &scales);
    for (row, scale) in context.chunks_exact(H).zip(scales) {
        out.extend(
            row.iter()
                .map(|x| (x / scale).round_ties_even().clamp(-127., 127.) as i8 as u8),
        );
    }
}

/// Each row is Q:384 F32, K:384 BF16, V:384 BF16. K/V already
/// use BF16 on the attention wire; packing here introduces no new rounding.
fn append_packed_qkv(out: &mut Vec<u8>, qkv: &[f32]) {
    assert_eq!(qkv.len() % (3 * H), 0);
    for row in qkv.chunks_exact(3 * H) {
        append_float_bytes(out, &row[..H]);
        append_bf16(out, &row[H..]);
    }
}
pub fn pack_qkv(qkv: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(qkv.len() / (3 * H) * H * 8);
    append_packed_qkv(&mut out, qkv);
    out
}
pub fn unpack_qkv(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if bytes.is_empty() || bytes.len() % (H * 8) != 0 || bytes.len() / (H * 8) > 1023 {
        return Err("packed qkv length".into());
    }
    let mut out = Vec::with_capacity(bytes.len() / (H * 8) * 3 * H);
    for row in bytes.chunks_exact(H * 8) {
        out.extend(floats(&row[..H * 4])?);
        let kv = read_bf16(&row[H * 4..]);
        if kv.iter().any(|v| !v.is_finite()) {
            return Err("packed qkv nonfinite".into());
        }
        out.extend(kv);
    }
    Ok(out)
}

/// Same local attention with its existing per-head Q quantization moved to
/// the client. Payload: Q i16, Q scales F32, K BF16, V BF16.
pub fn packed_attention_wire(
    qkv: &[f32],
    start: usize,
    rows: usize,
    head: usize,
    heads: usize,
    ks: usize,
    ke: usize,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows * heads * (D * 2 + 4) + (ke - ks) * heads * D * 4);
    packed::append_q14(
        &mut out,
        (start..start + rows).flat_map(|t| {
            qkv[t * 3 * H + head * D..t * 3 * H + (head + heads) * D]
                .chunks_exact(D)
                .map(packed::finite_query)
        }),
    );
    for side in 1..=2 {
        for t in ks..ke {
            append_bf16(
                &mut out,
                &qkv[t * 3 * H + side * H + head * D..t * 3 * H + side * H + (head + heads) * D],
            );
        }
    }
    out
}

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
    #[cfg(all(target_arch = "wasm32", feature = "integer-attention"))]
    if kn <= MAX_TOKENS {
        return integer_attention::run(layer, start, rows, heads, ks, kn, &q, &k, &v);
    }
    let mut out = vec![0.; rows * width];
    let mut scores = vec![0.; kn];
    #[cfg(feature = "profiling")]
    let mut costs = [0u64; 3];
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
            #[cfg(feature = "profiling")]
            let qk_start = profile::now();
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
            #[cfg(feature = "profiling")]
            let sm_start = {
                let now = profile::now();
                costs[0] += now - qk_start;
                now
            };
            #[cfg(target_arch = "wasm32")]
            let denom = softmax_row(active, max);
            #[cfg(not(target_arch = "wasm32"))]
            let mut denom = 0.;
            #[cfg(not(target_arch = "wasm32"))]
            for score in active.iter_mut() {
                *score = softmax_exp(*score - max);
                denom += *score;
            }
            #[cfg(feature = "profiling")]
            let av_start = {
                let now = profile::now();
                costs[1] += now - sm_start;
                now
            };
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
            #[cfg(feature = "profiling")]
            {
                costs[2] += profile::now() - av_start;
            }
        }
    }
    #[cfg(feature = "profiling")]
    for (name, cost) in ["qk_scores", "softmax", "av_weighted_sum"]
        .into_iter()
        .zip(costs)
    {
        profile::add(name, cost, (rows * heads) as u64);
    }
    Ok(out)
}

pub fn request(fields: [usize; 8], body: &[u8]) -> Vec<u8> {
    let mut b: Vec<u8> = fields
        .into_iter()
        .flat_map(|v| (v as u32).to_le_bytes())
        .collect();
    b.extend(body);
    b
}
pub fn decode_floats(b: &[u8]) -> Result<Vec<f32>, String> {
    floats(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "approx-gelu")]
    #[test]
    fn append_codecs_preserve_prefix_and_ieee_rounding() {
        let bits = [
            0u32, 0x80000000, 0x3f808000, 0x3f818000, 0x7f7fffff, 0xff7fffff, 0x7f800000,
            0xff800000, 0x7fc01234, 1, 0xffffabcd,
        ];
        for n in 1..=bits.len() {
            let x: Vec<_> = bits[..n].iter().map(|v| f32::from_bits(*v)).collect();
            let mut actual = vec![7, 9, 11];
            append_bf16(&mut actual, &x);
            let mut expected = vec![7, 9, 11];
            expected.extend(x.iter().flat_map(|v| bf16(*v).to_le_bytes()));
            assert_eq!(actual, expected);
        }
        let residual = vec![0.25; 3 * H];
        let context: Vec<_> = (0..3 * H).map(|i| (i % 17) as f32 - 8.).collect();
        assert_eq!(
            pack_finish(&residual, &context),
            pack_finish_bytes(&float_bytes(&residual), &context)
        );
    }

    #[test]
    fn packed_finish_matches_dynamic_quantization() {
        let n = 3;
        let context: Vec<f32> = (0..n * H)
            .map(|i| {
                if i < H {
                    0.
                } else {
                    ((i * 71 % 1021) as f32 - 510.) / 997.
                }
            })
            .collect();
        let residual = vec![0.; n * H];
        let packet = pack_finish(&residual, &context);
        assert_eq!(packet.len(), n * (H * 5 + 4));
        let scales = floats(&packet[n * H * 4..n * (H * 4 + 4)]).unwrap();
        let quant: Vec<_> = packet[n * (H * 4 + 4)..]
            .iter()
            .map(|v| *v as i8 as i16)
            .collect();
        let linear = Linear {
            #[cfg(feature = "strassen")]
            prepared: None,
            rows: H,
            cols: H,
            weights: (0..H * H).map(|i| (i * 17 % 255) as i16 - 127).collect(),
            scales: vec![0.003; H],
        };
        assert_eq!(
            float_bytes(&linear.apply(&context)),
            float_bytes(&linear.apply_quantized(&quant, &scales))
        );
    }

    #[test]
    fn packed_qkv_preserves_attention_wire() {
        let qkv: Vec<f32> = (0..7 * 3 * H)
            .map(|i| ((i * 71 % 1021) as f32 - 510.) / 997.)
            .collect();
        let packed = pack_qkv(&qkv);
        assert_eq!(packed.len(), 7 * H * 8);
        let decoded = unpack_qkv(&packed).unwrap();
        for heads in [1, 3, 12] {
            for head in (0..12).step_by(heads) {
                assert_eq!(
                    attention_wire(&qkv, 1, 3, head, heads, 0, 7),
                    attention_wire(&decoded, 1, 3, head, heads, 0, 7)
                );
            }
        }
        assert!(unpack_qkv(&packed[..packed.len() - 1]).is_err());
        assert!(unpack_qkv(&[]).is_err());
    }

    #[test]
    fn approximate_gelu_error_is_bounded_on_dense_grid() {
        let mut max_error = 0f32;
        for i in -120000..=120000 {
            let v = i as f32 / 10000.;
            let x = v * std::f32::consts::FRAC_1_SQRT_2;
            let t = 1. / (1. + 0.3275911 * x.abs());
            let p = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
                + 0.254829592)
                * t;
            let erf = (1. - p * softmax_exp(-x * x)).copysign(x);
            let approx = 0.5 * v * (1. + erf);
            let reference = 0.5 * v * (1. + libm::erff(x));
            max_error = max_error.max((approx - reference).abs());
        }
        assert!(max_error < 3e-6, "GELU max abs error {max_error}");
    }

    #[test]
    fn measured_length_profiles_cover_twelve_layers() {
        for tokens in [
            2, 64, 65, 128, 129, 160, 161, 256, 257, 288, 289, 511, 512, 513, 768, 1024, 1280, 1365,
        ] {
            let plan = query_plan(tokens).unwrap();
            assert_eq!(plan.iter().sum::<usize>(), 12);
            let mut first = 0;
            for count in plan {
                assert_eq!(next_count(tokens, first).unwrap(), *count);
                first += count;
            }
        }
        for n in [0, 1, DEFAULT_MAX_TOKENS + 1, 5000] {
            assert!(query_plan(n).is_err());
        }
        assert_eq!(query_plan(160).unwrap(), &[12]);
        assert_eq!(query_plan(161).unwrap(), &[12]);
        assert_eq!(query_plan(256).unwrap(), &[12]);
        assert_eq!(query_plan(288).unwrap(), &[12]);
        assert_eq!(query_plan(SINGLE_QUERY_MAX_TOKENS).unwrap(), &[12]);
        assert_eq!(query_plan(SINGLE_QUERY_MAX_TOKENS + 1).unwrap(), &[5, 7]);
        assert!(next_count(512, 1).is_err());
        assert_eq!(hidden_rows(512, 10), 129);
        assert_eq!(hidden_rows(512, 11), 65);
        assert_eq!(hidden_rows(512, 12), 1);
    }

    #[test]
    fn tiled_int8_dot_matches_scalar_for_every_output() {
        for cols in [384, 1536] {
            let x: Vec<i16> = (0..4 * cols)
                .map(|i| ((i * 71) % 255) as i16 - 127)
                .collect();
            let w: Vec<i16> = (0..4 * cols)
                .map(|i| ((i * 19 + 37) % 255) as i16 - 127)
                .collect();
            let got = dot_tile(&x, &w, cols);
            for t in 0..4 {
                for r in 0..4 {
                    let expected: i32 = x[t * cols..(t + 1) * cols]
                        .iter()
                        .zip(&w[r * cols..(r + 1) * cols])
                        .map(|(a, b)| *a as i32 * *b as i32)
                        .sum();
                    assert_eq!(got[t * 4 + r], expected);
                }
            }
        }
    }

    #[test]
    fn int8_dot_accumulates_without_i16_overflow() {
        for length in [384, 1536] {
            let a = vec![127i16; length];
            let b: Vec<i16> = (0..length)
                .map(|i| if i % 3 == 0 { -127 } else { 127 })
                .collect();
            let expected: i32 = a.iter().zip(&b).map(|(a, b)| *a as i32 * *b as i32).sum();
            assert_eq!(dot_i16(&a, &b), expected);
            let weights: Vec<i16> = (0..4)
                .flat_map(|r| b.iter().map(move |v| if r % 2 == 0 { *v } else { -*v }))
                .collect();
            assert_eq!(
                dot_four(&a, &weights),
                [expected, -expected, expected, -expected]
            );
        }
    }

    #[test]
    fn softmax_polynomial_has_bounded_relative_error() {
        for i in 0..=80000 {
            let x = -(i as f32) / 1000.;
            let expected = x.exp();
            assert!((softmax_exp(x) / expected - 1.).abs() < 2e-6, "x={x}");
        }
        assert_eq!(softmax_exp(-81.), 0.);
    }

    #[test]
    fn bf16_transport_rounds_ties_to_even() {
        assert_eq!(bf16(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(bf16(f32::from_bits(0x3f81_8000)), 0x3f82);
        assert_eq!(bf16(-0.0), 0x8000);
    }

    #[test]
    fn attention_uses_global_context_and_inclusive_local_window() {
        // Zero Q/K gives uniform probabilities. Position 129 is outside
        // the inclusive +/-64 window of position 64, but inside global attention.
        let total = 130;
        let start = 64;
        let mut qkv = vec![0.; total * 3 * H];
        for t in 0..total {
            for d in 0..D {
                qkv[t * 3 * H + 2 * H + d] = t as f32;
            }
        }
        for (layer, expected) in [(0, 64.5), (1, 64.)] {
            let (ks, ke) = key_range(layer, start, 1, total);
            let wire = attention_wire(&qkv, start, 1, 0, 1, ks, ke);
            let output = attention(layer, start, 1, 0, 1, ks, ke - ks, &wire).unwrap();
            assert!(output.iter().all(|v| (*v - expected).abs() < 1e-4));
            assert!(
                attention(layer, start, 1, 0, 1, ks, ke - ks, &wire[..wire.len() - 1]).is_err()
            );
        }
    }
}
