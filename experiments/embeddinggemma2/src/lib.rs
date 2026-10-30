//! Disposable text-only fixture probe. No public production API or durable model storage.
use serde::Deserialize;
use std::collections::BTreeMap;
pub mod harrier;

#[derive(Deserialize)]
struct Descriptor {
    shape: Vec<usize>,
    offset: usize,
    bytes: usize,
    kind: String,
}
struct Linear {
    rows: usize,
    cols: usize,
    w: Vec<i16>,
    s: Vec<f32>,
}
pub struct Model {
    linear: BTreeMap<String, Linear>,
    floats: BTreeMap<String, Vec<f32>>,
}
fn floats(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect()
}

impl Model {
    pub fn load(pack: &[u8]) -> Self {
        let n = u64::from_le_bytes(pack[..8].try_into().unwrap()) as usize;
        let meta: BTreeMap<String, Descriptor> = serde_json::from_slice(&pack[8..8 + n]).unwrap();
        let payload = &pack[8 + n..];
        let mut linear = BTreeMap::new();
        let mut fs = BTreeMap::new();
        for (name, d) in meta {
            let b = &payload[d.offset..d.offset + d.bytes];
            if d.kind == "i8" {
                let (rows, cols) = (d.shape[0], d.shape[1]);
                assert_eq!(cols % 8, 0);
                assert_eq!(rows % 4, 0);
                assert_eq!(b.len(), rows * cols + rows * 4);
                linear.insert(
                    name.trim_end_matches(".weight").into(),
                    Linear {
                        rows,
                        cols,
                        w: b[..rows * cols].iter().map(|v| *v as i8 as i16).collect(),
                        s: floats(&b[rows * cols..]),
                    },
                );
            } else {
                fs.insert(name, floats(b));
            }
        }
        Self { linear, floats: fs }
    }
    fn lin(&self, name: &str, x: &[f32]) -> Vec<f32> {
        self.linear[name].run(x)
    }
    fn norm(&self, name: &str, x: &[f32], width: usize) -> Vec<f32> {
        norm(x, width, Some(&self.floats[&format!("{name}.weight")]))
    }
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        self.forward_trace(input, None)
    }
    pub fn forward_trace(&self, input: &[f32], mut trace: Option<&mut Vec<Vec<f32>>>) -> Vec<f32> {
        assert_eq!(input.len() % 512, 0);
        let n = input.len() / 512;
        assert!((2..=256).contains(&n));
        let mut ple = self.lin("ple.per_layer_model_projection", input);
        for x in &mut ple {
            *x *= 512f32.powf(-0.5);
        }
        ple = self.norm("ple.per_layer_projection_norm", &ple, 512);
        let mut x = input.to_vec();
        for l in 0..24 {
            self.layer(&mut x, l, &ple, 24 * 512, l * 512);
            if let Some(trace) = trace.as_deref_mut() {
                trace.push(x.clone());
            }
        }
        self.finish(&x)
    }
    fn finish(&self, x: &[f32]) -> Vec<f32> {
        let n = x.len() / 512;
        let x = self.norm("norm", &x, 512);
        // The W8A8 reference quantizes the projection per token, so preserve this order.
        let x = self.lin("embedding_projection", &x);
        let mut pooled = vec![0f32; 768];
        for row in x.chunks_exact(768) {
            for (p, z) in pooled.iter_mut().zip(row) {
                *p += z / n as f32;
            }
        }
        let s = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
        for p in &mut pooled {
            *p /= s;
        }
        pooled
    }
    /// Stateless continuation: the caller carries full hidden states, never pooled chunks.
    pub fn step(
        &self,
        input: &[f32],
        hidden: &[f32],
        start: usize,
        count: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        assert!(input.len() % 512 == 0 && (1024..=256 * 512).contains(&input.len()));
        assert!((1..=2).contains(&count) && start < 24 && start + count <= 24);
        assert!(hidden.len() == input.len());
        let mut x = hidden.to_vec();
        let projection = &self.linear["ple.per_layer_model_projection"];
        for l in start..start + count {
            // Quantization scales are per input token and output row. Slicing the
            // output rows is identical to computing the full PLE matrix then slicing.
            let mut ple = projection.run_rows(input, l * 512, 512);
            for v in &mut ple {
                *v *= 512f32.powf(-0.5);
            }
            ple = self.norm("ple.per_layer_projection_norm", &ple, 512);
            self.layer(&mut x, l, &ple, 512, 0);
        }
        if start + count == 24 {
            (Vec::new(), self.finish(&x))
        } else {
            (x, Vec::new())
        }
    }
    fn layer(&self, x: &mut Vec<f32>, l: usize, ple: &[f32], ple_stride: usize, ple_offset: usize) {
        let n = x.len() / 512;
        let p = format!("layers.{l}");
        let h = self.norm(&format!("{p}.input_layernorm"), x, 512);
        let a = format!("{p}.self_attn");
        let d = if l % 6 == 5 { 512 } else { 256 };
        let kv = if l % 6 == 5 { 1 } else { 2 };
        let mut q = self.norm(
            &format!("{a}.q_norm"),
            &self.lin(&format!("{a}.q_proj"), &h),
            d,
        );
        let mut k = self.norm(
            &format!("{a}.k_norm"),
            &self.lin(&format!("{a}.k_proj"), &h),
            d,
        );
        let mut v = norm(&self.lin(&format!("{a}.v_proj"), &h), d, None);
        let base = if l % 6 == 5 { 1_000_000. } else { 10_000. };
        rope(&mut q, n, 4, d, base);
        rope(&mut k, n, kv, d, base);
        for z in k.iter_mut().chain(v.iter_mut()) {
            *z = bf16(*z);
        }
        let context = attention(&q, &k, &v, n, d, kv);
        let out = self.lin(&format!("{a}.o_proj"), &context);
        add(
            x,
            &self.norm(&format!("{p}.post_attention_layernorm"), &out, 512),
        );
        let h = self.norm(&format!("{p}.pre_feedforward_layernorm"), x, 512);
        let mut gate = self.lin(&format!("{p}.mlp.gate_proj"), &h);
        let up = self.lin(&format!("{p}.mlp.up_proj"), &h);
        for (g, u) in gate.iter_mut().zip(up) {
            *g = gelu(*g) * u;
        }
        let out = self.lin(&format!("{p}.mlp.down_proj"), &gate);
        add(
            x,
            &self.norm(&format!("{p}.post_feedforward_layernorm"), &out, 512),
        );
        let mut gate = self.lin(&format!("{p}.ple_block.per_layer_input_gate"), &x);
        for t in 0..n {
            for j in 0..512 {
                let i = t * 512 + j;
                gate[i] = gelu(gate[i]) * ple[t * ple_stride + ple_offset + j];
            }
        }
        let out = self.lin(&format!("{p}.ple_block.per_layer_projection"), &gate);
        add(
            x,
            &self.norm(
                &format!("{p}.ple_block.post_per_layer_input_norm"),
                &out,
                512,
            ),
        );
        let s = self.floats[&format!("{p}.layer_scalar")][0];
        for z in x {
            *z *= s;
        }
    }
}
impl Linear {
    fn run(&self, x: &[f32]) -> Vec<f32> {
        self.run_rows(x, 0, self.rows)
    }
    fn run_rows(&self, x: &[f32], first: usize, rows: usize) -> Vec<f32> {
        assert!(first + rows <= self.rows && rows % 4 == 0);
        let n = x.len() / self.cols;
        let mut q = vec![0i16; x.len()];
        let mut scales = vec![0f32; n];
        for (t, row) in x.chunks_exact(self.cols).enumerate() {
            let s = (row.iter().fold(0f32, |m, v| m.max(v.abs())) / 127.).max(1e-12);
            scales[t] = s;
            for (a, b) in q[t * self.cols..(t + 1) * self.cols].iter_mut().zip(row) {
                *a = (*b / s).round_ties_even().clamp(-127., 127.) as i16;
            }
        }
        let mut y = vec![0f32; n * rows];
        for t in 0..n {
            for r in (0..rows).step_by(4) {
                let sums = dot4(
                    &q[t * self.cols..(t + 1) * self.cols],
                    &self.w[(first + r) * self.cols..(first + r + 4) * self.cols],
                    self.cols,
                );
                for j in 0..4 {
                    y[t * rows + r + j] = sums[j] as f32 * scales[t] * self.s[first + r + j];
                }
            }
        }
        y
    }
}
#[cfg(not(target_arch = "wasm32"))]
fn dot4(x: &[i16], w: &[i16], cols: usize) -> [i32; 4] {
    let mut s = [0i32; 4];
    for k in 0..cols {
        for j in 0..4 {
            s[j] += x[k] as i32 * w[j * cols + k] as i32;
        }
    }
    s
}
#[cfg(target_arch = "wasm32")]
fn dot4(x: &[i16], w: &[i16], cols: usize) -> [i32; 4] {
    use core::arch::wasm32::*;
    unsafe {
        let mut a = i32x4_splat(0);
        let mut b = a;
        let mut c = a;
        let mut d = a;
        for k in (0..cols).step_by(8) {
            let v = v128_load(x.as_ptr().add(k).cast());
            a = i32x4_add(a, i32x4_dot_i16x8(v, v128_load(w.as_ptr().add(k).cast())));
            b = i32x4_add(
                b,
                i32x4_dot_i16x8(v, v128_load(w.as_ptr().add(cols + k).cast())),
            );
            c = i32x4_add(
                c,
                i32x4_dot_i16x8(v, v128_load(w.as_ptr().add(2 * cols + k).cast())),
            );
            d = i32x4_add(
                d,
                i32x4_dot_i16x8(v, v128_load(w.as_ptr().add(3 * cols + k).cast())),
            );
        }
        let sum = |v: v128| {
            i32x4_extract_lane::<0>(v)
                + i32x4_extract_lane::<1>(v)
                + i32x4_extract_lane::<2>(v)
                + i32x4_extract_lane::<3>(v)
        };
        [sum(a), sum(b), sum(c), sum(d)]
    }
}
fn norm(x: &[f32], width: usize, w: Option<&Vec<f32>>) -> Vec<f32> {
    let mut out = x.to_vec();
    for row in out.chunks_exact_mut(width) {
        let s = (row.iter().map(|v| v * v).sum::<f32>() / width as f32 + 1e-6).powf(-0.5);
        for (j, v) in row.iter_mut().enumerate() {
            *v = *v * s * w.map_or(1., |w| w[j]);
        }
    }
    out
}
fn gelu(x: f32) -> f32 {
    0.5 * x * (1. + (0.7978845608028654 * (x + 0.044715 * x * x * x)).tanh())
}
fn add(x: &mut [f32], y: &[f32]) {
    for (x, y) in x.iter_mut().zip(y) {
        *x += y;
    }
}
fn bf16(x: f32) -> f32 {
    let b = x.to_bits();
    f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff0000)
}
fn rope(x: &mut [f32], n: usize, heads: usize, d: usize, base: f32) {
    for t in 0..n {
        for j in 0..d / 2 {
            let angle = t as f32 / base.powf((2 * j) as f32 / d as f32);
            let (s, c) = angle.sin_cos();
            for h in 0..heads {
                let i = (t * heads + h) * d + j;
                let a = x[i];
                let b = x[i + d / 2];
                x[i] = a * c - b * s;
                x[i + d / 2] = b * c + a * s;
            }
        }
    }
}
fn attention(q: &[f32], k: &[f32], v: &[f32], n: usize, d: usize, kv: usize) -> Vec<f32> {
    let mut out = vec![0f32; n * 4 * d];
    for t in 0..n {
        for h in 0..4 {
            let head = h / (4 / kv);
            let query = &q[(t * 4 + h) * d..(t * 4 + h + 1) * d];
            let mut scores = vec![0f32; n];
            for s in 0..n {
                scores[s] = dot(query, &k[(s * kv + head) * d..(s * kv + head + 1) * d]);
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            for s in &mut scores {
                *s = (*s - max).exp();
            }
            let sum = scores.iter().sum::<f32>();
            for (s, score) in scores.iter().enumerate() {
                let p = *score / sum;
                let values = &v[(s * kv + head) * d..(s * kv + head + 1) * d];
                for j in 0..d {
                    out[(t * 4 + h) * d + j] += p * values[j];
                }
            }
        }
    }
    out
}
fn dot(x: &[f32], y: &[f32]) -> f32 {
    x.iter().zip(y).map(|(x, y)| x * y).sum()
}

#[cfg(target_arch = "wasm32")]
mod canister {
    use super::*;
    use std::cell::RefCell;
    thread_local! {
        static PACK:RefCell<Vec<u8>>=const {RefCell::new(Vec::new())};
        static MODEL:RefCell<Option<Model>>=const {RefCell::new(None)};
        static HARRIER:RefCell<Option<harrier::Harrier>>=const {RefCell::new(None)};
    }
    #[ic_cdk::update]
    fn upload(offset: u64, bytes: serde_bytes::ByteBuf) {
        assert!(ic_cdk::api::is_controller(&ic_cdk::api::msg_caller()));
        assert!(bytes.len() <= 1_800_000);
        PACK.with_borrow_mut(|p| {
            assert_eq!(offset as usize, p.len());
            assert!(p.len() + bytes.len() < 150_000_000);
            p.extend(bytes);
        });
    }
    #[ic_cdk::update]
    fn activate() {
        assert!(ic_cdk::api::is_controller(&ic_cdk::api::msg_caller()));
        let pack = PACK.with_borrow_mut(std::mem::take);
        MODEL.with_borrow_mut(|m| *m = Some(Model::load(&pack)));
    }
    #[ic_cdk::query]
    fn step(
        input: Vec<f32>,
        hidden: Vec<f32>,
        start: u32,
        count: u32,
    ) -> (Vec<f32>, Vec<f32>, u64) {
        assert!(input.iter().chain(&hidden).all(|x| x.is_finite()));
        let t = ic_cdk::api::performance_counter(0);
        let (state, embedding) = MODEL.with_borrow(|m| {
            m.as_ref()
                .unwrap()
                .step(&input, &hidden, start as usize, count as usize)
        });
        (state, embedding, ic_cdk::api::performance_counter(0) - t)
    }
    #[ic_cdk::update(name = "activateHarrier")]
    fn activate_harrier() {
        assert!(ic_cdk::api::is_controller(&ic_cdk::api::msg_caller()));
        let pack = PACK.with_borrow_mut(std::mem::take);
        HARRIER.with_borrow_mut(|m| *m = Some(harrier::Harrier::load(&pack)));
    }
    #[ic_cdk::query(name = "harrierEmbed")]
    fn harrier_embed(input: Vec<f32>) -> (Vec<f32>, u64) {
        assert!(input.iter().all(|x| x.is_finite()));
        let t = ic_cdk::api::performance_counter(0);
        let v = HARRIER.with_borrow(|m| m.as_ref().unwrap().forward(&input));
        (v, ic_cdk::api::performance_counter(0) - t)
    }
    #[ic_cdk::query(name = "harrierStep")]
    fn harrier_step(hidden: Vec<f32>, start: u32, count: u32) -> (Vec<f32>, Vec<f32>, u64) {
        assert!(hidden.iter().all(|x| x.is_finite()));
        let t = ic_cdk::api::performance_counter(0);
        let (state, v) = HARRIER.with_borrow(|m| {
            m.as_ref()
                .unwrap()
                .step(&hidden, start as usize, count as usize)
        });
        (state, v, ic_cdk::api::performance_counter(0) - t)
    }
    #[ic_cdk::query]
    fn embed(input: Vec<f32>) -> (Vec<f32>, u64) {
        inference(input)
    }
    #[ic_cdk::update(name = "embedUpdate")]
    fn embed_update(input: Vec<f32>) -> (Vec<f32>, u64) {
        assert!(ic_cdk::api::is_controller(&ic_cdk::api::msg_caller()));
        inference(input)
    }
    fn inference(input: Vec<f32>) -> (Vec<f32>, u64) {
        assert!(input.len() % 512 == 0 && (1024..=256 * 512).contains(&input.len()));
        let start = ic_cdk::api::performance_counter(0);
        let out = MODEL.with_borrow(|m| m.as_ref().unwrap().forward(&input));
        (out, ic_cdk::api::performance_counter(0) - start)
    }
}
