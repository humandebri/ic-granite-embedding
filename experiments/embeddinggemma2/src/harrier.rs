//! Fixed Harrier 270M text encoder; shares only the generic W8 linear kernel.
use super::{Model, add, bf16, dot, gelu, norm, rope};
pub struct Harrier {
    inner: Model,
}
impl Harrier {
    pub fn load(pack: &[u8]) -> Self {
        Self {
            inner: Model::load(pack),
        }
    }
    fn norm(&self, name: &str, x: &[f32], width: usize) -> Vec<f32> {
        let w: Vec<_> = self.inner.floats[&format!("{name}.weight")]
            .iter()
            .map(|x| x + 1.)
            .collect();
        norm(x, width, Some(&w))
    }
    fn layer(&self, x: &mut Vec<f32>, l: usize) {
        let n = x.len() / 640;
        let p = format!("layers.{l}");
        let a = format!("{p}.self_attn");
        let h = self.norm(&format!("{p}.input_layernorm"), x, 640);
        let mut q = self.norm(
            &format!("{a}.q_norm"),
            &self.inner.lin(&format!("{a}.q_proj"), &h),
            256,
        );
        let mut k = self.norm(
            &format!("{a}.k_norm"),
            &self.inner.lin(&format!("{a}.k_proj"), &h),
            256,
        );
        let mut v = self.inner.lin(&format!("{a}.v_proj"), &h);
        rope(&mut q, n, 4, 256, 1_000_000.);
        rope(&mut k, n, 1, 256, 1_000_000.);
        for z in k.iter_mut().chain(v.iter_mut()) {
            *z = bf16(*z);
        }
        let mut context = vec![0f32; n * 1024];
        for t in 0..n {
            for head in 0..4 {
                let query = &q[(t * 4 + head) * 256..(t * 4 + head + 1) * 256];
                let mut scores: Vec<_> = (0..=t)
                    .map(|s| dot(query, &k[s * 256..(s + 1) * 256]) / 16.)
                    .collect();
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                for s in &mut scores {
                    *s = (*s - max).exp();
                }
                let sum = scores.iter().sum::<f32>();
                for (s, score) in scores.iter().enumerate() {
                    let prob = *score / sum;
                    for j in 0..256 {
                        context[(t * 4 + head) * 256 + j] += prob * v[s * 256 + j];
                    }
                }
            }
        }
        let out = self.inner.lin(&format!("{a}.o_proj"), &context);
        add(
            x,
            &self.norm(&format!("{p}.post_attention_layernorm"), &out, 640),
        );
        let h = self.norm(&format!("{p}.pre_feedforward_layernorm"), x, 640);
        let mut gate = self.inner.lin(&format!("{p}.mlp.gate_proj"), &h);
        let up = self.inner.lin(&format!("{p}.mlp.up_proj"), &h);
        for (g, u) in gate.iter_mut().zip(up) {
            *g = gelu(*g) * u;
        }
        let out = self.inner.lin(&format!("{p}.mlp.down_proj"), &gate);
        add(
            x,
            &self.norm(&format!("{p}.post_feedforward_layernorm"), &out, 640),
        );
    }
    fn finish(&self, x: &[f32]) -> Vec<f32> {
        let mut v = self.norm("norm", &x[x.len() - 640..], 640);
        let scale = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut v {
            *x /= scale;
        }
        v
    }
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        self.forward_trace(input, None)
    }
    pub fn forward_trace(&self, input: &[f32], mut trace: Option<&mut Vec<Vec<f32>>>) -> Vec<f32> {
        assert!(input.len() % 640 == 0 && (1280..=256 * 640).contains(&input.len()));
        let mut x = input.to_vec();
        for l in 0..18 {
            self.layer(&mut x, l);
            if let Some(trace) = trace.as_deref_mut() {
                trace.push(x.clone());
            }
        }
        self.finish(&x)
    }
    pub fn step(&self, hidden: &[f32], start: usize, count: usize) -> (Vec<f32>, Vec<f32>) {
        assert!(hidden.len() % 640 == 0 && (1280..=256 * 640).contains(&hidden.len()));
        assert!((1..=2).contains(&count) && start < 18 && start + count <= 18);
        let mut x = hidden.to_vec();
        for l in start..start + count {
            self.layer(&mut x, l);
        }
        if start + count == 18 {
            (Vec::new(), self.finish(&x))
        } else {
            (x, Vec::new())
        }
    }
}
