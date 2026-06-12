use rand::Rng;
use serde_json::Value;
use rayon::prelude::*;

// ── math helpers ─────────────────────────────────────────────────────────────

fn softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let s = if sum > 0.0 { sum } else { 1.0 };
    exps.iter().map(|e| e / s).collect()
}

// w: [out x in], x: [in] → [out]  (single sample, inference only)
fn matvec(w: &[f32], x: &[f32], out: usize, inp: usize) -> Vec<f32> {
    let mut r = vec![0.0f32; out];
    for i in 0..out {
        let base = i * inp;
        let mut s = 0.0f32;
        for j in 0..inp { s += w[base + j] * x[j]; }
        r[i] = s;
    }
    r
}

// BATCHED matmul: W [out x in], X [batch x in] → Y [batch x out]
// Parallelised over batch rows - each row is independent.
fn matmul_batch(w: &[f32], x: &[f32], batch: usize, out: usize, inp: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; batch * out];
    y.par_chunks_mut(out)
        .enumerate()
        .for_each(|(b, row)| {
            let xb = b * inp;
            for i in 0..out {
                let wi = i * inp;
                let mut s = 0.0f32;
                for j in 0..inp { s += w[wi + j] * x[xb + j]; }
                row[i] = s;
            }
        });
    y
}

// Add bias in-place: y [batch x n] += bias [n]
fn add_bias_batch(y: &mut [f32], bias: &[f32], batch: usize, n: usize) {
    y.par_chunks_mut(n).for_each(|row| {
        for i in 0..n { row[i] += bias[i]; }
    });
    let _ = batch; // batch implicit via chunk count
}

// Softmax in-place over each row of x [batch x n]
fn softmax_batch_inplace(x: &mut [f32], _batch: usize, n: usize) {
    x.par_chunks_mut(n).for_each(|row| {
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() { *v = (*v - max).exp(); sum += *v; }
        let s = if sum > 0.0 { sum } else { 1.0 };
        for v in row.iter_mut() { *v /= s; }
    });
}

// Accumulate weight gradients: g_w [out x in] += dY^T @ X
// dY [batch x out], X [batch x in]
// Parallelised over output rows - each row i of g_w is independent.
fn accum_weight_grad(g_w: &mut [f32], dy: &[f32], x: &[f32],
                     batch: usize, out: usize, inp: usize) {
    g_w.par_chunks_mut(inp)
        .enumerate()
        .for_each(|(i, gwi)| {
            for b in 0..batch {
                let dy_bi = dy[b * out + i];
                if dy_bi == 0.0 { continue; }
                let xb = b * inp;
                for j in 0..inp { gwi[j] += dy_bi * x[xb + j]; }
            }
        });
}

// Accumulate bias grad: g_b [out] += sum_batch(dY)
fn accum_bias_grad(g_b: &mut [f32], dy: &[f32], batch: usize, out: usize) {
    for b in 0..batch {
        for i in 0..out { g_b[i] += dy[b * out + i]; }
    }
}

// Backprop through linear: dX [batch x in] += dY [batch x out] @ W [out x in]
// Parallelised over batch - each sample's dX slice is independent.
fn backprop_linear(dx: &mut [f32], dy: &[f32], w: &[f32],
                   batch: usize, out: usize, inp: usize) {
    dx.par_chunks_mut(inp)
        .enumerate()
        .for_each(|(b, dxb)| {
            for j in 0..inp {
                let mut s = 0.0f32;
                for i in 0..out { s += dy[b * out + i] * w[i * inp + j]; }
                dxb[j] += s;
            }
        });
    let _ = batch;
}

fn randn(n: usize, scale: f32) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen::<f32>() * 2.0 * scale - scale).collect()
}

fn zeros(n: usize) -> Vec<f32> { vec![0.0f32; n] }

fn adam_vec(p: &mut [f32], m: &mut [f32], v: &mut [f32], g: &[f32],
            lr: f32, b1: f32, b2: f32, eps: f32, bc1: f32, bc2: f32) {
    for i in 0..p.len() {
        m[i] = b1 * m[i] + (1.0 - b1) * g[i];
        v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
        let mh = m[i] / bc1;
        let vh = v[i] / bc2;
        p[i] -= lr * mh / (vh.sqrt() + eps);
    }
}

// ── AdaptiveSoftmax ───────────────────────────────────────────────────────────

pub struct AdaptiveSoftmax {
    pub hidden:     usize,
    pub vocab:      usize,
    pub head_size:  usize,
    pub tail1_size: usize,
    pub tail2_size: usize,
    pub dim1:       usize,
    pub dim2:       usize,

    pub w_head:  Vec<f32>,
    pub b_head:  Vec<f32>,
    pub w_proj1: Vec<f32>,
    pub w_tail1: Vec<f32>,
    pub b_tail1: Vec<f32>,
    pub w_proj2: Vec<f32>,
    pub w_tail2: Vec<f32>,
    pub b_tail2: Vec<f32>,

    g_w_head:  Vec<f32>, g_b_head:  Vec<f32>,
    g_w_proj1: Vec<f32>,
    g_w_tail1: Vec<f32>, g_b_tail1: Vec<f32>,
    g_w_proj2: Vec<f32>,
    g_w_tail2: Vec<f32>, g_b_tail2: Vec<f32>,

    m_w_head: Vec<f32>, v_w_head: Vec<f32>,
    m_b_head: Vec<f32>, v_b_head: Vec<f32>,
    m_w_proj1: Vec<f32>, v_w_proj1: Vec<f32>,
    m_w_tail1: Vec<f32>, v_w_tail1: Vec<f32>,
    m_b_tail1: Vec<f32>, v_b_tail1: Vec<f32>,
    m_w_proj2: Vec<f32>, v_w_proj2: Vec<f32>,
    m_w_tail2: Vec<f32>, v_w_tail2: Vec<f32>,
    m_b_tail2: Vec<f32>, v_b_tail2: Vec<f32>,

    pub adam_step: i32,
}

impl AdaptiveSoftmax {
    pub fn new(hidden: usize, vocab: usize) -> Self {
        let head_size  = vocab.min(2000);
        let tail1_size = vocab.saturating_sub(head_size).min(3000);
        let tail2_size = vocab.saturating_sub(head_size + tail1_size);
        let dim1 = (hidden / 2).max(1);
        let dim2 = (hidden / 4).max(1);
        let hs = head_size + 2;
        let H  = hidden;
        let sc = 0.01f32;

        macro_rules! w { ($r:expr, $c:expr) => { randn($r * $c, (2.0 / ($r + $c) as f32).sqrt() * sc) } }
        macro_rules! z { ($n:expr)         => { zeros($n) } }

        Self {
            hidden, vocab, head_size, tail1_size, tail2_size, dim1, dim2,

            w_head:  w!(hs,         H),    b_head:  z!(hs),
            w_proj1: w!(dim1,       H),
            w_tail1: w!(tail1_size, dim1), b_tail1: z!(tail1_size),
            w_proj2: w!(dim2,       H),
            w_tail2: w!(tail2_size, dim2), b_tail2: z!(tail2_size),

            g_w_head:  z!(hs*H),          g_b_head:  z!(hs),
            g_w_proj1: z!(dim1*H),
            g_w_tail1: z!(tail1_size*dim1),g_b_tail1: z!(tail1_size),
            g_w_proj2: z!(dim2*H),
            g_w_tail2: z!(tail2_size*dim2),g_b_tail2: z!(tail2_size),

            m_w_head:  z!(hs*H),           v_w_head:  z!(hs*H),
            m_b_head:  z!(hs),             v_b_head:  z!(hs),
            m_w_proj1: z!(dim1*H),         v_w_proj1: z!(dim1*H),
            m_w_tail1: z!(tail1_size*dim1),v_w_tail1: z!(tail1_size*dim1),
            m_b_tail1: z!(tail1_size),     v_b_tail1: z!(tail1_size),
            m_w_proj2: z!(dim2*H),         v_w_proj2: z!(dim2*H),
            m_w_tail2: z!(tail2_size*dim2),v_w_tail2: z!(tail2_size*dim2),
            m_b_tail2: z!(tail2_size),     v_b_tail2: z!(tail2_size),

            adam_step: 0,
        }
    }

    /// Full vocab log-probs for inference/sampling (single sample, unchanged).
    pub fn forward(&self, h: &[f32]) -> Vec<f32> {
        let hs = self.head_size + 2;

        let mut hl = matvec(&self.w_head, h, hs, self.hidden);
        for i in 0..hs { hl[i] += self.b_head[i]; }
        let hp = softmax(&hl);

        let proj1 = matvec(&self.w_proj1, h, self.dim1, self.hidden);
        let mut tl1 = matvec(&self.w_tail1, &proj1, self.tail1_size, self.dim1);
        for i in 0..self.tail1_size { tl1[i] += self.b_tail1[i]; }
        let tp1 = softmax(&tl1);

        let proj2 = matvec(&self.w_proj2, h, self.dim2, self.hidden);
        let mut tl2 = matvec(&self.w_tail2, &proj2, self.tail2_size, self.dim2);
        for i in 0..self.tail2_size { tl2[i] += self.b_tail2[i]; }
        let tp2 = softmax(&tl2);

        let mut lp = vec![f32::NEG_INFINITY; self.vocab];
        for i in 0..self.head_size { lp[i] = hp[i].max(1e-30).ln(); }
        let g1 = hp[self.head_size].max(1e-30);
        let g2 = hp[self.head_size + 1].max(1e-30);
        for i in 0..self.tail1_size {
            lp[self.head_size + i] = (g1 * tp1[i].max(1e-30)).ln();
        }
        for i in 0..self.tail2_size {
            lp[self.head_size + self.tail1_size + i] = (g2 * tp2[i].max(1e-30)).ln();
        }
        lp
    }

    /// BATCHED forward + backward for one time-step.
    /// h_batch: [batch x H] (flat row-major)
    /// targets: [batch]  (token ids, usize::MAX = masked/padding)
    /// Returns (total_loss, d_h_batch [batch x H])
    pub fn accum_batch(&mut self, h_batch: &[f32], targets: &[usize], batch: usize)
        -> (f32, Vec<f32>)
    {
        let H   = self.hidden;
        let hs  = self.head_size + 2;
        let d1  = self.dim1;
        let d2  = self.dim2;
        let ts1 = self.tail1_size;
        let ts2 = self.tail2_size;

        // ── Forward ──────────────────────────────────────────────────────────

        let mut head_logits = matmul_batch(&self.w_head, h_batch, batch, hs, H);
        add_bias_batch(&mut head_logits, &self.b_head, batch, hs);
        softmax_batch_inplace(&mut head_logits, batch, hs);

        let proj1 = matmul_batch(&self.w_proj1, h_batch, batch, d1, H);

        let mut tail1_logits = matmul_batch(&self.w_tail1, &proj1, batch, ts1, d1);
        add_bias_batch(&mut tail1_logits, &self.b_tail1, batch, ts1);
        softmax_batch_inplace(&mut tail1_logits, batch, ts1);

        let proj2 = matmul_batch(&self.w_proj2, h_batch, batch, d2, H);

        let mut tail2_logits = matmul_batch(&self.w_tail2, &proj2, batch, ts2, d2);
        add_bias_batch(&mut tail2_logits, &self.b_tail2, batch, ts2);
        softmax_batch_inplace(&mut tail2_logits, batch, ts2);

        // ── Loss + gradient of softmax outputs ───────────────────────────────

        let mut d_head  = head_logits.clone();  // (probs - onehot), filled below
        let mut d_tail1 = vec![0.0f32; batch * ts1];
        let mut d_tail2 = vec![0.0f32; batch * ts2];
        let mut total_loss = 0.0f32;

        for b in 0..batch {
            let t = targets[b];
            if t == usize::MAX {
                // masked - zero out d_head row so it contributes no gradient
                let base = b * hs;
                for k in 0..hs { d_head[base + k] = 0.0; }
                continue;
            }

            if t < self.head_size {
                let p = head_logits[b * hs + t].max(1e-30);
                total_loss -= p.ln();
                d_head[b * hs + t] -= 1.0;

            } else if t < self.head_size + ts1 {
                let ti = t - self.head_size;
                let pg = head_logits[b * hs + self.head_size].max(1e-30);
                let pt = tail1_logits[b * ts1 + ti].max(1e-30);
                total_loss -= pg.ln() + pt.ln();
                d_head[b * hs + self.head_size] -= 1.0;
                for i in 0..ts1 { d_tail1[b * ts1 + i] = tail1_logits[b * ts1 + i]; }
                d_tail1[b * ts1 + ti] -= 1.0;

            } else {
                let ti = t - self.head_size - ts1;
                let pg = head_logits[b * hs + self.head_size + 1].max(1e-30);
                let pt = tail2_logits[b * ts2 + ti].max(1e-30);
                total_loss -= pg.ln() + pt.ln();
                d_head[b * hs + self.head_size + 1] -= 1.0;
                for i in 0..ts2 { d_tail2[b * ts2 + i] = tail2_logits[b * ts2 + i]; }
                d_tail2[b * ts2 + ti] -= 1.0;
            }
        }

        // ── Weight grads ─────────────────────────────────────────────────────

        accum_weight_grad(&mut self.g_w_head,  &d_head,  h_batch, batch, hs,  H);
        accum_bias_grad  (&mut self.g_b_head,  &d_head,  batch, hs);

        accum_weight_grad(&mut self.g_w_tail1, &d_tail1, &proj1,  batch, ts1, d1);
        accum_bias_grad  (&mut self.g_b_tail1, &d_tail1, batch, ts1);

        accum_weight_grad(&mut self.g_w_tail2, &d_tail2, &proj2,  batch, ts2, d2);
        accum_bias_grad  (&mut self.g_b_tail2, &d_tail2, batch, ts2);

        // ── d_proj1, d_proj2 ─────────────────────────────────────────────────

        let mut d_proj1 = vec![0.0f32; batch * d1];
        backprop_linear(&mut d_proj1, &d_tail1, &self.w_tail1, batch, ts1, d1);
        accum_weight_grad(&mut self.g_w_proj1, &d_proj1, h_batch, batch, d1, H);

        let mut d_proj2 = vec![0.0f32; batch * d2];
        backprop_linear(&mut d_proj2, &d_tail2, &self.w_tail2, batch, ts2, d2);
        accum_weight_grad(&mut self.g_w_proj2, &d_proj2, h_batch, batch, d2, H);

        // ── d_h [batch x H] ──────────────────────────────────────────────────

        let mut dh = vec![0.0f32; batch * H];
        backprop_linear(&mut dh, &d_head,  &self.w_head,  batch, hs, H);
        backprop_linear(&mut dh, &d_proj1, &self.w_proj1, batch, d1, H);
        backprop_linear(&mut dh, &d_proj2, &self.w_proj2, batch, d2, H);

        (total_loss, dh)
    }

    pub fn reset_grads(&mut self) {
        self.g_w_head.iter_mut().for_each(|x| *x = 0.0);
        self.g_b_head.iter_mut().for_each(|x| *x = 0.0);
        self.g_w_proj1.iter_mut().for_each(|x| *x = 0.0);
        self.g_w_tail1.iter_mut().for_each(|x| *x = 0.0);
        self.g_b_tail1.iter_mut().for_each(|x| *x = 0.0);
        self.g_w_proj2.iter_mut().for_each(|x| *x = 0.0);
        self.g_w_tail2.iter_mut().for_each(|x| *x = 0.0);
        self.g_b_tail2.iter_mut().for_each(|x| *x = 0.0);
    }

    pub fn adam_update(&mut self, lr: f32) {
        self.adam_step += 1;
        let t   = self.adam_step;
        let b1  = 0.9f32; let b2 = 0.999f32; let eps = 1e-8f32;
        let bc1 = 1.0 - b1.powi(t);
        let bc2 = 1.0 - b2.powi(t);

        // Read gradient via raw ptr so we can mutate m/v/param in the same call
        // without cloning. Fields are disjoint so this is safe.
        macro_rules! adam {
            ($p:expr, $m:expr, $v:expr, $g:expr) => {{
                let g_slice = unsafe {
                    std::slice::from_raw_parts($g.as_ptr(), $g.len())
                };
                adam_vec(&mut $p, &mut $m, &mut $v, g_slice, lr, b1, b2, eps, bc1, bc2);
            }};
        }
        adam!(self.w_head,  self.m_w_head,  self.v_w_head,  self.g_w_head);
        adam!(self.b_head,  self.m_b_head,  self.v_b_head,  self.g_b_head);
        adam!(self.w_proj1, self.m_w_proj1, self.v_w_proj1, self.g_w_proj1);
        adam!(self.w_tail1, self.m_w_tail1, self.v_w_tail1, self.g_w_tail1);
        adam!(self.b_tail1, self.m_b_tail1, self.v_b_tail1, self.g_b_tail1);
        adam!(self.w_proj2, self.m_w_proj2, self.v_w_proj2, self.g_w_proj2);
        adam!(self.w_tail2, self.m_w_tail2, self.v_w_tail2, self.g_w_tail2);
        adam!(self.b_tail2, self.m_b_tail2, self.v_b_tail2, self.g_b_tail2);
    }

    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "hidden":     self.hidden,
            "vocab":      self.vocab,
            "head_size":  self.head_size,
            "tail1_size": self.tail1_size,
            "tail2_size": self.tail2_size,
            "dim1":       self.dim1,
            "dim2":       self.dim2,
            "w_head":     self.w_head,
            "b_head":     self.b_head,
            "w_proj1":    self.w_proj1,
            "w_tail1":    self.w_tail1,
            "b_tail1":    self.b_tail1,
            "w_proj2":    self.w_proj2,
            "w_tail2":    self.w_tail2,
            "b_tail2":    self.b_tail2,
            "adam_step":  self.adam_step,
        })
    }

    pub fn from_json(v: &Value) -> Option<Self> {
        let hidden     = v["hidden"].as_u64()? as usize;
        let vocab      = v["vocab"].as_u64()? as usize;
        let head_size  = v["head_size"].as_u64()? as usize;
        let tail1_size = v["tail1_size"].as_u64()? as usize;
        let tail2_size = v["tail2_size"].as_u64()? as usize;
        let dim1       = v["dim1"].as_u64()? as usize;
        let dim2       = v["dim2"].as_u64()? as usize;

        fn load_vec(v: &Value) -> Vec<f32> {
            v.as_array().map_or(vec![], |a| {
                a.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect()
            })
        }

        let hs = head_size + 2;
        let H  = hidden;

        Some(Self {
            hidden, vocab, head_size, tail1_size, tail2_size, dim1, dim2,
            w_head:  load_vec(&v["w_head"]),  b_head:  load_vec(&v["b_head"]),
            w_proj1: load_vec(&v["w_proj1"]),
            w_tail1: load_vec(&v["w_tail1"]), b_tail1: load_vec(&v["b_tail1"]),
            w_proj2: load_vec(&v["w_proj2"]),
            w_tail2: load_vec(&v["w_tail2"]), b_tail2: load_vec(&v["b_tail2"]),

            g_w_head:  zeros(hs*H),               g_b_head:  zeros(hs),
            g_w_proj1: zeros(dim1*H),
            g_w_tail1: zeros(tail1_size*dim1),     g_b_tail1: zeros(tail1_size),
            g_w_proj2: zeros(dim2*H),
            g_w_tail2: zeros(tail2_size*dim2),     g_b_tail2: zeros(tail2_size),

            m_w_head:  zeros(hs*H),  v_w_head:  zeros(hs*H),
            m_b_head:  zeros(hs),    v_b_head:  zeros(hs),
            m_w_proj1: zeros(dim1*H),v_w_proj1: zeros(dim1*H),
            m_w_tail1: zeros(tail1_size*dim1), v_w_tail1: zeros(tail1_size*dim1),
            m_b_tail1: zeros(tail1_size),      v_b_tail1: zeros(tail1_size),
            m_w_proj2: zeros(dim2*H),v_w_proj2: zeros(dim2*H),
            m_w_tail2: zeros(tail2_size*dim2), v_w_tail2: zeros(tail2_size*dim2),
            m_b_tail2: zeros(tail2_size),      v_b_tail2: zeros(tail2_size),

            adam_step: v["adam_step"].as_i64().unwrap_or(0) as i32,
        })
    }
}
