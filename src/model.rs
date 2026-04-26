use ndarray::{s, Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use rand_distr::Normal;
use crate::lstm_gpu::LSTMGpu;

pub struct LSTMModel {
    pub embed: Array2<f32>,
    pub w_x: Array2<f32>,
    pub w_h: Array2<f32>,
    pub b: Array1<f32>,
    pub w_out: Array2<f32>,
    pub b_out: Array1<f32>,
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub hidden_dim: usize,
    m_embed: Array2<f32>,
    v_embed: Array2<f32>,
    m_w_x: Array2<f32>,
    v_w_x: Array2<f32>,
    m_w_h: Array2<f32>,
    v_w_h: Array2<f32>,
    m_b: Array1<f32>,
    v_b: Array1<f32>,
    m_w_out: Array2<f32>,
    v_w_out: Array2<f32>,
    m_b_out: Array1<f32>,
    v_b_out: Array1<f32>,
    pub adam_t: usize,
    pub gpu: Option<LSTMGpu>,
}

#[derive(Clone)]
pub struct LSTMState {
    pub h: Array1<f32>,
    pub c: Array1<f32>,
}

struct StepCacheBatch {
    x: Array2<f32>,
    token_ids: Vec<usize>,
    h_prev: Array2<f32>,
    c_prev: Array2<f32>,
    i: Array2<f32>,
    f: Array2<f32>,
    o: Array2<f32>,
    g: Array2<f32>,
    c: Array2<f32>,
    c_tanh: Array2<f32>,
    h: Array2<f32>,
}

impl LSTMModel {
    pub fn new(vocab_size: usize, embed_dim: usize, hidden_dim: usize) -> Self {
        println!("\n================================");
        println!("           ARIA v7 LSTM           ");
        println!("================================");
        println!("  Vocabulary size:    {}", vocab_size);
        println!("  Embedding dim:      {}", embed_dim);
        println!("  Hidden dim:         {}", hidden_dim);
        println!("  Batch processing:   enabled");
        println!("  Optimizer:          Adam");
        println!("  Fused gates:        4*H single matmul");
        let total_params = Self::count_params(vocab_size, embed_dim, hidden_dim);
        println!("  Total parameters:   {:.1}M", total_params as f32 / 1e6);
        println!("================================\n");

        let scale_e = 1.0 / (embed_dim as f32).sqrt();
        let scale_i = 1.0 / (embed_dim as f32).sqrt();
        let scale_h = 1.0 / (hidden_dim as f32).sqrt();
        let fh = 4 * hidden_dim;

        let mut b = Array1::<f32>::zeros(fh);
        for k in hidden_dim..2 * hidden_dim { b[k] = 1.0; }

        let gpu = LSTMGpu::try_init();
        if gpu.is_some() {
            println!("GPU matmul: ENABLED");
        } else {
            println!("GPU matmul: not available, using CPU");
        }
        println!();

        LSTMModel {
            embed: Array2::random((vocab_size, embed_dim), Normal::new(0.0, scale_e).unwrap()),
            w_x: Array2::random((embed_dim, fh), Normal::new(0.0, scale_i).unwrap()),
            w_h: Array2::random((hidden_dim, fh), Normal::new(0.0, scale_h).unwrap()),
            b,
            w_out: Array2::random((hidden_dim, vocab_size), Normal::new(0.0, scale_h).unwrap()),
            b_out: Array1::<f32>::zeros(vocab_size),
            m_embed: Array2::zeros((vocab_size, embed_dim)),
            v_embed: Array2::zeros((vocab_size, embed_dim)),
            m_w_x: Array2::zeros((embed_dim, fh)),
            v_w_x: Array2::zeros((embed_dim, fh)),
            m_w_h: Array2::zeros((hidden_dim, fh)),
            v_w_h: Array2::zeros((hidden_dim, fh)),
            m_b: Array1::zeros(fh),
            v_b: Array1::zeros(fh),
            m_w_out: Array2::zeros((hidden_dim, vocab_size)),
            v_w_out: Array2::zeros((hidden_dim, vocab_size)),
            m_b_out: Array1::zeros(vocab_size),
            v_b_out: Array1::zeros(vocab_size),
            adam_t: 0,
            vocab_size, embed_dim, hidden_dim,
            gpu,
        }
    }

    fn count_params(vs: usize, ed: usize, hd: usize) -> usize {
        vs * ed + ed * 4 * hd + hd * 4 * hd + 4 * hd + hd * vs + vs
    }

    pub fn init_state(&self) -> LSTMState {
        LSTMState { h: Array1::zeros(self.hidden_dim), c: Array1::zeros(self.hidden_dim) }
    }

    fn matmul_2d(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
        let (m, k) = (a.nrows(), a.ncols());
        let n = b.ncols();

        if let Some(ref gpu) = self.gpu {
            if m * k * n > 100_000 {
                let a_slice = a.as_slice().unwrap_or_else(|| {
                    panic!("non-contiguous");
                });
                let b_slice = b.as_slice().unwrap_or_else(|| {
                    panic!("non-contiguous");
                });
                let result = gpu.matmul(a_slice, b_slice, m as u32, k as u32, n as u32);
                return Array2::from_shape_vec((m, n), result).unwrap();
            }
        }

        a.dot(b)
    }

    fn embed_lookup_batch(&self, token_ids: &[usize]) -> Array2<f32> {
        let bs = token_ids.len();
        let mut x = Array2::<f32>::zeros((bs, self.embed_dim));
        for (i, &t) in token_ids.iter().enumerate() {
            if t < self.vocab_size {
                x.row_mut(i).assign(&self.embed.row(t));
            }
        }
        x
    }

    fn step_batch(&self, token_ids: &[usize], h_prev: &Array2<f32>, c_prev: &Array2<f32>) -> StepCacheBatch {
        let x = self.embed_lookup_batch(token_ids);
        let h_dim = self.hidden_dim;

        let mut gates = self.matmul_2d(&x, &self.w_x);
        gates += &self.matmul_2d(h_prev, &self.w_h);
        for mut row in gates.axis_iter_mut(Axis(0)) { row += &self.b; }

        let mut ig = gates.slice(s![.., 0..h_dim]).to_owned();
        let mut fg = gates.slice(s![.., h_dim..2*h_dim]).to_owned();
        let mut og = gates.slice(s![.., 2*h_dim..3*h_dim]).to_owned();
        let mut gg = gates.slice(s![.., 3*h_dim..4*h_dim]).to_owned();

        ig.mapv_inplace(|v| 1.0 / (1.0 + (-v).exp()));
        fg.mapv_inplace(|v| 1.0 / (1.0 + (-v).exp()));
        og.mapv_inplace(|v| 1.0 / (1.0 + (-v).exp()));
        gg.mapv_inplace(|v| v.tanh());

        let c_new = &(&fg * c_prev) + &(&ig * &gg);
        let mut ct = c_new.clone();
        ct.mapv_inplace(|v| v.tanh());
        let h_new = &og * &ct;

        StepCacheBatch {
            x, token_ids: token_ids.to_vec(),
            h_prev: h_prev.clone(), c_prev: c_prev.clone(),
            i: ig, f: fg, o: og, g: gg,
            c: c_new, c_tanh: ct, h: h_new,
        }
    }

    pub fn step(&self, token_id: usize, state: &LSTMState) -> (Array1<f32>, LSTMState) {
        let hp = state.h.view().into_shape((1, self.hidden_dim)).unwrap().to_owned();
        let cp = state.c.view().into_shape((1, self.hidden_dim)).unwrap().to_owned();
        let cache = self.step_batch(&[token_id], &hp, &cp);
        let h_out = cache.h.row(0).to_owned();
        let c_out = cache.c.row(0).to_owned();
        let logits = &h_out.dot(&self.w_out) + &self.b_out;
        (logits, LSTMState { h: h_out, c: c_out })
    }

    pub fn forward_seq(&self, tokens: &[usize]) -> (Array1<f32>, LSTMState) {
        let mut state = self.init_state();
        let mut logits = Array1::zeros(self.vocab_size);
        for &t in tokens {
            let (l, s) = self.step(t, &state);
            logits = l; state = s;
        }
        (logits, state)
    }

    pub fn softmax(&self, x: &Array1<f32>) -> Array1<f32> {
        let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp = x.map(|a| (a - max).exp());
        let sum: f32 = exp.sum();
        if sum > 0.0 { &exp / sum } else { Array1::from_elem(x.len(), 1.0 / x.len() as f32) }
    }

    pub fn sample_greedy(&self, logits: &Array1<f32>) -> usize {
        logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0)
    }

    pub fn sample_top_k(&self, logits: &Array1<f32>, temperature: f32, top_k: usize) -> usize {
        let temp = temperature.max(0.05);
        let scaled: Array1<f32> = logits.map(|x| x / temp);
        let mut indexed: Vec<(usize, f32)> = scaled.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let k = top_k.min(indexed.len()).max(1);
        let top: Vec<(usize, f32)> = indexed.into_iter().take(k).collect();
        let mx = top[0].1;
        let exps: Vec<f32> = top.iter().map(|(_, v)| (v - mx).exp()).collect();
        let sum: f32 = exps.iter().sum();
        if sum <= 0.0 || !sum.is_finite() { return top[0].0; }
        let r: f32 = rand::random();
        let mut cum = 0.0f32;
        for (idx, e) in exps.iter().enumerate() {
            cum += e / sum;
            if r < cum { return top[idx].0; }
        }
        top[k - 1].0
    }

    pub fn sample_action(&self, logits: &Array1<f32>) -> (usize, f32) {
        let probs = self.softmax(logits);
        let r: f32 = rand::random();
        let mut cum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if r < cum { return (i, p.max(1e-20).ln()); }
        }
        let last = probs.len() - 1;
        (last, probs[last].max(1e-20).ln())
    }

    pub fn sample_action_with_temp(&self, logits: &Array1<f32>, temperature: f32) -> (usize, f32) {
        let id = self.sample_top_k(logits, temperature, 20);
        let probs = self.softmax(logits);
        (id, probs[id].max(1e-20).ln())
    }

    fn clip_grad_global(slices: &mut [&mut [f32]], max_norm: f32) {
        let mut sq = 0.0f64;
        for s in slices.iter() { for &v in s.iter() { sq += (v as f64) * (v as f64); } }
        let norm = sq.sqrt() as f32;
        if norm > max_norm && norm > 0.0 {
            let scale = max_norm / norm;
            for s in slices.iter_mut() { for v in s.iter_mut() { *v *= scale; } }
        }
    }

    fn adam_update_2d(p: &mut Array2<f32>, g: &Array2<f32>, m: &mut Array2<f32>, v: &mut Array2<f32>, t: usize, lr: f32) {
        let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
        let bc1 = 1.0 - b1.powi(t as i32);
        let bc2 = 1.0 - b2.powi(t as i32);
        let ps = p.as_slice_mut().unwrap();
        let gs = g.as_slice().unwrap();
        let ms = m.as_slice_mut().unwrap();
        let vs = v.as_slice_mut().unwrap();
        for k in 0..ps.len() {
            ms[k] = b1 * ms[k] + (1.0 - b1) * gs[k];
            vs[k] = b2 * vs[k] + (1.0 - b2) * gs[k] * gs[k];
            ps[k] -= lr * (ms[k] / bc1) / ((vs[k] / bc2).sqrt() + eps);
        }
    }

    fn adam_update_1d(p: &mut Array1<f32>, g: &Array1<f32>, m: &mut Array1<f32>, v: &mut Array1<f32>, t: usize, lr: f32) {
        let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
        let bc1 = 1.0 - b1.powi(t as i32);
        let bc2 = 1.0 - b2.powi(t as i32);
        let ps = p.as_slice_mut().unwrap();
        let gs = g.as_slice().unwrap();
        let ms = m.as_slice_mut().unwrap();
        let vs = v.as_slice_mut().unwrap();
        for k in 0..ps.len() {
            ms[k] = b1 * ms[k] + (1.0 - b1) * gs[k];
            vs[k] = b2 * vs[k] + (1.0 - b2) * gs[k] * gs[k];
            ps[k] -= lr * (ms[k] / bc1) / ((vs[k] / bc2).sqrt() + eps);
        }
    }

    pub fn train_batch(&mut self, sequences: &[Vec<usize>], learning_rate: f32) -> f32 {
        if sequences.is_empty() { return 0.0; }
        let bs = sequences.len();
        let max_len = sequences.iter().map(|s| s.len()).max().unwrap_or(0);
        if max_len < 2 { return 0.0; }

        let hd = self.hidden_dim;
        let vs = self.vocab_size;
        let ed = self.embed_dim;

        let mut tok_pad = vec![0usize; bs * max_len];
        let mut mask = vec![0.0f32; bs * max_len];
        for (b, seq) in sequences.iter().enumerate() {
            for (t, &tk) in seq.iter().enumerate() {
                tok_pad[b * max_len + t] = tk;
                mask[b * max_len + t] = 1.0;
            }
        }

        let gpu_ref = self.gpu.take();

        let mut h = Array2::<f32>::zeros((bs, hd));
        let mut c = Array2::<f32>::zeros((bs, hd));
        let mut caches: Vec<StepCacheBatch> = Vec::with_capacity(max_len);

        for t in 0..max_len {
            let toks: Vec<usize> = (0..bs).map(|b| tok_pad[b * max_len + t]).collect();
            let cache = self.step_batch(&toks, &h, &c);
            h = cache.h.clone();
            c = cache.c.clone();
            caches.push(cache);
        }

        let mut d_w_out = Array2::<f32>::zeros((hd, vs));
        let mut d_b_out = Array1::<f32>::zeros(vs);
        let mut dh_from_loss: Vec<Array2<f32>> = (0..max_len).map(|_| Array2::zeros((bs, hd))).collect();

        let mut total_loss = 0.0f64;
        let mut loss_count = 0usize;

        for t in 0..max_len.saturating_sub(1) {
            let h_t = &caches[t].h;
            let logits = self.matmul_2d(h_t, &self.w_out);

            for b in 0..bs {
                if mask[b * max_len + t + 1] < 0.5 { continue; }
                let target = tok_pad[b * max_len + t + 1];
                if target >= vs { continue; }

                let row = logits.row(b);
                let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut exps = vec![0.0f32; vs];
                let mut sum = 0.0f64;
                for k in 0..vs { let e = (row[k] - mx).exp(); exps[k] = e; sum += e as f64; }
                let inv = 1.0 / sum as f32;
                let p_tgt = exps[target] * inv;
                total_loss += -(p_tgt.max(1e-20) as f64).ln();
                loss_count += 1;

                let mut dl = vec![0.0f32; vs];
                for k in 0..vs { dl[k] = exps[k] * inv; }
                dl[target] -= 1.0;

                for k in 0..vs {
                    d_b_out[k] += dl[k];
                    for j in 0..hd { d_w_out[[j, k]] += h_t[[b, j]] * dl[k]; }
                }

                for j in 0..hd {
                    let mut acc = 0.0f32;
                    for k in 0..vs { acc += self.w_out[[j, k]] * dl[k]; }
                    dh_from_loss[t][[b, j]] += acc;
                }
            }
        }

        if loss_count == 0 { self.gpu = gpu_ref; return 0.0; }

        let mut d_embed = Array2::<f32>::zeros((vs, ed));
        let mut d_w_x = Array2::<f32>::zeros((ed, 4 * hd));
        let mut d_w_h = Array2::<f32>::zeros((hd, 4 * hd));
        let mut d_b_lstm = Array1::<f32>::zeros(4 * hd);

        let mut dh_next = Array2::<f32>::zeros((bs, hd));
        let mut dc_next = Array2::<f32>::zeros((bs, hd));

        for t in (0..max_len).rev() {
            let cache = &caches[t];
            let dh = &dh_next + &dh_from_loss[t];
            let do_ = &dh * &cache.c_tanh;
            let dc_tanh = &dh * &cache.o;
            let dtanh = cache.c_tanh.mapv(|v| 1.0 - v * v);
            let dc = &dc_next + &(&dc_tanh * &dtanh);
            let di = &dc * &cache.g;
            let df = &dc * &cache.c_prev;
            let dg = &dc * &cache.i;
            let di_pre = &di * &cache.i.mapv(|v| v * (1.0 - v));
            let df_pre = &df * &cache.f.mapv(|v| v * (1.0 - v));
            let do_pre = &do_ * &cache.o.mapv(|v| v * (1.0 - v));
            let dg_pre = &dg * &cache.g.mapv(|v| 1.0 - v * v);

            let mut d_gates = Array2::<f32>::zeros((bs, 4 * hd));
            d_gates.slice_mut(s![.., 0..hd]).assign(&di_pre);
            d_gates.slice_mut(s![.., hd..2*hd]).assign(&df_pre);
            d_gates.slice_mut(s![.., 2*hd..3*hd]).assign(&do_pre);
            d_gates.slice_mut(s![.., 3*hd..4*hd]).assign(&dg_pre);

            for b in 0..bs {
                if mask[b * max_len + t] < 0.5 { d_gates.row_mut(b).fill(0.0); }
            }

            d_w_x += &cache.x.t().dot(&d_gates);
            d_w_h += &cache.h_prev.t().dot(&d_gates);
            d_b_lstm += &d_gates.sum_axis(Axis(0));

            let dx = d_gates.dot(&self.w_x.t());
            for b in 0..bs {
                if mask[b * max_len + t] < 0.5 { continue; }
                let tok = cache.token_ids[b];
                if tok < vs {
                    for k in 0..ed { d_embed[[tok, k]] += dx[[b, k]]; }
                }
            }

            dh_next = d_gates.dot(&self.w_h.t());
            dc_next = &dc * &cache.f;
        }

        let scale = 1.0 / loss_count as f32;
        d_w_out.mapv_inplace(|v| v * scale);
        d_b_out.mapv_inplace(|v| v * scale);
        d_w_x.mapv_inplace(|v| v * scale);
        d_w_h.mapv_inplace(|v| v * scale);
        d_b_lstm.mapv_inplace(|v| v * scale);
        d_embed.mapv_inplace(|v| v * scale);

        {
            let mut all: Vec<&mut [f32]> = vec![
                d_w_out.as_slice_mut().unwrap(),
                d_b_out.as_slice_mut().unwrap(),
                d_w_x.as_slice_mut().unwrap(),
                d_w_h.as_slice_mut().unwrap(),
                d_b_lstm.as_slice_mut().unwrap(),
                d_embed.as_slice_mut().unwrap(),
            ];
            Self::clip_grad_global(&mut all, 5.0);
        }

        self.adam_t += 1;
        let t = self.adam_t;
        let lr = learning_rate;
        Self::adam_update_2d(&mut self.w_out, &d_w_out, &mut self.m_w_out, &mut self.v_w_out, t, lr);
        Self::adam_update_1d(&mut self.b_out, &d_b_out, &mut self.m_b_out, &mut self.v_b_out, t, lr);
        Self::adam_update_2d(&mut self.w_x, &d_w_x, &mut self.m_w_x, &mut self.v_w_x, t, lr);
        Self::adam_update_2d(&mut self.w_h, &d_w_h, &mut self.m_w_h, &mut self.v_w_h, t, lr);
        Self::adam_update_1d(&mut self.b, &d_b_lstm, &mut self.m_b, &mut self.v_b, t, lr);
        Self::adam_update_2d(&mut self.embed, &d_embed, &mut self.m_embed, &mut self.v_embed, t, lr);

        self.gpu = gpu_ref;
        (total_loss / loss_count as f64) as f32
    }

    pub fn backward_step(&mut self, tokens: &[usize], learning_rate: f32) -> f32 {
        self.train_batch(&[tokens.to_vec()], learning_rate)
    }

    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        use std::fs;
        let data = serde_json::json!({
            "vocab_size": self.vocab_size, "embed_dim": self.embed_dim, "hidden_dim": self.hidden_dim,
            "format": "v7_fused",
            "embed": self.embed.iter().copied().collect::<Vec<_>>(),
            "w_x": self.w_x.iter().copied().collect::<Vec<_>>(),
            "w_h": self.w_h.iter().copied().collect::<Vec<_>>(),
            "b": self.b.iter().copied().collect::<Vec<_>>(),
            "w_out": self.w_out.iter().copied().collect::<Vec<_>>(),
            "b_out": self.b_out.iter().copied().collect::<Vec<_>>(),
        });
        fs::write(path, serde_json::to_string(&data)?)?;
        Ok(())
    }

    pub fn load(path: &str, vocab_size: usize, embed_dim: usize, hidden_dim: usize) -> anyhow::Result<Self> {
        use std::fs;
        let content = fs::read_to_string(path)?;
        let data: serde_json::Value = serde_json::from_str(&content)?;
        let mut model = LSTMModel::new(vocab_size, embed_dim, hidden_dim);
        let fh = 4 * hidden_dim;
        if let Some(a) = data["embed"].as_array() {
            let v: Vec<f32> = a.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect();
            if v.len() == vocab_size * embed_dim { model.embed = Array2::from_shape_vec((vocab_size, embed_dim), v).unwrap(); }
        }
        if let Some(a) = data["w_x"].as_array() {
            let v: Vec<f32> = a.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect();
            if v.len() == embed_dim * fh { model.w_x = Array2::from_shape_vec((embed_dim, fh), v).unwrap(); }
        }
        if let Some(a) = data["w_h"].as_array() {
            let v: Vec<f32> = a.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect();
            if v.len() == hidden_dim * fh { model.w_h = Array2::from_shape_vec((hidden_dim, fh), v).unwrap(); }
        }
        if let Some(a) = data["b"].as_array() {
            let v: Vec<f32> = a.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect();
            if v.len() == fh { model.b = Array1::from_vec(v); }
        }
        if let Some(a) = data["w_out"].as_array() {
            let v: Vec<f32> = a.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect();
            if v.len() == hidden_dim * vocab_size { model.w_out = Array2::from_shape_vec((hidden_dim, vocab_size), v).unwrap(); }
        }
        if let Some(a) = data["b_out"].as_array() {
            let v: Vec<f32> = a.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect();
            if v.len() == vocab_size { model.b_out = Array1::from_vec(v); }
        }
        Ok(model)
    }
}
