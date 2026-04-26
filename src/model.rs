use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use rand_distr::Normal;
use rand::seq::SliceRandom;

pub struct LSTMModel {
    pub embed: Array2<f32>,

    pub w_ii: Array2<f32>,
    pub w_hi: Array2<f32>,
    pub b_i: Array1<f32>,

    pub w_if: Array2<f32>,
    pub w_hf: Array2<f32>,
    pub b_f: Array1<f32>,

    pub w_io: Array2<f32>,
    pub w_ho: Array2<f32>,
    pub b_o: Array1<f32>,

    pub w_ig: Array2<f32>,
    pub w_hg: Array2<f32>,
    pub b_g: Array1<f32>,

    pub w_out: Array2<f32>,
    pub b_out: Array1<f32>,

    pub vocab_size: usize,
    pub embed_dim: usize,
    pub hidden_dim: usize,
}

#[derive(Clone)]
pub struct LSTMState {
    pub h: Array1<f32>,
    pub c: Array1<f32>,
}

struct StepCache {
    x: Array1<f32>,
    token_id: usize,
    h_prev: Array1<f32>,
    c_prev: Array1<f32>,
    i: Array1<f32>,
    f: Array1<f32>,
    o: Array1<f32>,
    g: Array1<f32>,
    c: Array1<f32>,
    c_tanh: Array1<f32>,
    h: Array1<f32>,
    logits: Array1<f32>,
}

impl LSTMModel {
    pub fn new(vocab_size: usize, embed_dim: usize, hidden_dim: usize) -> Self {
        println!("\n================================");
        println!("           ARIA v3 LSTM           ");
        println!("================================");
        println!("Initializing model with:");
        println!("  - Vocabulary size: {}", vocab_size);
        println!("  - Embedding dimension: {}", embed_dim);
        println!("  - Hidden dimension: {}", hidden_dim);

        let scale_embed = 1.0 / (embed_dim as f32).sqrt();
        let scale_in = 1.0 / (embed_dim as f32).sqrt();
        let scale_hid = 1.0 / (hidden_dim as f32).sqrt();

        let total_params = Self::count_params(vocab_size, embed_dim, hidden_dim);
        println!("  - Total parameters: {:.1}M", total_params as f32 / 1e6);
        println!("================================\n");

        let mut b_f = Array1::zeros(hidden_dim);
        b_f.fill(1.0);

        LSTMModel {
            embed: Array2::random((vocab_size, embed_dim), Normal::new(0.0, scale_embed).unwrap()),

            w_ii: Array2::random((embed_dim, hidden_dim), Normal::new(0.0, scale_in).unwrap()),
            w_hi: Array2::random((hidden_dim, hidden_dim), Normal::new(0.0, scale_hid).unwrap()),
            b_i: Array1::zeros(hidden_dim),

            w_if: Array2::random((embed_dim, hidden_dim), Normal::new(0.0, scale_in).unwrap()),
            w_hf: Array2::random((hidden_dim, hidden_dim), Normal::new(0.0, scale_hid).unwrap()),
            b_f,

            w_io: Array2::random((embed_dim, hidden_dim), Normal::new(0.0, scale_in).unwrap()),
            w_ho: Array2::random((hidden_dim, hidden_dim), Normal::new(0.0, scale_hid).unwrap()),
            b_o: Array1::zeros(hidden_dim),

            w_ig: Array2::random((embed_dim, hidden_dim), Normal::new(0.0, scale_in).unwrap()),
            w_hg: Array2::random((hidden_dim, hidden_dim), Normal::new(0.0, scale_hid).unwrap()),
            b_g: Array1::zeros(hidden_dim),

            w_out: Array2::random((hidden_dim, vocab_size), Normal::new(0.0, scale_hid).unwrap()),
            b_out: Array1::zeros(vocab_size),

            vocab_size,
            embed_dim,
            hidden_dim,
        }
    }

    fn count_params(vocab_size: usize, embed_dim: usize, hidden_dim: usize) -> usize {
        let embed_params = vocab_size * embed_dim;
        let lstm_params = 4 * (embed_dim * hidden_dim + hidden_dim * hidden_dim + hidden_dim);
        let output_params = hidden_dim * vocab_size + vocab_size;
        embed_params + lstm_params + output_params
    }

    pub fn init_state(&self) -> LSTMState {
        LSTMState {
            h: Array1::zeros(self.hidden_dim),
            c: Array1::zeros(self.hidden_dim),
        }
    }

    fn sigmoid_arr(x: &Array1<f32>) -> Array1<f32> {
        x.map(|a| 1.0 / (1.0 + (-a).exp()))
    }

    fn tanh_arr(x: &Array1<f32>) -> Array1<f32> {
        x.map(|a| a.tanh())
    }

    fn step_cached(&self, token_id: usize, h_prev: &Array1<f32>, c_prev: &Array1<f32>) -> StepCache {
        let x = if token_id < self.vocab_size {
            self.embed.row(token_id).to_owned()
        } else {
            Array1::zeros(self.embed_dim)
        };

        let i_pre = &x.dot(&self.w_ii) + &h_prev.dot(&self.w_hi) + &self.b_i;
        let f_pre = &x.dot(&self.w_if) + &h_prev.dot(&self.w_hf) + &self.b_f;
        let o_pre = &x.dot(&self.w_io) + &h_prev.dot(&self.w_ho) + &self.b_o;
        let g_pre = &x.dot(&self.w_ig) + &h_prev.dot(&self.w_hg) + &self.b_g;

        let i = Self::sigmoid_arr(&i_pre);
        let f = Self::sigmoid_arr(&f_pre);
        let o = Self::sigmoid_arr(&o_pre);
        let g = Self::tanh_arr(&g_pre);

        let c = &(&f * c_prev) + &(&i * &g);
        let c_tanh = Self::tanh_arr(&c);
        let h = &o * &c_tanh;

        let logits = &h.dot(&self.w_out) + &self.b_out;

        StepCache {
            x,
            token_id,
            h_prev: h_prev.clone(),
            c_prev: c_prev.clone(),
            i,
            f,
            o,
            g,
            c,
            c_tanh,
            h,
            logits,
        }
    }

    pub fn step(&self, token_id: usize, state: &LSTMState) -> (Array1<f32>, LSTMState) {
        let cache = self.step_cached(token_id, &state.h, &state.c);
        (cache.logits, LSTMState { h: cache.h, c: cache.c })
    }

    pub fn forward_seq(&self, tokens: &[usize]) -> (Array1<f32>, LSTMState) {
        let mut state = self.init_state();
        let mut logits = Array1::zeros(self.vocab_size);

        for &token_id in tokens {
            let (next_logits, next_state) = self.step(token_id, &state);
            logits = next_logits;
            state = next_state;
        }

        (logits, state)
    }

    pub fn softmax(&self, x: &Array1<f32>) -> Array1<f32> {
        let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp = x.map(|a| (a - max).exp());
        let sum: f32 = exp.sum();
        if sum > 0.0 {
            &exp / sum
        } else {
            Array1::from_elem(x.len(), 1.0 / x.len() as f32)
        }
    }

    pub fn sample_greedy(&self, logits: &Array1<f32>) -> usize {
        let mut best = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best = i;
            }
        }
        best
    }

    pub fn sample_top_k(&self, logits: &Array1<f32>, temperature: f32, top_k: usize) -> usize {
        let temp = temperature.max(0.05);
        let scaled: Array1<f32> = logits.map(|x| x / temp);

        let mut indexed: Vec<(usize, f32)> = scaled.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let k = top_k.min(indexed.len()).max(1);
        let top: Vec<(usize, f32)> = indexed.into_iter().take(k).collect();

        let max_v = top.iter().map(|(_, v)| *v).fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = top.iter().map(|(_, v)| (v - max_v).exp()).collect();
        let sum: f32 = exps.iter().sum();

        if sum <= 0.0 || !sum.is_finite() {
            return top[0].0;
        }

        let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

        let r: f32 = rand::random();
        let mut cum = 0.0f32;
        for (idx, p) in probs.iter().enumerate() {
            cum += p;
            if r < cum {
                return top[idx].0;
            }
        }
        top[k - 1].0
    }

    pub fn sample_action(&self, logits: &Array1<f32>) -> (usize, f32) {
        let probs = self.softmax(logits);
        let mut cumsum = 0.0;
        let rand_val: f32 = rand::random();

        for (i, &prob) in probs.iter().enumerate() {
            cumsum += prob;
            if rand_val < cumsum {
                return (i, prob.max(1e-20).ln());
            }
        }
        let last = probs.len() - 1;
        (last, probs[last].max(1e-20).ln())
    }

    pub fn sample_action_with_temp(&self, logits: &Array1<f32>, temperature: f32) -> (usize, f32) {
        let id = self.sample_top_k(logits, temperature, 20);
        let probs = self.softmax(logits);
        (id, probs[id].max(1e-20).ln())
    }

    fn clip_grad(arr: &mut Array1<f32>, max_norm: f32) {
        let norm = arr.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > max_norm && norm > 0.0 {
            let scale = max_norm / norm;
            arr.mapv_inplace(|v| v * scale);
        }
    }

    fn clip_grad2(arr: &mut Array2<f32>, max_norm: f32) {
        let norm = arr.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > max_norm && norm > 0.0 {
            let scale = max_norm / norm;
            arr.mapv_inplace(|v| v * scale);
        }
    }

    pub fn backward_step(&mut self, tokens: &[usize], learning_rate: f32) -> f32 {
        if tokens.len() < 2 {
            return 0.0;
        }

        let mut h = Array1::<f32>::zeros(self.hidden_dim);
        let mut c = Array1::<f32>::zeros(self.hidden_dim);
        let mut caches: Vec<StepCache> = Vec::with_capacity(tokens.len());

        for &t in tokens {
            let cache = self.step_cached(t, &h, &c);
            h = cache.h.clone();
            c = cache.c.clone();
            caches.push(cache);
        }

        let mut d_embed = Array2::<f32>::zeros((self.vocab_size, self.embed_dim));

        let mut d_w_ii = Array2::<f32>::zeros((self.embed_dim, self.hidden_dim));
        let mut d_w_hi = Array2::<f32>::zeros((self.hidden_dim, self.hidden_dim));
        let mut d_b_i = Array1::<f32>::zeros(self.hidden_dim);

        let mut d_w_if = Array2::<f32>::zeros((self.embed_dim, self.hidden_dim));
        let mut d_w_hf = Array2::<f32>::zeros((self.hidden_dim, self.hidden_dim));
        let mut d_b_f = Array1::<f32>::zeros(self.hidden_dim);

        let mut d_w_io = Array2::<f32>::zeros((self.embed_dim, self.hidden_dim));
        let mut d_w_ho = Array2::<f32>::zeros((self.hidden_dim, self.hidden_dim));
        let mut d_b_o = Array1::<f32>::zeros(self.hidden_dim);

        let mut d_w_ig = Array2::<f32>::zeros((self.embed_dim, self.hidden_dim));
        let mut d_w_hg = Array2::<f32>::zeros((self.hidden_dim, self.hidden_dim));
        let mut d_b_g = Array1::<f32>::zeros(self.hidden_dim);

        let mut d_w_out = Array2::<f32>::zeros((self.hidden_dim, self.vocab_size));
        let mut d_b_out = Array1::<f32>::zeros(self.vocab_size);

        let mut dh_next = Array1::<f32>::zeros(self.hidden_dim);
        let mut dc_next = Array1::<f32>::zeros(self.hidden_dim);

        let mut total_loss = 0.0f32;
        let mut loss_count = 0usize;

        let n = tokens.len();
        for step in (0..n).rev() {
            let cache = &caches[step];

            let mut dh = dh_next.clone();

            if step + 1 < n {
                let target = tokens[step + 1];
                if target < self.vocab_size {
                    let probs = self.softmax(&cache.logits);
                    let p_target = probs[target].max(1e-20);
                    total_loss += -p_target.ln();
                    loss_count += 1;

                    let mut d_logits = probs.clone();
                    d_logits[target] -= 1.0;

                    let h_col = cache.h.view().into_shape((self.hidden_dim, 1)).unwrap();
                    let dl_row = d_logits.view().into_shape((1, self.vocab_size)).unwrap();
                    d_w_out = &d_w_out + &h_col.dot(&dl_row);
                    d_b_out = &d_b_out + &d_logits;

                    let dh_from_out = self.w_out.dot(&d_logits);
                    dh = &dh + &dh_from_out;
                }
            }

            let do_ = &dh * &cache.c_tanh;
            let dc_tanh = &dh * &cache.o;
            let dc = &dc_next + &(&dc_tanh * &cache.c_tanh.map(|v| 1.0 - v * v));

            let di = &dc * &cache.g;
            let df = &dc * &cache.c_prev;
            let dg = &dc * &cache.i;

            let di_pre = &di * &cache.i.map(|v| v * (1.0 - v));
            let df_pre = &df * &cache.f.map(|v| v * (1.0 - v));
            let do_pre = &do_ * &cache.o.map(|v| v * (1.0 - v));
            let dg_pre = &dg * &cache.g.map(|v| 1.0 - v * v);

            let x_col = cache.x.view().into_shape((self.embed_dim, 1)).unwrap();
            let h_prev_col = cache.h_prev.view().into_shape((self.hidden_dim, 1)).unwrap();

            let di_row = di_pre.view().into_shape((1, self.hidden_dim)).unwrap();
            let df_row = df_pre.view().into_shape((1, self.hidden_dim)).unwrap();
            let do_row = do_pre.view().into_shape((1, self.hidden_dim)).unwrap();
            let dg_row = dg_pre.view().into_shape((1, self.hidden_dim)).unwrap();

            d_w_ii = &d_w_ii + &x_col.dot(&di_row);
            d_w_if = &d_w_if + &x_col.dot(&df_row);
            d_w_io = &d_w_io + &x_col.dot(&do_row);
            d_w_ig = &d_w_ig + &x_col.dot(&dg_row);

            d_w_hi = &d_w_hi + &h_prev_col.dot(&di_row);
            d_w_hf = &d_w_hf + &h_prev_col.dot(&df_row);
            d_w_ho = &d_w_ho + &h_prev_col.dot(&do_row);
            d_w_hg = &d_w_hg + &h_prev_col.dot(&dg_row);

            d_b_i = &d_b_i + &di_pre;
            d_b_f = &d_b_f + &df_pre;
            d_b_o = &d_b_o + &do_pre;
            d_b_g = &d_b_g + &dg_pre;

            let dh_prev = &self.w_hi.dot(&di_pre)
                + &self.w_hf.dot(&df_pre)
                + &self.w_ho.dot(&do_pre)
                + &self.w_hg.dot(&dg_pre);

            let dc_prev = &dc * &cache.f;

            let dx = &self.w_ii.dot(&di_pre)
                + &self.w_if.dot(&df_pre)
                + &self.w_io.dot(&do_pre)
                + &self.w_ig.dot(&dg_pre);

            if cache.token_id < self.vocab_size {
                let mut row = d_embed.row_mut(cache.token_id);
                row += &dx;
            }

            dh_next = dh_prev;
            dc_next = dc_prev;
        }

        if loss_count == 0 {
            return 0.0;
        }

        let scale = 1.0 / loss_count as f32;
        d_w_out.mapv_inplace(|v| v * scale);
        d_b_out.mapv_inplace(|v| v * scale);
        d_w_ii.mapv_inplace(|v| v * scale);
        d_w_hi.mapv_inplace(|v| v * scale);
        d_b_i.mapv_inplace(|v| v * scale);
        d_w_if.mapv_inplace(|v| v * scale);
        d_w_hf.mapv_inplace(|v| v * scale);
        d_b_f.mapv_inplace(|v| v * scale);
        d_w_io.mapv_inplace(|v| v * scale);
        d_w_ho.mapv_inplace(|v| v * scale);
        d_b_o.mapv_inplace(|v| v * scale);
        d_w_ig.mapv_inplace(|v| v * scale);
        d_w_hg.mapv_inplace(|v| v * scale);
        d_b_g.mapv_inplace(|v| v * scale);
        d_embed.mapv_inplace(|v| v * scale);

        let clip = 5.0f32;
        Self::clip_grad2(&mut d_w_out, clip);
        Self::clip_grad(&mut d_b_out, clip);
        Self::clip_grad2(&mut d_w_ii, clip);
        Self::clip_grad2(&mut d_w_hi, clip);
        Self::clip_grad(&mut d_b_i, clip);
        Self::clip_grad2(&mut d_w_if, clip);
        Self::clip_grad2(&mut d_w_hf, clip);
        Self::clip_grad(&mut d_b_f, clip);
        Self::clip_grad2(&mut d_w_io, clip);
        Self::clip_grad2(&mut d_w_ho, clip);
        Self::clip_grad(&mut d_b_o, clip);
        Self::clip_grad2(&mut d_w_ig, clip);
        Self::clip_grad2(&mut d_w_hg, clip);
        Self::clip_grad(&mut d_b_g, clip);
        Self::clip_grad2(&mut d_embed, clip);

        let lr = learning_rate;
        self.w_out = &self.w_out - &(&d_w_out * lr);
        self.b_out = &self.b_out - &(&d_b_out * lr);
        self.w_ii = &self.w_ii - &(&d_w_ii * lr);
        self.w_hi = &self.w_hi - &(&d_w_hi * lr);
        self.b_i = &self.b_i - &(&d_b_i * lr);
        self.w_if = &self.w_if - &(&d_w_if * lr);
        self.w_hf = &self.w_hf - &(&d_w_hf * lr);
        self.b_f = &self.b_f - &(&d_b_f * lr);
        self.w_io = &self.w_io - &(&d_w_io * lr);
        self.w_ho = &self.w_ho - &(&d_w_ho * lr);
        self.b_o = &self.b_o - &(&d_b_o * lr);
        self.w_ig = &self.w_ig - &(&d_w_ig * lr);
        self.w_hg = &self.w_hg - &(&d_w_hg * lr);
        self.b_g = &self.b_g - &(&d_b_g * lr);
        self.embed = &self.embed - &(&d_embed * lr);

        let _ = Axis(0);
        let _ = SliceRandom::shuffle as fn(&mut [usize], &mut rand::rngs::ThreadRng);

        total_loss / loss_count as f32
    }

    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        use std::fs;

        let data = serde_json::json!({
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,

            "embed": self.embed.iter().copied().collect::<Vec<_>>(),
            "w_ii": self.w_ii.iter().copied().collect::<Vec<_>>(),
            "w_hi": self.w_hi.iter().copied().collect::<Vec<_>>(),
            "b_i": self.b_i.iter().copied().collect::<Vec<_>>(),
            "w_if": self.w_if.iter().copied().collect::<Vec<_>>(),
            "w_hf": self.w_hf.iter().copied().collect::<Vec<_>>(),
            "b_f": self.b_f.iter().copied().collect::<Vec<_>>(),
            "w_io": self.w_io.iter().copied().collect::<Vec<_>>(),
            "w_ho": self.w_ho.iter().copied().collect::<Vec<_>>(),
            "b_o": self.b_o.iter().copied().collect::<Vec<_>>(),
            "w_ig": self.w_ig.iter().copied().collect::<Vec<_>>(),
            "w_hg": self.w_hg.iter().copied().collect::<Vec<_>>(),
            "b_g": self.b_g.iter().copied().collect::<Vec<_>>(),
            "w_out": self.w_out.iter().copied().collect::<Vec<_>>(),
            "b_out": self.b_out.iter().copied().collect::<Vec<_>>(),
        });

        fs::write(path, serde_json::to_string_pretty(&data)?)?;
        Ok(())
    }

    pub fn load(path: &str, vocab_size: usize, embed_dim: usize, hidden_dim: usize) -> anyhow::Result<Self> {
        use std::fs;

        let content = fs::read_to_string(path)?;
        let data: serde_json::Value = serde_json::from_str(&content)?;

        let mut model = LSTMModel::new(vocab_size, embed_dim, hidden_dim);

        if let Some(embed_data) = data["embed"].as_array() {
            let vec: Vec<f32> = embed_data.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect();
            if vec.len() == vocab_size * embed_dim {
                model.embed = Array2::from_shape_vec((vocab_size, embed_dim), vec)?;
            }
        }

        if let Some(w_ii_data) = data["w_ii"].as_array() {
            let vec: Vec<f32> = w_ii_data.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect();
            if vec.len() == embed_dim * hidden_dim {
                model.w_ii = Array2::from_shape_vec((embed_dim, hidden_dim), vec)?;
            }
        }

        if let Some(w_hi_data) = data["w_hi"].as_array() {
            let vec: Vec<f32> = w_hi_data.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect();
            if vec.len() == hidden_dim * hidden_dim {
                model.w_hi = Array2::from_shape_vec((hidden_dim, hidden_dim), vec)?;
            }
        }

        if let Some(b_i_data) = data["b_i"].as_array() {
            let vec: Vec<f32> = b_i_data.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect();
            model.b_i = Array1::from_vec(vec);
        }

        if let Some(w_if_data) = data["w_if"].as_array() {
            let vec: Vec<f32> = w_if_data.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect();
            if vec.len() == embed_dim * hidden_dim {
                model.w_if = Array2::from_shape_vec((embed_dim, hidden_dim), vec)?;
            }
        }

        if let Some(w_hf_data) = data["w_hf"].as_array() {
            let vec: Vec<f32> = w_hf_data.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect();
            if vec.len() == hidden_dim * hidden_dim {
                model.w_hf = Array2::from_shape_vec((hidden_dim, hidden_dim), vec)?;
            }
        }

        if let Some(b_f_data) = data["b_f"].as_array() {
            let vec: Vec<f32> = b_f_data.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect();
            model.b_f = Array1::from_vec(vec);
        }

        if let Some(w_io_data) = data["w_io"].as_array() {
            let vec: Vec<f32> = w_io_data.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect();
            if vec.len() == embed_dim * hidden_dim {
                model.w_io = Array2::from_shape_vec((embed_dim, hidden_dim), vec)?;
            }
        }

        if let Some(w_ho_data) = data["w_ho"].as_array() {
            let vec: Vec<f32> = w_ho_data.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect();
            if vec.len() == hidden_dim * hidden_dim {
                model.w_ho = Array2::from_shape_vec((hidden_dim, hidden_dim), vec)?;
            }
        }

        if let Some(b_o_data) = data["b_o"].as_array() {
            let vec: Vec<f32> = b_o_data.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect();
            model.b_o = Array1::from_vec(vec);
        }

        if let Some(w_ig_data) = data["w_ig"].as_array() {
            let vec: Vec<f32> = w_ig_data.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect();
            if vec.len() == embed_dim * hidden_dim {
                model.w_ig = Array2::from_shape_vec((embed_dim, hidden_dim), vec)?;
            }
        }

        if let Some(w_hg_data) = data["w_hg"].as_array() {
            let vec: Vec<f32> = w_hg_data.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect();
            if vec.len() == hidden_dim * hidden_dim {
                model.w_hg = Array2::from_shape_vec((hidden_dim, hidden_dim), vec)?;
            }
        }

        if let Some(b_g_data) = data["b_g"].as_array() {
            let vec: Vec<f32> = b_g_data.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect();
            model.b_g = Array1::from_vec(vec);
        }

        if let Some(w_out_data) = data["w_out"].as_array() {
            let vec: Vec<f32> = w_out_data.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect();
            if vec.len() == hidden_dim * vocab_size {
                model.w_out = Array2::from_shape_vec((hidden_dim, vocab_size), vec)?;
            }
        }

        if let Some(b_out_data) = data["b_out"].as_array() {
            let vec: Vec<f32> = b_out_data.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect();
            model.b_out = Array1::from_vec(vec);
        }

        Ok(model)
    }
}
