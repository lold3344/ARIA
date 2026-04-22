use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use rand_distr::Normal;
use std::sync::Arc;

pub struct LSTMModel {
    pub embed: Arc<Array2<f32>>,
    
    pub w_ii: Arc<Array2<f32>>,
    pub w_hi: Arc<Array2<f32>>,
    pub b_i: Arc<Array1<f32>>,
    
    pub w_if: Arc<Array2<f32>>,
    pub w_hf: Arc<Array2<f32>>,
    pub b_f: Arc<Array1<f32>>,
    
    pub w_io: Arc<Array2<f32>>,
    pub w_ho: Arc<Array2<f32>>,
    pub b_o: Arc<Array1<f32>>,
    
    pub w_ig: Arc<Array2<f32>>,
    pub w_hg: Arc<Array2<f32>>,
    pub b_g: Arc<Array1<f32>>,
    
    pub w_out: Arc<Array2<f32>>,
    pub b_out: Arc<Array1<f32>>,
    
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub hidden_dim: usize,
}

#[derive(Clone)]
pub struct LSTMState {
    pub h: Array1<f32>,
    pub c: Array1<f32>,
}

impl LSTMModel {
    pub fn new(vocab_size: usize, embed_dim: usize, hidden_dim: usize) -> Self {
        let scale_embed = 1.0 / (embed_dim as f32).sqrt();
        let scale_hidden = 1.0 / (hidden_dim as f32).sqrt();
        
        LSTMModel {
            embed: Arc::new(Array2::random((vocab_size, embed_dim), Normal::new(0.0, scale_embed).unwrap())),
            
            w_ii: Arc::new(Array2::random((embed_dim, hidden_dim), Normal::new(0.0, scale_hidden).unwrap())),
            w_hi: Arc::new(Array2::random((hidden_dim, hidden_dim), Normal::new(0.0, scale_hidden).unwrap())),
            b_i: Arc::new(Array1::zeros(hidden_dim)),
            
            w_if: Arc::new(Array2::random((embed_dim, hidden_dim), Normal::new(0.0, scale_hidden).unwrap())),
            w_hf: Arc::new(Array2::random((hidden_dim, hidden_dim), Normal::new(0.0, scale_hidden).unwrap())),
            b_f: Arc::new(Array1::zeros(hidden_dim)),
            
            w_io: Arc::new(Array2::random((embed_dim, hidden_dim), Normal::new(0.0, scale_hidden).unwrap())),
            w_ho: Arc::new(Array2::random((hidden_dim, hidden_dim), Normal::new(0.0, scale_hidden).unwrap())),
            b_o: Arc::new(Array1::zeros(hidden_dim)),
            
            w_ig: Arc::new(Array2::random((embed_dim, hidden_dim), Normal::new(0.0, scale_hidden).unwrap())),
            w_hg: Arc::new(Array2::random((hidden_dim, hidden_dim), Normal::new(0.0, scale_hidden).unwrap())),
            b_g: Arc::new(Array1::zeros(hidden_dim)),
            
            w_out: Arc::new(Array2::random((hidden_dim, vocab_size), Normal::new(0.0, scale_hidden).unwrap())),
            b_out: Arc::new(Array1::zeros(vocab_size)),
            
            vocab_size,
            embed_dim,
            hidden_dim,
        }
    }

    pub fn init_state(&self) -> LSTMState {
        LSTMState {
            h: Array1::zeros(self.hidden_dim),
            c: Array1::zeros(self.hidden_dim),
        }
    }

    pub fn step(&self, token_id: usize, state: &LSTMState) -> (Array1<f32>, LSTMState) {
        if token_id >= self.embed.nrows() {
            return (Array1::zeros(self.b_out.len()), state.clone());
        }

        let x = self.embed.row(token_id).to_owned();

        let i = self.sigmoid(&(x.dot(&*self.w_ii) + state.h.dot(&*self.w_hi) + &*self.b_i));
        let f = self.sigmoid(&(x.dot(&*self.w_if) + state.h.dot(&*self.w_hf) + &*self.b_f));
        let o = self.sigmoid(&(x.dot(&*self.w_io) + state.h.dot(&*self.w_ho) + &*self.b_o));
        let g = self.tanh(&(x.dot(&*self.w_ig) + state.h.dot(&*self.w_hg) + &*self.b_g));

        let c_new = &(&f * &state.c) + &(&i * &g);
        let h_new = &o * &self.tanh(&c_new);

        let logits = h_new.dot(&*self.w_out) + &*self.b_out;

        (logits, LSTMState { h: h_new, c: c_new })
    }

    pub fn forward_seq(&self, tokens: &[usize]) -> (Array1<f32>, LSTMState) {
        let mut state = self.init_state();
        let mut logits = Array1::zeros(self.b_out.len());

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
        &exp / sum
    }

    pub fn sample_action(&self, logits: &Array1<f32>) -> (usize, f32) {
        let probs = self.softmax(logits);
        let mut cumsum = 0.0;
        let rand_val: f32 = rand::random();

        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if rand_val < cumsum {
                return (i, p.ln().max(-20.0));
            }
        }

        let last = probs.len() - 1;
        (last, probs[last].ln().max(-20.0))
    }

    fn sigmoid(&self, x: &Array1<f32>) -> Array1<f32> {
        x.map(|a| 1.0 / (1.0 + (-a).exp()))
    }

    fn tanh(&self, x: &Array1<f32>) -> Array1<f32> {
        x.map(|a| a.tanh())
    }

    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        use std::fs;
        
        let data = serde_json::json!({
            "embed": self.embed.as_ref().clone().into_shape((self.vocab_size * self.embed_dim,)).unwrap().to_vec(),
            "w_out": self.w_out.as_ref().clone().into_shape((self.hidden_dim * self.vocab_size,)).unwrap().to_vec(),
            "b_out": self.b_out.as_ref().to_vec(),
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
            model.embed = Arc::new(Array2::from_shape_vec((vocab_size, embed_dim), vec)?);
        }
        
        if let Some(w_out_data) = data["w_out"].as_array() {
            let vec: Vec<f32> = w_out_data.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect();
            model.w_out = Arc::new(Array2::from_shape_vec((hidden_dim, vocab_size), vec)?);
        }
        
        if let Some(b_out_data) = data["b_out"].as_array() {
            let vec: Vec<f32> = b_out_data.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect();
            model.b_out = Arc::new(Array1::from_vec(vec));
        }
        
        Ok(model)
    }
}