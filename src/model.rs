use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use rand_distr::Normal;

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
}

#[derive(Clone)]
pub struct LSTMState {
    pub h: Array1<f32>,
    pub c: Array1<f32>,
}

impl LSTMModel {
    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        use std::fs;
        
        let data = serde_json::json!({
            "embed": self.embed.to_shape_vec().unwrap().1,
            "w_ii": self.w_ii.to_shape_vec().unwrap().1,
            "w_hi": self.w_hi.to_shape_vec().unwrap().1,
            "b_i": self.b_i.to_vec(),
            "w_if": self.w_if.to_shape_vec().unwrap().1,
            "w_hf": self.w_hf.to_shape_vec().unwrap().1,
            "b_f": self.b_f.to_vec(),
            "w_io": self.w_io.to_shape_vec().unwrap().1,
            "w_ho": self.w_ho.to_shape_vec().unwrap().1,
            "b_o": self.b_o.to_vec(),
            "w_ig": self.w_ig.to_shape_vec().unwrap().1,
            "w_hg": self.w_hg.to_shape_vec().unwrap().1,
            "b_g": self.b_g.to_vec(),
            "w_out": self.w_out.to_shape_vec().unwrap().1,
            "b_out": self.b_out.to_vec(),
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
            model.embed = ndarray::Array2::from_shape_vec((vocab_size, embed_dim), vec)?;
        }
        
        if let Some(w_ii_data) = data["w_ii"].as_array() {
            let vec: Vec<f32> = w_ii_data.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect();
            model.w_ii = ndarray::Array2::from_shape_vec((embed_dim, hidden_dim), vec)?;
        }
        
        if let Some(w_hi_data) = data["w_hi"].as_array() {
            let vec: Vec<f32> = w_hi_data.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect();
            model.w_hi = ndarray::Array2::from_shape_vec((hidden_dim, hidden_dim), vec)?;
        }
        
        if let Some(b_i_data) = data["b_i"].as_array() {
            let vec: Vec<f32> = b_i_data.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect();
            model.b_i = ndarray::Array1::from_vec(vec);
        }
        
        if let Some(w_if_data) = data["w_if"].as_array() {
            let vec: Vec<f32> = w_if_data.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect();
            model.w_if = ndarray::Array2::from_shape_vec((embed_dim, hidden_dim), vec)?;
        }
        
        if let Some(w_hf_data) = data["w_hf"].as_array() {
            let vec: Vec<f32> = w_hf_data.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect();
            model.w_hf = ndarray::Array2::from_shape_vec((hidden_dim, hidden_dim), vec)?;
        }
        
        if let Some(b_f_data) = data["b_f"].as_array() {
            let vec: Vec<f32> = b_f_data.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect();
            model.b_f = ndarray::Array1::from_vec(vec);
        }
        
        if let Some(w_io_data) = data["w_io"].as_array() {
            let vec: Vec<f32> = w_io_data.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect();
            model.w_io = ndarray::Array2::from_shape_vec((embed_dim, hidden_dim), vec)?;
        }
        
        if let Some(w_ho_data) = data["w_ho"].as_array() {
            let vec: Vec<f32> = w_ho_data.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect();
            model.w_ho = ndarray::Array2::from_shape_vec((hidden_dim, hidden_dim), vec)?;
        }
        
        if let Some(b_o_data) = data["b_o"].as_array() {
            let vec: Vec<f32> = b_o_data.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect();
            model.b_o = ndarray::Array1::from_vec(vec);
        }
        
        if let Some(w_ig_data) = data["w_ig"].as_array() {
            let vec: Vec<f32> = w_ig_data.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect();
            model.w_ig = ndarray::Array2::from_shape_vec((embed_dim, hidden_dim), vec)?;
        }
        
        if let Some(w_hg_data) = data["w_hg"].as_array() {
            let vec: Vec<f32> = w_hg_data.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect();
            model.w_hg = ndarray::Array2::from_shape_vec((hidden_dim, hidden_dim), vec)?;
        }
        
        if let Some(b_g_data) = data["b_g"].as_array() {
            let vec: Vec<f32> = b_g_data.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect();
            model.b_g = ndarray::Array1::from_vec(vec);
        }
        
        if let Some(w_out_data) = data["w_out"].as_array() {
            let vec: Vec<f32> = w_out_data.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect();
            model.w_out = ndarray::Array2::from_shape_vec((hidden_dim, vocab_size), vec)?;
        }
        
        if let Some(b_out_data) = data["b_out"].as_array() {
            let vec: Vec<f32> = b_out_data.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect();
            model.b_out = ndarray::Array1::from_vec(vec);
        }
        
        Ok(model)
    }

    pub fn new(vocab_size: usize, embed_dim: usize, hidden_dim: usize) -> Self {
        let scale_embed = 1.0 / (embed_dim as f32).sqrt();
        let scale_hidden = 1.0 / (hidden_dim as f32).sqrt();
        
        LSTMModel {
            embed: Array2::random((vocab_size, embed_dim), Normal::new(0.0, scale_embed).unwrap()),
            
            w_ii: Array2::random((embed_dim, hidden_dim), Normal::new(0.0, scale_hidden).unwrap()),
            w_hi: Array2::random((hidden_dim, hidden_dim), Normal::new(0.0, scale_hidden).unwrap()),
            b_i: Array1::zeros(hidden_dim),
            
            w_if: Array2::random((embed_dim, hidden_dim), Normal::new(0.0, scale_hidden).unwrap()),
            w_hf: Array2::random((hidden_dim, hidden_dim), Normal::new(0.0, scale_hidden).unwrap()),
            b_f: Array1::zeros(hidden_dim),
            
            w_io: Array2::random((embed_dim, hidden_dim), Normal::new(0.0, scale_hidden).unwrap()),
            w_ho: Array2::random((hidden_dim, hidden_dim), Normal::new(0.0, scale_hidden).unwrap()),
            b_o: Array1::zeros(hidden_dim),
            
            w_ig: Array2::random((embed_dim, hidden_dim), Normal::new(0.0, scale_hidden).unwrap()),
            w_hg: Array2::random((hidden_dim, hidden_dim), Normal::new(0.0, scale_hidden).unwrap()),
            b_g: Array1::zeros(hidden_dim),
            
            w_out: Array2::random((hidden_dim, vocab_size), Normal::new(0.0, scale_hidden).unwrap()),
            b_out: Array1::zeros(vocab_size),
        }
    }

    pub fn init_state(&self, hidden_dim: usize) -> LSTMState {
        LSTMState {
            h: Array1::zeros(hidden_dim),
            c: Array1::zeros(hidden_dim),
        }
    }

    pub fn step(&self, token_id: usize, state: &LSTMState) -> (Array1<f32>, LSTMState) {
        if token_id >= self.embed.nrows() {
            let logits = Array1::zeros(self.b_out.len());
            return (logits, state.clone());
        }

        let x = self.embed.row(token_id).to_owned();

        let i = self.sigmoid(&(&x.dot(&self.w_ii) + &state.h.dot(&self.w_hi) + &self.b_i));
        let f = self.sigmoid(&(&x.dot(&self.w_if) + &state.h.dot(&self.w_hf) + &self.b_f));
        let o = self.sigmoid(&(&x.dot(&self.w_io) + &state.h.dot(&self.w_ho) + &self.b_o));
        let g = self.tanh(&(&x.dot(&self.w_ig) + &state.h.dot(&self.w_hg) + &self.b_g));

        let c_new = &(&f * &state.c) + &(&i * &g);
        let h_new = &o * &self.tanh(&c_new);

        let logits = &h_new.dot(&self.w_out) + &self.b_out;

        (logits, LSTMState { h: h_new, c: c_new })
    }

    pub fn forward_seq(&self, tokens: &[usize]) -> (Array1<f32>, LSTMState) {
        let mut state = self.init_state(self.w_hi.ncols());
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

        for (i, &prob) in probs.iter().enumerate() {
            cumsum += prob;
            if rand_val < cumsum {
                return (i, prob.ln().max(-20.0));
            }
        }

        (probs.len() - 1, probs[probs.len() - 1].ln().max(-20.0))
    }

    fn sigmoid(&self, x: &Array1<f32>) -> Array1<f32> {
        x.map(|a| 1.0 / (1.0 + (-a).exp()))
    }

    fn tanh(&self, x: &Array1<f32>) -> Array1<f32> {
        x.map(|a| a.tanh())
    }
}
