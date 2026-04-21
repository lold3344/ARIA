use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use rand_distr::Normal;

pub struct Model {
    pub embed: Array2<f32>,
    pub w1: Array2<f32>,
    pub b1: Array1<f32>,
    pub w2: Array2<f32>,
    pub b2: Array1<f32>,
    pub policy_head: Array2<f32>,
    pub value_head: Array2<f32>,
}

impl Model {
    pub fn new(vocab_size: usize, embed_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        let scale = 1.0 / (embed_dim as f32).sqrt();
        let normal = Normal::new(0.0, scale).unwrap();
        
        Model {
            embed: Array2::random((vocab_size, embed_dim), normal),
            w1: Array2::random((embed_dim, hidden_dim), Normal::new(0.0, 1.0 / (embed_dim as f32).sqrt()).unwrap()),
            b1: Array1::zeros(hidden_dim),
            w2: Array2::random((hidden_dim, output_dim), Normal::new(0.0, 1.0 / (hidden_dim as f32).sqrt()).unwrap()),
            b2: Array1::zeros(output_dim),
            policy_head: Array2::random((output_dim, vocab_size), Normal::new(0.0, 1.0 / (output_dim as f32).sqrt()).unwrap()),
            value_head: Array2::random((output_dim, 1), Normal::new(0.0, 1.0 / (output_dim as f32).sqrt()).unwrap()),
        }
    }

    pub fn forward(&self, input: &[usize]) -> (Array1<f32>, f32) {
        let mut context = Array1::zeros(self.embed.ncols());
        
        for &token_id in input {
            if token_id < self.embed.nrows() {
                context = context + &self.embed.row(token_id).to_owned();
            }
        }
        context = &context / (input.len() as f32);

        let hidden = context.dot(&self.w1);
        let hidden = &hidden + &self.b1;
        let hidden = self.relu(&hidden);

        let output = hidden.dot(&self.w2);
        let output = &output + &self.b2;
        let output = self.relu(&output);

        let logits = output.dot(&self.policy_head);
        let value = output.dot(&self.value_head);

        (logits, value[0])
    }

    pub fn relu(&self, x: &Array1<f32>) -> Array1<f32> {
        x.map(|a| a.max(0.0))
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

    pub fn get_action_log_prob(&self, logits: &Array1<f32>, action: usize) -> f32 {
        let probs = self.softmax(logits);
        if action < probs.len() {
            probs[action].ln().max(-20.0)
        } else {
            -20.0
        }
    }
}
