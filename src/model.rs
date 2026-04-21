use ndarray::{Array1, Array2};
use rand::thread_rng;
use rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
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
        let mut rng = thread_rng();
        let scale = 1.0 / (embed_dim as f32).sqrt();
        
        Model {
            embed: Array2::random_using(&mut rng, (vocab_size, embed_dim), StandardNormal)
                * scale,
            w1: Array2::random_using(&mut rng, (embed_dim, hidden_dim), StandardNormal)
                * (1.0 / (embed_dim as f32).sqrt()),
            b1: Array1::zeros(hidden_dim),
            w2: Array2::random_using(&mut rng, (hidden_dim, output_dim), StandardNormal)
                * (1.0 / (hidden_dim as f32).sqrt()),
            b2: Array1::zeros(output_dim),
            policy_head: Array2::random_using(&mut rng, (output_dim, vocab_size), StandardNormal)
                * (1.0 / (output_dim as f32).sqrt()),
            value_head: Array2::random_using(&mut rng, (output_dim, 1), StandardNormal)
                * (1.0 / (output_dim as f32).sqrt()),
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

        (logits, value[[0, 0]])
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
                return (i, prob.ln());
            }
        }
        
        (probs.len() - 1, probs[probs.len() - 1].ln())
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
