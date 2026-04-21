use ndarray::Array1;
use crate::model::Model;

pub struct PPOTrainer {
    pub learning_rate: f32,
    pub gamma: f32,
    pub gae_lambda: f32,
    pub clip_ratio: f32,
    pub entropy_coef: f32,
}

#[derive(Clone)]
pub struct Experience {
    pub tokens: Vec<usize>,
    pub action: usize,
    pub log_prob: f32,
    pub reward: f32,
    pub done: bool,
}

impl PPOTrainer {
    pub fn new() -> Self {
        PPOTrainer {
            learning_rate: 0.0003,
            gamma: 0.99,
            gae_lambda: 0.95,
            clip_ratio: 0.2,
            entropy_coef: 0.01,
        }
    }

    pub fn update(&self, model: &mut Model, batch: &[Experience]) {
        if batch.is_empty() {
            return;
        }

        let mut returns = vec![0.0; batch.len()];
        let mut value = 0.0;

        for i in (0..batch.len()).rev() {
            value = batch[i].reward + self.gamma * value * (if batch[i].done { 0.0 } else { 1.0 });
            returns[i] = value;
        }

        let mean_return = returns.iter().sum::<f32>() / returns.len() as f32;
        let _variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f32>() / returns.len() as f32;

        for exp in batch.iter() {
            let (logits, state_value) = model.forward(&exp.tokens);
            
            let new_log_prob = model.get_action_log_prob(&logits, exp.action);
            let ratio = (new_log_prob - exp.log_prob).exp();
            
            let advantage = (exp.reward - state_value).max(-1.0).min(1.0);
            
            let policy_loss = -(ratio * advantage).min(
                ((1.0 + self.clip_ratio) * advantage).min(
                    (1.0 - self.clip_ratio) * advantage
                )
            );

            let value_loss = (state_value - exp.reward).powi(2);
            
            let probs = model.softmax(&logits);
            let entropy = -probs.iter()
                .filter(|&&p| p > 1e-8)
                .map(|&p| p * p.ln())
                .sum::<f32>();

            let total_loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy;

            let lr_scaled = self.learning_rate / (batch.len() as f32);
            self.apply_gradient(model, total_loss, lr_scaled);
        }
    }

    fn apply_gradient(&self, model: &mut Model, loss: f32, lr: f32) {
        let dloss = loss * lr;
        
        for layer in model.policy_head.iter_mut() {
            *layer -= dloss * 0.1;
        }
        
        for layer in model.value_head.iter_mut() {
            *layer -= dloss * 0.1;
        }

        for layer in model.w2.iter_mut() {
            *layer -= dloss * 0.01;
        }

        for layer in model.b2.iter_mut() {
            *layer -= dloss * 0.01;
        }
    }
}
