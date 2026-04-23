use tch::{nn, Tensor, Device};
use tch::nn::RNN;

pub struct LSTMModel {
    pub vs: nn::VarStore,
    pub embed: nn::Embedding,
    pub lstm: nn::LSTM,
    pub linear: nn::Linear,
    pub hidden: i64,
}

impl LSTMModel {
    pub fn new(vocab: i64, embed_dim: i64, hidden: i64) -> Self {
        let vs = nn::VarStore::new(Device::cuda_if_available());
        let root = &vs.root();

        let embed = nn::embedding(root / "embed", vocab, embed_dim, Default::default());
        let lstm = nn::lstm(root / "lstm", embed_dim, hidden, Default::default());
        let linear = nn::linear(root / "linear", hidden, vocab, Default::default());

        Self { vs, embed, lstm, linear, hidden }
    }

    pub fn forward_seq(&self, input: &Tensor) -> (Tensor, nn::LSTMState) {
        let emb = input.apply(&self.embed);
        let (out, state) = self.lstm.seq(&emb);
        let logits: Tensor = out.apply(&self.linear);
        let last = logits.select(1, logits.size()[1] - 1);
        (last, state)
    }

    pub fn step(&self, input: i64, state: Option<nn::LSTMState>) -> (Tensor, nn::LSTMState) {
        let x = Tensor::from_slice(&[input]).view([1, 1]);
        let emb = x.apply(&self.embed);

        let (out, new_state) = match state {
            Some(s) => self.lstm.seq_init(&emb, &s),
            None => self.lstm.seq(&emb),
        };

        let logits: Tensor = out.apply(&self.linear);
        let last = logits.select(1, logits.size()[1] - 1);

        (last, new_state)
    }

    pub fn sample(&self, logits: &Tensor) -> i64 {
        logits.softmax(-1, tch::Kind::Float)
            .multinomial(1, true)
            .int64_value(&[0])
    }

    pub fn save(&self, path: &str) {
        let _ = self.vs.save(path);
    }

    pub fn load(path: &str, vocab: i64, embed: i64, hidden: i64) -> Self {
        let mut m = Self::new(vocab, embed, hidden);
        let _ = m.vs.load(path);
        m
    }
}

fn main() {}