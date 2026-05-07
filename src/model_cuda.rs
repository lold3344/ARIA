use std::fs;
use std::path::Path;
use std::time::Instant;
use std::thread;
use rayon::prelude::*;
use rand::seq::SliceRandom;
use tch::{Device, Tensor, Kind, nn};

use crate::tokenizer::Tokenizer;
use crate::lstm_cuda::LSTMCuda;

// =============================================================================
// LSTM STATE
// =============================================================================

#[derive(Clone)]
pub struct LSTMState {
    pub h: Vec<f32>,
    pub c: Vec<f32>,
}

// =============================================================================
// LSTM MODEL (CUDA / GPU)
// =============================================================================

pub struct LSTMModelCuda {
    vs: nn::VarStore,
    embed: Tensor,
    w_x: Tensor,
    w_h: Tensor,
    b: Tensor,
    w_out: Tensor,
    b_out: Tensor,

    m_embed: Tensor, v_embed: Tensor,
    m_w_x: Tensor, v_w_x: Tensor,
    m_w_h: Tensor, v_w_h: Tensor,
    m_b: Tensor, v_b: Tensor,
    m_w_out: Tensor, v_w_out: Tensor,
    m_b_out: Tensor, v_b_out: Tensor,

    pub vocab_size: usize,
    pub embed_dim: usize,
    pub hidden_dim: usize,
    adam_step: i64,
    cuda: Option<LSTMCuda>,
}

impl LSTMModelCuda {
    pub fn new(vocab_size: usize, embed_dim: usize, hidden_dim: usize) -> Self {
        let cuda = LSTMCuda::try_init();
        let device = cuda.as_ref()
            .map(|_| Device::Cuda(0))
            .unwrap_or(Device::Cpu);

        println!("\n================================");
        println!("           ARIA v8 CUDA           ");
        println!("================================");
        println!("  Vocabulary size:    {}", vocab_size);
        println!("  Embedding dim:      {}", embed_dim);
        println!("  Hidden dim:         {}", hidden_dim);
        println!("  Device:             {:?}", device);
        println!("  CUDA enabled:       {}", cuda.is_some());
        println!("================================\n");

        let vs = nn::VarStore::new(device);
        let fh = 4 * hidden_dim;
        let scale_e = 1.0 / (embed_dim as f64).sqrt();
        let scale_h = 1.0 / (hidden_dim as f64).sqrt();

        // Use VarStore.get with initializer closure
        let embed = vs.get("embed", &[vocab_size as i64, embed_dim as i64], |shape, _| {
            Tensor::randn(shape, (Kind::Float, device)) * scale_e
        });
        let w_x = vs.get("w_x", &[embed_dim as i64, fh as i64], |shape, _| {
            Tensor::randn(shape, (Kind::Float, device)) * scale_e
        });
        let w_h = vs.get("w_h", &[hidden_dim as i64, fh as i64], |shape, _| {
            Tensor::randn(shape, (Kind::Float, device)) * scale_h
        });
        let mut b = vs.get("b", &[fh as i64], |shape, _| {
            Tensor::zeros(shape, (Kind::Float, device))
        });
        b.narrow(0, hidden_dim as i64, hidden_dim as i64).fill_(1.0);
        let w_out = vs.get("w_out", &[hidden_dim as i64, vocab_size as i64], |shape, _| {
            Tensor::randn(shape, (Kind::Float, device)) * scale_h
        });
        let b_out = vs.get("b_out", &[vocab_size as i64], |shape, _| {
            Tensor::zeros(shape, (Kind::Float, device))
        });

        let zeros_like = |t: &Tensor| Tensor::zeros_like(t);

        LSTMModelCuda {
            vs,
            embed, w_x, w_h, b, w_out, b_out,
            m_embed: zeros_like(&embed), v_embed: zeros_like(&embed),
            m_w_x: zeros_like(&w_x), v_w_x: zeros_like(&w_x),
            m_w_h: zeros_like(&w_h), v_w_h: zeros_like(&w_h),
            m_b: zeros_like(&b), v_b: zeros_like(&b),
            m_w_out: zeros_like(&w_out), v_w_out: zeros_like(&w_out),
            m_b_out: zeros_like(&b_out), v_b_out: zeros_like(&b_out),
            vocab_size, embed_dim, hidden_dim,
            adam_step: 0,
            cuda,
        }
    }

    pub fn init_state(&self) -> LSTMState {
        LSTMState {
            h: vec![0.0; self.hidden_dim],
            c: vec![0.0; self.hidden_dim],
        }
    }

    fn embed_lookup(&self, token_ids: &[usize]) -> Tensor {
        let indices: Vec<i64> = token_ids.iter().map(|&i| i as i64).collect();
        let indices_t = Tensor::f_from_slice(&indices).unwrap().reshape([-1]);
        self.embed.index_select(0, &indices_t)
    }

    fn lstm_step(&self, x: &Tensor, h: &Tensor, c: &Tensor) -> (Tensor, Tensor) {
        let gates_x = x.matmul(&self.w_x);
        let gates_h = h.matmul(&self.w_h);
        let gates = &gates_x + &gates_h + &self.b;

        let hd = self.hidden_dim as i64;
        let i = gates.narrow(1, 0, hd).sigmoid();
        let f = gates.narrow(1, hd, hd).sigmoid();
        let o = gates.narrow(1, 2 * hd, hd).sigmoid();
        let g = gates.narrow(1, 3 * hd, hd).tanh();

        let c_new = &f * c + &i * &g;
        let h_new = &o * c_new.tanh();

        (h_new, c_new)
    }

    fn tensor_to_vec_f32(t: &Tensor) -> Vec<f32> {
        let cpu = t.to(Device::Cpu);
        let n = cpu.numel() as usize;
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            result.push(cpu.double_value(&[i as i64]) as f32);
        }
        result
    }

    pub fn forward_seq(&self, tokens: &[usize]) -> (Vec<f32>, LSTMState) {
        let batch = tokens.len();
        if batch == 0 {
            return (vec![0.0; self.vocab_size], self.init_state());
        }

        let device = self.vs.device();
        let mut h = Tensor::zeros(&[batch as i64, self.hidden_dim as i64], (Kind::Float, device));
        let mut c = Tensor::zeros(&[batch as i64, self.hidden_dim as i64], (Kind::Float, device));

        for &token in tokens {
            let x = self.embed_lookup(&[token]);
            let (h_new, c_new) = self.lstm_step(&x, &h, &c);
            h = h_new;
            c = c_new;
        }

        let logits = h.matmul(&self.w_out) + &self.b_out;

        (Self::tensor_to_vec_f32(&logits), LSTMState {
            h: Self::tensor_to_vec_f32(&h),
            c: Self::tensor_to_vec_f32(&c),
        })
    }

    pub fn step(&self, token_id: usize, state: &LSTMState) -> (Vec<f32>, LSTMState) {
        let device = self.vs.device();

        let h = Tensor::f_from_slice(&state.h).unwrap().reshape([1, self.hidden_dim as i64]).to_device(device);
        let c = Tensor::f_from_slice(&state.c).unwrap().reshape([1, self.hidden_dim as i64]).to_device(device);

        let x = self.embed_lookup(&[token_id]);
        let (h_new, c_new) = self.lstm_step(&x, &h, &c);

        let logits = h_new.matmul(&self.w_out) + &self.b_out;

        (Self::tensor_to_vec_f32(&logits), LSTMState {
            h: Self::tensor_to_vec_f32(&h_new),
            c: Self::tensor_to_vec_f32(&c_new),
        })
    }

    pub fn train_batch(&mut self, sequences: &[Vec<usize>], learning_rate: f64) -> f32 {
        if sequences.is_empty() || sequences.iter().all(|s| s.is_empty()) {
            return 0.0;
        }

        let valid_seqs: Vec<_> = sequences.iter().filter(|s| s.len() >= 2).collect();
        if valid_seqs.is_empty() {
            return 0.0;
        }

        let batch = valid_seqs.len();
        let max_len = valid_seqs.iter().map(|s| s.len()).max().unwrap_or(2);

        let mut input_ids = vec![0i64; batch * max_len];
        let mut target_ids = vec![0i64; batch * max_len];
        let mut mask = vec![0.0f64; batch * max_len];

        for (b, seq) in valid_seqs.iter().enumerate() {
            for (t, &tk) in seq.iter().enumerate() {
                if t < max_len - 1 {
                    input_ids[b * max_len + t] = tk as i64;
                    target_ids[b * max_len + t + 1] = tk as i64;
                    mask[b * max_len + t] = 1.0;
                }
            }
        }

        let device = self.vs.device();
        let input_t = Tensor::f_from_slice(&input_ids).unwrap().reshape([batch as i64, max_len as i64]).to_device(device);
        let target_t = Tensor::f_from_slice(&target_ids).unwrap().reshape([batch as i64, max_len as i64]).to_device(device);
        let mask_t = Tensor::f_from_slice(&mask).unwrap().reshape([batch as i64, max_len as i64]).to_device(device);

        let mut h = Tensor::zeros(&[batch as i64, self.hidden_dim as i64], (Kind::Float, device)).set_requires_grad(true);
        let mut c = Tensor::zeros(&[batch as i64, self.hidden_dim as i64], (Kind::Float, device)).set_requires_grad(true);

        let mut total_loss = Tensor::zeros(&[1], (Kind::Float, device));

        for step in 0..max_len.saturating_sub(1) {
            let t_i64 = step as i64;
            let tok = input_t.narrow(1, t_i64, 1).squeeze();
            let x = self.embed_lookup(&tensor_to_usize(&tok));

            let (h_new, c_new) = self.lstm_step(&x, &h, &c);
            let logits = h_new.matmul(&self.w_out) + &self.b_out;

            let target = target_t.narrow(1, t_i64 + 1, 1).squeeze();
            let m = mask_t.narrow(1, t_i64, 1).squeeze();
            let loss_step = logits.cross_entropy_for_logits(&target);
            let loss_masked = (loss_step * m).sum(Kind::Float);

            total_loss = total_loss + loss_masked;
            h = h_new.set_requires_grad(true);
            c = c_new.set_requires_grad(true);
        }

        let avg_loss = total_loss / (batch as f64);
        avg_loss.backward();

        // Clip and update
        self.clip_gradients(5.0);
        self.adam_update(learning_rate);
        self.clear_gradients();

        avg_loss.double_value(&[0]) as f32
    }

    fn clip_gradients(&mut self, max_norm: f32) {
        let g = self.embed.grad();
        let grad_norm = g.norm().double_value(&[0]) as f32;
        if grad_norm > max_norm {
            let scale = (max_norm / grad_norm) as f64;
            g.set_(g.copy() * scale);
        }
    }

    fn adam_update(&mut self, lr: f64) {
        self.adam_step += 1;
        let t = self.adam_step as i32;
        let b1: f64 = 0.9;
        let b2: f64 = 0.999;
        let eps: f64 = 1e-8;
        let bc1 = 1.0 - b1.powi(t);
        let bc2 = 1.0 - b2.powi(t);
        let lr_scaled = lr * bc2.sqrt() / bc1;

        self.adam_update_tensor(&mut self.embed, &mut self.m_embed, &mut self.v_embed, lr_scaled, b1, b2, eps);
        self.adam_update_tensor(&mut self.w_x, &mut self.m_w_x, &mut self.v_w_x, lr_scaled, b1, b2, eps);
        self.adam_update_tensor(&mut self.w_h, &mut self.m_w_h, &mut self.v_w_h, lr_scaled, b1, b2, eps);
        self.adam_update_tensor(&mut self.b, &mut self.m_b, &mut self.v_b, lr_scaled, b1, b2, eps);
        self.adam_update_tensor(&mut self.w_out, &mut self.m_w_out, &mut self.v_w_out, lr_scaled, b1, b2, eps);
        self.adam_update_tensor(&mut self.b_out, &mut self.m_b_out, &mut self.v_b_out, lr_scaled, b1, b2, eps);
    }

    fn adam_update_tensor(&self, param: &mut Tensor, m: &mut Tensor, v: &mut Tensor, lr: f64, b1: f64, b2: f64, eps: f64) {
        let g = param.grad();
        let g_copy = g.copy();
        let m_copy = m.copy();
        let v_copy = v.copy();

        let m_new = &m_copy * b1 + &g_copy * (1.0 - b1);
        let v_new = &v_copy * b2 + &(&g_copy * &g_copy) * (1.0 - b2);
        let denom = v_new.sqrt() + eps;
        let update = &m_new / denom;
        let param_new = param.copy() - update * lr;

        *param = param_new;
        *m = m_new;
        *v = v_new;
    }

    fn clear_gradients(&mut self) {
        self.embed.zero_grad();
        self.w_x.zero_grad();
        self.w_h.zero_grad();
        self.b.zero_grad();
        self.w_out.zero_grad();
        self.b_out.zero_grad();
    }

    pub fn sample_greedy(&self, logits: &[f32]) -> usize {
        logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    pub fn sample_top_k(&self, logits: &[f32], temperature: f32, top_k: usize) -> usize {
        let temp = temperature.max(0.05);
        let scaled: Vec<f32> = logits.iter().map(|x| x / temp).collect();
        let mut indexed: Vec<(usize, f32)> = scaled.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let k = top_k.min(indexed.len()).max(1);
        let top: Vec<(usize, f32)> = indexed.into_iter().take(k).collect();
        let mx = top[0].1;
        let exps: Vec<f32> = top.iter().map(|(_, v)| (v - mx).exp()).collect();
        let sum: f32 = exps.iter().sum();

        if sum <= 0.0 || !sum.is_finite() {
            return top[0].0;
        }

        let r: f32 = rand::random();
        let mut cum = 0.0f32;
        for (idx, e) in exps.iter().enumerate() {
            cum += e / sum;
            if r < cum {
                return top[idx].0;
            }
        }
        top[k - 1].0
    }

    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let embed_vec = Self::tensor_to_vec_f32(&self.embed);
        let w_x_vec = Self::tensor_to_vec_f32(&self.w_x);
        let w_h_vec = Self::tensor_to_vec_f32(&self.w_h);
        let b_vec = Self::tensor_to_vec_f32(&self.b);
        let w_out_vec = Self::tensor_to_vec_f32(&self.w_out);
        let b_out_vec = Self::tensor_to_vec_f32(&self.b_out);

        let data = serde_json::json!({
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "format": "v8_cuda",
            "embed": embed_vec,
            "w_x": w_x_vec,
            "w_h": w_h_vec,
            "b": b_vec,
            "w_out": w_out_vec,
            "b_out": b_out_vec,
        });

        std::fs::write(path, serde_json::to_string_pretty(&data)?)?;
        Ok(())
    }

    pub fn load(path: &str, vocab_size: usize, embed_dim: usize, hidden_dim: usize) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let data: serde_json::Value = serde_json::from_str(&content)?;

        let mut model = LSTMModelCuda::new(vocab_size, embed_dim, hidden_dim);
        let device = model.vs.device();

        macro_rules! load {
            ($field:ident, $key:expr) => {
                if let Some(arr) = data[$key].as_array() {
                    let v: Vec<f32> = arr.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect();
                    if v.len() == model.$field.numel() as usize {
                        let t = Tensor::f_from_slice(&v).unwrap().reshape(model.$field.f_shape().unwrap()).to_device(device);
                        model.$field.f_copy_(&t).unwrap();
                    }
                }
            };
        }

        load!(embed, "embed");
        load!(w_x, "w_x");
        load!(w_h, "w_h");
        load!(b, "b");
        load!(w_out, "w_out");
        load!(b_out, "b_out");

        Ok(model)
    }

    pub fn backward_step(&mut self, tokens: &[usize], learning_rate: f64) -> f32 {
        self.train_batch(&[tokens.to_vec()], learning_rate)
    }
}

// Helper: extract 1D tensor as Vec<usize>
fn tensor_to_usize(t: &Tensor) -> Vec<usize> {
    let cpu = t.to(Device::Cpu);
    let n = cpu.numel() as usize;
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        result.push(cpu.int64_value(&[i as i64]) as usize);
    }
    result
}

// =============================================================================
// PRETRAINING
// =============================================================================

const LEARNING_RATE: f64 = 0.001;
const MAX_TOKENS_PER_SEQ: usize = 80;
const MIN_TOKENS_PER_SEQ: usize = 4;
const PRETRAIN_EPOCHS: usize = 10;
const PRETRAIN_BATCH_SIZE: usize = 256;

pub fn pretrain_from_files(
    model: &mut LSTMModelCuda,
    tokenizer: &mut Tokenizer,
    data_dir: &str,
) -> anyhow::Result<()> {
    let path = Path::new(data_dir);
    if !path.exists() {
        println!("Data directory not found: {}", data_dir);
        return Ok(());
    }

    let start_time = Instant::now();
    let num_cpus = thread::available_parallelism().map(|n| n.get()).unwrap_or(8);

    println!("\n===============================================");
    println!("       ARIA - MAX PERFORMANCE TRAINING        ");
    println!("===============================================");
    println!("Learning rate:   {}", LEARNING_RATE);
    println!("Epochs:          {}", PRETRAIN_EPOCHS);
    println!("Batch size:      {}", PRETRAIN_BATCH_SIZE);
    println!("CPU threads:     {}", num_cpus);
    println!("Max seq length:  {}", MAX_TOKENS_PER_SEQ);
    println!("Optimizer:       Adam");
    println!("===============================================\n");

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus)
        .build_global()
        .ok();

    println!("Stage 1: Loading text files (parallel)...");
    let read_start = Instant::now();

    let file_contents: Vec<(String, String)> = fs::read_dir(path)?
        .par_bridge()
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let file_path = entry.path();
            if file_path.extension().map_or(false, |ext| ext == "txt") {
                if let Ok(content) = fs::read_to_string(&file_path) {
                    if !content.trim().is_empty() {
                        let filename = file_path.file_name()?.to_string_lossy().to_string();
                        return Some((filename, content));
                    }
                }
            }
            None
        })
        .collect();

    let read_time = read_start.elapsed();

    if file_contents.is_empty() {
        println!("No text files found.");
        return Ok(());
    }

    println!("Loaded {} files in {:.3}s\n", file_contents.len(), read_time.as_secs_f32());

    println!("Stage 2: Tokenizing (parallel)...");
    let process_start = Instant::now();

    let all_sentences: Vec<Vec<String>> = file_contents
        .par_iter()
        .map(|(_, content)| {
            content
                .split(|c| c == '.' || c == '\n' || c == '!' || c == '?')
                .map(|s| s.trim().to_string())
                .filter(|s| s.len() > 5)
                .collect()
        })
        .collect();

    let mut all_sequences: Vec<Vec<usize>> = Vec::new();

    for sentences in all_sentences {
        for sentence in sentences {
            let tokens = tokenizer.encode(&sentence);
            if tokens.len() >= MIN_TOKENS_PER_SEQ && tokens.len() <= MAX_TOKENS_PER_SEQ {
                all_sequences.push(tokens);
            }
        }
    }

    let process_time = process_start.elapsed();
    let total_tokens: usize = all_sequences.iter().map(|s| s.len()).sum();

    println!("{} sequences, {} tokens in {:.3}s\n", all_sequences.len(), total_tokens, process_time.as_secs_f32());

    if all_sequences.is_empty() {
        println!("No usable sequences.");
        return Ok(());
    }

    println!("Stage 3: Training on GPU (CUDA)...");
    let num_batches = (all_sequences.len() + PRETRAIN_BATCH_SIZE - 1) / PRETRAIN_BATCH_SIZE;
    println!("{} batches per epoch\n", num_batches);

    let training_start = Instant::now();

    for epoch in 0..PRETRAIN_EPOCHS {
        let epoch_start = Instant::now();

        let mut shuffled = all_sequences.clone();
        shuffled.shuffle(&mut rand::thread_rng());

        let mut total_loss = 0.0f32;
        let mut batch_count = 0usize;

        for chunk in shuffled.chunks(PRETRAIN_BATCH_SIZE) {
            let batch: Vec<Vec<usize>> = chunk.to_vec();
            let loss = model.train_batch(&batch, LEARNING_RATE);
            if loss.is_finite() && loss.is_normal() {
                total_loss += loss;
                batch_count += 1;
            }
        }

        let avg = total_loss / batch_count.max(1) as f32;
        let et = epoch_start.elapsed();
        let speed = all_sequences.len() as f32 / et.as_secs_f32();
        println!("Epoch {}/{}: loss={:.6}  {:.1}s  {:.0} seq/s", epoch + 1, PRETRAIN_EPOCHS, avg, et.as_secs_f32(), speed);
    }

    let training_time = training_start.elapsed();
    let total_time = start_time.elapsed();

    println!("\n===============================================");
    println!("TRAINING COMPLETE");
    println!("  I/O:       {:.3}s", read_time.as_secs_f32());
    println!("  Process:   {:.3}s", process_time.as_secs_f32());
    println!("  Training:  {:.3}s", training_time.as_secs_f32());
    println!("  Total:     {:.3}s", total_time.as_secs_f32());
    println!("  Sequences: {}", all_sequences.len());
    println!("  Tokens:    {}", total_tokens);
    println!("===============================================\n");

    Ok(())
}