#![recursion_limit = "256"]

use aria::model_cuda::LSTMModelCuda;
use aria::tokenizer::Tokenizer;
use std::fs::File;
use std::io::{BufRead, BufReader};
use rand::seq::SliceRandom;

#[derive(serde::Deserialize)]
struct DialogRecord { text: String }

fn load_tiny_subset(path: &str, max_records: usize, tokenizer: &mut Tokenizer) -> anyhow::Result<(Vec<Vec<usize>>, Vec<Vec<f32>>)> {
    let f = File::open(path)?;
    let r = BufReader::new(f);
    let mut seqs = vec![];
    let mut masks = vec![];
    for line in r.lines() {
        let line = line?;
        if line.trim().is_empty() { continue; }
        let rec: DialogRecord = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let (s, m) = tokenizer.encode_dialog(&rec.text);
        if s.len() < 6 { continue; }
        seqs.push(s);
        masks.push(m);
        if seqs.len() >= max_records { break; }
        if seqs.len() % 10000 == 0 {
            println!("  loaded {} records", seqs.len());
        }
    }
    Ok((seqs, masks))
}

fn greedy_generate(model: &LSTMModelCuda, tokenizer: &mut Tokenizer, prompt: &str, max_tokens: usize) -> String {
    let ids = tokenizer.encode(prompt);
    let input = &ids[..ids.len().saturating_sub(1)];
    let (mut logits, mut state) = model.forward_seq(input);
    let mut generated = vec![];
    for _ in 0..max_tokens {
        let mut masked = logits.clone();
        tokenizer.mask_logits(&mut masked);
        let mut best = 0usize;
        let mut best_val = masked[0];
        for (i, &v) in masked.iter().enumerate() {
            if v > best_val { best = i; best_val = v; }
        }
        if best == 0 || best == 3 || best >= tokenizer.vocab_size() { break; }
        generated.push(best);
        let (nl, ns) = model.step(best, &state);
        state = ns;
        logits = nl;
    }
    tokenizer.decode(&generated).trim().to_string()
}

fn main() -> anyhow::Result<()> {
    let model_path = "aria json/aria_checkpoint.json";
    let tokenizer_path = "aria json/aria_tokenizer.json";
    let data_path = "data base/DataBase_roles.jsonl";

    let mut tokenizer = Tokenizer::load(tokenizer_path)?;
    let mut model = LSTMModelCuda::load_checkpoint(model_path)?;

    println!("Loading tiny subset...");
    let (mut seqs, mut masks) = load_tiny_subset(data_path, 50_000, &mut tokenizer)?;
    println!("Loaded {} sequences", seqs.len());

    let val_count = (seqs.len() / 20).max(1).min(seqs.len() / 10);
    let train_count = seqs.len() - val_count;
    let mut train_seqs = seqs.split_off(val_count);
    let mut train_masks = masks.split_off(val_count);
    let val_seqs = seqs;
    let val_masks = masks;

    let prompts = [
        "Пользователь: привет\nАссистент:",
        "Пользователь: как дела\nАссистент:",
        "Пользователь: сколько будет 1 плюс 1\nАссистент:",
        "Пользователь: что ты умеешь\nАссистент:",
        "Пользователь: расскажи о себе\nАссистент:",
    ];

    let batch_size = 128;
    let lr = 0.0003f64;
    let total_steps = 100usize;
    let batches_per_step = 10usize;

    println!("\nStarting tiny supervised fine-tuning: {} train / {} val / {} steps * {} batches",
        train_count, val_count, total_steps, batches_per_step);
    let mut rng = rand::thread_rng();
    for step in 0..total_steps {
        train_seqs.shuffle(&mut rng);
        train_masks.shuffle(&mut rng);

        let mut step_loss = 0.0f32;
        let mut step_batches = 0usize;
        for (s, m) in train_seqs.chunks(batch_size).zip(train_masks.chunks(batch_size)).take(batches_per_step) {
            let s: Vec<Vec<usize>> = s.iter().cloned().collect();
            let m: Vec<Vec<f32>> = m.iter().cloned().collect();
            let loss = model.train_batch_masked(&s, &m, lr);
            if loss.is_finite() { step_loss += loss; step_batches += 1; }
        }

        if step % 10 == 0 || step == total_steps - 1 {
            let avg = step_loss / step_batches.max(1) as f32;
            let mut val_loss = 0.0f32;
            let mut val_batches = 0usize;
            for (s, m) in val_seqs.chunks(batch_size).zip(val_masks.chunks(batch_size)) {
                let s: Vec<Vec<usize>> = s.iter().cloned().collect();
                let m: Vec<Vec<f32>> = m.iter().cloned().collect();
                let loss = model.train_batch_masked(&s, &m, 0.0);
                if loss.is_finite() { val_loss += loss; val_batches += 1; }
            }
            let val_avg = val_loss / val_batches.max(1) as f32;
            println!("\n[step {}] train_loss={:.4}  val_loss={:.4}", step, avg, val_avg);
            for prompt in &prompts {
                let out = greedy_generate(&model, &mut tokenizer, prompt, 30);
                println!("  [{}] -> {}", prompt, out);
            }
        }
    }

    println!("\nSaving checkpoint...");
    model.save_checkpoint("aria json/aria_checkpoint_tiny.json")?;
    Ok(())
}
