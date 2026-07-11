#![recursion_limit = "256"]

use aria::transformer_cuda::TransformerModel;
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

fn greedy_generate(model: &TransformerModel, tokenizer: &mut Tokenizer, prompt: &str, max_tokens: usize) -> String {
    let ids = tokenizer.encode_prompt(prompt);
    let (mut logits, mut kv) = model.forward_seq(&ids);
    let mut generated = vec![];
    for _ in 0..max_tokens {
        tokenizer.mask_logits(&mut logits);
        let best = model.sample_greedy(&logits);
        if best == 0 || best == 3 || best >= tokenizer.vocab_size() { break; }
        generated.push(best);
        let (nl, nkv) = model.step(best, &kv);
        kv = nkv;
        logits = nl;
    }
    tokenizer.decode(&generated).trim().to_string()
}

fn main() -> anyhow::Result<()> {
    let model_path = "aria json/aria_checkpoint.gguf";
    let data_path = "data base/DataBase_roles.jsonl";

    let (mut model, mut tokenizer) = TransformerModel::load_checkpoint(model_path)?;

    println!("Loading tiny subset...");
    let (mut seqs, mut masks) = load_tiny_subset(data_path, 50_000, &mut tokenizer)?;
    println!("Loaded {} sequences", seqs.len());

    let val_count = (seqs.len() / 20).max(1).min(seqs.len() / 10);
    let mut train_seqs = seqs.split_off(val_count);
    let mut train_masks = masks.split_off(val_count);
    let val_seqs = seqs;
    let val_masks = masks;

    let prompts = [
        "привет",
        "как дела",
        "сколько будет 1 плюс 1",
        "что ты умеешь",
        "расскажи о себе",
    ];

    let batch_size = 64usize;
    let lr = 0.0003f32;
    let total_steps = 100usize;
    let batches_per_step = 10usize;

    println!("\nStarting tiny fine-tuning: {} train / {} val / {} steps * {} batches",
        train_seqs.len(), val_seqs.len(), total_steps, batches_per_step);
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
    model.save_checkpoint("aria json/aria_checkpoint_tiny.gguf", &tokenizer)?;
    Ok(())
}
