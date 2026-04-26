use std::fs;
use std::path::Path;
use std::time::Instant;
use crate::model::LSTMModel;
use crate::tokenizer::Tokenizer;
use rayon::prelude::*;
use rand::seq::SliceRandom;

const LEARNING_RATE: f32 = 0.001;
const MAX_TOKENS_PER_SEQ: usize = 80;
const MIN_TOKENS_PER_SEQ: usize = 4;
const EPOCHS: usize = 5;
const BATCH_SIZE: usize = 32;

pub fn pretrain_from_files(
    model: &mut LSTMModel,
    tokenizer: &mut Tokenizer,
    data_dir: &str,
) -> anyhow::Result<()> {
    let path = Path::new(data_dir);
    if !path.exists() {
        println!("Data directory not found: {}", data_dir);
        return Ok(());
    }

    let start_time = Instant::now();

    println!("\nARIA - Batch Training");
    println!("===============================================");
    println!("Learning rate:   {}", LEARNING_RATE);
    println!("Epochs:          {}", EPOCHS);
    println!("Batch size:      {}", BATCH_SIZE);
    println!("Max seq length:  {}", MAX_TOKENS_PER_SEQ);
    println!("Optimizer:       Adam");
    println!("===============================================\n");

    println!("Stage 1: Loading text files...");
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

    println!("Loaded {} files in {:.3}s", file_contents.len(), read_time.as_secs_f32());
    println!(
        "Total: {:.2} MB\n",
        file_contents.iter().map(|(_, c)| c.len()).sum::<usize>() as f32 / 1_000_000.0
    );

    println!("Stage 2: Tokenizing...");
    let process_start = Instant::now();

    let all_sentences: Vec<(String, Vec<String>)> = file_contents
        .par_iter()
        .map(|(filename, content)| {
            let sentences: Vec<String> = content
                .split(|c| c == '.' || c == '\n' || c == '!' || c == '?')
                .map(|s| s.trim().to_string())
                .filter(|s| s.len() > 5)
                .collect();
            (filename.clone(), sentences)
        })
        .collect();

    let mut all_sequences: Vec<Vec<usize>> = Vec::new();

    for (filename, sentences) in all_sentences {
        let mut cnt = 0usize;
        for sentence in sentences {
            let tokens = tokenizer.encode(&sentence);
            if tokens.len() >= MIN_TOKENS_PER_SEQ && tokens.len() <= MAX_TOKENS_PER_SEQ {
                all_sequences.push(tokens);
                cnt += 1;
            }
        }
        println!("  {} ({} seqs)", filename, cnt);
    }

    let process_time = process_start.elapsed();
    let total_tokens: usize = all_sequences.iter().map(|s| s.len()).sum();

    println!(
        "\n{} sequences, {} tokens in {:.3}s",
        all_sequences.len(), total_tokens, process_time.as_secs_f32()
    );
    println!("Vocabulary: {}\n", tokenizer.vocab_size());

    if all_sequences.is_empty() {
        println!("No usable sequences.");
        return Ok(());
    }

    println!("Stage 3: Training...");
    let num_batches = (all_sequences.len() + BATCH_SIZE - 1) / BATCH_SIZE;
    println!("{} batches per epoch\n", num_batches);

    let training_start = Instant::now();

    for epoch in 0..EPOCHS {
        let epoch_start = Instant::now();
        let mut shuffled = all_sequences.clone();
        shuffled.shuffle(&mut rand::thread_rng());

        let mut epoch_loss = 0.0f32;
        let mut batch_count = 0usize;
        let report_every = (num_batches / 5).max(1);

        for chunk in shuffled.chunks(BATCH_SIZE) {
            let batch: Vec<Vec<usize>> = chunk.to_vec();
            let loss = model.train_batch(&batch, LEARNING_RATE);
            if loss.is_finite() { epoch_loss += loss; batch_count += 1; }

            if batch_count > 0 && batch_count % report_every == 0 {
                let avg = epoch_loss / batch_count as f32;
                let elapsed = epoch_start.elapsed().as_secs_f32();
                let speed = (batch_count * BATCH_SIZE) as f32 / elapsed;
                println!(
                    "  epoch {}/{}  batch {}/{}  loss={:.4}  {:.0} seq/s",
                    epoch + 1, EPOCHS, batch_count, num_batches, avg, speed
                );
            }
        }

        let et = epoch_start.elapsed();
        if batch_count > 0 {
            let avg = epoch_loss / batch_count as f32;
            let speed = all_sequences.len() as f32 / et.as_secs_f32();
            println!("Epoch {}/{}: loss={:.6}  {:.1}s  {:.0} seq/s\n", epoch + 1, EPOCHS, avg, et.as_secs_f32(), speed);
        }
    }

    let training_time = training_start.elapsed();
    let total_time = start_time.elapsed();

    println!("===============================================");
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
