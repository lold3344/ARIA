use std::fs;
use std::path::Path;
use std::time::Instant;
use crate::model::LSTMModel;
use crate::tokenizer::Tokenizer;
use rayon::prelude::*;
use rand::seq::SliceRandom;

const LEARNING_RATE: f32 = 0.05;
const MAX_TOKENS_PER_SEQ: usize = 80;
const MIN_TOKENS_PER_SEQ: usize = 4;
const EPOCHS: usize = 5;

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

    println!("\nARIA - Training with Next-Token Prediction");
    println!("===============================================");
    println!("Learning rate:   {}", LEARNING_RATE);
    println!("Epochs:          {}", EPOCHS);
    println!("Max seq length:  {}", MAX_TOKENS_PER_SEQ);
    println!("Loss:            Cross Entropy with Softmax");
    println!("Optimizer:       SGD with gradient clipping");
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
        println!("No text files found. Pre-training skipped.");
        return Ok(());
    }

    println!("Loaded {} files in {:.3}s", file_contents.len(), read_time.as_secs_f32());
    println!(
        "Total size: {:.2} MB\n",
        file_contents.iter().map(|(_, c)| c.len()).sum::<usize>() as f32 / 1_000_000.0
    );

    println!("Stage 2: Splitting into sentences and tokenizing...");
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
        let mut count_for_file = 0usize;
        for sentence in sentences {
            let tokens = tokenizer.encode(&sentence);
            if tokens.len() >= MIN_TOKENS_PER_SEQ && tokens.len() <= MAX_TOKENS_PER_SEQ {
                all_sequences.push(tokens);
                count_for_file += 1;
            }
        }
        println!("  Processed {} ({} sequences)", filename, count_for_file);
    }

    let process_time = process_start.elapsed();
    let total_tokens: usize = all_sequences.iter().map(|s| s.len()).sum();

    println!(
        "\nCreated {} sequences ({} tokens) in {:.3}s",
        all_sequences.len(),
        total_tokens,
        process_time.as_secs_f32()
    );
    println!("Vocabulary size: {}\n", tokenizer.vocab_size());

    if all_sequences.is_empty() {
        println!("No usable sequences. Skipping training.");
        return Ok(());
    }

    println!("Stage 3: Training (BPTT, full LSTM gradient)...");
    println!("Sequences per epoch: {}\n", all_sequences.len());

    let training_start = Instant::now();

    for epoch in 0..EPOCHS {
        let mut shuffled = all_sequences.clone();
        shuffled.shuffle(&mut rand::thread_rng());

        let mut epoch_loss = 0.0f32;
        let mut count = 0usize;

        let report_every = (shuffled.len() / 10).max(1);

        for (idx, tokens) in shuffled.iter().enumerate() {
            if tokens.len() >= 2 {
                let loss = model.backward_step(tokens, LEARNING_RATE);
                if loss.is_finite() {
                    epoch_loss += loss;
                    count += 1;
                }
            }

            if idx > 0 && idx % report_every == 0 {
                let avg = if count > 0 { epoch_loss / count as f32 } else { 0.0 };
                println!(
                    "  epoch {}/{}  step {}/{}  avg_loss = {:.4}",
                    epoch + 1, EPOCHS, idx, shuffled.len(), avg
                );
            }
        }

        if count > 0 {
            let avg_loss = epoch_loss / count as f32;
            println!("Epoch {}/{} complete: avg loss = {:.6}\n", epoch + 1, EPOCHS, avg_loss);
        }
    }

    let training_time = training_start.elapsed();
    let total_time = start_time.elapsed();

    println!("===============================================");
    println!("TRAINING RESULTS:");
    println!("===============================================");
    println!("File I/O:            {:.3}s", read_time.as_secs_f32());
    println!("Processing:          {:.3}s", process_time.as_secs_f32());
    println!("Training:            {:.3}s", training_time.as_secs_f32());
    println!("TOTAL TIME:          {:.3}s", total_time.as_secs_f32());
    println!("");
    println!("Total sequences:     {}", all_sequences.len());
    println!("Epochs:              {}", EPOCHS);
    println!("Total tokens:        {}", total_tokens);
    println!("Learning method:     Next-token prediction with Cross Entropy + BPTT");
    println!("===============================================\n");

    Ok(())
}
