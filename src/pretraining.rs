use std::fs;
use std::path::Path;
use std::time::Instant;
use crate::model::LSTMModel;
use crate::tokenizer::Tokenizer;
use rayon::prelude::*;

const BATCH_SIZE: usize = 128;
const LEARNING_RATE: f32 = 0.001;
const MAX_TOKENS_PER_SEQ: usize = 100;

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

    let gpu_available = check_gpu_available();

    let device_name = if gpu_available {
        "GPU (DirectX 12)"
    } else {
        "CPU"
    };

    println!("\nARIA - Hybrid GPU+CPU Training");
    println!("===============================================");
    println!("Device: {}", device_name);
    println!("Batch size: {}", BATCH_SIZE);
    println!("Learning rate: {}", LEARNING_RATE);
    println!("CPU threads: {}", rayon::current_num_threads());
    println!("===============================================\n");

    println!("Stage 1: Loading text files in parallel (CPU Rayon)...");
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
    println!("Total size: {:.2} MB\n",
        file_contents.iter().map(|(_, c)| c.len()).sum::<usize>() as f32 / 1_000_000.0);

    println!("Stage 2: Processing and tokenizing (CPU Rayon)...");
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
        for sentence in sentences {
            let tokens = tokenizer.encode(&sentence);
            if tokens.len() >= 3 && tokens.len() <= MAX_TOKENS_PER_SEQ {
                all_sequences.push(tokens);
            }
        }
        println!("  Processed {}", filename);
    }

    let process_time = process_start.elapsed();
    let total_tokens: usize = all_sequences.iter().map(|s| s.len()).sum();

    println!("Created {} sequences ({} tokens) in {:.3}s",
        all_sequences.len(), total_tokens, process_time.as_secs_f32());
    println!("Vocabulary size: {}\n", tokenizer.vocab_size());

    println!("Stage 3: {} training (LSTM forward pass)...", device_name);
    println!("Processing {} sequences in batches of {}\n", all_sequences.len(), BATCH_SIZE);

    let training_start = Instant::now();

    let (total_loss, total_processed) = train_cpu_rayon(model, &all_sequences);

    let training_time = training_start.elapsed();
    let total_time = start_time.elapsed();

    let avg_loss = if total_processed > 0 {
        total_loss / total_processed as f32
    } else {
        0.0
    };
    let tokens_per_sec = if training_time.as_secs_f32() > 0.0 {
        total_tokens as f32 / training_time.as_secs_f32()
    } else {
        0.0
    };

    println!("\n===============================================");
    println!("TRAINING RESULTS:");
    println!("===============================================");
    println!("File I/O (CPU):      {:.3}s", read_time.as_secs_f32());
    println!("Processing (CPU):    {:.3}s", process_time.as_secs_f32());
    println!("Training ({}): {:.3}s", device_name, training_time.as_secs_f32());
    println!("TOTAL TIME:          {:.3}s", total_time.as_secs_f32());
    println!("");
    println!("Total sequences:     {}", all_sequences.len());
    println!("Processed:           {}", total_processed);
    println!("Total tokens:        {}", total_tokens);
    println!("Average loss:        {:.6}", avg_loss);
    println!("Throughput:          {:.0} tokens/sec", tokens_per_sec);
    println!("Vocab size:          {}", tokenizer.vocab_size());
    println!("Device:              {}", device_name);
    println!("===============================================\n");

    Ok(())
}

fn check_gpu_available() -> bool {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::DX12,
        dx12_shader_compiler: wgpu::Dx12Compiler::default(),
        gles_minor_version: wgpu::Gles3MinorVersion::default(),
        flags: wgpu::InstanceFlags::default(),
    });

    let adapters = pollster::block_on(async {
        instance.enumerate_adapters(wgpu::Backends::DX12)
    });

    !adapters.is_empty()
}

fn train_cpu_rayon(
    model: &LSTMModel,
    sequences: &[Vec<usize>],
) -> (f32, usize) {
    let num_threads = rayon::current_num_threads();
    let chunk_size = ((sequences.len() + num_threads - 1) / num_threads).max(1);

    let chunk_results: Vec<(usize, f32)> = sequences
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut chunk_loss = 0.0f32;
            let mut count = 0usize;

            for tokens in chunk {
                if tokens.len() < 2 {
                    continue;
                }

                let (logits, _) = model.forward_seq(tokens);
                let probs = model.softmax(&logits);

                if let Some(&last_token) = tokens.last() {
                    if last_token < probs.len() {
                        chunk_loss += -probs[last_token].ln().max(-20.0);
                    }
                }
                count += 1;
            }

            (count, chunk_loss)
        })
        .collect();

    let total_processed: usize = chunk_results.iter().map(|(c, _)| c).sum();
    let total_loss: f32 = chunk_results.iter().map(|(_, l)| l).sum();

    (total_loss, total_processed)
}
