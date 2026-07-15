#![recursion_limit = "256"]

use aria::transformer_cuda::TransformerModel;
use aria::tokenizer::Tokenizer;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

#[derive(serde::Deserialize)]
struct DialogRecord { text: String }

fn list_jsonl_files(dir: &str) -> anyhow::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let p = entry.path();
        if p.is_file() && p.extension().and_then(|s| s.to_str()) == Some("jsonl") {
            files.push(p);
        }
    }
    files.sort();
    Ok(files)
}

fn feed_tokenizer(data_dir: &str, tokenizer: &mut Tokenizer, max_lines: usize) -> anyhow::Result<()> {
    let files = list_jsonl_files(data_dir)?;
    if files.is_empty() {
        anyhow::bail!("No .jsonl files found in {}", data_dir);
    }
    println!("Found {} .jsonl file(s) for vocab:", files.len());
    for p in &files { println!("  - {}", p.display()); }

    const CHUNK: usize = 50_000;
    let mut batch: Vec<String> = Vec::with_capacity(CHUNK);
    let mut count = 0usize;

    'outer: for path in &files {
        let f = File::open(path)?;
        let r = BufReader::new(f);
        for line in r.lines() {
            if count >= max_lines { break 'outer; }
            let line = line?;
            if line.trim().is_empty() { continue; }
            let rec: DialogRecord = match serde_json::from_str(&line) {
                Ok(r) => r,
                Err(_) => continue,
            };
            batch.push(rec.text);
            count += 1;

            if batch.len() >= CHUNK {
                tokenizer.feed_batch(&batch);
                batch.clear();
                if count % 200_000 == 0 {
                    println!("  vocab: processed {} records", count);
                }
            }
        }
    }
    if !batch.is_empty() {
        tokenizer.feed_batch(&batch);
    }
    println!("  vocab: total records processed: {}", count);
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let data_dir = "data base";
    let checkpoint_path = "aria json/aria_checkpoint.gguf";
    let tokenizer_path = "aria json/aria_tokenizer.json";

    // Transformer hyperparams (250M params, ARIA Medium)
    let d_model    = 896;
    let num_heads  = 14;
    let num_layers = 20;
    let ffn_dim    = 3584;
    let max_seq    = 512;

    let vocab_lines: usize = std::env::var("ARIA_VOCAB_LINES")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(2_000_000);

    println!("Building vocabulary from dialog file (max {} records)...", vocab_lines);
    let mut tokenizer = Tokenizer::new();
    feed_tokenizer(data_dir, &mut tokenizer, vocab_lines)?;
    tokenizer.freeze();
    let vocab_size = tokenizer.vocab_size();
    println!("Vocab size: {}\n", vocab_size);

    // Remove stale cache
    let cache_path = format!("{}/sequences_cache_transformer_v{}_len{}.bin",
                             data_dir, vocab_size, max_seq);
    if std::path::Path::new(&cache_path).exists() {
        println!("Removing stale cache: {}", cache_path);
        std::fs::remove_file(&cache_path)?;
    }

    println!("Initializing fresh Transformer model...");
    let mut model = TransformerModel::new(vocab_size, d_model, num_heads, num_layers, ffn_dim, max_seq);

    println!("Starting supervised dialog training...");
    aria::transformer_cuda::pretrain_from_files(&mut model, &mut tokenizer, data_dir, checkpoint_path)?;

    println!("\nSaving final checkpoint...");
    model.save_checkpoint(checkpoint_path, &tokenizer)?;

    println!("Done.");
    Ok(())
}
