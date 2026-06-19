#![recursion_limit = "256"]

use aria::transformer_cuda::TransformerModel;
use aria::tokenizer::Tokenizer;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(serde::Deserialize)]
struct DialogRecord { text: String }

fn feed_tokenizer(path: &str, tokenizer: &mut Tokenizer, max_lines: usize) -> anyhow::Result<()> {
    let f = File::open(path)?;
    let r = BufReader::new(f);
    let mut count = 0usize;
    for line in r.lines().take(max_lines) {
        let line = line?;
        if line.trim().is_empty() { continue; }
        let rec: DialogRecord = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(_) => continue,
        };
        tokenizer.encode(&rec.text);
        count += 1;
        if count % 100_000 == 0 {
            println!("  vocab: processed {} records", count);
        }
    }
    println!("  vocab: total records processed: {}", count);
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let data_dir = "data base";
    let dialog_path = "data base/DataBase_roles.jsonl";
    let checkpoint_path = "aria json/aria_checkpoint.json";
    let tokenizer_path = "aria json/aria_tokenizer.json";

    // Transformer hyperparams
    let d_model    = 768;
    let num_heads  = 12;
    let num_layers = 12;
    let ffn_dim    = 3072;
    let max_seq    = 256;

    let vocab_lines: usize = std::env::var("ARIA_VOCAB_LINES")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(2_000_000);

    println!("Building vocabulary from dialog file (max {} records)...", vocab_lines);
    let mut tokenizer = Tokenizer::new();
    feed_tokenizer(dialog_path, &mut tokenizer, vocab_lines)?;
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
    aria::transformer_cuda::pretrain_from_files(&mut model, &mut tokenizer, data_dir, checkpoint_path, tokenizer_path)?;

    println!("\nSaving final checkpoint...");
    model.save_checkpoint(checkpoint_path)?;
    tokenizer.save(tokenizer_path)?;

    println!("Done.");
    Ok(())
}
