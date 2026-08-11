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

fn prompt_dataset_selection(files: &[std::path::PathBuf]) -> anyhow::Result<Vec<std::path::PathBuf>> {
    use std::io::{self, Write};

    if files.is_empty() {
        anyhow::bail!("No .jsonl files found");
    }

    println!("Found {} .jsonl file(s):", files.len());
    for (i, p) in files.iter().enumerate() {
        println!("  {:2} - {}", i + 1, p.display());
    }
    println!("\nChoose datasets to use:");
    println!("  - enter numbers separated by spaces (e.g. \"1 3 7\")");
    println!("  - or type \"all\" to use every file");
    print!("> ");
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let input = input.trim();

    if input.eq_ignore_ascii_case("all") || input.is_empty() {
        return Ok(files.to_vec());
    }

    let mut selected = Vec::new();
    for token in input.split_whitespace() {
        let idx: usize = token.parse()
            .map_err(|_| anyhow::anyhow!("'{}' is not a valid number", token))?;
        if idx == 0 || idx > files.len() {
            anyhow::bail!("number {} out of range (1..{})", idx, files.len());
        }
        selected.push(files[idx - 1].clone());
    }

    if selected.is_empty() {
        anyhow::bail!("No datasets selected");
    }

    Ok(selected)
}

fn main() -> anyhow::Result<()> {
    let data_dir = "data base";
    let checkpoint_path = "aria json/aria_checkpoint.gguf";
    // Tokenizer is now embedded in the GGUF checkpoint; no separate JSON file needed.

    // Transformer hyperparams (250M params, ARIA Medium)
    let d_model    = 896;
    let num_heads  = 14;
    let num_layers = 20;
    let ffn_dim    = 3584;
    let max_seq    = 512;

    let vocab_lines: usize = std::env::var("ARIA_VOCAB_LINES")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(2_000_000);

    println!("[train_debug] Building vocabulary from dialog file (max {} records)...", vocab_lines);
    let mut tokenizer = Tokenizer::new();
    feed_tokenizer(data_dir, &mut tokenizer, vocab_lines)?;
    tokenizer.freeze();
    let vocab_size = tokenizer.vocab_size();
    println!("Vocab size: {}\n", vocab_size);

    // List and choose datasets
    let all_files = list_jsonl_files(data_dir)?;
    let selected_files = prompt_dataset_selection(&all_files)?;
    println!("Selected {} dataset(s):", selected_files.len());
    for p in &selected_files { println!("  - {}", p.display()); }
    println!();

    // Build cache while GPU memory is still free.
    let max_seqs: Option<usize> = std::env::var("ARIA_MAX_SEQS")
        .ok().and_then(|s| s.parse().ok());
    let (_, _, n) = aria::transformer_cuda::prepare_seq_cache(
        &mut tokenizer, data_dir, &selected_files, max_seq, 2, max_seqs
    )?;
    println!("[train_debug] Cache ready: {} sequences\n", n);

    println!("[train_debug] Initializing fresh Transformer model...");
    let mut model = TransformerModel::new(vocab_size, d_model, num_heads, num_layers, ffn_dim, max_seq);

    println!("[train_debug] Starting supervised dialog training with gradient diagnostics...");
    aria::transformer_cuda::pretrain_from_files(
        &mut model, &mut tokenizer, data_dir, &selected_files, checkpoint_path
    )?;

    println!("\n[train_debug] Saving final checkpoint...");
    model.save_checkpoint(checkpoint_path, &tokenizer)?;

    println!("[train_debug] Done.");
    Ok(())
}
