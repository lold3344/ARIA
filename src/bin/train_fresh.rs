#![recursion_limit = "256"]

use aria::model_cuda::LSTMModelCuda;
use aria::tokenizer::Tokenizer;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(serde::Deserialize)]
struct DialogRecord { text: String }

fn collect_text_for_tokenizer(path: &str) -> anyhow::Result<String> {
    let f = File::open(path)?;
    let r = BufReader::new(f);
    let mut out = String::new();
    for line in r.lines().take(2_000_000) {
        let line = line?;
        if line.trim().is_empty() { continue; }
        let rec: DialogRecord = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(_) => continue,
        };
        out.push_str(&rec.text);
        out.push(' ');
    }
    Ok(out)
}

fn main() -> anyhow::Result<()> {
    let data_dir = "data base";
    let dialog_path = "data base/DataBase_roles.jsonl";
    let checkpoint_path = "aria json/aria_checkpoint.json";
    let tokenizer_path = "aria json/aria_tokenizer.json";

    let embed_dim = 1024;
    let hidden_dim = 2048;

    println!("Building vocabulary from dialog file...");
    let mut tokenizer = Tokenizer::new();
    let text = collect_text_for_tokenizer(dialog_path)?;
    tokenizer.encode(&text);  // accumulate word frequencies
    tokenizer.freeze();
    let vocab_size = tokenizer.vocab_size();
    println!("Vocab size: {}\n", vocab_size);

    println!("Initializing fresh model...");
    let mut model = LSTMModelCuda::new(vocab_size, embed_dim, hidden_dim);

    println!("Starting supervised dialog training...");
    aria::model_cuda::pretrain_from_files(&mut model, &mut tokenizer, data_dir, checkpoint_path, tokenizer_path)?;

    println!("\nSaving final checkpoint...");
    model.save_checkpoint(checkpoint_path)?;
    tokenizer.save(tokenizer_path)?;

    println!("Done.");
    Ok(())
}
