#![recursion_limit = "256"]

mod model_cuda;
mod adaptive_softmax;
mod tokenizer;
mod storage;
mod db;
mod lstm_cuda;
mod math_train;

use std::io::{self, Write};
use std::fs;
use uuid::Uuid;
use crate::model_cuda::LSTMModelCuda;
use crate::tokenizer::Tokenizer;
use crate::storage::{EncryptionManager, DialogEntry, get_current_timestamp};

struct Stats {
    total_messages: u32,
    positive_rewards: u32,
    negative_rewards: u32,
    total_loss: f32,
}

#[derive(Clone, Copy)]
enum SamplingMode {
    Greedy,
    TopK { k: usize, temperature: f32 },
    TopP { p: f32, temperature: f32 },
}

fn main() -> anyhow::Result<()> {
    println!("=====================================");
    println!("                 ARIA                ");
    println!("Adaptive Reasoning Intelligence Agent");
    println!("=====================================\n");

    let encryption_key = storage::EncryptionManager::generate_key();
    println!("Encryption key: {}\n", &encryption_key[..16]);

    let encryptor = EncryptionManager::new(&encryption_key)?;

    let json_dir = "aria json";
    fs::create_dir_all(&json_dir)?;
    let db_path = "aria json/aria_dialogs.json";
    db::init_db(db_path)?;

    let data_dir = "data base";
    fs::create_dir_all(&data_dir)?;

    for filename in &["DataBase.txt", "Words.txt"] {
        let filepath = format!("{}/{}", data_dir, filename);
        if !std::path::Path::new(&filepath).exists() {
            fs::write(&filepath, "")?;
        }
    }

    println!("Data directory: {}\n", data_dir);

    let session_id = Uuid::new_v4().to_string();
    println!("Session ID: {}\n", session_id);

    let embed_dim = 1024;
    let hidden_dim = 2048;

    let checkpoint_path = "aria json/aria_checkpoint.json";
    let tokenizer_path = "aria json/aria_tokenizer.json";

    let checkpoint_exists = std::path::Path::new(checkpoint_path).exists();
    let tokenizer_exists = std::path::Path::new(tokenizer_path).exists();
    let continue_train = std::env::var("ARIA_CONTINUE_TRAIN").is_ok();

    let (mut model, mut tokenizer) = if checkpoint_exists && tokenizer_exists && !continue_train {
        println!("Loading checkpoint and tokenizer...");
        let t = Tokenizer::load(tokenizer_path)?;
        let m = LSTMModelCuda::load_checkpoint(checkpoint_path)?;
        println!("Checkpoint loaded. Vocabulary: {}\n", t.vocab_size());
        (m, t)
    } else if checkpoint_exists && tokenizer_exists && continue_train {
        println!("Loading checkpoint and tokenizer for continued training...");
        let mut t = Tokenizer::load(tokenizer_path)?;
        let mut m = LSTMModelCuda::load_checkpoint(checkpoint_path)?;
        println!("Checkpoint loaded. Continuing training...\n");
        model_cuda::pretrain_from_files(&mut m, &mut t, data_dir, checkpoint_path, tokenizer_path).ok();
        m.save_checkpoint(checkpoint_path).ok();
        t.save(tokenizer_path).ok();
        (m, t)
    } else {
        let mut t = Tokenizer::new();
        let m = train_fresh(&mut t, data_dir, checkpoint_path, tokenizer_path, embed_dim, hidden_dim)?;
        m.save_checkpoint(checkpoint_path).ok();
        t.save(tokenizer_path).ok();
        (m, t)
    };

    println!("Vocabulary: {}", tokenizer.vocab_size());
    println!("\nCommands:");
    println!("  stats            - show statistics");
    println!("  settings         - show current sampling settings");
    println!("  mode greedy      - greedy decoding");
    println!("  mode topk        - top-k sampling (default, k=20)");
    println!("  mode topp        - nucleus (top-p) sampling (default p=0.9)");
    println!("  temp <0.1-2.0>   - set temperature");
    println!("  topk <n>         - set top-k value");
    println!("  topp <0.0-1.0>   - set top-p value");
    println!("  exit             - quit\n");

    let mut stats = Stats {
        total_messages: 0, positive_rewards: 0, negative_rewards: 0, total_loss: 0.0,
    };

    let mut mode = SamplingMode::TopK { k: 20, temperature: 0.7 };

    loop {
        print!("You: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input == "exit" { println!("\nGoodbye"); break; }
        if input == "stats" {
            println!("\n=== Statistics ===");
            println!("Total messages: {}", stats.total_messages);
            println!("Positive: {}", stats.positive_rewards);
            println!("Negative: {}", stats.negative_rewards);
            if stats.total_messages > 0 {
                println!("Avg loss: {:.4}", stats.total_loss / stats.total_messages as f32);
            }
            println!();
            continue;
        }
        if input == "settings" {
            match mode {
                SamplingMode::Greedy => println!("\nMode: greedy\n"),
                SamplingMode::TopK { k, temperature } => println!("\nMode: top-k  k={}  temp={:.2}\n", k, temperature),
                SamplingMode::TopP { p, temperature } => println!("\nMode: top-p  p={:.2}  temp={:.2}\n", p, temperature),
            }
            continue;
        }
        if input == "mode greedy" { mode = SamplingMode::Greedy; println!("Mode: greedy\n"); continue; }
        if input == "mode topk"   { mode = SamplingMode::TopK { k: 20, temperature: 0.7 }; println!("Mode: top-k  k=20  temp=0.70\n"); continue; }
        if input == "mode topp"   { mode = SamplingMode::TopP { p: 0.9, temperature: 0.7 }; println!("Mode: top-p  p=0.90  temp=0.70\n"); continue; }

        if let Some(rest) = input.strip_prefix("temp ") {
            if let Ok(t) = rest.trim().parse::<f32>() {
                let t = t.clamp(0.05, 2.0);
                mode = match mode {
                    SamplingMode::Greedy               => SamplingMode::TopK { k: 20, temperature: t },
                    SamplingMode::TopK { k, .. }       => SamplingMode::TopK { k, temperature: t },
                    SamplingMode::TopP { p, .. }       => SamplingMode::TopP { p, temperature: t },
                };
                println!("Temperature set to {:.2}\n", t);
            } else {
                println!("Usage: temp <0.05-2.0>\n");
            }
            continue;
        }
        if let Some(rest) = input.strip_prefix("topk ") {
            if let Ok(k) = rest.trim().parse::<usize>() {
                let temperature = match mode { SamplingMode::TopK { temperature, .. } | SamplingMode::TopP { temperature, .. } => temperature, _ => 0.7 };
                mode = SamplingMode::TopK { k: k.max(1), temperature };
                println!("Mode: top-k  k={}  temp={:.2}\n", k.max(1), temperature);
            } else {
                println!("Usage: topk <n>\n");
            }
            continue;
        }
        if let Some(rest) = input.strip_prefix("topp ") {
            if let Ok(p) = rest.trim().parse::<f32>() {
                let temperature = match mode { SamplingMode::TopK { temperature, .. } | SamplingMode::TopP { temperature, .. } => temperature, _ => 0.7 };
                let p = p.clamp(0.01, 1.0);
                mode = SamplingMode::TopP { p, temperature };
                println!("Mode: top-p  p={:.2}  temp={:.2}\n", p, temperature);
            } else {
                println!("Usage: topp <0.0-1.0>\n");
            }
            continue;
        }
        if input.is_empty() { continue; }

        // Build role-formatted prompt for the model.
        let prompt = format!("Пользователь: {}\nАссистент:", input);
        let tokens = tokenizer.encode(&prompt);
        if tokens.len() < 3 { continue; }

        let user_entry_id = Uuid::new_v4().to_string();
        let user_encrypted = encryptor.encrypt(input)?;
        let user_entry = DialogEntry {
            id: user_entry_id.clone(), role: "user".to_string(),
            encrypted_message: user_encrypted, timestamp: get_current_timestamp(), reward: None,
        };
        db::insert_dialog(db_path, &user_entry, &session_id)?;

        print!("ARIA: ");
        io::stdout().flush()?;

        let input_no_end = &tokens[..tokens.len() - 1];
        let (mut current_logits, mut current_state) = model.forward_seq(input_no_end);
        let mut generated_tokens: Vec<usize> = Vec::new();

        // Greedy / short decoding for stable dialog: stop at END or after 40 tokens.
        for _ in 0..40 {
            tokenizer.mask_logits(&mut current_logits);

            let action = match mode {
                SamplingMode::Greedy => model.sample_greedy(&current_logits),
                SamplingMode::TopK { k, temperature } => model.sample_top_k(&current_logits, temperature, k),
                SamplingMode::TopP { p, temperature } => model.sample_top_p(&current_logits, temperature, p),
            };

            if action >= tokenizer.vocab_size() { break; }
            if action == 3 || action == 0 { break; }
            if action == 1 { continue; }

            generated_tokens.push(action);

            let (nl, ns) = model.step(action, &current_state);
            current_state = ns;
            current_logits = nl;
        }

        let response_text = tokenizer.decode(&generated_tokens);
        println!("{}\n", response_text);
        let aria_entry_id = Uuid::new_v4().to_string();
        let aria_encrypted = encryptor.encrypt(&response_text)?;
        let aria_entry = DialogEntry {
            id: aria_entry_id.clone(), role: "aria".to_string(),
            encrypted_message: aria_encrypted, timestamp: get_current_timestamp(), reward: None,
        };
        db::insert_dialog(db_path, &aria_entry, &session_id)?;
        stats.total_messages += 1;
    }

    Ok(())
}

fn train_fresh(tokenizer: &mut Tokenizer, data_dir: &str, checkpoint_path: &str, tokenizer_path: &str, embed_dim: usize, hidden_dim: usize) -> anyhow::Result<LSTMModelCuda> {
    println!("Building vocabulary from data...\n");

    let path = std::path::Path::new(data_dir);
    let mut all_text = String::new();
    if path.exists() {
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let p = entry.path();
            let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
            if ext == "txt" {
                if let Ok(content) = std::fs::read_to_string(&p) {
                    all_text.push_str(&content);
                    all_text.push(' ');
                }
            } else if ext == "jsonl" {
                if let Ok(content) = std::fs::read_to_string(&p) {
                    for line in content.lines() {
                        let line = line.trim();
                        if line.is_empty() { continue; }
                        if let Ok(obj) = serde_json::from_str::<serde_json::Value>(line) {
                            if let Some(text) = obj.get("text").and_then(|v| v.as_str()) {
                                all_text.push_str(text);
                                all_text.push(' ');
                            }
                        }
                    }
                }
            }
        }
    }

    let _ = tokenizer.encode(&all_text);
    tokenizer.freeze();
    let actual_vocab = tokenizer.vocab_size();
    println!("Vocabulary built: {} words", actual_vocab);

    let mut model = LSTMModelCuda::new(actual_vocab, embed_dim, hidden_dim);

    println!("Pre-training...");
    model_cuda::pretrain_from_files(&mut model, tokenizer, data_dir, checkpoint_path, tokenizer_path).ok();

    println!("Math curriculum...");
    math_train::train_math_curriculum(&mut model, tokenizer, "data base/Math_Learn", 0.0003).ok();

    println!("Training complete.\n");
    Ok(model)
}
