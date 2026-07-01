#![recursion_limit = "256"]

mod transformer_cuda;
mod tokenizer;
mod storage;
mod db;

use std::io::{self, Write};
use std::fs;
use uuid::Uuid;
use crate::transformer_cuda::TransformerModel;
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

fn print_logo() {
    let logo_path = "screenshots/ARIA-LOGO-ANSI.txt";
    if let Ok(content) = fs::read_to_string(logo_path) {
        let esc = '\x1b';
        for line in content.lines() {
            println!("{}", line.replace("\\e", &esc.to_string()));
        }
        print!("\x1b[0m");
    } else {
        println!("=====================================");
        println!("                 ARIA                ");
        println!("Adaptive Reasoning Intelligence Agent");
        println!("=====================================");
    }
    println!();
}

fn main() -> anyhow::Result<()> {
    print_logo();

    // Persist encryption key so dialogs can be decrypted across sessions
    let key_path = "aria json/aria_key.hex";
    let encryption_key = if std::path::Path::new(key_path).exists() {
        fs::read_to_string(key_path).unwrap_or_else(|_| storage::EncryptionManager::generate_key())
    } else {
        let k = storage::EncryptionManager::generate_key();
        fs::create_dir_all("aria json").ok();
        fs::write(key_path, &k).ok();
        k
    };
    println!("Encryption key: {}\n", &encryption_key[..16]);

    let encryptor = EncryptionManager::new(&encryption_key)?;

    let json_dir = "aria json";
    fs::create_dir_all(&json_dir)?;
    let db_path = "aria json/aria_dialogs.json";
    db::init_db(db_path)?;

    let data_dir = "data base";
    fs::create_dir_all(&data_dir)?;


    println!("Data directory: {}\n", data_dir);

    let session_id = Uuid::new_v4().to_string();
    println!("Session ID: {}\n", session_id);

    let checkpoint_path = "aria json/aria_checkpoint.json";
    let tokenizer_path = "aria json/aria_tokenizer.json";

    let checkpoint_exists = std::path::Path::new(checkpoint_path).exists();
    let tokenizer_exists = std::path::Path::new(tokenizer_path).exists();
    let continue_train = std::env::var("ARIA_CONTINUE_TRAIN").is_ok();

    let (mut model, mut tokenizer) = if checkpoint_exists && tokenizer_exists && !continue_train {
        println!("Loading checkpoint and tokenizer...");
        let t = Tokenizer::load(tokenizer_path)?;
        let m = TransformerModel::load_checkpoint(checkpoint_path)?;
        println!("Checkpoint loaded. Vocabulary: {}\n", t.vocab_size());
        (m, t)
    } else if checkpoint_exists && tokenizer_exists && continue_train {
        println!("Loading checkpoint and tokenizer for continued training...");
        let mut t = Tokenizer::load(tokenizer_path)?;
        let mut m = TransformerModel::load_checkpoint(checkpoint_path)?;
        println!("Checkpoint loaded. Continuing training...\n");
        crate::transformer_cuda::pretrain_from_files(&mut m, &mut t, data_dir, checkpoint_path, tokenizer_path).ok();
        m.save_checkpoint(checkpoint_path).ok();
        t.save(tokenizer_path).ok();
        (m, t)
    } else {
        let mut t = Tokenizer::new();
        let m = train_fresh(&mut t, data_dir, checkpoint_path, tokenizer_path)?;
        m.save_checkpoint(checkpoint_path).ok();
        t.save(tokenizer_path).ok();
        (m, t)
    };

    // Chat is inference-only from here on — drop training buffers.
    model.free_training_buffers();

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

    // Load full dialog history from persistent JSON memory
    let mut history: Vec<(String, String)> = {
        let entries = db::load_recent_dialogs(db_path, usize::MAX / 4).unwrap_or_default();
        let mut pairs: Vec<(String, String)> = Vec::new();
        let mut i = 0;
        while i + 1 < entries.len() {
            if entries[i].role == "user" && entries[i + 1].role == "aria" {
                let user_text = encryptor.decrypt(&entries[i].encrypted_message).unwrap_or_default();
                let aria_text = encryptor.decrypt(&entries[i + 1].encrypted_message).unwrap_or_default();
                if !user_text.is_empty() && !aria_text.is_empty() {
                    pairs.push((user_text, aria_text));
                }
                i += 2;
            } else {
                i += 1;
            }
        }
        pairs
    };
    const MAX_CONTEXT_TOKENS: usize = 220;

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

        // Build context: pack as many recent exchanges as fit + current message
        let current_prompt = tokenizer.encode_prompt(input);
        // reserve room for the current prompt inside the window
        let budget = MAX_CONTEXT_TOKENS.saturating_sub(current_prompt.len());
        let mut hist_tokens: Vec<usize> = Vec::new();
        for (user_text, aria_text) in history.iter().rev() {
            let mut user_tokens = tokenizer.encode_prompt(user_text);
            user_tokens.pop(); // remove trailing ASSISTANT token, we append the response instead
            let aria_tokens = tokenizer.encode(aria_text);
            let turn_len = user_tokens.len() + aria_tokens.len() + 1; // +1 for END
            if 1 + hist_tokens.len() + turn_len >= budget { break; } // +1 for START
            // prepend this (older) turn before the already-collected newer turns
            let mut turn: Vec<usize> = user_tokens;
            turn.extend(aria_tokens);
            turn.push(3usize); // END
            turn.extend(hist_tokens);
            hist_tokens = turn;
        }
        // Append current user message (current_prompt already starts with START)
        let full: Vec<usize> = hist_tokens.iter().chain(current_prompt.iter()).copied().collect();
        let tokens = if full.len() <= MAX_CONTEXT_TOKENS + 30 { full } else { current_prompt };
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

        let mut ids = tokens.clone();
        let mut generated_tokens: Vec<usize> = Vec::new();

        for step in 0..120 {
            let mut current_logits = model.forward_gpu(&ids);
            tokenizer.mask_logits(&mut current_logits);

            let action = match mode {
                SamplingMode::Greedy => model.sample_greedy(&current_logits),
                SamplingMode::TopK { k, temperature } => model.sample_top_k(&current_logits, temperature, k),
                SamplingMode::TopP { p, temperature } => model.sample_top_p(&current_logits, temperature, p),
            };

            if action >= tokenizer.vocab_size() { break; }
            if (action == 3 || action == 0) && step >= 3 { break; }
            if action == 1 { continue; }

            generated_tokens.push(action);
            ids.push(action);
            if ids.len() >= 256 { break; }
        }

        let response_text = tokenizer.decode(&generated_tokens);
        println!("{}\n", response_text);
        history.push((input.to_string(), response_text.clone()));
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

fn train_fresh(tokenizer: &mut Tokenizer, data_dir: &str, checkpoint_path: &str, tokenizer_path: &str) -> anyhow::Result<TransformerModel> {
    use std::io::{BufRead, BufReader};

    // Limit vocab building to avoid OOM — full corpus not needed for BPE
    let vocab_lines: usize = std::env::var("ARIA_VOCAB_LINES")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(500_000);

    println!("Building vocabulary from data (max {} lines)...\n", vocab_lines);

    let path = std::path::Path::new(data_dir);
    if path.exists() {
        let mut total_lines = 0usize;
        'outer: for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let p = entry.path();
            let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
            if ext == "txt" {
                if let Ok(f) = std::fs::File::open(&p) {
                    for line in BufReader::new(f).lines().flatten() {
                        tokenizer.encode(&line);
                        total_lines += 1;
                        if total_lines >= vocab_lines { break 'outer; }
                    }
                }
            } else if ext == "jsonl" {
                if let Ok(f) = std::fs::File::open(&p) {
                    for line in BufReader::new(f).lines().flatten() {
                        let line = line.trim().to_string();
                        if line.is_empty() { continue; }
                        if let Ok(obj) = serde_json::from_str::<serde_json::Value>(&line) {
                            if let Some(text) = obj.get("text").and_then(|v| v.as_str()) {
                                tokenizer.encode(text);
                            }
                        }
                        total_lines += 1;
                        if total_lines >= vocab_lines { break 'outer; }
                    }
                }
            }
        }
        println!("Processed {} lines for vocabulary.", total_lines);
    }

    tokenizer.freeze();
    let actual_vocab = tokenizer.vocab_size();
    println!("Vocabulary built: {} tokens", actual_vocab);

    let mut model = TransformerModel::new(actual_vocab, 768, 12, 12, 3072, 256);

    println!("Pre-training...");
    crate::transformer_cuda::pretrain_from_files(&mut model, tokenizer, data_dir, checkpoint_path, tokenizer_path).ok();

    println!("Training complete.\n");
    Ok(model)
}
