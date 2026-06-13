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

    let db_path = "aria_dialogs.json";
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

    let model_path = "aria_model.json";
    let tokenizer_path = "aria_tokenizer.json";

    let model_exists = std::path::Path::new(model_path).exists();
    let tokenizer_exists = std::path::Path::new(tokenizer_path).exists();

    let (mut model, mut tokenizer) = if model_exists && tokenizer_exists {
        println!("Loading pre-trained model and tokenizer...");
        let t = Tokenizer::load(tokenizer_path)?;
        let actual_vocab = t.vocab_size();
        match LSTMModelCuda::load(model_path, actual_vocab, embed_dim, hidden_dim) {
            Ok(m) => {
                println!("Model loaded. Vocabulary: {}\n", actual_vocab);
                (m, t)
            },
            _ => {
                println!("Failed to load model. Training from scratch.\n");
                let mut t = Tokenizer::new();
                let m = train_fresh(&mut t, data_dir, embed_dim, hidden_dim)?;
                m.save(model_path).ok();
                t.save(tokenizer_path).ok();
                (m, t)
            }
        }
    } else {
        let mut t = Tokenizer::new();
        let m = train_fresh(&mut t, data_dir, embed_dim, hidden_dim)?;
        m.save(model_path).ok();
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

        let tokens = tokenizer.encode(input);
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
        let mut response_words: Vec<String> = Vec::new();
        let mut generated_tokens: Vec<usize> = Vec::new();

        for _ in 0..50 {
            let action = match mode {
                SamplingMode::Greedy => model.sample_greedy(&current_logits),
                SamplingMode::TopK { k, temperature } => model.sample_top_k(&current_logits, temperature, k),
                SamplingMode::TopP { p, temperature } => model.sample_top_p(&current_logits, temperature, p),
            };

            if action >= tokenizer.vocab_size() { break; }
            if action == 3 { break; }
            if action == 0 { break; }

            if action == 1 {
                let (nl, ns) = model.step(action, &current_state);
                current_state = ns; current_logits = nl;
                continue;
            }

            if let Some(word) = tokenizer.id_to_word(action) {
                if word == "<END>" || word == "<PAD>" { break; }
                if !word.starts_with('<') {
                    print!("{} ", word);
                    io::stdout().flush()?;
                    response_words.push(word);
                    generated_tokens.push(action);
                }
            }

            let (nl, ns) = model.step(action, &current_state);
            current_state = ns;
            current_logits = nl;
        }

        println!("\n");

        let response_text = response_words.join(" ");
        let aria_entry_id = Uuid::new_v4().to_string();
        let aria_encrypted = encryptor.encrypt(&response_text)?;
        let aria_entry = DialogEntry {
            id: aria_entry_id.clone(), role: "aria".to_string(),
            encrypted_message: aria_encrypted, timestamp: get_current_timestamp(), reward: None,
        };
        db::insert_dialog(db_path, &aria_entry, &session_id)?;
        stats.total_messages += 1;

        print!("Rate (I like / I dont like / skip): ");
        io::stdout().flush()?;

        let mut rating = String::new();
        io::stdin().read_line(&mut rating)?;
        let rating = rating.trim();

        let reward = if rating.starts_with("I like") {
            stats.positive_rewards += 1; 1.0f32
        } else if rating.starts_with("I dont like") {
            stats.negative_rewards += 1; -1.0f32
        } else { 0.0f32 };

        if reward != 0.0 {
            db::update_dialog_reward(db_path, &aria_entry_id, reward)?;
            println!("Reward: {}", if reward > 0.0 { "+1" } else { "-1" });

            let mut combined: Vec<usize> = input_no_end.to_vec();
            combined.extend(generated_tokens.iter().copied());
            combined.push(3);

            if combined.len() >= 3 {
                if reward > 0.0 {
                    // Reinforce: gradient descent, make this response more likely.
                    let loss = model.backward_step(&combined, 0.0005);
                    stats.total_loss += loss;
                    println!("Reinforced (loss = {:.4})\n", loss);
                } else {
                    // Punish: gradient ascent, make this response less likely.
                    let loss = model.backward_step(&combined, -0.0002);
                    stats.total_loss += loss.abs();
                    println!("Penalized (loss = {:.4})\n", loss);
                }
            }
        }
    }

    Ok(())
}

fn train_fresh(tokenizer: &mut Tokenizer, data_dir: &str, embed_dim: usize, hidden_dim: usize) -> anyhow::Result<LSTMModelCuda> {
    println!("Building vocabulary from data...\n");

    let path = std::path::Path::new(data_dir);
    let mut all_text = String::new();
    if path.exists() {
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            if entry.path().extension().map_or(false, |e| e == "txt") {
                if let Ok(content) = std::fs::read_to_string(entry.path()) {
                    all_text.push_str(&content);
                    all_text.push(' ');
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
    model_cuda::pretrain_from_files(&mut model, tokenizer, data_dir).ok();

    println!("Math curriculum...");
    math_train::train_math_curriculum(&mut model, tokenizer, "Math_Learn", 0.0003).ok();

    println!("Training complete.\n");
    Ok(model)
}
