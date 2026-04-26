mod model;
mod tokenizer;
mod storage;
mod db;
mod pretraining;
mod lstm_gpu;

use std::io::{self, Write};
use std::fs;
use uuid::Uuid;
use crate::model::LSTMModel;
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

    for filename in &["Books.txt", "Reddit.txt", "Poetry.txt", "summary.txt", "fan fiction.txt", "words.txt", "news.txt"] {
        let filepath = format!("{}/{}", data_dir, filename);
        if !std::path::Path::new(&filepath).exists() {
            fs::write(&filepath, "")?;
        }
    }

    println!("Data directory: {}\n", data_dir);

    let session_id = Uuid::new_v4().to_string();
    println!("Session ID: {}\n", session_id);

    let embed_dim = 512;
    let hidden_dim = 1024;

    let model_path = "aria_model.json";
    let tokenizer_path = "aria_tokenizer.json";

    let model_exists = std::path::Path::new(model_path).exists();
    let tokenizer_exists = std::path::Path::new(tokenizer_path).exists();

    let (mut model, mut tokenizer) = if model_exists && tokenizer_exists {
        println!("Loading pre-trained model and tokenizer...");
        let t = Tokenizer::load(tokenizer_path)?;
        let actual_vocab = t.vocab_size();
        match LSTMModel::load(model_path, actual_vocab, embed_dim, hidden_dim) {
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
    println!("  stats       - show statistics");
    println!("  mode greedy - greedy decoding");
    println!("  mode topk   - top-k sampling (default)");
    println!("  exit        - quit\n");

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
        if input == "mode greedy" { mode = SamplingMode::Greedy; println!("Mode: greedy\n"); continue; }
        if input == "mode topk" { mode = SamplingMode::TopK { k: 20, temperature: 0.7 }; println!("Mode: top-k\n"); continue; }
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

            if reward > 0.0 {
                let mut combined: Vec<usize> = input_no_end.to_vec();
                combined.extend(generated_tokens.iter().copied());
                combined.push(3);
                if combined.len() >= 3 {
                    let loss = model.backward_step(&combined, 0.0005);
                    stats.total_loss += loss;
                    println!("Reinforced (loss = {:.4})\n", loss);
                }
            } else {
                println!("Noted (negative reward logged)\n");
            }
        }
    }

    Ok(())
}

fn train_fresh(tokenizer: &mut Tokenizer, data_dir: &str, embed_dim: usize, hidden_dim: usize) -> anyhow::Result<LSTMModel> {
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
    let actual_vocab = tokenizer.vocab_size();
    println!("Vocabulary built: {} words", actual_vocab);
    tokenizer.freeze();

    let mut model = LSTMModel::new(actual_vocab, embed_dim, hidden_dim);

    println!("Pre-training...");
    pretraining::pretrain_from_files(&mut model, tokenizer, data_dir).ok();

    println!("Training complete.\n");
    Ok(model)
}
