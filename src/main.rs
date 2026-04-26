mod model;
mod tokenizer;
mod storage;
mod db;
mod pretraining;

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

    println!("Data directory structure created at: {}", data_dir);
    println!("Files created:");
    println!("  - data base/Books.txt");
    println!("  - data base/Reddit.txt");
    println!("  - data base/Poetry.txt");
    println!("  - data base/summary.txt");
    println!("  - data base/fan fiction.txt");
    println!("  - data base/words.txt");
    println!("  - data base/news.txt\n");

    let session_id = Uuid::new_v4().to_string();
    println!("Session ID: {}\n", session_id);

    let embed_dim = 512;
    let hidden_dim = 1024;
    let vocab_size = 8000;

    let model_path = "aria_model.json";
    let tokenizer_path = "aria_tokenizer.json";

    let model_exists = std::path::Path::new(model_path).exists();
    let tokenizer_exists = std::path::Path::new(tokenizer_path).exists();

    let (mut model, mut tokenizer) = if model_exists && tokenizer_exists {
        println!("Loading pre-trained model and tokenizer...");
        match (LSTMModel::load(model_path, vocab_size, embed_dim, hidden_dim), Tokenizer::load(tokenizer_path)) {
            (Ok(m), Ok(t)) => {
                println!("Model loaded from: {}", model_path);
                println!("Tokenizer loaded from: {}", tokenizer_path);
                println!("Vocabulary size: {}\n", t.vocab_size());
                (m, t)
            },
            _ => {
                println!("Failed to load. Training from scratch.\n");
                let mut t = Tokenizer::new();
                let mut m = LSTMModel::new(vocab_size, embed_dim, hidden_dim);

                println!("Pre-training on text files from: {}", data_dir);
                pretraining::pretrain_from_files(&mut m, &mut t, data_dir).ok();

                println!("\nSaving model to: {}", model_path);
                m.save(model_path).ok();
                println!("Saving tokenizer to: {}", tokenizer_path);
                t.save(tokenizer_path).ok();
                println!("Training complete. Ready for chat.\n");

                (m, t)
            }
        }
    } else {
        println!("Initializing...");
        println!("  - Embedding dim: {}", embed_dim);
        println!("  - Hidden dim: {}", hidden_dim);
        println!("  - Vocabulary: {}", vocab_size);
        println!();

        let mut t = Tokenizer::new();
        let mut m = LSTMModel::new(vocab_size, embed_dim, hidden_dim);

        println!("Pre-training on text files from: {}", data_dir);
        pretraining::pretrain_from_files(&mut m, &mut t, data_dir).ok();

        println!("\nSaving model to: {}", model_path);
        m.save(model_path).ok();
        println!("Saving tokenizer to: {}", tokenizer_path);
        t.save(tokenizer_path).ok();
        println!("Training complete. Ready for chat.\n");

        (m, t)
    };

    println!("Tokenizer vocabulary size: {}", tokenizer.vocab_size());
    println!("\nCommands:");
    println!("  stats                - show statistics");
    println!("  mode greedy          - switch to greedy decoding");
    println!("  mode topk            - switch to top-k sampling (default)");
    println!("  Reward: 'I like' at the start of response");
    println!("  Punishment: 'I dont like' at the start of response");
    println!("  Type 'exit' to quit\n");

    let mut stats = Stats {
        total_messages: 0,
        positive_rewards: 0,
        negative_rewards: 0,
        total_loss: 0.0,
    };

    let mut mode = SamplingMode::TopK { k: 20, temperature: 0.7 };

    loop {
        print!("You: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input == "exit" {
            println!("\nGoodbye");
            break;
        }

        if input == "stats" {
            println!("\n=== Statistics ===");
            println!("Total messages: {}", stats.total_messages);
            println!("Positive rewards: {}", stats.positive_rewards);
            println!("Negative rewards: {}", stats.negative_rewards);
            if stats.total_messages > 0 {
                println!("Avg loss: {:.4}", stats.total_loss / stats.total_messages as f32);
            }
            println!();
            continue;
        }

        if input == "mode greedy" {
            mode = SamplingMode::Greedy;
            println!("Sampling mode: greedy\n");
            continue;
        }

        if input == "mode topk" {
            mode = SamplingMode::TopK { k: 20, temperature: 0.7 };
            println!("Sampling mode: top-k (k=20, temp=0.7)\n");
            continue;
        }

        if input.is_empty() {
            continue;
        }

        let tokens = tokenizer.encode(input);
        if tokens.is_empty() {
            continue;
        }

        let user_entry_id = Uuid::new_v4().to_string();
        let user_encrypted = encryptor.encrypt(input)?;

        let user_entry = DialogEntry {
            id: user_entry_id.clone(),
            role: "user".to_string(),
            encrypted_message: user_encrypted,
            timestamp: get_current_timestamp(),
            reward: None,
        };

        db::insert_dialog(db_path, &user_entry, &session_id)?;

        print!("ARIA: ");
        io::stdout().flush()?;

        let (mut current_logits, mut current_state) = model.forward_seq(&tokens);
        let mut response_words: Vec<String> = Vec::new();
        let mut generated_tokens: Vec<usize> = Vec::new();

        for _ in 0..25 {
            let action = match mode {
                SamplingMode::Greedy => model.sample_greedy(&current_logits),
                SamplingMode::TopK { k, temperature } => model.sample_top_k(&current_logits, temperature, k),
            };

            if action >= tokenizer.vocab_size() {
                break;
            }

            if action < 2 {
                if action == 3 {
                    break;
                }
                let (next_logits, next_state) = model.step(action, &current_state);
                current_state = next_state;
                current_logits = next_logits;
                continue;
            }

            if let Some(word) = tokenizer.id_to_word(action) {
                if word == "<END>" {
                    break;
                }
                if !word.starts_with('<') && !word.ends_with('>') {
                    print!("{} ", word);
                    io::stdout().flush()?;
                    response_words.push(word);
                    generated_tokens.push(action);
                }
            }

            let (next_logits, next_state) = model.step(action, &current_state);
            current_state = next_state;
            current_logits = next_logits;
        }

        println!("\n");

        let response_text = response_words.join(" ");

        let aria_entry_id = Uuid::new_v4().to_string();
        let aria_encrypted = encryptor.encrypt(&response_text)?;

        let aria_entry = DialogEntry {
            id: aria_entry_id.clone(),
            role: "aria".to_string(),
            encrypted_message: aria_encrypted,
            timestamp: get_current_timestamp(),
            reward: None,
        };

        db::insert_dialog(db_path, &aria_entry, &session_id)?;

        stats.total_messages += 1;

        print!("Rate (I like / I dont like / skip): ");
        io::stdout().flush()?;

        let mut rating = String::new();
        io::stdin().read_line(&mut rating)?;
        let rating = rating.trim();

        let reward = if rating.starts_with("I like") {
            stats.positive_rewards += 1;
            1.0f32
        } else if rating.starts_with("I dont like") {
            stats.negative_rewards += 1;
            -1.0f32
        } else {
            0.0f32
        };

        if reward != 0.0 {
            db::update_dialog_reward(db_path, &aria_entry_id, reward)?;
            println!("Reward: {}", if reward > 0.0 { "+1" } else { "-1" });

            let mut combined: Vec<usize> = tokens.clone();
            combined.extend(generated_tokens.iter().copied());

            if combined.len() >= 2 {
                let lr = if reward > 0.0 { 0.02 } else { -0.02 };
                let loss = model.backward_step(&combined, lr);
                stats.total_loss += loss;
                println!("Online update applied (loss = {:.4})\n", loss);
            } else {
                println!();
            }
        }
    }

    Ok(())
}
