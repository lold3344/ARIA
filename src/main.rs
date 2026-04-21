mod model;
mod ppo;
mod tokenizer;
mod storage;
mod db;

use std::io::{self, Write};
use uuid::Uuid;
use crate::model::Model;
use crate::ppo::{PPOTrainer, Experience};
use crate::tokenizer::Tokenizer;
use crate::storage::{EncryptionManager, DialogEntry, get_current_timestamp};

fn main() -> anyhow::Result<()> {
    println!("╔════════════════════════════════════╗");
    println!("║     ARIA - Adaptive Reasoning      ║");
    println!("║        Intelligence Agent         ║");
    println!("╚════════════════════════════════════╝\n");

    let encryption_key = storage::EncryptionManager::generate_key();
    println!("🔐 Generated encryption key: {}\n", &encryption_key[..16]);

    let encryptor = EncryptionManager::new(&encryption_key)?;
    
    let db_path = "aria_dialogs.json";
    db::init_db(db_path)?;

    let session_id = Uuid::new_v4().to_string();
    println!("📝 Session ID: {}\n", session_id);

    let mut tokenizer = Tokenizer::new();
    
    let vocab_size = 5000;
    let embed_dim = 256;
    let hidden_dim = 1024;
    let output_dim = 1024;
    
    println!("🧠 Initializing model...");
    println!("   - Vocabulary size: {}", vocab_size);
    println!("   - Embedding dim: {}", embed_dim);
    println!("   - Hidden dim: {}", hidden_dim);
    println!("   - Output dim: {}", output_dim);
    println!("   - Parameters: ~1M\n");

    let mut model = Model::new(vocab_size, embed_dim, hidden_dim, output_dim);
    let trainer = PPOTrainer::new();
    let mut batch = Vec::new();

    println!("💬 Type 'exit' to quit, rate responses with: 👍 or 👎\n");

    loop {
        print!("You: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input == "exit" {
            println!("\n✨ Goodbye!");
            break;
        }

        if input.is_empty() {
            continue;
        }

        let tokens = tokenizer.encode(input);
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

        let (logits, value) = model.forward(&tokens);
        let (action, log_prob) = model.sample_action(&logits);

        let response_tokens: Vec<usize> = (0..5)
            .map(|_| {
                let (logits, _) = model.forward(&tokens);
                let (action, _) = model.sample_action(&logits);
                action % tokenizer.vocab_size()
            })
            .collect();

        let response_text = if response_tokens.is_empty() {
            "I'm learning from our conversation.".to_string()
        } else {
            tokenizer.decode(&response_tokens)
        };

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

        println!("ARIA: {}\n", response_text);

        print!("Rate response 👍 or 👎: ");
        io::stdout().flush()?;

        let mut rating = String::new();
        io::stdin().read_line(&mut rating)?;
        let rating = rating.trim();

        let reward = if rating.contains('👍') || rating.to_lowercase() == "yes" || rating == "y" {
            1.0
        } else {
            -0.5
        };

        db::update_dialog_reward(db_path, &aria_entry_id, reward)?;

        let experience = Experience {
            tokens: tokens.clone(),
            action,
            log_prob,
            reward,
            done: false,
        };

        batch.push(experience);

        if batch.len() >= 4 {
            println!("\n🔄 Training on batch of {} experiences...", batch.len());
            trainer.update(&mut model, &batch);
            batch.clear();
            println!("✅ Model updated\n");
        }

        println!("📊 Stats:");
        println!("   - Value estimate: {:.4}", value);
        println!("   - Reward: {:.1}", reward);
        println!("   - Batch size: {}\n", batch.len());
    }

    Ok(())
}
