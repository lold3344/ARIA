mod model;
mod tokenizer;
mod storage;
mod db;
mod pretraining;

use std::io::{self, Write};
use std::fs;
use uuid::Uuid;
use tch::Tensor;
use crate::model::LSTMModel;
use crate::tokenizer::Tokenizer;
use crate::storage::EncryptionManager;

struct Stats {
    total_messages: u32,
    positive_rewards: u32,
    negative_rewards: u32,
    total_loss: f32,
}

fn main() -> anyhow::Result<()> {
    println!("=====================================");
    println!("ARIA - Adaptive Reasoning Intelligence Agent");
    println!("=====================================\n");

    let encryption_key = EncryptionManager::generate_key();
    println!("Encryption key: {}\n", &encryption_key[..16]);

    let _encryptor = EncryptionManager::new(&encryption_key)?;

    let db_path = "aria_dialogs.json";
    db::init_db(db_path)?;

    let data_dir = "data base";
    fs::create_dir_all(&data_dir)?;

    let files = [
        "Books.txt",
        "Reddit.txt",
        "Poetry.txt",
        "summary.txt",
        "fan fiction.txt",
        "words.txt",
        "news.txt",
    ];

    for filename in &files {
        let filepath = format!("{}/{}", data_dir, filename);
        if !std::path::Path::new(&filepath).exists() {
            fs::write(&filepath, "")?;
        }
    }

    let _session_id = Uuid::new_v4().to_string();

    let embed_dim = 384;
    let hidden_dim = 1536;
    let vocab_size = 8000;

    let model_path = "aria_model.ot";
    let tokenizer_path = "aria_tokenizer.json";

    let (model, mut tokenizer) = if std::path::Path::new(model_path).exists()
        && std::path::Path::new(tokenizer_path).exists()
    {
        let m = LSTMModel::load(model_path, vocab_size, embed_dim, hidden_dim);
        let t = Tokenizer::load(tokenizer_path)?;
        (m, t)
    } else {
        let m = LSTMModel::new(vocab_size, embed_dim, hidden_dim);
        let t = Tokenizer::new();
        (m, t)
    };

    let mut _stats = Stats {
        total_messages: 0,
        positive_rewards: 0,
        negative_rewards: 0,
        total_loss: 0.0,
    };

    loop {
        print!("You: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input == "exit" {
            break;
        }

        let tokens = tokenizer.encode(input);
        if tokens.is_empty() {
            continue;
        }

        let vec_tokens: Vec<i64> = tokens.iter().map(|x| *x as i64).collect();

        let input_tensor = Tensor::f_from_slice(&vec_tokens)?
            .view([1, vec_tokens.len() as i64]);

        let (logits, mut state) = model.forward_seq(&input_tensor);
        let mut action = model.sample(&logits);

        let mut response_tokens = vec![action as usize];

        for _ in 0..12 {
            let (logits, new_state) = model.step(action, Some(state));
            state = new_state;

            let next = model.sample(&logits);
            if next < 2 {
                break;
            }

            response_tokens.push(next as usize);
            action = next;
        }

        let response = tokenizer.decode(&response_tokens);
        println!("ARIA: {}\n", response);

        _stats.total_messages += 1;
    }

    Ok(())
}