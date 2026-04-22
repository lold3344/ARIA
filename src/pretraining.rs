use std::fs;
use std::path::Path;
use crate::model::LSTMModel;
use crate::tokenizer::Tokenizer;

pub fn pretrain_from_files(model: &mut LSTMModel, tokenizer: &mut Tokenizer, data_dir: &str) -> anyhow::Result<()> {
    let path = Path::new(data_dir);
    
    if !path.exists() {
        return Ok(());
    }

    let mut total_loss = 0.0;
    let mut count = 0;

    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let file_path = entry.path();

        if file_path.extension().map_or(false, |ext| ext == "txt") {
            let content = match fs::read_to_string(&file_path) {
                Ok(c) => c,
                Err(_) => continue,
            };

            if content.trim().is_empty() {
                continue;
            }

            println!("Pre-training on: {}", file_path.file_name().unwrap_or_default().to_string_lossy());
            
            let sentences: Vec<&str> = content.split(|c| c == '.' || c == '\n').collect();

            for sentence in sentences {
                let sentence = sentence.trim();
                if sentence.is_empty() || sentence.len() < 5 {
                    continue;
                }

                let tokens = tokenizer.encode(sentence);
                if tokens.len() < 3 {
                    continue;
                }

                for i in 1..tokens.len().min(10) {
                    let input = &tokens[0..i];
                    let target = tokens[i];

                    let (logits, _) = model.forward_seq(input);
                    let probs = model.softmax(&logits);

                    if target < probs.len() {
                        let loss = -probs[target].ln().max(-20.0);
                        total_loss += loss;
                        count += 1;

                        if count % 100 == 0 {
                            println!("  Processed {} tokens, avg loss: {:.4}", count, total_loss / count as f32);
                        }

                        apply_pretrain_gradient(model, target, &probs, 0.001);
                    }
                }
            }
        }
    }

    if count > 0 {
        println!("Pre-training complete. Total loss: {:.4}, tokens: {}", total_loss / count as f32, count);
    } else {
        println!("No text files found or all files are empty. Pre-training skipped.");
    }

    Ok(())
}

fn apply_pretrain_gradient(model: &mut LSTMModel, target: usize, probs: &ndarray::Array1<f32>, lr: f32) {
    let mut grad = probs.clone();
    grad[target] -= 1.0;
    grad = &grad * (lr / probs.len() as f32);

    for layer in model.w_out.iter_mut() {
        *layer -= grad.mean().unwrap_or(0.0) * 0.1;
    }

    for layer in model.b_out.iter_mut() {
        *layer -= grad.mean().unwrap_or(0.0) * 0.1;
    }

    for layer in model.w_hi.iter_mut() {
        *layer -= grad.mean().unwrap_or(0.0) * 0.05;
    }

    for layer in model.w_hf.iter_mut() {
        *layer -= grad.mean().unwrap_or(0.0) * 0.05;
    }

    for layer in model.w_ho.iter_mut() {
        *layer -= grad.mean().unwrap_or(0.0) * 0.05;
    }
}
