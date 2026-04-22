use std::fs;
use std::path::Path;
use crate::model::LSTMModel;
use crate::tokenizer::Tokenizer;

pub fn pretrain_from_files(model: &mut LSTMModel, tokenizer: &mut Tokenizer, data_dir: &str) -> anyhow::Result<()> {
    let path = Path::new(data_dir);
    
    if !path.exists() {
        return Ok(());
    }

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

                    let (_logits, _) = model.forward_seq(input);
                }
            }
        }
    }

    Ok(())
}