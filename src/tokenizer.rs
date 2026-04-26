use std::collections::HashMap;
use std::fs;

pub struct Tokenizer {
    word_to_id: HashMap<String, usize>,
    id_to_word: HashMap<usize, String>,
    max_vocab: usize,
    frozen: bool,
}

impl Tokenizer {
    pub fn new() -> Self {
        let mut tokenizer = Tokenizer {
            word_to_id: HashMap::new(),
            id_to_word: HashMap::new(),
            max_vocab: 0,
            frozen: false,
        };

        let special = vec!["<PAD>", "<UNK>", "<START>", "<END>"];
        for (i, token) in special.iter().enumerate() {
            tokenizer.word_to_id.insert(token.to_string(), i);
            tokenizer.id_to_word.insert(i, token.to_string());
        }

        tokenizer
    }

    pub fn set_max_vocab(&mut self, max_vocab: usize) {
        self.max_vocab = max_vocab;
    }

    pub fn freeze(&mut self) {
        self.frozen = true;
    }

    pub fn encode(&mut self, text: &str) -> Vec<usize> {
        let lowercase = text.to_lowercase();
        let words: Vec<&str> = lowercase.split_whitespace().collect();

        let mut tokens = vec![2];

        for word in words {
            let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
            if clean.is_empty() {
                continue;
            }

            if let Some(&id) = self.word_to_id.get(clean) {
                tokens.push(id);
            } else if self.frozen || (self.max_vocab > 0 && self.word_to_id.len() >= self.max_vocab) {
                tokens.push(1);
            } else {
                let id = self.word_to_id.len();
                self.word_to_id.insert(clean.to_string(), id);
                self.id_to_word.insert(id, clean.to_string());
                tokens.push(id);
            }
        }

        tokens.push(3);
        tokens
    }

    pub fn decode(&self, token_ids: &[usize]) -> String {
        token_ids.iter()
            .filter_map(|&id| {
                let word = self.id_to_word.get(&id)?;
                if word.starts_with('<') && word.ends_with('>') {
                    None
                } else {
                    Some(word.clone())
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub fn vocab_size(&self) -> usize {
        self.word_to_id.len()
    }

    pub fn id_to_word(&self, id: usize) -> Option<String> {
        self.id_to_word.get(&id).cloned()
    }

    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let data = serde_json::json!({
            "word_to_id": self.word_to_id,
            "id_to_word": self.id_to_word.iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect::<HashMap<String, String>>(),
        });
        fs::write(path, serde_json::to_string_pretty(&data)?)?;
        Ok(())
    }

    pub fn load(path: &str) -> anyhow::Result<Self> {
        let content = fs::read_to_string(path)?;
        let data: serde_json::Value = serde_json::from_str(&content)?;

        let mut tokenizer = Tokenizer::new();

        if let Some(w2i) = data["word_to_id"].as_object() {
            for (word, id) in w2i {
                if let Some(id_num) = id.as_u64() {
                    tokenizer.word_to_id.insert(word.clone(), id_num as usize);
                }
            }
        }

        if let Some(i2w) = data["id_to_word"].as_object() {
            for (id_str, word) in i2w {
                if let Ok(id_num) = id_str.parse::<usize>() {
                    if let Some(w) = word.as_str() {
                        tokenizer.id_to_word.insert(id_num, w.to_string());
                    }
                }
            }
        }

        tokenizer.frozen = true;
        Ok(tokenizer)
    }
}
