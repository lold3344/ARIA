use std::collections::HashMap;

pub struct Tokenizer {
    word_to_id: HashMap<String, usize>,
    id_to_word: HashMap<usize, String>,
}

impl Tokenizer {
    pub fn new() -> Self {
        let mut tokenizer = Tokenizer {
            word_to_id: HashMap::new(),
            id_to_word: HashMap::new(),
        };

        let special_tokens = vec!["<PAD>", "<UNK>", "<START>", "<END>"];
        for (i, token) in special_tokens.iter().enumerate() {
            tokenizer.word_to_id.insert(token.to_string(), i);
            tokenizer.id_to_word.insert(i, token.to_string());
        }

        tokenizer
    }

    pub fn encode(&mut self, text: &str) -> Vec<usize> {
        let words: Vec<&str> = text.to_lowercase()
            .split_whitespace()
            .collect();

        let mut tokens = vec![self.word_to_id["<START>"]];

        for word in words {
            let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric());
            if !clean_word.is_empty() {
                let id = self.word_to_id.len();
                let token_id = *self.word_to_id.entry(clean_word.to_string())
                    .or_insert(id);
                
                if token_id >= self.word_to_id.len() - 1 {
                    self.id_to_word.insert(token_id, clean_word.to_string());
                }
                tokens.push(token_id);
            }
        }

        tokens.push(self.word_to_id["<END>"]);
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

    pub fn add_word(&mut self, word: &str) -> usize {
        let id = self.word_to_id.len();
        let token_id = *self.word_to_id.entry(word.to_string())
            .or_insert(id);
        if !self.id_to_word.contains_key(&token_id) {
            self.id_to_word.insert(token_id, word.to_string());
        }
        token_id
    }
}
