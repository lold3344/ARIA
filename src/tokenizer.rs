use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;
use rayon::prelude::*;

const PAD: usize   = 0;
const UNK: usize   = 1;
const START: usize = 2;
const END: usize   = 3;

pub struct Tokenizer {
    merges:      Vec<(String, String)>,
    merge_rank:  HashMap<(String, String), usize>,
    token_to_id: HashMap<String, usize>,
    id_to_token: HashMap<usize, String>,
    target_vocab: usize,
    frozen:      bool,
    word_freq:   HashMap<String, usize>,
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn clean_word(w: &str) -> String {
    w.trim_matches(|c: char| !c.is_alphanumeric() && c != '-')
     .to_lowercase()
}

// Standard BPE training: returns merge rules in priority order.
fn train_bpe(word_freq: &HashMap<String, usize>, num_merges: usize) -> Vec<(String, String)> {
    let mut vocab: Vec<(Vec<String>, usize)> = word_freq
        .iter()
        .filter(|(_, &f)| f > 0)
        .map(|(word, &freq)| {
            let chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
            (chars, freq)
        })
        .collect();

    let mut merges: Vec<(String, String)> = Vec::with_capacity(num_merges);

    for step in 0..num_merges {
        // Count pairs in parallel.
        let pair_freq: HashMap<(String, String), usize> = vocab
            .par_iter()
            .fold(
                HashMap::new,
                |mut acc, (word, freq)| {
                    for i in 0..word.len().saturating_sub(1) {
                        let pair = (word[i].clone(), word[i + 1].clone());
                        *acc.entry(pair).or_insert(0) += freq;
                    }
                    acc
                },
            )
            .reduce(HashMap::new, |mut a, b| {
                for (k, v) in b { *a.entry(k).or_insert(0) += v; }
                a
            });

        if pair_freq.is_empty() { break; }

        let best = pair_freq
            .iter()
            .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)))
            .map(|(p, _)| p.clone())
            .unwrap();

        if step % 500 == 0 {
            println!("  BPE step {}/{} best={}{} (freq={})",
                step, num_merges, best.0, best.1,
                pair_freq[&best]);
            let _ = std::io::stdout().flush();
        }

        merges.push(best.clone());

        let merged = format!("{}{}", best.0, best.1);
        vocab = vocab
            .into_par_iter()
            .map(|(word, freq)| {
                let mut out: Vec<String> = Vec::with_capacity(word.len());
                let mut i = 0;
                while i < word.len() {
                    if i + 1 < word.len() && word[i] == best.0 && word[i + 1] == best.1 {
                        out.push(merged.clone());
                        i += 2;
                    } else {
                        out.push(word[i].clone());
                        i += 1;
                    }
                }
                (out, freq)
            })
            .collect();
    }

    merges
}

// Apply learned merges to a single word, return list of sub-word tokens.
fn apply_merges(word: &str, merge_rank: &HashMap<(String, String), usize>) -> Vec<String> {
    if word.is_empty() {
        return vec![];
    }
    let mut pieces: Vec<String> = word.chars().map(|c| c.to_string()).collect();

    loop {
        if pieces.len() <= 1 {
            break;
        }
        // Find pair with lowest (earliest) rank.
        let mut best_rank = usize::MAX;
        let mut best_idx  = usize::MAX;
        for i in 0..pieces.len() - 1 {
            let key = (pieces[i].clone(), pieces[i + 1].clone());
            if let Some(&rank) = merge_rank.get(&key) {
                if rank < best_rank {
                    best_rank = rank;
                    best_idx  = i;
                }
            }
        }
        if best_idx == usize::MAX {
            break;
        }
        let merged = format!("{}{}", pieces[best_idx], pieces[best_idx + 1]);
        pieces.remove(best_idx + 1);
        pieces[best_idx] = merged;
    }
    pieces
}

// ── Tokenizer impl ────────────────────────────────────────────────────────────

impl Tokenizer {
    pub fn new() -> Self {
        Self::with_vocab(8_000)
    }

    pub fn with_vocab(target: usize) -> Self {
        let mut t = Self {
            merges:       Vec::new(),
            merge_rank:   HashMap::new(),
            token_to_id:  HashMap::new(),
            id_to_token:  HashMap::new(),
            target_vocab: target,
            frozen:       false,
            word_freq:    HashMap::new(),
        };
        t.add_special("<PAD>",   PAD);
        t.add_special("<UNK>",   UNK);
        t.add_special("<START>", START);
        t.add_special("<END>",   END);
        t
    }

    fn add_special(&mut self, tok: &str, id: usize) {
        self.token_to_id.insert(tok.to_string(), id);
        self.id_to_token.insert(id, tok.to_string());
    }

    pub fn freeze(&mut self) {
        let num_merges = self.target_vocab.saturating_sub(4 + 512);
        // Drop words that appear only once - they add noise and slow BPE down.
        let min_freq = 2;
        let filtered: HashMap<String, usize> = self.word_freq.iter()
            .filter(|(_, &f)| f >= min_freq)
            .map(|(w, &f)| (w.clone(), f))
            .collect();
        println!("BPE: training {} merges on {} unique words (min_freq={})...",
                 num_merges, filtered.len(), min_freq);
        let word_freq = filtered;

        let merges = train_bpe(&word_freq, num_merges);
        self.merge_rank = merges.iter().enumerate()
            .map(|(i, (a, b))| ((a.clone(), b.clone()), i))
            .collect();
        self.merges = merges;

        // Build vocab: collect all chars appearing in corpus.
        let mut chars: HashSet<String> = HashSet::new();
        for word in self.word_freq.keys() {
            for c in word.chars() {
                chars.insert(c.to_string());
            }
        }
        let mut sorted_chars: Vec<String> = chars.into_iter().collect();
        sorted_chars.sort();

        let mut next_id = 4usize;
        for tok in &sorted_chars {
            if !self.token_to_id.contains_key(tok) {
                self.token_to_id.insert(tok.clone(), next_id);
                self.id_to_token.insert(next_id, tok.clone());
                next_id += 1;
            }
        }

        // Add merged tokens in merge order.
        for (a, b) in &self.merges {
            let tok = format!("{}{}", a, b);
            if !self.token_to_id.contains_key(&tok) {
                self.token_to_id.insert(tok.clone(), next_id);
                self.id_to_token.insert(next_id, tok);
                next_id += 1;
            }
        }

        println!("BPE: vocab size = {}", self.token_to_id.len());
        self.frozen = true;
    }

    pub fn encode(&mut self, text: &str) -> Vec<usize> {
        if !self.frozen {
            // Accumulate word frequencies for BPE training.
            for word in text.split_whitespace() {
                let clean = clean_word(word);
                if clean.len() >= 2 {
                    *self.word_freq.entry(clean).or_insert(0) += 1;
                }
            }
            return vec![];
        }

        let mut tokens = vec![START];
        for word in text.split_whitespace() {
            let clean = clean_word(word);
            if clean.is_empty() {
                continue;
            }
            let pieces = apply_merges(&clean, &self.merge_rank);
            for piece in pieces {
                let id = self.token_to_id.get(&piece).copied().unwrap_or(UNK);
                tokens.push(id);
            }
        }
        tokens.push(END);
        tokens
    }

    pub fn vocab_size(&self) -> usize {
        self.token_to_id.len()
    }

    pub fn id_to_word(&self, id: usize) -> Option<String> {
        self.id_to_token.get(&id).cloned()
    }

    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let id_to_word_str: HashMap<String, String> = self.id_to_token
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect();

        let merges_list: Vec<[String; 2]> = self.merges
            .iter()
            .map(|(a, b)| [a.clone(), b.clone()])
            .collect();

        let data = serde_json::json!({
            "version": "bpe",
            "word_to_id": self.token_to_id,
            "id_to_word": id_to_word_str,
            "merges": merges_list,
        });
        fs::write(path, serde_json::to_string(&data)?)?;
        Ok(())
    }

    pub fn load(path: &str) -> anyhow::Result<Self> {
        let content = fs::read_to_string(path)?;
        let data: serde_json::Value = serde_json::from_str(&content)?;

        let mut t = Tokenizer::new();
        t.token_to_id.clear();
        t.id_to_token.clear();

        if let Some(obj) = data["word_to_id"].as_object() {
            for (word, id) in obj {
                if let Some(id_num) = id.as_u64() {
                    t.token_to_id.insert(word.clone(), id_num as usize);
                }
            }
        }

        if let Some(obj) = data["id_to_word"].as_object() {
            for (id_str, word) in obj {
                if let (Ok(id_num), Some(w)) = (id_str.parse::<usize>(), word.as_str()) {
                    t.id_to_token.insert(id_num, w.to_string());
                }
            }
        }

        if let Some(arr) = data["merges"].as_array() {
            for item in arr {
                if let Some(pair) = item.as_array() {
                    if pair.len() == 2 {
                        if let (Some(a), Some(b)) = (pair[0].as_str(), pair[1].as_str()) {
                            t.merges.push((a.to_string(), b.to_string()));
                        }
                    }
                }
            }
        }

        t.merge_rank = t.merges.iter().enumerate()
            .map(|(i, (a, b))| ((a.clone(), b.clone()), i))
            .collect();

        t.frozen = true;
        Ok(t)
    }
}
