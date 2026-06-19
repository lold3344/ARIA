use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;
use rayon::prelude::*;

const PAD:        usize = 0;
const UNK:        usize = 1;
const START:      usize = 2;
const END:        usize = 3;
const SPACE:      usize = 4;
const USER:       usize = 5;
const ASSISTANT:  usize = 6;
const SPACE_TOKEN:    &str = "<SP>";
const USER_TOKEN:     &str = "<USER>";
const ASSISTANT_TOKEN:&str = "<ASSISTANT>";

const PUNCTUATION: &[&str] = &[".", ",", "!", "?", ";", ":", "\"", "'", "(", ")", "-"];

#[inline]
fn is_punctuation(c: char) -> bool {
    matches!(c, '.' | ',' | '!' | '?' | ';' | ':' | '"' | '\'' | '(' | ')' | '-')
}

// Profanity / vulgar language roots used to filter vocabulary and generation.
const BAD_ROOTS: &[&str] = &[
    "хуй", "хуе", "хуя", "хуи", "хул",
    "пизд", "пидор", "пидар", "пидр",
    "еба", "ебу", "ебе", "ебо", "еби", "ебл", "ебн", "ебя",
    "бля", "сук", "жоп", "муд", "говн", "дерьм", "уеб", "выеб", "заеб",
    "долбоеб", "пиздабол", "охуе", "ахуе", "наху",
    "сос", "соса", "соси", "посос", "сосн", "сосу", "сосет", "сосешь",
];

fn is_bad_token(tok: &str) -> bool {
    if tok.starts_with('<') { return false; }
    let t = tok.to_lowercase();
    BAD_ROOTS.iter().any(|&r| t.contains(r))
}

fn is_bad_word(word: &str) -> bool {
    let w = word.to_lowercase();
    for &r in BAD_ROOTS {
        if let Some(pos) = w.find(r) {
            let left = pos.checked_sub(1).and_then(|i| w.chars().nth(i)).map(|c| c.is_alphabetic()).unwrap_or(false);
            let right = w[pos + r.len()..].chars().next().map(|c| c.is_alphabetic()).unwrap_or(false);
            // Block root if it touches a word boundary (prefix or suffix)
            if !left || !right { return true; }
        }
    }
    false
}

// ─────────────────────────────────────────────────────────────
//  Unicode whitelist: кириллица + базовая пунктуация + цифры
// ─────────────────────────────────────────────────────────────

#[inline]
fn is_allowed_char(c: char) -> bool {
    matches!(c,
        'а'..='я' | 'А'..='Я' | 'ё' | 'Ё'  // кириллица
        | '0'..='9'
        | ' ' | '-' | ',' | '.' | '!' | '?' | ':' | ';' | '\'' | '"' | '(' | ')'
    )
}

#[inline]
fn is_cyrillic(c: char) -> bool {
    matches!(c, 'а'..='я' | 'А'..='Я' | 'ё' | 'Ё')
}

// Возвращает true если строка ≥90% кириллицы (по буквам)
fn cyrillic_ratio(s: &str) -> bool {
    let letters: usize = s.chars().filter(|c| c.is_alphabetic()).count();
    if letters == 0 { return false; }
    let cyr: usize = s.chars().filter(|c| is_cyrillic(*c)).count();
    cyr * 100 / letters >= 90
}

// Очистка слова: только whitelist символы, ё→е, нижний регистр
fn clean_word(w: &str) -> String {
    w.chars()
     .filter(|&c| is_allowed_char(c))
     .map(|c| match c {
         'ё' | 'Ё' => 'е',  // нормализация ё→е для стабильности
         _ => c.to_lowercase().next().unwrap_or(c),
     })
     .collect::<String>()
     .trim_matches(|c: char| !c.is_alphanumeric())
     .to_string()
}

// Фильтрует строку: убирает non-whitelist символы полностью
fn clean_line(s: &str) -> String {
    s.chars().filter(|&c| is_allowed_char(c)).collect()
}

// ─────────────────────────────────────────────────────────────
//  Fast incremental BPE training (CPU + Rayon)
// ─────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────
//  Tokenizer
// ─────────────────────────────────────────────────────────────
pub struct Tokenizer {
    merges:       Vec<(String, String)>,
    merge_rank:   HashMap<(String, String), usize>,
    token_to_id:  HashMap<String, usize>,
    id_to_token:  HashMap<usize, String>,
    // Set of token IDs that pass the Cyrillic whitelist (for constrained decoding)
    allowed_ids:  HashSet<usize>,
    // Set of token IDs that contain profanity / vulgar substrings
    bad_ids:      HashSet<usize>,
    target_vocab: usize,
    frozen:       bool,
    word_freq:    HashMap<String, usize>,
    // Versioning: hash of (sorted merges list) — prevents loading stale caches
    merges_hash:  u64,
}

fn hash_merges(merges: &[(String, String)]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for (a, b) in merges {
        for byte in a.bytes().chain(b.bytes()).chain(std::iter::once(b'|')) {
            h ^= byte as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

// ─────────────────────────────────────────────────────────────
//  BPE training — fast incremental CPU implementation
//
//  Key idea: maintain a pair_freq HashMap and update only the
//  pairs *affected* by each merge, instead of recomputing all
//  pairs from scratch on every step.  This turns O(steps × words)
//  into O(steps × affected_words) which is 100-1000× faster.
//
//  Words are stored as Vec<u32> (token ids) to avoid String
//  heap allocations inside the hot loop.
// ─────────────────────────────────────────────────────────────
fn bpe_get_id(tok: &str, tok_to_id: &mut HashMap<String, u32>, id_to_tok: &mut Vec<String>) -> u32 {
    if let Some(&id) = tok_to_id.get(tok) { return id; }
    let id = id_to_tok.len() as u32;
    tok_to_id.insert(tok.to_string(), id);
    id_to_tok.push(tok.to_string());
    id
}

fn train_bpe(word_freq: &HashMap<String, usize>, num_merges: usize) -> Vec<(String, String)> {
    // ── Build initial token table (single chars) ──────────────
    let mut id_to_tok: Vec<String> = Vec::new();
    let mut tok_to_id: HashMap<String, u32> = HashMap::new();

    // Each word: (token_ids, freq).  Only Cyrillic chars + digits.
    // Build char→id first (single-threaded, needs mut tok_to_id/id_to_tok)
    let mut vocab: Vec<(Vec<u32>, usize)> = word_freq.iter()
        .filter(|(_, &f)| f > 0)
        .filter_map(|(w, &f)| {
            let ids: Vec<u32> = w.chars()
                .filter(|c| is_cyrillic(*c) || c.is_ascii_digit())
                .map(|c| bpe_get_id(&c.to_string(), &mut tok_to_id, &mut id_to_tok))
                .collect();
            if ids.is_empty() { None } else { Some((ids, f)) }
        })
        .collect();

    let n_words = vocab.len();

    // ── Initial pair frequency count (parallel) ───────────────
    let mut pair_freq: HashMap<(u32, u32), i64> = vocab.par_iter()
        .fold(HashMap::new, |mut acc, (ids, f)| {
            for w in ids.windows(2) {
                *acc.entry((w[0], w[1])).or_insert(0) += *f as i64;
            }
            acc
        })
        .reduce(HashMap::new, |mut a, b| {
            for (k, v) in b { *a.entry(k).or_insert(0) += v; }
            a
        });

    // For each pair, which word indices contain it — for incremental update
    // We rebuild this lazily: recount from pair_freq after each merge.
    // To avoid O(n_words) scan per step we maintain pair→word_set.
    // Build it once, then update incrementally.
    let mut pair_to_words: HashMap<(u32, u32), Vec<usize>> = {
        let mut m: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
        for (wi, (ids, _)) in vocab.iter().enumerate() {
            for w in ids.windows(2) {
                m.entry((w[0], w[1])).or_default().push(wi);
            }
        }
        m
    };

    let mut merges: Vec<(String, String)> = Vec::with_capacity(num_merges);

    println!("BPE: {} merges on {} words...", num_merges, n_words);
    let _ = std::io::stdout().flush();

    for step in 0..num_merges {
        // ── Find best pair ────────────────────────────────────
        let best = pair_freq.iter()
            .filter(|(_, &v)| v > 0)
            .max_by(|(pa, &va), (pb, &vb)| {
                va.cmp(&vb).then_with(|| {
                    // tie-break: lexicographic on token strings for determinism
                    let sa = format!("{}{}", id_to_tok[pa.0 as usize], id_to_tok[pa.1 as usize]);
                    let sb = format!("{}{}", id_to_tok[pb.0 as usize], id_to_tok[pb.1 as usize]);
                    sb.cmp(&sa)
                })
            });

        let (&(a_id, b_id), &freq) = match best {
            Some(r) => r,
            None => break,
        };
        if freq <= 0 { break; }

        let a_tok  = id_to_tok[a_id as usize].clone();
        let b_tok  = id_to_tok[b_id as usize].clone();
        let merged = format!("{}{}", a_tok, b_tok);

        // Only allow fully-Cyrillic merges
        if !merged.chars().all(|c| is_cyrillic(c) || c.is_ascii_digit()) {
            pair_freq.insert((a_id, b_id), 0);
            continue;
        }

        let ab_id = bpe_get_id(&merged, &mut tok_to_id, &mut id_to_tok);

        if step % 500 == 0 {
            println!("  BPE step {}/{} best={} (freq={})", step, num_merges, merged, freq);
            let _ = std::io::stdout().flush();
        }

        merges.push((a_tok.clone(), b_tok.clone()));

        // ── Apply merge to affected words only ────────────────
        let affected: Vec<usize> = pair_to_words
            .get(&(a_id, b_id))
            .cloned()
            .unwrap_or_default();

        for wi in &affected {
            let (ids, wf) = &mut vocab[*wi];
            if ids.len() < 2 { continue; }
            let f = *wf as i64;

            let mut i = 0;
            while i < ids.len().saturating_sub(1) {
                if ids[i] == a_id && ids[i + 1] == b_id {
                    // Remove pairs touching this position from pair_freq
                    if i > 0 {
                        *pair_freq.entry((ids[i - 1], ids[i])).or_insert(0) -= f;
                        *pair_freq.entry((ids[i - 1], ab_id)).or_insert(0) += f;
                        pair_to_words.entry((ids[i - 1], ab_id)).or_default().push(*wi);
                    }
                    // ids[i+1] might become ab_id's right neighbour
                    let right_exists = i + 2 < ids.len();
                    if right_exists {
                        *pair_freq.entry((ids[i + 1], ids[i + 2])).or_insert(0) -= f;
                        *pair_freq.entry((ab_id, ids[i + 2])).or_insert(0) += f;
                        pair_to_words.entry((ab_id, ids[i + 2])).or_default().push(*wi);
                    }
                    // Remove the merged pair itself
                    *pair_freq.entry((a_id, b_id)).or_insert(0) -= f;

                    ids[i] = ab_id;
                    ids.remove(i + 1);
                    // Don't advance i — newly merged token may pair with next
                } else {
                    i += 1;
                }
            }
        }

        // Remove exhausted entry
        pair_freq.remove(&(a_id, b_id));
        pair_to_words.remove(&(a_id, b_id));
    }

    merges
}

fn encode_word(word: &str, tokens: &mut Vec<usize>, token_to_id: &HashMap<String, usize>, merge_rank: &HashMap<(String, String), usize>) {
    let w: String = word.chars().filter(|c| is_cyrillic(*c) || c.is_ascii_digit()).collect();
    if w.is_empty() { return; }
    let pieces = apply_merges(&w, merge_rank);
    for piece in pieces {
        if let Some(&id) = token_to_id.get(&piece) {
            tokens.push(id);
        }
    }
}

fn apply_merges(word: &str, merge_rank: &HashMap<(String, String), usize>) -> Vec<String> {
    if word.is_empty() { return vec![]; }
    let mut pieces: Vec<String> = word.chars()
        .filter(|c| is_cyrillic(*c) || c.is_ascii_digit())
        .map(|c| c.to_string())
        .collect();
    loop {
        if pieces.len() <= 1 { break; }
        let mut best_rank = usize::MAX;
        let mut best_idx  = usize::MAX;
        for i in 0..pieces.len() - 1 {
            if let Some(&rank) = merge_rank.get(&(pieces[i].clone(), pieces[i+1].clone())) {
                if rank < best_rank { best_rank = rank; best_idx = i; }
            }
        }
        if best_idx == usize::MAX { break; }
        let merged = format!("{}{}", pieces[best_idx], pieces[best_idx+1]);
        pieces.remove(best_idx + 1);
        pieces[best_idx] = merged;
    }
    pieces
}

// ─────────────────────────────────────────────────────────────
//  Tokenizer impl
// ─────────────────────────────────────────────────────────────
impl Tokenizer {
    pub fn new() -> Self { Self::with_vocab(32_000) }

    pub fn with_vocab(target: usize) -> Self {
        let mut t = Self {
            merges:       Vec::new(),
            merge_rank:   HashMap::new(),
            token_to_id:  HashMap::new(),
            id_to_token:  HashMap::new(),
            allowed_ids:  HashSet::new(),
            bad_ids:      HashSet::new(),
            target_vocab: target,
            frozen:       false,
            word_freq:    HashMap::new(),
            merges_hash:  0,
        };
        t.add_special("<PAD>",   PAD);
        t.add_special("<UNK>",   UNK);
        t.add_special("<START>", START);
        t.add_special("<END>",   END);
        t.add_special(SPACE_TOKEN, SPACE);
        t.add_special(USER_TOKEN, USER);
        t.add_special(ASSISTANT_TOKEN, ASSISTANT);
        t
    }

    fn add_special(&mut self, tok: &str, id: usize) {
        self.token_to_id.insert(tok.to_string(), id);
        self.id_to_token.insert(id, tok.to_string());
    }

    pub fn freeze(&mut self) {
        let num_merges = self.target_vocab.saturating_sub(4 + 512);

        // Only keep words with ≥90% Cyrillic content and min_freq >= 5
        let filtered: HashMap<String, usize> = self.word_freq.iter()
            .filter(|(w, &f)| f >= 5 && cyrillic_ratio(w) && !is_bad_word(w))
            .map(|(w, &f)| (w.clone(), f))
            .collect();

        println!("BPE: training {} merges on {} unique words (min_freq=5, cyrillic≥90%)...",
                 num_merges, filtered.len());

        let merges = train_bpe(&filtered, num_merges);
        self.merges_hash = hash_merges(&merges);
        self.merge_rank  = merges.iter().enumerate()
            .map(|(i, (a, b))| ((a.clone(), b.clone()), i))
            .collect();
        self.merges = merges;

        // Build vocab: only Cyrillic chars from filtered words
        let mut chars: HashSet<String> = HashSet::new();
        for word in filtered.keys() {
            for c in word.chars() {
                if is_cyrillic(c) || c.is_ascii_digit() {
                    chars.insert(c.to_string());
                }
            }
        }
        let mut sorted_chars: Vec<String> = chars.into_iter().collect();
        sorted_chars.sort();

        let mut next_id = 7usize; // 0-6 reserved for special tokens incl. SPACE, USER, ASSISTANT
        for tok in &sorted_chars {
            if !self.token_to_id.contains_key(tok) {
                self.token_to_id.insert(tok.clone(), next_id);
                self.id_to_token.insert(next_id, tok.clone());
                next_id += 1;
            }
        }
        for (a, b) in &self.merges {
            let tok = format!("{}{}", a, b);
            if !self.token_to_id.contains_key(&tok) {
                self.token_to_id.insert(tok.clone(), next_id);
                self.id_to_token.insert(next_id, tok);
                next_id += 1;
            }
        }
        // Punctuation tokens
        for p in PUNCTUATION {
            if !self.token_to_id.contains_key(*p) {
                self.token_to_id.insert(p.to_string(), next_id);
                self.id_to_token.insert(next_id, p.to_string());
                next_id += 1;
            }
        }

        // Build allowed_ids: special tokens, space, punctuation, Cyrillic/digit tokens
        self.allowed_ids = self.token_to_id.iter()
            .filter(|(tok, _)| {
                tok.starts_with('<') // special tokens allowed
                || *tok == SPACE_TOKEN
                || PUNCTUATION.contains(&tok.as_str())
                || tok.chars().all(|c| is_cyrillic(c) || c.is_ascii_digit())
            })
            .map(|(_, &id)| id)
            .collect();

        // Build bad_ids: profanity / vulgar tokens that must not be generated
        self.bad_ids = self.token_to_id.iter()
            .filter(|(tok, _)| is_bad_token(tok))
            .map(|(_, &id)| id)
            .collect();
        self.allowed_ids.retain(|id| !self.bad_ids.contains(id));

        println!("BPE: vocab size = {}  allowed_ids = {}  bad_ids = {}",
                 self.token_to_id.len(), self.allowed_ids.len(), self.bad_ids.len());
        self.frozen = true;
    }

    // Feed a large batch of texts in parallel for vocabulary building.
    // Must be called before freeze(). Much faster than calling encode() line by line.
    pub fn feed_batch(&mut self, texts: &[String]) {
        assert!(!self.frozen, "feed_batch called after freeze");
        // Count word frequencies in parallel, then merge into self.word_freq
        let merged: HashMap<String, usize> = texts.par_iter()
            .fold(HashMap::new, |mut acc, text| {
                let cleaned = clean_line(text);
                for word in cleaned.split_whitespace() {
                    let w = clean_word(word);
                    if w.len() >= 2 && cyrillic_ratio(&w) && !is_bad_word(&w) {
                        *acc.entry(w).or_insert(0) += 1;
                    }
                }
                acc
            })
            .reduce(HashMap::new, |mut a, b| {
                for (k, v) in b { *a.entry(k).or_insert(0) += v; }
                a
            });
        for (k, v) in merged {
            *self.word_freq.entry(k).or_insert(0) += v;
        }
    }

    pub fn encode(&mut self, text: &str) -> Vec<usize> {
        if !self.frozen {
            // Pre-training: accumulate word frequencies — only Russian words
            let cleaned = clean_line(text);
            for word in cleaned.split_whitespace() {
                let w = clean_word(word);
                // UNK not counted in statistics — skip unknown/short/non-Cyrillic
                if w.len() >= 2 && cyrillic_ratio(&w) && !is_bad_word(&w) {
                    *self.word_freq.entry(w).or_insert(0) += 1;
                }
            }
            return vec![];
        }

        let mut tokens = vec![START];
        self.encode_text_to_tokens(&clean_line(&text.replace('\n', " ")), &mut tokens);
        tokens.push(END);
        tokens
    }

    // Encode a user prompt for inference, matching the SFT training format:
    //   <START> <USER> [user tokens] <ASSISTANT>
    // The model then generates tokens starting from the ASSISTANT position.
    pub fn encode_prompt(&self, user_text: &str) -> Vec<usize> {
        let mut tokens = vec![START];
        tokens.push(USER);
        self.encode_text_to_tokens(&clean_line(user_text), &mut tokens);
        tokens.push(ASSISTANT);
        tokens
    }

    // Encode text and build a target mask for supervised dialog fine-tuning.
    // Uses explicit role tokens:
    //   <START> <USER> ... <ASSISTANT> ... <END>
    // mask[t] == 1.0 means the target token at position t (seq[t+1]) is inside the
    // assistant turn and should contribute to the loss. Only supports the format:
    //   Пользователь: ...\nАссистент: ...
    pub fn encode_dialog(&mut self, text: &str) -> (Vec<usize>, Vec<f32>) {
        let mut tokens = vec![START];
        let turns: Vec<&str> = text.split('\n').collect();

        let mut assistant_token_start: Option<usize> = None;
        for turn in turns {
            let trimmed = turn.trim();
            if trimmed.is_empty() { continue; }
            if let Some(rest) = trimmed.strip_prefix("Пользователь:") {
                tokens.push(USER);
                self.encode_text_to_tokens(&clean_line(rest), &mut tokens);
            } else if let Some(rest) = trimmed.strip_prefix("Ассистент:") {
                tokens.push(ASSISTANT);
                assistant_token_start = Some(tokens.len());
                self.encode_text_to_tokens(&clean_line(rest), &mut tokens);
            } else {
                // Unknown turn header: treat as user text (not trained).
                tokens.push(USER);
                self.encode_text_to_tokens(&clean_line(trimmed), &mut tokens);
            }
        }
        tokens.push(END);

        // One mask entry per target position (tokens.len() - 1). Only assistant tokens are
        // trained; the final <END> is also trained so the model learns to stop.
        let mut mask = vec![0.0f32; tokens.len() - 1];
        if let Some(start) = assistant_token_start {
            // To predict the first assistant content token, we train on the target position
            // right after the ASSISTANT token (i.e. at index start - 1, whose target is
            // tokens[start]). The final END target is also trained.
            let from = start.saturating_sub(1);
            let to = tokens.len().saturating_sub(1); // exclusive
            for t in from..to {
                if t < mask.len() { mask[t] = 1.0f32; }
            }
        }
        (tokens, mask)
    }

    fn encode_text_to_tokens(&self, text: &str, tokens: &mut Vec<usize>) {
        let mut word_buf = String::new();
        for c in text.chars() {
            if c.is_alphabetic() || c.is_ascii_digit() {
                word_buf.push(c);
            } else if c == ' ' {
                if !word_buf.is_empty() {
                    encode_word(&word_buf, tokens, &self.token_to_id, &self.merge_rank);
                    word_buf.clear();
                }
                if tokens.last() != Some(&SPACE) {
                    tokens.push(SPACE);
                }
            } else if is_punctuation(c) {
                if !word_buf.is_empty() {
                    encode_word(&word_buf, tokens, &self.token_to_id, &self.merge_rank);
                    word_buf.clear();
                }
                if let Some(&id) = self.token_to_id.get(&c.to_string()) {
                    tokens.push(id);
                }
            }
        }
        if !word_buf.is_empty() {
            encode_word(&word_buf, tokens, &self.token_to_id, &self.merge_rank);
        }
    }

    pub fn vocab_size(&self) -> usize { self.token_to_id.len() }

    pub fn id_to_word(&self, id: usize) -> Option<String> {
        self.id_to_token.get(&id).cloned()
    }

    // Decode a sequence of token IDs back into a string.
    // BPE subwords are concatenated directly; punctuation is preserved; SPACE becomes a real space.
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut out = String::new();
        for &id in ids {
            if id == SPACE {
                out.push(' ');
                continue;
            }
            if let Some(word) = self.id_to_token.get(&id) {
                if word.starts_with('<') { continue; }
                out.push_str(word);
            }
        }
        out
    }

    // Constrained decoding: mask logits of non-Cyrillic and profanity tokens to -inf
    pub fn mask_logits(&self, logits: &mut Vec<f32>) {
        for (id, v) in logits.iter_mut().enumerate() {
            if !self.allowed_ids.contains(&id) || self.bad_ids.contains(&id) {
                *v = f32::NEG_INFINITY;
            }
        }
    }

    // Sanity check: encode → decode roundtrip
    pub fn roundtrip_check(&mut self, text: &str) -> bool {
        let ids = self.encode(text);
        let decoded = self.decode(&ids);
        // Check decoded contains only allowed chars
        decoded.chars().all(|c| is_allowed_char(c) || c == ' ')
    }

    pub fn merges_hash(&self) -> u64 { self.merges_hash }

    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let id_to_word_str: HashMap<String, String> = self.id_to_token.iter()
            .map(|(k, v)| (k.to_string(), v.clone())).collect();
        let merges_list: Vec<[String; 2]> = self.merges.iter()
            .map(|(a, b)| [a.clone(), b.clone()]).collect();
        let data = serde_json::json!({
            "version": "bpe_v3_roles",
            "merges_hash": self.merges_hash,
            "word_to_id":  self.token_to_id,
            "id_to_word":  id_to_word_str,
            "merges":      merges_list,
        });
        fs::write(path, serde_json::to_string(&data)?)?;
        Ok(())
    }

    pub fn load(path: &str) -> anyhow::Result<Self> {
        let data: serde_json::Value = serde_json::from_str(&fs::read_to_string(path)?)?;

        // Version guard — refuse to load old tokenizers
        let ver = data["version"].as_str().unwrap_or("");
        if ver != "bpe_v3_roles" {
            anyhow::bail!("Stale tokenizer (version='{}'), please retrain from scratch", ver);
        }

        let mut t = Tokenizer::new();
        t.token_to_id.clear();
        t.id_to_token.clear();

        if let Some(obj) = data["word_to_id"].as_object() {
            for (word, id) in obj {
                if let Some(n) = id.as_u64() {
                    t.token_to_id.insert(word.clone(), n as usize);
                }
            }
        }
        if let Some(obj) = data["id_to_word"].as_object() {
            for (id_str, word) in obj {
                if let (Ok(n), Some(w)) = (id_str.parse::<usize>(), word.as_str()) {
                    t.id_to_token.insert(n, w.to_string());
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

        t.merges_hash = data["merges_hash"].as_u64().unwrap_or(0);
        t.merge_rank  = t.merges.iter().enumerate()
            .map(|(i, (a, b))| ((a.clone(), b.clone()), i)).collect();

        // Ensure reserved special tokens exist in a loaded tokenizer
        if !t.token_to_id.contains_key(SPACE_TOKEN) {
            t.token_to_id.insert(SPACE_TOKEN.to_string(), SPACE);
            t.id_to_token.insert(SPACE, SPACE_TOKEN.to_string());
        }
        if !t.token_to_id.contains_key(USER_TOKEN) {
            t.token_to_id.insert(USER_TOKEN.to_string(), USER);
            t.id_to_token.insert(USER, USER_TOKEN.to_string());
        }
        if !t.token_to_id.contains_key(ASSISTANT_TOKEN) {
            t.token_to_id.insert(ASSISTANT_TOKEN.to_string(), ASSISTANT);
            t.id_to_token.insert(ASSISTANT, ASSISTANT_TOKEN.to_string());
        }
        let mut next_id = t.token_to_id.len();
        for p in PUNCTUATION {
            if !t.token_to_id.contains_key(*p) {
                t.token_to_id.insert(p.to_string(), next_id);
                t.id_to_token.insert(next_id, p.to_string());
                next_id += 1;
            }
        }

        // Rebuild allowed_ids and bad_ids
        t.allowed_ids = t.token_to_id.iter()
            .filter(|(tok, _)| {
                tok.starts_with('<')
                || *tok == SPACE_TOKEN
                || PUNCTUATION.contains(&tok.as_str())
                || tok.chars().all(|c| is_cyrillic(c) || c.is_ascii_digit())
            })
            .map(|(_, &id)| id)
            .collect();
        t.bad_ids = t.token_to_id.iter()
            .filter(|(tok, _)| is_bad_token(tok))
            .map(|(_, &id)| id)
            .collect();
        t.allowed_ids.retain(|id| !t.bad_ids.contains(id));

        t.frozen = true;
        Ok(t)
    }
}
