use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;
use rayon::prelude::*;
use ocl::{Buffer, Context, Device, DeviceType, Kernel, Platform, Program, Queue};

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
//  GPU BPE kernels (OpenCL — только для BPE training)
// ─────────────────────────────────────────────────────────────
const BPE_KERNELS: &str = r#"
inline int next_live(__global const int* tokens, int pos, int off, int raw_len) {
    for (int k = pos + 1; k < off + raw_len; k++) {
        if (tokens[k] != -1) return k;
    }
    return -1;
}

__kernel void count_pairs(
    __global const int* tokens,
    __global const int* offsets,
    __global const int* raw_lens,
    __global const int* freqs,
    __global       int* pair_counts,
    int vocab_size
) {
    int wi = get_global_id(0);
    int off     = offsets[wi];
    int raw_len = raw_lens[wi];
    int freq    = freqs[wi];
    int prev = -1;
    for (int i = off; i < off + raw_len; i++) {
        int t = tokens[i];
        if (t == -1) continue;
        if (prev >= 0) atomic_add(&pair_counts[prev * vocab_size + t], freq);
        prev = t;
    }
}

__kernel void apply_merge(
    __global       int* tokens,
    __global const int* offsets,
    __global const int* raw_lens,
    int merge_a, int merge_b, int merge_ab
) {
    int wi = get_global_id(0);
    int off     = offsets[wi];
    int raw_len = raw_lens[wi];
    for (int i = off; i < off + raw_len; i++) {
        if (tokens[i] != merge_a) continue;
        int j = next_live(tokens, i, off, raw_len);
        if (j < 0) continue;
        if (tokens[j] == merge_b) { tokens[i] = merge_ab; tokens[j] = -1; }
    }
}

__kernel void reduce_argmax(
    __global const int* counts,
    __global       int* partial_idx,
    __global       int* partial_val,
    int n,
    __local int* lcl_val,
    __local int* lcl_idx
) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int lsz = get_local_size(0);
    int grp = get_group_id(0);
    int val = (gid < n) ? counts[gid] : 0;
    int idx = (gid < n) ? gid : -1;
    lcl_val[lid] = val; lcl_idx[lid] = idx;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsz >> 1; s > 0; s >>= 1) {
        if (lid < s) {
            if (lcl_val[lid+s] > lcl_val[lid] ||
                (lcl_val[lid+s] == lcl_val[lid] && lcl_idx[lid+s] < lcl_idx[lid])) {
                lcl_val[lid] = lcl_val[lid+s];
                lcl_idx[lid] = lcl_idx[lid+s];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) { partial_val[grp] = lcl_val[0]; partial_idx[grp] = lcl_idx[0]; }
}
"#;

struct GpuBpe {
    queue:        Queue,
    #[allow(dead_code)] context: Context,
    #[allow(dead_code)] n_words: usize,
    buf_tokens:   Buffer<i32>,
    buf_offsets:  Buffer<i32>,
    buf_raw_lens: Buffer<i32>,
    buf_freqs:    Buffer<i32>,
    buf_counts:   Buffer<i32>,
    buf_part_idx: Buffer<i32>,
    buf_part_val: Buffer<i32>,
    n_groups:     usize,
    vocab_cap:    usize,
    local_size:   usize,
    k_count:      Kernel,
    k_merge:      Kernel,
    k_argmax:     Kernel,
}

impl GpuBpe {
    fn try_init(flat_tokens: &[i32], offsets: &[i32], raw_lens: &[i32], freqs: &[i32], vocab_cap: usize) -> Option<Self> {
        let platform = Platform::list().into_iter()
            .find(|p| { let n = p.name().unwrap_or_default().to_lowercase();
                        n.contains("nvidia") || n.contains("amd") || n.contains("intel") })
            .or_else(|| Platform::list().into_iter().next())?;
        let device = Device::list(platform, Some(DeviceType::GPU)).ok()?.into_iter().next()?;
        let context = Context::builder().platform(platform).devices(device).build().ok()?;
        let queue   = Queue::new(&context, device, None).ok()?;
        let program = Program::builder().src(BPE_KERNELS).devices(device).build(&context).ok()?;

        let n_words    = offsets.len();
        let pc_size    = vocab_cap * vocab_cap;
        let local_size = 256usize;
        let n_groups   = (pc_size + local_size - 1) / local_size;

        let buf_tokens   = Buffer::<i32>::builder().queue(queue.clone()).len(flat_tokens.len()).copy_host_slice(flat_tokens).build().ok()?;
        let buf_offsets  = Buffer::<i32>::builder().queue(queue.clone()).len(n_words).copy_host_slice(offsets).build().ok()?;
        let buf_raw_lens = Buffer::<i32>::builder().queue(queue.clone()).len(n_words).copy_host_slice(raw_lens).build().ok()?;
        let buf_freqs    = Buffer::<i32>::builder().queue(queue.clone()).len(n_words).copy_host_slice(freqs).build().ok()?;
        let buf_counts   = Buffer::<i32>::builder().queue(queue.clone()).len(pc_size).fill_val(0i32).build().ok()?;
        let buf_part_idx = Buffer::<i32>::builder().queue(queue.clone()).len(n_groups).build().ok()?;
        let buf_part_val = Buffer::<i32>::builder().queue(queue.clone()).len(n_groups).build().ok()?;

        let k_count = Kernel::builder().program(&program).name("count_pairs").queue(queue.clone())
            .global_work_size(n_words)
            .arg(&buf_tokens).arg(&buf_offsets).arg(&buf_raw_lens).arg(&buf_freqs).arg(&buf_counts).arg(0i32)
            .build().ok()?;
        let k_merge = Kernel::builder().program(&program).name("apply_merge").queue(queue.clone())
            .global_work_size(n_words)
            .arg(&buf_tokens).arg(&buf_offsets).arg(&buf_raw_lens).arg(0i32).arg(0i32).arg(0i32)
            .build().ok()?;
        let global_argmax = ((pc_size + local_size - 1) / local_size) * local_size;
        let k_argmax = Kernel::builder().program(&program).name("reduce_argmax").queue(queue.clone())
            .global_work_size(global_argmax).local_work_size(local_size)
            .arg(&buf_counts).arg(&buf_part_idx).arg(&buf_part_val).arg(0i32)
            .arg_local::<i32>(local_size).arg_local::<i32>(local_size)
            .build().ok()?;

        println!("[BPE-GPU] {} | tokens={}K words={} pc_buf={}MB",
            device.name().unwrap_or_default(), flat_tokens.len() / 1000, n_words, pc_size * 4 / 1_000_000);

        Some(GpuBpe { queue, context, n_words, buf_tokens, buf_offsets, buf_raw_lens, buf_freqs,
                      buf_counts, buf_part_idx, buf_part_val, n_groups, vocab_cap, local_size,
                      k_count, k_merge, k_argmax })
    }

    fn ensure_vocab_cap(&mut self, v: usize) -> Option<()> {
        if v <= self.vocab_cap { return Some(()); }
        let pc_size  = v * v;
        let n_groups = (pc_size + self.local_size - 1) / self.local_size;
        self.buf_counts   = Buffer::<i32>::builder().queue(self.queue.clone()).len(pc_size).fill_val(0i32).build().ok()?;
        self.buf_part_idx = Buffer::<i32>::builder().queue(self.queue.clone()).len(n_groups).build().ok()?;
        self.buf_part_val = Buffer::<i32>::builder().queue(self.queue.clone()).len(n_groups).build().ok()?;
        self.k_count.set_arg(4, &self.buf_counts).ok()?;
        let global = ((pc_size + self.local_size - 1) / self.local_size) * self.local_size;
        self.k_argmax.set_arg(0, &self.buf_counts).ok()?;
        self.k_argmax.set_arg(1, &self.buf_part_idx).ok()?;
        self.k_argmax.set_arg(2, &self.buf_part_val).ok()?;
        unsafe { self.k_argmax.set_default_global_work_size(ocl::SpatialDims::One(global)); }
        self.vocab_cap = v;
        self.n_groups  = n_groups;
        Some(())
    }

    fn step(&mut self, vocab_size: usize) -> Option<(usize, usize, i32)> {
        let pc_size = vocab_size * vocab_size;
        self.ensure_vocab_cap(vocab_size)?;
        self.buf_counts.cmd().fill(0i32, Some(pc_size)).enq().ok()?;
        self.k_count.set_arg(5, vocab_size as i32).ok()?;
        unsafe { self.k_count.enq().ok()?; }
        self.k_argmax.set_arg(3, pc_size as i32).ok()?;
        unsafe { self.k_argmax.enq().ok()?; }
        let mut part_val = vec![0i32; self.n_groups];
        let mut part_idx = vec![0i32; self.n_groups];
        self.buf_part_val.read(&mut part_val).enq().ok()?;
        self.buf_part_idx.read(&mut part_idx).enq().ok()?;
        let (best_grp, &freq) = part_val.iter().enumerate()
            .filter(|(_, &v)| v > 0)
            .max_by(|(ia, &va), (ib, &vb)| va.cmp(&vb).then_with(|| ib.cmp(ia)))?;
        if freq == 0 { return None; }
        let idx = part_idx[best_grp] as usize;
        Some((idx / vocab_size, idx % vocab_size, freq))
    }

    fn apply_merge(&mut self, a: i32, b: i32, ab: i32) -> Option<()> {
        self.k_merge.set_arg(3, a).ok()?;
        self.k_merge.set_arg(4, b).ok()?;
        self.k_merge.set_arg(5, ab).ok()?;
        unsafe { self.k_merge.enq().ok()?; }
        Some(())
    }
}

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
//  BPE training
// ─────────────────────────────────────────────────────────────
fn train_bpe(word_freq: &HashMap<String, usize>, num_merges: usize) -> Vec<(String, String)> {
    let words: Vec<(&str, usize)> = word_freq.iter()
        .filter(|(_, &f)| f > 0)
        .map(|(w, &f)| (w.as_str(), f))
        .collect();

    let mut tok_to_id: HashMap<String, u32> = HashMap::new();
    let mut id_to_tok: Vec<String> = Vec::new();

    let get_id = |tok: &str, m: &mut HashMap<String, u32>, v: &mut Vec<String>| -> u32 {
        if let Some(&id) = m.get(tok) { return id; }
        let id = v.len() as u32;
        m.insert(tok.to_string(), id);
        v.push(tok.to_string());
        id
    };

    let mut flat_tokens: Vec<i32> = Vec::new();
    let mut offsets:     Vec<i32> = Vec::new();
    let mut raw_lens:    Vec<i32> = Vec::new();
    let mut freqs:       Vec<i32> = Vec::new();

    for &(word, freq) in &words {
        offsets.push(flat_tokens.len() as i32);
        // Only allow Cyrillic chars in BPE vocab — filter anything else
        let chars: Vec<i32> = word.chars()
            .filter(|c| is_cyrillic(*c) || c.is_ascii_digit())
            .map(|c| get_id(&c.to_string(), &mut tok_to_id, &mut id_to_tok) as i32)
            .collect();
        if chars.is_empty() {
            // pop the offset we just pushed — word has no valid chars
            offsets.pop();
            continue;
        }
        raw_lens.push(chars.len() as i32);
        freqs.push(freq as i32);
        flat_tokens.extend_from_slice(&chars);
    }

    if flat_tokens.is_empty() {
        return Vec::new();
    }

    let vocab_cap = id_to_tok.len() + num_merges + 64;
    let mut gpu = GpuBpe::try_init(&flat_tokens, &offsets, &raw_lens, &freqs, vocab_cap);
    if gpu.is_none() {
        println!("  [BPE] GPU unavailable, falling back to CPU");
        return train_bpe_cpu(word_freq, num_merges);
    }
    let g = gpu.as_mut().unwrap();

    let mut merges: Vec<(String, String)> = Vec::with_capacity(num_merges);

    for step in 0..num_merges {
        let vocab_size = id_to_tok.len();
        let (a_id, b_id, freq) = match g.step(vocab_size) {
            Some(r) => r,
            None => break,
        };

        let a_tok  = id_to_tok[a_id].clone();
        let b_tok  = id_to_tok[b_id].clone();
        let merged = format!("{}{}", a_tok, b_tok);

        // Reject merges that produce non-Cyrillic tokens
        if !merged.chars().all(|c| is_cyrillic(c) || c.is_ascii_digit()) {
            continue;
        }

        let merged_id = get_id(&merged, &mut tok_to_id, &mut id_to_tok) as i32;

        if step % 500 == 0 {
            println!("  BPE step {}/{} best={} (freq={})", step, num_merges, merged, freq);
            let _ = std::io::stdout().flush();
        }

        merges.push((a_tok, b_tok));
        if g.apply_merge(a_id as i32, b_id as i32, merged_id).is_none() {
            println!("  [BPE] apply_merge failed at step {}", step);
            break;
        }
    }

    merges
}

fn train_bpe_cpu(word_freq: &HashMap<String, usize>, num_merges: usize) -> Vec<(String, String)> {
    let mut vocab: Vec<(Vec<String>, usize)> = word_freq.iter()
        .filter(|(_, &f)| f > 0)
        .map(|(w, &f)| (w.chars()
            .filter(|c| is_cyrillic(*c) || c.is_ascii_digit())
            .map(|c| c.to_string()).collect::<Vec<_>>(), f))
        .filter(|(v, _)| !v.is_empty())
        .collect();

    let mut merges = Vec::with_capacity(num_merges);
    for step in 0..num_merges {
        let pair_freq = cpu_count_pairs(&vocab);
        if pair_freq.is_empty() { break; }
        let (best, &freq) = pair_freq.iter()
            .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0))).unwrap();
        if step % 500 == 0 {
            println!("  BPE step {}/{} best={}{} (freq={})", step, num_merges, best.0, best.1, freq);
            let _ = std::io::stdout().flush();
        }
        let merged = format!("{}{}", best.0, best.1);
        if !merged.chars().all(|c| is_cyrillic(c) || c.is_ascii_digit()) { continue; }
        merges.push(best.clone());
        let best = best.clone();
        vocab = vocab.into_par_iter().map(|(word, f)| {
            let mut out = Vec::with_capacity(word.len());
            let mut i = 0;
            while i < word.len() {
                if i + 1 < word.len() && word[i] == best.0 && word[i+1] == best.1 {
                    out.push(merged.clone()); i += 2;
                } else { out.push(word[i].clone()); i += 1; }
            }
            (out, f)
        }).collect();
    }
    merges
}

fn cpu_count_pairs(vocab: &[(Vec<String>, usize)]) -> HashMap<(String, String), usize> {
    vocab.par_iter()
        .fold(HashMap::new, |mut acc, (word, freq)| {
            for i in 0..word.len().saturating_sub(1) {
                *acc.entry((word[i].clone(), word[i+1].clone())).or_insert(0) += freq;
            }
            acc
        })
        .reduce(HashMap::new, |mut a, b| {
            for (k, v) in b { *a.entry(k).or_insert(0) += v; }
            a
        })
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
    pub fn new() -> Self { Self::with_vocab(8_000) }

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
            // right after the ASSISTANT token. The final END target is also trained.
            let from = start;
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
