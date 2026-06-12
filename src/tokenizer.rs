use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;
use rayon::prelude::*;
use ocl::{Buffer, Context, Device, DeviceType, Kernel, Platform, Program, Queue};

const PAD: usize   = 0;
const UNK: usize   = 1;
const START: usize = 2;
const END: usize   = 3;

// ── GPU-resident BPE kernels ──────────────────────────────────────────────────
//
// Tokens are stored as a flat i32 array on GPU the ENTIRE training run.
// Merged tokens are marked -1 (tombstone) in place - no memory moves.
//
// count_pairs: each work-item = one word.
//   Skips tombstones, accumulates freq into pair_counts[a * vocab_size + b].
//
// apply_merge: each work-item = one position in flat token array.
//   If tokens[i]==a and next non-tombstone == b, replace tokens[i]=merged,
//   mark tokens[j]=-1.
//
// reduce_argmax: two-pass parallel reduction over pair_counts[V*V].
//   Pass 1: each work-group finds its local max -> writes to partial[group].
//   Pass 2: single work-group reduces partial[] -> out[0]=idx, out[1]=freq.
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
        if (prev >= 0) {
            atomic_add(&pair_counts[prev * vocab_size + t], freq);
        }
        prev = t;
    }
}

__kernel void apply_merge(
    __global       int* tokens,
    __global const int* offsets,
    __global const int* raw_lens,
    int merge_a,
    int merge_b,
    int merge_ab
) {
    int wi = get_global_id(0);
    int off     = offsets[wi];
    int raw_len = raw_lens[wi];

    for (int i = off; i < off + raw_len; i++) {
        if (tokens[i] != merge_a) continue;
        int j = next_live(tokens, i, off, raw_len);
        if (j < 0) continue;
        if (tokens[j] == merge_b) {
            tokens[i] = merge_ab;
            tokens[j] = -1;
        }
    }
}

// Parallel reduction: find argmax of pair_counts[0..n].
// Each work-group reduces its chunk into partial_idx/partial_val.
// Call with global_size = n (rounded up to multiple of local_size).
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
    lcl_val[lid] = val;
    lcl_idx[lid] = idx;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = lsz >> 1; s > 0; s >>= 1) {
        if (lid < s) {
            if (lcl_val[lid + s] > lcl_val[lid] ||
                (lcl_val[lid + s] == lcl_val[lid] && lcl_idx[lid + s] < lcl_idx[lid])) {
                lcl_val[lid] = lcl_val[lid + s];
                lcl_idx[lid] = lcl_idx[lid + s];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        partial_val[grp] = lcl_val[0];
        partial_idx[grp] = lcl_idx[0];
    }
}
"#;

// GPU-resident BPE state. Flat token array lives on GPU the whole training run.
// Tombstones (-1) replace the second token of each merged pair in-place.
struct GpuBpe {
    queue:        Queue,
    context:      Context,
    n_words:      usize,
    // GPU buffers (never reallocated after init)
    buf_tokens:   Buffer<i32>,
    buf_offsets:  Buffer<i32>,
    buf_raw_lens: Buffer<i32>,
    buf_freqs:    Buffer<i32>,
    buf_counts:   Buffer<i32>,
    buf_part_idx: Buffer<i32>,  // partial argmax results
    buf_part_val: Buffer<i32>,
    n_groups:     usize,
    vocab_cap:    usize,
    local_size:   usize,
    // pre-compiled kernels
    k_count:      Kernel,
    k_merge:      Kernel,
    k_argmax:     Kernel,
}

impl GpuBpe {
    fn try_init(
        flat_tokens: &[i32],
        offsets: &[i32],
        raw_lens: &[i32],
        freqs: &[i32],
        vocab_cap: usize,
    ) -> Option<Self> {
        let platform = Platform::list().into_iter()
            .find(|p| {
                let name = p.name().unwrap_or_default().to_lowercase();
                name.contains("nvidia") || name.contains("amd") || name.contains("intel")
            })
            .or_else(|| Platform::list().into_iter().next())?;
        let device = Device::list(platform, Some(DeviceType::GPU))
            .ok()?.into_iter().next()?;
        let context = Context::builder()
            .platform(platform).devices(device).build().ok()?;
        let queue = Queue::new(&context, device, None).ok()?;

        let program = Program::builder()
            .src(BPE_KERNELS).devices(device).build(&context).ok()?;

        let n_words  = offsets.len();
        let pc_size  = vocab_cap * vocab_cap;
        let local_size: usize = 256;
        let n_groups = (pc_size + local_size - 1) / local_size;

        let buf_tokens = Buffer::<i32>::builder()
            .queue(queue.clone()).len(flat_tokens.len())
            .copy_host_slice(flat_tokens).build().ok()?;
        let buf_offsets = Buffer::<i32>::builder()
            .queue(queue.clone()).len(n_words)
            .copy_host_slice(offsets).build().ok()?;
        let buf_raw_lens = Buffer::<i32>::builder()
            .queue(queue.clone()).len(n_words)
            .copy_host_slice(raw_lens).build().ok()?;
        let buf_freqs = Buffer::<i32>::builder()
            .queue(queue.clone()).len(n_words)
            .copy_host_slice(freqs).build().ok()?;
        let buf_counts = Buffer::<i32>::builder()
            .queue(queue.clone()).len(pc_size).fill_val(0i32).build().ok()?;
        let buf_part_idx = Buffer::<i32>::builder()
            .queue(queue.clone()).len(n_groups).build().ok()?;
        let buf_part_val = Buffer::<i32>::builder()
            .queue(queue.clone()).len(n_groups).build().ok()?;

        let k_count = Kernel::builder()
            .program(&program).name("count_pairs")
            .queue(queue.clone())
            .global_work_size(n_words)
            .arg(&buf_tokens)
            .arg(&buf_offsets)
            .arg(&buf_raw_lens)
            .arg(&buf_freqs)
            .arg(&buf_counts)
            .arg(0i32)
            .build().ok()?;

        let k_merge = Kernel::builder()
            .program(&program).name("apply_merge")
            .queue(queue.clone())
            .global_work_size(n_words)
            .arg(&buf_tokens)
            .arg(&buf_offsets)
            .arg(&buf_raw_lens)
            .arg(0i32)
            .arg(0i32)
            .arg(0i32)
            .build().ok()?;

        let global_argmax = ((pc_size + local_size - 1) / local_size) * local_size;
        let k_argmax = Kernel::builder()
            .program(&program).name("reduce_argmax")
            .queue(queue.clone())
            .global_work_size(global_argmax)
            .local_work_size(local_size)
            .arg(&buf_counts)
            .arg(&buf_part_idx)
            .arg(&buf_part_val)
            .arg(0i32) // n - set each step
            .arg_local::<i32>(local_size)
            .arg_local::<i32>(local_size)
            .build().ok()?;

        println!("[BPE-GPU] {} | tokens={}K words={} pc_buf={}MB",
            device.name().unwrap_or_default(),
            flat_tokens.len() / 1000,
            n_words,
            pc_size * 4 / 1_000_000);

        Some(GpuBpe {
            queue, context, n_words,
            buf_tokens, buf_offsets, buf_raw_lens, buf_freqs, buf_counts,
            buf_part_idx, buf_part_val, n_groups,
            vocab_cap, local_size,
            k_count, k_merge, k_argmax,
        })
    }

    fn ensure_vocab_cap(&mut self, v: usize) -> Option<()> {
        if v <= self.vocab_cap { return Some(()); }
        let pc_size  = v * v;
        let n_groups = (pc_size + self.local_size - 1) / self.local_size;
        self.buf_counts = Buffer::<i32>::builder()
            .queue(self.queue.clone()).len(pc_size).fill_val(0i32).build().ok()?;
        self.buf_part_idx = Buffer::<i32>::builder()
            .queue(self.queue.clone()).len(n_groups).build().ok()?;
        self.buf_part_val = Buffer::<i32>::builder()
            .queue(self.queue.clone()).len(n_groups).build().ok()?;
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

    // One BPE step fully on GPU. Returns (best_a_id, best_b_id, freq).
    fn step(&mut self, vocab_size: usize) -> Option<(usize, usize, i32)> {
        let pc_size = vocab_size * vocab_size;
        self.ensure_vocab_cap(vocab_size)?;

        // Zero counts, count pairs.
        self.buf_counts.cmd().fill(0i32, Some(pc_size)).enq().ok()?;
        self.k_count.set_arg(5, vocab_size as i32).ok()?;
        unsafe { self.k_count.enq().ok()?; }

        // GPU argmax reduction - reads only n_groups*8 bytes back (~2KB).
        self.k_argmax.set_arg(3, pc_size as i32).ok()?;
        unsafe { self.k_argmax.enq().ok()?; }

        let mut part_val = vec![0i32; self.n_groups];
        let mut part_idx = vec![0i32; self.n_groups];
        self.buf_part_val.read(&mut part_val).enq().ok()?;
        self.buf_part_idx.read(&mut part_idx).enq().ok()?;

        // Final reduce on CPU over n_groups results (negligible).
        let (best_grp, &freq) = part_val.iter().enumerate()
            .filter(|(_, &v)| v > 0)
            .max_by(|(ia, &va), (ib, &vb)| va.cmp(&vb).then_with(|| ib.cmp(ia)))?;
        if freq == 0 { return None; }
        let idx = part_idx[best_grp] as usize;

        Some((idx / vocab_size, idx % vocab_size, freq))
    }

    // Apply merge on GPU in-place (tombstone pattern).
    fn apply_merge(&mut self, a: i32, b: i32, ab: i32) -> Option<()> {
        self.k_merge.set_arg(3, a).ok()?;
        self.k_merge.set_arg(4, b).ok()?;
        self.k_merge.set_arg(5, ab).ok()?;
        unsafe { self.k_merge.enq().ok()?; }
        Some(())
    }
}

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

// BPE training: tokens live on GPU for entire run, CPU only does argmax.
fn train_bpe(word_freq: &HashMap<String, usize>, num_merges: usize) -> Vec<(String, String)> {
    // Build initial char-level vocab.
    let words: Vec<(&str, usize)> = word_freq.iter()
        .filter(|(_, &f)| f > 0)
        .map(|(w, &f)| (w.as_str(), f))
        .collect();

    // Token <-> u32 ID mapping (CPU side, grows as merges happen).
    let mut tok_to_id: HashMap<String, u32> = HashMap::new();
    let mut id_to_tok: Vec<String> = Vec::new();

    let mut get_id = |tok: &str,
                      m: &mut HashMap<String, u32>,
                      v: &mut Vec<String>| -> u32 {
        if let Some(&id) = m.get(tok) { return id; }
        let id = v.len() as u32;
        m.insert(tok.to_string(), id);
        v.push(tok.to_string());
        id
    };

    // Build flat token arrays for GPU upload.
    let mut flat_tokens: Vec<i32> = Vec::new();
    let mut offsets:  Vec<i32> = Vec::new();
    let mut raw_lens: Vec<i32> = Vec::new();
    let mut freqs:    Vec<i32> = Vec::new();

    for &(word, freq) in &words {
        offsets.push(flat_tokens.len() as i32);
        let chars: Vec<i32> = word.chars().map(|c| {
            get_id(&c.to_string(), &mut tok_to_id, &mut id_to_tok) as i32
        }).collect();
        raw_lens.push(chars.len() as i32);
        freqs.push(freq as i32);
        flat_tokens.extend_from_slice(&chars);
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

        let a_tok = id_to_tok[a_id].clone();
        let b_tok = id_to_tok[b_id].clone();
        let merged = format!("{}{}", a_tok, b_tok);
        let merged_id = get_id(&merged, &mut tok_to_id, &mut id_to_tok) as i32;

        if step % 500 == 0 {
            println!("  BPE step {}/{} best={} (freq={})",
                step, num_merges, merged, freq);
            let _ = std::io::stdout().flush();
        }

        merges.push((a_tok, b_tok));
        if g.apply_merge(a_id as i32, b_id as i32, merged_id).is_none() {
            println!("  [BPE] apply_merge failed at step {}, aborting GPU", step);
            break;
        }
    }

    merges
}

// Pure CPU fallback (rayon) - same algorithm as before.
fn train_bpe_cpu(word_freq: &HashMap<String, usize>, num_merges: usize) -> Vec<(String, String)> {
    let mut vocab: Vec<(Vec<String>, usize)> = word_freq.iter()
        .filter(|(_, &f)| f > 0)
        .map(|(w, &f)| (w.chars().map(|c| c.to_string()).collect(), f))
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
        merges.push(best.clone());
        let best = best.clone();
        vocab = vocab.into_par_iter().map(|(word, freq)| {
            let mut out = Vec::with_capacity(word.len());
            let mut i = 0;
            while i < word.len() {
                if i + 1 < word.len() && word[i] == best.0 && word[i+1] == best.1 {
                    out.push(merged.clone()); i += 2;
                } else { out.push(word[i].clone()); i += 1; }
            }
            (out, freq)
        }).collect();
    }
    merges
}

fn cpu_count_pairs(vocab: &[(Vec<String>, usize)]) -> HashMap<(String, String), usize> {
    vocab
        .par_iter()
        .fold(HashMap::new, |mut acc, (word, freq)| {
            for i in 0..word.len().saturating_sub(1) {
                let pair = (word[i].clone(), word[i + 1].clone());
                *acc.entry(pair).or_insert(0) += freq;
            }
            acc
        })
        .reduce(HashMap::new, |mut a, b| {
            for (k, v) in b { *a.entry(k).or_insert(0) += v; }
            a
        })
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
