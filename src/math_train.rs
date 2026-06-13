use std::fs;
use rand::Rng;
use rand::seq::SliceRandom;
use serde_json::Value;

// ─────────────────────────────────────────────────────────────
//  Math sample: a plain text sequence for the LSTM to learn
// ─────────────────────────────────────────────────────────────
pub struct MathSample {
    pub text: String,
    pub difficulty: u8,   // 1-5 for curriculum ordering
}

// ─────────────────────────────────────────────────────────────
//  Curriculum stage configuration
// ─────────────────────────────────────────────────────────────
pub struct CurriculumStage {
    pub name:        &'static str,
    pub max_difficulty: u8,
    pub epochs:      usize,
    pub lr_scale:    f64,
}

pub const CURRICULUM: &[CurriculumStage] = &[
    CurriculumStage { name: "Арифметика",   max_difficulty: 1, epochs: 2, lr_scale: 1.0 },
    CurriculumStage { name: "Алгебра",      max_difficulty: 2, epochs: 2, lr_scale: 0.8 },
    CurriculumStage { name: "Задачи",       max_difficulty: 3, epochs: 2, lr_scale: 0.6 },
    CurriculumStage { name: "Геометрия",    max_difficulty: 4, epochs: 1, lr_scale: 0.5 },
    CurriculumStage { name: "Олимпиад",     max_difficulty: 5, epochs: 1, lr_scale: 0.4 },
];

// ─────────────────────────────────────────────────────────────
//  Synthetic data generator
// ─────────────────────────────────────────────────────────────
pub fn generate_synthetic(n: usize) -> Vec<MathSample> {
    let mut rng = rand::thread_rng();
    let mut out = Vec::with_capacity(n);

    for _ in 0..n {
        let kind: u8 = rng.gen_range(0..10);
        let (text, diff) = match kind {
            // ── уровень 1: базовая арифметика ──────────────────
            0 => {
                let a: i64 = rng.gen_range(1..100);
                let b: i64 = rng.gen_range(1..100);
                (format!("{} + {} = {}", a, b, a + b), 1)
            }
            1 => {
                let a: i64 = rng.gen_range(1..100);
                let b: i64 = rng.gen_range(1..=a);
                (format!("{} - {} = {}", a, b, a - b), 1)
            }
            2 => {
                let a: i64 = rng.gen_range(1..30);
                let b: i64 = rng.gen_range(1..30);
                (format!("{} * {} = {}", a, b, a * b), 1)
            }
            3 => {
                let b: i64 = rng.gen_range(1..20);
                let q: i64 = rng.gen_range(1..20);
                let a = b * q;
                (format!("{} / {} = {}", a, b, q), 1)
            }
            // ── уровень 2: простые уравнения ───────────────────
            4 => {
                let a: i64 = rng.gen_range(1..50);
                let b: i64 = rng.gen_range(1..50);
                let c: i64 = a + b;
                (format!("x + {} = {}, x = {}", b, c, a), 2)
            }
            5 => {
                let a: i64 = rng.gen_range(2..20);
                let x: i64 = rng.gen_range(1..20);
                let b = a * x;
                (format!("{} * x = {}, x = {}", a, b, x), 2)
            }
            6 => {
                let a: i64 = rng.gen_range(1..50);
                let b: i64 = rng.gen_range(1..50);
                let c: i64 = rng.gen_range(1..50);
                (format!("{} + {} + {} = {}", a, b, c, a + b + c), 2)
            }
            // ── уровень 3: цепочки вычислений ──────────────────
            7 => {
                let a: i64 = rng.gen_range(2..20);
                let b: i64 = rng.gen_range(2..10);
                let c: i64 = rng.gen_range(1..30);
                let res = a * b + c;
                (format!("{} * {} + {} = {}", a, b, c, res), 3)
            }
            8 => {
                let a: i64 = rng.gen_range(10..100);
                let b: i64 = rng.gen_range(2..10);
                let c: i64 = rng.gen_range(1..20);
                let res = (a - c) / b;
                let a_adj = res * b + c;
                (format!("({} - {}) / {} = {}", a_adj, c, b, res), 3)
            }
            // ── уровень 4: процент, дроби ───────────────────────
            _ => {
                let whole: i64 = rng.gen_range(10..200);
                let pct: i64 = rng.gen_range(1..50) * 5;
                let res = whole * pct / 100;
                (format!("{}% от {} = {}", pct, whole, res), 4)
            }
        };

        out.push(MathSample { text, difficulty: diff });
    }

    out
}

// ─────────────────────────────────────────────────────────────
//  Load real JSONL data from Math_Learn/
//  Extracts: problem text + step texts + final answer
// ─────────────────────────────────────────────────────────────
pub fn load_jsonl(path: &str) -> Vec<MathSample> {
    let data = match fs::read_to_string(path) {
        Ok(d) => d,
        Err(_) => return Vec::new(),
    };

    let mut out = Vec::new();
    for line in data.lines() {
        let line = line.trim();
        if line.is_empty() { continue; }
        let v: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let problem = v["question"]["problem"].as_str().unwrap_or("").trim().to_string();
        let answer  = v["question"]["ground_truth_answer"].as_str().unwrap_or("").trim().to_string();
        if problem.is_empty() { continue; }

        // Flatten step chain into one reasoning text
        let mut steps_text = String::new();
        if let Some(steps) = v["label"]["steps"].as_array() {
            for step in steps {
                // Prefer chosen_completion text
                if let Some(idx) = step["chosen_completion"].as_u64() {
                    if let Some(text) = step["completions"][idx as usize]["text"].as_str() {
                        if !steps_text.is_empty() { steps_text.push(' '); }
                        steps_text.push_str(text.trim());
                    }
                }
            }
        }

        // Full sequence: "Вопрос: <problem> Решение: <steps> Ответ: <answer>"
        let full = if steps_text.is_empty() {
            format!("Вопрос: {} Ответ: {}", problem, answer)
        } else {
            format!("Вопрос: {} Решение: {} Ответ: {}", problem, steps_text, answer)
        };

        // Rough difficulty based on solution length
        let diff = match full.len() {
            0..=100   => 2,
            101..=300 => 3,
            301..=600 => 4,
            _         => 5,
        };

        out.push(MathSample { text: full, difficulty: diff });
    }

    out
}

// ─────────────────────────────────────────────────────────────
//  Load all Math_Learn JSONL files
// ─────────────────────────────────────────────────────────────
pub fn load_all_math(math_dir: &str) -> Vec<MathSample> {
    let mut all = Vec::new();
    let dir = std::path::Path::new(math_dir);
    if !dir.exists() { return all; }

    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "jsonl") {
                let path_str = path.to_str().unwrap_or("");
                let loaded = load_jsonl(path_str);
                println!("  [Math] {} → {} samples", path.file_name().unwrap().to_str().unwrap_or("?"), loaded.len());
                all.extend(loaded);
            }
        }
    }

    // Add synthetic data
    let synth = generate_synthetic(50_000);
    println!("  [Math] synthetic → {} samples", synth.len());
    all.extend(synth);

    all
}

// ─────────────────────────────────────────────────────────────
//  Filter samples by max difficulty (for curriculum)
// ─────────────────────────────────────────────────────────────
pub fn filter_by_difficulty(samples: &[MathSample], max_diff: u8) -> Vec<&MathSample> {
    samples.iter().filter(|s| s.difficulty <= max_diff).collect()
}

// ─────────────────────────────────────────────────────────────
//  Convert MathSamples → tokenized sequences
// ─────────────────────────────────────────────────────────────
pub fn tokenize_math<'a>(
    samples: &'a [&'a MathSample],
    tokenizer: &mut crate::tokenizer::Tokenizer,
    min_len: usize,
    max_len: usize,
) -> Vec<Vec<usize>> {
    let mut seqs = Vec::new();
    for s in samples {
        let toks = tokenizer.encode(&s.text);
        if toks.len() >= min_len && toks.len() <= max_len {
            seqs.push(toks);
        }
    }
    seqs
}

// ─────────────────────────────────────────────────────────────
//  Full curriculum training loop (called from main/pretrain)
// ─────────────────────────────────────────────────────────────
pub fn train_math_curriculum(
    model: &mut crate::model_cuda::LSTMModelCuda,
    tokenizer: &mut crate::tokenizer::Tokenizer,
    math_dir: &str,
    base_lr: f64,
) -> anyhow::Result<()> {
    println!("\n╔══════════════════════════════════════╗");
    println!("║     ARIA  Math Curriculum Learning   ║");
    println!("╚══════════════════════════════════════╝");

    let all_samples = load_all_math(math_dir);
    println!("Total math samples: {}\n", all_samples.len());

    if all_samples.is_empty() {
        println!("No math data found in {}", math_dir);
        return Ok(());
    }

    let mut rng = rand::thread_rng();

    for stage in CURRICULUM {
        let filtered = filter_by_difficulty(&all_samples, stage.max_difficulty);
        if filtered.is_empty() {
            println!("Stage '{}': no samples, skipping", stage.name);
            continue;
        }

        let lr = base_lr * stage.lr_scale;
        println!("┌─ Stage: {}  (diff ≤ {})  samples: {}  epochs: {}  lr: {:.6}",
                 stage.name, stage.max_difficulty, filtered.len(), stage.epochs, lr);

        let seqs = tokenize_math(&filtered, tokenizer, 4, 256);
        if seqs.is_empty() {
            println!("└─ No sequences after tokenization, skipping\n");
            continue;
        }
        println!("│  Tokenized sequences: {}", seqs.len());

        let batch_size = 256usize;

        for epoch in 0..stage.epochs {
            let mut shuffled = seqs.clone();
            shuffled.shuffle(&mut rng);

            let total_batches = (shuffled.len() + batch_size - 1) / batch_size;
            let mut total_loss = 0.0f32;
            let mut batches = 0usize;
            let t0 = std::time::Instant::now();

            for chunk in shuffled.chunks(batch_size) {
                let loss = model.train_batch(chunk, lr);
                if loss.is_finite() { total_loss += loss; batches += 1; }
            }

            let avg = total_loss / batches.max(1) as f32;
            let seq_s = seqs.len() as f32 / t0.elapsed().as_secs_f32();
            println!("│  epoch {}/{}  loss={:.4}  {:.0} seq/s",
                     epoch + 1, stage.epochs, avg, seq_s);
        }

        println!("└─ done\n");
    }

    println!("Math curriculum complete.\n");
    Ok(())
}
