use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() -> anyhow::Result<()> {
    let path = "data base/DataBase_roles.jsonl";
    let f = File::open(path)?;
    let r = BufReader::new(f);

    let mut total = 0usize;
    let mut user_tokens = 0usize;
    let mut assistant_tokens = 0usize;
    let mut turns_counts = vec![];
    let mut total_lines = 0usize;
    let mut broken = 0usize;

    for line in r.lines() {
        total_lines += 1;
        let line = match line {
            Ok(l) => l,
            Err(_) => { broken += 1; continue; }
        };
        if line.trim().is_empty() { continue; }
        let v: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => { broken += 1; continue; }
        };
        let text = v.get("text").and_then(|x| x.as_str()).unwrap_or("");
        if text.is_empty() { continue; }
        total += 1;

        let turns = text.split("\n").count();
        turns_counts.push(turns);

        for part in text.split("\n") {
            if part.starts_with("Пользователь:") {
                user_tokens += part.len();
            } else if part.starts_with("Ассистент:") {
                assistant_tokens += part.len();
            }
        }

        if total % 100_000 == 0 {
            println!("processed {} lines", total);
        }
    }

    println!("\n=== Data Audit ===");
    println!("total lines: {}", total_lines);
    println!("valid records: {}", total);
    println!("broken/empty: {}", broken);
    println!("user_chars: {}  assistant_chars: {}  ratio: {:.3}",
        user_tokens, assistant_tokens, user_tokens as f64 / assistant_tokens.max(1) as f64);

    turns_counts.sort();
    let n = turns_counts.len();
    println!("turns: min={} max={} median={} mean={:.2}",
        turns_counts[0], turns_counts[n-1], turns_counts[n/2],
        turns_counts.iter().sum::<usize>() as f64 / n as f64);

    Ok(())
}
