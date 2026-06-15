use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(serde::Deserialize)]
struct DialogRecord { text: String }

fn main() -> anyhow::Result<()> {
    let path = "data base/DataBase_roles.jsonl";
    let f = File::open(path)?;
    let r = BufReader::new(f);

    let mut words: HashMap<String, usize> = HashMap::new();
    let mut total = 0usize;
    for line in r.lines().take(1_000_000) {
        let line = line?;
        if line.trim().is_empty() { continue; }
        let rec: DialogRecord = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(_) => continue,
        };
        if let Some(assistant) = rec.text.split("\n").find(|s| s.trim().starts_with("Ассистент:")) {
            let text = assistant.strip_prefix("Ассистент:").unwrap_or(assistant);
            for word in text.split_whitespace() {
                let w = word.to_lowercase().chars().filter(|c| c.is_alphabetic()).collect::<String>();
                if w.len() >= 3 {
                    *words.entry(w).or_insert(0) += 1;
                    total += 1;
                }
            }
        }
    }

    let mut v: Vec<_> = words.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1));

    println!("Top 30 words in assistant responses ({} total words):", total);
    for (i, (w, c)) in v.iter().take(30).enumerate() {
        println!("{:2}. {} - {} ({:.3}%)", i + 1, w, c, *c as f64 * 100.0 / total.max(1) as f64);
    }
    Ok(())
}
