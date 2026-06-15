#![recursion_limit = "256"]

use aria::tokenizer::Tokenizer;
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    let tokenizer_path = "aria json/aria_tokenizer.json";
    let t1 = Tokenizer::load(tokenizer_path)?;
    let t2 = Tokenizer::load(tokenizer_path)?;

    println!("vocab_size load1 = {}", t1.vocab_size());
    println!("vocab_size load2 = {}", t2.vocab_size());
    assert_eq!(t1.vocab_size(), t2.vocab_size(), "vocab size unstable");

    let mut id_map1: HashMap<usize, String> = HashMap::new();
    let mut id_map2: HashMap<usize, String> = HashMap::new();
    for id in 0..t1.vocab_size() {
        if let Some(w) = t1.id_to_word(id) { id_map1.insert(id, w); }
        if let Some(w) = t2.id_to_word(id) { id_map2.insert(id, w); }
    }
    for id in 0..t1.vocab_size() {
        assert_eq!(id_map1.get(&id), id_map2.get(&id), "token id {} differs between loads", id);
    }
    println!("token IDs stable across loads");

    let samples = [
        "привет",
        "как дела",
        "сколько будет 1 плюс 1",
        "Пользователь: привет\nАссистент: привет",
        "расскажи о себе. что ты умеешь?",
        "пока, до встречи!",
    ];
    for s in &samples {
        let mut enc = Tokenizer::load(tokenizer_path)?;
        let ids = enc.encode(s);
        let dec = enc.decode(&ids);
        println!("in:  {}", s);
        println!("ids: {:?}", ids);
        println!("out: {}", dec);
        println!("rt:  {}\n", if dec == *s { "OK" } else { "MISMATCH" });
    }

    Ok(())
}
