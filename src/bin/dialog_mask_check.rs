#![recursion_limit = "256"]

use aria::tokenizer::Tokenizer;

fn main() -> anyhow::Result<()> {
    let tokenizer_path = "aria json/aria_tokenizer.json";
    let mut tokenizer = Tokenizer::load(tokenizer_path)?;

    let samples = [
        "Пользователь: привет\nАссистент: привет",
        "Пользователь: как дела\nАссистент: хорошо, спасибо",
        "Пользователь: сколько будет 1 плюс 1\nАссистент: 2",
    ];

    for s in &samples {
        let (ids, mask) = tokenizer.encode_dialog(s);
        let dec = tokenizer.decode(&ids);
        println!("text: {}", s);
        println!("ids:  {:?}", ids);
        println!("mask: {:?}", mask);
        println!("dec:  {}", dec);
        println!("mask sum: {} / {}\n", mask.iter().sum::<f32>(), mask.len());
    }

    Ok(())
}
