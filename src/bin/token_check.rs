#![recursion_limit = "256"]

use aria::tokenizer::Tokenizer;

fn main() -> anyhow::Result<()> {
    let mut tokenizer = Tokenizer::new();

    // Feed a small dialog corpus to build the vocab
    let corpus = "Пользователь: привет
Ассистент: привет
Пользователь: как дела
Ассистент: хорошо спасибо
Пользователь: сколько будет 1 плюс 1
Ассистент: 2
Пользователь: что ты умеешь
Ассистент: я могу отвечать на вопросы
";
    tokenizer.encode(corpus);
    tokenizer.freeze();

    println!("vocab_size = {}", tokenizer.vocab_size());
    println!("reserved USER id = 5, ASSISTANT id = 6\n");

    for sample in &[
        "Пользователь: привет\nАссистент: привет",
        "Пользователь: как дела\nАссистент: хорошо",
        "Пользователь: сколько будет 1 плюс 1\nАссистент: 2",
    ] {
        let (ids, mask) = tokenizer.encode_dialog(sample);
        let dec = tokenizer.decode(&ids);
        let assistant_tokens: Vec<_> = ids.iter().zip(mask.iter()).filter(|(_, &m)| m > 0.5).map(|(&id, _)| id).collect();
        println!("text: {}", sample);
        println!("ids : {:?}", ids);
        println!("mask: {:?}", mask);
        println!("dec : {}", dec);
        println!("assistant content ids: {:?}", assistant_tokens);
        println!("mask sum: {} / {}\n", mask.iter().sum::<f32>(), mask.len());
    }

    Ok(())
}
