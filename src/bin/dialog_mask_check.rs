#![recursion_limit = "256"]

fn main() -> anyhow::Result<()> {
    let checkpoint_path = "aria json/aria_checkpoint.gguf";
    println!("Loading tokenizer from embedded GGUF checkpoint: {}", checkpoint_path);
    let (_, mut tokenizer) = aria::transformer_cuda::TransformerModel::load_checkpoint(checkpoint_path)?;

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
