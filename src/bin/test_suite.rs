#![recursion_limit = "256"]

use aria::model_cuda::LSTMModelCuda;
use aria::tokenizer::Tokenizer;
use std::fs;
use std::time::Instant;

fn run_case(model: &LSTMModelCuda, tokenizer: &mut Tokenizer, name: &str, prompt: &str) -> String {
    let start = Instant::now();
    let response = model.decode_top_k(tokenizer, prompt, 50, 20, 0.7, 1.2);
    let elapsed = start.elapsed().as_secs_f32();
    println!("[{}] {:.2}s prompt: {}\n  -> {}\n", name, elapsed, prompt, response);
    response
}

fn main() -> anyhow::Result<()> {
    let model_path = "aria json/aria_checkpoint.json";
    let tokenizer_path = "aria json/aria_tokenizer.json";

    println!("Loading tokenizer...");
    let mut tokenizer = Tokenizer::load(tokenizer_path)?;
    println!("Loading checkpoint...");
    let model = LSTMModelCuda::load_checkpoint(model_path)?;
    println!("Ready. vocab={}\n", tokenizer.vocab_size());

    fs::create_dir_all("logs")?;

    let mut log = String::new();
    log.push_str(&format!("ARIA validation snapshot\nmodel: {}\ntokenizer: {}\nvocab: {}\n\n",
        model_path, tokenizer_path, tokenizer.vocab_size()));

    let cases = vec![
        ("math_1", "сколько будет 1 плюс 1"),
        ("math_2", "сколько будет 2 плюс 2"),
        ("math_3", "сколько будет 10 минус 3"),
        ("greeting", "как дела"),
        ("what_is_x", "что такое любовь"),
        ("dialog_1", "привет"),
        ("dialog_2", "как тебя зовут"),
        ("dialog_3", "что ты умеешь"),
        ("dialog_4", "расскажи о себе"),
        ("dialog_5", "пока"),
    ];

    for (name, prompt) in cases {
        let response = run_case(&model, &mut tokenizer, name, prompt);
        log.push_str(&format!("[{}] {}\n  -> {}\n\n", name, prompt, response));
    }

    fs::write("logs/validation_log.txt", log)?;
    println!("Results saved to logs/validation_log.txt");
    Ok(())
}
