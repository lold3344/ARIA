#![recursion_limit = "256"]

use aria::model_cuda::LSTMModelCuda;
use aria::tokenizer::Tokenizer;
use std::env;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    let prompt = args.get(1).cloned().unwrap_or_else(|| "привет".to_string());

    let model_path = "aria json/aria_checkpoint.json";
    let tokenizer_path = "aria json/aria_tokenizer.json";

    println!("Loading tokenizer from {}...", tokenizer_path);
    let mut tokenizer = Tokenizer::load(tokenizer_path)?;
    let vocab = tokenizer.vocab_size();

    println!("Loading checkpoint from {}...", model_path);
    let model = LSTMModelCuda::load_checkpoint(model_path)?;
    println!("Model ready. vocab={} embed={} hidden={}", vocab, model.embed_dim, model.hidden_dim);

    let max_tokens = 50usize;
    let k = 20usize;
    let temperature = 0.7f32;
    let repetition_penalty = 1.2f32;

    println!("Prompt: {}", prompt);
    print!("Response: ");
    let response = model.decode_top_k(&mut tokenizer, &prompt, max_tokens, k, temperature, repetition_penalty);
    println!("{}", response);

    Ok(())
}
