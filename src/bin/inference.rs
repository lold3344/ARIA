#![recursion_limit = "256"]

use aria::transformer_cuda::TransformerModel;
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
    let model = TransformerModel::load_checkpoint(model_path)?;
    println!("Model ready. vocab={}", vocab);

    let max_tokens = 50usize;
    let k = 20usize;
    let temperature = 0.7f32;

    println!("Prompt: {}", prompt);
    let ids = tokenizer.encode_prompt(&prompt);
    let (mut logits, mut kv) = model.forward_seq(&ids);
    let mut generated = vec![];
    for step in 0..max_tokens {
        tokenizer.mask_logits(&mut logits);
        let token = model.sample_top_k(&logits, temperature, k);
        if token >= tokenizer.vocab_size() { break; }
        if (token == 0 || token == 3) && step >= 3 { break; }
        if token == 1 { continue; }
        generated.push(token);
        let (nl, nkv) = model.step(token, &kv);
        kv = nkv;
        logits = nl;
    }
    println!("Response: {}", tokenizer.decode(&generated));

    Ok(())
}
