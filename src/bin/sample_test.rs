#![recursion_limit = "256"]

use aria::transformer_cuda::TransformerModel;
use aria::tokenizer::Tokenizer;

fn run_topk(model: &mut TransformerModel, tokenizer: &Tokenizer, prompt: &str, temp: f32, k: usize) -> String {
    let mut ids = tokenizer.encode_prompt(prompt);
    let mut generated: Vec<usize> = Vec::new();
    for _ in 0..40 {
        let mut logits = model.forward_gpu(&ids);
        tokenizer.mask_logits(&mut logits);
        let token = model.sample_top_k(&logits, temp, k);
        if token == 0 || token == 3 || token >= tokenizer.vocab_size() { break; }
        generated.push(token);
        ids.push(token);
        if ids.len() >= 256 { break; }
    }
    tokenizer.decode(&generated)
}

fn run_topp(model: &mut TransformerModel, tokenizer: &Tokenizer, prompt: &str, temp: f32, p: f32) -> String {
    let mut ids = tokenizer.encode_prompt(prompt);
    let mut generated: Vec<usize> = Vec::new();
    for _ in 0..40 {
        let mut logits = model.forward_gpu(&ids);
        tokenizer.mask_logits(&mut logits);
        let token = model.sample_top_p(&logits, temp, p);
        if token == 0 || token == 3 || token >= tokenizer.vocab_size() { break; }
        generated.push(token);
        ids.push(token);
        if ids.len() >= 256 { break; }
    }
    tokenizer.decode(&generated)
}

fn main() -> anyhow::Result<()> {
    let model_path = "aria json/aria_checkpoint.gguf";

    let (mut model, tokenizer) = TransformerModel::load_checkpoint(model_path)?;
    model.free_training_buffers();

    let cases = [
        "привет",
        "как дела",
        "сколько будет 1 плюс 1",
        "что такое любовь",
        "расскажи о себе",
    ];

    for &(k, temp) in &[(10, 0.7f32), (20, 0.7), (20, 1.0), (20, 1.3)] {
        println!("\n=== top-k={}  temp={:.1} ===", k, temp);
        for prompt in &cases {
            let out = run_topk(&mut model, &tokenizer, prompt, temp, k);
            println!("  [{}] -> {}", prompt, out);
        }
    }

    for &(p, temp) in &[(0.9f32, 0.7f32), (0.9, 1.0)] {
        println!("\n=== top-p={:.2}  temp={:.1} ===", p, temp);
        for prompt in &cases {
            let out = run_topp(&mut model, &tokenizer, prompt, temp, p);
            println!("  [{}] -> {}", prompt, out);
        }
    }

    Ok(())
}
