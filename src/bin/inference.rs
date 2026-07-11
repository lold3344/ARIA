#![recursion_limit = "256"]

use aria::transformer_cuda::TransformerModel;
use std::env;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    let prompt = args.get(1).cloned().unwrap_or_else(|| "привет".to_string());

    let model_path = "aria json/aria_checkpoint.gguf";

    println!("Loading checkpoint from {}...", model_path);
    let (mut model, tokenizer) = TransformerModel::load_checkpoint(model_path)?;
    let vocab = tokenizer.vocab_size();
    model.free_training_buffers();
    println!("Model ready (inference mode). vocab={}", vocab);

    let max_tokens = 50usize;
    let k = 20usize;
    let temperature = 0.7f32;
    let repeat_penalty = 1.3f32;

    println!("Prompt: {}", prompt);
    let mut ids = tokenizer.encode_prompt(&prompt);
    let mut generated: Vec<usize> = Vec::new();

    let t0 = Instant::now();
    for step in 0..max_tokens {
        let mut logits = model.forward_gpu(&ids);
        tokenizer.mask_logits(&mut logits);

        for &prev in &generated {
            if prev < logits.len() {
                if logits[prev] > 0.0 { logits[prev] /= repeat_penalty; }
                else { logits[prev] *= repeat_penalty; }
            }
        }

        let token = model.sample_top_k(&logits, temperature, k);
        if token >= vocab { break; }
        if (token == 0 || token == 3) && step >= 3 { break; }
        if token == 1 { continue; }

        generated.push(token);
        ids.push(token);
        if ids.len() >= 256 { break; }
    }
    let elapsed = t0.elapsed().as_secs_f32();
    let tps = generated.len() as f32 / elapsed.max(0.001);

    println!("Response: {}", tokenizer.decode(&generated));
    println!("[{} tokens in {:.2}s = {:.1} tok/s]", generated.len(), elapsed, tps);

    Ok(())
}
