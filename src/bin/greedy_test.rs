#![recursion_limit = "256"]

use aria::transformer_cuda::TransformerModel;
use aria::tokenizer::Tokenizer;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let model_path = "aria json/aria_checkpoint.json";
    let tokenizer_path = "aria json/aria_tokenizer.json";

    let tokenizer = Tokenizer::load(tokenizer_path)?;
    let mut model = TransformerModel::load_checkpoint(model_path)?;
    model.free_training_buffers();

    let cases = [
        "привет",
        "как дела",
        "сколько будет 1 плюс 1",
        "что такое любовь",
        "расскажи о себе",
    ];

    for prompt in &cases {
        let mut ids = tokenizer.encode_prompt(prompt);
        let mut generated: Vec<usize> = Vec::new();
        let t0 = Instant::now();
        for _ in 0..50 {
            let mut logits = model.forward_gpu(&ids);
            tokenizer.mask_logits(&mut logits);
            let best = model.sample_greedy(&logits);
            if best == 0 || best == 3 || best >= tokenizer.vocab_size() { break; }
            generated.push(best);
            ids.push(best);
            if ids.len() >= 256 { break; }
        }
        let elapsed = t0.elapsed().as_secs_f32();
        let tps = generated.len() as f32 / elapsed.max(0.001);
        let out = tokenizer.decode(&generated);
        println!("[{}] -> {}  ({} tok, {:.1} tok/s)", prompt, out, generated.len(), tps);
    }

    Ok(())
}
