#![recursion_limit = "256"]

use aria::model_cuda::LSTMModelCuda;
use aria::tokenizer::Tokenizer;

fn run_topk(model: &LSTMModelCuda, tokenizer: &mut Tokenizer, prompt: &str, temp: f32, k: usize) -> String {
    let ids = tokenizer.encode_prompt(prompt);
    let (mut logits, mut state) = model.forward_seq(&ids);
    let mut generated = vec![];
    for _ in 0..40 {
        tokenizer.mask_logits(&mut logits);
        let token = model.sample_top_k(&logits, temp, k);
        if token == 0 || token == 3 || token >= tokenizer.vocab_size() { break; }
        generated.push(token);
        let (nl, ns) = model.step(token, &state);
        state = ns;
        logits = nl;
    }
    tokenizer.decode(&generated)
}

fn run_topp(model: &LSTMModelCuda, tokenizer: &mut Tokenizer, prompt: &str, temp: f32, p: f32) -> String {
    let ids = tokenizer.encode_prompt(prompt);
    let (mut logits, mut state) = model.forward_seq(&ids);
    let mut generated = vec![];
    for _ in 0..40 {
        tokenizer.mask_logits(&mut logits);
        let token = model.sample_top_p(&logits, temp, p);
        if token == 0 || token == 3 || token >= tokenizer.vocab_size() { break; }
        generated.push(token);
        let (nl, ns) = model.step(token, &state);
        state = ns;
        logits = nl;
    }
    tokenizer.decode(&generated)
}

fn main() -> anyhow::Result<()> {
    let model_path = "aria json/aria_checkpoint.json";
    let tokenizer_path = "aria json/aria_tokenizer.json";

    let mut tokenizer = Tokenizer::load(tokenizer_path)?;
    let model = LSTMModelCuda::load_checkpoint(model_path)?;

    let cases = [
        "привет",
        "как дела",
        "сколько будет 1 плюс 1",
        "что такое любовь",
        "расскажи о себе",
    ];

    // Top-K с разными temp и k
    for &(k, temp) in &[(10, 0.7f32), (20, 0.7), (40, 0.7), (10, 1.0), (20, 1.0), (40, 1.0), (20, 1.3)] {
        println!("\n=== top-k={}  temp={:.1} ===", k, temp);
        for prompt in &cases {
            let out = run_topk(&model, &mut tokenizer, prompt, temp, k);
            println!("  [{}] -> {}", prompt, out);
        }
    }

    // Top-P с разными temp и p
    for &(p, temp) in &[(0.85f32, 0.7f32), (0.9, 0.7), (0.95, 0.7), (0.85, 1.0), (0.9, 1.0), (0.95, 1.0)] {
        println!("\n=== top-p={:.2}  temp={:.1} ===", p, temp);
        for prompt in &cases {
            let out = run_topp(&model, &mut tokenizer, prompt, temp, p);
            println!("  [{}] -> {}", prompt, out);
        }
    }

    Ok(())
}
