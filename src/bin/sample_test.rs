#![recursion_limit = "256"]

use aria::model_cuda::LSTMModelCuda;
use aria::tokenizer::Tokenizer;

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

    for &temp in &[0.7f32, 1.0, 1.3] {
        println!("\n=== top-k=20  temp={:.1} ===", temp);
        for prompt in &cases {
            let ids = tokenizer.encode_prompt(prompt);
            let (mut logits, mut state) = model.forward_seq(&ids);
            let mut generated = vec![];
            for _ in 0..40 {
                tokenizer.mask_logits(&mut logits);
                let token = model.sample_top_k(&logits, temp, 20);
                if token == 0 || token == 3 || token >= tokenizer.vocab_size() { break; }
                generated.push(token);
                let (nl, ns) = model.step(token, &state);
                state = ns;
                logits = nl;
            }
            let out = tokenizer.decode(&generated);
            println!("[{}] -> {}", prompt, out);
        }
    }

    Ok(())
}
