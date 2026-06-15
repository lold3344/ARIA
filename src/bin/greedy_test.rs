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

    for prompt in &cases {
        let ids = tokenizer.encode(prompt);
        let input = &ids[..ids.len().saturating_sub(1)];
        let (mut logits, mut state) = model.forward_seq(input);
        let mut generated = vec![];
        for _ in 0..50 {
            tokenizer.mask_logits(&mut logits);
            let mut best = 0usize;
            let mut best_val = logits[0];
            for (i, &v) in logits.iter().enumerate() {
                if v > best_val { best = i; best_val = v; }
            }
            if best == 0 || best == 3 || best >= tokenizer.vocab_size() { break; }
            generated.push(best);
            let (nl, ns) = model.step(best, &state);
            state = ns;
            logits = nl;
        }
        let out = tokenizer.decode(&generated);
        println!("[{}] -> {}", prompt, out);
    }

    Ok(())
}
