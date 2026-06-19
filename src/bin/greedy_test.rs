#![recursion_limit = "256"]

use aria::transformer_cuda::TransformerModel;
use aria::tokenizer::Tokenizer;

fn main() -> anyhow::Result<()> {
    let model_path = "aria json/aria_checkpoint.json";
    let tokenizer_path = "aria json/aria_tokenizer.json";

    let mut tokenizer = Tokenizer::load(tokenizer_path)?;
    let model = TransformerModel::load_checkpoint(model_path)?;

    let cases = [
        "привет",
        "как дела",
        "сколько будет 1 плюс 1",
        "что такое любовь",
        "расскажи о себе",
    ];

    for prompt in &cases {
        let ids = tokenizer.encode_prompt(prompt);
        let (mut logits, mut kv) = model.forward_seq(&ids);
        let mut generated = vec![];
        for _ in 0..50 {
            tokenizer.mask_logits(&mut logits);
            let best = model.sample_greedy(&logits);
            if best == 0 || best == 3 || best >= tokenizer.vocab_size() { break; }
            generated.push(best);
            let (nl, nkv) = model.step(best, &kv);
            kv = nkv;
            logits = nl;
        }
        let out = tokenizer.decode(&generated);
        println!("[{}] -> {}", prompt, out);
    }

    Ok(())
}
