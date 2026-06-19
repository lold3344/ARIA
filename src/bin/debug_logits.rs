#![recursion_limit = "256"]

use aria::transformer_cuda::TransformerModel;
use aria::tokenizer::Tokenizer;

fn main() -> anyhow::Result<()> {
    let model_path = "aria json/aria_checkpoint.json";
    let tokenizer_path = "aria json/aria_tokenizer.json";

    let mut tokenizer = Tokenizer::load(tokenizer_path)?;
    let model = TransformerModel::load_checkpoint(model_path)?;

    let prompt = "привет";
    let ids = tokenizer.encode_prompt(prompt);
    println!("Prompt tokens: {:?}", ids);

    let (logits0, state0) = model.forward_seq(&ids);

    let mut logits0m = logits0.clone();
    tokenizer.mask_logits(&mut logits0m);
    println!("\n--- Logits for special token ids 0..=6 after ASSISTANT ---");
    for id in 0..=6 {
        println!("  id={} ({:?})  raw={:.4}  masked={}",
            id, tokenizer.id_to_word(id),
            logits0[id],
            if logits0m[id].is_finite() { format!("{:.4}", logits0m[id]) } else { "-inf".to_string() });
    }

    println!("\n--- Generation trace (5 steps) ---");
    let mut logits = logits0;
    let mut state = state0;
    for step in 0..5 {
        let mut masked = logits.clone();
        tokenizer.mask_logits(&mut masked);

        let mut best = 0usize;
        let mut best_val = masked[0];
        for (i, &v) in masked.iter().enumerate() {
            if v > best_val { best = i; best_val = v; }
        }

        println!("  step {}: best_id={} logit={:.4} token={:?}  {}",
            step, best, best_val, tokenizer.id_to_word(best),
            if best == 0 { "→ BREAK (PAD)" } else if best == 3 { "→ BREAK (END)" } else { "→ push" });

        if best == 0 || best == 3 || best >= tokenizer.vocab_size() { break; }
        let (nl, ns) = model.step(best, &state);
        state = ns;
        logits = nl;
    }

    Ok(())
}
