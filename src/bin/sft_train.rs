#![recursion_limit = "256"]

use aria::model_cuda::LSTMModelCuda;
use aria::tokenizer::Tokenizer;

fn main() -> anyhow::Result<()> {
    let model_path = "aria json/aria_checkpoint.json";
    let tokenizer_path = "aria json/aria_tokenizer.json";
    let data_dir = "data base";

    let mut tokenizer = Tokenizer::load(tokenizer_path)?;
    let mut model = LSTMModelCuda::load_checkpoint(model_path)?;

    println!("Starting supervised dialog fine-tuning from checkpoint...");
    aria::model_cuda::pretrain_from_files(&mut model, &mut tokenizer, data_dir, model_path, tokenizer_path)?;

    println!("\nFine-tuning complete. Running greedy test...");
    let prompts = [
        "Пользователь: привет\nАссистент:",
        "Пользователь: как дела\nАссистент:",
        "Пользователь: сколько будет 1 плюс 1\nАссистент:",
        "Пользователь: что ты умеешь\nАссистент:",
        "Пользователь: расскажи о себе\nАссистент:",
    ];
    for prompt in &prompts {
        let ids = tokenizer.encode(prompt);
        let input = &ids[..ids.len().saturating_sub(1)];
        let (mut logits, mut state) = model.forward_seq(input);
        let mut generated = vec![];
        for _ in 0..40 {
            let mut masked = logits.clone();
            tokenizer.mask_logits(&mut masked);
            let mut best = 0usize;
            let mut best_val = masked[0];
            for (i, &v) in masked.iter().enumerate() {
                if v > best_val { best = i; best_val = v; }
            }
            if best == 0 || best == 3 || best >= tokenizer.vocab_size() { break; }
            generated.push(best);
            let (nl, ns) = model.step(best, &state);
            state = ns;
            logits = nl;
        }
        let out = tokenizer.decode(&generated).trim().to_string();
        println!("[{}] -> {}", prompt.replace('\n', " | "), out);
    }

    Ok(())
}
