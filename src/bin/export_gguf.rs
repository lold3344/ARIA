// Exports ARIA training checkpoint to inference-only Q4_0 GGUF
// Usage: cargo run --bin export_gguf --release -- [input.gguf] [output.gguf]

use aria::transformer_cuda::TransformerModel;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let input  = args.get(1).map(|s| s.as_str()).unwrap_or("aria json/aria_checkpoint.gguf");
    let output = args.get(2).map(|s| s.as_str()).unwrap_or("aria json/aria_inference.gguf");

    println!("Loading checkpoint: {}", input);
    let (model, _tokenizer) = TransformerModel::load_checkpoint(input)?;

    println!("Exporting Q4_0 inference GGUF: {}", output);
    model.save_gguf_inference(output)?;

    let in_size  = std::fs::metadata(input)?.len()  / 1024 / 1024;
    let out_size = std::fs::metadata(output)?.len() / 1024 / 1024;
    println!("Checkpoint: {}MB  →  Inference: {}MB  ({}x smaller)",
        in_size, out_size, in_size / out_size.max(1));

    Ok(())
}
