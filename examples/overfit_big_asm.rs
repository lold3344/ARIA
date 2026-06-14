use aria::model_cuda::LSTMModelCuda;

fn main() {
    let vocab = 7530usize;
    let embed = 1024usize;
    let hidden = 2048usize;
    let batch = 32usize;
    let seq_len = 20usize;
    let lr = 1e-3f64;

    std::env::set_var("ARIA_LSTM_OPT", "1");
    std::env::set_var("ARIA_ASM_OPT", "1");
    std::env::set_var("ARIA_CLIP", "5.0");

    let mut model = LSTMModelCuda::new(vocab, embed, hidden);

    let seq: Vec<usize> = (0..seq_len).map(|i| i % vocab).collect();
    let sequences: Vec<Vec<usize>> = (0..batch).map(|_| seq.clone()).collect();

    for step in 0..100 {
        let loss = model.train_batch(&sequences, lr);
        if step % 10 == 0 {
            println!("step {:4} loss={:.6}", step, loss);
        }
        if loss.is_nan() || loss.is_infinite() {
            println!("NaN/Inf at step {}", step);
            break;
        }
    }
}
