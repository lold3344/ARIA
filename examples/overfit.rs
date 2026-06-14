use aria::model_cuda::LSTMModelCuda;

fn main() {
    let vocab = 100usize;
    let embed = 64usize;
    let hidden = 64usize;
    let batch = 10usize;
    let seq_len = 10usize;
    let lr = 1e-3f64;

    std::env::set_var("ARIA_SIMPLE_SOFTMAX", "1");
    std::env::set_var("ARIA_LSTM_OPT", "1");
    std::env::set_var("ARIA_ASM_OPT", "1");
    std::env::set_var("ARIA_CLIP", "5.0");

    let mut model = LSTMModelCuda::new(vocab, embed, hidden);

    // generate a single deterministic sequence and repeat it for overfit
    let seq: Vec<usize> = (0..seq_len).map(|i| i % vocab).collect();
    let sequences: Vec<Vec<usize>> = (0..batch).map(|_| seq.clone()).collect();

    for step in 0..500 {
        let loss = model.train_batch(&sequences, lr);
        if step % 50 == 0 {
            println!("step {:4} loss={:.6}", step, loss);
        }
        if loss < 0.5 {
            println!("Converged at step {} loss={:.6}", step, loss);
            break;
        }
    }
}
