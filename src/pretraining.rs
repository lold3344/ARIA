use std::fs;
use crate::model::LSTMModel;
use crate::tokenizer::Tokenizer;

pub fn pretrain_from_files(
    model: &LSTMModel,
    tokenizer: &mut Tokenizer,
    data_dir: &str
) -> anyhow::Result<()> {

    if !std::path::Path::new(data_dir).exists() {
        return Ok(());
    }

    for entry in fs::read_dir(data_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().map_or(false, |e| e == "txt") {
            let content = fs::read_to_string(&path).unwrap_or_default();

            for sentence in content.split(|c| c == '.' || c == '\n') {
                let sentence = sentence.trim();
                if sentence.len() < 5 {
                    continue;
                }

                let tokens = tokenizer.encode(sentence);
                if tokens.len() < 3 {
                    continue;
                }

                let vec_tokens: Vec<i64> = tokens.iter().map(|x| *x as i64).collect();

                for i in 1..tokens.len().min(10) {
                    let slice = &vec_tokens[0..i];

                    let input_tensor = tch::Tensor::f_from_slice(slice)?
                        .view([1, slice.len() as i64]);

                    let _ = model.forward_seq(&input_tensor);
                }
            }
        }
    }

    Ok(())
}
