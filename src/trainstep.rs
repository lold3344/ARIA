fn train_step<B: AutodiffBackend>(
    device: &B::Device,
    model: &mut SimpleGPUModel<B>,
    optimizer: &mut burn::optim::Adam<B>,
    inputs: &Vec<Vec<i64>>,
    targets: &Vec<i64>,
) {

    let x = Tensor::<B, 2>::from_data(inputs.clone(), device);
    let y = Tensor::<B, 1>::from_data(targets.clone(), device);

    let logits = model.forward(x);
    let loss = CrossEntropyLoss::new().forward(logits, y);

    optimizer.backward_step(&loss);
}