@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> hidden: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

fn tanh_approx(x: f32) -> f32 {
    return tanh(x);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let input_len = arrayLength(&input);
    
    if i >= input_len {
        return;
    }

    let x = input[i];
    let h = hidden[i % arrayLength(&hidden)];
    let w = weights[i % arrayLength(&weights)];

    let f = sigmoid(x * w);
    let i_gate = sigmoid(x * w);
    let o = sigmoid(x * w);
    let g = tanh_approx(x * w);

    let c = f * h + i_gate * g;
    let h_new = o * tanh_approx(c);

    output[i] = h_new;
}
