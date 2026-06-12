use std::fs;
use std::path::Path;
use std::time::Instant;
use std::thread;
use std::io::Write;
use crate::adaptive_softmax::AdaptiveSoftmax;

use ocl::{Buffer, Queue, Program, Kernel, MemFlags};
use rayon::prelude::*;
use rand::seq::SliceRandom;
use rand::Rng;

use crate::tokenizer::Tokenizer;
use crate::lstm_cuda::GpuContext;

// =============================================================================
// OpenCL KERNELS
// =============================================================================

const KERNELS_SRC: &str = r#"

inline void atomic_add_float(__global float* addr, float val) {
    union { unsigned int u32; float f32; } next, expected, current;
    current.f32 = *addr;
    do {
        expected.f32 = current.f32;
        next.f32 = expected.f32 + val;
        current.u32 = atomic_cmpxchg((volatile __global unsigned int*)addr, expected.u32, next.u32);
    } while (current.u32 != expected.u32);
}

// Embedding lookup: out[b*E + d] = embed[ids[b]*E + d]
__kernel void embedding_fwd(
    __global const float* embed,
    __global const int*   ids,
    __global       float* out,
    int E)
{
    int b = get_global_id(0);
    int d = get_global_id(1);
    out[b*E + d] = embed[ids[b]*E + d];
}

// Embedding backward (scatter-add)
__kernel void embedding_bwd(
    __global const float* d_out,
    __global const int*   ids,
    __global       float* d_embed,
    int E)
{
    int b = get_global_id(0);
    int d = get_global_id(1);
    atomic_add_float(&d_embed[ids[b]*E + d], d_out[b*E + d]);
}

// Add bias: out[b*N + i] += bias[i]
__kernel void add_bias(
    __global float*       out,
    __global const float* bias,
    int N)
{
    int b = get_global_id(0);
    int i = get_global_id(1);
    out[b*N + i] += bias[i];
}

// GEMM: C = alpha*A*B + beta*C
// A: [M x K], B: [K x N], C: [M x N]  row-major
__kernel void gemm(
    __global const float* A,
    __global const float* B,
    __global       float* C,
    int M, int N, int K,
    float alpha, float beta)
{
    int row = get_global_id(0); // [0, M)
    int col = get_global_id(1); // [0, N)
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; k++)
        sum += A[row*K + k] * B[k*N + col];
    C[row*N + col] = alpha * sum + beta * C[row*N + col];
}

// GEMM_TN: C = alpha * A^T * B + beta*C
// A: [K x M] stored, A^T is [M x K]; B: [K x N]; C: [M x N]
__kernel void gemm_tn(
    __global const float* A,
    __global const float* B,
    __global       float* C,
    int M, int N, int K,
    float alpha, float beta)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; k++)
        sum += A[k*M + row] * B[k*N + col];
    C[row*N + col] = alpha * sum + beta * C[row*N + col];
}

// GEMM_NT: C = alpha * A * B^T + beta*C
// A: [M x K]; B: [N x K] stored, B^T is [K x N]; C: [M x N]
__kernel void gemm_nt(
    __global const float* A,
    __global const float* B,
    __global       float* C,
    int M, int N, int K,
    float alpha, float beta)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; k++)
        sum += A[row*K + k] * B[col*K + k];
    C[row*N + col] = alpha * sum + beta * C[row*N + col];
}

// LSTM forward: gates[b, 4*H] + c_prev[b, H] -> h[b,H], c[b,H]
__kernel void lstm_fwd(
    __global const float* gates,
    __global const float* c_prev,
    __global       float* h_out,
    __global       float* c_out,
    int H)
{
    int b = get_global_id(0);
    int j = get_global_id(1);
    float gi = gates[b*4*H + 0*H + j];
    float gf = gates[b*4*H + 1*H + j];
    float go = gates[b*4*H + 2*H + j];
    float gg = gates[b*4*H + 3*H + j];

    float i_g = 1.0f / (1.0f + exp(-gi));
    float f_g = 1.0f / (1.0f + exp(-gf));
    float o_g = 1.0f / (1.0f + exp(-go));
    float g_g = tanh(gg);

    float c = f_g * c_prev[b*H + j] + i_g * g_g;
    float h = o_g * tanh(c);

    c_out[b*H + j] = c;
    h_out[b*H + j] = h;
}

// LSTM backward
__kernel void lstm_bwd(
    __global const float* gates_raw,
    __global const float* c_prev,
    __global const float* c_cur,
    __global const float* d_h,
    __global const float* d_c_next,
    __global       float* d_gates,
    __global       float* d_c_prev,
    int H)
{
    int b = get_global_id(0);
    int j = get_global_id(1);

    float gi = gates_raw[b*4*H + 0*H + j];
    float gf = gates_raw[b*4*H + 1*H + j];
    float go = gates_raw[b*4*H + 2*H + j];
    float gg = gates_raw[b*4*H + 3*H + j];

    float i_g = 1.0f / (1.0f + exp(-gi));
    float f_g = 1.0f / (1.0f + exp(-gf));
    float o_g = 1.0f / (1.0f + exp(-go));
    float g_g = tanh(gg);

    float c  = c_cur[b*H + j];
    float tc = tanh(c);

    float dc = d_h[b*H + j] * o_g * (1.0f - tc*tc) + d_c_next[b*H + j];

    float di  = dc * g_g;
    float df  = dc * c_prev[b*H + j];
    float do_ = d_h[b*H + j] * tc;
    float dg  = dc * i_g;

    d_gates[b*4*H + 0*H + j] = di  * i_g * (1.0f - i_g);
    d_gates[b*4*H + 1*H + j] = df  * f_g * (1.0f - f_g);
    d_gates[b*4*H + 2*H + j] = do_ * o_g * (1.0f - o_g);
    d_gates[b*4*H + 3*H + j] = dg  * (1.0f - g_g*g_g);

    d_c_prev[b*H + j] = dc * f_g;
}

// Softmax cross-entropy forward (per-sample, one work-item per sample)
__kernel void ce_fwd(
    __global const float* logits,
    __global const int*   targets,
    __global       float* loss_out,
    __global       float* probs_out,
    int V)
{
    int b = get_global_id(0);
    __global const float* row = logits + b*V;

    float mx = row[0];
    for (int i = 1; i < V; i++) if (row[i] > mx) mx = row[i];

    float s = 0.0f;
    for (int i = 0; i < V; i++) s += exp(row[i] - mx);

    loss_out[b] = -(row[targets[b]] - mx - log(s));

    __global float* p = probs_out + b*V;
    for (int i = 0; i < V; i++) p[i] = exp(row[i] - mx) / s;
}

// Softmax cross-entropy backward
__kernel void ce_bwd(
    __global const float* probs,
    __global const int*   targets,
    __global       float* d_logits,
    int V, float inv_batch)
{
    int b = get_global_id(0);
    int v = get_global_id(1);
    float g = probs[b*V + v];
    if (v == targets[b]) g -= 1.0f;
    d_logits[b*V + v] = g * inv_batch;
}

// Reduce sum over batch dimension: out[i] = sum_b x[b*N+i]
__kernel void reduce_sum_batch(
    __global const float* x,
    __global       float* out,
    int batch, int N)
{
    int i = get_global_id(0);
    float s = 0.0f;
    for (int b = 0; b < batch; b++) s += x[b*N + i];
    out[i] = s;
}

// Adam update (in-place)
__kernel void adam_update(
    __global       float* param,
    __global       float* m,
    __global       float* v,
    __global const float* grad,
    float lr, float b1, float b2, float eps, float bc1, float bc2)
{
    int i = get_global_id(0);
    float g  = grad[i];
    float m_ = b1 * m[i] + (1.0f - b1) * g;
    float v_ = b2 * v[i] + (1.0f - b2) * g * g;
    m[i] = m_;
    v[i] = v_;
    param[i] -= lr * (m_ / bc1) / (sqrt(v_ / bc2) + eps);
}

// Gradient L2 norm squared (atomic add into out[0])
__kernel void grad_norm_sq(
    __global const float* grad,
    __global       float* out,
    int n)
{
    int i = get_global_id(0);
    if (i >= n) return;
    float g = grad[i];
    // Use atomic float add via integer CAS trick
    volatile __global int* out_int = (volatile __global int*)out;
    float old_val, new_val;
    int old_int, new_int;
    do {
        old_val = *(__global float*)out_int;
        new_val = old_val + g * g;
        old_int = as_int(old_val);
        new_int = as_int(new_val);
    } while (atomic_cmpxchg(out_int, old_int, new_int) != old_int);
}

// Scale gradient
__kernel void grad_scale(
    __global float* grad,
    float scale)
{
    int i = get_global_id(0);
    grad[i] *= scale;
}

// ---- AdaptiveSoftmax kernels ----

// matmul: C[batch x out] = A[batch x in] * W^T[out x in]  (W row-major)
__kernel void asm_linear(
    __global const float* A,
    __global const float* W,
    __global const float* bias,
    __global       float* C,
    int in_dim, int out_dim)
{
    int b = get_global_id(0);
    int o = get_global_id(1);
    float s = bias[o];
    int wa = o * in_dim;
    int aa = b * in_dim;
    for (int k = 0; k < in_dim; k++) s += A[aa+k] * W[wa+k];
    C[b * out_dim + o] = s;
}

// row-wise softmax in-place: x[batch x n]
__kernel void asm_softmax(
    __global float* x,
    int n)
{
    int b = get_global_id(0);
    __global float* row = x + b * n;
    float mx = row[0];
    for (int i = 1; i < n; i++) if (row[i] > mx) mx = row[i];
    float s = 0.0f;
    for (int i = 0; i < n; i++) { row[i] = exp(row[i] - mx); s += row[i]; }
    for (int i = 0; i < n; i++) row[i] /= s;
}

// compute loss and softmax gradient given targets
// probs[batch x n] in-place -> d_probs (probs - onehot)
// loss_out[batch] output
__kernel void asm_ce_grad(
    __global float*       probs,
    __global const int*   targets,
    __global       float* loss_out,
    int n, int offset)
{
    int b = get_global_id(0);
    int t = targets[b];
    if (t < 0) { loss_out[b] = 0.0f; return; }
    int ti = t - offset;
    float p = probs[b * n + ti];
    loss_out[b] = -(p > 1e-30f ? log(p) : -30.0f);
    probs[b * n + ti] -= 1.0f;
}

// weight grad: G_W[out x in] += d_out^T [out x batch] * A [batch x in]
// one work-item per (o, k)
__kernel void asm_wgrad(
    __global const float* d_out,
    __global const float* A,
    __global       float* G_W,
    int batch, int in_dim, int out_dim)
{
    int o = get_global_id(0);
    int k = get_global_id(1);
    float s = 0.0f;
    for (int b = 0; b < batch; b++) s += d_out[b * out_dim + o] * A[b * in_dim + k];
    G_W[o * in_dim + k] += s;
}

// bias grad: G_b[out] += sum_b d_out[b, out]
__kernel void asm_bgrad(
    __global const float* d_out,
    __global       float* G_b,
    int batch, int out_dim)
{
    int o = get_global_id(0);
    float s = 0.0f;
    for (int b = 0; b < batch; b++) s += d_out[b * out_dim + o];
    G_b[o] += s;
}

// input grad: d_A[batch x in] += d_out[batch x out] * W[out x in]
__kernel void asm_igrad(
    __global const float* d_out,
    __global const float* W,
    __global       float* d_A,
    int batch, int in_dim, int out_dim)
{
    int b = get_global_id(0);
    int k = get_global_id(1);
    float s = 0.0f;
    for (int o = 0; o < out_dim; o++) s += d_out[b * out_dim + o] * W[o * in_dim + k];
    d_A[b * in_dim + k] += s;
}

"#;

// =============================================================================
// LSTM STATE (host Vecs — small, copied cheaply; GPU state held in train loop)
// =============================================================================

pub struct LSTMState {
    pub h: Vec<f32>,
    pub c: Vec<f32>,
}

// =============================================================================
// MODEL
// =============================================================================

pub struct LSTMModelCuda {
    queue:   Queue,
    program: Program,

    pub embed: Buffer<f32>,
    pub w_x:   Buffer<f32>,
    pub w_h:   Buffer<f32>,
    pub b:     Buffer<f32>,

    m_embed: Buffer<f32>, v_embed: Buffer<f32>,
    m_w_x:   Buffer<f32>, v_w_x:   Buffer<f32>,
    m_w_h:   Buffer<f32>, v_w_h:   Buffer<f32>,
    m_b:     Buffer<f32>, v_b:     Buffer<f32>,

    pub adaptive_sm: AdaptiveSoftmax,

    pub vocab_size: usize,
    pub embed_dim:  usize,
    pub hidden_dim: usize,
    adam_step: i32,

    // Pre-built kernels
    k_embed_fwd:      Kernel,
    k_embed_bwd:      Kernel,
    k_add_bias:       Kernel,
    k_lstm_fwd:       Kernel,
    k_lstm_bwd:       Kernel,
    k_gemm:           Kernel,
    k_gemm_tn:        Kernel,
    k_gemm_nt:        Kernel,
    k_reduce_sum:     Kernel,
    k_adam:           Kernel,
    k_grad_norm_sq:   Kernel,
    k_grad_scale:     Kernel,
    // AdaptiveSoftmax GPU kernels
    k_asm_linear:     Kernel,
    k_asm_softmax:    Kernel,
    k_asm_ce_grad:    Kernel,
    k_asm_wgrad:      Kernel,
    k_asm_bgrad:      Kernel,
    k_asm_igrad:      Kernel,
    // AdaptiveSoftmax GPU weights
    asm_head_size:  usize,
    asm_tail1_size: usize,
    asm_tail2_size: usize,
    asm_dim1:       usize,
    asm_dim2:       usize,
    g_w_head:  Buffer<f32>, g_b_head:  Buffer<f32>,
    g_w_proj1: Buffer<f32>,
    g_w_tail1: Buffer<f32>, g_b_tail1: Buffer<f32>,
    g_w_proj2: Buffer<f32>,
    g_w_tail2: Buffer<f32>, g_b_tail2: Buffer<f32>,
    gm_w_head: Buffer<f32>, gv_w_head: Buffer<f32>,
    gm_b_head: Buffer<f32>, gv_b_head: Buffer<f32>,
    gm_w_proj1:Buffer<f32>, gv_w_proj1:Buffer<f32>,
    gm_w_tail1:Buffer<f32>, gv_w_tail1:Buffer<f32>,
    gm_b_tail1:Buffer<f32>, gv_b_tail1:Buffer<f32>,
    gm_w_proj2:Buffer<f32>, gv_w_proj2:Buffer<f32>,
    gm_w_tail2:Buffer<f32>, gv_w_tail2:Buffer<f32>,
    gm_b_tail2:Buffer<f32>, gv_b_tail2:Buffer<f32>,
    asm_adam_step: i32,
}

fn randn_vec(n: usize, scale: f32) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen::<f32>() * 2.0 * scale - scale).collect()
}

fn gpu_buf(queue: &Queue, data: &[f32]) -> Buffer<f32> {
    Buffer::builder()
        .queue(queue.clone())
        .flags(MemFlags::new().read_write().copy_host_ptr())
        .len(data.len())
        .copy_host_slice(data)
        .build().unwrap()
}

fn gpu_zeros(queue: &Queue, n: usize) -> Buffer<f32> {
    Buffer::builder()
        .queue(queue.clone())
        .flags(MemFlags::new().read_write())
        .len(n)
        .fill_val(0.0f32)
        .build().unwrap()
}

fn gpu_zeros_i32(queue: &Queue, n: usize) -> Buffer<i32> {
    Buffer::builder()
        .queue(queue.clone())
        .flags(MemFlags::new().read_write())
        .len(n)
        .fill_val(0i32)
        .build().unwrap()
}

fn buf_to_vec(buf: &Buffer<f32>) -> Vec<f32> {
    let mut v = vec![0.0f32; buf.len()];
    buf.read(&mut v).enq().unwrap();
    v
}

impl LSTMModelCuda {
    pub fn new(vocab_size: usize, embed_dim: usize, hidden_dim: usize) -> Self {
        let gpu = GpuContext::try_init().expect("No OpenCL GPU found. Install NVIDIA/AMD drivers.");
        let queue = gpu.queue;
        let context = gpu.context;

        let program = Program::builder()
            .src(KERNELS_SRC)
            .build(&context)
            .expect("OpenCL kernel compilation failed");

        let se = (1.0 / embed_dim  as f64).sqrt() as f32;
        let sh = (1.0 / hidden_dim as f64).sqrt() as f32;
        let fh = 4 * hidden_dim;

        let embed_data = randn_vec(vocab_size * embed_dim, se);
        let w_x_data   = randn_vec(embed_dim  * fh,       se);
        let w_h_data   = randn_vec(hidden_dim * fh,       sh);
        let mut b_data = vec![0.0f32; fh];
        for i in hidden_dim..2*hidden_dim { b_data[i] = 1.0; }

        let embed = gpu_buf(&queue, &embed_data);
        let w_x   = gpu_buf(&queue, &w_x_data);
        let w_h   = gpu_buf(&queue, &w_h_data);
        let b     = gpu_buf(&queue, &b_data);

        let adaptive_sm = AdaptiveSoftmax::new(hidden_dim, vocab_size);
        let asm_head_size  = adaptive_sm.head_size;
        let asm_tail1_size = adaptive_sm.tail1_size;
        let asm_tail2_size = adaptive_sm.tail2_size;
        let asm_dim1       = adaptive_sm.dim1;
        let asm_dim2       = adaptive_sm.dim2;
        let hs = asm_head_size + 2;

        println!("================================");
        println!("        ARIA  OpenCL GPU        ");
        println!("================================");
        println!("  Vocab:   {}", vocab_size);
        println!("  Embed:   {}", embed_dim);
        println!("  Hidden:  {}", hidden_dim);
        let lstm_params = vocab_size*embed_dim + embed_dim*fh + hidden_dim*fh + fh;
        let asm_params  = (adaptive_sm.head_size+2)*hidden_dim
            + adaptive_sm.dim1*hidden_dim + adaptive_sm.tail1_size*adaptive_sm.dim1
            + adaptive_sm.dim2*hidden_dim + adaptive_sm.tail2_size*adaptive_sm.dim2;
        println!("  Params:  ~{:.1}M", (lstm_params + asm_params) as f64 / 1e6);
        println!("  ASoftmax: head={} tail1={} tail2={}",
                 adaptive_sm.head_size, adaptive_sm.tail1_size, adaptive_sm.tail2_size);
        println!("================================\n");

        // Dummy buffers for kernel arg-slot registration
        let d1f = gpu_zeros(&queue, 1);
        let d1i = gpu_zeros_i32(&queue, 1);
        let zero_i = 0i32;
        let zero_f = 0.0f32;

        let k_embed_fwd = Kernel::builder().program(&program).name("embedding_fwd").queue(queue.clone())
            .arg(&embed).arg(&d1i).arg(&d1f).arg(&zero_i).build().unwrap();
        let k_embed_bwd = Kernel::builder().program(&program).name("embedding_bwd").queue(queue.clone())
            .arg(&d1f).arg(&d1i).arg(&d1f).arg(&zero_i).build().unwrap();
        let k_add_bias = Kernel::builder().program(&program).name("add_bias").queue(queue.clone())
            .arg(&d1f).arg(&d1f).arg(&zero_i).build().unwrap();
        let k_lstm_fwd = Kernel::builder().program(&program).name("lstm_fwd").queue(queue.clone())
            .arg(&d1f).arg(&d1f).arg(&d1f).arg(&d1f).arg(&zero_i).build().unwrap();
        let k_lstm_bwd = Kernel::builder().program(&program).name("lstm_bwd").queue(queue.clone())
            .arg(&d1f).arg(&d1f).arg(&d1f).arg(&d1f).arg(&d1f)
            .arg(&d1f).arg(&d1f).arg(&zero_i).build().unwrap();
        let k_gemm = Kernel::builder().program(&program).name("gemm").queue(queue.clone())
            .arg(&d1f).arg(&d1f).arg(&d1f)
            .arg(&zero_i).arg(&zero_i).arg(&zero_i).arg(&zero_f).arg(&zero_f).build().unwrap();
        let k_gemm_tn = Kernel::builder().program(&program).name("gemm_tn").queue(queue.clone())
            .arg(&d1f).arg(&d1f).arg(&d1f)
            .arg(&zero_i).arg(&zero_i).arg(&zero_i).arg(&zero_f).arg(&zero_f).build().unwrap();
        let k_gemm_nt = Kernel::builder().program(&program).name("gemm_nt").queue(queue.clone())
            .arg(&d1f).arg(&d1f).arg(&d1f)
            .arg(&zero_i).arg(&zero_i).arg(&zero_i).arg(&zero_f).arg(&zero_f).build().unwrap();
        let k_reduce_sum = Kernel::builder().program(&program).name("reduce_sum_batch").queue(queue.clone())
            .arg(&d1f).arg(&d1f).arg(&zero_i).arg(&zero_i).build().unwrap();
        let k_adam = Kernel::builder().program(&program).name("adam_update").queue(queue.clone())
            .arg(&d1f).arg(&d1f).arg(&d1f).arg(&d1f)
            .arg(&zero_f).arg(&zero_f).arg(&zero_f).arg(&zero_f).arg(&zero_f).arg(&zero_f).build().unwrap();
        let k_grad_norm_sq = Kernel::builder().program(&program).name("grad_norm_sq").queue(queue.clone())
            .arg(&d1f).arg(&d1f).arg(&zero_i).build().unwrap();
        let k_grad_scale = Kernel::builder().program(&program).name("grad_scale").queue(queue.clone())
            .arg(&d1f).arg(&zero_f).build().unwrap();

        let k_asm_linear = Kernel::builder().program(&program).name("asm_linear").queue(queue.clone())
            .arg(&d1f).arg(&d1f).arg(&d1f).arg(&d1f).arg(&zero_i).arg(&zero_i).build().unwrap();
        let k_asm_softmax = Kernel::builder().program(&program).name("asm_softmax").queue(queue.clone())
            .arg(&d1f).arg(&zero_i).build().unwrap();
        let k_asm_ce_grad = Kernel::builder().program(&program).name("asm_ce_grad").queue(queue.clone())
            .arg(&d1f).arg(&d1i).arg(&d1f).arg(&zero_i).arg(&zero_i).build().unwrap();
        let k_asm_wgrad = Kernel::builder().program(&program).name("asm_wgrad").queue(queue.clone())
            .arg(&d1f).arg(&d1f).arg(&d1f).arg(&zero_i).arg(&zero_i).arg(&zero_i).build().unwrap();
        let k_asm_bgrad = Kernel::builder().program(&program).name("asm_bgrad").queue(queue.clone())
            .arg(&d1f).arg(&d1f).arg(&zero_i).arg(&zero_i).build().unwrap();
        let k_asm_igrad = Kernel::builder().program(&program).name("asm_igrad").queue(queue.clone())
            .arg(&d1f).arg(&d1f).arg(&d1f).arg(&zero_i).arg(&zero_i).arg(&zero_i).build().unwrap();

        // Upload ASM weights to GPU
        let g_w_head  = gpu_buf(&queue, &adaptive_sm.w_head);
        let g_b_head  = gpu_buf(&queue, &adaptive_sm.b_head);
        let g_w_proj1 = gpu_buf(&queue, &adaptive_sm.w_proj1);
        let g_w_tail1 = gpu_buf(&queue, &adaptive_sm.w_tail1);
        let g_b_tail1 = gpu_buf(&queue, &adaptive_sm.b_tail1);
        let g_w_proj2 = gpu_buf(&queue, &adaptive_sm.w_proj2);
        let g_w_tail2 = gpu_buf(&queue, &adaptive_sm.w_tail2);
        let g_b_tail2 = gpu_buf(&queue, &adaptive_sm.b_tail2);

        LSTMModelCuda {
            m_embed: gpu_zeros(&queue, vocab_size * embed_dim),
            v_embed: gpu_zeros(&queue, vocab_size * embed_dim),
            m_w_x:   gpu_zeros(&queue, embed_dim  * fh),
            v_w_x:   gpu_zeros(&queue, embed_dim  * fh),
            m_w_h:   gpu_zeros(&queue, hidden_dim * fh),
            v_w_h:   gpu_zeros(&queue, hidden_dim * fh),
            m_b:     gpu_zeros(&queue, fh),
            v_b:     gpu_zeros(&queue, fh),
            k_embed_fwd, k_embed_bwd, k_add_bias,
            k_lstm_fwd, k_lstm_bwd,
            k_gemm, k_gemm_tn, k_gemm_nt,
            k_reduce_sum, k_adam, k_grad_norm_sq, k_grad_scale,
            k_asm_linear, k_asm_softmax, k_asm_ce_grad,
            k_asm_wgrad, k_asm_bgrad, k_asm_igrad,
            asm_head_size, asm_tail1_size, asm_tail2_size, asm_dim1, asm_dim2,
            g_w_head, g_b_head, g_w_proj1,
            g_w_tail1, g_b_tail1, g_w_proj2,
            g_w_tail2, g_b_tail2,
            gm_w_head:  gpu_zeros(&queue, hs * hidden_dim),
            gv_w_head:  gpu_zeros(&queue, hs * hidden_dim),
            gm_b_head:  gpu_zeros(&queue, hs),
            gv_b_head:  gpu_zeros(&queue, hs),
            gm_w_proj1: gpu_zeros(&queue, asm_dim1 * hidden_dim),
            gv_w_proj1: gpu_zeros(&queue, asm_dim1 * hidden_dim),
            gm_w_tail1: gpu_zeros(&queue, asm_tail1_size * asm_dim1),
            gv_w_tail1: gpu_zeros(&queue, asm_tail1_size * asm_dim1),
            gm_b_tail1: gpu_zeros(&queue, asm_tail1_size),
            gv_b_tail1: gpu_zeros(&queue, asm_tail1_size),
            gm_w_proj2: gpu_zeros(&queue, asm_dim2 * hidden_dim),
            gv_w_proj2: gpu_zeros(&queue, asm_dim2 * hidden_dim),
            gm_w_tail2: gpu_zeros(&queue, asm_tail2_size * asm_dim2),
            gv_w_tail2: gpu_zeros(&queue, asm_tail2_size * asm_dim2),
            gm_b_tail2: gpu_zeros(&queue, asm_tail2_size),
            gv_b_tail2: gpu_zeros(&queue, asm_tail2_size),
            asm_adam_step: 0,
            queue, program,
            embed, w_x, w_h, b,
            adaptive_sm,
            vocab_size, embed_dim, hidden_dim,
            adam_step: 0,
        }
    }

    pub fn init_state(&self) -> LSTMState {
        LSTMState {
            h: vec![0.0f32; self.hidden_dim],
            c: vec![0.0f32; self.hidden_dim],
        }
    }

    // -------------------------------------------------------------------------
    // GEMM helpers — use pre-built kernels, set args dynamically
    // -------------------------------------------------------------------------
    fn gemm(&self, a: &Buffer<f32>, b: &Buffer<f32>, c: &Buffer<f32>,
            m: usize, n: usize, k: usize, alpha: f32, beta: f32) {
        unsafe {
            self.k_gemm.set_arg(0, a).unwrap();
            self.k_gemm.set_arg(1, b).unwrap();
            self.k_gemm.set_arg(2, c).unwrap();
            self.k_gemm.set_arg(3, &(m as i32)).unwrap();
            self.k_gemm.set_arg(4, &(n as i32)).unwrap();
            self.k_gemm.set_arg(5, &(k as i32)).unwrap();
            self.k_gemm.set_arg(6, &alpha).unwrap();
            self.k_gemm.set_arg(7, &beta).unwrap();
            self.k_gemm.cmd().global_work_size([m, n]).enq().unwrap();
        }
    }

    fn gemm_tn(&self, a: &Buffer<f32>, b: &Buffer<f32>, c: &Buffer<f32>,
               m: usize, n: usize, k: usize, alpha: f32, beta: f32) {
        unsafe {
            self.k_gemm_tn.set_arg(0, a).unwrap();
            self.k_gemm_tn.set_arg(1, b).unwrap();
            self.k_gemm_tn.set_arg(2, c).unwrap();
            self.k_gemm_tn.set_arg(3, &(m as i32)).unwrap();
            self.k_gemm_tn.set_arg(4, &(n as i32)).unwrap();
            self.k_gemm_tn.set_arg(5, &(k as i32)).unwrap();
            self.k_gemm_tn.set_arg(6, &alpha).unwrap();
            self.k_gemm_tn.set_arg(7, &beta).unwrap();
            self.k_gemm_tn.cmd().global_work_size([m, n]).enq().unwrap();
        }
    }

    fn gemm_nt(&self, a: &Buffer<f32>, b: &Buffer<f32>, c: &Buffer<f32>,
               m: usize, n: usize, k: usize, alpha: f32, beta: f32) {
        unsafe {
            self.k_gemm_nt.set_arg(0, a).unwrap();
            self.k_gemm_nt.set_arg(1, b).unwrap();
            self.k_gemm_nt.set_arg(2, c).unwrap();
            self.k_gemm_nt.set_arg(3, &(m as i32)).unwrap();
            self.k_gemm_nt.set_arg(4, &(n as i32)).unwrap();
            self.k_gemm_nt.set_arg(5, &(k as i32)).unwrap();
            self.k_gemm_nt.set_arg(6, &alpha).unwrap();
            self.k_gemm_nt.set_arg(7, &beta).unwrap();
            self.k_gemm_nt.cmd().global_work_size([m, n]).enq().unwrap();
        }
    }

    // -------------------------------------------------------------------------
    // Single-token step (inference)
    // -------------------------------------------------------------------------
    fn step_internal(&self, token_id: usize, h_in: &[f32], c_in: &[f32])
        -> (Vec<f32>, Vec<f32>, Vec<f32>)
    {
        let E  = self.embed_dim;
        let H  = self.hidden_dim;
        let _V = self.vocab_size;
        let fh = 4 * H;

        let ids   = gpu_buf_i32(&self.queue, &[token_id as i32]);
        let h_gpu = gpu_buf(&self.queue, h_in);
        let c_gpu = gpu_buf(&self.queue, c_in);

        // Embedding lookup
        let mut x = gpu_zeros(&self.queue, E);
        unsafe {
            self.k_embed_fwd.set_arg(0, &self.embed).unwrap();
            self.k_embed_fwd.set_arg(1, &ids).unwrap();
            self.k_embed_fwd.set_arg(2, &x).unwrap();
            self.k_embed_fwd.set_arg(3, &(E as i32)).unwrap();
            self.k_embed_fwd.cmd().global_work_size([1usize, E]).enq().unwrap();
        }

        // gates = x @ W_x + h @ W_h + bias
        let mut gates = gpu_zeros(&self.queue, fh);
        self.gemm(&x, &self.w_x, &gates, 1, fh, E, 1.0, 0.0);
        self.gemm(&h_gpu, &self.w_h, &gates, 1, fh, H, 1.0, 1.0);
        unsafe {
            self.k_add_bias.set_arg(0, &gates).unwrap();
            self.k_add_bias.set_arg(1, &self.b).unwrap();
            self.k_add_bias.set_arg(2, &(fh as i32)).unwrap();
            self.k_add_bias.cmd().global_work_size([1usize, fh]).enq().unwrap();
        }

        // LSTM
        let mut h_out = gpu_zeros(&self.queue, H);
        let mut c_out = gpu_zeros(&self.queue, H);
        unsafe {
            self.k_lstm_fwd.set_arg(0, &gates).unwrap();
            self.k_lstm_fwd.set_arg(1, &c_gpu).unwrap();
            self.k_lstm_fwd.set_arg(2, &h_out).unwrap();
            self.k_lstm_fwd.set_arg(3, &c_out).unwrap();
            self.k_lstm_fwd.set_arg(4, &(H as i32)).unwrap();
            self.k_lstm_fwd.cmd().global_work_size([1usize, H]).enq().unwrap();
        }

        let h_cpu = buf_to_vec(&h_out);
        let logits = self.adaptive_sm.forward(&h_cpu);
        (logits, h_cpu, buf_to_vec(&c_out))
    }

    pub fn forward_seq(&self, tokens: &[usize]) -> (Vec<f32>, LSTMState) {
        if tokens.is_empty() {
            return (vec![0.0f32; self.vocab_size], self.init_state());
        }
        let mut h = vec![0.0f32; self.hidden_dim];
        let mut c = vec![0.0f32; self.hidden_dim];
        let mut logits = vec![0.0f32; self.vocab_size];
        for &tok in tokens {
            let (l, nh, nc) = self.step_internal(tok, &h, &c);
            logits = l; h = nh; c = nc;
        }
        (logits, LSTMState { h, c })
    }

    pub fn step(&self, token_id: usize, state: &LSTMState) -> (Vec<f32>, LSTMState) {
        let (logits, h, c) = self.step_internal(token_id, &state.h, &state.c);
        (logits, LSTMState { h, c })
    }

    // -------------------------------------------------------------------------
    // Training
    // -------------------------------------------------------------------------
    pub fn train_batch(&mut self, sequences: &[Vec<usize>], learning_rate: f64) -> f32 {
        let valid: Vec<&Vec<usize>> = sequences.iter().filter(|s| s.len() >= 2).collect();
        if valid.is_empty() { return 0.0; }

        let batch   = valid.len();
        let max_len = valid.iter().map(|s| s.len()).max().unwrap();
        let steps   = max_len.saturating_sub(1);
        let E  = self.embed_dim;
        let H  = self.hidden_dim;
        let fh = 4 * H;

        // Build padded input/target arrays (CPU side, used for adaptive softmax)
        let mut input_flat  = vec![0i32; batch * max_len];
        let mut target_flat = vec![0i32; batch * max_len];
        let mut mask_flat   = vec![0.0f32; batch * max_len];

        for (b, seq) in valid.iter().enumerate() {
            for (t, &tk) in seq.iter().enumerate() {
                if t + 1 < seq.len() {
                    input_flat [b * max_len + t] = tk as i32;
                    target_flat[b * max_len + t] = seq[t+1] as i32;
                    mask_flat  [b * max_len + t] = 1.0;
                }
            }
        }

        // OPT: upload the entire input token matrix once (steps * batch) instead of
        // allocating a fresh gpu_buf_i32 every step in the forward loop.
        // Layout: tok_all[t * batch + b] = input token for step t, sample b
        let mut tok_all_cpu = vec![0i32; steps * batch];
        for t in 0..steps {
            for b in 0..batch {
                tok_all_cpu[t * batch + b] = input_flat[b * max_len + t];
            }
        }
        // Gradient accumulators (zeroed once per batch)
        let mut d_embed = gpu_zeros(&self.queue, self.vocab_size * E);
        let mut d_w_x   = gpu_zeros(&self.queue, E  * fh);
        let mut d_w_h   = gpu_zeros(&self.queue, H  * fh);
        let mut d_b     = gpu_zeros(&self.queue, fh);

        // Store forward activations for BPTT
        let mut xs_list:    Vec<Buffer<f32>> = Vec::with_capacity(steps);
        let mut gates_list: Vec<Buffer<f32>> = Vec::with_capacity(steps);
        let mut h_list:     Vec<Buffer<f32>> = Vec::with_capacity(steps + 1);
        let mut c_list:     Vec<Buffer<f32>> = Vec::with_capacity(steps + 1);
        h_list.push(gpu_zeros(&self.queue, batch * H));
        c_list.push(gpu_zeros(&self.queue, batch * H));

        let mut total_loss = 0.0f32;

        // Pre-allocate reusable per-step buffers (avoid GPU alloc inside the loops)
        let mut tok_step_buf = gpu_zeros_i32(&self.queue, batch);
        let mut x_buf        = gpu_zeros(&self.queue, batch * E);

        // ---- FORWARD ----
        for t in 0..steps {
            let tok_step = &tok_all_cpu[t * batch .. (t + 1) * batch];
            tok_step_buf.write(tok_step).enq().unwrap();

            let mut x = gpu_zeros(&self.queue, batch * E);
            unsafe {
                self.k_embed_fwd.set_arg(0, &self.embed).unwrap();
                self.k_embed_fwd.set_arg(1, &tok_step_buf).unwrap();
                self.k_embed_fwd.set_arg(2, &x).unwrap();
                self.k_embed_fwd.set_arg(3, &(E as i32)).unwrap();
                self.k_embed_fwd.cmd().global_work_size([batch, E]).enq().unwrap();
            }

            let mut gates = gpu_zeros(&self.queue, batch * fh);
            self.gemm(&x, &self.w_x, &gates, batch, fh, E, 1.0, 0.0);
            self.gemm(&h_list[t], &self.w_h, &gates, batch, fh, H, 1.0, 1.0);
            unsafe {
                self.k_add_bias.set_arg(0, &gates).unwrap();
                self.k_add_bias.set_arg(1, &self.b).unwrap();
                self.k_add_bias.set_arg(2, &(fh as i32)).unwrap();
                self.k_add_bias.cmd().global_work_size([batch, fh]).enq().unwrap();
            }

            let mut h_new = gpu_zeros(&self.queue, batch * H);
            let mut c_new = gpu_zeros(&self.queue, batch * H);
            unsafe {
                self.k_lstm_fwd.set_arg(0, &gates).unwrap();
                self.k_lstm_fwd.set_arg(1, &c_list[t]).unwrap();
                self.k_lstm_fwd.set_arg(2, &h_new).unwrap();
                self.k_lstm_fwd.set_arg(3, &c_new).unwrap();
                self.k_lstm_fwd.set_arg(4, &(H as i32)).unwrap();
                self.k_lstm_fwd.cmd().global_work_size([batch, H]).enq().unwrap();
            }

            // h_list stores GPU buffers for BPTT; we'll bulk-read them after forward pass.
            h_list.push(h_new);
            c_list.push(c_new);
            xs_list.push(x);
            gates_list.push(gates);
        }

        // ---- OPT: ONE bulk GPU→CPU read for all hidden states ----
        // Read all h_list[1..=steps] in a single flat Vec instead of steps calls.
        let mut h_all_cpu = vec![0.0f32; (steps + 1) * batch * H];
        // h_list[0] is zeros (initial state), skip it
        for t in 0..=steps {
            let slice = &mut h_all_cpu[t * batch * H .. (t + 1) * batch * H];
            h_list[t].read(slice).enq().unwrap();
        }

        // ---- LOSS + OUTPUT GRAD (AdaptiveSoftmax on GPU) ----
        let hs   = self.asm_head_size + 2;
        let d1   = self.asm_dim1;
        let d2   = self.asm_dim2;
        let ts1  = self.asm_tail1_size;
        let ts2  = self.asm_tail2_size;

        // grad accumulators for ASM weights (zeroed once per batch)
        let dg_w_head  = gpu_zeros(&self.queue, hs  * H);
        let dg_b_head  = gpu_zeros(&self.queue, hs);
        let dg_w_proj1 = gpu_zeros(&self.queue, d1  * H);
        let dg_w_tail1 = gpu_zeros(&self.queue, ts1 * d1);
        let dg_b_tail1 = gpu_zeros(&self.queue, ts1);
        let dg_w_proj2 = gpu_zeros(&self.queue, d2  * H);
        let dg_w_tail2 = gpu_zeros(&self.queue, ts2 * d2);
        let dg_b_tail2 = gpu_zeros(&self.queue, ts2);

        // Pre-allocate reusable per-step buffers (avoid alloc inside loop)
        let head_logits  = gpu_zeros(&self.queue, batch * hs);
        let proj1        = gpu_zeros(&self.queue, batch * d1);
        let tail1_logits = gpu_zeros(&self.queue, batch * ts1);
        let proj2        = gpu_zeros(&self.queue, batch * d2);
        let tail2_logits = gpu_zeros(&self.queue, batch * ts2);
        let loss_buf     = gpu_zeros(&self.queue, batch);
        let d_proj1_buf  = gpu_zeros(&self.queue, batch * d1);
        let d_proj2_buf  = gpu_zeros(&self.queue, batch * d2);
        let zero_d1      = gpu_zeros(&self.queue, d1); // zero bias for proj
        let zero_d2      = gpu_zeros(&self.queue, d2);
        let tgt_gpu      = gpu_zeros_i32(&self.queue, batch);
        let head_tgt_gpu  = gpu_zeros_i32(&self.queue, batch);
        let tail1_tgt_gpu = gpu_zeros_i32(&self.queue, batch);
        let tail2_tgt_gpu = gpu_zeros_i32(&self.queue, batch);
        // d_h per step still needs separate buffers for backward pass
        let mut d_h_steps: Vec<Buffer<f32>> = (0..steps)
            .map(|_| gpu_zeros(&self.queue, batch * H))
            .collect();

        let mut tgt_cpu       = vec![0i32; batch];
        let mut head_tgt_cpu  = vec![0i32; batch];
        let mut tail1_tgt_cpu = vec![0i32; batch];
        let mut tail2_tgt_cpu = vec![0i32; batch];

        for t in 0..steps {
            let h_gpu = &h_list[t + 1];

            // build target arrays on CPU, upload once
            for b in 0..batch {
                let tk = if mask_flat[b * max_len + t] < 0.5 { -1i32 }
                         else { target_flat[b * max_len + t] };
                tgt_cpu[b] = tk;
                head_tgt_cpu[b] = if tk < 0 { -1 }
                    else if (tk as usize) < self.asm_head_size { tk }
                    else if (tk as usize) < self.asm_head_size + ts1 { self.asm_head_size as i32 }
                    else { (self.asm_head_size + 1) as i32 };
                tail1_tgt_cpu[b] = if tk < 0 || (tk as usize) < self.asm_head_size
                    || (tk as usize) >= self.asm_head_size + ts1 { -1 }
                    else { tk - self.asm_head_size as i32 };
                tail2_tgt_cpu[b] = if tk < 0 || (tk as usize) < self.asm_head_size + ts1 { -1 }
                    else { tk - (self.asm_head_size + ts1) as i32 };
            }
            tgt_gpu.write(&tgt_cpu).enq().unwrap();
            head_tgt_gpu.write(&head_tgt_cpu).enq().unwrap();
            tail1_tgt_gpu.write(&tail1_tgt_cpu).enq().unwrap();
            tail2_tgt_gpu.write(&tail2_tgt_cpu).enq().unwrap();

            unsafe {
                // head linear + softmax
                self.k_asm_linear.set_arg(0, h_gpu).unwrap();
                self.k_asm_linear.set_arg(1, &self.g_w_head).unwrap();
                self.k_asm_linear.set_arg(2, &self.g_b_head).unwrap();
                self.k_asm_linear.set_arg(3, &head_logits).unwrap();
                self.k_asm_linear.set_arg(4, &(H as i32)).unwrap();
                self.k_asm_linear.set_arg(5, &(hs as i32)).unwrap();
                self.k_asm_linear.cmd().global_work_size([batch, hs]).enq().unwrap();
                self.k_asm_softmax.set_arg(0, &head_logits).unwrap();
                self.k_asm_softmax.set_arg(1, &(hs as i32)).unwrap();
                self.k_asm_softmax.cmd().global_work_size([batch]).enq().unwrap();

                // proj1 -> tail1
                self.k_asm_linear.set_arg(0, h_gpu).unwrap();
                self.k_asm_linear.set_arg(1, &self.g_w_proj1).unwrap();
                self.k_asm_linear.set_arg(2, &zero_d1).unwrap();
                self.k_asm_linear.set_arg(3, &proj1).unwrap();
                self.k_asm_linear.set_arg(4, &(H as i32)).unwrap();
                self.k_asm_linear.set_arg(5, &(d1 as i32)).unwrap();
                self.k_asm_linear.cmd().global_work_size([batch, d1]).enq().unwrap();
                self.k_asm_linear.set_arg(0, &proj1).unwrap();
                self.k_asm_linear.set_arg(1, &self.g_w_tail1).unwrap();
                self.k_asm_linear.set_arg(2, &self.g_b_tail1).unwrap();
                self.k_asm_linear.set_arg(3, &tail1_logits).unwrap();
                self.k_asm_linear.set_arg(4, &(d1 as i32)).unwrap();
                self.k_asm_linear.set_arg(5, &(ts1 as i32)).unwrap();
                self.k_asm_linear.cmd().global_work_size([batch, ts1]).enq().unwrap();
                self.k_asm_softmax.set_arg(0, &tail1_logits).unwrap();
                self.k_asm_softmax.set_arg(1, &(ts1 as i32)).unwrap();
                self.k_asm_softmax.cmd().global_work_size([batch]).enq().unwrap();

                // proj2 -> tail2
                self.k_asm_linear.set_arg(0, h_gpu).unwrap();
                self.k_asm_linear.set_arg(1, &self.g_w_proj2).unwrap();
                self.k_asm_linear.set_arg(2, &zero_d2).unwrap();
                self.k_asm_linear.set_arg(3, &proj2).unwrap();
                self.k_asm_linear.set_arg(4, &(H as i32)).unwrap();
                self.k_asm_linear.set_arg(5, &(d2 as i32)).unwrap();
                self.k_asm_linear.cmd().global_work_size([batch, d2]).enq().unwrap();
                self.k_asm_linear.set_arg(0, &proj2).unwrap();
                self.k_asm_linear.set_arg(1, &self.g_w_tail2).unwrap();
                self.k_asm_linear.set_arg(2, &self.g_b_tail2).unwrap();
                self.k_asm_linear.set_arg(3, &tail2_logits).unwrap();
                self.k_asm_linear.set_arg(4, &(d2 as i32)).unwrap();
                self.k_asm_linear.set_arg(5, &(ts2 as i32)).unwrap();
                self.k_asm_linear.cmd().global_work_size([batch, ts2]).enq().unwrap();
                self.k_asm_softmax.set_arg(0, &tail2_logits).unwrap();
                self.k_asm_softmax.set_arg(1, &(ts2 as i32)).unwrap();
                self.k_asm_softmax.cmd().global_work_size([batch]).enq().unwrap();

                // CE grad + loss
                loss_buf.cmd().fill(0.0f32, None).enq().unwrap();
                self.k_asm_ce_grad.set_arg(0, &head_logits).unwrap();
                self.k_asm_ce_grad.set_arg(1, &head_tgt_gpu).unwrap();
                self.k_asm_ce_grad.set_arg(2, &loss_buf).unwrap();
                self.k_asm_ce_grad.set_arg(3, &(hs as i32)).unwrap();
                self.k_asm_ce_grad.set_arg(4, &0i32).unwrap();
                self.k_asm_ce_grad.cmd().global_work_size([batch]).enq().unwrap();
                self.k_asm_ce_grad.set_arg(0, &tail1_logits).unwrap();
                self.k_asm_ce_grad.set_arg(1, &tail1_tgt_gpu).unwrap();
                self.k_asm_ce_grad.cmd().global_work_size([batch]).enq().unwrap();
                self.k_asm_ce_grad.set_arg(0, &tail2_logits).unwrap();
                self.k_asm_ce_grad.set_arg(1, &tail2_tgt_gpu).unwrap();
                self.k_asm_ce_grad.set_arg(3, &(ts2 as i32)).unwrap();
                self.k_asm_ce_grad.cmd().global_work_size([batch]).enq().unwrap();
            }
            let loss_cpu_v = buf_to_vec(&loss_buf);
            total_loss += loss_cpu_v.iter().sum::<f32>() / batch as f32;

            unsafe {
                // weight grads head
                self.k_asm_wgrad.set_arg(0, &head_logits).unwrap();
                self.k_asm_wgrad.set_arg(1, h_gpu).unwrap();
                self.k_asm_wgrad.set_arg(2, &dg_w_head).unwrap();
                self.k_asm_wgrad.set_arg(3, &(batch as i32)).unwrap();
                self.k_asm_wgrad.set_arg(4, &(H as i32)).unwrap();
                self.k_asm_wgrad.set_arg(5, &(hs as i32)).unwrap();
                self.k_asm_wgrad.cmd().global_work_size([hs, H]).enq().unwrap();
                self.k_asm_bgrad.set_arg(0, &head_logits).unwrap();
                self.k_asm_bgrad.set_arg(1, &dg_b_head).unwrap();
                self.k_asm_bgrad.set_arg(2, &(batch as i32)).unwrap();
                self.k_asm_bgrad.set_arg(3, &(hs as i32)).unwrap();
                self.k_asm_bgrad.cmd().global_work_size([hs]).enq().unwrap();
                // weight grads tail1
                self.k_asm_wgrad.set_arg(0, &tail1_logits).unwrap();
                self.k_asm_wgrad.set_arg(1, &proj1).unwrap();
                self.k_asm_wgrad.set_arg(2, &dg_w_tail1).unwrap();
                self.k_asm_wgrad.set_arg(4, &(d1 as i32)).unwrap();
                self.k_asm_wgrad.set_arg(5, &(ts1 as i32)).unwrap();
                self.k_asm_wgrad.cmd().global_work_size([ts1, d1]).enq().unwrap();
                self.k_asm_bgrad.set_arg(0, &tail1_logits).unwrap();
                self.k_asm_bgrad.set_arg(1, &dg_b_tail1).unwrap();
                self.k_asm_bgrad.set_arg(3, &(ts1 as i32)).unwrap();
                self.k_asm_bgrad.cmd().global_work_size([ts1]).enq().unwrap();
                // weight grads tail2
                self.k_asm_wgrad.set_arg(0, &tail2_logits).unwrap();
                self.k_asm_wgrad.set_arg(1, &proj2).unwrap();
                self.k_asm_wgrad.set_arg(2, &dg_w_tail2).unwrap();
                self.k_asm_wgrad.set_arg(4, &(d2 as i32)).unwrap();
                self.k_asm_wgrad.set_arg(5, &(ts2 as i32)).unwrap();
                self.k_asm_wgrad.cmd().global_work_size([ts2, d2]).enq().unwrap();
                self.k_asm_bgrad.set_arg(0, &tail2_logits).unwrap();
                self.k_asm_bgrad.set_arg(1, &dg_b_tail2).unwrap();
                self.k_asm_bgrad.set_arg(3, &(ts2 as i32)).unwrap();
                self.k_asm_bgrad.cmd().global_work_size([ts2]).enq().unwrap();

                // d_proj1 = d_tail1 @ w_tail1; w_proj1 grad
                d_proj1_buf.cmd().fill(0.0f32, None).enq().unwrap();
                self.k_asm_igrad.set_arg(0, &tail1_logits).unwrap();
                self.k_asm_igrad.set_arg(1, &self.g_w_tail1).unwrap();
                self.k_asm_igrad.set_arg(2, &d_proj1_buf).unwrap();
                self.k_asm_igrad.set_arg(3, &(batch as i32)).unwrap();
                self.k_asm_igrad.set_arg(4, &(d1 as i32)).unwrap();
                self.k_asm_igrad.set_arg(5, &(ts1 as i32)).unwrap();
                self.k_asm_igrad.cmd().global_work_size([batch, d1]).enq().unwrap();
                self.k_asm_wgrad.set_arg(0, &d_proj1_buf).unwrap();
                self.k_asm_wgrad.set_arg(1, h_gpu).unwrap();
                self.k_asm_wgrad.set_arg(2, &dg_w_proj1).unwrap();
                self.k_asm_wgrad.set_arg(4, &(H as i32)).unwrap();
                self.k_asm_wgrad.set_arg(5, &(d1 as i32)).unwrap();
                self.k_asm_wgrad.cmd().global_work_size([d1, H]).enq().unwrap();

                // d_proj2 = d_tail2 @ w_tail2; w_proj2 grad
                d_proj2_buf.cmd().fill(0.0f32, None).enq().unwrap();
                self.k_asm_igrad.set_arg(0, &tail2_logits).unwrap();
                self.k_asm_igrad.set_arg(1, &self.g_w_tail2).unwrap();
                self.k_asm_igrad.set_arg(2, &d_proj2_buf).unwrap();
                self.k_asm_igrad.set_arg(4, &(d2 as i32)).unwrap();
                self.k_asm_igrad.set_arg(5, &(ts2 as i32)).unwrap();
                self.k_asm_igrad.cmd().global_work_size([batch, d2]).enq().unwrap();
                self.k_asm_wgrad.set_arg(0, &d_proj2_buf).unwrap();
                self.k_asm_wgrad.set_arg(1, h_gpu).unwrap();
                self.k_asm_wgrad.set_arg(2, &dg_w_proj2).unwrap();
                self.k_asm_wgrad.set_arg(4, &(H as i32)).unwrap();
                self.k_asm_wgrad.set_arg(5, &(d2 as i32)).unwrap();
                self.k_asm_wgrad.cmd().global_work_size([d2, H]).enq().unwrap();

                // d_h[t]
                let d_h = &d_h_steps[t];
                self.k_asm_igrad.set_arg(0, &head_logits).unwrap();
                self.k_asm_igrad.set_arg(1, &self.g_w_head).unwrap();
                self.k_asm_igrad.set_arg(2, d_h).unwrap();
                self.k_asm_igrad.set_arg(3, &(batch as i32)).unwrap();
                self.k_asm_igrad.set_arg(4, &(H as i32)).unwrap();
                self.k_asm_igrad.set_arg(5, &(hs as i32)).unwrap();
                self.k_asm_igrad.cmd().global_work_size([batch, H]).enq().unwrap();
                self.k_asm_igrad.set_arg(0, &d_proj1_buf).unwrap();
                self.k_asm_igrad.set_arg(1, &self.g_w_proj1).unwrap();
                self.k_asm_igrad.set_arg(5, &(d1 as i32)).unwrap();
                self.k_asm_igrad.cmd().global_work_size([batch, H]).enq().unwrap();
                self.k_asm_igrad.set_arg(0, &d_proj2_buf).unwrap();
                self.k_asm_igrad.set_arg(1, &self.g_w_proj2).unwrap();
                self.k_asm_igrad.set_arg(5, &(d2 as i32)).unwrap();
                self.k_asm_igrad.cmd().global_work_size([batch, H]).enq().unwrap();
            }
        }


        // ---- BACKWARD THROUGH LSTM ----
        let mut d_c_next     = gpu_zeros(&self.queue, batch * H);
        let mut d_gates_bwd  = gpu_zeros(&self.queue, batch * fh);
        let mut d_c_prev_bwd = gpu_zeros(&self.queue, batch * H);

        for t in (0..steps).rev() {
            unsafe {
                d_gates_bwd.cmd().fill(0.0f32, None).enq().unwrap();
                d_c_prev_bwd.cmd().fill(0.0f32, None).enq().unwrap();

                self.k_lstm_bwd.set_arg(0, &gates_list[t]).unwrap();
                self.k_lstm_bwd.set_arg(1, &c_list[t]).unwrap();
                self.k_lstm_bwd.set_arg(2, &c_list[t+1]).unwrap();
                self.k_lstm_bwd.set_arg(3, &d_h_steps[t]).unwrap();
                self.k_lstm_bwd.set_arg(4, &d_c_next).unwrap();
                self.k_lstm_bwd.set_arg(5, &d_gates_bwd).unwrap();
                self.k_lstm_bwd.set_arg(6, &d_c_prev_bwd).unwrap();
                self.k_lstm_bwd.set_arg(7, &(H as i32)).unwrap();
                self.k_lstm_bwd.cmd().global_work_size([batch, H]).enq().unwrap();
            }
            // ping-pong: swap d_c_next and d_c_prev_bwd (no GPU->CPU->GPU round-trip)
            std::mem::swap(&mut d_c_next, &mut d_c_prev_bwd);

            self.gemm_tn(&h_list[t],  &d_gates_bwd, &d_w_h, H, fh, batch, 1.0, 1.0);
            self.gemm_tn(&xs_list[t], &d_gates_bwd, &d_w_x, E, fh, batch, 1.0, 1.0);

            unsafe {
                self.k_reduce_sum.set_arg(0, &d_gates_bwd).unwrap();
                self.k_reduce_sum.set_arg(1, &d_b).unwrap();
                self.k_reduce_sum.set_arg(2, &(batch as i32)).unwrap();
                self.k_reduce_sum.set_arg(3, &(fh as i32)).unwrap();
                self.k_reduce_sum.cmd().global_work_size([fh]).enq().unwrap();
            }

            unsafe { x_buf.cmd().fill(0.0f32, None).enq().unwrap(); }
            self.gemm_nt(&d_gates_bwd, &self.w_x, &x_buf, batch, E, fh, 1.0, 0.0);

            let tok_step = &tok_all_cpu[t * batch .. (t + 1) * batch];
            tok_step_buf.write(tok_step).enq().unwrap();
            unsafe {
                self.k_embed_bwd.set_arg(0, &x_buf).unwrap();
                self.k_embed_bwd.set_arg(1, &tok_step_buf).unwrap();
                self.k_embed_bwd.set_arg(2, &d_embed).unwrap();
                self.k_embed_bwd.set_arg(3, &(E as i32)).unwrap();
                self.k_embed_bwd.cmd().global_work_size([batch, E]).enq().unwrap();
            }
        }

        // ---- ADAM UPDATE ----

        self.adam_step += 1;
        let t  = self.adam_step;
        let b1 = 0.9f32; let b2 = 0.999f32; let eps = 1e-8f32;
        let bc1 = 1.0 - b1.powi(t);
        let bc2 = 1.0 - b2.powi(t);
        let lr  = learning_rate as f32;

        let grads = [&d_embed, &d_w_x, &d_w_h, &d_b];

        let mut norm_sq_buf = gpu_zeros(&self.queue, 1);
        for (i, (n, param, m_buf, v_buf)) in [
            (self.vocab_size * E, &mut self.embed, &mut self.m_embed, &mut self.v_embed),
            (E * fh,  &mut self.w_x,   &mut self.m_w_x,   &mut self.v_w_x),
            (H * fh,  &mut self.w_h,   &mut self.m_w_h,   &mut self.v_w_h),
            (fh,      &mut self.b,     &mut self.m_b,     &mut self.v_b),
        ].iter_mut().enumerate() {
            let grad = grads[i];
            unsafe { norm_sq_buf.cmd().fill(0.0f32, None).enq().unwrap(); }
            unsafe {
                self.k_grad_norm_sq.set_arg(0, grad).unwrap();
                self.k_grad_norm_sq.set_arg(1, &norm_sq_buf).unwrap();
                self.k_grad_norm_sq.set_arg(2, &(*n as i32)).unwrap();
                self.k_grad_norm_sq.cmd().global_work_size([*n]).enq().unwrap();
            }
            let norm = buf_to_vec(&norm_sq_buf)[0].sqrt();
            if norm > 5.0 {
                let scale = 5.0f32 / norm;
                unsafe {
                    self.k_grad_scale.set_arg(0, grad).unwrap();
                    self.k_grad_scale.set_arg(1, &scale).unwrap();
                    self.k_grad_scale.cmd().global_work_size([*n]).enq().unwrap();
                }
            }
            unsafe {
                self.k_adam.set_arg(0, &**param).unwrap();
                self.k_adam.set_arg(1, &**m_buf).unwrap();
                self.k_adam.set_arg(2, &**v_buf).unwrap();
                self.k_adam.set_arg(3, grad).unwrap();
                self.k_adam.set_arg(4, &lr).unwrap();
                self.k_adam.set_arg(5, &b1).unwrap();
                self.k_adam.set_arg(6, &b2).unwrap();
                self.k_adam.set_arg(7, &eps).unwrap();
                self.k_adam.set_arg(8, &bc1).unwrap();
                self.k_adam.set_arg(9, &bc2).unwrap();
                self.k_adam.cmd().global_work_size([*n]).enq().unwrap();
            }
        }

        // ASM Adam
        self.asm_adam_step += 1;
        let at = self.asm_adam_step;
        let abc1 = 1.0 - 0.9f32.powi(at);
        let abc2 = 1.0 - 0.999f32.powi(at);
        let asm_grads: &[(&Buffer<f32>, &Buffer<f32>, &Buffer<f32>, &Buffer<f32>, usize)] = &[
            (&dg_w_head,  &self.g_w_head,  &self.gm_w_head,  &self.gv_w_head,  hs * H),
            (&dg_b_head,  &self.g_b_head,  &self.gm_b_head,  &self.gv_b_head,  hs),
            (&dg_w_proj1, &self.g_w_proj1, &self.gm_w_proj1, &self.gv_w_proj1, d1 * H),
            (&dg_w_tail1, &self.g_w_tail1, &self.gm_w_tail1, &self.gv_w_tail1, ts1 * d1),
            (&dg_b_tail1, &self.g_b_tail1, &self.gm_b_tail1, &self.gv_b_tail1, ts1),
            (&dg_w_proj2, &self.g_w_proj2, &self.gm_w_proj2, &self.gv_w_proj2, d2 * H),
            (&dg_w_tail2, &self.g_w_tail2, &self.gm_w_tail2, &self.gv_w_tail2, ts2 * d2),
            (&dg_b_tail2, &self.g_b_tail2, &self.gm_b_tail2, &self.gv_b_tail2, ts2),
        ];
        for (grad, param, m_buf, v_buf, n) in asm_grads {
            unsafe {
                self.k_adam.set_arg(0, *param).unwrap();
                self.k_adam.set_arg(1, *m_buf).unwrap();
                self.k_adam.set_arg(2, *v_buf).unwrap();
                self.k_adam.set_arg(3, *grad).unwrap();
                self.k_adam.set_arg(4, &lr).unwrap();
                self.k_adam.set_arg(5, &0.9f32).unwrap();
                self.k_adam.set_arg(6, &0.999f32).unwrap();
                self.k_adam.set_arg(7, &1e-8f32).unwrap();
                self.k_adam.set_arg(8, &abc1).unwrap();
                self.k_adam.set_arg(9, &abc2).unwrap();
                self.k_adam.cmd().global_work_size([*n]).enq().unwrap();
            }
        }

        total_loss / steps as f32
    }

    pub fn backward_step(&mut self, tokens: &[usize], learning_rate: f64) -> f32 {
        self.train_batch(&[tokens.to_vec()], learning_rate)
    }

    // -------------------------------------------------------------------------
    // Sampling
    // -------------------------------------------------------------------------
    pub fn sample_greedy(&self, logits: &[f32]) -> usize {
        logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i).unwrap_or(0)
    }

    pub fn sample_top_k(&self, logits: &[f32], temperature: f32, top_k: usize) -> usize {
        let temp = temperature.max(0.05);
        let scaled: Vec<f32> = logits.iter().map(|x| x / temp).collect();
        let mut indexed: Vec<(usize, f32)> = scaled.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let k = top_k.min(indexed.len()).max(1);
        let top: Vec<(usize, f32)> = indexed.into_iter().take(k).collect();
        let mx = top[0].1;
        let exps: Vec<f32> = top.iter().map(|(_, v)| (v - mx).exp()).collect();
        let sum: f32 = exps.iter().sum();
        if sum <= 0.0 || !sum.is_finite() { return top[0].0; }
        let r: f32 = rand::random();
        let mut cum = 0.0f32;
        for (i, e) in exps.iter().enumerate() {
            cum += e / sum;
            if r < cum { return top[i].0; }
        }
        top[k-1].0
    }

    pub fn sample_top_p(&self, logits: &[f32], temperature: f32, top_p: f32) -> usize {
        let temp = temperature.max(0.05);
        let scaled: Vec<f32> = logits.iter().map(|x| x / temp).collect();
        let mut indexed: Vec<(usize, f32)> = scaled.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mx = indexed[0].1;
        let exps: Vec<f32> = indexed.iter().map(|(_, v)| (v - mx).exp()).collect();
        let sum: f32 = exps.iter().sum();
        if sum <= 0.0 || !sum.is_finite() { return indexed[0].0; }
        let p = top_p.clamp(0.01, 1.0);
        let mut cum = 0.0f32;
        let mut candidates: Vec<(usize, f32)> = Vec::new();
        for (i, (idx, _)) in indexed.iter().enumerate() {
            let prob = exps[i] / sum;
            candidates.push((*idx, prob));
            cum += prob;
            if cum >= p { break; }
        }
        let r: f32 = rand::random();
        let cs: f32 = candidates.iter().map(|(_, p)| p).sum();
        let mut s = 0.0f32;
        for (idx, prob) in &candidates {
            s += prob / cs;
            if r < s { return *idx; }
        }
        candidates.last().map(|(i, _)| *i).unwrap_or(0)
    }

    // -------------------------------------------------------------------------
    // Save / Load
    // -------------------------------------------------------------------------
    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let data = serde_json::json!({
            "vocab_size": self.vocab_size,
            "embed_dim":  self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "format":     "v11_adaptive",
            "embed": buf_to_vec(&self.embed),
            "w_x":   buf_to_vec(&self.w_x),
            "w_h":   buf_to_vec(&self.w_h),
            "b":     buf_to_vec(&self.b),
            "adaptive_sm": self.adaptive_sm.to_json(),
        });
        fs::write(path, serde_json::to_string(&data)?)?;
        Ok(())
    }

    pub fn load(path: &str, vocab_size: usize, embed_dim: usize, hidden_dim: usize) -> anyhow::Result<Self> {
        let data: serde_json::Value = serde_json::from_str(&fs::read_to_string(path)?)?;
        let mut model = LSTMModelCuda::new(vocab_size, embed_dim, hidden_dim);
        let fh = 4 * hidden_dim;

        macro_rules! load_gpu {
            ($field:ident, $key:expr, $n:expr) => {
                if let Some(arr) = data[$key].as_array() {
                    let v: Vec<f32> = arr.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect();
                    if v.len() == $n { model.$field = gpu_buf(&model.queue, &v); }
                }
            };
        }

        load_gpu!(embed, "embed", vocab_size * embed_dim);
        load_gpu!(w_x,   "w_x",  embed_dim  * fh);
        load_gpu!(w_h,   "w_h",  hidden_dim * fh);
        load_gpu!(b,     "b",    fh);

        if let Some(asm) = AdaptiveSoftmax::from_json(&data["adaptive_sm"]) {
            model.adaptive_sm = asm;
        }

        Ok(model)
    }
}

// =============================================================================
// PRETRAINING
// =============================================================================

const LEARNING_RATE:       f64   = 0.0003;
const MAX_TOKENS_PER_SEQ:  usize = 80;
const MIN_TOKENS_PER_SEQ:  usize = 4;
const PRETRAIN_EPOCHS:     usize = 5;
const PRETRAIN_BATCH_SIZE: usize = 32;

pub fn pretrain_from_files(
    model: &mut LSTMModelCuda,
    tokenizer: &mut Tokenizer,
    data_dir: &str,
) -> anyhow::Result<()> {
    let path = Path::new(data_dir);
    if !path.exists() { println!("Data dir not found."); return Ok(()); }

    let start = Instant::now();
    let num_cpus = thread::available_parallelism().map(|n| n.get()).unwrap_or(4);

    println!("\n===============================================");
    println!("       ARIA - OpenCL GPU TRAINING             ");
    println!("===============================================");
    println!("LR: {}  Epochs: {}  Batch: {}",
             LEARNING_RATE, PRETRAIN_EPOCHS, PRETRAIN_BATCH_SIZE);
    println!("===============================================\n");

    rayon::ThreadPoolBuilder::new().num_threads(num_cpus).build_global().ok();

    let files: Vec<String> = fs::read_dir(path)?
        .par_bridge()
        .filter_map(|e| {
            let e = e.ok()?;
            let p = e.path();
            if p.extension().map_or(false, |x| x == "txt") {
                let c = fs::read_to_string(&p).ok()?;
                if !c.trim().is_empty() { return Some(c); }
            }
            None
        })
        .collect();

    println!("Loaded {} files", files.len());
    if files.is_empty() { println!("No text files."); return Ok(()); }

    let mut all_seqs: Vec<Vec<usize>> = files.iter()
        .flat_map(|c| {
            c.split(|ch| ch == '.' || ch == '\n' || ch == '!' || ch == '?')
             .map(|s| s.trim().to_string())
             .filter(|s| s.len() > 5)
             .collect::<Vec<_>>()
        })
        .filter_map(|s| {
            let t = tokenizer.encode(&s);
            if t.len() >= MIN_TOKENS_PER_SEQ && t.len() <= MAX_TOKENS_PER_SEQ { Some(t) } else { None }
        })
        .collect();

    println!("{} sequences\n", all_seqs.len());
    if all_seqs.is_empty() { println!("No usable sequences."); return Ok(()); }

    let total_batches = (all_seqs.len() + PRETRAIN_BATCH_SIZE - 1) / PRETRAIN_BATCH_SIZE;

    let mut current_lr = LEARNING_RATE;
    for epoch in 0..PRETRAIN_EPOCHS {
        let ep = Instant::now();
        let mut last_report = Instant::now();
        all_seqs.shuffle(&mut rand::thread_rng());
        let mut total_loss = 0.0f32;
        let mut batches = 0usize;
        let mut seqs_done = 0usize;

        for chunk in all_seqs.chunks(PRETRAIN_BATCH_SIZE) {
            let loss = model.train_batch(chunk, current_lr);
            if loss.is_finite() { total_loss += loss; batches += 1; }
            seqs_done += chunk.len();

            if last_report.elapsed().as_secs_f32() >= 10.0 {
                let avg      = total_loss / batches.max(1) as f32;
                let elapsed  = ep.elapsed().as_secs_f32();
                let seq_s    = seqs_done as f32 / elapsed;
                let remaining = total_batches.saturating_sub(batches);
                println!("  Epoch {}/{}  |  batch {}/{}  ({} remaining)  |  loss={:.4}  |  {:.0} seq/s",
                         epoch+1, PRETRAIN_EPOCHS,
                         batches, total_batches, remaining,
                         avg, seq_s);
                std::io::stdout().flush().ok();
                last_report = Instant::now();
            }
        }

        let avg = total_loss / batches.max(1) as f32;
        let et = ep.elapsed();
        let seq_s = all_seqs.len() as f32 / et.as_secs_f32();
        println!("Epoch {}/{} done  |  loss={:.6}  |  {:.1}s  |  {:.0} seq/s  |  lr={:.6}",
                 epoch+1, PRETRAIN_EPOCHS, avg, et.as_secs_f32(), seq_s, current_lr);
        current_lr *= 0.85;
    }

    println!("\nTotal: {:.1}s", start.elapsed().as_secs_f32());
    Ok(())
}

// -------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------

fn gpu_buf_i32(queue: &Queue, data: &[i32]) -> Buffer<i32> {
    Buffer::builder()
        .queue(queue.clone())
        .flags(MemFlags::new().read_write().copy_host_ptr())
        .len(data.len())
        .copy_host_slice(data)
        .build().unwrap()
}