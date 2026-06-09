use std::fs;
use std::path::Path;
use std::time::Instant;
use std::thread;

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
    pub w_out: Buffer<f32>,
    pub b_out: Buffer<f32>,

    m_embed: Buffer<f32>, v_embed: Buffer<f32>,
    m_w_x:   Buffer<f32>, v_w_x:   Buffer<f32>,
    m_w_h:   Buffer<f32>, v_w_h:   Buffer<f32>,
    m_b:     Buffer<f32>, v_b:     Buffer<f32>,
    m_w_out: Buffer<f32>, v_w_out: Buffer<f32>,
    m_b_out: Buffer<f32>, v_b_out: Buffer<f32>,

    pub vocab_size: usize,
    pub embed_dim:  usize,
    pub hidden_dim: usize,
    adam_step: i32,
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

        println!("================================");
        println!("        ARIA  OpenCL GPU        ");
        println!("================================");
        println!("  Vocab:   {}", vocab_size);
        println!("  Embed:   {}", embed_dim);
        println!("  Hidden:  {}", hidden_dim);
        println!("  Params:  ~{:.1}M",
            (vocab_size*embed_dim + embed_dim*4*hidden_dim + hidden_dim*4*hidden_dim
             + 4*hidden_dim + hidden_dim*vocab_size + vocab_size) as f64 / 1e6);
        println!("================================\n");

        let se = (1.0 / embed_dim  as f64).sqrt() as f32;
        let sh = (1.0 / hidden_dim as f64).sqrt() as f32;
        let fh = 4 * hidden_dim;

        let embed_data = randn_vec(vocab_size * embed_dim, se);
        let w_x_data   = randn_vec(embed_dim  * fh,       se);
        let w_h_data   = randn_vec(hidden_dim * fh,       sh);
        let mut b_data = vec![0.0f32; fh];
        for i in hidden_dim..2*hidden_dim { b_data[i] = 1.0; } // forget gate bias
        let w_out_data = randn_vec(hidden_dim * vocab_size, sh);
        let b_out_data = vec![0.0f32; vocab_size];

        let embed  = gpu_buf(&queue, &embed_data);
        let w_x    = gpu_buf(&queue, &w_x_data);
        let w_h    = gpu_buf(&queue, &w_h_data);
        let b      = gpu_buf(&queue, &b_data);
        let w_out  = gpu_buf(&queue, &w_out_data);
        let b_out  = gpu_buf(&queue, &b_out_data);

        LSTMModelCuda {
            m_embed: gpu_zeros(&queue, vocab_size * embed_dim),
            v_embed: gpu_zeros(&queue, vocab_size * embed_dim),
            m_w_x:   gpu_zeros(&queue, embed_dim  * fh),
            v_w_x:   gpu_zeros(&queue, embed_dim  * fh),
            m_w_h:   gpu_zeros(&queue, hidden_dim * fh),
            v_w_h:   gpu_zeros(&queue, hidden_dim * fh),
            m_b:     gpu_zeros(&queue, fh),
            v_b:     gpu_zeros(&queue, fh),
            m_w_out: gpu_zeros(&queue, hidden_dim * vocab_size),
            v_w_out: gpu_zeros(&queue, hidden_dim * vocab_size),
            m_b_out: gpu_zeros(&queue, vocab_size),
            v_b_out: gpu_zeros(&queue, vocab_size),
            queue, program,
            embed, w_x, w_h, b, w_out, b_out,
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
    // GEMM helpers
    // -------------------------------------------------------------------------
    fn gemm(&self, a: &Buffer<f32>, b: &Buffer<f32>, c: &mut Buffer<f32>,
            m: usize, n: usize, k: usize, alpha: f32, beta: f32) {
        let kern = Kernel::builder()
            .program(&self.program).name("gemm").queue(self.queue.clone())
            .global_work_size([m, n])
            .arg(a).arg(b).arg(c)
            .arg(m as i32).arg(n as i32).arg(k as i32)
            .arg(alpha).arg(beta)
            .build().unwrap();
        unsafe { kern.enq().unwrap(); }
    }

    fn gemm_tn(&self, a: &Buffer<f32>, b: &Buffer<f32>, c: &mut Buffer<f32>,
               m: usize, n: usize, k: usize, alpha: f32, beta: f32) {
        let kern = Kernel::builder()
            .program(&self.program).name("gemm_tn").queue(self.queue.clone())
            .global_work_size([m, n])
            .arg(a).arg(b).arg(c)
            .arg(m as i32).arg(n as i32).arg(k as i32)
            .arg(alpha).arg(beta)
            .build().unwrap();
        unsafe { kern.enq().unwrap(); }
    }

    fn gemm_nt(&self, a: &Buffer<f32>, b: &Buffer<f32>, c: &mut Buffer<f32>,
               m: usize, n: usize, k: usize, alpha: f32, beta: f32) {
        let kern = Kernel::builder()
            .program(&self.program).name("gemm_nt").queue(self.queue.clone())
            .global_work_size([m, n])
            .arg(a).arg(b).arg(c)
            .arg(m as i32).arg(n as i32).arg(k as i32)
            .arg(alpha).arg(beta)
            .build().unwrap();
        unsafe { kern.enq().unwrap(); }
    }

    // -------------------------------------------------------------------------
    // Single-token step (inference)
    // -------------------------------------------------------------------------
    fn step_internal(&self, token_id: usize, h_in: &[f32], c_in: &[f32])
        -> (Vec<f32>, Vec<f32>, Vec<f32>)
    {
        let E  = self.embed_dim;
        let H  = self.hidden_dim;
        let V  = self.vocab_size;
        let fh = 4 * H;

        let ids   = gpu_buf_i32(&self.queue, &[token_id as i32]);
        let h_gpu = gpu_buf(&self.queue, h_in);
        let c_gpu = gpu_buf(&self.queue, c_in);

        // Embedding lookup
        let mut x = gpu_zeros(&self.queue, E);
        let kern = Kernel::builder()
            .program(&self.program).name("embedding_fwd").queue(self.queue.clone())
            .global_work_size([1usize, E])
            .arg(&self.embed).arg(&ids).arg(&mut x).arg(E as i32)
            .build().unwrap();
        unsafe { kern.enq().unwrap(); }

        // gates = x @ W_x + h @ W_h
        let mut gates = gpu_zeros(&self.queue, fh);
        self.gemm(&x, &self.w_x, &mut gates, 1, fh, E, 1.0, 0.0);
        self.gemm(&h_gpu, &self.w_h, &mut gates, 1, fh, H, 1.0, 1.0);

        // + bias
        let kern = Kernel::builder()
            .program(&self.program).name("add_bias").queue(self.queue.clone())
            .global_work_size([1usize, fh])
            .arg(&mut gates).arg(&self.b).arg(fh as i32)
            .build().unwrap();
        unsafe { kern.enq().unwrap(); }

        // LSTM
        let mut h_out = gpu_zeros(&self.queue, H);
        let mut c_out = gpu_zeros(&self.queue, H);
        let kern = Kernel::builder()
            .program(&self.program).name("lstm_fwd").queue(self.queue.clone())
            .global_work_size([1usize, H])
            .arg(&gates).arg(&c_gpu).arg(&mut h_out).arg(&mut c_out).arg(H as i32)
            .build().unwrap();
        unsafe { kern.enq().unwrap(); }

        // logits = h_out @ W_out + b_out
        let mut logits = gpu_zeros(&self.queue, V);
        self.gemm(&h_out, &self.w_out, &mut logits, 1, V, H, 1.0, 0.0);
        let kern = Kernel::builder()
            .program(&self.program).name("add_bias").queue(self.queue.clone())
            .global_work_size([1usize, V])
            .arg(&mut logits).arg(&self.b_out).arg(V as i32)
            .build().unwrap();
        unsafe { kern.enq().unwrap(); }

        (buf_to_vec(&logits), buf_to_vec(&h_out), buf_to_vec(&c_out))
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
        let V  = self.vocab_size;
        let fh = 4 * H;

        // Build padded input/target arrays
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

        // Gradient accumulators
        let mut d_embed = gpu_zeros(&self.queue, self.vocab_size * E);
        let mut d_w_x   = gpu_zeros(&self.queue, E  * fh);
        let mut d_w_h   = gpu_zeros(&self.queue, H  * fh);
        let mut d_b     = gpu_zeros(&self.queue, fh);
        let mut d_w_out = gpu_zeros(&self.queue, H  * V);
        let mut d_b_out = gpu_zeros(&self.queue, V);

        // Store forward activations for BPTT
        let mut xs_list:    Vec<Buffer<f32>> = Vec::with_capacity(steps);
        let mut gates_list: Vec<Buffer<f32>> = Vec::with_capacity(steps);
        let mut h_list:     Vec<Buffer<f32>> = Vec::with_capacity(steps + 1);
        let mut c_list:     Vec<Buffer<f32>> = Vec::with_capacity(steps + 1);
        h_list.push(gpu_zeros(&self.queue, batch * H));
        c_list.push(gpu_zeros(&self.queue, batch * H));

        let mut total_loss = 0.0f32;

        // ---- FORWARD ----
        for t in 0..steps {
            let tok_col: Vec<i32> = (0..batch).map(|b| input_flat[b * max_len + t]).collect();
            let tok_gpu = gpu_buf_i32(&self.queue, &tok_col);

            let mut x = gpu_zeros(&self.queue, batch * E);
            let kern = Kernel::builder()
                .program(&self.program).name("embedding_fwd").queue(self.queue.clone())
                .global_work_size([batch, E])
                .arg(&self.embed).arg(&tok_gpu).arg(&mut x).arg(E as i32)
                .build().unwrap();
            unsafe { kern.enq().unwrap(); }

            let mut gates = gpu_zeros(&self.queue, batch * fh);
            self.gemm(&x,         &self.w_x, &mut gates, batch, fh, E, 1.0, 0.0);
            self.gemm(&h_list[t], &self.w_h, &mut gates, batch, fh, H, 1.0, 1.0);

            let kern = Kernel::builder()
                .program(&self.program).name("add_bias").queue(self.queue.clone())
                .global_work_size([batch, fh])
                .arg(&mut gates).arg(&self.b).arg(fh as i32)
                .build().unwrap();
            unsafe { kern.enq().unwrap(); }

            let mut h_new = gpu_zeros(&self.queue, batch * H);
            let mut c_new = gpu_zeros(&self.queue, batch * H);
            let kern = Kernel::builder()
                .program(&self.program).name("lstm_fwd").queue(self.queue.clone())
                .global_work_size([batch, H])
                .arg(&gates).arg(&c_list[t])
                .arg(&mut h_new).arg(&mut c_new).arg(H as i32)
                .build().unwrap();
            unsafe { kern.enq().unwrap(); }

            xs_list.push(x);
            gates_list.push(gates);
            h_list.push(h_new);
            c_list.push(c_new);
        }

        // ---- LOSS + OUTPUT GRAD ----
        let mut d_h_steps: Vec<Buffer<f32>> = (0..steps)
            .map(|_| gpu_zeros(&self.queue, batch * H)).collect();

        for t in 0..steps {
            let tgt_col: Vec<i32> = (0..batch).map(|b| {
                if mask_flat[b * max_len + t] > 0.5 { target_flat[b * max_len + t] } else { 0 }
            }).collect();
            let tgt_gpu = gpu_buf_i32(&self.queue, &tgt_col);

            let mut logits = gpu_zeros(&self.queue, batch * V);
            self.gemm(&h_list[t+1], &self.w_out, &mut logits, batch, V, H, 1.0, 0.0);
            let kern = Kernel::builder()
                .program(&self.program).name("add_bias").queue(self.queue.clone())
                .global_work_size([batch, V])
                .arg(&mut logits).arg(&self.b_out).arg(V as i32)
                .build().unwrap();
            unsafe { kern.enq().unwrap(); }

            let mut loss_buf = gpu_zeros(&self.queue, batch);
            let mut probs    = gpu_zeros(&self.queue, batch * V);
            let kern = Kernel::builder()
                .program(&self.program).name("ce_fwd").queue(self.queue.clone())
                .global_work_size([batch])
                .arg(&logits).arg(&tgt_gpu)
                .arg(&mut loss_buf).arg(&mut probs).arg(V as i32)
                .build().unwrap();
            unsafe { kern.enq().unwrap(); }

            let loss_cpu = buf_to_vec(&loss_buf);
            total_loss += loss_cpu.iter().sum::<f32>() / batch as f32;

            let mut d_logits = gpu_zeros(&self.queue, batch * V);
            let kern = Kernel::builder()
                .program(&self.program).name("ce_bwd").queue(self.queue.clone())
                .global_work_size([batch, V])
                .arg(&probs).arg(&tgt_gpu).arg(&mut d_logits)
                .arg(V as i32).arg(1.0f32 / batch as f32)
                .build().unwrap();
            unsafe { kern.enq().unwrap(); }

            // d_W_out += h^T @ d_logits
            self.gemm_tn(&h_list[t+1], &d_logits, &mut d_w_out, H, V, batch, 1.0, 1.0);

            // d_b_out += sum_batch(d_logits)
            let kern = Kernel::builder()
                .program(&self.program).name("reduce_sum_batch").queue(self.queue.clone())
                .global_work_size([V])
                .arg(&d_logits).arg(&mut d_b_out).arg(batch as i32).arg(V as i32)
                .build().unwrap();
            unsafe { kern.enq().unwrap(); }

            // d_h = d_logits @ W_out^T
            self.gemm_nt(&d_logits, &self.w_out, &mut d_h_steps[t], batch, H, V, 1.0, 1.0);
        }

        // ---- BACKWARD THROUGH LSTM ----
        let mut d_c_next = gpu_zeros(&self.queue, batch * H);

        for t in (0..steps).rev() {
            let mut d_gates  = gpu_zeros(&self.queue, batch * fh);
            let mut d_c_prev = gpu_zeros(&self.queue, batch * H);

            let kern = Kernel::builder()
                .program(&self.program).name("lstm_bwd").queue(self.queue.clone())
                .global_work_size([batch, H])
                .arg(&gates_list[t]).arg(&c_list[t]).arg(&c_list[t+1])
                .arg(&d_h_steps[t]).arg(&d_c_next)
                .arg(&mut d_gates).arg(&mut d_c_prev).arg(H as i32)
                .build().unwrap();
            unsafe { kern.enq().unwrap(); }

            d_c_next = d_c_prev;

            // d_W_h += h_prev^T @ d_gates
            self.gemm_tn(&h_list[t], &d_gates, &mut d_w_h, H, fh, batch, 1.0, 1.0);
            // d_W_x += x^T @ d_gates
            self.gemm_tn(&xs_list[t], &d_gates, &mut d_w_x, E, fh, batch, 1.0, 1.0);

            // d_b += sum_batch(d_gates)
            let kern = Kernel::builder()
                .program(&self.program).name("reduce_sum_batch").queue(self.queue.clone())
                .global_work_size([fh])
                .arg(&d_gates).arg(&mut d_b).arg(batch as i32).arg(fh as i32)
                .build().unwrap();
            unsafe { kern.enq().unwrap(); }

            // d_x = d_gates @ W_x^T
            let mut d_x = gpu_zeros(&self.queue, batch * E);
            self.gemm_nt(&d_gates, &self.w_x, &mut d_x, batch, E, fh, 1.0, 0.0);

            // embedding backward
            let tok_col: Vec<i32> = (0..batch).map(|b| input_flat[b * max_len + t]).collect();
            let tok_gpu = gpu_buf_i32(&self.queue, &tok_col);
            let kern = Kernel::builder()
                .program(&self.program).name("embedding_bwd").queue(self.queue.clone())
                .global_work_size([batch, E])
                .arg(&d_x).arg(&tok_gpu).arg(&mut d_embed).arg(E as i32)
                .build().unwrap();
            unsafe { kern.enq().unwrap(); }
        }

        // ---- ADAM UPDATE ----
        self.adam_step += 1;
        let t  = self.adam_step;
        let b1 = 0.9f32; let b2 = 0.999f32; let eps = 1e-8f32;
        let bc1 = 1.0 - b1.powi(t);
        let bc2 = 1.0 - b2.powi(t);
        let lr  = learning_rate as f32;

        let grads = [&d_embed, &d_w_x, &d_w_h, &d_b, &d_w_out, &d_b_out];

        for (i, (n, param, m_buf, v_buf)) in [
            (self.vocab_size * E, &mut self.embed, &mut self.m_embed, &mut self.v_embed),
            (E * fh,  &mut self.w_x,   &mut self.m_w_x,   &mut self.v_w_x),
            (H * fh,  &mut self.w_h,   &mut self.m_w_h,   &mut self.v_w_h),
            (fh,      &mut self.b,     &mut self.m_b,     &mut self.v_b),
            (H * V,   &mut self.w_out, &mut self.m_w_out, &mut self.v_w_out),
            (V,       &mut self.b_out, &mut self.m_b_out, &mut self.v_b_out),
        ].iter_mut().enumerate() {
            let grad = grads[i];
            // Clip gradient
            let mut norm_sq_buf = gpu_zeros(&self.queue, 1);
            let kern = Kernel::builder()
                .program(&self.program).name("grad_norm_sq").queue(self.queue.clone())
                .global_work_size([*n])
                .arg(grad).arg(&mut norm_sq_buf).arg(*n as i32)
                .build().unwrap();
            unsafe { kern.enq().unwrap(); }
            let norm = buf_to_vec(&norm_sq_buf)[0].sqrt();
            if norm > 5.0 {
                let scale = 5.0f32 / norm;
                let kern = Kernel::builder()
                    .program(&self.program).name("grad_scale").queue(self.queue.clone())
                    .global_work_size([*n])
                    .arg(grad).arg(scale)
                    .build().unwrap();
                unsafe { kern.enq().unwrap(); }
            }
            // Adam
            let kern = Kernel::builder()
                .program(&self.program).name("adam_update").queue(self.queue.clone())
                .global_work_size([*n])
                .arg(&**param).arg(&**m_buf).arg(&**v_buf).arg(grad)
                .arg(lr).arg(b1).arg(b2).arg(eps).arg(bc1).arg(bc2)
                .build().unwrap();
            unsafe { kern.enq().unwrap(); }
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
            "vocab_size":  self.vocab_size,
            "embed_dim":   self.embed_dim,
            "hidden_dim":  self.hidden_dim,
            "format":      "v10_opencl",
            "embed":  buf_to_vec(&self.embed),
            "w_x":    buf_to_vec(&self.w_x),
            "w_h":    buf_to_vec(&self.w_h),
            "b":      buf_to_vec(&self.b),
            "w_out":  buf_to_vec(&self.w_out),
            "b_out":  buf_to_vec(&self.b_out),
        });
        fs::write(path, serde_json::to_string_pretty(&data)?)?;
        Ok(())
    }

    pub fn load(path: &str, vocab_size: usize, embed_dim: usize, hidden_dim: usize) -> anyhow::Result<Self> {
        let data: serde_json::Value = serde_json::from_str(&fs::read_to_string(path)?)?;
        let mut model = LSTMModelCuda::new(vocab_size, embed_dim, hidden_dim);
        let fh = 4 * hidden_dim;

        macro_rules! load {
            ($field:ident, $key:expr, $n:expr) => {
                if let Some(arr) = data[$key].as_array() {
                    let v: Vec<f32> = arr.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect();
                    if v.len() == $n {
                        model.$field = gpu_buf(&model.queue, &v);
                    }
                }
            };
        }

        load!(embed, "embed",  vocab_size * embed_dim);
        load!(w_x,   "w_x",   embed_dim  * fh);
        load!(w_h,   "w_h",   hidden_dim * fh);
        load!(b,     "b",     fh);
        load!(w_out, "w_out", hidden_dim * vocab_size);
        load!(b_out, "b_out", vocab_size);

        Ok(model)
    }
}

// =============================================================================
// PRETRAINING
// =============================================================================

const LEARNING_RATE:       f64   = 0.001;
const MAX_TOKENS_PER_SEQ:  usize = 80;
const MIN_TOKENS_PER_SEQ:  usize = 4;
const PRETRAIN_EPOCHS:     usize = 10;
const PRETRAIN_BATCH_SIZE: usize = 64;

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

    for epoch in 0..PRETRAIN_EPOCHS {
        let ep = Instant::now();
        all_seqs.shuffle(&mut rand::thread_rng());
        let mut total_loss = 0.0f32;
        let mut batches = 0usize;

        for chunk in all_seqs.chunks(PRETRAIN_BATCH_SIZE) {
            let loss = model.train_batch(chunk, LEARNING_RATE);
            if loss.is_finite() { total_loss += loss; batches += 1; }
        }

        let avg = total_loss / batches.max(1) as f32;
        let et = ep.elapsed();
        println!("Epoch {}/{}: loss={:.6}  {:.1}s  {:.0} seq/s",
                 epoch+1, PRETRAIN_EPOCHS, avg, et.as_secs_f32(),
                 all_seqs.len() as f32 / et.as_secs_f32());
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
