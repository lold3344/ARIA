use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use std::thread;
use std::io::Write;

use cudarc::driver::{CudaStream, CudaSlice, CudaModule, LaunchConfig};
use cudarc::driver::PushKernelArg;
use cudarc::nvrtc::Ptx;
use cudarc::cublas::{CudaBlas, GemmConfig, Gemm, sys::cublasOperation_t};

use rand::seq::SliceRandom;
use rand::Rng;

use crate::adaptive_softmax::AdaptiveSoftmax;
use crate::tokenizer::Tokenizer;
use crate::lstm_cuda::GpuContext;

// ─────────────────────────────────────────────────────────────
//  PTX kernel names (compiled from src/kernels.cu via build.rs)
// ─────────────────────────────────────────────────────────────
const KERNEL_NAMES: &[&str] = &[
    "embedding_fwd", "embedding_bwd", "add_bias",
    "fused_lstm_fwd", "fused_lstm_bwd",
    "asm_linear", "asm_softmax", "asm_ce_grad",
    "asm_wgrad", "asm_bgrad", "asm_igrad",
    "reduce_sum_batch", "adam_update",
    "norm_reduce", "clip_if_needed", "zero_float",
];

// ─────────────────────────────────────────────────────────────
//  LSTM state (CPU side, small)
// ─────────────────────────────────────────────────────────────
pub struct LSTMState {
    pub h: Vec<f32>,
    pub c: Vec<f32>,
}

// ─────────────────────────────────────────────────────────────
//  Model
// ─────────────────────────────────────────────────────────────
pub struct LSTMModelCuda {
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    blas:   CudaBlas,

    // LSTM weights (VRAM)
    pub embed: CudaSlice<f32>,
    pub w_x:   CudaSlice<f32>,   // [E, 4H]  row-major
    pub w_h:   CudaSlice<f32>,   // [H, 4H]
    pub b:     CudaSlice<f32>,   // [4H]

    // Adam moments
    m_embed: CudaSlice<f32>, v_embed: CudaSlice<f32>,
    m_w_x:   CudaSlice<f32>, v_w_x:   CudaSlice<f32>,
    m_w_h:   CudaSlice<f32>, v_w_h:   CudaSlice<f32>,
    m_b:     CudaSlice<f32>, v_b:     CudaSlice<f32>,

    // AdaptiveSoftmax (CPU weights, GPU buffers managed per-batch)
    pub adaptive_sm: AdaptiveSoftmax,

    // ASM GPU weights + Adam moments
    g_w_head:  CudaSlice<f32>, g_b_head:  CudaSlice<f32>,
    g_w_proj1: CudaSlice<f32>,
    g_w_tail1: CudaSlice<f32>, g_b_tail1: CudaSlice<f32>,
    g_w_proj2: CudaSlice<f32>,
    g_w_tail2: CudaSlice<f32>, g_b_tail2: CudaSlice<f32>,

    gm_w_head: CudaSlice<f32>, gv_w_head: CudaSlice<f32>,
    gm_b_head: CudaSlice<f32>, gv_b_head: CudaSlice<f32>,
    gm_w_proj1:CudaSlice<f32>, gv_w_proj1:CudaSlice<f32>,
    gm_w_tail1:CudaSlice<f32>, gv_w_tail1:CudaSlice<f32>,
    gm_b_tail1:CudaSlice<f32>, gv_b_tail1:CudaSlice<f32>,
    gm_w_proj2:CudaSlice<f32>, gv_w_proj2:CudaSlice<f32>,
    gm_w_tail2:CudaSlice<f32>, gv_w_tail2:CudaSlice<f32>,
    gm_b_tail2:CudaSlice<f32>, gv_b_tail2:CudaSlice<f32>,

    pub vocab_size: usize,
    pub embed_dim:  usize,
    pub hidden_dim: usize,
    adam_step:     i32,
    asm_adam_step: i32,

    asm_head_size:  usize,
    asm_tail1_size: usize,
    asm_tail2_size: usize,
    asm_dim1:       usize,
    asm_dim2:       usize,
}

// ─────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────
fn randn_vec(n: usize, scale: f32) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen::<f32>() * 2.0 * scale - scale).collect()
}

fn upload(stream: &Arc<CudaStream>, data: &[f32]) -> CudaSlice<f32> {
    stream.clone_htod(data).unwrap()
}

fn zeros_gpu(stream: &Arc<CudaStream>, n: usize) -> CudaSlice<f32> {
    stream.alloc_zeros::<f32>(n).unwrap()
}

fn download(stream: &Arc<CudaStream>, buf: &CudaSlice<f32>) -> Vec<f32> {
    stream.clone_dtoh(buf).unwrap()
}

// cuBLAS SGEMM wrapper  C = alpha*op(A)*op(B) + beta*C
// cuBLAS is column-major so we use the transposed form:
//   C^T = alpha * B^T * A^T + beta * C^T
// Dimensions (row-major view): A[M,K]  B[K,N]  C[M,N]
fn sgemm(
    blas: &CudaBlas,
    transa: bool, transb: bool,
    m: usize, n: usize, k: usize,
    alpha: f32, beta: f32,
    a: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    c: &mut CudaSlice<f32>,
) {
    use cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T};
    if !transa && !transb {
        // C[M,N] = A[M,K]*B[K,N]  →  cuBLAS(N,N): C^T[N,M] = B^T[N,K]*A^T[K,M]
        // pass (B,A): A_cublas=B(col[N,K],lda=N), B_cublas=A(col[K,M],ldb=K)
        unsafe {
            blas.gemm(GemmConfig::<f32> {
                transa: CUBLAS_OP_N, transb: CUBLAS_OP_N,
                m: n as i32, n: m as i32, k: k as i32,
                alpha, lda: n as i32, ldb: k as i32, beta, ldc: n as i32,
            }, b, a, c).unwrap();
        }
    } else if transa && !transb {
        // C[M,N] = A^T[M,K]*B[K,N], A stored [K,M]
        // C^T[N,M] = B^T[N,K]*A[K,M] → cuBLAS(N,T): A_cublas=B(col[N,K],lda=N), B_cublas=A(col[M,K],transb=T,ldb=M)
        unsafe {
            blas.gemm(GemmConfig::<f32> {
                transa: CUBLAS_OP_N, transb: CUBLAS_OP_T,
                m: n as i32, n: m as i32, k: k as i32,
                alpha, lda: n as i32, ldb: m as i32, beta, ldc: n as i32,
            }, b, a, c).unwrap();
        }
    } else {
        // C[M,N] = A[M,K]*B^T[K,N], B stored [N,K]
        // C^T[N,M] = B[N,K]*A^T[K,M] → cuBLAS(T,N): A_cublas=B(col[K,N],transa=T,lda=K), B_cublas=A(col[K,M],ldb=K)
        unsafe {
            blas.gemm(GemmConfig::<f32> {
                transa: CUBLAS_OP_T, transb: CUBLAS_OP_N,
                m: n as i32, n: m as i32, k: k as i32,
                alpha, lda: k as i32, ldb: k as i32, beta, ldc: n as i32,
            }, b, a, c).unwrap();
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  Launch helpers
// ─────────────────────────────────────────────────────────────
fn cfg1d(n: usize) -> LaunchConfig {
    let threads = 256usize;
    let blocks  = (n + threads - 1) / threads;
    LaunchConfig { grid_dim: (blocks as u32, 1, 1), block_dim: (threads as u32, 1, 1), shared_mem_bytes: 0 }
}

fn cfg2d(bx: usize, by: usize) -> LaunchConfig {
    let ty = by.min(1024);
    let gy = (by + ty - 1) / ty;
    LaunchConfig { grid_dim: (bx as u32, gy as u32, 1), block_dim: (ty as u32, 1, 1), shared_mem_bytes: 0 }
}

// ─────────────────────────────────────────────────────────────
//  impl LSTMModelCuda
// ─────────────────────────────────────────────────────────────
impl LSTMModelCuda {
    pub fn new(vocab_size: usize, embed_dim: usize, hidden_dim: usize) -> Self {
        let gpu    = GpuContext::try_init().expect("No CUDA GPU found");
        let ctx    = gpu.ctx;
        let stream = gpu.stream;
        let blas   = gpu.blas;

        // Load PTX compiled by build.rs (nvcc -> kernels.ptx in OUT_DIR)
        let out_dir  = env!("OUT_DIR");
        let ptx_path = std::path::PathBuf::from(out_dir).join("kernels.ptx");
        let ptx    = Ptx::from_file(&ptx_path);
        let module = ctx.load_module(ptx)
            .unwrap_or_else(|_| panic!("Failed to load kernels.ptx from {:?}", ptx_path));

        let se = (1.0 / embed_dim  as f64).sqrt() as f32;
        let sh = (1.0 / hidden_dim as f64).sqrt() as f32;
        let fh = 4 * hidden_dim;

        let embed_data = randn_vec(vocab_size * embed_dim, se);
        let w_x_data   = randn_vec(embed_dim  * fh,       se);
        let w_h_data   = randn_vec(hidden_dim * fh,       sh);
        let mut b_data = vec![0.0f32; fh];
        for i in hidden_dim..2*hidden_dim { b_data[i] = 1.0; }

        let adaptive_sm    = AdaptiveSoftmax::new(hidden_dim, vocab_size);
        let asm_head_size  = adaptive_sm.head_size;
        let asm_tail1_size = adaptive_sm.tail1_size;
        let asm_tail2_size = adaptive_sm.tail2_size;
        let asm_dim1       = adaptive_sm.dim1;
        let asm_dim2       = adaptive_sm.dim2;
        let hs = asm_head_size + 2;

        let lstm_params = vocab_size*embed_dim + embed_dim*fh + hidden_dim*fh + fh;
        let asm_params  = hs*hidden_dim + asm_dim1*hidden_dim + asm_tail1_size*asm_dim1
            + asm_dim2*hidden_dim + asm_tail2_size*asm_dim2;

        println!("================================");
        println!("        ARIA  CUDA / cuBLAS     ");
        println!("================================");
        println!("  Vocab:   {}", vocab_size);
        println!("  Embed:   {}", embed_dim);
        println!("  Hidden:  {}", hidden_dim);
        println!("  Params:  ~{:.1}M", (lstm_params + asm_params) as f64 / 1e6);
        println!("  ASoftmax: head={} tail1={} tail2={}",
                 asm_head_size, asm_tail1_size, asm_tail2_size);
        println!("================================\n");

        LSTMModelCuda {
            embed: upload(&stream, &embed_data),
            w_x:   upload(&stream, &w_x_data),
            w_h:   upload(&stream, &w_h_data),
            b:     upload(&stream, &b_data),
            m_embed: zeros_gpu(&stream, vocab_size * embed_dim),
            v_embed: zeros_gpu(&stream, vocab_size * embed_dim),
            m_w_x: zeros_gpu(&stream, embed_dim * fh),
            v_w_x: zeros_gpu(&stream, embed_dim * fh),
            m_w_h: zeros_gpu(&stream, hidden_dim * fh),
            v_w_h: zeros_gpu(&stream, hidden_dim * fh),
            m_b:   zeros_gpu(&stream, fh),
            v_b:   zeros_gpu(&stream, fh),

            g_w_head:  upload(&stream, &adaptive_sm.w_head),
            g_b_head:  upload(&stream, &adaptive_sm.b_head),
            g_w_proj1: upload(&stream, &adaptive_sm.w_proj1),
            g_w_tail1: upload(&stream, &adaptive_sm.w_tail1),
            g_b_tail1: upload(&stream, &adaptive_sm.b_tail1),
            g_w_proj2: upload(&stream, &adaptive_sm.w_proj2),
            g_w_tail2: upload(&stream, &adaptive_sm.w_tail2),
            g_b_tail2: upload(&stream, &adaptive_sm.b_tail2),

            gm_w_head:  zeros_gpu(&stream, hs * hidden_dim),
            gv_w_head:  zeros_gpu(&stream, hs * hidden_dim),
            gm_b_head:  zeros_gpu(&stream, hs),
            gv_b_head:  zeros_gpu(&stream, hs),
            gm_w_proj1: zeros_gpu(&stream, asm_dim1 * hidden_dim),
            gv_w_proj1: zeros_gpu(&stream, asm_dim1 * hidden_dim),
            gm_w_tail1: zeros_gpu(&stream, asm_tail1_size * asm_dim1),
            gv_w_tail1: zeros_gpu(&stream, asm_tail1_size * asm_dim1),
            gm_b_tail1: zeros_gpu(&stream, asm_tail1_size),
            gv_b_tail1: zeros_gpu(&stream, asm_tail1_size),
            gm_w_proj2: zeros_gpu(&stream, asm_dim2 * hidden_dim),
            gv_w_proj2: zeros_gpu(&stream, asm_dim2 * hidden_dim),
            gm_w_tail2: zeros_gpu(&stream, asm_tail2_size * asm_dim2),
            gv_w_tail2: zeros_gpu(&stream, asm_tail2_size * asm_dim2),
            gm_b_tail2: zeros_gpu(&stream, asm_tail2_size),
            gv_b_tail2: zeros_gpu(&stream, asm_tail2_size),

            stream, module, blas,
            adaptive_sm,
            vocab_size, embed_dim, hidden_dim,
            adam_step: 0, asm_adam_step: 0,
            asm_head_size, asm_tail1_size, asm_tail2_size, asm_dim1, asm_dim2,
        }
    }

    pub fn init_state(&self) -> LSTMState {
        LSTMState {
            h: vec![0.0f32; self.hidden_dim],
            c: vec![0.0f32; self.hidden_dim],
        }
    }

    // ─── single-step inference ───────────────────────────────
    fn step_internal(&self, token_id: usize, h_in: &[f32], c_in: &[f32])
        -> (Vec<f32>, Vec<f32>, Vec<f32>)
    {
        let E  = self.embed_dim;
        let H  = self.hidden_dim;
        let fh = 4 * H;
        let stream = Arc::clone(&self.stream);

        let ids   = stream.clone_htod(&[token_id as i32]).unwrap();
        let h_gpu = stream.clone_htod(h_in).unwrap();
        let c_gpu = stream.clone_htod(c_in).unwrap();

        let mut x     = stream.alloc_zeros::<f32>(E).unwrap();
        let mut gates = stream.alloc_zeros::<f32>(fh).unwrap();
        let mut h_out = stream.alloc_zeros::<f32>(H).unwrap();
        let mut c_out = stream.alloc_zeros::<f32>(H).unwrap();

        let f_emb_fwd = self.module.load_function("embedding_fwd").unwrap();
        let e_i = E as i32;
        unsafe {
            stream.launch_builder(&f_emb_fwd)
                .arg(&self.embed)
                .arg(&ids)
                .arg(&mut x)
                .arg(&e_i)
                .launch(cfg2d(1, E)).unwrap();
        }

        sgemm(&self.blas, false, false, 1, fh, E, 1.0, 0.0, &x, &self.w_x, &mut gates);
        sgemm(&self.blas, false, false, 1, fh, H, 1.0, 1.0, &h_gpu, &self.w_h, &mut gates);

        let f_bias = self.module.load_function("add_bias").unwrap();
        let one_i = 1i32;
        let fh_i  = fh as i32;
        unsafe {
            stream.launch_builder(&f_bias)
                .arg(&mut gates)
                .arg(&self.b)
                .arg(&one_i)
                .arg(&fh_i)
                .launch(cfg1d(fh)).unwrap();
        }

        let f_lstm = self.module.load_function("fused_lstm_fwd").unwrap();
        let h_i = H as i32;
        unsafe {
            stream.launch_builder(&f_lstm)
                .arg(&gates)
                .arg(&c_gpu)
                .arg(&mut h_out)
                .arg(&mut c_out)
                .arg(&h_i)
                .launch(cfg2d(1, H)).unwrap();
        }

        let h_cpu  = download(&self.stream, &h_out);
        let logits = self.adaptive_sm.forward(&h_cpu);
        (logits, h_cpu, download(&self.stream, &c_out))
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

    // ─────────────────────────────────────────────────────────
    //  train_batch — cuBLAS GEMM + fused LSTM + GPU Adam
    // ─────────────────────────────────────────────────────────
    pub fn train_batch(&mut self, sequences: &[Vec<usize>], learning_rate: f64) -> f32 {
        let valid: Vec<&Vec<usize>> = sequences.iter().filter(|s| s.len() >= 2).collect();
        if valid.is_empty() { return 0.0; }

        let batch   = valid.len();
        let max_len = valid.iter().map(|s| s.len()).max().unwrap();
        let steps   = max_len.saturating_sub(1);
        let E  = self.embed_dim;
        let H  = self.hidden_dim;
        let fh = 4 * H;

        // Build padded CPU arrays
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

        // Transpose tokens: tok_all[t * batch + b]
        let mut tok_all_cpu = vec![0i32; steps * batch];
        for t in 0..steps {
            for b in 0..batch {
                tok_all_cpu[t * batch + b] = input_flat[b * max_len + t];
            }
        }

        let stream = Arc::clone(&self.stream);

        // ── Gradient accumulators ──
        let mut d_embed = stream.alloc_zeros::<f32>(self.vocab_size * E).unwrap();
        let mut d_w_x   = stream.alloc_zeros::<f32>(E  * fh).unwrap();
        let mut d_w_h   = stream.alloc_zeros::<f32>(H  * fh).unwrap();
        let mut d_b     = stream.alloc_zeros::<f32>(fh).unwrap();

        // ── Pre-allocate ALL step buffers ──
        let mut xs_list:    Vec<CudaSlice<f32>> = (0..steps).map(|_| stream.alloc_zeros::<f32>(batch * E).unwrap()).collect();
        let mut gates_list: Vec<CudaSlice<f32>> = (0..steps).map(|_| stream.alloc_zeros::<f32>(batch * fh).unwrap()).collect();
        let mut h_list:     Vec<CudaSlice<f32>> = (0..=steps).map(|_| stream.alloc_zeros::<f32>(batch * H).unwrap()).collect();
        let mut c_list:     Vec<CudaSlice<f32>> = (0..=steps).map(|_| stream.alloc_zeros::<f32>(batch * H).unwrap()).collect();

        // GPU total loss (single float, no CPU sync per step)
        let mut gpu_total_loss: CudaSlice<f32> = stream.alloc_zeros::<f32>(1).unwrap();

        let f_emb_fwd  = self.module.load_function("embedding_fwd").unwrap();
        let f_bias     = self.module.load_function("add_bias").unwrap();
        let f_lstm_fwd = self.module.load_function("fused_lstm_fwd").unwrap();

        // ── FORWARD ──────────────────────────────────────────
        for t in 0..steps {
            let tok_step = &tok_all_cpu[t * batch .. (t + 1) * batch];
            let tok_buf  = stream.clone_htod(tok_step).unwrap();

            let batch_i = batch as i32;
            let e_i     = E as i32;
            let fh_i    = fh as i32;
            let h_i     = H as i32;
            unsafe {
                stream.launch_builder(&f_emb_fwd)
                    .arg(&self.embed)
                    .arg(&tok_buf)
                    .arg(&mut xs_list[t])
                    .arg(&e_i)
                    .launch(cfg2d(batch, E)).unwrap();
            }

            sgemm(&self.blas, false, false, batch, fh, E, 1.0, 0.0,
                  &xs_list[t], &self.w_x, &mut gates_list[t]);
            sgemm(&self.blas, false, false, batch, fh, H, 1.0, 1.0,
                  &h_list[t], &self.w_h, &mut gates_list[t]);

            unsafe {
                stream.launch_builder(&f_bias)
                    .arg(&mut gates_list[t])
                    .arg(&self.b)
                    .arg(&batch_i)
                    .arg(&fh_i)
                    .launch(cfg1d(batch * fh)).unwrap();
                let (c_prev, c_next) = c_list.split_at_mut(t + 1);
                stream.launch_builder(&f_lstm_fwd)
                    .arg(&gates_list[t])
                    .arg(&c_prev[t])
                    .arg(&mut h_list[t+1])
                    .arg(&mut c_next[0])
                    .arg(&h_i)
                    .launch(cfg2d(batch, H)).unwrap();
            }
        }

        // ── ASM forward + backward (loss + d_h per step) ─────
        let hs  = self.asm_head_size + 2;
        let d1  = self.asm_dim1;
        let d2  = self.asm_dim2;
        let ts1 = self.asm_tail1_size;
        let ts2 = self.asm_tail2_size;

        let zero_d1 = stream.alloc_zeros::<f32>(d1).unwrap();
        let zero_d2 = stream.alloc_zeros::<f32>(d2).unwrap();

        let mut dg_w_head  = stream.alloc_zeros::<f32>(hs  * H).unwrap();
        let mut dg_b_head  = stream.alloc_zeros::<f32>(hs).unwrap();
        let mut dg_w_proj1 = stream.alloc_zeros::<f32>(d1  * H).unwrap();
        let mut dg_w_tail1 = stream.alloc_zeros::<f32>(ts1 * d1).unwrap();
        let mut dg_b_tail1 = stream.alloc_zeros::<f32>(ts1).unwrap();
        let mut dg_w_proj2 = stream.alloc_zeros::<f32>(d2  * H).unwrap();
        let mut dg_w_tail2 = stream.alloc_zeros::<f32>(ts2 * d2).unwrap();
        let mut dg_b_tail2 = stream.alloc_zeros::<f32>(ts2).unwrap();

        let mut d_h_steps: Vec<CudaSlice<f32>> = (0..steps).map(|_| stream.alloc_zeros::<f32>(batch * H).unwrap()).collect();

        let f_asm_lin = self.module.load_function("asm_linear").unwrap();
        let f_asm_sm  = self.module.load_function("asm_softmax").unwrap();
        let f_asm_ce  = self.module.load_function("asm_ce_grad").unwrap();
        let f_asm_wg  = self.module.load_function("asm_wgrad").unwrap();
        let f_asm_bg  = self.module.load_function("asm_bgrad").unwrap();
        let f_asm_ig  = self.module.load_function("asm_igrad").unwrap();

        let mut tgt_cpu       = vec![0i32; batch];
        let mut head_tgt_cpu  = vec![0i32; batch];
        let mut tail1_tgt_cpu = vec![0i32; batch];
        let mut tail2_tgt_cpu = vec![0i32; batch];

        for t in 0..steps {
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

            let head_tgt_gpu  = stream.clone_htod(&head_tgt_cpu).unwrap();
            let tail1_tgt_gpu = stream.clone_htod(&tail1_tgt_cpu).unwrap();
            let tail2_tgt_gpu = stream.clone_htod(&tail2_tgt_cpu).unwrap();

            let h_gpu = &h_list[t + 1];

            let mut head_logits  = stream.alloc_zeros::<f32>(batch * hs).unwrap();
            let mut proj1        = stream.alloc_zeros::<f32>(batch * d1).unwrap();
            let mut tail1_logits = stream.alloc_zeros::<f32>(batch * ts1).unwrap();
            let mut proj2        = stream.alloc_zeros::<f32>(batch * d2).unwrap();
            let mut tail2_logits = stream.alloc_zeros::<f32>(batch * ts2).unwrap();
            let mut d_proj1_buf  = stream.alloc_zeros::<f32>(batch * d1).unwrap();
            let mut d_proj2_buf  = stream.alloc_zeros::<f32>(batch * d2).unwrap();

            let batch_i = batch as i32;
            let h_i  = H   as i32;
            let hs_i = hs  as i32;
            let d1_i = d1  as i32;
            let d2_i = d2  as i32;
            let ts1_i = ts1 as i32;
            let ts2_i = ts2 as i32;
            let asm_head_i  = self.asm_head_size as i32;
            let asm_htail_i = (self.asm_head_size + ts1) as i32;
            let zero_i = 0i32;

            unsafe {
                // head
                stream.launch_builder(&f_asm_lin)
                    .arg(h_gpu).arg(&self.g_w_head).arg(&self.g_b_head)
                    .arg(&mut head_logits).arg(&h_i).arg(&hs_i).arg(&batch_i)
                    .launch(cfg2d(batch, hs)).unwrap();
                stream.launch_builder(&f_asm_sm)
                    .arg(&mut head_logits).arg(&hs_i).arg(&batch_i)
                    .launch(cfg1d(batch)).unwrap();

                // proj1 -> tail1
                stream.launch_builder(&f_asm_lin)
                    .arg(h_gpu).arg(&self.g_w_proj1).arg(&zero_d1)
                    .arg(&mut proj1).arg(&h_i).arg(&d1_i).arg(&batch_i)
                    .launch(cfg2d(batch, d1)).unwrap();
                stream.launch_builder(&f_asm_lin)
                    .arg(&proj1).arg(&self.g_w_tail1).arg(&self.g_b_tail1)
                    .arg(&mut tail1_logits).arg(&d1_i).arg(&ts1_i).arg(&batch_i)
                    .launch(cfg2d(batch, ts1)).unwrap();
                stream.launch_builder(&f_asm_sm)
                    .arg(&mut tail1_logits).arg(&ts1_i).arg(&batch_i)
                    .launch(cfg1d(batch)).unwrap();

                // proj2 -> tail2
                stream.launch_builder(&f_asm_lin)
                    .arg(h_gpu).arg(&self.g_w_proj2).arg(&zero_d2)
                    .arg(&mut proj2).arg(&h_i).arg(&d2_i).arg(&batch_i)
                    .launch(cfg2d(batch, d2)).unwrap();
                stream.launch_builder(&f_asm_lin)
                    .arg(&proj2).arg(&self.g_w_tail2).arg(&self.g_b_tail2)
                    .arg(&mut tail2_logits).arg(&d2_i).arg(&ts2_i).arg(&batch_i)
                    .launch(cfg2d(batch, ts2)).unwrap();
                stream.launch_builder(&f_asm_sm)
                    .arg(&mut tail2_logits).arg(&ts2_i).arg(&batch_i)
                    .launch(cfg1d(batch)).unwrap();

                // CE grad + loss accumulation
                stream.launch_builder(&f_asm_ce)
                    .arg(&mut head_logits).arg(&head_tgt_gpu).arg(&mut gpu_total_loss)
                    .arg(&hs_i).arg(&zero_i).arg(&batch_i)
                    .launch(cfg1d(batch)).unwrap();
                stream.launch_builder(&f_asm_ce)
                    .arg(&mut tail1_logits).arg(&tail1_tgt_gpu).arg(&mut gpu_total_loss)
                    .arg(&ts1_i).arg(&asm_head_i).arg(&batch_i)
                    .launch(cfg1d(batch)).unwrap();
                stream.launch_builder(&f_asm_ce)
                    .arg(&mut tail2_logits).arg(&tail2_tgt_gpu).arg(&mut gpu_total_loss)
                    .arg(&ts2_i).arg(&asm_htail_i).arg(&batch_i)
                    .launch(cfg1d(batch)).unwrap();

                // Weight grads — head
                stream.launch_builder(&f_asm_wg)
                    .arg(&head_logits).arg(h_gpu).arg(&mut dg_w_head)
                    .arg(&batch_i).arg(&h_i).arg(&hs_i)
                    .launch(LaunchConfig { grid_dim: (hs as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }).unwrap();
                stream.launch_builder(&f_asm_bg)
                    .arg(&head_logits).arg(&mut dg_b_head).arg(&batch_i).arg(&hs_i)
                    .launch(cfg1d(hs)).unwrap();

                // tail1
                stream.launch_builder(&f_asm_wg)
                    .arg(&tail1_logits).arg(&proj1).arg(&mut dg_w_tail1)
                    .arg(&batch_i).arg(&d1_i).arg(&ts1_i)
                    .launch(LaunchConfig { grid_dim: (ts1 as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }).unwrap();
                stream.launch_builder(&f_asm_bg)
                    .arg(&tail1_logits).arg(&mut dg_b_tail1).arg(&batch_i).arg(&ts1_i)
                    .launch(cfg1d(ts1)).unwrap();

                // tail2
                stream.launch_builder(&f_asm_wg)
                    .arg(&tail2_logits).arg(&proj2).arg(&mut dg_w_tail2)
                    .arg(&batch_i).arg(&d2_i).arg(&ts2_i)
                    .launch(LaunchConfig { grid_dim: (ts2 as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }).unwrap();
                stream.launch_builder(&f_asm_bg)
                    .arg(&tail2_logits).arg(&mut dg_b_tail2).arg(&batch_i).arg(&ts2_i)
                    .launch(cfg1d(ts2)).unwrap();

                // d_proj1
                stream.launch_builder(&f_asm_ig)
                    .arg(&tail1_logits).arg(&self.g_w_tail1).arg(&mut d_proj1_buf)
                    .arg(&batch_i).arg(&d1_i).arg(&ts1_i)
                    .launch(cfg2d(batch, d1)).unwrap();
                stream.launch_builder(&f_asm_wg)
                    .arg(&d_proj1_buf).arg(h_gpu).arg(&mut dg_w_proj1)
                    .arg(&batch_i).arg(&h_i).arg(&d1_i)
                    .launch(LaunchConfig { grid_dim: (d1 as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }).unwrap();

                // d_proj2
                stream.launch_builder(&f_asm_ig)
                    .arg(&tail2_logits).arg(&self.g_w_tail2).arg(&mut d_proj2_buf)
                    .arg(&batch_i).arg(&d2_i).arg(&ts2_i)
                    .launch(cfg2d(batch, d2)).unwrap();
                stream.launch_builder(&f_asm_wg)
                    .arg(&d_proj2_buf).arg(h_gpu).arg(&mut dg_w_proj2)
                    .arg(&batch_i).arg(&h_i).arg(&d2_i)
                    .launch(LaunchConfig { grid_dim: (d2 as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }).unwrap();

                // d_h[t]
                let d_h = &mut d_h_steps[t];
                stream.launch_builder(&f_asm_ig)
                    .arg(&head_logits).arg(&self.g_w_head).arg(&mut *d_h)
                    .arg(&batch_i).arg(&h_i).arg(&hs_i)
                    .launch(cfg2d(batch, H)).unwrap();
                stream.launch_builder(&f_asm_ig)
                    .arg(&d_proj1_buf).arg(&self.g_w_proj1).arg(&mut *d_h)
                    .arg(&batch_i).arg(&h_i).arg(&d1_i)
                    .launch(cfg2d(batch, H)).unwrap();
                stream.launch_builder(&f_asm_ig)
                    .arg(&d_proj2_buf).arg(&self.g_w_proj2).arg(&mut *d_h)
                    .arg(&batch_i).arg(&h_i).arg(&d2_i)
                    .launch(cfg2d(batch, H)).unwrap();
            }
        }

        // ── BACKWARD THROUGH LSTM ─────────────────────────────
        let f_lstm_bwd = self.module.load_function("fused_lstm_bwd").unwrap();
        let f_red      = self.module.load_function("reduce_sum_batch").unwrap();

        let mut d_c_next     = stream.alloc_zeros::<f32>(batch * H).unwrap();
        let mut d_gates_bwd  = stream.alloc_zeros::<f32>(batch * fh).unwrap();
        let mut d_c_prev_bwd = stream.alloc_zeros::<f32>(batch * H).unwrap();
        let mut d_x_buf      = stream.alloc_zeros::<f32>(batch * E).unwrap();

        let f_emb_bwd = self.module.load_function("embedding_bwd").unwrap();

        for t in (0..steps).rev() {
            stream.memset_zeros(&mut d_gates_bwd).unwrap();
            stream.memset_zeros(&mut d_c_prev_bwd).unwrap();

            let batch_i = batch as i32;
            let h_i     = H  as i32;
            let fh_i    = fh as i32;
            let e_i     = E  as i32;
            unsafe {
                stream.launch_builder(&f_lstm_bwd)
                    .arg(&gates_list[t])
                    .arg(&c_list[t])
                    .arg(&c_list[t+1])
                    .arg(&d_h_steps[t])
                    .arg(&d_c_next)
                    .arg(&mut d_gates_bwd)
                    .arg(&mut d_c_prev_bwd)
                    .arg(&h_i)
                    .launch(cfg2d(batch, H)).unwrap();
            }
            std::mem::swap(&mut d_c_next, &mut d_c_prev_bwd);

            // d_w_h += h[t]^T @ d_gates   [H, fh]
            sgemm(&self.blas, true, false, H, fh, batch, 1.0, 1.0,
                  &h_list[t], &d_gates_bwd, &mut d_w_h);
            // d_w_x += x[t]^T @ d_gates   [E, fh]
            sgemm(&self.blas, true, false, E, fh, batch, 1.0, 1.0,
                  &xs_list[t], &d_gates_bwd, &mut d_w_x);

            unsafe {
                stream.launch_builder(&f_red)
                    .arg(&d_gates_bwd)
                    .arg(&mut d_b)
                    .arg(&batch_i)
                    .arg(&fh_i)
                    .launch(cfg1d(fh)).unwrap();
            }

            // d_x = d_gates @ w_x^T   [batch, E]
            stream.memset_zeros(&mut d_x_buf).unwrap();
            sgemm(&self.blas, false, true, batch, E, fh, 1.0, 0.0,
                  &d_gates_bwd, &self.w_x, &mut d_x_buf);

            let tok_step = &tok_all_cpu[t * batch .. (t + 1) * batch];
            let tok_gpu  = stream.clone_htod(tok_step).unwrap();
            unsafe {
                stream.launch_builder(&f_emb_bwd)
                    .arg(&d_x_buf)
                    .arg(&tok_gpu)
                    .arg(&mut d_embed)
                    .arg(&e_i)
                    .launch(cfg2d(batch, E)).unwrap();
            }
        }

        // ── ADAM UPDATE ───────────────────────────────────────
        self.adam_step += 1;
        let t   = self.adam_step;
        let b1  = 0.9f32; let b2 = 0.999f32; let eps = 1e-8f32;
        let bc1 = 1.0 - b1.powi(t);
        let bc2 = 1.0 - b2.powi(t);
        let lr  = learning_rate as f32;

        let f_norm = self.module.load_function("norm_reduce").unwrap();
        let f_adam = self.module.load_function("adam_update").unwrap();

        let wg: usize = 256;
        let clip_val = 5.0f32;

        // Max ngroups across all LSTM params (w_h = H*4H is largest)
        let max_n = [self.vocab_size * E, E * fh, H * fh, fh].iter().copied().max().unwrap();
        let max_ngroups = (max_n + wg - 1) / wg;
        let mut partial_buf = stream.alloc_zeros::<f32>(max_ngroups).unwrap();

        macro_rules! lstm_adam {
            ($param:expr, $m:expr, $v:expr, $grad:expr, $n:expr) => {{
                let nn: usize = $n;
                let n_i = nn as i32;
                let ngroups = (nn + wg - 1) / wg;

                // GPU norm reduction → CPU finish → GPU scale if needed
                stream.memset_zeros(&mut partial_buf).unwrap();
                unsafe {
                    stream.launch_builder(&f_norm)
                        .arg(&$grad)
                        .arg(&mut partial_buf)
                        .arg(&n_i)
                        .launch(LaunchConfig {
                            grid_dim: (ngroups as u32, 1, 1),
                            block_dim: (wg as u32, 1, 1),
                            shared_mem_bytes: (wg * 4) as u32,
                        }).unwrap();
                }
                let partial_cpu = download(&stream, &partial_buf);
                let norm = partial_cpu[..ngroups].iter().sum::<f32>().sqrt();
                if norm > clip_val {
                    let scale = clip_val / norm;
                    // scale gradient in-place using adam_update trick: just pre-scale grad
                    // We do this with a single kernel launch using elem-wise multiply
                    // For simplicity: download, scale, re-upload (only when clipping fires)
                    let mut g_cpu = download(&stream, &$grad);
                    for v in &mut g_cpu { *v *= scale; }
                    $grad = upload(&stream, &g_cpu);
                }
                unsafe {
                    stream.launch_builder(&f_adam)
                        .arg(&mut $param).arg(&mut $m).arg(&mut $v).arg(&$grad)
                        .arg(&lr).arg(&b1).arg(&b2).arg(&eps).arg(&bc1).arg(&bc2).arg(&n_i)
                        .launch(cfg1d(nn)).unwrap();
                }
            }};
        }

        lstm_adam!(self.embed,  self.m_embed, self.v_embed, d_embed, self.vocab_size * E);
        lstm_adam!(self.w_x,    self.m_w_x,   self.v_w_x,   d_w_x,  E * fh);
        lstm_adam!(self.w_h,    self.m_w_h,   self.v_w_h,   d_w_h,  H * fh);
        lstm_adam!(self.b,      self.m_b,     self.v_b,     d_b,    fh);

        // ASM Adam
        self.asm_adam_step += 1;
        let at   = self.asm_adam_step;
        let abc1 = 1.0 - 0.9f32.powi(at);
        let abc2 = 1.0 - 0.999f32.powi(at);

        macro_rules! asm_adam {
            ($grad:expr, $param:expr, $m:expr, $v:expr, $n:expr) => {{
                let nn: usize = $n;
                let n_i = nn as i32;
                let b1_a = 0.9f32; let b2_a = 0.999f32; let eps_a = 1e-8f32;
                unsafe {
                    stream.launch_builder(&f_adam)
                        .arg(&mut $param).arg(&mut $m).arg(&mut $v).arg(&$grad)
                        .arg(&lr).arg(&b1_a).arg(&b2_a).arg(&eps_a).arg(&abc1).arg(&abc2).arg(&n_i)
                        .launch(cfg1d(nn)).unwrap();
                }
            }};
        }

        asm_adam!(dg_w_head,  self.g_w_head,  self.gm_w_head,  self.gv_w_head,  hs * H);
        asm_adam!(dg_b_head,  self.g_b_head,  self.gm_b_head,  self.gv_b_head,  hs);
        asm_adam!(dg_w_proj1, self.g_w_proj1, self.gm_w_proj1, self.gv_w_proj1, d1 * H);
        asm_adam!(dg_w_tail1, self.g_w_tail1, self.gm_w_tail1, self.gv_w_tail1, ts1 * d1);
        asm_adam!(dg_b_tail1, self.g_b_tail1, self.gm_b_tail1, self.gv_b_tail1, ts1);
        asm_adam!(dg_w_proj2, self.g_w_proj2, self.gm_w_proj2, self.gv_w_proj2, d2 * H);
        asm_adam!(dg_w_tail2, self.g_w_tail2, self.gm_w_tail2, self.gv_w_tail2, ts2 * d2);
        asm_adam!(dg_b_tail2, self.g_b_tail2, self.gm_b_tail2, self.gv_b_tail2, ts2);

        // Single GPU→CPU readback for loss
        let loss_val = download(&self.stream, &gpu_total_loss)[0];
        loss_val / (steps * batch) as f32
    }

    pub fn backward_step(&mut self, tokens: &[usize], learning_rate: f64) -> f32 {
        self.train_batch(&[tokens.to_vec()], learning_rate)
    }

    // ── Sampling ──────────────────────────────────────────────
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

    // ── Save / Load ───────────────────────────────────────────
    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let data = serde_json::json!({
            "vocab_size": self.vocab_size,
            "embed_dim":  self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "format":     "v12_cuda",
            "embed": download(&self.stream, &self.embed),
            "w_x":   download(&self.stream, &self.w_x),
            "w_h":   download(&self.stream, &self.w_h),
            "b":     download(&self.stream, &self.b),
            "adaptive_sm": self.adaptive_sm.to_json(),
        });
        fs::write(path, serde_json::to_string(&data)?)?;
        Ok(())
    }

    pub fn load(path: &str, vocab_size: usize, embed_dim: usize, hidden_dim: usize)
        -> anyhow::Result<Self>
    {
        let data: serde_json::Value = serde_json::from_str(&fs::read_to_string(path)?)?;
        let mut model = LSTMModelCuda::new(vocab_size, embed_dim, hidden_dim);
        let fh = 4 * hidden_dim;

        macro_rules! load_gpu {
            ($field:ident, $key:expr, $n:expr) => {
                if let Some(arr) = data[$key].as_array() {
                    let v: Vec<f32> = arr.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect();
                    if v.len() == $n { model.$field = upload(&model.stream, &v); }
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

// ─────────────────────────────────────────────────────────────
//  Sequence cache helpers
// ─────────────────────────────────────────────────────────────
fn save_seq_cache(path: &str, seqs: &[Vec<usize>]) {
    let mut buf: Vec<u8> = Vec::with_capacity(seqs.len() * 20 * 2);
    let count = seqs.len() as u32;
    buf.extend_from_slice(&count.to_le_bytes());
    for seq in seqs {
        buf.extend_from_slice(&(seq.len() as u16).to_le_bytes());
        for &tok in seq {
            buf.extend_from_slice(&(tok as u16).to_le_bytes());
        }
    }
    if let Ok(mut f) = fs::File::create(path) { f.write_all(&buf).ok(); }
}

fn load_seq_cache(path: &str) -> Option<Vec<Vec<usize>>> {
    let data = fs::read(path).ok()?;
    if data.len() < 4 { return None; }
    let count = u32::from_le_bytes(data[0..4].try_into().ok()?) as usize;
    let mut seqs = Vec::with_capacity(count);
    let mut i = 4usize;
    for _ in 0..count {
        if i + 2 > data.len() { return None; }
        let len = u16::from_le_bytes(data[i..i+2].try_into().ok()?) as usize;
        i += 2;
        if i + len * 2 > data.len() { return None; }
        let seq: Vec<usize> = (0..len)
            .map(|k| u16::from_le_bytes(data[i+k*2..i+k*2+2].try_into().unwrap()) as usize)
            .collect();
        i += len * 2;
        seqs.push(seq);
    }
    Some(seqs)
}

// ─────────────────────────────────────────────────────────────
//  Pre-training entry point
// ─────────────────────────────────────────────────────────────
const LEARNING_RATE:       f64   = 0.0003;
const MAX_TOKENS_PER_SEQ:  usize = 80;
const MIN_TOKENS_PER_SEQ:  usize = 4;
const PRETRAIN_EPOCHS:     usize = 5;
const PRETRAIN_BATCH_SIZE: usize = 512;

pub fn pretrain_from_files(
    model: &mut LSTMModelCuda,
    tokenizer: &mut Tokenizer,
    data_dir: &str,
) -> anyhow::Result<()> {
    let path = Path::new(data_dir);
    if !path.exists() { println!("Data dir not found."); return Ok(()); }

    let start    = Instant::now();
    let _num_cpus = thread::available_parallelism().map(|n| n.get()).unwrap_or(4);

    println!("\n===============================================");
    println!("       ARIA - CUDA cuBLAS TRAINING            ");
    println!("===============================================");
    println!("LR: {}  Epochs: {}  Batch: {}", LEARNING_RATE, PRETRAIN_EPOCHS, PRETRAIN_BATCH_SIZE);
    println!("===============================================\n");

    let cache_path = format!("{}/sequences_cache.bin", data_dir);
    let mut all_seqs: Vec<Vec<usize>> = if let Some(cached) = load_seq_cache(&cache_path) {
        println!("Loaded {} sequences from cache\n", cached.len());
        cached
    } else {
        use rayon::prelude::*;
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
        if files.is_empty() { return Ok(()); }

        let seqs: Vec<Vec<usize>> = files.iter()
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

        println!("Saving sequence cache...");
        save_seq_cache(&cache_path, &seqs);
        seqs
    };

    println!("{} sequences\n", all_seqs.len());
    if all_seqs.is_empty() { return Ok(()); }

    let total_batches = (all_seqs.len() + PRETRAIN_BATCH_SIZE - 1) / PRETRAIN_BATCH_SIZE;
    let mut current_lr = LEARNING_RATE;

    for epoch in 0..PRETRAIN_EPOCHS {
        let ep = Instant::now();
        let mut last_report = Instant::now();
        all_seqs.shuffle(&mut rand::thread_rng());
        let mut total_loss = 0.0f32;
        let mut batches    = 0usize;
        let mut seqs_done  = 0usize;

        for chunk in all_seqs.chunks(PRETRAIN_BATCH_SIZE) {
            let loss = model.train_batch(chunk, current_lr);
            if loss.is_finite() { total_loss += loss; batches += 1; }
            seqs_done += chunk.len();

            if last_report.elapsed().as_secs_f32() >= 10.0 {
                let avg       = total_loss / batches.max(1) as f32;
                let elapsed   = ep.elapsed().as_secs_f32();
                let seq_s     = seqs_done as f32 / elapsed;
                let remaining = total_batches.saturating_sub(batches);
                println!("  Epoch {}/{}  |  batch {}/{}  ({} remaining)  |  loss={:.4}  |  {:.0} seq/s",
                         epoch+1, PRETRAIN_EPOCHS, batches, total_batches, remaining, avg, seq_s);
                std::io::stdout().flush().ok();
                last_report = Instant::now();
            }
        }

        let avg   = total_loss / batches.max(1) as f32;
        let et    = ep.elapsed();
        let seq_s = all_seqs.len() as f32 / et.as_secs_f32();
        println!("Epoch {}/{} done  |  loss={:.6}  |  {:.1}s  |  {:.0} seq/s  |  lr={:.6}",
                 epoch+1, PRETRAIN_EPOCHS, avg, et.as_secs_f32(), seq_s, current_lr);
        current_lr *= 0.85;
    }

    println!("\nTotal: {:.1}s", start.elapsed().as_secs_f32());
    Ok(())
}
