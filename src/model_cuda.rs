use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use std::io::{Write, BufWriter, Read, Seek, SeekFrom};
use std::cell::Cell;

use cudarc::driver::{CudaStream, CudaSlice, CudaModule, LaunchConfig, PinnedHostSlice};
use cudarc::driver::PushKernelArg;
use cudarc::nvrtc::Ptx;
use cudarc::cublas::{CudaBlas, GemmConfig, Gemm, sys::cublasOperation_t};
use half::f16;

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
    "reduce_sum_batch", "reduce_sum", "adam_update", "adam_update_f16", "sgd_update_f16", "scale_f16",
    "norm_reduce", "norm_reduce_f16", "clip_if_needed", "clip_if_needed_f16",
    "zero_float",
];

// ─────────────────────────────────────────────────────────────
//  LSTM state (CPU side, small)
// ─────────────────────────────────────────────────────────────
pub struct LSTMState {
    pub h: Vec<f32>,
    pub c: Vec<f32>,
}

// ──────────────────────────────────────────────────────────────
//  Model
// ─────────────────────────────────────────────────────────────
struct CudaFns {
    emb_fwd:  cudarc::driver::CudaFunction,
    emb_bwd:  cudarc::driver::CudaFunction,
    bias:     cudarc::driver::CudaFunction,
    lstm_fwd: cudarc::driver::CudaFunction,
    lstm_bwd: cudarc::driver::CudaFunction,
    red_sum:  cudarc::driver::CudaFunction,
    red_one:  cudarc::driver::CudaFunction,
    asm_lin:  cudarc::driver::CudaFunction,
    asm_sm:   cudarc::driver::CudaFunction,
    asm_ce:   cudarc::driver::CudaFunction,
    asm_wg:   cudarc::driver::CudaFunction,
    asm_bg:   cudarc::driver::CudaFunction,
    asm_ig:   cudarc::driver::CudaFunction,
    norm_red: cudarc::driver::CudaFunction,
    norm_red_f16: cudarc::driver::CudaFunction,
    clip:     cudarc::driver::CudaFunction,
    clip_f16: cudarc::driver::CudaFunction,
    scale_f16: cudarc::driver::CudaFunction,
    adam:     cudarc::driver::CudaFunction,
    adam_f16: cudarc::driver::CudaFunction,
    sgd_f16:  cudarc::driver::CudaFunction,
}

// ──────────────────────────────────────────────────────────────
//  Per-batch reusable GPU buffers (allocated once, reused)
//  All activations/grads/weights are f16; Adam moments are f32.
// ──────────────────────────────────────────────────────────────
struct BatchBufs {
    batch:  usize,
    steps:  usize,
    xs:          Vec<CudaSlice<f16>>,
    gates:       Vec<CudaSlice<f16>>,
    h:           Vec<CudaSlice<f16>>,
    c:           Vec<CudaSlice<f16>>,
    head_logits: Vec<CudaSlice<f16>>,
    simple_logits: Vec<CudaSlice<f16>>,
    d_simple_logits: Vec<CudaSlice<f16>>,
    proj1:       Vec<CudaSlice<f16>>,
    tail1_log:   Vec<CudaSlice<f16>>,
    proj2:       Vec<CudaSlice<f16>>,
    tail2_log:   Vec<CudaSlice<f16>>,
    d_proj1:     Vec<CudaSlice<f16>>,
    d_proj2:     Vec<CudaSlice<f16>>,
    d_h:         Vec<CudaSlice<f16>>,
    d_embed:     CudaSlice<f16>,
    d_w_x:       CudaSlice<f16>,
    d_w_h:       CudaSlice<f16>,
    d_b:         CudaSlice<f16>,
    d_gates:     CudaSlice<f16>,
    d_c_next:    CudaSlice<f16>,
    d_c_prev:    CudaSlice<f16>,
    d_x:         CudaSlice<f16>,
    dg_w_head:   CudaSlice<f16>,
    dg_b_head:   CudaSlice<f16>,
    dg_w_simple: CudaSlice<f16>,
    dg_b_simple: CudaSlice<f16>,
    dg_w_proj1:  CudaSlice<f16>,
    dg_w_tail1:  CudaSlice<f16>,
    dg_b_tail1:  CudaSlice<f16>,
    dg_w_proj2:  CudaSlice<f16>,
    dg_w_tail2:  CudaSlice<f16>,
    dg_b_tail2:  CudaSlice<f16>,
    gpu_loss:    [CudaSlice<f32>; 2],
    partial:     CudaSlice<f32>,
    zero_d1:     CudaSlice<f16>,
    zero_d2:     CudaSlice<f16>,
    tok_bufs:    Vec<CudaSlice<i32>>,
    head_tgt:    Vec<CudaSlice<i32>>,
    simple_tgt:  Vec<CudaSlice<i32>>,
    tail1_tgt:   Vec<CudaSlice<i32>>,
    tail2_tgt:   Vec<CudaSlice<i32>>,
    loss_part:   CudaSlice<f32>,
}

impl BatchBufs {
    fn alloc(stream: &Arc<CudaStream>, batch: usize, steps: usize,
             E: usize, H: usize, fh: usize,
             hs: usize, d1: usize, d2: usize, ts1: usize, ts2: usize,
             vocab: usize, max_ngroups: usize) -> Self {
        let az    = |n| stream.alloc_zeros::<f16>(n).unwrap();
        let az_f32= |n| stream.alloc_zeros::<f32>(n).unwrap();
        let azi   = |n| stream.alloc_zeros::<i32>(n).unwrap();
        Self {
            batch, steps,
            xs:          (0..steps).map(|_| az(batch*E)).collect(),
            gates:       (0..steps).map(|_| az(batch*fh)).collect(),
            h:           (0..=steps).map(|_| az(batch*H)).collect(),
            c:           (0..=steps).map(|_| az(batch*H)).collect(),
            head_logits: (0..steps).map(|_| az(batch*hs)).collect(),
            simple_logits: (0..steps).map(|_| az(batch*vocab)).collect(),
            d_simple_logits: (0..steps).map(|_| az(batch*vocab)).collect(),
            proj1:       (0..steps).map(|_| az(batch*d1)).collect(),
            tail1_log:   (0..steps).map(|_| az(batch*ts1)).collect(),
            proj2:       (0..steps).map(|_| az(batch*d2)).collect(),
            tail2_log:   (0..steps).map(|_| az(batch*ts2)).collect(),
            d_proj1:     (0..steps).map(|_| az(batch*d1)).collect(),
            d_proj2:     (0..steps).map(|_| az(batch*d2)).collect(),
            d_h:         (0..steps).map(|_| az(batch*H)).collect(),
            d_embed: az(vocab*E), d_w_x: az(E*fh), d_w_h: az(H*fh), d_b: az(fh),
            d_gates: az(batch*fh), d_c_next: az(batch*H), d_c_prev: az(batch*H), d_x: az(batch*E),
            dg_w_head: az(hs*H), dg_b_head: az(hs),
            dg_w_simple: az(vocab*H), dg_b_simple: az(vocab),
            dg_w_proj1: az(d1*H), dg_w_tail1: az(ts1*d1), dg_b_tail1: az(ts1),
            dg_w_proj2: az(d2*H), dg_w_tail2: az(ts2*d2), dg_b_tail2: az(ts2),
            gpu_loss: [az_f32(1), az_f32(1)], partial: az_f32(max_ngroups),
            zero_d1: az(d1), zero_d2: az(d2),
            tok_bufs:  (0..steps).map(|_| azi(batch)).collect(),
            head_tgt:  (0..steps).map(|_| azi(batch)).collect(),
            simple_tgt: (0..steps).map(|_| azi(batch)).collect(),
            tail1_tgt: (0..steps).map(|_| azi(batch)).collect(),
            tail2_tgt: (0..steps).map(|_| azi(batch)).collect(),
            loss_part: az_f32(3 * ((batch + 31) / 32) * steps),
        }
    }

    fn fits(&self, batch: usize, steps: usize) -> bool {
        self.batch >= batch && self.steps >= steps
    }
}

// ──────────────────────────────────────────────────────────────
pub struct LSTMModelCuda {
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    blas:   CudaBlas,
    fns:    CudaFns,
    bufs:   Option<BatchBufs>,

    // weights + activations (f16)
    pub embed: CudaSlice<f16>,
    pub w_x:   CudaSlice<f16>,
    pub w_h:   CudaSlice<f16>,
    pub b:     CudaSlice<f16>,

    // Adam moments (f32 for stability)
    m_embed: CudaSlice<f32>, v_embed: CudaSlice<f32>,
    m_w_x:   CudaSlice<f32>, v_w_x:   CudaSlice<f32>,
    m_w_h:   CudaSlice<f32>, v_w_h:   CudaSlice<f32>,
    m_b:     CudaSlice<f32>, v_b:     CudaSlice<f32>,

    pub adaptive_sm: AdaptiveSoftmax,

    // Optional simple linear+softmax head for debugging (used when ARIA_SIMPLE_SOFTMAX=1)
    w_simple: CudaSlice<f16>,
    b_simple: CudaSlice<f16>,
    m_w_simple: CudaSlice<f32>, v_w_simple: CudaSlice<f32>,
    m_b_simple: CudaSlice<f32>, v_b_simple: CudaSlice<f32>,
    simple_adam_step: i32,

    g_w_head:  CudaSlice<f16>, g_b_head:  CudaSlice<f16>,
    g_w_proj1: CudaSlice<f16>,
    g_w_tail1: CudaSlice<f16>, g_b_tail1: CudaSlice<f16>,
    g_w_proj2: CudaSlice<f16>,
    g_w_tail2: CudaSlice<f16>, g_b_tail2: CudaSlice<f16>,

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
    loss_idx:      usize,
    host_loss:     [PinnedHostSlice<f32>; 2],

    asm_head_size:  usize,
    asm_tail1_size: usize,
    asm_tail2_size: usize,
    asm_dim1:       usize,
    asm_dim2:       usize,
    use_simple_softmax: bool,

    profile_step: Cell<usize>,
}

// ──────────────────────────────────────────────────────────────
//  Helpers
// ──────────────────────────────────────────────────────────────
fn randn_vec(n: usize, scale: f32) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen::<f32>() * 2.0 * scale - scale).collect()
}

fn to_f16(v: &[f32]) -> Vec<f16> { v.iter().map(|&x| f16::from_f32(x)).collect() }
fn from_f16(v: &[f16]) -> Vec<f32> { v.iter().map(|&x| x.to_f32()).collect() }

fn upload_f16(stream: &Arc<CudaStream>, data: &[f32]) -> CudaSlice<f16> {
    let h = to_f16(data);
    stream.clone_htod(&h).unwrap()
}

fn zeros_f16(stream: &Arc<CudaStream>, n: usize) -> CudaSlice<f16> {
    stream.alloc_zeros::<f16>(n).unwrap()
}

fn zeros_f32(stream: &Arc<CudaStream>, n: usize) -> CudaSlice<f32> {
    stream.alloc_zeros::<f32>(n).unwrap()
}

fn download_f16(stream: &Arc<CudaStream>, buf: &CudaSlice<f16>) -> Vec<f32> {
    stream.synchronize().unwrap();
    let h: Vec<f16> = stream.clone_dtoh(buf).unwrap();
    from_f16(&h)
}

fn hf(v: f32) -> f16 { f16::from_f32(v) }

// Generic cuBLAS GEMM wrapper for row-major C = alpha*op(A)*op(B) + beta*C
fn gemm<T: cudarc::driver::DeviceRepr>(
    blas: &CudaBlas,
    transa: bool, transb: bool,
    m: usize, n: usize, k: usize,
    alpha: T, beta: T,
    a: &CudaSlice<T>,
    b: &CudaSlice<T>,
    c: &mut CudaSlice<T>,
) where CudaBlas: Gemm<T> {
    use cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T};
    if !transa && !transb {
        unsafe {
            blas.gemm(GemmConfig::<T> {
                transa: CUBLAS_OP_N, transb: CUBLAS_OP_N,
                m: n as i32, n: m as i32, k: k as i32,
                alpha, lda: n as i32, ldb: k as i32, beta, ldc: n as i32,
            }, b, a, c).unwrap();
        }
    } else if transa && !transb {
        unsafe {
            blas.gemm(GemmConfig::<T> {
                transa: CUBLAS_OP_N, transb: CUBLAS_OP_T,
                m: n as i32, n: m as i32, k: k as i32,
                alpha, lda: n as i32, ldb: m as i32, beta, ldc: n as i32,
            }, b, a, c).unwrap();
        }
    } else {
        unsafe {
            blas.gemm(GemmConfig::<T> {
                transa: CUBLAS_OP_T, transb: CUBLAS_OP_N,
                m: n as i32, n: m as i32, k: k as i32,
                alpha, lda: k as i32, ldb: k as i32, beta, ldc: n as i32,
            }, b, a, c).unwrap();
        }
    }
}

// ──────────────────────────────────────────────────────────────
//  Launch helpers
// ──────────────────────────────────────────────────────────────
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

fn cfg_warp_per_row(batch: usize) -> LaunchConfig {
    let rows_per_block = 32usize;
    let gy = (batch + rows_per_block - 1) / rows_per_block;
    LaunchConfig {
        grid_dim: (1, gy as u32, 1),
        block_dim: (32, rows_per_block as u32, 1),
        shared_mem_bytes: (4 * rows_per_block) as u32,
    }
}

fn cfg_reduce(n: usize) -> LaunchConfig {
    let threads = 256usize;
    let blocks = (n + threads - 1) / threads;
    LaunchConfig { grid_dim: (blocks as u32, 1, 1), block_dim: (threads as u32, 1, 1), shared_mem_bytes: (threads * 4) as u32 }
}

// ──────────────────────────────────────────────────────────────
//  impl LSTMModelCuda
// ──────────────────────────────────────────────────────────────
impl LSTMModelCuda {
    pub fn new(vocab_size: usize, embed_dim: usize, hidden_dim: usize) -> Self {
        let gpu    = GpuContext::try_init().expect("No CUDA GPU found");
        let ctx    = gpu.ctx;
        let stream = gpu.stream;
        let blas   = gpu.blas;

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

        let use_simple_softmax = std::env::var("ARIA_SIMPLE_SOFTMAX").is_ok();

        let simple_params = if use_simple_softmax { vocab_size * hidden_dim + vocab_size } else { 0 };
        let lstm_params = vocab_size*embed_dim + embed_dim*fh + hidden_dim*fh + fh;
        let asm_params  = hs*hidden_dim + asm_dim1*hidden_dim + asm_tail1_size*asm_dim1
            + asm_dim2*hidden_dim + asm_tail2_size*asm_dim2;

        println!("================================");
        println!("        ARIA  CUDA / cuBLAS     ");
        println!("================================");
        println!("  Vocab:   {}", vocab_size);
        println!("  Embed:   {}", embed_dim);
        println!("  Hidden:  {}", hidden_dim);
        println!("  Params:  ~{:.1}M", (lstm_params + asm_params + simple_params) as f64 / 1e6);
        if use_simple_softmax {
            println!("  Softmax: simple linear (DEBUG)");
        } else {
            println!("  ASoftmax: head={} tail1={} tail2={}",
                     asm_head_size, asm_tail1_size, asm_tail2_size);
        }
        println!("  Precision: FP16 (FP32 Adam moments)");
        println!("================================\n");

        let w_simple_data = if use_simple_softmax { randn_vec(vocab_size * hidden_dim, (2.0 / (vocab_size + hidden_dim) as f32).sqrt() * 0.01f32) } else { vec![] };
        let b_simple_data = if use_simple_softmax { vec![0.0f32; vocab_size] } else { vec![] };

        let fns = CudaFns {
            emb_fwd:  module.load_function("embedding_fwd").unwrap(),
            emb_bwd:  module.load_function("embedding_bwd").unwrap(),
            bias:     module.load_function("add_bias").unwrap(),
            lstm_fwd: module.load_function("fused_lstm_fwd").unwrap(),
            lstm_bwd: module.load_function("fused_lstm_bwd").unwrap(),
            red_sum:  module.load_function("reduce_sum_batch").unwrap(),
            red_one:  module.load_function("reduce_sum").unwrap(),
            asm_lin:  module.load_function("asm_linear").unwrap(),
            asm_sm:   module.load_function("asm_softmax").unwrap(),
            asm_ce:   module.load_function("asm_ce_grad").unwrap(),
            asm_wg:   module.load_function("asm_wgrad").unwrap(),
            asm_bg:   module.load_function("asm_bgrad").unwrap(),
            asm_ig:   module.load_function("asm_igrad").unwrap(),
            norm_red: module.load_function("norm_reduce").unwrap(),
            norm_red_f16: module.load_function("norm_reduce_f16").unwrap(),
            clip:     module.load_function("clip_if_needed").unwrap(),
            clip_f16: module.load_function("clip_if_needed_f16").unwrap(),
            adam:     module.load_function("adam_update").unwrap(),
            adam_f16: module.load_function("adam_update_f16").unwrap(),
            sgd_f16:  module.load_function("sgd_update_f16").unwrap(),
            scale_f16: module.load_function("scale_f16").unwrap(),
        };

        LSTMModelCuda {
            embed: upload_f16(&stream, &embed_data),
            w_x:   upload_f16(&stream, &w_x_data),
            w_h:   upload_f16(&stream, &w_h_data),
            b:     upload_f16(&stream, &b_data),
            m_embed: zeros_f32(&stream, vocab_size * embed_dim),
            v_embed: zeros_f32(&stream, vocab_size * embed_dim),
            m_w_x: zeros_f32(&stream, embed_dim * fh),
            v_w_x: zeros_f32(&stream, embed_dim * fh),
            m_w_h: zeros_f32(&stream, hidden_dim * fh),
            v_w_h: zeros_f32(&stream, hidden_dim * fh),
            m_b:   zeros_f32(&stream, fh),
            v_b:   zeros_f32(&stream, fh),

            w_simple: if use_simple_softmax { upload_f16(&stream, &w_simple_data) } else { zeros_f16(&stream, 1) },
            b_simple: if use_simple_softmax { upload_f16(&stream, &b_simple_data) } else { zeros_f16(&stream, 1) },
            m_w_simple: if use_simple_softmax { zeros_f32(&stream, vocab_size * hidden_dim) } else { zeros_f32(&stream, 1) },
            v_w_simple: if use_simple_softmax { zeros_f32(&stream, vocab_size * hidden_dim) } else { zeros_f32(&stream, 1) },
            m_b_simple: if use_simple_softmax { zeros_f32(&stream, vocab_size) } else { zeros_f32(&stream, 1) },
            v_b_simple: if use_simple_softmax { zeros_f32(&stream, vocab_size) } else { zeros_f32(&stream, 1) },
            simple_adam_step: 0,

            g_w_head:  upload_f16(&stream, &adaptive_sm.w_head),
            g_b_head:  upload_f16(&stream, &adaptive_sm.b_head),
            g_w_proj1: upload_f16(&stream, &adaptive_sm.w_proj1),
            g_w_tail1: upload_f16(&stream, &adaptive_sm.w_tail1),
            g_b_tail1: upload_f16(&stream, &adaptive_sm.b_tail1),
            g_w_proj2: upload_f16(&stream, &adaptive_sm.w_proj2),
            g_w_tail2: upload_f16(&stream, &adaptive_sm.w_tail2),
            g_b_tail2: upload_f16(&stream, &adaptive_sm.b_tail2),

            gm_w_head:  zeros_f32(&stream, hs * hidden_dim),
            gv_w_head:  zeros_f32(&stream, hs * hidden_dim),
            gm_b_head:  zeros_f32(&stream, hs),
            gv_b_head:  zeros_f32(&stream, hs),
            gm_w_proj1: zeros_f32(&stream, asm_dim1 * hidden_dim),
            gv_w_proj1: zeros_f32(&stream, asm_dim1 * hidden_dim),
            gm_w_tail1: zeros_f32(&stream, asm_tail1_size * asm_dim1),
            gv_w_tail1: zeros_f32(&stream, asm_tail1_size * asm_dim1),
            gm_b_tail1: zeros_f32(&stream, asm_tail1_size),
            gv_b_tail1: zeros_f32(&stream, asm_tail1_size),
            gm_w_proj2: zeros_f32(&stream, asm_dim2 * hidden_dim),
            gv_w_proj2: zeros_f32(&stream, asm_dim2 * hidden_dim),
            gm_w_tail2: zeros_f32(&stream, asm_tail2_size * asm_dim2),
            gv_w_tail2: zeros_f32(&stream, asm_tail2_size * asm_dim2),
            gm_b_tail2: zeros_f32(&stream, asm_tail2_size),
            gv_b_tail2: zeros_f32(&stream, asm_tail2_size),

            stream, module, blas, fns, bufs: None,
            adaptive_sm,
            vocab_size, embed_dim, hidden_dim,
            adam_step: 0, asm_adam_step: 0, loss_idx: 0,
            asm_head_size, asm_tail1_size, asm_tail2_size, asm_dim1, asm_dim2,
            use_simple_softmax,
            profile_step: Cell::new(0),
            host_loss: [
                unsafe { ctx.alloc_pinned::<f32>(1).unwrap() },
                unsafe { ctx.alloc_pinned::<f32>(1).unwrap() },
            ],
        }
    }

    pub fn init_state(&self) -> LSTMState {
        LSTMState { h: vec![0.0f32; self.hidden_dim], c: vec![0.0f32; self.hidden_dim] }
    }

    fn step_internal(&self, token_id: usize, h_in: &[f32], c_in: &[f32])
        -> (Vec<f32>, Vec<f32>, Vec<f32>)
    {
        let E  = self.embed_dim;
        let H  = self.hidden_dim;
        let fh = 4 * H;
        let stream = Arc::clone(&self.stream);

        let ids   = stream.clone_htod(&[token_id as i32]).unwrap();
        let h_gpu = upload_f16(&stream, h_in);
        let c_gpu = upload_f16(&stream, c_in);

        let mut x     = zeros_f16(&stream, E);
        let mut gates = zeros_f16(&stream, fh);
        let mut h_out = zeros_f16(&stream, H);
        let mut c_out = zeros_f16(&stream, H);

        let e_i = E as i32; let fh_i = fh as i32; let h_i = H as i32; let one_i = 1i32;
        unsafe {
            stream.launch_builder(&self.fns.emb_fwd)
                .arg(&self.embed).arg(&ids).arg(&mut x).arg(&e_i)
                .launch(cfg2d(1, E)).unwrap();
        }
        gemm(&self.blas, false, false, 1, fh, E, hf(1.0), hf(0.0), &x, &self.w_x, &mut gates);
        gemm(&self.blas, false, false, 1, fh, H, hf(1.0), hf(1.0), &h_gpu, &self.w_h, &mut gates);
        unsafe {
            stream.launch_builder(&self.fns.bias)
                .arg(&mut gates).arg(&self.b).arg(&one_i).arg(&fh_i)
                .launch(cfg1d(fh)).unwrap();
            stream.launch_builder(&self.fns.lstm_fwd)
                .arg(&gates).arg(&c_gpu).arg(&mut h_out).arg(&mut c_out).arg(&h_i)
                .launch(cfg2d(1, H)).unwrap();
        }

        let h_cpu  = download_f16(&self.stream, &h_out);
        let logits = self.adaptive_sm.forward(&h_cpu);
        (logits, h_cpu, download_f16(&self.stream, &c_out))
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

    // ──────────────────────────────────────────────────────────────
    //  train_batch
    // ──────────────────────────────────────────────────────────────
    pub fn train_batch(&mut self, sequences: &[Vec<usize>], learning_rate: f64) -> f32 {
        let stream = Arc::clone(&self.stream);
        let li = self.loss_idx;
        let prev_li = 1 - li;
        // Wait for the previous async loss copy to finish, then read it.
        stream.synchronize().unwrap();
        let prev_loss = if self.profile_step.get() > 0 {
            self.host_loss[prev_li].as_slice().unwrap()[0]
        } else {
            0.0
        };

        let valid: Vec<&Vec<usize>> = sequences.iter().filter(|s| s.len() >= 2).collect();
        if valid.is_empty() { return 0.0; }

        let batch   = valid.len();
        let max_len = valid.iter().map(|s| s.len()).max().unwrap();
        let steps   = max_len.saturating_sub(1);
        let E  = self.embed_dim;
        let H  = self.hidden_dim;
        let fh = 4 * H;

        let do_profile = self.profile_step.get() < 5;
        let t0 = if do_profile { Some(Instant::now()) } else { None };

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

        let mut tok_all_cpu = vec![0i32; steps * batch];
        for t in 0..steps {
            for b in 0..batch { tok_all_cpu[t * batch + b] = input_flat[b * max_len + t]; }
        }
        let t_prep = t0.map(|t| t.elapsed().as_secs_f32());

        let hs  = self.asm_head_size + 2;
        let d1  = self.asm_dim1;
        let d2  = self.asm_dim2;
        let ts1 = self.asm_tail1_size;
        let ts2 = self.asm_tail2_size;
        let wg  = 256usize;
        let max_n = [
            self.vocab_size * E, E * fh, H * fh, fh,
            if self.use_simple_softmax { self.vocab_size * H } else { 0 },
            if self.use_simple_softmax { self.vocab_size } else { 0 },
        ].iter().copied().max().unwrap();
        let max_ngroups = (max_n + wg - 1) / wg;

        if self.bufs.as_ref().map_or(true, |b| !b.fits(batch, steps)) {
            self.bufs = Some(BatchBufs::alloc(
                &stream, batch, steps, E, H, fh, hs, d1, d2, ts1, ts2,
                self.vocab_size, max_ngroups,
            ));
        }
        let bk = self.bufs.as_mut().unwrap();

        stream.memset_zeros(&mut bk.d_embed).unwrap();
        stream.memset_zeros(&mut bk.d_w_x).unwrap();
        stream.memset_zeros(&mut bk.d_w_h).unwrap();
        stream.memset_zeros(&mut bk.d_b).unwrap();
        stream.memset_zeros(&mut bk.dg_w_head).unwrap();
        stream.memset_zeros(&mut bk.dg_b_head).unwrap();
        if self.use_simple_softmax {
            stream.memset_zeros(&mut bk.dg_w_simple).unwrap();
            stream.memset_zeros(&mut bk.dg_b_simple).unwrap();
        }
        stream.memset_zeros(&mut bk.dg_w_proj1).unwrap();
        stream.memset_zeros(&mut bk.dg_w_tail1).unwrap();
        stream.memset_zeros(&mut bk.dg_b_tail1).unwrap();
        stream.memset_zeros(&mut bk.dg_w_proj2).unwrap();
        stream.memset_zeros(&mut bk.dg_w_tail2).unwrap();
        stream.memset_zeros(&mut bk.dg_b_tail2).unwrap();
        let li = self.loss_idx;
        let prev_li = 1 - li;
        stream.memset_zeros(&mut bk.loss_part).unwrap();
        stream.memset_zeros(&mut bk.gpu_loss[li]).unwrap();
        stream.memset_zeros(&mut bk.h[0]).unwrap();
        stream.memset_zeros(&mut bk.c[0]).unwrap();
        for t in 0..steps { stream.memset_zeros(&mut bk.d_h[t]).unwrap(); }

        for t in 0..steps {
            let tok_step = &tok_all_cpu[t * batch .. (t+1) * batch];
            stream.memcpy_htod(tok_step, &mut bk.tok_bufs[t]).unwrap();
        }

        let mut head_tgt_cpu  = vec![0i32; batch * steps];
        let mut simple_tgt_cpu = vec![0i32; batch * steps];
        let mut tail1_tgt_cpu = vec![0i32; batch * steps];
        let mut tail2_tgt_cpu = vec![0i32; batch * steps];
        for t in 0..steps {
            for b in 0..batch {
                let tk = if mask_flat[b * max_len + t] < 0.5 { -1i32 } else { target_flat[b * max_len + t] };
                simple_tgt_cpu[t * batch + b] = tk;
                head_tgt_cpu[t * batch + b] = if tk < 0 { -1 }
                    else if (tk as usize) < self.asm_head_size { tk }
                    else if (tk as usize) < self.asm_head_size + ts1 { self.asm_head_size as i32 }
                    else { (self.asm_head_size + 1) as i32 };
                tail1_tgt_cpu[t * batch + b] = if tk < 0 || (tk as usize) < self.asm_head_size
                    || (tk as usize) >= self.asm_head_size + ts1 { -1 }
                    else { tk - self.asm_head_size as i32 };
                tail2_tgt_cpu[t * batch + b] = if tk < 0 || (tk as usize) < self.asm_head_size + ts1 { -1 }
                    else { tk - (self.asm_head_size + ts1) as i32 };
            }
            let off = t * batch;
            stream.memcpy_htod(&head_tgt_cpu[off..off+batch],  &mut bk.head_tgt[t]).unwrap();
            if self.use_simple_softmax {
                stream.memcpy_htod(&simple_tgt_cpu[off..off+batch], &mut bk.simple_tgt[t]).unwrap();
            }
            stream.memcpy_htod(&tail1_tgt_cpu[off..off+batch], &mut bk.tail1_tgt[t]).unwrap();
            stream.memcpy_htod(&tail2_tgt_cpu[off..off+batch], &mut bk.tail2_tgt[t]).unwrap();
        }

        let t_upload = t0.map(|t| t.elapsed().as_secs_f32());
        let t_fwd_start = if do_profile { Some(Instant::now()) } else { None };

        let batch_i = batch as i32;
        let e_i     = E   as i32;
        let fh_i    = fh  as i32;
        let h_i     = H   as i32;
        let hs_i    = hs  as i32;
        let _d1_i    = d1  as i32;
        let _d2_i    = d2  as i32;
        let ts1_i   = ts1 as i32;
        let ts2_i   = ts2 as i32;
        let zero_i  = 0i32;
        let _asm_head_i  = self.asm_head_size as i32;
        let _asm_htail_i = (self.asm_head_size + ts1) as i32;
        let a1 = hf(1.0);
        let a0 = hf(0.0);

        // ── FORWARD ─────────────────────────────────────────────
        for t in 0..steps {
            stream.memset_zeros(&mut bk.gates[t]).unwrap();
            unsafe {
                stream.launch_builder(&self.fns.emb_fwd)
                    .arg(&self.embed).arg(&bk.tok_bufs[t])
                    .arg(&mut bk.xs[t]).arg(&e_i)
                    .launch(cfg2d(batch, E)).unwrap();
            }
            gemm(&self.blas, false, false, batch, fh, E, a1, a0, &bk.xs[t],  &self.w_x, &mut bk.gates[t]);
            gemm(&self.blas, false, false, batch, fh, H, a1, a1, &bk.h[t],   &self.w_h, &mut bk.gates[t]);
            unsafe {
                stream.launch_builder(&self.fns.bias)
                    .arg(&mut bk.gates[t]).arg(&self.b).arg(&batch_i).arg(&fh_i)
                    .launch(cfg1d(batch * fh)).unwrap();
                let (c_prev, c_next) = bk.c.split_at_mut(t + 1);
                stream.launch_builder(&self.fns.lstm_fwd)
                    .arg(&bk.gates[t]).arg(&c_prev[t])
                    .arg(&mut bk.h[t+1]).arg(&mut c_next[0]).arg(&h_i)
                    .launch(cfg2d(batch, H)).unwrap();
            }
        }
        let t_fwd = t_fwd_start.map(|t| t.elapsed().as_secs_f32());
        let t_asm_start = if do_profile { Some(Instant::now()) } else { None };

        // ── ASM fwd + bwd ─────────────────────────────────────────
        let rows_per_block = 32usize;
        let blocks_per_batch = (batch + rows_per_block - 1) / rows_per_block;
        let part_stride = 3 * blocks_per_batch;
        let part_total = part_stride * steps;

        let vocab_i = self.vocab_size as i32;
        for t in 0..steps {
            if self.use_simple_softmax {
                gemm(&self.blas, false, true, batch, self.vocab_size, H, a1, a0, &bk.h[t+1], &self.w_simple, &mut bk.simple_logits[t]);
                unsafe {
                    stream.launch_builder(&self.fns.bias)
                        .arg(&mut bk.simple_logits[t]).arg(&self.b_simple).arg(&batch_i).arg(&vocab_i)
                        .launch(cfg1d(batch * self.vocab_size)).unwrap();
                    stream.launch_builder(&self.fns.asm_sm)
                        .arg(&mut bk.simple_logits[t]).arg(&vocab_i).arg(&batch_i)
                        .launch(cfg_warp_per_row(batch)).unwrap();
                    let loss_off = (t * part_stride) as i32;
                    stream.launch_builder(&self.fns.asm_ce)
                        .arg(&mut bk.simple_logits[t]).arg(&bk.simple_tgt[t]).arg(&mut bk.loss_part)
                        .arg(&vocab_i).arg(&zero_i).arg(&batch_i).arg(&loss_off)
                        .launch(cfg_warp_per_row(batch)).unwrap();
                }
                gemm(&self.blas, true, false, self.vocab_size, H, batch, a1, a1, &bk.simple_logits[t], &bk.h[t+1], &mut bk.dg_w_simple);
                unsafe {
                    stream.launch_builder(&self.fns.red_sum)
                        .arg(&bk.simple_logits[t]).arg(&mut bk.dg_b_simple).arg(&batch_i).arg(&vocab_i)
                        .launch(cfg1d(self.vocab_size)).unwrap();
                }
                let dh = &mut bk.d_h[t];
                gemm(&self.blas, false, false, batch, H, self.vocab_size, a1, a1, &bk.simple_logits[t], &self.w_simple, &mut *dh);
                continue;
            }

            gemm(&self.blas, false, true, batch, hs, H, a1, a0, &bk.h[t+1], &self.g_w_head, &mut bk.head_logits[t]);
            unsafe {
                stream.launch_builder(&self.fns.bias)
                    .arg(&mut bk.head_logits[t]).arg(&self.g_b_head).arg(&batch_i).arg(&hs_i)
                    .launch(cfg1d(batch * hs)).unwrap();
                stream.launch_builder(&self.fns.asm_sm)
                    .arg(&mut bk.head_logits[t]).arg(&hs_i).arg(&batch_i)
                    .launch(cfg_warp_per_row(batch)).unwrap();
            }

            gemm(&self.blas, false, true, batch, d1, H, a1, a0, &bk.h[t+1], &self.g_w_proj1, &mut bk.proj1[t]);
            gemm(&self.blas, false, true, batch, ts1, d1, a1, a0, &bk.proj1[t], &self.g_w_tail1, &mut bk.tail1_log[t]);
            unsafe {
                stream.launch_builder(&self.fns.bias)
                    .arg(&mut bk.tail1_log[t]).arg(&self.g_b_tail1).arg(&batch_i).arg(&ts1_i)
                    .launch(cfg1d(batch * ts1)).unwrap();
                stream.launch_builder(&self.fns.asm_sm)
                    .arg(&mut bk.tail1_log[t]).arg(&ts1_i).arg(&batch_i)
                    .launch(cfg_warp_per_row(batch)).unwrap();
            }

            gemm(&self.blas, false, true, batch, d2, H, a1, a0, &bk.h[t+1], &self.g_w_proj2, &mut bk.proj2[t]);
            gemm(&self.blas, false, true, batch, ts2, d2, a1, a0, &bk.proj2[t], &self.g_w_tail2, &mut bk.tail2_log[t]);
            unsafe {
                stream.launch_builder(&self.fns.bias)
                    .arg(&mut bk.tail2_log[t]).arg(&self.g_b_tail2).arg(&batch_i).arg(&ts2_i)
                    .launch(cfg1d(batch * ts2)).unwrap();
                stream.launch_builder(&self.fns.asm_sm)
                    .arg(&mut bk.tail2_log[t]).arg(&ts2_i).arg(&batch_i)
                    .launch(cfg_warp_per_row(batch)).unwrap();

                let loss_head_off  = (t * part_stride) as i32;
                let loss_tail1_off = loss_head_off + blocks_per_batch as i32;
                let loss_tail2_off = loss_head_off + 2 * blocks_per_batch as i32;
                stream.launch_builder(&self.fns.asm_ce)
                    .arg(&mut bk.head_logits[t]).arg(&bk.head_tgt[t]).arg(&mut bk.loss_part)
                    .arg(&hs_i).arg(&zero_i).arg(&batch_i).arg(&loss_head_off)
                    .launch(cfg_warp_per_row(batch)).unwrap();
                stream.launch_builder(&self.fns.asm_ce)
                    .arg(&mut bk.tail1_log[t]).arg(&bk.tail1_tgt[t]).arg(&mut bk.loss_part)
                    .arg(&ts1_i).arg(&zero_i).arg(&batch_i).arg(&loss_tail1_off)
                    .launch(cfg_warp_per_row(batch)).unwrap();
                stream.launch_builder(&self.fns.asm_ce)
                    .arg(&mut bk.tail2_log[t]).arg(&bk.tail2_tgt[t]).arg(&mut bk.loss_part)
                    .arg(&ts2_i).arg(&zero_i).arg(&batch_i).arg(&loss_tail2_off)
                    .launch(cfg_warp_per_row(batch)).unwrap();
            }

            gemm(&self.blas, true, false, hs, H, batch, a1, a1, &bk.head_logits[t], &bk.h[t+1], &mut bk.dg_w_head);
            gemm(&self.blas, true, false, ts1, d1, batch, a1, a1, &bk.tail1_log[t], &bk.proj1[t], &mut bk.dg_w_tail1);
            gemm(&self.blas, true, false, ts2, d2, batch, a1, a1, &bk.tail2_log[t], &bk.proj2[t], &mut bk.dg_w_tail2);

            unsafe {
                stream.launch_builder(&self.fns.red_sum)
                    .arg(&bk.head_logits[t]).arg(&mut bk.dg_b_head).arg(&batch_i).arg(&hs_i)
                    .launch(cfg1d(hs)).unwrap();
                stream.launch_builder(&self.fns.red_sum)
                    .arg(&bk.tail1_log[t]).arg(&mut bk.dg_b_tail1).arg(&batch_i).arg(&ts1_i)
                    .launch(cfg1d(ts1)).unwrap();
                stream.launch_builder(&self.fns.red_sum)
                    .arg(&bk.tail2_log[t]).arg(&mut bk.dg_b_tail2).arg(&batch_i).arg(&ts2_i)
                    .launch(cfg1d(ts2)).unwrap();
            }

            gemm(&self.blas, false, false, batch, d1, ts1, a1, a0, &bk.tail1_log[t], &self.g_w_tail1, &mut bk.d_proj1[t]);
            gemm(&self.blas, true, false, d1, H, batch, a1, a1, &bk.d_proj1[t], &bk.h[t+1], &mut bk.dg_w_proj1);
            gemm(&self.blas, false, false, batch, d2, ts2, a1, a0, &bk.tail2_log[t], &self.g_w_tail2, &mut bk.d_proj2[t]);
            gemm(&self.blas, true, false, d2, H, batch, a1, a1, &bk.d_proj2[t], &bk.h[t+1], &mut bk.dg_w_proj2);

            let dh = &mut bk.d_h[t];
            gemm(&self.blas, false, false, batch, H, hs, a1, a1, &bk.head_logits[t], &self.g_w_head, &mut *dh);
            gemm(&self.blas, false, false, batch, H, d1, a1, a1, &bk.d_proj1[t], &self.g_w_proj1, &mut *dh);
            gemm(&self.blas, false, false, batch, H, d2, a1, a1, &bk.d_proj2[t], &self.g_w_proj2, &mut *dh);
        }
        unsafe {
            let n_i = part_total as i32 ;
            stream.launch_builder(&self.fns.red_one)
                .arg(&bk.loss_part).arg(&mut bk.gpu_loss[li]).arg(&n_i)
                .launch(cfg_reduce(part_total)).unwrap();
        }
        let t_asm = t_asm_start.map(|t| t.elapsed().as_secs_f32());
        let t_bwd_start = if do_profile { Some(Instant::now()) } else { None };

        // ── BACKWARD THROUGH LSTM ─────────────────────────────────────
        stream.memset_zeros(&mut bk.d_c_next).unwrap();
        for t in (0..steps).rev() {
            stream.memset_zeros(&mut bk.d_gates).unwrap();
            stream.memset_zeros(&mut bk.d_c_prev).unwrap();
            unsafe {
                let (c_prev_sl, c_next_sl) = bk.c.split_at(t + 1);
                stream.launch_builder(&self.fns.lstm_bwd)
                    .arg(&bk.gates[t]).arg(&c_prev_sl[t]).arg(&c_next_sl[0])
                    .arg(&bk.d_h[t]).arg(&bk.d_c_next)
                    .arg(&mut bk.d_gates).arg(&mut bk.d_c_prev).arg(&h_i)
                    .launch(cfg2d(batch, H)).unwrap();
            }
            std::mem::swap(&mut bk.d_c_next, &mut bk.d_c_prev);

            if t > 0 {
                gemm(&self.blas, false, true, batch, H, fh, a1, a1, &bk.d_gates, &self.w_h, &mut bk.d_h[t - 1]);
            }

            gemm(&self.blas, true, false, H, fh, batch, a1, a1, &bk.h[t],    &bk.d_gates, &mut bk.d_w_h);
            gemm(&self.blas, true, false, E, fh, batch, a1, a1, &bk.xs[t],   &bk.d_gates, &mut bk.d_w_x);
            unsafe {
                stream.launch_builder(&self.fns.red_sum)
                    .arg(&bk.d_gates).arg(&mut bk.d_b).arg(&batch_i).arg(&fh_i)
                    .launch(cfg1d(fh)).unwrap();
            }
            stream.memset_zeros(&mut bk.d_x).unwrap();
            gemm(&self.blas, false, true, batch, E, fh, a1, a0, &bk.d_gates, &self.w_x, &mut bk.d_x);
            unsafe {
                stream.launch_builder(&self.fns.emb_bwd)
                    .arg(&bk.d_x).arg(&bk.tok_bufs[t]).arg(&mut bk.d_embed).arg(&e_i)
                    .launch(cfg2d(batch, E)).unwrap();
            }
        }

        // ── OPTIMIZER UPDATE ────────────────────────────────────────
        let t_bwd = t_bwd_start.map(|t| t.elapsed().as_secs_f32());
        let t_adam_start = if do_profile { Some(Instant::now()) } else { None };
        let skip_adam = std::env::var("ARIA_SKIP_ADAM").is_ok();
        let use_sgd   = std::env::var("ARIA_USE_SGD").is_ok();
        let opt_lstm  = std::env::var("ARIA_LSTM_OPT").ok().and_then(|s| s.parse::<i32>().ok()).unwrap_or(1) != 0;
        let opt_asm   = std::env::var("ARIA_ASM_OPT").ok().and_then(|s| s.parse::<i32>().ok()).unwrap_or(1) != 0;
        let no_clip   = std::env::var("ARIA_NO_CLIP").is_ok();
        let t_adam_opt = if skip_adam { None } else { t_adam_start };
        if !skip_adam {
        let b1  = 0.9f32; let b2 = 0.999f32; let eps_adam = 1e-8f32;
        let lr  = learning_rate as f32;
        let clip_val: f32 = std::env::var("ARIA_CLIP")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(1.0f32);

        macro_rules! opt_f16 {
            ($param:expr, $m:expr, $v:expr, $grad:expr, $n:expr, $t:expr, $clip:expr) => {{
                let nn: usize = $n;
                let n_i = nn as i32;
                if $clip {
                    let ngroups = (nn + wg - 1) / wg;
                    stream.memset_zeros(&mut bk.partial).unwrap();
                    unsafe {
                        stream.launch_builder(&self.fns.norm_red_f16)
                            .arg(&$grad).arg(&mut bk.partial).arg(&n_i)
                            .launch(LaunchConfig { grid_dim: (ngroups as u32,1,1), block_dim: (wg as u32,1,1), shared_mem_bytes: (wg*4) as u32 }).unwrap();
                        stream.launch_builder(&self.fns.clip_f16)
                            .arg(&bk.partial).arg(&mut $grad).arg(&(ngroups as i32)).arg(&n_i).arg(&clip_val)
                            .launch(LaunchConfig { grid_dim: (1,1,1), block_dim: (wg as u32,1,1), shared_mem_bytes: (wg*4) as u32 }).unwrap();
                    }
                }
                if use_sgd {
                    unsafe {
                        stream.launch_builder(&self.fns.sgd_f16)
                            .arg(&mut $param).arg(&$grad).arg(&lr).arg(&n_i)
                            .launch(cfg1d(nn)).unwrap();
                    }
                } else {
                    unsafe {
                        stream.launch_builder(&self.fns.adam_f16)
                            .arg(&mut $param).arg(&mut $m).arg(&mut $v).arg(&$grad)
                            .arg(&lr).arg(&b1).arg(&b2).arg(&eps_adam).arg(&$t).arg(&$t).arg(&n_i)
                            .launch(cfg1d(nn)).unwrap();
                    }
                }
            }};
        }

        if opt_lstm {
            self.adam_step += 1;
            let t_adam_i = self.adam_step;
            let bc1 = 1.0 - b1.powi(t_adam_i);
            let _bc2 = 1.0 - b2.powi(t_adam_i);
            opt_f16!(self.embed, self.m_embed, self.v_embed, bk.d_embed, self.vocab_size * E, bc1, !no_clip);
            opt_f16!(self.w_x,   self.m_w_x,  self.v_w_x,   bk.d_w_x,  E * fh, bc1, !no_clip);
            opt_f16!(self.w_h,   self.m_w_h,  self.v_w_h,   bk.d_w_h,  H * fh, bc1, !no_clip);
            opt_f16!(self.b,     self.m_b,    self.v_b,     bk.d_b,    fh, bc1, !no_clip);
        }

        if opt_asm {
            if self.use_simple_softmax {
                self.simple_adam_step += 1;
                let at = self.simple_adam_step;
                let abc1 = 1.0 - 0.9f32.powi(at);
                opt_f16!(self.w_simple, self.m_w_simple, self.v_w_simple, bk.dg_w_simple, self.vocab_size * H, abc1, !no_clip);
                opt_f16!(self.b_simple, self.m_b_simple, self.v_b_simple, bk.dg_b_simple, self.vocab_size, abc1, !no_clip);
            } else {
                self.asm_adam_step += 1;
                let at   = self.asm_adam_step;
                let abc1 = 1.0 - 0.9f32.powi(at);
                opt_f16!(self.g_w_head,  self.gm_w_head,  self.gv_w_head,  bk.dg_w_head,  hs * H, abc1, !no_clip);
                opt_f16!(self.g_b_head,  self.gm_b_head,  self.gv_b_head,  bk.dg_b_head,  hs, abc1, !no_clip);
                opt_f16!(self.g_w_proj1, self.gm_w_proj1, self.gv_w_proj1, bk.dg_w_proj1, d1 * H, abc1, !no_clip);
                opt_f16!(self.g_w_tail1, self.gm_w_tail1, self.gv_w_tail1, bk.dg_w_tail1, ts1 * d1, abc1, !no_clip);
                opt_f16!(self.g_b_tail1, self.gm_b_tail1, self.gv_b_tail1, bk.dg_b_tail1, ts1, abc1, !no_clip);
                opt_f16!(self.g_w_proj2, self.gm_w_proj2, self.gv_w_proj2, bk.dg_w_proj2, d2 * H, abc1, !no_clip);
                opt_f16!(self.g_w_tail2, self.gm_w_tail2, self.gv_w_tail2, bk.dg_w_tail2, ts2 * d2, abc1, !no_clip);
                opt_f16!(self.g_b_tail2, self.gm_b_tail2, self.gv_b_tail2, bk.dg_b_tail2, ts2, abc1, !no_clip);
            }
        }

        } // end if !skip_adam

        let t_adam = t_adam_opt.map(|t| t.elapsed().as_secs_f32());

        // Enqueue async loss copy for the next call to read; no CPU sync here.
        stream.memcpy_dtoh(&bk.gpu_loss[li], &mut self.host_loss[li]).unwrap();
        self.loss_idx = prev_li;

        if do_profile {
            self.profile_step.set(self.profile_step.get() + 1);
            println!(
                "    [profile] prep={:.3}s upload={:.3}s fwd={:.3}s asm={:.3}s bwd={:.3}s adam={:.3}s",
                t_prep.unwrap_or(0.0), t_upload.unwrap_or(0.0), t_fwd.unwrap_or(0.0),
                t_asm.unwrap_or(0.0), t_bwd.unwrap_or(0.0), t_adam.unwrap_or(0.0)
            );
        }

        // Return loss from the previous batch (already synchronized at the top).
        prev_loss / (steps * batch).max(1) as f32
    }

    pub fn backward_step(&mut self, tokens: &[usize], learning_rate: f64) -> f32 {
        self.train_batch(&[tokens.to_vec()], learning_rate)
    }

    // ── Sampling ──────────────────────────────────────────────────────────
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

    // ── Decoding ──────────────────────────────────────────────────────────
    fn apply_repetition_penalty(logits: &mut [f32], history: &[usize], penalty: f32) {
        if penalty <= 1.0 { return; }
        for &id in history {
            if id < logits.len() && logits[id] > 0.0 {
                logits[id] /= penalty;
            } else if id < logits.len() {
                logits[id] *= penalty;
            }
        }
    }

    pub fn decode_greedy(&self, tokenizer: &mut Tokenizer, prompt: &str, max_tokens: usize) -> String {
        let mut tokens = tokenizer.encode(prompt);
        if tokens.len() < 2 { tokens.push(3); }
        let input = &tokens[..tokens.len().saturating_sub(1)];
        let (_, mut state) = self.forward_seq(input);
        let mut generated: Vec<usize> = Vec::new();
        let mut last_logits = vec![0.0f32; self.vocab_size];
        for _ in 0..max_tokens {
            let (logits, new_state) = if generated.is_empty() {
                self.forward_seq(&tokens)
            } else {
                self.step(*generated.last().unwrap(), &state)
            };
            last_logits = logits;
            state = new_state;
            let mut masked = last_logits.clone();
            tokenizer.mask_logits(&mut masked);
            let id = self.sample_greedy(&masked);
            if id == 0 || id == 3 || id >= tokenizer.vocab_size() { break; }
            generated.push(id);
        }
        tokenizer.decode(&generated)
    }

    pub fn decode_top_k(&self, tokenizer: &mut Tokenizer, prompt: &str, max_tokens: usize,
                        k: usize, temperature: f32, repetition_penalty: f32) -> String
    {
        let mut tokens = tokenizer.encode(prompt);
        if tokens.len() < 2 { tokens.push(3); }
        let input = &tokens[..tokens.len().saturating_sub(1)];
        let (_, mut state) = self.forward_seq(input);
        let mut generated: Vec<usize> = Vec::new();
        let mut last_logits = vec![0.0f32; self.vocab_size];
        for _ in 0..max_tokens {
            let (logits, new_state) = if generated.is_empty() {
                self.forward_seq(&tokens)
            } else {
                self.step(*generated.last().unwrap(), &state)
            };
            last_logits = logits;
            state = new_state;
            let mut masked = last_logits.clone();
            tokenizer.mask_logits(&mut masked);
            Self::apply_repetition_penalty(&mut masked, &generated, repetition_penalty);
            let id = self.sample_top_k(&masked, temperature, k);
            if id == 0 || id == 3 || id >= tokenizer.vocab_size() { break; }
            generated.push(id);
        }
        tokenizer.decode(&generated)
    }

    // ── Save / Load (legacy JSON, weights only) ──────────────────────────────
    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let data = serde_json::json!({
            "vocab_size": self.vocab_size,
            "embed_dim":  self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "format":     "v13_cuda_fp16",
            "embed": download_f16(&self.stream, &self.embed),
            "w_x":   download_f16(&self.stream, &self.w_x),
            "w_h":   download_f16(&self.stream, &self.w_h),
            "b":     download_f16(&self.stream, &self.b),
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
                    if v.len() == $n { model.$field = upload_f16(&model.stream, &v); }
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

    // ── Full checkpoint (weights + optimizer + config) ─────────────────────────
    pub fn save_checkpoint(&self, path: &str) -> anyhow::Result<()> {
        use base64::{Engine as _, engine::general_purpose::STANDARD as B64};

        fn f16_b64(stream: &Arc<CudaStream>, buf: &CudaSlice<f16>) -> String {
            stream.synchronize().unwrap();
            let h: Vec<f16> = stream.clone_dtoh(buf).unwrap();
            let bytes: Vec<u8> = h.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
            B64.encode(&bytes)
        }
        fn f32_b64(stream: &Arc<CudaStream>, buf: &CudaSlice<f32>) -> String {
            stream.synchronize().unwrap();
            let h: Vec<f32> = stream.clone_dtoh(buf).unwrap();
            let bytes: Vec<u8> = h.iter().flat_map(|x| x.to_le_bytes()).collect();
            B64.encode(&bytes)
        }

        let (w_simple_b64, b_simple_b64) = if self.use_simple_softmax {
            (f16_b64(&self.stream, &self.w_simple), f16_b64(&self.stream, &self.b_simple))
        } else {
            (String::new(), String::new())
        };
        let (m_w_simple_b64, v_w_simple_b64, m_b_simple_b64, v_b_simple_b64) = if self.use_simple_softmax {
            (f32_b64(&self.stream, &self.m_w_simple), f32_b64(&self.stream, &self.v_w_simple),
             f32_b64(&self.stream, &self.m_b_simple), f32_b64(&self.stream, &self.v_b_simple))
        } else {
            (String::new(), String::new(), String::new(), String::new())
        };

        let data = serde_json::json!({
            "format": "aria_checkpoint_v1",
            "vocab_size": self.vocab_size,
            "embed_dim":  self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "asm_head_size": self.asm_head_size,
            "asm_tail1_size": self.asm_tail1_size,
            "asm_tail2_size": self.asm_tail2_size,
            "asm_dim1": self.asm_dim1,
            "asm_dim2": self.asm_dim2,
            "use_simple_softmax": self.use_simple_softmax,
            "adam_step": self.adam_step,
            "asm_adam_step": self.asm_adam_step,
            "simple_adam_step": self.simple_adam_step,
            "weights": {
                "embed": f16_b64(&self.stream, &self.embed),
                "w_x":   f16_b64(&self.stream, &self.w_x),
                "w_h":   f16_b64(&self.stream, &self.w_h),
                "b":     f16_b64(&self.stream, &self.b),
                "g_w_head":  f16_b64(&self.stream, &self.g_w_head),
                "g_b_head":  f16_b64(&self.stream, &self.g_b_head),
                "g_w_proj1": f16_b64(&self.stream, &self.g_w_proj1),
                "g_w_tail1": f16_b64(&self.stream, &self.g_w_tail1),
                "g_b_tail1": f16_b64(&self.stream, &self.g_b_tail1),
                "g_w_proj2": f16_b64(&self.stream, &self.g_w_proj2),
                "g_w_tail2": f16_b64(&self.stream, &self.g_w_tail2),
                "g_b_tail2": f16_b64(&self.stream, &self.g_b_tail2),
                "w_simple": w_simple_b64,
                "b_simple": b_simple_b64,
            },
            "adam": {
                "m_embed": f32_b64(&self.stream, &self.m_embed), "v_embed": f32_b64(&self.stream, &self.v_embed),
                "m_w_x":   f32_b64(&self.stream, &self.m_w_x),   "v_w_x":   f32_b64(&self.stream, &self.v_w_x),
                "m_w_h":   f32_b64(&self.stream, &self.m_w_h),   "v_w_h":   f32_b64(&self.stream, &self.v_w_h),
                "m_b":     f32_b64(&self.stream, &self.m_b),     "v_b":     f32_b64(&self.stream, &self.v_b),
                "m_w_simple": m_w_simple_b64,
                "v_w_simple": v_w_simple_b64,
                "m_b_simple": m_b_simple_b64,
                "v_b_simple": v_b_simple_b64,
                "gm_w_head":  f32_b64(&self.stream, &self.gm_w_head),  "gv_w_head":  f32_b64(&self.stream, &self.gv_w_head),
                "gm_b_head":  f32_b64(&self.stream, &self.gm_b_head),  "gv_b_head":  f32_b64(&self.stream, &self.gv_b_head),
                "gm_w_proj1": f32_b64(&self.stream, &self.gm_w_proj1), "gv_w_proj1": f32_b64(&self.stream, &self.gv_w_proj1),
                "gm_w_tail1": f32_b64(&self.stream, &self.gm_w_tail1), "gv_w_tail1": f32_b64(&self.stream, &self.gv_w_tail1),
                "gm_b_tail1": f32_b64(&self.stream, &self.gm_b_tail1), "gv_b_tail1": f32_b64(&self.stream, &self.gv_b_tail1),
                "gm_w_proj2": f32_b64(&self.stream, &self.gm_w_proj2), "gv_w_proj2": f32_b64(&self.stream, &self.gv_w_proj2),
                "gm_w_tail2": f32_b64(&self.stream, &self.gm_w_tail2), "gv_w_tail2": f32_b64(&self.stream, &self.gv_w_tail2),
                "gm_b_tail2": f32_b64(&self.stream, &self.gm_b_tail2), "gv_b_tail2": f32_b64(&self.stream, &self.gv_b_tail2),
            },
            "adaptive_sm": self.adaptive_sm.to_json(),
        });
        fs::write(path, serde_json::to_string(&data)?)?;
        Ok(())
    }

    pub fn load_checkpoint(path: &str) -> anyhow::Result<Self> {
        use base64::{Engine as _, engine::general_purpose::STANDARD as B64};

        let data: serde_json::Value = serde_json::from_str(&fs::read_to_string(path)?)?;
        let vocab_size = data["vocab_size"].as_u64().unwrap_or(0) as usize;
        let embed_dim  = data["embed_dim"].as_u64().unwrap_or(0) as usize;
        let hidden_dim = data["hidden_dim"].as_u64().unwrap_or(0) as usize;
        if vocab_size == 0 || embed_dim == 0 || hidden_dim == 0 {
            anyhow::bail!("checkpoint missing dims");
        }

        let mut model = LSTMModelCuda::new(vocab_size, embed_dim, hidden_dim);

        fn decode_f16(s: &str) -> Vec<f32> {
            if s.is_empty() { return vec![]; }
            let bytes = B64.decode(s).unwrap_or_default();
            bytes.chunks_exact(2)
                .map(|b| f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
                .collect()
        }
        fn decode_f32(s: &str) -> Vec<f32> {
            if s.is_empty() { return vec![]; }
            let bytes = B64.decode(s).unwrap_or_default();
            bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
        }
        macro_rules! load16 {
            ($field:ident, $key:expr) => {
                if let Some(s) = data["weights"][$key].as_str() {
                    let v = decode_f16(s);
                    model.$field = upload_f16(&model.stream, &v);
                }
            };
        }
        macro_rules! load32 {
            ($field:ident, $key:expr) => {
                if let Some(s) = data["adam"][$key].as_str() {
                    let v = decode_f32(s);
                    let n = v.len();
                    let mut slice = model.stream.alloc_zeros::<f32>(n).unwrap();
                    model.stream.memcpy_htod(&v, &mut slice).unwrap();
                    model.$field = slice;
                }
            };
        }

        load16!(embed, "embed"); load16!(w_x, "w_x"); load16!(w_h, "w_h"); load16!(b, "b");
        load16!(g_w_head, "g_w_head"); load16!(g_b_head, "g_b_head");
        load16!(g_w_proj1, "g_w_proj1");
        load16!(g_w_tail1, "g_w_tail1"); load16!(g_b_tail1, "g_b_tail1");
        load16!(g_w_proj2, "g_w_proj2");
        load16!(g_w_tail2, "g_w_tail2"); load16!(g_b_tail2, "g_b_tail2");
        if model.use_simple_softmax {
            load16!(w_simple, "w_simple"); load16!(b_simple, "b_simple");
        }

        load32!(m_embed, "m_embed"); load32!(v_embed, "v_embed");
        load32!(m_w_x, "m_w_x");     load32!(v_w_x, "v_w_x");
        load32!(m_w_h, "m_w_h");     load32!(v_w_h, "v_w_h");
        load32!(m_b, "m_b");         load32!(v_b, "v_b");
        if model.use_simple_softmax {
            load32!(m_w_simple, "m_w_simple"); load32!(v_w_simple, "v_w_simple");
            load32!(m_b_simple, "m_b_simple"); load32!(v_b_simple, "v_b_simple");
        }
        load32!(gm_w_head, "gm_w_head");  load32!(gv_w_head, "gv_w_head");
        load32!(gm_b_head, "gm_b_head");  load32!(gv_b_head, "gv_b_head");
        load32!(gm_w_proj1, "gm_w_proj1");load32!(gv_w_proj1, "gv_w_proj1");
        load32!(gm_w_tail1, "gm_w_tail1");load32!(gv_w_tail1, "gv_w_tail1");
        load32!(gm_b_tail1, "gm_b_tail1");load32!(gv_b_tail1, "gv_b_tail1");
        load32!(gm_w_proj2, "gm_w_proj2");load32!(gv_w_proj2, "gv_w_proj2");
        load32!(gm_w_tail2, "gm_w_tail2");load32!(gv_w_tail2, "gv_w_tail2");
        load32!(gm_b_tail2, "gm_b_tail2");load32!(gv_b_tail2, "gv_b_tail2");

        if let Some(asm) = AdaptiveSoftmax::from_json(&data["adaptive_sm"]) {
            model.adaptive_sm = asm;
        }
        if let Some(v) = data["adam_step"].as_i64() { model.adam_step = v as i32; }
        if let Some(v) = data["asm_adam_step"].as_i64() { model.asm_adam_step = v as i32; }
        if let Some(v) = data["simple_adam_step"].as_i64() { model.simple_adam_step = v as i32; }
        Ok(model)
    }
}

// ──────────────────────────────────────────────────────────────
//  Sequence cache helpers
// ──────────────────────────────────────────────────────────────
// Packed cache v2: header count, then each seq = real_len (u32) + tokens[real_len]
fn save_seq_cache_packed(path: &str, seqs: &[Vec<usize>], fixed_len: usize) {
    let file = match fs::File::create(path) { Ok(f) => f, Err(_) => return };
    let mut w = BufWriter::new(file);
    let count = seqs.len().min(u32::MAX as usize) as u32;
    w.write_all(&count.to_le_bytes()).ok();
    for seq in seqs.iter().take(count as usize) {
        let len = seq.len().min(fixed_len).min(u32::MAX as usize) as u32;
        w.write_all(&len.to_le_bytes()).ok();
        for &tok in &seq[..len as usize] { w.write_all(&(tok as u32).to_le_bytes()).ok(); }
    }
    w.flush().ok();
}

fn load_seq_cache_packed_chunked(path: &str, fixed_len: usize, max_seqs: usize) -> Option<Vec<Vec<usize>>> {
    let mut file = fs::File::open(path).ok()?;
    let mut header = [0u8; 4];
    file.read_exact(&mut header).ok()?;
    let file_count = u32::from_le_bytes(header) as usize;
    let count = file_count.min(max_seqs);
    let mut seqs = Vec::with_capacity(count);
    for _ in 0..count {
        let mut len_buf = [0u8; 4];
        file.read_exact(&mut len_buf).ok()?;
        let len = u32::from_le_bytes(len_buf) as usize;
        let len = len.min(fixed_len);
        let mut buf = vec![0u8; len * 4];
        file.read_exact(&mut buf).ok()?;
        let seq: Vec<usize> = (0..len)
            .map(|k| u32::from_le_bytes(buf[k*4..k*4+4].try_into().unwrap()) as usize)
            .collect();
        seqs.push(seq);
    }
    Some(seqs)
}

#[allow(dead_code)]
fn load_seq_cache_packed(path: &str, fixed_len: usize) -> Option<Vec<Vec<usize>>> {
    load_seq_cache_packed_chunked(path, fixed_len, usize::MAX)
}

#[allow(dead_code)]
fn save_seq_cache(path: &str, seqs: &[Vec<usize>]) {
    let mut buf: Vec<u8> = Vec::with_capacity(seqs.len() * 20 * 2);
    let count = seqs.len() as u32;
    buf.extend_from_slice(&count.to_le_bytes());
    for seq in seqs {
        buf.extend_from_slice(&(seq.len() as u16).to_le_bytes());
        for &tok in seq { buf.extend_from_slice(&(tok as u16).to_le_bytes()); }
    }
    if let Ok(mut f) = fs::File::create(path) { f.write_all(&buf).ok(); }
}
#[allow(dead_code)]
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

// ──────────────────────────────────────────────────────────────
//  Pre-training entry point
// ──────────────────────────────────────────────────────────────
const LEARNING_RATE:       f64   = 0.0003;
const MAX_TOKENS_PER_SEQ:  usize = 80;
const MIN_TOKENS_PER_SEQ:  usize = 6;
const PRETRAIN_EPOCHS:     usize = 5;
const PRETRAIN_BATCH_SIZE: usize = 1024;
const MAX_SEQS_PER_EPOCH:  usize = 500_000;

pub fn pretrain_from_files(model: &mut LSTMModelCuda, tokenizer: &mut Tokenizer, data_dir: &str, checkpoint_path: &str, tokenizer_path: &str) -> anyhow::Result<()> {
    let _max_seqs: usize = std::env::var("ARIA_MAX_SEQS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(MAX_SEQS_PER_EPOCH);

    let path = Path::new(data_dir);
    if !path.exists() { println!("Data dir not found."); return Ok(()); }

    let start = Instant::now();

    println!("\n===============================================");
    println!("       ARIA - CUDA FP16 TRAINING              ");
    println!("===============================================");
    println!("LR: {}  Epochs: {}  Batch: {}  SeqLen: {}", LEARNING_RATE, PRETRAIN_EPOCHS, PRETRAIN_BATCH_SIZE, MAX_TOKENS_PER_SEQ);
    println!("===============================================\n");

    let cache_path = format!("{}/sequences_cache_v{}_len{}.bin", data_dir, tokenizer.vocab_size(), MAX_TOKENS_PER_SEQ);
    let _ = fs::remove_file(format!("{}/sequences_cache.bin", data_dir));
    let _ = fs::remove_file(format!("{}/sequences_cache_v{}.bin", data_dir, tokenizer.vocab_size()));

    let mut all_seqs: Vec<Vec<usize>> = if let Some(cached) = load_seq_cache_packed_chunked(&cache_path, MAX_TOKENS_PER_SEQ, MAX_SEQS_PER_EPOCH) {
        println!("Loaded {} sequences from packed cache\n", cached.len());
        cached
    } else {
        use rayon::prelude::*;
        #[derive(serde::Deserialize)]
        struct DialogRecord { text: String }

        let entries: Vec<(std::path::PathBuf, String)> = fs::read_dir(path)?
            .par_bridge()
            .filter_map(|e| {
                let e = e.ok()?;
                let p = e.path();
                let ext = p.extension().and_then(|x| x.to_str())?;
                if ext == "txt" {
                    let c = fs::read_to_string(&p).ok()?;
                    if !c.trim().is_empty() { return Some((p, c)); }
                } else if ext == "jsonl" {
                    let c = fs::read_to_string(&p).ok()?;
                    if !c.trim().is_empty() { return Some((p, c)); }
                }
                None
            })
            .collect();

        println!("Loaded {} files", entries.len());
        if entries.is_empty() { return Ok(()); }

        println!("Streaming tokenized sequences to cache...");
        let cache_file = fs::File::create(&cache_path)?;
        let mut w = BufWriter::new(cache_file);
        w.write_all(&0u32.to_le_bytes())?;

        let mut count: u32 = 0;
        for (p, c) in &entries {
            let ext = p.extension().and_then(|x| x.to_str()).unwrap_or("");
            if ext == "jsonl" {
                for line in c.lines() {
                    let line = line.trim();
                    if line.is_empty() { continue; }
                    let text: String = match serde_json::from_str::<DialogRecord>(line) {
                        Ok(rec) => rec.text,
                        Err(_) => continue,
                    };
                    if text.trim().len() <= 5 { continue; }
                    let mut t = tokenizer.encode(&text);
                    if t.len() < MIN_TOKENS_PER_SEQ { continue; }
                    if t.len() > MAX_TOKENS_PER_SEQ { t.truncate(MAX_TOKENS_PER_SEQ); }
                    let len = t.len().min(u32::MAX as usize) as u32;
                    w.write_all(&len.to_le_bytes())?;
                    for &tok in &t { w.write_all(&(tok as u32).to_le_bytes())?; }
                    count = count.saturating_add(1);
                    if count >= MAX_SEQS_PER_EPOCH as u32 { break; }
                }
            } else {
                for s in c.split(|ch| ch == '.' || ch == '\n' || ch == '!' || ch == '?') {
                    let s = s.trim();
                    if s.len() <= 5 { continue; }
                    let mut t = tokenizer.encode(s);
                    if t.len() < MIN_TOKENS_PER_SEQ { continue; }
                    if t.len() > MAX_TOKENS_PER_SEQ { t.truncate(MAX_TOKENS_PER_SEQ); }
                    let len = t.len().min(u32::MAX as usize) as u32;
                    w.write_all(&len.to_le_bytes())?;
                    for &tok in &t { w.write_all(&(tok as u32).to_le_bytes())?; }
                    count = count.saturating_add(1);
                    if count >= MAX_SEQS_PER_EPOCH as u32 { break; }
                }
            }
            if count >= MAX_SEQS_PER_EPOCH as u32 { break; }
        }
        w.flush()?;
        drop(w);
        let mut cache_file = fs::File::options().write(true).open(&cache_path)?;
        cache_file.seek(SeekFrom::Start(0))?;
        cache_file.write_all(&count.to_le_bytes())?;
        println!("Saved packed cache: {} sequences\n", count);

        load_seq_cache_packed_chunked(&cache_path, MAX_TOKENS_PER_SEQ, MAX_SEQS_PER_EPOCH)
            .unwrap_or_default()
    };

    println!("{} sequences total\n", all_seqs.len());
    if all_seqs.is_empty() { return Ok(()); }

    let use_count = all_seqs.len().min(MAX_SEQS_PER_EPOCH);
    let total_batches = (use_count + PRETRAIN_BATCH_SIZE - 1) / PRETRAIN_BATCH_SIZE;

    // Train / validation split: keep last 5% as eval, never shuffled into train
    let val_count = (use_count / 20).max(1).min(use_count / 10);
    let train_count = use_count - val_count;
    let (train_seqs, val_seqs) = all_seqs.split_at(train_count);
    println!("Using {} train + {} val sequences per epoch\n", train_count, val_count);

    let mut current_lr: f64 = std::env::var("ARIA_LR")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(LEARNING_RATE);
    let fixed_len = MAX_TOKENS_PER_SEQ;
    let train_batches = (train_count + PRETRAIN_BATCH_SIZE - 1) / PRETRAIN_BATCH_SIZE;
    let val_batches   = (val_count + PRETRAIN_BATCH_SIZE - 1) / PRETRAIN_BATCH_SIZE;

    for epoch in 0..PRETRAIN_EPOCHS {
        let ep = Instant::now();
        let mut last_report = Instant::now();
        let mut epoch_train = train_seqs.to_vec();
        epoch_train.shuffle(&mut rand::thread_rng());
        let mut total_loss = 0.0f32;
        let mut batches    = 0usize;
        let mut seqs_done  = 0usize;

        let mut batch_buf: Vec<Vec<usize>> = Vec::with_capacity(PRETRAIN_BATCH_SIZE);
        for chunk in epoch_train.chunks(PRETRAIN_BATCH_SIZE) {
            batch_buf.clear();
            batch_buf.extend(chunk.iter().cloned());
            let loss = model.train_batch(&batch_buf, current_lr);
            if loss.is_finite() { total_loss += loss; batches += 1; }
            if batches <= 10 {
                println!("    [batch {}] loss={:.4} total_loss={:.4} avg={:.4}", batches, loss, total_loss, total_loss / batches.max(1) as f32);
            }
            seqs_done += chunk.len();

            if last_report.elapsed().as_secs_f32() >= 1.0 {
                let avg       = total_loss / batches.max(1) as f32;
                let elapsed   = ep.elapsed().as_secs_f32();
                let seq_s     = seqs_done as f32 / elapsed;
                let remaining = train_batches.saturating_sub(batches);
                let tokens_s  = seq_s * fixed_len as f32;
                println!("  Epoch {}/{}  |  batch {}/{}  ({} remaining)  |  loss={:.4}  |  {:.0} seq/s  ({:.0} tok/s)",
                         epoch+1, PRETRAIN_EPOCHS, batches, train_batches, remaining, avg, seq_s, tokens_s);
                std::io::stdout().flush().ok();
                last_report = Instant::now();
            }
        }

        // Validation pass (no gradient update)
        let mut val_loss = 0.0f32;
        let mut val_batches_done = 0usize;
        for chunk in val_seqs.chunks(PRETRAIN_BATCH_SIZE) {
            batch_buf.clear();
            batch_buf.extend(chunk.iter().cloned());
            let loss = model.train_batch(&batch_buf, 0.0);
            if loss.is_finite() { val_loss += loss; val_batches_done += 1; }
        }
        let val_avg = val_loss / val_batches_done.max(1) as f32;

        let avg   = total_loss / batches.max(1) as f32;
        let et    = ep.elapsed();
        let seq_s = seqs_done as f32 / et.as_secs_f32();
        let tok_s = seq_s * fixed_len as f32;
        println!("Epoch {}/{} done  |  train_loss={:.6}  |  val_loss={:.6}  |  {:.1}s  |  {:.0} seq/s  ({:.0} tok/s)  |  lr={:.6}",
                 epoch+1, PRETRAIN_EPOCHS, avg, val_avg, et.as_secs_f32(), seq_s, tok_s, current_lr);
        if !checkpoint_path.is_empty() {
            println!("  Saving checkpoint to {} ...", checkpoint_path);
            model.save_checkpoint(checkpoint_path).ok();
        }
        if !tokenizer_path.is_empty() {
            println!("  Saving tokenizer to {} ...", tokenizer_path);
            tokenizer.save(tokenizer_path).ok();
        }
        current_lr *= 0.85;
    }

    println!("\nTotal: {:.1}s", start.elapsed().as_secs_f32());
    Ok(())
}
