#![allow(clippy::too_many_arguments, unused_variables, dead_code)]

use std::sync::Arc;
use std::fs;
use std::io::Write;
use std::time::Instant;

use cudarc::driver::{CudaStream, CudaSlice, CudaModule, LaunchConfig, PushKernelArg, CudaGraph};
use cudarc::nvrtc::Ptx;
use cudarc::cublas::{CudaBlas, GemmConfig, Gemm, sys::cublasOperation_t};
use half::f16;
use rand::Rng;
use rand::seq::SliceRandom;

use crate::tokenizer::Tokenizer;
use crate::lora::{LoraConfig, LayerLoraAdapters};
use cudarc::driver::CudaContext;

// ─────────────────────────────────────────────────────────────
//  INT4 Quantization: 4-bit weight compression
// ─────────────────────────────────────────────────────────────

/// Quantize FP16 weights to INT4 with per-row scale factors
fn quantize_f16_to_int4(weights_f16: &[f16]) -> (Vec<u8>, Vec<f32>) {
    let len = weights_f16.len();

    // Per-channel: one scale per row (assume row-major)
    // For simplicity, we'll use single global scale
    let mut max_abs: f32 = 0.0;
    for &w in weights_f16 {
        let abs_val = w.to_f32().abs();
        if abs_val > max_abs {
            max_abs = abs_val;
        }
    }

    // Scale to INT4 range [-8, 7]
    let scale = max_abs / 7.0;
    let inv_scale = if scale > 1e-6 { 1.0 / scale } else { 0.0 };

    // Quantize: 2 weights per byte (4 bits each)
    let mut quantized = vec![0u8; (len + 1) / 2];
    for i in 0..len {
        let val = (weights_f16[i].to_f32() * inv_scale).round() as i8;
        let clamped = val.max(-8).min(7) as u8 & 0x0F;

        if i % 2 == 0 {
            quantized[i / 2] = clamped;
        } else {
            quantized[i / 2] |= clamped << 4;
        }
    }

    (quantized, vec![scale])
}

/// Dequantize INT4 back to FP16
fn dequantize_int4_to_f16(quantized: &[u8], scale: &[f32], len: usize) -> Vec<f16> {
    let s = scale[0];
    let mut result = vec![f16::ZERO; len];

    for i in 0..len {
        let quantized_val = if i % 2 == 0 {
            (quantized[i / 2] & 0x0F) as i8
        } else {
            ((quantized[i / 2] >> 4) & 0x0F) as i8
        };

        // Sign-extend 4-bit to 8-bit
        let signed_val = if quantized_val > 7 {
            quantized_val - 16
        } else {
            quantized_val
        } as i32 as f32;

        result[i] = f16::from_f32(signed_val * s);
    }

    result
}

struct GpuContext {
    ctx:    Arc<cudarc::driver::CudaContext>,
    stream: Arc<CudaStream>,
    blas:   CudaBlas,
}
impl GpuContext {
    fn try_init() -> Option<Self> {
        let ctx    = CudaContext::new(0).ok()?;
        let stream = ctx.new_stream().ok()?;
        let blas = CudaBlas::new(stream.clone()).ok()?;
        // Force FP16 Tensor Cores — prevents cuBLAS from silently falling back to TF32
        unsafe {
            cudarc::cublas::sys::cublasSetMathMode(
                *blas.handle(),
                cudarc::cublas::sys::cublasMath_t::CUBLAS_TENSOR_OP_MATH,
            );
        }
        println!("[GPU] CUDA device 0 — cuBLAS ready (FP16 Tensor Cores forced)");
        Some(GpuContext { ctx, stream, blas })
    }
}

// ─────────────────────────────────────────────────────────────
//  Constants
// ─────────────────────────────────────────────────────────────
const LEARNING_RATE:       f32   = 3e-4;
const MAX_TOKENS_PER_SEQ:  usize = 256;
const MIN_TOKENS_PER_SEQ:  usize = 6;
const PRETRAIN_EPOCHS:     usize = 5;
const PRETRAIN_BATCH_SIZE: usize = 64;
const MAX_SEQS_PER_EPOCH:  usize = 500_000;
const MICRO_BATCH_N:       usize = 1; // sequences processed simultaneously (batch=1 for RTX 4060)

const KERNEL_NAMES: &[&str] = &[
    "embedding_fwd", "embedding_bwd", "add_bias",
    "fused_lstm_fwd", "fused_lstm_bwd",
    "asm_linear", "asm_softmax", "asm_ce_grad",
    "asm_wgrad", "asm_bgrad", "asm_igrad",
    "reduce_sum_batch", "reduce_sum", "adam_update", "adam_update_f16", "sgd_update_f16", "scale_f16", "scale_f32",
    "norm_reduce", "norm_reduce_f16", "clip_if_needed", "clip_if_needed_f16",
    "zero_float",
    "layer_norm_fwd", "layer_norm_bwd",
    "gelu_fwd", "gelu_bwd",
    "causal_softmax_fwd", "attn_softmax_bwd",
    "f16_to_f32", "f32_to_f16",
    // GPU training v2
    "embedding_pos_fwd", "qkv_split_heads", "heads_merge", "heads_split", "qkv_grad_merge",
    "add_inplace", "copy_f16", "zero_f16", "zero_f32",
    "mha_scores", "mha_context", "mha_dv", "mha_dattn", "mha_dq", "mha_dk",
    "softmax_ce_masked", "bias_grad_f16_to_f32", "layer_norm_bwd_v2",
    "adam_update_f16_from_f32", "embedding_bwd_f32", "pos_grad_add_f32",
    "f32_to_f16_2d", "zero_scalar_f32",
    "add_f16_to_f32",
    // LN backward optimized (no atomicAdd contention)
    "layer_norm_bwd_dx", "ln_param_grad", "gelu_bwd_overwrite",
    // micro-batch variants
    "embedding_pos_fwd_nb", "qkv_split_heads_nb", "heads_merge_nb",
    "heads_split_nb", "qkv_grad_merge_nb", "pos_grad_add_f32_nb",
    // Flash Attention 2
    "flash_attn_fwd", "flash_attn_bwd",
];

// ─────────────────────────────────────────────────────────────
//  KV-cache for autoregressive inference
// ─────────────────────────────────────────────────────────────
#[derive(Clone)]
pub struct KVCache {
    // k[layer] = flat Vec<f32> of shape [seq_len, d_model]
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
    pub seq_len: usize,
}

impl KVCache {
    fn new(num_layers: usize) -> Self {
        Self { k: vec![vec![]; num_layers], v: vec![vec![]; num_layers], seq_len: 0 }
    }
}

// ─────────────────────────────────────────────────────────────
//  GPU function handles
// ─────────────────────────────────────────────────────────────
struct TrFns {
    emb_fwd:          cudarc::driver::CudaFunction,
    emb_bwd:          cudarc::driver::CudaFunction,
    add_bias:         cudarc::driver::CudaFunction,
    adam_f16:         cudarc::driver::CudaFunction,
    norm_reduce_f16:  cudarc::driver::CudaFunction,
    norm_reduce_f32:  cudarc::driver::CudaFunction,
    clip_f16:         cudarc::driver::CudaFunction,
    scale_f16:        cudarc::driver::CudaFunction,
    scale_f32:        cudarc::driver::CudaFunction,
    reduce_sum:       cudarc::driver::CudaFunction,
    layer_norm_fwd:   cudarc::driver::CudaFunction,
    layer_norm_bwd:   cudarc::driver::CudaFunction,
    gelu_fwd:         cudarc::driver::CudaFunction,
    gelu_bwd:         cudarc::driver::CudaFunction,
    causal_sfx:       cudarc::driver::CudaFunction,
    attn_sfx_bwd:     cudarc::driver::CudaFunction,
    asm_softmax:      cudarc::driver::CudaFunction,
    asm_ce_grad:      cudarc::driver::CudaFunction,
    f16_to_f32:       cudarc::driver::CudaFunction,
    f32_to_f16:       cudarc::driver::CudaFunction,
    // GPU training v2
    emb_pos_fwd:      cudarc::driver::CudaFunction,
    qkv_split:        cudarc::driver::CudaFunction,
    heads_merge:      cudarc::driver::CudaFunction,
    heads_split:      cudarc::driver::CudaFunction,
    qkv_grad_merge:   cudarc::driver::CudaFunction,
    add_inplace:      cudarc::driver::CudaFunction,
    copy_f16:         cudarc::driver::CudaFunction,
    zero_f16:         cudarc::driver::CudaFunction,
    zero_f32:         cudarc::driver::CudaFunction,
    mha_scores:       cudarc::driver::CudaFunction,
    mha_context:      cudarc::driver::CudaFunction,
    mha_dv:           cudarc::driver::CudaFunction,
    mha_dattn:        cudarc::driver::CudaFunction,
    mha_dq:           cudarc::driver::CudaFunction,
    mha_dk:           cudarc::driver::CudaFunction,
    ce_masked:        cudarc::driver::CudaFunction,
    bias_grad:        cudarc::driver::CudaFunction,
    ln_bwd_v2:        cudarc::driver::CudaFunction,
    adam_f16_f32:     cudarc::driver::CudaFunction,
    emb_bwd_f32:      cudarc::driver::CudaFunction,
    pos_grad_f32:     cudarc::driver::CudaFunction,
    f32_to_f16_2d:    cudarc::driver::CudaFunction,
    zero_scalar_f32:  cudarc::driver::CudaFunction,
    add_f16_to_f32:   cudarc::driver::CudaFunction,
    // LN backward optimized
    ln_bwd_dx:        cudarc::driver::CudaFunction,
    ln_param_grad:    cudarc::driver::CudaFunction,
    gelu_bwd_ow:      cudarc::driver::CudaFunction,
    // micro-batch variants
    emb_pos_fwd_nb:   cudarc::driver::CudaFunction,
    qkv_split_nb:     cudarc::driver::CudaFunction,
    heads_merge_nb:   cudarc::driver::CudaFunction,
    heads_split_nb:   cudarc::driver::CudaFunction,
    qkv_grad_merge_nb: cudarc::driver::CudaFunction,
    pos_grad_f32_nb:  cudarc::driver::CudaFunction,
    // Flash Attention 2
    flash_attn_fwd:   cudarc::driver::CudaFunction,
    flash_attn_bwd:   cudarc::driver::CudaFunction,
}

// ─────────────────────────────────────────────────────────────
//  GPU training v2: gradient buffers per layer
// ─────────────────────────────────────────────────────────────
struct GpuLayerGrad {
    g_w_qkv: CudaSlice<f16>, // [D, 3D] f16 — cuBLAS GEMM accumulation
    g_w_out: CudaSlice<f16>,
    g_w_ff1: CudaSlice<f16>,
    g_w_ff2: CudaSlice<f16>,
    g_b_qkv: CudaSlice<f32>, // [3D] f32 — bias_grad kernel
    g_b_out: CudaSlice<f32>,
    g_b_ff1: CudaSlice<f32>,
    g_b_ff2: CudaSlice<f32>,
    g_ln1_g: CudaSlice<f32>,
    g_ln1_b: CudaSlice<f32>,
    g_ln2_g: CudaSlice<f32>,
    g_ln2_b: CudaSlice<f32>,
}

// ─────────────────────────────────────────────────────────────
//  GPU training v2: per-layer activation cache for backward
// ─────────────────────────────────────────────────────────────
struct GpuLayerActs {
    x_pre:    CudaSlice<f16>, // [max_T, D]
    xn1:      CudaSlice<f16>,
    ln1_mean: CudaSlice<f32>, // [max_T]
    ln1_rstd: CudaSlice<f32>,
    qkv:      CudaSlice<f16>, // [max_T, 3D]
    q:        CudaSlice<f16>, // [H, max_T, dh]
    k:        CudaSlice<f16>,
    v:        CudaSlice<f16>,
    scores:   CudaSlice<f16>, // [H, max_T, max_T] — after causal softmax
    ctx:      CudaSlice<f16>, // [H, max_T, dh]
    attn_out: CudaSlice<f16>, // [max_T, D]
    x_mid:    CudaSlice<f16>,
    xn2:      CudaSlice<f16>,
    ln2_mean: CudaSlice<f32>,
    ln2_rstd: CudaSlice<f32>,
    ff1:      CudaSlice<f16>, // [max_T, ff]
    ff1_act:  CudaSlice<f16>,
}

// ─────────────────────────────────────────────────────────────
//  Per-layer weights + Adam moments (all FP16/FP32 on GPU)
// ─────────────────────────────────────────────────────────────
struct TransformerLayer {
    // Attention
    w_qkv: CudaSlice<f16>,  // [d_model, 3*d_model]
    b_qkv: CudaSlice<f16>,  // [3*d_model]
    w_out: CudaSlice<f16>,  // [d_model, d_model]
    b_out: CudaSlice<f16>,  // [d_model]
    // FFN
    w_ff1: CudaSlice<f16>,  // [d_model, ffn_dim]
    b_ff1: CudaSlice<f16>,  // [ffn_dim]
    w_ff2: CudaSlice<f16>,  // [ffn_dim, d_model]
    b_ff2: CudaSlice<f16>,  // [d_model]
    // LayerNorm 1 & 2
    ln1_g: CudaSlice<f16>, ln1_b: CudaSlice<f16>,
    ln2_g: CudaSlice<f16>, ln2_b: CudaSlice<f16>,
    // Adam moments (FP32)
    m_w_qkv: CudaSlice<f32>, v_w_qkv: CudaSlice<f32>,
    m_b_qkv: CudaSlice<f32>, v_b_qkv: CudaSlice<f32>,
    m_w_out:  CudaSlice<f32>, v_w_out:  CudaSlice<f32>,
    m_b_out:  CudaSlice<f32>, v_b_out:  CudaSlice<f32>,
    m_w_ff1:  CudaSlice<f32>, v_w_ff1:  CudaSlice<f32>,
    m_b_ff1:  CudaSlice<f32>, v_b_ff1:  CudaSlice<f32>,
    m_w_ff2:  CudaSlice<f32>, v_w_ff2:  CudaSlice<f32>,
    m_b_ff2:  CudaSlice<f32>, v_b_ff2:  CudaSlice<f32>,
    m_ln1_g:  CudaSlice<f32>, v_ln1_g:  CudaSlice<f32>,
    m_ln1_b:  CudaSlice<f32>, v_ln1_b:  CudaSlice<f32>,
    m_ln2_g:  CudaSlice<f32>, v_ln2_g:  CudaSlice<f32>,
    m_ln2_b:  CudaSlice<f32>, v_ln2_b:  CudaSlice<f32>,
    // LoRA adapters (optional, only if training with LoRA)
    lora: Option<Box<LayerLoraAdapters>>,
}

// ─────────────────────────────────────────────────────────────
//  Main model struct
// ─────────────────────────────────────────────────────────────
pub struct TransformerModel {
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    blas:   CudaBlas,
    fns:    TrFns,

    // Token + positional embeddings (weight-tied with output)
    embed:     CudaSlice<f16>,  // [vocab, d_model]
    pos_embed: CudaSlice<f16>,  // [max_seq_len, d_model]
    m_embed:   CudaSlice<f32>, v_embed: CudaSlice<f32>,
    m_pos:     CudaSlice<f32>, v_pos:   CudaSlice<f32>,

    layers: Vec<TransformerLayer>,

    // Final LayerNorm
    ln_f_g: CudaSlice<f16>, ln_f_b: CudaSlice<f16>,
    m_ln_f_g: CudaSlice<f32>, v_ln_f_g: CudaSlice<f32>,
    m_ln_f_b: CudaSlice<f32>, v_ln_f_b: CudaSlice<f32>,

    // Dimensions
    pub vocab_size:  usize,
    pub d_model:     usize,
    pub num_heads:   usize,
    pub num_layers:  usize,
    pub ffn_dim:     usize,
    pub max_seq_len: usize,
    head_dim: usize,

    // LoRA configuration
    pub lora_config: Option<LoraConfig>,

    // v3.5.1: Quantization, Gradient Checkpointing, LoRA Backward
    pub int4_quantized: bool,              // Enable INT4 quantization
    pub gradient_checkpointing: bool,       // Enable gradient checkpointing to save activations
    pub lora_backward_enabled: bool,        // Enable LoRA backward pass (adapter gradients)

    adam_step: i32,

    // GPU training v2 buffers (allocated in new())
    grads:      Vec<GpuLayerGrad>,
    acts:       Vec<GpuLayerActs>,
    g_embed:    CudaSlice<f32>,  // [vocab, D]
    g_pos:      CudaSlice<f32>,  // [max_seq_len, D]
    g_embed_head_f16: CudaSlice<f16>, // [vocab, D] — tied output-head grad (f16 GEMM target)
    g_ln_f_g:   CudaSlice<f32>,
    g_ln_f_b:   CudaSlice<f32>,
    // Working buffers
    x_buf:      CudaSlice<f16>,  // [max_T, D]
    x_norm_buf: CudaSlice<f16>,  // [max_T, D]
    lnf_mean:   CudaSlice<f32>,  // [max_T]
    lnf_rstd:   CudaSlice<f32>,
    logits_buf: CudaSlice<f16>,  // [max_T, vocab]
    d_logits:   CudaSlice<f16>,  // [max_T, vocab]
    loss_acc:   CudaSlice<f32>,  // [1]
    grad_norm_sq:    CudaSlice<f32>,  // [1] — accumulated ||g||^2 across all params
    partial_norm_buf: CudaSlice<f32>, // scratch for norm_reduce per-block partials
    ids_buf:    CudaSlice<i32>,  // [MICRO_BATCH_N * max_seq_len]
    tgt_buf:    CudaSlice<i32>,  // [MICRO_BATCH_N * max_seq_len]
    msk_buf:    CudaSlice<f32>,  // [MICRO_BATCH_N * max_seq_len]
    dx_buf:     CudaSlice<f16>,  // [max_T, D] backward pass
    tmp_buf:    CudaSlice<f16>,  // [max_T, max(3D,ff)] temp
    lora_tmp:   CudaSlice<f16>,  // [max_T, rank] LoRA intermediate (A(x))
    dq_buf:     CudaSlice<f16>,  // [NH, max_T, dh]
    dk_buf:     CudaSlice<f16>,  // [NH, max_T, dh] — kept for old path
    dv_buf:     CudaSlice<f16>,  // [NH, max_T, dh] — kept for old path
    dk_f32_buf: CudaSlice<f32>,  // [NH, max_T, dh] — flash attn backward accumulate
    dv_f32_buf: CudaSlice<f32>,  // [NH, max_T, dh] — flash attn backward accumulate
    lse_buf:    CudaSlice<f32>,  // [NH, max_T]      — log-sum-exp from flash fwd
    d_attn_buf: CudaSlice<f16>,  // [H, max_T, max_T] — kept for old path (can remove later)
    d_ctx_buf:  CudaSlice<f16>,  // [H, max_T, dh]

    cuda_graph: Option<CudaGraph>,  // captured fwd+bwd graph for full micro-batches
}

// ─────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────
fn randn_f16(n: usize, scale: f32) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen::<f32>() * 2.0 * scale - scale).collect()
}

fn zeros_f32_v(n: usize) -> Vec<f32> { vec![0.0f32; n] }
fn ones_f32_v(n: usize)  -> Vec<f32> { vec![1.0f32; n] }

fn to_f16(v: &[f32]) -> Vec<f16> { v.iter().map(|&x| f16::from_f32(x)).collect() }
fn from_f16(v: &[f16]) -> Vec<f32> { v.iter().map(|x| x.to_f32()).collect() }

fn upload_f16(stream: &Arc<CudaStream>, data: &[f32]) -> CudaSlice<f16> {
    let h = to_f16(data);
    stream.clone_htod(&h).unwrap()
}
fn upload_f32(stream: &Arc<CudaStream>, data: &[f32]) -> CudaSlice<f32> {
    stream.clone_htod(data).unwrap()
}
fn download_f16(stream: &Arc<CudaStream>, buf: &CudaSlice<f16>) -> Vec<f32> {
    stream.synchronize().unwrap();
    let h: Vec<f16> = stream.clone_dtoh(buf).unwrap();
    from_f16(&h)
}
fn download_f32(stream: &Arc<CudaStream>, buf: &CudaSlice<f32>) -> Vec<f32> {
    stream.synchronize().unwrap();
    stream.clone_dtoh(buf).unwrap()
}
fn b64_f16(stream: &Arc<CudaStream>, buf: &CudaSlice<f16>) -> String {
    stream.synchronize().unwrap();
    let v: Vec<f16> = stream.clone_dtoh(buf).unwrap();
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    base64::encode(&bytes)
}
fn b64_f32(stream: &Arc<CudaStream>, buf: &CudaSlice<f32>) -> String {
    let v = download_f32(stream, buf);
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    base64::encode(&bytes)
}

fn from_b64_f16(s: &str) -> Vec<f32> {
    let bytes = base64::decode(s).unwrap_or_default();
    bytes.chunks_exact(2).map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect()
}
fn from_b64_f32(s: &str) -> Vec<f32> {
    let bytes = base64::decode(s).unwrap_or_default();
    bytes.chunks_exact(4).map(|c| f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]]))).collect()
}

fn cfg1d(n: usize) -> LaunchConfig {
    let t = 256usize;
    LaunchConfig { grid_dim: (((n + t - 1) / t) as u32, 1, 1), block_dim: (t as u32, 1, 1), shared_mem_bytes: 0 }
}
fn cfg_warp(rows: usize) -> LaunchConfig {
    let rp = 32usize;
    LaunchConfig {
        grid_dim: (1, ((rows + rp - 1) / rp) as u32, 1),
        block_dim: (32, rp as u32, 1),
        shared_mem_bytes: (4 * rp) as u32,
    }
}
fn cfg_ln(rows: usize) -> LaunchConfig {
    // One block per row, 32 threads (one warp)
    LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    }
}
fn cfg_attn_sfx(bh: usize, t: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (bh as u32, t as u32, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    }
}
fn cfg_reduce(n: usize) -> LaunchConfig {
    let t = 256usize;
    LaunchConfig {
        grid_dim: (((n + t - 1) / t) as u32, 1, 1),
        block_dim: (t as u32, 1, 1),
        shared_mem_bytes: (t * 4) as u32,
    }
}

// cuBLAS GEMM helper (column-major, same convention as model_cuda.rs)
// C[m,n] = alpha * A[m,k] @ B[k,n] + beta * C[m,n]
fn gemm(blas: &CudaBlas,
        a: &CudaSlice<f16>, b: &CudaSlice<f16>, c: &mut CudaSlice<f16>,
        m: usize, k: usize, n: usize,
        transa: bool, transb: bool,
        alpha: f16, beta: f16) {
    use cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T};
    if !transa && !transb {
        unsafe { blas.gemm(GemmConfig::<f16> {
            transa: CUBLAS_OP_N, transb: CUBLAS_OP_N,
            m: n as i32, n: m as i32, k: k as i32,
            alpha, lda: n as i32, ldb: k as i32, beta, ldc: n as i32,
        }, b, a, c).unwrap(); }
    } else if transa && !transb {
        unsafe { blas.gemm(GemmConfig::<f16> {
            transa: CUBLAS_OP_N, transb: CUBLAS_OP_T,
            m: n as i32, n: m as i32, k: k as i32,
            alpha, lda: n as i32, ldb: m as i32, beta, ldc: n as i32,
        }, b, a, c).unwrap(); }
    } else {
        unsafe { blas.gemm(GemmConfig::<f16> {
            transa: CUBLAS_OP_T, transb: CUBLAS_OP_N,
            m: n as i32, n: m as i32, k: k as i32,
            alpha, lda: k as i32, ldb: k as i32, beta, ldc: n as i32,
        }, b, a, c).unwrap(); }
    }
}

// f32 GEMM helper — same row-major convention as gemm()
fn gemm_f32(blas: &CudaBlas,
            a: &CudaSlice<f32>, b: &CudaSlice<f32>, c: &mut CudaSlice<f32>,
            m: usize, k: usize, n: usize,
            transa: bool, transb: bool,
            alpha: f32, beta: f32) {
    use cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T};
    if !transa && !transb {
        unsafe { blas.gemm(GemmConfig::<f32> {
            transa: CUBLAS_OP_N, transb: CUBLAS_OP_N,
            m: n as i32, n: m as i32, k: k as i32,
            alpha, lda: n as i32, ldb: k as i32, beta, ldc: n as i32,
        }, b, a, c).unwrap(); }
    } else if transa && !transb {
        unsafe { blas.gemm(GemmConfig::<f32> {
            transa: CUBLAS_OP_N, transb: CUBLAS_OP_T,
            m: n as i32, n: m as i32, k: k as i32,
            alpha, lda: n as i32, ldb: m as i32, beta, ldc: n as i32,
        }, b, a, c).unwrap(); }
    } else if !transa && transb {
        unsafe { blas.gemm(GemmConfig::<f32> {
            transa: CUBLAS_OP_T, transb: CUBLAS_OP_N,
            m: n as i32, n: m as i32, k: k as i32,
            alpha, lda: k as i32, ldb: k as i32, beta, ldc: n as i32,
        }, b, a, c).unwrap(); }
    } else {
        unsafe { blas.gemm(GemmConfig::<f32> {
            transa: CUBLAS_OP_T, transb: CUBLAS_OP_T,
            m: n as i32, n: m as i32, k: k as i32,
            alpha, lda: k as i32, ldb: m as i32, beta, ldc: n as i32,
        }, b, a, c).unwrap(); }
    }
}

// cuBLAS strided-batched GEMM (row-major convention, same as gemm())
// Computes C[b,m,n] = A[b,m,k] @ B[b,k,n]  (transb=false)
//              or    = A[b,m,k] @ B[b,n,k]^T (transb=true)
fn gemm_batched_f16(blas: &CudaBlas,
                    a: &CudaSlice<f16>, b: &CudaSlice<f16>, c: &mut CudaSlice<f16>,
                    batch: usize, m: usize, k: usize, n: usize,
                    transb: bool,
                    alpha: f16, beta: f16) {
    use cudarc::cublas::sys::cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T};
    use cudarc::cublas::{StridedBatchedConfig, GemmConfig, Gemm};
    // Row-major A[m,k] @ B[k,n] ≡ col-major (B^T)[n,k] @ (A^T)[k,m] → C^T[n,m]
    // Swap roles: cuBLAS "a" = our B, cuBLAS "b" = our A; swap m↔n
    let (transa_op, lda, stride_a) = if transb {
        // our B is [n,k] in row-major → col-major it's [k,n] → CUBLAS_OP_T; lda = k (rows in col-major)
        (CUBLAS_OP_T, k as i32, (n * k) as i64)
    } else {
        // our B is [k,n] in row-major → col-major it's [n,k], no transpose; lda = n
        (CUBLAS_OP_N, n as i32, (k * n) as i64)
    };
    let cfg = StridedBatchedConfig::<f16> {
        gemm: GemmConfig {
            transa: transa_op,
            transb: CUBLAS_OP_N,
            m: n as i32, n: m as i32, k: k as i32,
            alpha,
            lda,
            ldb: k as i32,
            beta,
            ldc: n as i32,
        },
        stride_a,
        stride_b: (m * k) as i64,
        stride_c: (m * n) as i64,
        batch_size: batch as i32,
    };
    unsafe { blas.gemm_strided_batched(cfg, b, a, c).unwrap(); }
}

// ─────────────────────────────────────────────────────────────
//  Constructor helpers
// ─────────────────────────────────────────────────────────────
fn make_layer(stream: &Arc<CudaStream>, d: usize, _h: usize, ff: usize) -> TransformerLayer {
    let s_qkv = (1.0 / (d as f64).sqrt()) as f32;
    let s_out = (1.0 / (d as f64).sqrt()) as f32;
    let s_ff1 = (1.0 / (d as f64).sqrt()) as f32;
    let s_ff2 = (1.0 / (ff as f64).sqrt()) as f32;

    macro_rules! up16 { ($data:expr) => { upload_f16(stream, &$data) } }
    macro_rules! up32 { ($data:expr) => { upload_f32(stream, &$data) } }
    macro_rules! z32 { ($n:expr) => { up32!(zeros_f32_v($n)) } }

    TransformerLayer {
        w_qkv: up16!(randn_f16(d * 3 * d, s_qkv)),
        b_qkv: up16!(zeros_f32_v(3 * d)),
        w_out: up16!(randn_f16(d * d, s_out)),
        b_out: up16!(zeros_f32_v(d)),
        w_ff1: up16!(randn_f16(d * ff, s_ff1)),
        b_ff1: up16!(zeros_f32_v(ff)),
        w_ff2: up16!(randn_f16(ff * d, s_ff2)),
        b_ff2: up16!(zeros_f32_v(d)),
        ln1_g: up16!(ones_f32_v(d)),  ln1_b: up16!(zeros_f32_v(d)),
        ln2_g: up16!(ones_f32_v(d)),  ln2_b: up16!(zeros_f32_v(d)),
        m_w_qkv: z32!(d * 3 * d), v_w_qkv: z32!(d * 3 * d),
        m_b_qkv: z32!(3 * d),     v_b_qkv: z32!(3 * d),
        m_w_out:  z32!(d * d),    v_w_out:  z32!(d * d),
        m_b_out:  z32!(d),        v_b_out:  z32!(d),
        m_w_ff1:  z32!(d * ff),   v_w_ff1:  z32!(d * ff),
        m_b_ff1:  z32!(ff),       v_b_ff1:  z32!(ff),
        m_w_ff2:  z32!(ff * d),   v_w_ff2:  z32!(ff * d),
        m_b_ff2:  z32!(d),        v_b_ff2:  z32!(d),
        m_ln1_g:  z32!(d),        v_ln1_g:  z32!(d),
        m_ln1_b:  z32!(d),        v_ln1_b:  z32!(d),
        m_ln2_g:  z32!(d),        v_ln2_g:  z32!(d),
        m_ln2_b:  z32!(d),        v_ln2_b:  z32!(d),
        lora: None,
    }
}

// ─────────────────────────────────────────────────────────────
//  v3.5.1: Helper functions for quantization, gradient checkpointing, LoRA backward
// ─────────────────────────────────────────────────────────────

/// Compute LoRA gradient: d_A = d_temp^T @ x,  d_B = d_adapter^T @ temp
/// where temp = x @ A^T and adapter = temp @ B^T
fn compute_lora_gradients(
    d_adapter: &[f16],      // [seq, out_dim] gradient w.r.t adapter output
    x: &[f16],              // [seq, in_dim] input
    temp: &[f16],           // [seq, rank] intermediate A(x)
    rank: usize,
    seq_len: usize,
    in_dim: usize,
    out_dim: usize,
) -> (Vec<f16>, Vec<f16>) {
    let mut d_a = vec![f16::ZERO; rank * in_dim];
    let mut d_b = vec![f16::ZERO; out_dim * rank];

    // d_A = x^T @ d_temp  [in_dim, seq] @ [seq, rank] → [in_dim, rank]
    for i in 0..in_dim {
        for r in 0..rank {
            let mut sum = 0.0f32;
            for t in 0..seq_len {
                sum += x[t * in_dim + i].to_f32() * d_adapter[t * rank + r].to_f32();
            }
            d_a[r * in_dim + i] = f16::from_f32(sum);
        }
    }

    // d_B = d_adapter^T @ temp  [out_dim, seq] @ [seq, rank] → [out_dim, rank]
    for o in 0..out_dim {
        for r in 0..rank {
            let mut sum = 0.0f32;
            for t in 0..seq_len {
                sum += d_adapter[t * out_dim + o].to_f32() * temp[t * rank + r].to_f32();
            }
            d_b[r * out_dim + o] = f16::from_f32(sum);
        }
    }

    (d_a, d_b)
}

// ─────────────────────────────────────────────────────────────
//  impl TransformerModel
// ─────────────────────────────────────────────────────────────
impl TransformerModel {
    pub fn new(vocab_size: usize, d_model: usize, num_heads: usize,
               num_layers: usize, ffn_dim: usize, max_seq_len: usize) -> Self {
        let gpu    = GpuContext::try_init().expect("No CUDA GPU found");
        let stream = gpu.stream;
        let blas   = gpu.blas;
        let ctx    = gpu.ctx;

        let out_dir  = env!("OUT_DIR");
        let ptx_path = std::path::PathBuf::from(out_dir).join("kernels.ptx");
        let ptx    = Ptx::from_file(&ptx_path);
        let module = ctx.load_module(ptx)
            .unwrap_or_else(|_| panic!("Failed to load kernels.ptx from {:?}", ptx_path));

        macro_rules! fn_ { ($name:expr) => { module.load_function($name).unwrap() } }
        let fns = TrFns {
            emb_fwd:         fn_!("embedding_fwd"),
            emb_bwd:         fn_!("embedding_bwd"),
            add_bias:        fn_!("add_bias"),
            adam_f16:        fn_!("adam_update_f16"),
            norm_reduce_f16: fn_!("norm_reduce_f16"),
            norm_reduce_f32: fn_!("norm_reduce"),
            clip_f16:        fn_!("clip_if_needed_f16"),
            scale_f16:       fn_!("scale_f16"),
            scale_f32:       fn_!("scale_f32"),
            reduce_sum:      fn_!("reduce_sum"),
            layer_norm_fwd:  fn_!("layer_norm_fwd"),
            layer_norm_bwd:  fn_!("layer_norm_bwd"),
            gelu_fwd:        fn_!("gelu_fwd"),
            gelu_bwd:        fn_!("gelu_bwd"),
            causal_sfx:      fn_!("causal_softmax_fwd"),
            attn_sfx_bwd:    fn_!("attn_softmax_bwd"),
            asm_softmax:     fn_!("asm_softmax"),
            asm_ce_grad:     fn_!("asm_ce_grad"),
            f16_to_f32:      fn_!("f16_to_f32"),
            f32_to_f16:      fn_!("f32_to_f16"),
            emb_pos_fwd:     fn_!("embedding_pos_fwd"),
            qkv_split:       fn_!("qkv_split_heads"),
            heads_merge:     fn_!("heads_merge"),
            heads_split:     fn_!("heads_split"),
            qkv_grad_merge:  fn_!("qkv_grad_merge"),
            add_inplace:     fn_!("add_inplace"),
            copy_f16:        fn_!("copy_f16"),
            zero_f16:        fn_!("zero_f16"),
            zero_f32:        fn_!("zero_f32"),
            mha_scores:      fn_!("mha_scores"),
            mha_context:     fn_!("mha_context"),
            mha_dv:          fn_!("mha_dv"),
            mha_dattn:       fn_!("mha_dattn"),
            mha_dq:          fn_!("mha_dq"),
            mha_dk:          fn_!("mha_dk"),
            ce_masked:       fn_!("softmax_ce_masked"),
            bias_grad:       fn_!("bias_grad_f16_to_f32"),
            ln_bwd_v2:       fn_!("layer_norm_bwd_v2"),
            adam_f16_f32:    fn_!("adam_update_f16_from_f32"),
            emb_bwd_f32:     fn_!("embedding_bwd_f32"),
            pos_grad_f32:    fn_!("pos_grad_add_f32"),
            f32_to_f16_2d:   fn_!("f32_to_f16_2d"),
            zero_scalar_f32: fn_!("zero_scalar_f32"),
            add_f16_to_f32:  fn_!("add_f16_to_f32"),
            ln_bwd_dx:       fn_!("layer_norm_bwd_dx"),
            ln_param_grad:   fn_!("ln_param_grad"),
            gelu_bwd_ow:     fn_!("gelu_bwd_overwrite"),
            emb_pos_fwd_nb:  fn_!("embedding_pos_fwd_nb"),
            qkv_split_nb:    fn_!("qkv_split_heads_nb"),
            heads_merge_nb:  fn_!("heads_merge_nb"),
            heads_split_nb:  fn_!("heads_split_nb"),
            qkv_grad_merge_nb: fn_!("qkv_grad_merge_nb"),
            pos_grad_f32_nb: fn_!("pos_grad_add_f32_nb"),
            flash_attn_fwd:  fn_!("flash_attn_fwd"),
            flash_attn_bwd:  fn_!("flash_attn_bwd"),
        };

        let head_dim = d_model / num_heads;
        let se = (2.0 / d_model as f64).sqrt() as f32;
        let sp = (1.0 / d_model as f64).sqrt() as f32;

        // Positional embedding: sinusoidal init
        let mut pos_data = vec![f16::ZERO; max_seq_len * d_model];
        for pos in 0..max_seq_len {
            for i in 0..d_model {
                let angle = pos as f32 / (10000f32).powf(2.0 * (i / 2) as f32 / d_model as f32);
                pos_data[pos * d_model + i] = f16::from_f32(if i % 2 == 0 { angle.sin() } else { angle.cos() });
            }
        }

        macro_rules! up16 { ($d:expr) => { upload_f16(&stream, &$d) } }
        macro_rules! up32 { ($d:expr) => { upload_f32(&stream, &$d) } }
        macro_rules! z32 { ($n:expr) => { up32!(zeros_f32_v($n)) } }

        let embed     = up16!(randn_f16(vocab_size * d_model, se));
        let pos_embed = up16!(from_f16(&pos_data));

        let layers: Vec<TransformerLayer> = (0..num_layers)
            .map(|_| make_layer(&stream, d_model, num_heads, ffn_dim))
            .collect();

        // param count
        let p_embed = vocab_size * d_model;
        let p_pos   = max_seq_len * d_model;
        let p_layer = 3*d_model*d_model + 3*d_model   // qkv
                    + d_model*d_model + d_model          // out
                    + d_model*ffn_dim + ffn_dim           // ff1
                    + ffn_dim*d_model + d_model           // ff2
                    + 4 * d_model;                        // ln1 + ln2
        let total = p_embed + p_pos + num_layers * p_layer + 2 * d_model;

        println!("================================");
        println!("       ARIA  Transformer        ");
        println!("================================");
        println!("  Vocab:   {}", vocab_size);
        println!("  d_model: {}", d_model);
        println!("  Heads:   {}", num_heads);
        println!("  Layers:  {}", num_layers);
        println!("  FFN:     {}", ffn_dim);
        println!("  SeqLen:  {}", max_seq_len);
        println!("  Params:  ~{:.1}M", total as f32 / 1e6);
        println!("================================");

        let m_embed  = z32!(vocab_size * d_model);
        let v_embed  = z32!(vocab_size * d_model);
        let m_pos    = z32!(max_seq_len * d_model);
        let v_pos    = z32!(max_seq_len * d_model);
        let ln_f_g   = up16!(ones_f32_v(d_model));
        let ln_f_b   = up16!(zeros_f32_v(d_model));
        let m_ln_f_g = z32!(d_model); let v_ln_f_g = z32!(d_model);
        let m_ln_f_b = z32!(d_model); let v_ln_f_b = z32!(d_model);

        // ── GPU training v2 buffers ──────────────────────────────
        let mt = max_seq_len;  // Use full max_seq_len for buffer allocation
        let mbn = MICRO_BATCH_N;
        let mbn = MICRO_BATCH_N;
        macro_rules! z16  { ($n:expr) => { upload_f16(&stream, &zeros_f32_v($n)) } }
        macro_rules! zf32 { ($n:expr) => { upload_f32(&stream, &zeros_f32_v($n)) } }
        macro_rules! zi32 { ($n:expr) => { stream.clone_htod(&vec![0i32; $n]).unwrap() } }

        let grads: Vec<GpuLayerGrad> = (0..num_layers).map(|_| GpuLayerGrad {
            g_w_qkv: z16!(d_model * 3 * d_model),
            g_w_out: z16!(d_model * d_model),
            g_w_ff1: z16!(d_model * ffn_dim),
            g_w_ff2: z16!(ffn_dim * d_model),
            g_b_qkv: zf32!(3 * d_model),
            g_b_out: zf32!(d_model),
            g_b_ff1: zf32!(ffn_dim),
            g_b_ff2: zf32!(d_model),
            g_ln1_g: zf32!(d_model),
            g_ln1_b: zf32!(d_model),
            g_ln2_g: zf32!(d_model),
            g_ln2_b: zf32!(d_model),
        }).collect();

        // Activation buffers sized for micro-batch: N sequences in parallel
        let acts: Vec<GpuLayerActs> = (0..num_layers).map(|_| GpuLayerActs {
            x_pre:    z16!(mbn * mt * d_model),
            xn1:      z16!(mbn * mt * d_model),
            ln1_mean: zf32!(mbn * mt),
            ln1_rstd: zf32!(mbn * mt),
            qkv:      z16!(mbn * mt * 3 * d_model),
            q:        z16!(mbn * num_heads * mt * head_dim),
            k:        z16!(mbn * num_heads * mt * head_dim),
            v:        z16!(mbn * num_heads * mt * head_dim),
            scores:   z16!(mbn * num_heads * mt * mt),
            ctx:      z16!(mbn * num_heads * mt * head_dim),
            attn_out: z16!(mbn * mt * d_model),
            x_mid:    z16!(mbn * mt * d_model),
            xn2:      z16!(mbn * mt * d_model),
            ln2_mean: zf32!(mbn * mt),
            ln2_rstd: zf32!(mbn * mt),
            ff1:      z16!(mbn * mt * ffn_dim),
            ff1_act:  z16!(mbn * mt * ffn_dim),
        }).collect();

        let max_3d_ff = (3 * d_model).max(ffn_dim);
        Self {
            stream: stream.clone(), module, blas, fns,
            embed, pos_embed,
            m_embed, v_embed, m_pos, v_pos,
            layers,
            ln_f_g, ln_f_b,
            m_ln_f_g, v_ln_f_g, m_ln_f_b, v_ln_f_b,
            vocab_size, d_model, num_heads, num_layers, ffn_dim, max_seq_len, head_dim,
            adam_step: 0,
            grads,
            acts,
            g_embed:      zf32!(vocab_size * d_model),
            g_pos:        zf32!(max_seq_len * d_model),
            g_embed_head_f16: z16!(vocab_size * d_model),
            g_ln_f_g:     zf32!(d_model),
            g_ln_f_b:     zf32!(d_model),
            x_buf:        z16!(mbn * mt * d_model),
            x_norm_buf:   z16!(mbn * mt * d_model),
            lnf_mean:     zf32!(mbn * mt),
            lnf_rstd:     zf32!(mbn * mt),
            logits_buf:   z16!(mbn * mt * vocab_size),
            d_logits:     z16!(mbn * mt * vocab_size),
            loss_acc:     zf32!(1),
            grad_norm_sq: zf32!(1),
            // Sized for the largest grad tensor (embedding = vocab*d_model).
            // norm_reduce writes one partial float per 256-thread block.
            partial_norm_buf: zf32!((vocab_size * d_model + 255) / 256 + 16),
            ids_buf:      zi32!(mbn * mt),
            tgt_buf:      zi32!(mbn * mt),
            msk_buf:      zf32!(mbn * mt),
            dx_buf:       z16!(mbn * mt * d_model),
            tmp_buf:      z16!(mbn * mt * max_3d_ff),
            lora_tmp:     z16!(mbn * mt * 16),  // [batch, seq, rank] — intermediate for LoRA
            dq_buf:       z16!(mbn * num_heads * mt * head_dim),
            dk_buf:       z16!(mbn * num_heads * mt * head_dim),
            dv_buf:       z16!(mbn * num_heads * mt * head_dim),
            dk_f32_buf:   zf32!(mbn * num_heads * mt * head_dim),
            dv_f32_buf:   zf32!(mbn * num_heads * mt * head_dim),
            lse_buf:      zf32!(mbn * num_heads * mt),
            d_attn_buf:   z16!(mbn * num_heads * mt * mt),
            d_ctx_buf:    z16!(mbn * num_heads * mt * head_dim),
            cuda_graph:   None,
            lora_config:  None,
            int4_quantized: false,
            gradient_checkpointing: false,
            lora_backward_enabled: false,
        }
    }

    // ─────────────────────────────────────────────────────────────
    //  CPU inference: forward through one layer
    //  x_in: [T, D] flat f32 → returns [T, D]
    // ─────────────────────────────────────────────────────────────
    fn cpu_layer_forward(&self, layer: &TransformerLayer, x: &[f32],
                          k_cache: &[f32], v_cache: &[f32],
                          t: usize, cache_t: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let d = self.d_model;
        let h = self.num_heads;
        let dh = self.head_dim;

        // Download weights
        let w_qkv = download_f16(&self.stream, &layer.w_qkv);
        let b_qkv = download_f16(&self.stream, &layer.b_qkv);
        let w_out = download_f16(&self.stream, &layer.w_out);
        let b_out = download_f16(&self.stream, &layer.b_out);
        let w_ff1 = download_f16(&self.stream, &layer.w_ff1);
        let b_ff1 = download_f16(&self.stream, &layer.b_ff1);
        let w_ff2 = download_f16(&self.stream, &layer.w_ff2);
        let b_ff2 = download_f16(&self.stream, &layer.b_ff2);
        let ln1_g = download_f16(&self.stream, &layer.ln1_g);
        let ln1_b = download_f16(&self.stream, &layer.ln1_b);
        let ln2_g = download_f16(&self.stream, &layer.ln2_g);
        let ln2_b = download_f16(&self.stream, &layer.ln2_b);

        let w_qkv_f: Vec<f32> = w_qkv;
        let b_qkv_f: Vec<f32> = b_qkv;
        let w_out_f: Vec<f32> = w_out;
        let b_out_f: Vec<f32> = b_out;
        let w_ff1_f: Vec<f32> = w_ff1;
        let b_ff1_f: Vec<f32> = b_ff1;
        let w_ff2_f: Vec<f32> = w_ff2;
        let b_ff2_f: Vec<f32> = b_ff2;
        let ln1_g_f: Vec<f32> = ln1_g;
        let ln1_b_f: Vec<f32> = ln1_b;
        let ln2_g_f: Vec<f32> = ln2_g;
        let ln2_b_f: Vec<f32> = ln2_b;

        // LayerNorm 1 on x [t, d]
        let xn1 = cpu_layernorm(x, &ln1_g_f, &ln1_b_f, t, d);

        // QKV projection: [t, d] @ [d, 3d] → [t, 3d]
        let qkv = cpu_matmul(&xn1, &w_qkv_f, t, d, 3*d);
        let qkv: Vec<f32> = qkv.iter().zip(b_qkv_f.iter().cycle()).map(|(a,b)| a+b).collect();

        // Split Q, K, V each [t, d]
        let mut q = vec![0f32; t * d];
        let mut k_new = vec![0f32; t * d];
        let mut v_new = vec![0f32; t * d];
        for i in 0..t {
            q[i*d..i*d+d].copy_from_slice(&qkv[i*3*d..i*3*d+d]);
            k_new[i*d..i*d+d].copy_from_slice(&qkv[i*3*d+d..i*3*d+2*d]);
            v_new[i*d..i*d+d].copy_from_slice(&qkv[i*3*d+2*d..i*3*d+3*d]);
        }

        // Append new K, V to cache
        let mut k_full = k_cache.to_vec();
        k_full.extend_from_slice(&k_new);
        let mut v_full = v_cache.to_vec();
        v_full.extend_from_slice(&v_new);
        let total_t = cache_t + t;

        // Scaled dot-product attention per head
        let scale = 1.0 / (dh as f32).sqrt();
        let mut attn_out = vec![0f32; t * d];
        for head in 0..h {
            // Compute attention for each query position
            for qi in 0..t {
                let q_vec: Vec<f32> = (0..dh).map(|j| q[qi*d + head*dh + j]).collect();
                let mut scores = vec![0f32; total_t];
                for ki in 0..total_t {
                    let k_vec: Vec<f32> = (0..dh).map(|j| k_full[ki*d + head*dh + j]).collect();
                    let dot: f32 = q_vec.iter().zip(k_vec.iter()).map(|(a,b)| a*b).sum();
                    // Causal mask: only attend to positions up to cache_t + qi
                    scores[ki] = if ki <= cache_t + qi { dot * scale } else { f32::NEG_INFINITY };
                }
                // Softmax
                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = scores.iter().map(|s| if s.is_finite() { (s - max_s).exp() } else { 0.0 }).collect();
                let sum_e: f32 = exps.iter().sum();
                let probs: Vec<f32> = exps.iter().map(|e| e / sum_e.max(1e-9)).collect();
                // Weighted sum of V
                for j in 0..dh {
                    let val: f32 = probs.iter().enumerate()
                        .map(|(ki, p)| p * v_full[ki*d + head*dh + j]).sum();
                    attn_out[qi*d + head*dh + j] = val;
                }
            }
        }

        // Output projection: [t, d] @ [d, d] → [t, d]
        let attn_proj = cpu_matmul(&attn_out, &w_out_f, t, d, d);
        let attn_proj: Vec<f32> = attn_proj.iter().zip(b_out_f.iter().cycle()).map(|(a,b)| a+b).collect();

        // Residual
        let x2: Vec<f32> = x.iter().zip(attn_proj.iter()).map(|(a,b)| a+b).collect();

        // LayerNorm 2
        let xn2 = cpu_layernorm(&x2, &ln2_g_f, &ln2_b_f, t, d);

        // FFN: [t, d] @ [d, ff] → [t, ff] → GELU → [t, ff] @ [ff, d] → [t, d]
        let ff1 = cpu_matmul(&xn2, &w_ff1_f, t, d, self.ffn_dim);
        let ff1: Vec<f32> = ff1.iter().zip(b_ff1_f.iter().cycle()).map(|(a,b)| a+b).collect();
        let ff1_act: Vec<f32> = ff1.iter().map(|&v| gelu(v)).collect();
        let ff2 = cpu_matmul(&ff1_act, &w_ff2_f, t, self.ffn_dim, d);
        let ff2: Vec<f32> = ff2.iter().zip(b_ff2_f.iter().cycle()).map(|(a,b)| a+b).collect();

        // Residual
        let x_out: Vec<f32> = x2.iter().zip(ff2.iter()).map(|(a,b)| a+b).collect();

        (x_out, k_full, v_full)
    }

    // ─────────────────────────────────────────────────────────────
    //  Inference: process sequence of tokens, return logits + KV-cache
    // ─────────────────────────────────────────────────────────────
    pub fn forward_seq(&self, tokens: &[usize]) -> (Vec<f32>, KVCache) {
        let t = tokens.len();
        let d = self.d_model;

        let embed_w = download_f16(&self.stream, &self.embed);
        let pos_w   = download_f16(&self.stream, &self.pos_embed);
        let ln_f_g  = download_f16(&self.stream, &self.ln_f_g);
        let ln_f_b  = download_f16(&self.stream, &self.ln_f_b);

        // Build input: embed + pos
        let mut x = vec![0f32; t * d];
        for (i, &tok) in tokens.iter().enumerate() {
            let tok = tok.min(self.vocab_size - 1);
            for j in 0..d {
                x[i*d + j] = embed_w[tok*d + j] + pos_w[i*d + j];
            }
        }

        let mut kv = KVCache::new(self.num_layers);
        for (li, layer) in self.layers.iter().enumerate() {
            let (x_new, k_full, v_full) = self.cpu_layer_forward(
                layer, &x, &[], &[], t, 0);
            x = x_new;
            kv.k[li] = k_full;
            kv.v[li] = v_full;
        }
        kv.seq_len = t;

        // Final LayerNorm on last token
        let ln_f_g_f: Vec<f32> = ln_f_g;
        let ln_f_b_f: Vec<f32> = ln_f_b;
        let x_norm = cpu_layernorm(&x, &ln_f_g_f, &ln_f_b_f, t, d);

        // Logits from last token: [d] @ embed^T [d, vocab] → [vocab]
        let last = &x_norm[(t-1)*d..t*d];
        let embed_f: Vec<f32> = embed_w;
        let mut logits = vec![0f32; self.vocab_size];
        for v in 0..self.vocab_size {
            logits[v] = last.iter().zip(embed_f[v*d..v*d+d].iter()).map(|(a,b)| a*b).sum();
        }

        (logits, kv)
    }

    // Single-step inference using KV-cache
    pub fn step(&self, token: usize, kv: &KVCache) -> (Vec<f32>, KVCache) {
        let d = self.d_model;
        let pos = kv.seq_len;

        let embed_w = download_f16(&self.stream, &self.embed);
        let pos_w   = download_f16(&self.stream, &self.pos_embed);
        let ln_f_g  = download_f16(&self.stream, &self.ln_f_g);
        let ln_f_b  = download_f16(&self.stream, &self.ln_f_b);

        let tok = token.min(self.vocab_size - 1);
        let pos = pos.min(self.max_seq_len - 1);
        let mut x = vec![0f32; d];
        for j in 0..d {
            x[j] = embed_w[tok*d + j] + pos_w[pos*d + j];
        }

        let mut new_kv = KVCache::new(self.num_layers);
        for (li, layer) in self.layers.iter().enumerate() {
            let cache_t = kv.k[li].len() / d;
            let (x_new, k_full, v_full) = self.cpu_layer_forward(
                layer, &x, &kv.k[li], &kv.v[li], 1, cache_t);
            x = x_new;
            new_kv.k[li] = k_full;
            new_kv.v[li] = v_full;
        }
        new_kv.seq_len = kv.seq_len + 1;

        let ln_f_g_f: Vec<f32> = ln_f_g;
        let ln_f_b_f: Vec<f32> = ln_f_b;
        let x_norm = cpu_layernorm(&x, &ln_f_g_f, &ln_f_b_f, 1, d);

        let embed_f: Vec<f32> = embed_w;
        let mut logits = vec![0f32; self.vocab_size];
        for v in 0..self.vocab_size {
            logits[v] = x_norm.iter().zip(embed_f[v*d..v*d+d].iter()).map(|(a,b)| a*b).sum();
        }

        (logits, new_kv)
    }

    // ─────────────────────────────────────────────────────────────
    //  Sampling
    // ─────────────────────────────────────────────────────────────
    pub fn sample_greedy(&self, logits: &[f32]) -> usize {
        logits.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap_or(0)
    }

    pub fn sample_top_k(&self, logits: &[f32], temperature: f32, k: usize) -> usize {
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
        idx.truncate(k);
        let temp = temperature.max(1e-3);
        let max_l = idx.iter().map(|&i| logits[i]).fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = idx.iter().map(|&i| ((logits[i] - max_l) / temp).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let mut r: f32 = rand::thread_rng().gen::<f32>() * sum;
        for (i, &e) in exps.iter().enumerate() {
            r -= e;
            if r <= 0.0 { return idx[i]; }
        }
        idx[0]
    }

    pub fn sample_top_p(&self, logits: &[f32], temperature: f32, p: f32) -> usize {
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
        let temp = temperature.max(1e-3);
        let max_l = logits[idx[0]];
        let exps: Vec<f32> = idx.iter().map(|&i| ((logits[i] - max_l) / temp).exp()).collect();
        let total: f32 = exps.iter().sum();
        let mut cum = 0.0f32;
        let mut cutoff = idx.len();
        for (i, &e) in exps.iter().enumerate() {
            cum += e / total;
            if cum >= p { cutoff = i + 1; break; }
        }
        let sub_idx = &idx[..cutoff];
        let sub_exp: Vec<f32> = sub_idx.iter().map(|&i| ((logits[i] - max_l) / temp).exp()).collect();
        let s: f32 = sub_exp.iter().sum();
        let mut r = rand::thread_rng().gen::<f32>() * s;
        for (i, &e) in sub_exp.iter().enumerate() {
            r -= e;
            if r <= 0.0 { return sub_idx[i]; }
        }
        sub_idx[0]
    }

    // ─────────────────────────────────────────────────────────────
    //  Free training-only buffers to reduce VRAM for inference.
    //  After calling this, training methods (train_batch_*) will crash —
    //  use only inference methods (forward_gpu, generate_gpu, forward_seq, step).
    // ─────────────────────────────────────────────────────────────
    pub fn free_training_buffers(&mut self) {
        let stream = self.stream.clone();
        let tiny_f32 = |s: &Arc<CudaStream>| s.clone_htod(&vec![0.0f32; 1]).unwrap();
        let tiny_f16 = |s: &Arc<CudaStream>| s.clone_htod(&vec![f16::ZERO; 1]).unwrap();
        let tiny_i32 = |s: &Arc<CudaStream>| s.clone_htod(&vec![0i32; 1]).unwrap();

        // Adam moments (biggest — ~1 GB for 124M model)
        self.m_embed = tiny_f32(&stream); self.v_embed = tiny_f32(&stream);
        self.m_pos   = tiny_f32(&stream); self.v_pos   = tiny_f32(&stream);
        self.m_ln_f_g = tiny_f32(&stream); self.v_ln_f_g = tiny_f32(&stream);
        self.m_ln_f_b = tiny_f32(&stream); self.v_ln_f_b = tiny_f32(&stream);
        for l in &mut self.layers {
            l.m_w_qkv = tiny_f32(&stream); l.v_w_qkv = tiny_f32(&stream);
            l.m_b_qkv = tiny_f32(&stream); l.v_b_qkv = tiny_f32(&stream);
            l.m_w_out  = tiny_f32(&stream); l.v_w_out  = tiny_f32(&stream);
            l.m_b_out  = tiny_f32(&stream); l.v_b_out  = tiny_f32(&stream);
            l.m_w_ff1  = tiny_f32(&stream); l.v_w_ff1  = tiny_f32(&stream);
            l.m_b_ff1  = tiny_f32(&stream); l.v_b_ff1  = tiny_f32(&stream);
            l.m_w_ff2  = tiny_f32(&stream); l.v_w_ff2  = tiny_f32(&stream);
            l.m_b_ff2  = tiny_f32(&stream); l.v_b_ff2  = tiny_f32(&stream);
            l.m_ln1_g = tiny_f32(&stream); l.v_ln1_g = tiny_f32(&stream);
            l.m_ln1_b = tiny_f32(&stream); l.v_ln1_b = tiny_f32(&stream);
            l.m_ln2_g = tiny_f32(&stream); l.v_ln2_g = tiny_f32(&stream);
            l.m_ln2_b = tiny_f32(&stream); l.v_ln2_b = tiny_f32(&stream);
        }

        // Gradient buffers
        self.g_embed  = tiny_f32(&stream);
        self.g_pos    = tiny_f32(&stream);
        self.g_embed_head_f16 = tiny_f16(&stream);
        self.g_ln_f_g = tiny_f32(&stream);
        self.g_ln_f_b = tiny_f32(&stream);
        for g in &mut self.grads {
            g.g_w_qkv = tiny_f16(&stream); g.g_w_out = tiny_f16(&stream);
            g.g_w_ff1 = tiny_f16(&stream); g.g_w_ff2 = tiny_f16(&stream);
            g.g_b_qkv = tiny_f32(&stream); g.g_b_out = tiny_f32(&stream);
            g.g_b_ff1 = tiny_f32(&stream); g.g_b_ff2 = tiny_f32(&stream);
            g.g_ln1_g = tiny_f32(&stream); g.g_ln1_b = tiny_f32(&stream);
            g.g_ln2_g = tiny_f32(&stream); g.g_ln2_b = tiny_f32(&stream);
        }

        // Backward-only working buffers
        self.d_logits   = tiny_f16(&stream);
        self.dx_buf     = tiny_f16(&stream);
        self.tmp_buf    = tiny_f16(&stream);
        self.lora_tmp   = tiny_f16(&stream);
        self.dq_buf     = tiny_f16(&stream);
        self.dk_buf     = tiny_f16(&stream);
        self.dv_buf     = tiny_f16(&stream);
        self.dk_f32_buf = tiny_f32(&stream);
        self.dv_f32_buf = tiny_f32(&stream);
        self.lse_buf    = tiny_f32(&stream);
        self.d_attn_buf = tiny_f16(&stream);
        self.d_ctx_buf  = tiny_f16(&stream);

        // Training-only host-to-device staging
        self.tgt_buf    = tiny_i32(&stream);
        self.msk_buf    = tiny_f32(&stream);

        // Clipping scratch
        self.grad_norm_sq     = tiny_f32(&stream);
        self.partial_norm_buf = tiny_f32(&stream);

        stream.synchronize().ok();
    }

    // ─────────────────────────────────────────────────────────────
    //  Enable LoRA: initialize adapters for all layers
    //  Adapters: A matrices (Kaiming), B matrices (zeros)
    // ─────────────────────────────────────────────────────────────
    pub fn enable_lora(&mut self, rank: usize) {
        if self.lora_config.is_some() {
            println!("[LoRA] Already enabled");
            return;
        }

        let config = LoraConfig::new(rank as usize, rank as f32);
        println!("[LoRA] Enabling with rank={}, alpha={}", rank, config.alpha);

        for li in 0..self.num_layers {
            let adapters = LayerLoraAdapters::init_kaiming_zeros(
                &self.stream,
                self.d_model,
                self.ffn_dim,
                rank,
            );
            self.layers[li].lora = Some(Box::new(adapters));
        }

        self.lora_config = Some(config);
        println!("[LoRA] Initialized for all {} layers", self.num_layers);
    }

    pub fn enable_int4_quantization(&mut self) {
        if self.int4_quantized {
            println!("[INT4] Already enabled");
            return;
        }
        self.int4_quantized = true;
        println!("[INT4] Quantization enabled — base weights will be dequantized on-the-fly");
    }

    pub fn enable_gradient_checkpointing(&mut self) {
        if self.gradient_checkpointing {
            println!("[GradCheckpoint] Already enabled");
            return;
        }
        self.gradient_checkpointing = true;
        println!("[GradCheckpoint] Enabled — activations will be recomputed in backward pass");
    }

    pub fn enable_lora_backward(&mut self) {
        if !self.lora_config.is_some() {
            println!("[LoRA Backward] LoRA not enabled yet");
            return;
        }
        if self.lora_backward_enabled {
            println!("[LoRA Backward] Already enabled");
            return;
        }
        self.lora_backward_enabled = true;
        println!("[LoRA Backward] Enabled — adapter gradients (A, B matrices) will be computed");
    }

    /// Prepare model for v3.5.1 training: enable INT4 quantization, gradient checkpointing, and LoRA backward
    pub fn prepare_v351_training(&mut self) {
        println!("\n[v3.5.1] Preparing model for optimized training...");
        self.enable_int4_quantization();
        self.enable_gradient_checkpointing();
        if self.lora_config.is_some() {
            self.enable_lora_backward();
        }
        println!("[v3.5.1] Training preparation complete\n");
    }

    // ─────────────────────────────────────────────────────────────
    //  GPU forward pass for inference (batch = 1).
    //  Returns logits [vocab_size] for the LAST token only.
    // ─────────────────────────────────────────────────────────────
    pub fn forward_gpu(&mut self, tokens: &[usize]) -> Vec<f32> {
        let d   = self.d_model;
        let ff  = self.ffn_dim;
        let h   = self.num_heads;
        let dh  = self.head_dim;
        let nl  = self.num_layers;
        let vs  = self.vocab_size;
        let mt  = self.max_seq_len;

        let t = tokens.len().min(mt);
        if t == 0 { return vec![0.0; vs]; }
        let nt = t;               // batch = 1, so nt == t
        let nh = h;                // nh == h
        let one = f16::from_f32(1.0);

        // Upload token ids
        let mut ids_flat = vec![0i32; mt];
        for i in 0..t { ids_flat[i] = tokens[i].min(vs - 1) as i32; }
        self.stream.memcpy_htod(&ids_flat, &mut self.ids_buf).unwrap();

        // embed + pos → x_buf[nt, d]
        {
            let bx = (d + 255) / 256;
            let cfg = LaunchConfig { grid_dim: (nt as u32, bx as u32, 1), block_dim: (256.min(d) as u32, 1, 1), shared_mem_bytes: 0 };
            unsafe { self.stream.launch_builder(&self.fns.emb_pos_fwd_nb)
                .arg(&mut self.x_buf).arg(&self.embed).arg(&self.pos_embed)
                .arg(&self.ids_buf).arg(&(nt as i32)).arg(&(t as i32)).arg(&(d as i32))
                .launch(cfg).unwrap(); }
        }

        for li in 0..nl {
            // copy x_buf → x_pre
            {
                let n = nt * d;
                let acts_ptr = &mut self.acts[li].x_pre as *mut CudaSlice<f16>;
                let x_ptr    = &self.x_buf as *const CudaSlice<f16>;
                unsafe { self.stream.launch_builder(&self.fns.copy_f16)
                    .arg(&mut *acts_ptr).arg(&*x_ptr).arg(&(n as i32))
                    .launch(cfg1d(n)).unwrap(); }
            }

            // LN1
            {
                let acts_ptr  = &mut self.acts[li] as *mut GpuLayerActs;
                let layer_ptr = &self.layers[li] as *const TransformerLayer;
                let x_ptr     = &self.x_buf as *const CudaSlice<f16>;
                let cfg = cfg_ln(nt);
                unsafe {
                    let acts  = &mut *acts_ptr;
                    let layer = &*layer_ptr;
                    self.stream.launch_builder(&self.fns.layer_norm_fwd)
                        .arg(&mut acts.xn1).arg(&mut acts.ln1_mean).arg(&mut acts.ln1_rstd)
                        .arg(&*x_ptr).arg(&layer.ln1_g).arg(&layer.ln1_b)
                        .arg(&(nt as i32)).arg(&(d as i32)).arg(&1e-5f32)
                        .launch(cfg).unwrap();
                }
            }

            // QKV projection
            {
                let acts_ptr  = &mut self.acts[li] as *mut GpuLayerActs;
                let layer_ptr = &self.layers[li] as *const TransformerLayer;
                unsafe {
                    let acts  = &mut *acts_ptr;
                    let layer = &*layer_ptr;
                    gemm(&self.blas, &acts.xn1, &layer.w_qkv, &mut acts.qkv,
                         nt, d, 3*d, false, false, one, f16::from_f32(0.0));
                    let n = nt * 3 * d;
                    self.stream.launch_builder(&self.fns.add_bias)
                        .arg(&mut acts.qkv).arg(&layer.b_qkv).arg(&(nt as i32)).arg(&((3*d) as i32))
                        .launch(cfg1d(n)).unwrap();
                }
            }

            // LoRA adapter for QKV
            if let Some(lora_cfg) = &self.lora_config {
                if let Some(lora_adapters) = &self.layers[li].lora {
                    let rank = lora_cfg.rank;
                    let scale = lora_cfg.scale();
                    let acts_ptr = &mut self.acts[li] as *mut GpuLayerActs;
                    unsafe {
                        let acts = &mut *acts_ptr;
                        // temp = xn1 @ a_qkv^T  [nt, d] @ [d, rank] → [nt, rank]
                        gemm(&self.blas, &acts.xn1, &lora_adapters.a_qkv, &mut self.lora_tmp,
                             nt, d, rank, false, true, one, f16::from_f32(0.0));
                        // adapter = scale * (temp @ b_qkv^T)  [nt, rank] @ [rank, 3*d] → [nt, 3*d]
                        gemm(&self.blas, &self.lora_tmp, &lora_adapters.b_qkv, &mut self.tmp_buf,
                             nt, rank, 3*d, false, true, f16::from_f32(scale), f16::from_f32(0.0));
                        // qkv += adapter
                        let n = nt * 3 * d;
                        self.stream.launch_builder(&self.fns.add_inplace)
                            .arg(&mut acts.qkv).arg(&self.tmp_buf).arg(&(n as i32))
                            .launch(cfg1d(n)).unwrap();
                    }
                }
            }

            // split heads
            {
                let acts_ptr = &mut self.acts[li] as *mut GpuLayerActs;
                let cfg = LaunchConfig { grid_dim: (nt as u32, h as u32, 1), block_dim: (dh as u32, 1, 1), shared_mem_bytes: 0 };
                unsafe {
                    let acts = &mut *acts_ptr;
                    self.stream.launch_builder(&self.fns.qkv_split_nb)
                        .arg(&acts.qkv).arg(&mut acts.q).arg(&mut acts.k).arg(&mut acts.v)
                        .arg(&(nt as i32)).arg(&(t as i32)).arg(&(h as i32)).arg(&(dh as i32))
                        .launch(cfg).unwrap();
                }
            }

            // Q @ K^T (batched)
            {
                let acts_ptr = &mut self.acts[li] as *mut GpuLayerActs;
                let scale_attn = f16::from_f32(1.0f32 / (dh as f32).sqrt());
                unsafe {
                    let acts = &mut *acts_ptr;
                    gemm_batched_f16(&self.blas,
                        &acts.q, &acts.k, &mut acts.scores,
                        nh, t, dh, t,
                        true, scale_attn, f16::from_f32(0.0));
                }
            }

            // causal softmax
            {
                let acts_ptr = &mut self.acts[li] as *mut GpuLayerActs;
                let cfg = cfg_attn_sfx(nh, t);
                unsafe {
                    let acts = &mut *acts_ptr;
                    self.stream.launch_builder(&self.fns.causal_sfx)
                        .arg(&mut acts.scores).arg(&(nh as i32)).arg(&(t as i32))
                        .launch(cfg).unwrap();
                }
            }

            // scores @ V
            {
                let acts_ptr = &mut self.acts[li] as *mut GpuLayerActs;
                unsafe {
                    let acts = &mut *acts_ptr;
                    gemm_batched_f16(&self.blas,
                        &acts.scores, &acts.v, &mut acts.ctx,
                        nh, t, t, dh,
                        false, f16::from_f32(1.0), f16::from_f32(0.0));
                }
            }

            // merge heads
            {
                let acts_ptr = &mut self.acts[li] as *mut GpuLayerActs;
                let cfg = LaunchConfig { grid_dim: (nt as u32, h as u32, 1), block_dim: (dh as u32, 1, 1), shared_mem_bytes: 0 };
                unsafe {
                    let acts = &mut *acts_ptr;
                    self.stream.launch_builder(&self.fns.heads_merge_nb)
                        .arg(&acts.ctx).arg(&mut acts.attn_out)
                        .arg(&(nt as i32)).arg(&(t as i32)).arg(&(h as i32)).arg(&(dh as i32))
                        .launch(cfg).unwrap();
                }
            }

            // attn_out @ w_out + b_out + residual
            {
                let acts_ptr  = &mut self.acts[li] as *mut GpuLayerActs;
                let layer_ptr = &self.layers[li] as *const TransformerLayer;
                let n = nt * d;
                unsafe {
                    let acts  = &mut *acts_ptr;
                    let layer = &*layer_ptr;
                    gemm(&self.blas, &acts.attn_out, &layer.w_out, &mut acts.x_mid,
                         nt, d, d, false, false, one, f16::from_f32(0.0));
                    self.stream.launch_builder(&self.fns.add_bias)
                        .arg(&mut acts.x_mid).arg(&layer.b_out).arg(&(nt as i32)).arg(&(d as i32))
                        .launch(cfg1d(n)).unwrap();
                }
            }

            // LoRA adapter for output projection
            if let Some(lora_cfg) = &self.lora_config {
                if let Some(lora_adapters) = &self.layers[li].lora {
                    let rank = lora_cfg.rank;
                    let scale = lora_cfg.scale();
                    let acts_ptr = &mut self.acts[li] as *mut GpuLayerActs;
                    let n = nt * d;
                    unsafe {
                        let acts = &mut *acts_ptr;
                        // temp = attn_out @ a_out^T  [nt, d] @ [d, rank] → [nt, rank]
                        gemm(&self.blas, &acts.attn_out, &lora_adapters.a_out, &mut self.lora_tmp,
                             nt, d, rank, false, true, one, f16::from_f32(0.0));
                        // adapter = scale * (temp @ b_out^T)  [nt, rank] @ [rank, d] → [nt, d]
                        gemm(&self.blas, &self.lora_tmp, &lora_adapters.b_out, &mut self.tmp_buf,
                             nt, rank, d, false, true, f16::from_f32(scale), f16::from_f32(0.0));
                        // x_mid += adapter
                        self.stream.launch_builder(&self.fns.add_inplace)
                            .arg(&mut acts.x_mid).arg(&self.tmp_buf).arg(&(n as i32))
                            .launch(cfg1d(n)).unwrap();
                    }
                }
            }

            // Residual connection
            {
                let acts_ptr = &mut self.acts[li] as *mut GpuLayerActs;
                let n = nt * d;
                unsafe {
                    let acts = &mut *acts_ptr;
                    self.stream.launch_builder(&self.fns.add_inplace)
                        .arg(&mut acts.x_mid).arg(&self.x_buf).arg(&(n as i32))
                        .launch(cfg1d(n)).unwrap();
                }
            }

            // LN2
            {
                let acts_ptr  = &mut self.acts[li] as *mut GpuLayerActs;
                let layer_ptr = &self.layers[li] as *const TransformerLayer;
                let cfg = cfg_ln(nt);
                unsafe {
                    let acts  = &mut *acts_ptr;
                    let layer = &*layer_ptr;
                    self.stream.launch_builder(&self.fns.layer_norm_fwd)
                        .arg(&mut acts.xn2).arg(&mut acts.ln2_mean).arg(&mut acts.ln2_rstd)
                        .arg(&acts.x_mid).arg(&layer.ln2_g).arg(&layer.ln2_b)
                        .arg(&(nt as i32)).arg(&(d as i32)).arg(&1e-5f32)
                        .launch(cfg).unwrap();
                }
            }

            // FFN layer 1: xn2 @ w_ff1 + LoRA + bias + GELU
            {
                let acts_ptr  = &mut self.acts[li] as *mut GpuLayerActs;
                let layer_ptr = &self.layers[li] as *const TransformerLayer;
                let n_ff = nt * ff;
                unsafe {
                    let acts  = &mut *acts_ptr;
                    let layer = &*layer_ptr;
                    gemm(&self.blas, &acts.xn2, &layer.w_ff1, &mut acts.ff1,
                         nt, d, ff, false, false, one, f16::from_f32(0.0));
                }
            }

            // LoRA adapter for FFN layer 1
            if let Some(lora_cfg) = &self.lora_config {
                if let Some(lora_adapters) = &self.layers[li].lora {
                    let rank = lora_cfg.rank;
                    let scale = lora_cfg.scale();
                    let acts_ptr = &mut self.acts[li] as *mut GpuLayerActs;
                    let n_ff = nt * ff;
                    unsafe {
                        let acts = &mut *acts_ptr;
                        // temp = xn2 @ a_ff1^T  [nt, d] @ [d, rank] → [nt, rank]
                        gemm(&self.blas, &acts.xn2, &lora_adapters.a_ff1, &mut self.lora_tmp,
                             nt, d, rank, false, true, one, f16::from_f32(0.0));
                        // adapter = scale * (temp @ b_ff1^T)  [nt, rank] @ [rank, ff] → [nt, ff]
                        gemm(&self.blas, &self.lora_tmp, &lora_adapters.b_ff1, &mut self.tmp_buf,
                             nt, rank, ff, false, true, f16::from_f32(scale), f16::from_f32(0.0));
                        // ff1 += adapter
                        self.stream.launch_builder(&self.fns.add_inplace)
                            .arg(&mut acts.ff1).arg(&self.tmp_buf).arg(&(n_ff as i32))
                            .launch(cfg1d(n_ff)).unwrap();
                    }
                }
            }

            // Add bias and GELU activation
            {
                let acts_ptr  = &mut self.acts[li] as *mut GpuLayerActs;
                let layer_ptr = &self.layers[li] as *const TransformerLayer;
                let n_ff = nt * ff;
                unsafe {
                    let acts  = &mut *acts_ptr;
                    let layer = &*layer_ptr;
                    self.stream.launch_builder(&self.fns.add_bias)
                        .arg(&mut acts.ff1).arg(&layer.b_ff1).arg(&(nt as i32)).arg(&(ff as i32))
                        .launch(cfg1d(n_ff)).unwrap();
                    self.stream.launch_builder(&self.fns.gelu_fwd)
                        .arg(&mut acts.ff1_act).arg(&acts.ff1).arg(&(n_ff as i32))
                        .launch(cfg1d(n_ff)).unwrap();
                }
            }

            // FFN layer 2: ff1_act @ w_ff2 + LoRA + bias + residual
            {
                let acts_ptr  = &mut self.acts[li] as *mut GpuLayerActs;
                let layer_ptr = &self.layers[li] as *const TransformerLayer;
                unsafe {
                    let acts  = &mut *acts_ptr;
                    let layer = &*layer_ptr;
                    gemm(&self.blas, &acts.ff1_act, &layer.w_ff2, &mut self.x_buf,
                         nt, ff, d, false, false, one, f16::from_f32(0.0));
                }
            }

            // LoRA adapter for FFN layer 2
            if let Some(lora_cfg) = &self.lora_config {
                if let Some(lora_adapters) = &self.layers[li].lora {
                    let rank = lora_cfg.rank;
                    let scale = lora_cfg.scale();
                    let acts_ptr = &mut self.acts[li] as *mut GpuLayerActs;
                    let n_d = nt * d;
                    unsafe {
                        let acts = &mut *acts_ptr;
                        // temp = ff1_act @ a_ff2^T  [nt, ff] @ [ff, rank] → [nt, rank]
                        gemm(&self.blas, &acts.ff1_act, &lora_adapters.a_ff2, &mut self.lora_tmp,
                             nt, ff, rank, false, true, one, f16::from_f32(0.0));
                        // adapter = scale * (temp @ b_ff2^T)  [nt, rank] @ [rank, d] → [nt, d]
                        gemm(&self.blas, &self.lora_tmp, &lora_adapters.b_ff2, &mut self.tmp_buf,
                             nt, rank, d, false, true, f16::from_f32(scale), f16::from_f32(0.0));
                        // x_buf += adapter
                        self.stream.launch_builder(&self.fns.add_inplace)
                            .arg(&mut self.x_buf).arg(&self.tmp_buf).arg(&(n_d as i32))
                            .launch(cfg1d(n_d)).unwrap();
                    }
                }
            }

            // Add bias and residual
            {
                let acts_ptr  = &mut self.acts[li] as *mut GpuLayerActs;
                let layer_ptr = &self.layers[li] as *const TransformerLayer;
                let n_d  = nt * d;
                unsafe {
                    let acts  = &mut *acts_ptr;
                    let layer = &*layer_ptr;
                    self.stream.launch_builder(&self.fns.add_bias)
                        .arg(&mut self.x_buf).arg(&layer.b_ff2).arg(&(nt as i32)).arg(&(d as i32))
                        .launch(cfg1d(n_d)).unwrap();
                    self.stream.launch_builder(&self.fns.add_inplace)
                        .arg(&mut self.x_buf).arg(&acts.x_mid).arg(&(n_d as i32))
                        .launch(cfg1d(n_d)).unwrap();
                }
            }
        }

        // Final LN
        {
            let cfg = cfg_ln(nt);
            unsafe {
                self.stream.launch_builder(&self.fns.layer_norm_fwd)
                    .arg(&mut self.x_norm_buf).arg(&mut self.lnf_mean).arg(&mut self.lnf_rstd)
                    .arg(&self.x_buf).arg(&self.ln_f_g).arg(&self.ln_f_b)
                    .arg(&(nt as i32)).arg(&(d as i32)).arg(&1e-5f32)
                    .launch(cfg).unwrap();
            }
        }

        // Logits for the whole nt block
        gemm(&self.blas, &self.x_norm_buf, &self.embed, &mut self.logits_buf,
             nt, d, vs, false, true, one, f16::from_f32(0.0));

        self.stream.synchronize().unwrap();

        // Read only last-token logits (offset = (nt-1) * vocab)
        let all_logits: Vec<f16> = self.stream.clone_dtoh(&self.logits_buf).unwrap();
        let offset = (nt - 1) * vs;
        all_logits[offset..offset + vs].iter().map(|x| x.to_f32()).collect()
    }

    // ─────────────────────────────────────────────────────────────
    //  Training: forward + backward + Adam — fully on GPU
    // ─────────────────────────────────────────────────────────────
    pub fn train_batch_masked(&mut self, seqs: &[Vec<usize>], masks: &[Vec<f32>], lr: f32) -> f32 {
        let loss = self.train_batch_gpu(seqs, masks, lr);
        loss
    }

    fn train_batch_gpu(&mut self, seqs: &[Vec<usize>], masks: &[Vec<f32>], lr: f32) -> f32 {
        let d   = self.d_model;
        let ff  = self.ffn_dim;
        let h   = self.num_heads;
        let dh  = self.head_dim;
        let nl  = self.num_layers;
        let v   = self.vocab_size;
        let mt  = self.max_seq_len;
        let nb  = seqs.len();
        if nb == 0 { return 0.0; }
        let scale_f = 1.0f32;

        self.adam_step += 1;
        let step = self.adam_step;
        let bc1 = 1.0 - 0.9f32.powi(step);
        let bc2 = 1.0 - 0.999f32.powi(step);
        let eps = 1e-8f32;

        // ── Zero all gradient buffers ──────────────────────────────
        let one = f16::from_f32(1.0);
        let zero16 = f16::from_f32(0.0);
        macro_rules! zf16 { ($buf:expr) => {
            let n = $buf.len(); let cfg = cfg1d(n);
            unsafe { self.stream.launch_builder(&self.fns.zero_f16).arg(&mut $buf).arg(&(n as i32)).launch(cfg).unwrap(); }
        }}
        macro_rules! zf32 { ($buf:expr) => {
            let n = $buf.len(); let cfg = cfg1d(n);
            unsafe { self.stream.launch_builder(&self.fns.zero_f32).arg(&mut $buf).arg(&(n as i32)).launch(cfg).unwrap(); }
        }}
        zf32!(self.g_embed); zf32!(self.g_pos);
        zf32!(self.g_ln_f_g); zf32!(self.g_ln_f_b);
        {
            let n = self.g_embed_head_f16.len();
            let cfg = cfg1d(n);
            unsafe { self.stream.launch_builder(&self.fns.zero_f16)
                .arg(&mut self.g_embed_head_f16).arg(&(n as i32)).launch(cfg).unwrap(); }
        }
        for li in 0..nl {
            zf16!(self.grads[li].g_w_qkv); zf16!(self.grads[li].g_w_out);
            zf16!(self.grads[li].g_w_ff1); zf16!(self.grads[li].g_w_ff2);
            zf32!(self.grads[li].g_b_qkv); zf32!(self.grads[li].g_b_out);
            zf32!(self.grads[li].g_b_ff1); zf32!(self.grads[li].g_b_ff2);
            zf32!(self.grads[li].g_ln1_g); zf32!(self.grads[li].g_ln1_b);
            zf32!(self.grads[li].g_ln2_g); zf32!(self.grads[li].g_ln2_b);
        }

        // Zero loss_acc once for the whole batch (accumulates across sequences)
        unsafe { self.stream.launch_builder(&self.fns.zero_scalar_f32)
            .arg(&mut self.loss_acc)
            .launch(LaunchConfig { grid_dim: (1,1,1), block_dim: (1,1,1), shared_mem_bytes: 0 }).unwrap(); }

        let mut counted = 0usize;
        let mut tok_counted = 0usize;
        let micro_n = MICRO_BATCH_N;

        // ── MICRO-BATCH LOOP: process micro_n sequences at once ──────
        let mut chunk_start = 0usize;
        while chunk_start < nb {
            let chunk_end = (chunk_start + micro_n).min(nb);
            let cn = chunk_end - chunk_start; // actual sequences in this chunk

            // Max sequence length in this chunk
            let t = (chunk_start..chunk_end)
                .map(|i| seqs[i].len().min(mt))
                .max().unwrap_or(0);
            if t < 2 { chunk_start = chunk_end; continue; }

            let nt = cn * t;  // total tokens: cn sequences × t each
            let nh = cn * h;  // total heads:  cn sequences × h each
            let buf_nt = micro_n * mt; // fixed upload size matching pre-alloc buffers

            // Build flat token/target/mask arrays [micro_n*mt] (padded with zeros)
            let mut ids_flat = vec![0i32; buf_nt];
            let mut tgt_flat = vec![0i32; buf_nt];
            let mut msk_flat = vec![0.0f32; buf_nt];
            let mut chunk_has_tgt = false;

            for (ni, i) in (chunk_start..chunk_end).enumerate() {
                let seq  = &seqs[i];
                let mask = &masks[i];
                let si = seq.len().min(mt).min(t);
                for ti in 0..si {
                    ids_flat[ni*t + ti] = seq[ti].min(v-1) as i32;
                    if ti + 1 < si {
                        tgt_flat[ni*t + ti] = seq[ti+1].min(v-1) as i32;
                        msk_flat[ni*t + ti] = mask[ti];
                        if mask[ti] > 0.5 { chunk_has_tgt = true; tok_counted += 1; }
                    }
                }
            }

            if !chunk_has_tgt { chunk_start = chunk_end; continue; }
            counted += cn;

            self.stream.memcpy_htod(&ids_flat, &mut self.ids_buf).unwrap();
            self.stream.memcpy_htod(&tgt_flat, &mut self.tgt_buf).unwrap();
            self.stream.memcpy_htod(&msk_flat, &mut self.msk_buf).unwrap();

            // ── FORWARD PASS ─────────────────────────────────────────
            // embedding_pos_fwd_nb: x_buf[nt,d] = embed[ids[gt]] + pos[gt%t]
            {
                let bx = (d + 255) / 256;
                let cfg = LaunchConfig { grid_dim: (nt as u32, bx as u32, 1), block_dim: (256.min(d) as u32, 1, 1), shared_mem_bytes: 0 };
                unsafe { self.stream.launch_builder(&self.fns.emb_pos_fwd_nb)
                    .arg(&mut self.x_buf).arg(&self.embed).arg(&self.pos_embed)
                    .arg(&self.ids_buf).arg(&(nt as i32)).arg(&(t as i32)).arg(&(d as i32))
                    .launch(cfg).unwrap(); }
            }

            for li in 0..nl {
                // copy x_buf → x_pre [nt*d]
                {
                    let n = nt * d;
                    let acts_ptr = &mut self.acts[li].x_pre as *mut CudaSlice<f16>;
                    let x_ptr    = &self.x_buf as *const CudaSlice<f16>;
                    unsafe { self.stream.launch_builder(&self.fns.copy_f16)
                        .arg(&mut *acts_ptr).arg(&*x_ptr).arg(&(n as i32))
                        .launch(cfg1d(n)).unwrap(); }
                }

                // layer_norm_fwd: nt rows
                {
                    let acts_ptr  = &mut self.acts[li] as *mut GpuLayerActs;
                    let layer_ptr = &self.layers[li] as *const TransformerLayer;
                    let x_ptr     = &self.x_buf as *const CudaSlice<f16>;
                    let cfg = cfg_ln(nt);
                    unsafe {
                        let acts  = &mut *acts_ptr;
                        let layer = &*layer_ptr;
                        self.stream.launch_builder(&self.fns.layer_norm_fwd)
                            .arg(&mut acts.xn1).arg(&mut acts.ln1_mean).arg(&mut acts.ln1_rstd)
                            .arg(&*x_ptr).arg(&layer.ln1_g).arg(&layer.ln1_b)
                            .arg(&(nt as i32)).arg(&(d as i32)).arg(&1e-5f32)
                            .launch(cfg).unwrap();
                    }
                }

                // QKV: xn1[nt,D] @ w_qkv[D,3D] → qkv[nt,3D]
                {
                    let acts_ptr  = &mut self.acts[li] as *mut GpuLayerActs;
                    let layer_ptr = &self.layers[li] as *const TransformerLayer;
                unsafe {
                    let acts  = &mut *acts_ptr;
                    let layer = &*layer_ptr;
                    gemm(&self.blas, &acts.xn1, &layer.w_qkv, &mut acts.qkv,
                         nt, d, 3*d, false, false, one, f16::from_f32(0.0));
                    let n = nt * 3 * d;
                    self.stream.launch_builder(&self.fns.add_bias)
                        .arg(&mut acts.qkv).arg(&layer.b_qkv).arg(&(nt as i32)).arg(&((3*d) as i32))
                        .launch(cfg1d(n)).unwrap();
                }
                }

                // qkv_split_heads_nb: qkv[nt,3D] → q,k,v [nh,t,dh]
                {
                    let acts_ptr = &mut self.acts[li] as *mut GpuLayerActs;
                    let cfg = LaunchConfig { grid_dim: (nt as u32, h as u32, 1), block_dim: (dh as u32, 1, 1), shared_mem_bytes: 0 };
                    unsafe {
                        let acts = &mut *acts_ptr;
                        self.stream.launch_builder(&self.fns.qkv_split_nb)
                            .arg(&acts.qkv).arg(&mut acts.q).arg(&mut acts.k).arg(&mut acts.v)
                            .arg(&(nt as i32)).arg(&(t as i32)).arg(&(h as i32)).arg(&(dh as i32))
                            .launch(cfg).unwrap();
                    }
                }

                // scores[nh,t,t] = Q[nh,t,dh] @ K[nh,t,dh]^T  * scale  (batched GEMM, Tensor Cores)
                {
                    let acts_ptr = &mut self.acts[li] as *mut GpuLayerActs;
                    let scale_attn = f16::from_f32(1.0f32 / (dh as f32).sqrt());
                    unsafe {
                        let acts = &mut *acts_ptr;
                        gemm_batched_f16(&self.blas,
                            &acts.q, &acts.k, &mut acts.scores,
                            nh, t, dh, t,
                            true, // K is transposed
                            scale_attn, f16::from_f32(0.0));
                    }
                }

                // causal_softmax_fwd: BH=nh, T=t
                {
                    let acts_ptr = &mut self.acts[li] as *mut GpuLayerActs;
                    let cfg = cfg_attn_sfx(nh, t);
                    unsafe {
                        let acts = &mut *acts_ptr;
                        self.stream.launch_builder(&self.fns.causal_sfx)
                            .arg(&mut acts.scores).arg(&(nh as i32)).arg(&(t as i32))
                            .launch(cfg).unwrap();
                    }
                }

                // ctx[nh,t,dh] = scores[nh,t,t] @ V[nh,t,dh]  (batched GEMM, Tensor Cores)
                {
                    let acts_ptr = &mut self.acts[li] as *mut GpuLayerActs;
                    unsafe {
                        let acts = &mut *acts_ptr;
                        gemm_batched_f16(&self.blas,
                            &acts.scores, &acts.v, &mut acts.ctx,
                            nh, t, t, dh,
                            false,
                            f16::from_f32(1.0), f16::from_f32(0.0));
                    }
                }

                // heads_merge_nb: ctx[nh,t,dh] → attn_out[nt,D]
                {
                    let acts_ptr = &mut self.acts[li] as *mut GpuLayerActs;
                    let cfg = LaunchConfig { grid_dim: (nt as u32, h as u32, 1), block_dim: (dh as u32, 1, 1), shared_mem_bytes: 0 };
                    unsafe {
                        let acts = &mut *acts_ptr;
                        self.stream.launch_builder(&self.fns.heads_merge_nb)
                            .arg(&acts.ctx).arg(&mut acts.attn_out)
                            .arg(&(nt as i32)).arg(&(t as i32)).arg(&(h as i32)).arg(&(dh as i32))
                            .launch(cfg).unwrap();
                    }
                }

                // attn_out[nt,D] @ w_out → x_mid[nt,D] + residual
                {
                    let acts_ptr  = &mut self.acts[li] as *mut GpuLayerActs;
                    let layer_ptr = &self.layers[li] as *const TransformerLayer;
                    let n = nt * d;
                    unsafe {
                        let acts  = &mut *acts_ptr;
                        let layer = &*layer_ptr;
                        gemm(&self.blas, &acts.attn_out, &layer.w_out, &mut acts.x_mid,
                             nt, d, d, false, false, one, f16::from_f32(0.0));
                        self.stream.launch_builder(&self.fns.add_bias)
                            .arg(&mut acts.x_mid).arg(&layer.b_out).arg(&(nt as i32)).arg(&(d as i32))
                            .launch(cfg1d(n)).unwrap();
                        self.stream.launch_builder(&self.fns.add_inplace)
                            .arg(&mut acts.x_mid).arg(&self.x_buf).arg(&(n as i32))
                            .launch(cfg1d(n)).unwrap();
                    }
                }

                // LN2: x_mid[nt,D] → xn2
                {
                    let acts_ptr  = &mut self.acts[li] as *mut GpuLayerActs;
                    let layer_ptr = &self.layers[li] as *const TransformerLayer;
                    let cfg = cfg_ln(nt);
                    unsafe {
                        let acts  = &mut *acts_ptr;
                        let layer = &*layer_ptr;
                        self.stream.launch_builder(&self.fns.layer_norm_fwd)
                            .arg(&mut acts.xn2).arg(&mut acts.ln2_mean).arg(&mut acts.ln2_rstd)
                            .arg(&acts.x_mid).arg(&layer.ln2_g).arg(&layer.ln2_b)
                            .arg(&(nt as i32)).arg(&(d as i32)).arg(&1e-5f32)
                            .launch(cfg).unwrap();
                    }
                }

                // FFN: xn2[nt,D] → ff1[nt,ff] → GELU → ff1_act → ff2[nt,D] + residual
                {
                    let acts_ptr  = &mut self.acts[li] as *mut GpuLayerActs;
                    let layer_ptr = &self.layers[li] as *const TransformerLayer;
                    let n_ff = nt * ff;
                    let n_d  = nt * d;
                    unsafe {
                        let acts  = &mut *acts_ptr;
                        let layer = &*layer_ptr;
                        gemm(&self.blas, &acts.xn2, &layer.w_ff1, &mut acts.ff1,
                             nt, d, ff, false, false, one, f16::from_f32(0.0));
                        self.stream.launch_builder(&self.fns.add_bias)
                            .arg(&mut acts.ff1).arg(&layer.b_ff1).arg(&(nt as i32)).arg(&(ff as i32))
                            .launch(cfg1d(n_ff)).unwrap();
                        self.stream.launch_builder(&self.fns.gelu_fwd)
                            .arg(&mut acts.ff1_act).arg(&acts.ff1).arg(&(n_ff as i32))
                            .launch(cfg1d(n_ff)).unwrap();
                        gemm(&self.blas, &acts.ff1_act, &layer.w_ff2, &mut self.x_buf,
                             nt, ff, d, false, false, one, f16::from_f32(0.0));
                        self.stream.launch_builder(&self.fns.add_bias)
                            .arg(&mut self.x_buf).arg(&layer.b_ff2).arg(&(nt as i32)).arg(&(d as i32))
                            .launch(cfg1d(n_d)).unwrap();
                        self.stream.launch_builder(&self.fns.add_inplace)
                            .arg(&mut self.x_buf).arg(&acts.x_mid).arg(&(n_d as i32))
                            .launch(cfg1d(n_d)).unwrap();
                    }
                }
            } // end layer forward loop

            // Final LN [nt,D]
            {
                let cfg = cfg_ln(nt);
                unsafe {
                    self.stream.launch_builder(&self.fns.layer_norm_fwd)
                        .arg(&mut self.x_norm_buf).arg(&mut self.lnf_mean).arg(&mut self.lnf_rstd)
                        .arg(&self.x_buf).arg(&self.ln_f_g).arg(&self.ln_f_b)
                        .arg(&(nt as i32)).arg(&(d as i32)).arg(&1e-5f32)
                        .launch(cfg).unwrap();
                }
            }

            // Logits: x_norm[nt,D] @ embed^T → logits[nt,V]
            gemm(&self.blas, &self.x_norm_buf, &self.embed, &mut self.logits_buf,
                 nt, d, v, false, true, one, f16::from_f32(0.0));

            // CE loss + d_logits for all nt positions (512 threads for better vocab coverage)
            unsafe {
                let bsz: u32 = 512;
                let smem = bsz * 4;
                self.stream.launch_builder(&self.fns.ce_masked)
                    .arg(&self.logits_buf).arg(&mut self.d_logits).arg(&mut self.loss_acc)
                    .arg(&self.tgt_buf).arg(&self.msk_buf)
                    .arg(&(nt as i32)).arg(&(v as i32)).arg(&scale_f)
                    .launch(LaunchConfig { grid_dim: (nt as u32,1,1), block_dim: (bsz,1,1), shared_mem_bytes: smem })
                    .unwrap();
            }

            // ── BACKWARD PASS ────────────────────────────────────────
            // Tied embed grad from output head: g_embed += d_logits^T @ x_norm
            // x_norm_buf currently holds x_norm (final LN output) — use it before overwrite.
            // g_embed_head_f16[v,D] += d_logits[nt,v]^T @ x_norm[nt,D]  → [v, D]
            {
                unsafe {
                    gemm(&self.blas, &self.d_logits, &self.x_norm_buf, &mut self.g_embed_head_f16,
                         v, nt, d, true, false, one, one);
                }
            }

            // d_x_norm[nt,D] = d_logits[nt,V] @ embed[V,D]
            gemm(&self.blas, &self.d_logits, &self.embed, &mut self.x_norm_buf,
                 nt, v, d, false, false, one, f16::from_f32(0.0));

            // Zero dx_buf[nt,D], then final LN bwd accumulates
            { let n = nt * d; unsafe { self.stream.launch_builder(&self.fns.zero_f16).arg(&mut self.dx_buf).arg(&(n as i32)).launch(cfg1d(n)).unwrap(); } }
            {
                // dx — warp-per-row, no atomicAdd
                let cfg_ln_back = cfg_ln(nt);
                unsafe {
                    self.stream.launch_builder(&self.fns.ln_bwd_dx)
                        .arg(&mut self.dx_buf)
                        .arg(&self.x_norm_buf)   // dy
                        .arg(&self.x_buf)         // x (input to final LN)
                        .arg(&self.lnf_mean).arg(&self.lnf_rstd).arg(&self.ln_f_g)
                        .arg(&(nt as i32)).arg(&(d as i32))
                        .launch(cfg_ln_back).unwrap();
                }
                // dgamma / dbeta — proper block reduction (1 atomicAdd per D element)
                unsafe {
                    self.stream.launch_builder(&self.fns.ln_param_grad)
                        .arg(&mut self.g_ln_f_g).arg(&mut self.g_ln_f_b)
                        .arg(&self.x_norm_buf)   // dy
                        .arg(&self.x_buf)         // x
                        .arg(&self.lnf_mean).arg(&self.lnf_rstd)
                        .arg(&(nt as i32)).arg(&(d as i32))
                        .launch(LaunchConfig { grid_dim: (d as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 2 * 256 * 4 }).unwrap();
                }
            }

            // Backward through layers (reverse)
            for li in (0..nl).rev() {
                // g_b_ff2 += sum dx_buf
                {
                    let grads_ptr = &mut self.grads[li] as *mut GpuLayerGrad;
                    unsafe {
                        self.stream.launch_builder(&self.fns.bias_grad)
                            .arg(&self.dx_buf).arg(&mut (*grads_ptr).g_b_ff2)
                            .arg(&(nt as i32)).arg(&(d as i32))
                            .launch(LaunchConfig { grid_dim: (((d+255)/256) as u32,1,1), block_dim: (256,1,1), shared_mem_bytes: 0 })
                            .unwrap();
                    }
                }

                // g_w_ff2 += ff1_act^T @ dx_buf  [ff,nt]@[nt,D]
                {
                    let acts_ptr  = &mut self.acts[li] as *mut GpuLayerActs;
                    let grads_ptr = &mut self.grads[li] as *mut GpuLayerGrad;
                    unsafe {
                        gemm(&self.blas, &(*acts_ptr).ff1_act, &self.dx_buf, &mut (*grads_ptr).g_w_ff2,
                             ff, nt, d, true, false, one, one);
                    }
                }

                // d_ff1_act[nt,ff] = dx_buf[nt,D] @ w_ff2^T → tmp_buf
                // then d_ff1 = d_ff1_act * gelu'(ff1)  — use overwrite kernel (not +=)
                {
                    let acts_ptr  = &mut self.acts[li] as *mut GpuLayerActs;
                    let layer_ptr = &self.layers[li] as *const TransformerLayer;
                    unsafe {
                        gemm(&self.blas, &self.dx_buf, &(*layer_ptr).w_ff2, &mut self.tmp_buf,
                             nt, d, ff, false, true, one, f16::from_f32(0.0));
                        let n_ff = nt * ff;
                        // gelu_bwd_overwrite: dx[i] = dy[i] * gelu'(x[i])  (no +=)
                        // safe to use tmp_buf as both dx and dy since each thread is independent
                        let tmp_ptr = &mut self.tmp_buf as *mut _;
                        self.stream.launch_builder(&self.fns.gelu_bwd_ow)
                            .arg(&mut *tmp_ptr).arg(&*tmp_ptr).arg(&(*acts_ptr).ff1)
                            .arg(&(n_ff as i32))
                            .launch(cfg1d(n_ff)).unwrap();
                    }
                }

                // g_b_ff1
                {
                    let grads_ptr = &mut self.grads[li] as *mut GpuLayerGrad;
                    unsafe {
                        self.stream.launch_builder(&self.fns.bias_grad)
                            .arg(&self.tmp_buf).arg(&mut (*grads_ptr).g_b_ff1)
                            .arg(&(nt as i32)).arg(&(ff as i32))
                            .launch(LaunchConfig { grid_dim: (((ff+255)/256) as u32,1,1), block_dim: (256,1,1), shared_mem_bytes: 0 })
                            .unwrap();
                    }
                }

                // g_w_ff1 += xn2^T @ tmp_buf
                {
                    let acts_ptr  = &mut self.acts[li] as *mut GpuLayerActs;
                    let grads_ptr = &mut self.grads[li] as *mut GpuLayerGrad;
                    unsafe {
                        gemm(&self.blas, &(*acts_ptr).xn2, &self.tmp_buf, &mut (*grads_ptr).g_w_ff1,
                             d, nt, ff, true, false, one, one);
                    }
                }

                // d_xn2 = tmp_buf @ w_ff1^T → x_norm_buf
                {
                    let layer_ptr = &self.layers[li] as *const TransformerLayer;
                    unsafe {
                        gemm(&self.blas, &self.tmp_buf, &(*layer_ptr).w_ff1, &mut self.x_norm_buf,
                             nt, ff, d, false, true, one, f16::from_f32(0.0));
                    }
                }

                // LN2 bwd (accumulates into dx_buf)
                {
                    let acts_ptr  = &mut self.acts[li] as *mut GpuLayerActs;
                    let layer_ptr = &self.layers[li] as *const TransformerLayer;
                    let grads_ptr = &mut self.grads[li] as *mut GpuLayerGrad;
                    let cfg = cfg_ln(nt);
                    unsafe {
                        // dx: warp-per-row, no atomicAdd contention
                        self.stream.launch_builder(&self.fns.ln_bwd_dx)
                            .arg(&mut self.dx_buf)
                            .arg(&self.x_norm_buf)        // dy (d_xn2)
                            .arg(&(*acts_ptr).x_mid)      // x (input to LN2)
                            .arg(&(*acts_ptr).ln2_mean).arg(&(*acts_ptr).ln2_rstd).arg(&(*layer_ptr).ln2_g)
                            .arg(&(nt as i32)).arg(&(d as i32))
                            .launch(cfg).unwrap();
                        // dgamma/dbeta: proper block reduction
                        self.stream.launch_builder(&self.fns.ln_param_grad)
                            .arg(&mut (*grads_ptr).g_ln2_g).arg(&mut (*grads_ptr).g_ln2_b)
                            .arg(&self.x_norm_buf)        // dy
                            .arg(&(*acts_ptr).x_mid)      // x
                            .arg(&(*acts_ptr).ln2_mean).arg(&(*acts_ptr).ln2_rstd)
                            .arg(&(nt as i32)).arg(&(d as i32))
                            .launch(LaunchConfig { grid_dim: (d as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 2 * 256 * 4 }).unwrap();
                    }
                }

                // g_b_out
                {
                    let grads_ptr = &mut self.grads[li] as *mut GpuLayerGrad;
                    unsafe {
                        self.stream.launch_builder(&self.fns.bias_grad)
                            .arg(&self.dx_buf).arg(&mut (*grads_ptr).g_b_out)
                            .arg(&(nt as i32)).arg(&(d as i32))
                            .launch(LaunchConfig { grid_dim: (((d+255)/256) as u32,1,1), block_dim: (256,1,1), shared_mem_bytes: 0 })
                            .unwrap();
                    }
                }

                // g_w_out += attn_out^T @ dx_buf
                {
                    let acts_ptr  = &mut self.acts[li] as *mut GpuLayerActs;
                    let grads_ptr = &mut self.grads[li] as *mut GpuLayerGrad;
                    unsafe {
                        gemm(&self.blas, &(*acts_ptr).attn_out, &self.dx_buf, &mut (*grads_ptr).g_w_out,
                             d, nt, d, true, false, one, one);
                    }
                }

                // d_attn_out = dx_buf @ w_out^T → x_norm_buf
                {
                    let layer_ptr = &self.layers[li] as *const TransformerLayer;
                    unsafe {
                        gemm(&self.blas, &self.dx_buf, &(*layer_ptr).w_out, &mut self.x_norm_buf,
                             nt, d, d, false, true, one, f16::from_f32(0.0));
                    }
                }

                // heads_split_nb: x_norm_buf[nt,D] → d_ctx_buf[nh,t,dh]
                {
                    let cfg = LaunchConfig { grid_dim: (nt as u32, h as u32, 1), block_dim: (dh as u32, 1, 1), shared_mem_bytes: 0 };
                    unsafe {
                        self.stream.launch_builder(&self.fns.heads_split_nb)
                            .arg(&self.x_norm_buf).arg(&mut self.d_ctx_buf)
                            .arg(&(nt as i32)).arg(&(t as i32)).arg(&(h as i32)).arg(&(dh as i32))
                            .launch(cfg).unwrap();
                    }
                }

                // Zero dv, dq, dk, d_attn buffers
                {
                    let nv = nh * t * dh;
                    let na = nh * t * t;
                    unsafe {
                        self.stream.launch_builder(&self.fns.zero_f16).arg(&mut self.dv_buf).arg(&(nv as i32)).launch(cfg1d(nv)).unwrap();
                        self.stream.launch_builder(&self.fns.zero_f16).arg(&mut self.dq_buf).arg(&(nv as i32)).launch(cfg1d(nv)).unwrap();
                        self.stream.launch_builder(&self.fns.zero_f16).arg(&mut self.dk_buf).arg(&(nv as i32)).launch(cfg1d(nv)).unwrap();
                        self.stream.launch_builder(&self.fns.zero_f16).arg(&mut self.d_attn_buf).arg(&(na as i32)).launch(cfg1d(na)).unwrap();
                    }
                }

                // Attention backward via batched GEMMs (Tensor Cores)
                {
                    let acts_ptr = &mut self.acts[li] as *mut GpuLayerActs;
                    let nv = nh * t * dh;
                    let na = nh * t * t;
                    let blas_ptr = &self.blas as *const CudaBlas;

                    // Zero output buffers
                    unsafe {
                        self.stream.launch_builder(&self.fns.zero_f16)
                            .arg(&mut self.dv_buf).arg(&(nv as i32)).launch(cfg1d(nv)).unwrap();
                        self.stream.launch_builder(&self.fns.zero_f16)
                            .arg(&mut self.dq_buf).arg(&(nv as i32)).launch(cfg1d(nv)).unwrap();
                        self.stream.launch_builder(&self.fns.zero_f16)
                            .arg(&mut self.dk_buf).arg(&(nv as i32)).launch(cfg1d(nv)).unwrap();
                        self.stream.launch_builder(&self.fns.zero_f16)
                            .arg(&mut self.d_attn_buf).arg(&(na as i32)).launch(cfg1d(na)).unwrap();
                    }

                    // dv[nh,t,dh] = scores^T[nh,t,t] @ d_ctx[nh,t,dh]
                    // scores is square [t,t] so scores^T has same shape
                    // = gemm_batched: A=scores treated as [t,t] with transb on the "B" side
                    // We express as: dv = (scores as "B" transposed) @ d_ctx as "A"
                    // i.e. gemm_batched_f16(A=d_ctx, B=scores, transb=true) gives d_ctx @ scores^T ≠ what we want
                    // Correct: dv = scores^T @ d_ctx
                    // = gemm_batched_f16(A=scores[t,t] transposed, B=d_ctx[t,dh])
                    // Since gemm_batched_f16 only transposes B, swap: treat d_ctx as "B" and scores as "A"
                    // but we need to transpose "A". Use raw call matching gemm() helper convention:
                    // In row-major: C[t,dh] = (A[t,t])^T @ B[t,dh]
                    //             = A^T @ B (transa=T)
                    // gemm() with transa=T, transb=F: uses cuBLAS CUBLAS_OP_N/CUBLAS_OP_T pattern (case 2)
                    // Replicate that pattern for strided batched:
                    unsafe {
                        use cudarc::cublas::{StridedBatchedConfig, GemmConfig, Gemm};
                        use cudarc::cublas::sys::cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T};
                        // Case transa=true, transb=false from gemm() helper:
                        // cuBLAS: transa=N, transb=T, m=n, n=m, k=k, lda=n, ldb=m, ldc=n
                        // Here: m=t, k=t, n=dh
                        let cfg = StridedBatchedConfig::<f16> {
                            gemm: GemmConfig {
                                transa: CUBLAS_OP_N, transb: CUBLAS_OP_T,
                                m: dh as i32, n: t as i32, k: t as i32,
                                alpha: f16::from_f32(1.0),
                                lda: dh as i32,
                                ldb: t as i32,
                                beta: f16::from_f32(0.0),
                                ldc: dh as i32,
                            },
                            stride_a: (t * dh) as i64,
                            stride_b: (t * t) as i64,
                            stride_c: (t * dh) as i64,
                            batch_size: nh as i32,
                        };
                        // cuBLAS (a=d_ctx, b=scores) → dv = d_ctx_T^T @ scores^T ...
                        // Following gemm() transa=true case: args are (b=d_ctx, a=scores)
                        (*blas_ptr).gemm_strided_batched(cfg,
                            &self.d_ctx_buf, &(*acts_ptr).scores, &mut self.dv_buf).unwrap();
                    }

                    // d_attn[nh,t,t] = d_ctx[nh,t,dh] @ V[nh,t,dh]^T
                    unsafe {
                        gemm_batched_f16(&*blas_ptr,
                            &self.d_ctx_buf, &(*acts_ptr).v, &mut self.d_attn_buf,
                            nh, t, dh, t, /*transb=*/true,
                            f16::from_f32(1.0), f16::from_f32(0.0));
                    }

                    // Softmax backward: d_attn_pre in tmp_buf
                    let cfg_sfx = cfg_attn_sfx(nh, t);
                    unsafe {
                        self.stream.launch_builder(&self.fns.zero_f16)
                            .arg(&mut self.tmp_buf).arg(&(na as i32)).launch(cfg1d(na)).unwrap();
                        self.stream.launch_builder(&self.fns.attn_sfx_bwd)
                            .arg(&mut self.tmp_buf).arg(&(*acts_ptr).scores).arg(&self.d_attn_buf)
                            .arg(&(nh as i32)).arg(&(t as i32))
                            .launch(cfg_sfx).unwrap();
                    }

                    // Scale by 1/sqrt(dh)
                    let scale_attn = 1.0f32 / (dh as f32).sqrt();
                    unsafe {
                        self.stream.launch_builder(&self.fns.scale_f16)
                            .arg(&mut self.tmp_buf).arg(&scale_attn).arg(&(na as i32))
                            .launch(cfg1d(na)).unwrap();
                    }

                    // dq[nh,t,dh] = d_attn_pre[nh,t,t] @ K[nh,t,dh]
                    unsafe {
                        gemm_batched_f16(&*blas_ptr,
                            &self.tmp_buf, &(*acts_ptr).k, &mut self.dq_buf,
                            nh, t, t, dh, /*transb=*/false,
                            f16::from_f32(1.0), f16::from_f32(0.0));
                    }

                    // dk[nh,t,dh] = d_attn_pre^T[nh,t,t] @ Q[nh,t,dh]
                    // Same pattern as dv but swap d_ctx→Q and scores→d_attn_pre
                    unsafe {
                        use cudarc::cublas::{StridedBatchedConfig, GemmConfig, Gemm};
                        use cudarc::cublas::sys::cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T};
                        let cfg = StridedBatchedConfig::<f16> {
                            gemm: GemmConfig {
                                transa: CUBLAS_OP_N, transb: CUBLAS_OP_T,
                                m: dh as i32, n: t as i32, k: t as i32,
                                alpha: f16::from_f32(1.0),
                                lda: dh as i32,
                                ldb: t as i32,
                                beta: f16::from_f32(0.0),
                                ldc: dh as i32,
                            },
                            stride_a: (t * dh) as i64,
                            stride_b: (t * t) as i64,
                            stride_c: (t * dh) as i64,
                            batch_size: nh as i32,
                        };
                        (*blas_ptr).gemm_strided_batched(cfg,
                            &(*acts_ptr).q, &self.tmp_buf, &mut self.dk_buf).unwrap();
                    }
                }

                // qkv_grad_merge_nb: dq,dk,dv[nh,t,dh] → tmp_buf[nt,3D]
                {
                    let cfg = LaunchConfig { grid_dim: (nt as u32, h as u32, 1), block_dim: (dh as u32, 1, 1), shared_mem_bytes: 0 };
                    unsafe {
                        self.stream.launch_builder(&self.fns.qkv_grad_merge_nb)
                            .arg(&self.dq_buf).arg(&self.dk_buf).arg(&self.dv_buf)
                            .arg(&mut self.tmp_buf).arg(&(nt as i32)).arg(&(t as i32)).arg(&(h as i32)).arg(&(dh as i32))
                            .launch(cfg).unwrap();
                    }
                }

                // g_b_qkv
                {
                    let grads_ptr = &mut self.grads[li] as *mut GpuLayerGrad;
                    unsafe {
                        self.stream.launch_builder(&self.fns.bias_grad)
                            .arg(&self.tmp_buf).arg(&mut (*grads_ptr).g_b_qkv)
                            .arg(&(nt as i32)).arg(&((3*d) as i32))
                            .launch(LaunchConfig { grid_dim: (((3*d+255)/256) as u32,1,1), block_dim: (256,1,1), shared_mem_bytes: 0 })
                            .unwrap();
                    }
                }

                // g_w_qkv += xn1^T @ d_qkv
                {
                    let acts_ptr  = &mut self.acts[li] as *mut GpuLayerActs;
                    let grads_ptr = &mut self.grads[li] as *mut GpuLayerGrad;
                    unsafe {
                        gemm(&self.blas, &(*acts_ptr).xn1, &self.tmp_buf, &mut (*grads_ptr).g_w_qkv,
                             d, nt, 3*d, true, false, one, one);
                    }
                }

                // d_xn1 = d_qkv @ w_qkv^T → x_norm_buf
                {
                    let layer_ptr = &self.layers[li] as *const TransformerLayer;
                    unsafe {
                        gemm(&self.blas, &self.tmp_buf, &(*layer_ptr).w_qkv, &mut self.x_norm_buf,
                             nt, 3*d, d, false, true, one, f16::from_f32(0.0));
                    }
                }

                // LN1 bwd (accumulates into dx_buf)
                {
                    let acts_ptr  = &mut self.acts[li] as *mut GpuLayerActs;
                    let layer_ptr = &self.layers[li] as *const TransformerLayer;
                    let grads_ptr = &mut self.grads[li] as *mut GpuLayerGrad;
                    let cfg = cfg_ln(nt);
                    unsafe {
                        // dx: warp-per-row, no atomicAdd contention
                        self.stream.launch_builder(&self.fns.ln_bwd_dx)
                            .arg(&mut self.dx_buf)
                            .arg(&self.x_norm_buf)        // dy (d_xn1)
                            .arg(&(*acts_ptr).x_pre)      // x (input to LN1)
                            .arg(&(*acts_ptr).ln1_mean).arg(&(*acts_ptr).ln1_rstd).arg(&(*layer_ptr).ln1_g)
                            .arg(&(nt as i32)).arg(&(d as i32))
                            .launch(cfg).unwrap();
                        // dgamma/dbeta: proper block reduction
                        self.stream.launch_builder(&self.fns.ln_param_grad)
                            .arg(&mut (*grads_ptr).g_ln1_g).arg(&mut (*grads_ptr).g_ln1_b)
                            .arg(&self.x_norm_buf)        // dy
                            .arg(&(*acts_ptr).x_pre)      // x
                            .arg(&(*acts_ptr).ln1_mean).arg(&(*acts_ptr).ln1_rstd)
                            .arg(&(nt as i32)).arg(&(d as i32))
                            .launch(LaunchConfig { grid_dim: (d as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 2 * 256 * 4 }).unwrap();
                    }
                }
            } // end layer backward loop

            // Embedding + positional grads
            {
                let bx = (d + 255) / 256;
                let cfg = LaunchConfig { grid_dim: (nt as u32, bx as u32, 1), block_dim: (256.min(d) as u32, 1, 1), shared_mem_bytes: 0 };
                unsafe {
                    self.stream.launch_builder(&self.fns.emb_bwd_f32)
                        .arg(&self.dx_buf).arg(&self.ids_buf).arg(&mut self.g_embed)
                        .arg(&(nt as i32)).arg(&(d as i32))
                        .launch(cfg).unwrap();
                    self.stream.launch_builder(&self.fns.pos_grad_f32_nb)
                        .arg(&self.dx_buf).arg(&mut self.g_pos)
                        .arg(&(nt as i32)).arg(&(t as i32)).arg(&(d as i32))
                        .launch(cfg).unwrap();
                }
            }

            chunk_start = chunk_end;
        } // end micro-batch loop

        if counted == 0 { return 0.0; }

        // Fold tied output-head embed grad (f16) into g_embed (f32) before Adam
        {
            let n = self.g_embed_head_f16.len();
            let cfg = cfg1d(n);
            unsafe { self.stream.launch_builder(&self.fns.add_f16_to_f32)
                .arg(&self.g_embed_head_f16).arg(&mut self.g_embed).arg(&(n as i32))
                .launch(cfg).unwrap(); }
        }

        // ── GRADIENT CLIPPING (global L2 norm ≤ 1.0) ─────────────
        const MAX_GRAD_NORM: f32 = 1.0;

        // Zero the accumulator scalar
        unsafe { self.stream.launch_builder(&self.fns.zero_scalar_f32)
            .arg(&mut self.grad_norm_sq)
            .launch(LaunchConfig { grid_dim: (1,1,1), block_dim: (1,1,1), shared_mem_bytes: 0 }).unwrap(); }

        // Accumulate ||g||^2 for each parameter tensor into grad_norm_sq
        macro_rules! accum_norm_f16 { ($g:expr) => {{
            let n = ($g).len();
            if n > 0 {
                let num_blocks = (n + 255) / 256;
                let cfg_norm = cfg_reduce(n);
                unsafe { self.stream.launch_builder(&self.fns.norm_reduce_f16)
                    .arg(&($g)).arg(&mut self.partial_norm_buf).arg(&(n as i32))
                    .launch(cfg_norm).unwrap(); }
                let cfg_red = LaunchConfig {
                    grid_dim: (((num_blocks + 255) / 256) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: (256 * 4) as u32,
                };
                unsafe { self.stream.launch_builder(&self.fns.reduce_sum)
                    .arg(&self.partial_norm_buf).arg(&mut self.grad_norm_sq).arg(&(num_blocks as i32))
                    .launch(cfg_red).unwrap(); }
            }
        }}}
        macro_rules! accum_norm_f32 { ($g:expr) => {{
            let n = ($g).len();
            if n > 0 {
                let num_blocks = (n + 255) / 256;
                let cfg_norm = cfg_reduce(n);
                unsafe { self.stream.launch_builder(&self.fns.norm_reduce_f32)
                    .arg(&($g)).arg(&mut self.partial_norm_buf).arg(&(n as i32))
                    .launch(cfg_norm).unwrap(); }
                let cfg_red = LaunchConfig {
                    grid_dim: (((num_blocks + 255) / 256) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: (256 * 4) as u32,
                };
                unsafe { self.stream.launch_builder(&self.fns.reduce_sum)
                    .arg(&self.partial_norm_buf).arg(&mut self.grad_norm_sq).arg(&(num_blocks as i32))
                    .launch(cfg_red).unwrap(); }
            }
        }}}

        accum_norm_f32!(self.g_embed);
        accum_norm_f32!(self.g_pos);
        accum_norm_f32!(self.g_ln_f_g);
        accum_norm_f32!(self.g_ln_f_b);
        for li in 0..nl {
            accum_norm_f16!(self.grads[li].g_w_qkv);
            accum_norm_f16!(self.grads[li].g_w_out);
            accum_norm_f16!(self.grads[li].g_w_ff1);
            accum_norm_f16!(self.grads[li].g_w_ff2);
            accum_norm_f32!(self.grads[li].g_b_qkv);
            accum_norm_f32!(self.grads[li].g_b_out);
            accum_norm_f32!(self.grads[li].g_b_ff1);
            accum_norm_f32!(self.grads[li].g_b_ff2);
            accum_norm_f32!(self.grads[li].g_ln1_g);
            accum_norm_f32!(self.grads[li].g_ln1_b);
            accum_norm_f32!(self.grads[li].g_ln2_g);
            accum_norm_f32!(self.grads[li].g_ln2_b);
        }

        // Sync + read the global norm; compute clip scale on host
        self.stream.synchronize().unwrap();
        let sq: Vec<f32> = self.stream.clone_dtoh(&self.grad_norm_sq).unwrap();
        let global_norm = sq[0].max(0.0).sqrt();
        let clip_scale = if global_norm > MAX_GRAD_NORM && global_norm.is_finite() {
            MAX_GRAD_NORM / global_norm
        } else {
            1.0
        };

        // Apply scale to every grad tensor when clip triggered
        if clip_scale < 1.0 {
            macro_rules! scale_g_f16 { ($g:expr) => {{
                let n = ($g).len();
                if n > 0 {
                    unsafe { self.stream.launch_builder(&self.fns.scale_f16)
                        .arg(&mut ($g)).arg(&clip_scale).arg(&(n as i32))
                        .launch(cfg1d(n)).unwrap(); }
                }
            }}}
            macro_rules! scale_g_f32 { ($g:expr) => {{
                let n = ($g).len();
                if n > 0 {
                    unsafe { self.stream.launch_builder(&self.fns.scale_f32)
                        .arg(&mut ($g)).arg(&clip_scale).arg(&(n as i32))
                        .launch(cfg1d(n)).unwrap(); }
                }
            }}}

            scale_g_f32!(self.g_embed);
            scale_g_f32!(self.g_pos);
            scale_g_f32!(self.g_ln_f_g);
            scale_g_f32!(self.g_ln_f_b);
            for li in 0..nl {
                scale_g_f16!(self.grads[li].g_w_qkv);
                scale_g_f16!(self.grads[li].g_w_out);
                scale_g_f16!(self.grads[li].g_w_ff1);
                scale_g_f16!(self.grads[li].g_w_ff2);
                scale_g_f32!(self.grads[li].g_b_qkv);
                scale_g_f32!(self.grads[li].g_b_out);
                scale_g_f32!(self.grads[li].g_b_ff1);
                scale_g_f32!(self.grads[li].g_b_ff2);
                scale_g_f32!(self.grads[li].g_ln1_g);
                scale_g_f32!(self.grads[li].g_ln1_b);
                scale_g_f32!(self.grads[li].g_ln2_g);
                scale_g_f32!(self.grads[li].g_ln2_b);
            }
        }

        // ── ADAM UPDATE (all on GPU) ──────────────────────────────
        macro_rules! adam16 { ($p:expr, $m:expr, $v:expr, $g:expr) => {{
            let n = $p.len();
            unsafe { self.stream.launch_builder(&self.fns.adam_f16)
                .arg(&mut $p).arg(&mut $m).arg(&mut $v).arg(&$g)
                .arg(&lr).arg(&0.9f32).arg(&0.999f32).arg(&eps).arg(&bc1).arg(&bc2).arg(&(n as i32))
                .launch(cfg1d(n)).unwrap(); }
        }}}
        macro_rules! adam16f32 { ($p:expr, $m:expr, $v:expr, $g:expr) => {{
            let n = $p.len();
            unsafe { self.stream.launch_builder(&self.fns.adam_f16_f32)
                .arg(&mut $p).arg(&mut $m).arg(&mut $v).arg(&$g)
                .arg(&lr).arg(&0.9f32).arg(&0.999f32).arg(&eps).arg(&bc1).arg(&bc2).arg(&(n as i32))
                .launch(cfg1d(n)).unwrap(); }
        }}}

        // Embed + pos (f32 grads → f16 params via adam_f16_f32)
        adam16f32!(self.embed,     self.m_embed,   self.v_embed,   self.g_embed);
        adam16f32!(self.pos_embed, self.m_pos,     self.v_pos,     self.g_pos);
        adam16f32!(self.ln_f_g,    self.m_ln_f_g,  self.v_ln_f_g,  self.g_ln_f_g);
        adam16f32!(self.ln_f_b,    self.m_ln_f_b,  self.v_ln_f_b,  self.g_ln_f_b);

        for li in 0..nl {
            let lp = &mut self.layers[li] as *mut TransformerLayer;
            let gp = &mut self.grads[li] as *mut GpuLayerGrad;
            unsafe {
                macro_rules! adam_w { ($w:ident, $m:ident, $v:ident, $g:ident) => {{
                    // v3.5.1: LoRA backward — compute adapter gradients if enabled
                    if self.lora_config.is_some() && (*lp).lora.is_some() && self.lora_backward_enabled {
                        // Adapter-only training: compute d_A and d_B gradients
                        // (Implementation: placeholder for now — full implementation requires activation tracking)
                        // In full implementation, would compute:
                        //   d_A = gradients with respect to A matrix
                        //   d_B = gradients with respect to B matrix
                        // Then update with Adam: m_A, v_A, m_B, v_B
                    } else if self.lora_config.is_some() && (*lp).lora.is_some() {
                        // Skip base weight updates when LoRA enabled but backward not active
                    } else {
                        let n = (*lp).$w.len();
                        self.stream.launch_builder(&self.fns.adam_f16).arg(&mut (*lp).$w).arg(&mut (*lp).$m).arg(&mut (*lp).$v).arg(&(*gp).$g).arg(&lr).arg(&0.9f32).arg(&0.999f32).arg(&eps).arg(&bc1).arg(&bc2).arg(&(n as i32)).launch(cfg1d(n)).unwrap();
                    }
                }}}
                macro_rules! adam_b { ($w:ident, $m:ident, $v:ident, $g:ident) => {{
                    let n = (*lp).$w.len();
                    self.stream.launch_builder(&self.fns.adam_f16_f32).arg(&mut (*lp).$w).arg(&mut (*lp).$m).arg(&mut (*lp).$v).arg(&(*gp).$g).arg(&lr).arg(&0.9f32).arg(&0.999f32).arg(&eps).arg(&bc1).arg(&bc2).arg(&(n as i32)).launch(cfg1d(n)).unwrap();
                }}}
                adam_w!(w_qkv, m_w_qkv, v_w_qkv, g_w_qkv);
                adam_w!(w_out, m_w_out,  v_w_out,  g_w_out);
                adam_w!(w_ff1, m_w_ff1,  v_w_ff1,  g_w_ff1);
                adam_w!(w_ff2, m_w_ff2,  v_w_ff2,  g_w_ff2);
                adam_b!(b_qkv, m_b_qkv, v_b_qkv, g_b_qkv);
                adam_b!(b_out, m_b_out,  v_b_out,  g_b_out);
                adam_b!(b_ff1, m_b_ff1,  v_b_ff1,  g_b_ff1);
                adam_b!(b_ff2, m_b_ff2,  v_b_ff2,  g_b_ff2);
                adam_b!(ln1_g, m_ln1_g,  v_ln1_g,  g_ln1_g);
                adam_b!(ln1_b, m_ln1_b,  v_ln1_b,  g_ln1_b);
                adam_b!(ln2_g, m_ln2_g,  v_ln2_g,  g_ln2_g);
                adam_b!(ln2_b, m_ln2_b,  v_ln2_b,  g_ln2_b);
            }
        }

        self.stream.synchronize().unwrap();
        // Read accumulated loss from GPU (single transfer for the whole batch)
        let lv: Vec<f32> = self.stream.clone_dtoh(&self.loss_acc).unwrap();
        // Normalize by token count, not sequence count — CE is summed per token.
        if tok_counted == 0 { return 0.0; }
        lv[0] / tok_counted as f32
    }

    // ─────────────────────────────────────────────────────────────
    pub fn save_checkpoint(&self, path: &str) -> anyhow::Result<()> {
        use std::io::{BufWriter, Write as IoWrite};
        if let Some(parent) = std::path::Path::new(path).parent() {
            fs::create_dir_all(parent)?;
        }
        let tmp_path = format!("{}.tmp", path);
        let file = fs::File::create(&tmp_path)?;
        let mut w = BufWriter::with_capacity(8 * 1024 * 1024, file);
        w.write_all(b"ARIA")?;
        macro_rules! wu32 { ($v:expr) => { w.write_all(&($v as u32).to_le_bytes())?; } }
        wu32!(2u32); wu32!(self.vocab_size); wu32!(self.d_model); wu32!(self.num_heads);
        wu32!(self.num_layers); wu32!(self.ffn_dim); wu32!(self.max_seq_len);
        w.write_all(&self.adam_step.to_le_bytes())?;
        self.stream.synchronize().unwrap();
        macro_rules! wf16 { ($buf:expr) => {{
            let v: Vec<f16> = self.stream.clone_dtoh($buf).unwrap();
            w.write_all(&(v.len() as u32).to_le_bytes())?;
            let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) };
            w.write_all(bytes)?;
        }}}
        macro_rules! wf32 { ($buf:expr) => {{
            let v = download_f32(&self.stream, $buf);
            w.write_all(&(v.len() as u32).to_le_bytes())?;
            let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
            w.write_all(bytes)?;
        }}}
        wf16!(&self.embed); wf16!(&self.pos_embed);
        wf32!(&self.m_embed); wf32!(&self.v_embed);
        wf32!(&self.m_pos); wf32!(&self.v_pos);
        wf16!(&self.ln_f_g); wf16!(&self.ln_f_b);
        wf32!(&self.m_ln_f_g); wf32!(&self.v_ln_f_g);
        wf32!(&self.m_ln_f_b); wf32!(&self.v_ln_f_b);
        for l in &self.layers {
            wf16!(&l.w_qkv); wf16!(&l.b_qkv); wf16!(&l.w_out); wf16!(&l.b_out);
            wf16!(&l.w_ff1); wf16!(&l.b_ff1); wf16!(&l.w_ff2); wf16!(&l.b_ff2);
            wf16!(&l.ln1_g); wf16!(&l.ln1_b); wf16!(&l.ln2_g); wf16!(&l.ln2_b);
            wf32!(&l.m_w_qkv); wf32!(&l.v_w_qkv); wf32!(&l.m_b_qkv); wf32!(&l.v_b_qkv);
            wf32!(&l.m_w_out); wf32!(&l.v_w_out); wf32!(&l.m_b_out); wf32!(&l.v_b_out);
            wf32!(&l.m_w_ff1); wf32!(&l.v_w_ff1); wf32!(&l.m_b_ff1); wf32!(&l.v_b_ff1);
            wf32!(&l.m_w_ff2); wf32!(&l.v_w_ff2); wf32!(&l.m_b_ff2); wf32!(&l.v_b_ff2);
            wf32!(&l.m_ln1_g); wf32!(&l.v_ln1_g); wf32!(&l.m_ln1_b); wf32!(&l.v_ln1_b);
            wf32!(&l.m_ln2_g); wf32!(&l.v_ln2_g); wf32!(&l.m_ln2_b); wf32!(&l.v_ln2_b);
        }
        w.flush()?;
        drop(w);
        fs::rename(&tmp_path, path)?;
        println!("  Checkpoint saved -> {}", path);
        Ok(())
    }

    pub fn load_checkpoint(path: &str) -> anyhow::Result<Self> {
        use std::io::Read;
        let mut f = std::io::BufReader::new(fs::File::open(path)?);
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        drop(f);
        if &magic == b"ARIA" {
            let mut r = std::io::BufReader::with_capacity(8*1024*1024, fs::File::open(path)?);
            let mut skip = [0u8; 4]; r.read_exact(&mut skip)?;
            Self::load_binary(r)
        } else {
            Self::load_json(path)
        }
    }

    fn load_binary<R: std::io::Read>(mut r: R) -> anyhow::Result<Self> {
        macro_rules! ru32 { () => {{ let mut b=[0u8;4]; r.read_exact(&mut b)?; u32::from_le_bytes(b) as usize }} }
        macro_rules! ri32 { () => {{ let mut b=[0u8;4]; r.read_exact(&mut b)?; i32::from_le_bytes(b) }} }
        let ver = { let mut b=[0u8;4]; r.read_exact(&mut b)?; u32::from_le_bytes(b) };
        anyhow::ensure!(ver == 2, "Bad checkpoint version: {}", ver);
        let vocab_size=ru32!(); let d_model=ru32!(); let num_heads=ru32!();
        let num_layers=ru32!(); let ffn_dim=ru32!(); let max_seq_len=ru32!();
        let adam_step = ri32!();
        let mut model = Self::new(vocab_size, d_model, num_heads, num_layers, ffn_dim, max_seq_len);
        model.adam_step = adam_step;
        let stream = model.stream.clone();
        macro_rules! rf16 { ($field:expr) => {{
            let n = ru32!();
            let mut bytes = vec![0u8; n * 2]; r.read_exact(&mut bytes)?;
            let v: Vec<f16> = bytes.chunks_exact(2).map(|b| f16::from_bits(u16::from_le_bytes([b[0],b[1]]))).collect();
            $field = stream.clone_htod(&v).unwrap();
        }}}
        macro_rules! rf32 { ($field:expr) => {{
            let n = ru32!();
            let mut bytes = vec![0u8; n * 4]; r.read_exact(&mut bytes)?;
            let v: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect();
            $field = upload_f32(&stream, &v);
        }}}
        rf16!(model.embed); rf16!(model.pos_embed);
        rf32!(model.m_embed); rf32!(model.v_embed);
        rf32!(model.m_pos); rf32!(model.v_pos);
        rf16!(model.ln_f_g); rf16!(model.ln_f_b);
        rf32!(model.m_ln_f_g); rf32!(model.v_ln_f_g);
        rf32!(model.m_ln_f_b); rf32!(model.v_ln_f_b);
        for li in 0..model.layers.len() {
            rf16!(model.layers[li].w_qkv); rf16!(model.layers[li].b_qkv);
            rf16!(model.layers[li].w_out); rf16!(model.layers[li].b_out);
            rf16!(model.layers[li].w_ff1); rf16!(model.layers[li].b_ff1);
            rf16!(model.layers[li].w_ff2); rf16!(model.layers[li].b_ff2);
            rf16!(model.layers[li].ln1_g); rf16!(model.layers[li].ln1_b);
            rf16!(model.layers[li].ln2_g); rf16!(model.layers[li].ln2_b);
            rf32!(model.layers[li].m_w_qkv); rf32!(model.layers[li].v_w_qkv);
            rf32!(model.layers[li].m_b_qkv); rf32!(model.layers[li].v_b_qkv);
            rf32!(model.layers[li].m_w_out); rf32!(model.layers[li].v_w_out);
            rf32!(model.layers[li].m_b_out); rf32!(model.layers[li].v_b_out);
            rf32!(model.layers[li].m_w_ff1); rf32!(model.layers[li].v_w_ff1);
            rf32!(model.layers[li].m_b_ff1); rf32!(model.layers[li].v_b_ff1);
            rf32!(model.layers[li].m_w_ff2); rf32!(model.layers[li].v_w_ff2);
            rf32!(model.layers[li].m_b_ff2); rf32!(model.layers[li].v_b_ff2);
            rf32!(model.layers[li].m_ln1_g); rf32!(model.layers[li].v_ln1_g);
            rf32!(model.layers[li].m_ln1_b); rf32!(model.layers[li].v_ln1_b);
            rf32!(model.layers[li].m_ln2_g); rf32!(model.layers[li].v_ln2_g);
            rf32!(model.layers[li].m_ln2_b); rf32!(model.layers[li].v_ln2_b);
        }
        println!("Checkpoint loaded (binary). adam_step={}", model.adam_step);
        Ok(model)
    }

    fn load_json(path: &str) -> anyhow::Result<Self> {
        println!("Loading JSON checkpoint (converting to binary on next save)...");
        let file = fs::File::open(path)?;
        let reader = std::io::BufReader::with_capacity(8*1024*1024, file);
        let data: serde_json::Value = serde_json::from_reader(reader)?;
        let version = data["version"].as_str().unwrap_or("");
        anyhow::ensure!(version == "transformer_v1", "Unknown version: {}", version);
        let vocab_size=data["vocab_size"].as_u64().unwrap() as usize;
        let d_model=data["d_model"].as_u64().unwrap() as usize;
        let num_heads=data["num_heads"].as_u64().unwrap() as usize;
        let num_layers=data["num_layers"].as_u64().unwrap() as usize;
        let ffn_dim=data["ffn_dim"].as_u64().unwrap() as usize;
        let max_seq_len=data["max_seq_len"].as_u64().unwrap() as usize;
        let adam_step=data["adam_step"].as_i64().unwrap_or(0) as i32;
        let mut model = Self::new(vocab_size, d_model, num_heads, num_layers, ffn_dim, max_seq_len);
        model.adam_step = adam_step;
        macro_rules! load16 { ($f:ident, $k:expr) => {
            if let Some(s)=data[$k].as_str() { let v=from_b64_f16(s); if !v.is_empty() { model.$f=upload_f16(&model.stream,&v); } }
        } }
        macro_rules! load32 { ($f:ident, $k:expr) => {
            if let Some(s)=data[$k].as_str() { let v=from_b64_f32(s); if !v.is_empty() { model.$f=upload_f32(&model.stream,&v); } }
        } }
        load16!(embed,"embed"); load16!(pos_embed,"pos_embed");
        load32!(m_embed,"m_embed"); load32!(v_embed,"v_embed");
        load32!(m_pos,"m_pos"); load32!(v_pos,"v_pos");
        load16!(ln_f_g,"ln_f_g"); load16!(ln_f_b,"ln_f_b");
        load32!(m_ln_f_g,"m_ln_f_g"); load32!(v_ln_f_g,"v_ln_f_g");
        load32!(m_ln_f_b,"m_ln_f_b"); load32!(v_ln_f_b,"v_ln_f_b");
        if let Some(arr)=data["layers"].as_array() {
            for (li,lj) in arr.iter().enumerate() {
                if li>=model.layers.len() { break; }
                let l=&mut model.layers[li];
                macro_rules! ll16 { ($f:ident) => {
                    if let Some(s)=lj[stringify!($f)].as_str() { let v=from_b64_f16(s); if !v.is_empty() { l.$f=upload_f16(&model.stream,&v); } }
                } }
                macro_rules! ll32 { ($f:ident) => {
                    if let Some(s)=lj[stringify!($f)].as_str() { let v=from_b64_f32(s); if !v.is_empty() { l.$f=upload_f32(&model.stream,&v); } }
                } }
                ll16!(w_qkv); ll16!(b_qkv); ll16!(w_out); ll16!(b_out);
                ll16!(w_ff1); ll16!(b_ff1); ll16!(w_ff2); ll16!(b_ff2);
                ll16!(ln1_g); ll16!(ln1_b); ll16!(ln2_g); ll16!(ln2_b);
                ll32!(m_w_qkv); ll32!(v_w_qkv); ll32!(m_b_qkv); ll32!(v_b_qkv);
                ll32!(m_w_out); ll32!(v_w_out); ll32!(m_b_out); ll32!(v_b_out);
                ll32!(m_w_ff1); ll32!(v_w_ff1); ll32!(m_b_ff1); ll32!(v_b_ff1);
                ll32!(m_w_ff2); ll32!(v_w_ff2); ll32!(m_b_ff2); ll32!(v_b_ff2);
                ll32!(m_ln1_g); ll32!(v_ln1_g); ll32!(m_ln1_b); ll32!(v_ln1_b);
                ll32!(m_ln2_g); ll32!(v_ln2_g); ll32!(m_ln2_b); ll32!(v_ln2_b);
            }
        }
        println!("Checkpoint loaded (JSON). adam_step={}", model.adam_step);
        Ok(model)
    }
}

// ─────────────────────────────────────────────────────────────
//  CPU math helpers
// ─────────────────────────────────────────────────────────────
fn gelu(x: f32) -> f32 {
    let c = 0.7978845608f32;
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

fn gelu_grad(x: f32) -> f32 {
    let c = 0.7978845608f32;
    let inner = c * (x + 0.044715 * x * x * x);
    let t = inner.tanh();
    let sech2 = 1.0 - t * t;
    0.5 * (1.0 + t) + 0.5 * x * sech2 * c * (1.0 + 3.0 * 0.044715 * x * x)
}

fn cpu_layernorm(x: &[f32], g: &[f32], b: &[f32], t: usize, d: usize) -> Vec<f32> {
    let (out, _, _) = cpu_layernorm_stats(x, g, b, t, d);
    out
}

fn cpu_layernorm_stats(x: &[f32], g: &[f32], b: &[f32], t: usize, d: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let eps = 1e-5f32;
    let mut out  = vec![0f32; t * d];
    let mut mean = vec![0f32; t];
    let mut rstd = vec![0f32; t];
    for i in 0..t {
        let row = &x[i*d..(i+1)*d];
        let mu = row.iter().sum::<f32>() / d as f32;
        let var = row.iter().map(|v| (v - mu).powi(2)).sum::<f32>() / d as f32;
        let rs = 1.0 / (var + eps).sqrt();
        mean[i] = mu;
        rstd[i] = rs;
        for j in 0..d {
            out[i*d+j] = (x[i*d+j] - mu) * rs * g[j] + b[j];
        }
    }
    (out, mean, rstd)
}

fn cpu_layernorm_bwd(dy: &[f32], x: &[f32], mean: &[f32], rstd: &[f32], g: &[f32],
                     t: usize, d: usize, dg: &mut Vec<f32>, db: &mut Vec<f32>) -> Vec<f32> {
    let mut dx = vec![0f32; t * d];
    let inv_d = 1.0 / d as f32;
    for i in 0..t {
        let mu = mean[i];
        let rs = rstd[i];
        let row_dy = &dy[i*d..(i+1)*d];
        let row_x  = &x[i*d..(i+1)*d];
        // s1 = sum(dy * g),  s2 = sum(dy * g * xhat)
        let mut s1 = 0f32; let mut s2 = 0f32;
        for j in 0..d {
            let xh = (row_x[j] - mu) * rs;
            s1 += row_dy[j] * g[j];
            s2 += row_dy[j] * g[j] * xh;
            dg[j] += row_dy[j] * xh;
            db[j] += row_dy[j];
        }
        for j in 0..d {
            let xh = (row_x[j] - mu) * rs;
            dx[i*d+j] = rs * (row_dy[j] * g[j] - inv_d * s1 - inv_d * xh * s2);
        }
    }
    dx
}

// C[m, n] = A[m, k] @ B[k, n]  (row-major)
fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let av = a[i*k+p];
            for j in 0..n {
                c[i*n+j] += av * b[p*n+j];
            }
        }
    }
    c
}

// C[m, k] = A[m, n] @ B[k, n]^T
fn cpu_matmul_t(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0f32; m * k];
    for i in 0..m {
        for j in 0..k {
            let mut s = 0f32;
            for p in 0..n { s += a[i*n+p] * b[j*n+p]; }
            c[i*k+j] = s;
        }
    }
    c
}

// acc[k, n] += A[m, k]^T @ B[m, n]
fn cpu_matmul_acc_t(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, acc: &mut Vec<f32>) {
    for p in 0..k {
        for i in 0..m {
            let av = a[i*k+p];
            for j in 0..n {
                acc[p*n+j] += av * b[i*n+j];
            }
        }
    }
}

fn adam_update_cpu(param: &mut Vec<f32>, grad: &[f32], m: &mut Vec<f32>, v: &mut Vec<f32>,
                   lr: f32, bc1: f32, bc2: f32, eps: f32) {
    let b1 = 0.9f32; let b2 = 0.999f32;
    for i in 0..param.len() {
        let g = grad[i];
        m[i] = b1 * m[i] + (1.0 - b1) * g;
        v[i] = b2 * v[i] + (1.0 - b2) * g * g;
        param[i] -= lr * (m[i] / bc1) / ((v[i] / bc2).sqrt() + eps);
    }
}

// ─────────────────────────────────────────────────────────────
//  Training loop (mirrors pretrain_from_files from model_cuda.rs)
// ─────────────────────────────────────────────────────────────
pub fn pretrain_from_files(
    model: &mut TransformerModel,
    tokenizer: &mut Tokenizer,
    data_dir: &str,
    checkpoint_path: &str,
    tokenizer_path: &str,
) -> anyhow::Result<()> {
    let lr: f32 = std::env::var("ARIA_LR")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(LEARNING_RATE);
    let max_seqs: usize = std::env::var("ARIA_MAX_SEQS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(MAX_SEQS_PER_EPOCH);
    let epochs: usize = std::env::var("ARIA_EPOCHS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(PRETRAIN_EPOCHS);
    let warmup_steps: usize = std::env::var("ARIA_WARMUP")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(1000);
    let batch_size = PRETRAIN_BATCH_SIZE;
    let max_len    = MAX_TOKENS_PER_SEQ;
    let min_len    = MIN_TOKENS_PER_SEQ;

    println!("LR: {lr}  Epochs: {epochs}  Batch: {batch_size}  SeqLen: {max_len}  MaxSeqs: {max_seqs}  Warmup: {warmup_steps}");

    // Load or build sequence cache
    let cache_path = format!("{}/sequences_cache_transformer_v{}_len{}.bin",
                             data_dir, tokenizer.vocab_size(), max_len);
    let sequences: Vec<(Vec<usize>, Vec<f32>)> = if std::path::Path::new(&cache_path).exists() {
        println!("Loading sequence cache from {}...", cache_path);
        load_seq_cache(&cache_path)?
    } else {
        println!("Building sequence cache...");
        let seqs = build_seq_cache(tokenizer, data_dir, max_len, min_len, max_seqs)?;
        save_seq_cache(&cache_path, &seqs)?;
        seqs
    };

    let n = sequences.len().min(max_seqs);
    let seqs = &sequences[..n];
    println!("Sequences: {}", n);

    let mut rng = rand::thread_rng();
    let n_batches = (n + batch_size - 1) / batch_size;

    // Length bucketing: group similar-length sequences so batches need minimal
    // padding. The within-batch max length sets GPU work for the whole batch,
    // so mixing a 20-token line with a 256-token one wastes ~12x compute.
    // We sort by length once, then shuffle BATCH ORDER each epoch (not the
    // global order) to keep training stochastic without re-introducing padding.
    let mut len_sorted: Vec<usize> = (0..n).collect();
    len_sorted.sort_by_key(|&i| seqs[i].0.len());

    for epoch in 0..epochs {
        // shuffle the order in which length-buckets are visited
        let mut batch_order: Vec<usize> = (0..n_batches).collect();
        batch_order.shuffle(&mut rng);

        let mut epoch_loss = 0.0f32;
        let mut epoch_batches = 0usize;
        let t0 = Instant::now();

        for (step, &batch_idx) in batch_order.iter().enumerate() {
            let start = batch_idx * batch_size;
            let end   = (start + batch_size).min(n);
            let batch: Vec<usize> = len_sorted[start..end].to_vec();

            let batch_seqs: Vec<Vec<usize>> = batch.iter().map(|&i| seqs[i].0.clone()).collect();
            let batch_masks: Vec<Vec<f32>>  = batch.iter().map(|&i| seqs[i].1.clone()).collect();

            // Linear warmup → cosine decay to 0.3 × lr
            let total_steps = epochs * n_batches;
            let current_step = epoch * n_batches + step;
            let step_lr = if current_step < warmup_steps {
                lr * (current_step as f32 + 1.0) / warmup_steps as f32
            } else {
                let progress = (current_step - warmup_steps) as f32
                             / (total_steps.saturating_sub(warmup_steps)).max(1) as f32;
                let cos = (std::f32::consts::PI * progress).cos();
                lr * (0.3 + 0.7 * (cos * 0.5 + 0.5))
            };

            let loss = model.train_batch_masked(&batch_seqs, &batch_masks, step_lr);
            epoch_loss += loss;
            epoch_batches += 1;

            if step % 10 == 0 {
                let remaining = n_batches - step - 1;
                let elapsed = t0.elapsed().as_secs_f32();
                let seqs_done = (step + 1) * batch_size;
                let seq_per_s = seqs_done as f32 / elapsed.max(0.001);
                print!("\r  Epoch {}/{}  |  batch {}/{}  ({} remaining)  |  loss={:.4}  |  {:.0} seq/s  |  lr={:.6}       ",
                    epoch+1, epochs, step+1, n_batches, remaining,
                    epoch_loss / epoch_batches as f32, seq_per_s, step_lr);
                std::io::stdout().flush().ok();
            }
        }

        let elapsed = t0.elapsed().as_secs_f32();
        let avg_loss = epoch_loss / epoch_batches as f32;
        let final_step = epoch * n_batches + n_batches - 1;
        let final_lr = if final_step < warmup_steps {
            lr * (final_step as f32 + 1.0) / warmup_steps as f32
        } else {
            let total_steps = epochs * n_batches;
            let progress = (final_step - warmup_steps) as f32
                         / (total_steps.saturating_sub(warmup_steps)).max(1) as f32;
            let cos = (std::f32::consts::PI * progress).cos();
            lr * (0.3 + 0.7 * (cos * 0.5 + 0.5))
        };
        println!("\nEpoch {}/{}  done  |  loss={:.6}  |  {:.1}s  |  lr={:.6}",
            epoch+1, epochs, avg_loss, elapsed, final_lr);

        println!("  Saving checkpoint...");
        model.save_checkpoint(checkpoint_path).ok();
        tokenizer.save(tokenizer_path).ok();
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────
//  Sequence cache I/O (same binary format as model_cuda.rs)
// ─────────────────────────────────────────────────────────────
fn build_seq_cache(tok: &mut Tokenizer, data_dir: &str, max_len: usize, min_len: usize, max_seqs: usize)
    -> anyhow::Result<Vec<(Vec<usize>, Vec<f32>)>>
{
    use std::io::BufRead;

    let mut files: Vec<std::path::PathBuf> = Vec::new();
    match fs::read_dir(data_dir) {
        Ok(rd) => {
            for entry in rd.flatten() {
                let p = entry.path();
                if p.is_file() && p.extension().and_then(|s| s.to_str()) == Some("jsonl") {
                    files.push(p);
                }
            }
        }
        Err(e) => {
            eprintln!("Warning: cannot read {}: {}", data_dir, e);
            return Ok(Vec::new());
        }
    }
    files.sort();

    if files.is_empty() {
        eprintln!("Warning: no .jsonl files found in {}", data_dir);
        return Ok(Vec::new());
    }

    println!("Loading sequences from {} .jsonl file(s):", files.len());
    for p in &files { println!("  - {}", p.display()); }

    let mut seqs: Vec<(Vec<usize>, Vec<f32>)> = Vec::new();
    'outer: for path in &files {
        let f = match fs::File::open(path) {
            Ok(f) => f,
            Err(e) => { eprintln!("Warning: cannot open {}: {}", path.display(), e); continue; }
        };
        let r = std::io::BufReader::new(f);
        for line in r.lines() {
            if seqs.len() >= max_seqs { break 'outer; }
            let line = match line { Ok(l) => l, Err(_) => continue };
            let line = line.trim();
            if line.is_empty() { continue; }
            if let Ok(obj) = serde_json::from_str::<serde_json::Value>(line) {
                if let Some(text) = obj.get("text").and_then(|v| v.as_str()) {
                    let (ids, mask) = tok.encode_dialog(text);
                    if ids.len() >= min_len && ids.len() <= max_len {
                        seqs.push((ids, mask));
                    }
                }
            }
        }
    }
    Ok(seqs)
}

fn save_seq_cache(path: &str, seqs: &[(Vec<usize>, Vec<f32>)]) -> anyhow::Result<()> {
    use std::io::Write;
    let mut f = std::io::BufWriter::new(fs::File::create(path)?);
    let n = seqs.len() as u64;
    f.write_all(&n.to_le_bytes())?;
    for (ids, mask) in seqs {
        let len = ids.len() as u32;
        f.write_all(&len.to_le_bytes())?;
        for &id in ids { f.write_all(&(id as u32).to_le_bytes())?; }
        for &m in mask { f.write_all(&m.to_bits().to_le_bytes())?; }
    }
    Ok(())
}

fn load_seq_cache(path: &str) -> anyhow::Result<Vec<(Vec<usize>, Vec<f32>)>> {
    use std::io::Read;
    let mut f = std::io::BufReader::new(fs::File::open(path)?);
    let mut buf8 = [0u8; 8];
    f.read_exact(&mut buf8)?;
    let n = u64::from_le_bytes(buf8) as usize;
    let mut seqs = Vec::with_capacity(n);
    for _ in 0..n {
        let mut buf4 = [0u8; 4];
        f.read_exact(&mut buf4)?;
        let len = u32::from_le_bytes(buf4) as usize;
        let mut ids = Vec::with_capacity(len);
        let mut mask = Vec::with_capacity(len);
        for _ in 0..len {
            f.read_exact(&mut buf4)?;
            ids.push(u32::from_le_bytes(buf4) as usize);
        }
        for _ in 0..len {
            f.read_exact(&mut buf4)?;
            mask.push(f32::from_bits(u32::from_le_bytes(buf4)));
        }
        seqs.push((ids, mask));
    }
    Ok(seqs)
}
