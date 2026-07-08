/// LoRA (Low-Rank Adaptation) support for ARIA Transformer
/// Allows efficient fine-tuning with ~1% trainable parameters

use cudarc::driver::CudaSlice;
use half::f16;

/// LoRA configuration
#[derive(Clone, Debug)]
pub struct LoraConfig {
    pub rank: usize,          // LoRA rank (typically 8-16)
    pub alpha: f32,           // scaling factor (usually = rank)
    pub enabled: bool,
}

impl LoraConfig {
    pub fn new(rank: usize, alpha: f32) -> Self {
        LoraConfig {
            rank,
            alpha,
            enabled: true,
        }
    }

    pub fn scale(&self) -> f32 {
        if self.rank > 0 {
            self.alpha / self.rank as f32
        } else {
            1.0
        }
    }
}

/// LoRA adapters for a single Transformer layer
/// Structure: output = W(x) + scale * B(A(x))
/// where W is frozen (base model), A and B are trainable adapters
#[derive(Clone)]
pub struct LayerLoraAdapters {
    // Attention Q,K,V projection adapters
    pub a_qkv: CudaSlice<f16>,  // [rank, d_model]
    pub b_qkv: CudaSlice<f16>,  // [3*d_model, rank]
    pub m_a_qkv: CudaSlice<f32>,
    pub v_a_qkv: CudaSlice<f32>,
    pub m_b_qkv: CudaSlice<f32>,
    pub v_b_qkv: CudaSlice<f32>,

    // Attention output projection adapters
    pub a_out: CudaSlice<f16>,  // [rank, d_model]
    pub b_out: CudaSlice<f16>,  // [d_model, rank]
    pub m_a_out: CudaSlice<f32>,
    pub v_a_out: CudaSlice<f32>,
    pub m_b_out: CudaSlice<f32>,
    pub v_b_out: CudaSlice<f32>,

    // FFN first layer adapters
    pub a_ff1: CudaSlice<f16>,  // [rank, d_model]
    pub b_ff1: CudaSlice<f16>,  // [ffn_dim, rank]
    pub m_a_ff1: CudaSlice<f32>,
    pub v_a_ff1: CudaSlice<f32>,
    pub m_b_ff1: CudaSlice<f32>,
    pub v_b_ff1: CudaSlice<f32>,

    // FFN second layer adapters
    pub a_ff2: CudaSlice<f16>,  // [rank, ffn_dim]
    pub b_ff2: CudaSlice<f16>,  // [d_model, rank]
    pub m_a_ff2: CudaSlice<f32>,
    pub v_a_ff2: CudaSlice<f32>,
    pub m_b_ff2: CudaSlice<f32>,
    pub v_b_ff2: CudaSlice<f32>,
}

/// Complete LoRA model - adapters for all layers
pub struct ModelLoraAdapters {
    pub layers: Vec<LayerLoraAdapters>,
    pub config: LoraConfig,
}

impl ModelLoraAdapters {
    pub fn num_params(&self, d_model: usize, ffn_dim: usize) -> usize {
        let rank = self.config.rank;

        // Per-layer:
        // QKV: rank*d_model + 3*d_model*rank
        // Out: rank*d_model + d_model*rank
        // FF1: rank*d_model + ffn_dim*rank
        // FF2: rank*ffn_dim + d_model*rank
        let per_layer = rank * d_model * 2          // a_qkv + b_qkv
                      + rank * d_model * 2          // a_out + b_out
                      + rank * d_model + rank * ffn_dim  // ff1
                      + rank * ffn_dim + rank * d_model; // ff2

        per_layer * self.layers.len()
    }
}
