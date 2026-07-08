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

impl LayerLoraAdapters {
    /// Initialize LoRA adapters: A matrices with Kaiming uniform, B matrices with zeros
    pub fn init_kaiming_zeros(
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        d_model: usize,
        ffn_dim: usize,
        rank: usize,
    ) -> Self {
        use rand::Rng;

        let mut rng = rand::thread_rng();

        // Kaiming uniform: scale = sqrt(6 / fan_in)
        let kaiming_scale = |fan_in: usize| (6.0 / fan_in as f32).sqrt();

        // A matrices: Kaiming uniform initialization
        let a_qkv_data: Vec<f32> = (0..rank * d_model)
            .map(|_| rng.gen_range(-1.0..1.0) * kaiming_scale(d_model))
            .collect();
        let a_out_data: Vec<f32> = (0..rank * d_model)
            .map(|_| rng.gen_range(-1.0..1.0) * kaiming_scale(d_model))
            .collect();
        let a_ff1_data: Vec<f32> = (0..rank * d_model)
            .map(|_| rng.gen_range(-1.0..1.0) * kaiming_scale(d_model))
            .collect();
        let a_ff2_data: Vec<f32> = (0..rank * ffn_dim)
            .map(|_| rng.gen_range(-1.0..1.0) * kaiming_scale(ffn_dim))
            .collect();

        // B matrices: zero initialization
        let b_qkv_data = vec![0.0f32; 3 * d_model * rank];
        let b_out_data = vec![0.0f32; d_model * rank];
        let b_ff1_data = vec![0.0f32; ffn_dim * rank];
        let b_ff2_data = vec![0.0f32; d_model * rank];

        // Convert to f16 and upload to GPU
        let to_f16 = |v: &[f32]| v.iter().map(|&x| half::f16::from_f32(x)).collect::<Vec<_>>();
        let upload = |data: Vec<f16>| stream.clone_htod(&data).unwrap();

        // Adam moments (FP32): initialize to zero
        let zeros_f32 = |n: usize| vec![0.0f32; n];
        let upload_f32 = |data: Vec<f32>| stream.clone_htod(&data).unwrap();

        Self {
            a_qkv: upload(to_f16(&a_qkv_data)),
            b_qkv: upload(to_f16(&b_qkv_data)),
            m_a_qkv: upload_f32(zeros_f32(rank * d_model)),
            v_a_qkv: upload_f32(zeros_f32(rank * d_model)),
            m_b_qkv: upload_f32(zeros_f32(3 * d_model * rank)),
            v_b_qkv: upload_f32(zeros_f32(3 * d_model * rank)),

            a_out: upload(to_f16(&a_out_data)),
            b_out: upload(to_f16(&b_out_data)),
            m_a_out: upload_f32(zeros_f32(rank * d_model)),
            v_a_out: upload_f32(zeros_f32(rank * d_model)),
            m_b_out: upload_f32(zeros_f32(d_model * rank)),
            v_b_out: upload_f32(zeros_f32(d_model * rank)),

            a_ff1: upload(to_f16(&a_ff1_data)),
            b_ff1: upload(to_f16(&b_ff1_data)),
            m_a_ff1: upload_f32(zeros_f32(rank * d_model)),
            v_a_ff1: upload_f32(zeros_f32(rank * d_model)),
            m_b_ff1: upload_f32(zeros_f32(ffn_dim * rank)),
            v_b_ff1: upload_f32(zeros_f32(ffn_dim * rank)),

            a_ff2: upload(to_f16(&a_ff2_data)),
            b_ff2: upload(to_f16(&b_ff2_data)),
            m_a_ff2: upload_f32(zeros_f32(rank * ffn_dim)),
            v_a_ff2: upload_f32(zeros_f32(rank * ffn_dim)),
            m_b_ff2: upload_f32(zeros_f32(d_model * rank)),
            v_b_ff2: upload_f32(zeros_f32(d_model * rank)),
        }
    }
}
