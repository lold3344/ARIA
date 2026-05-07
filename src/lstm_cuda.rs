use tch::{Device, Tensor};

pub struct LSTMCuda {
    pub device: Device,
}

impl LSTMCuda {
    pub fn try_init() -> Option<Self> {
        let device = Device::cuda_if_available();
        if !device.is_cuda() {
            println!("[LSTMCuda] No CUDA device found.");
            return None;
        }

        println!("[LSTMCuda] CUDA device detected: {:?}", device);

        // Allocate a tiny tensor to force driver initialization
        let _probe = Tensor::zeros(&[1], (tch::Kind::Float, device));
        println!("[LSTMCuda] CUDA ready.\n");

        Some(LSTMCuda { device })
    }

    pub fn is_available(&self) -> bool {
        Device::cuda_if_available().is_cuda()
    }

    pub fn get_memory_info() -> (f64, f64, f64) {
        if !Device::cuda_if_available().is_cuda() {
            return (0.0, 0.0, 0.0);
        }
        // tch doesn't expose per-GPU memory directly
        (0.0, 0.0, 0.0)
    }

    pub fn synchronize(&self) {
        // Force sync by doing a no-op CUDA operation
        let _ = Tensor::zeros(&[1], (tch::Kind::Float, self.device));
    }
}

impl Default for LSTMCuda {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
        }
    }
}