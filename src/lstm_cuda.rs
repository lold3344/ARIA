use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, LaunchAsync, LaunchConfig};
use cudarc::cublas::{CudaBlas, Gemm};
use std::sync::Arc;

pub struct GpuContext {
    pub device: Arc<CudaDevice>,
    pub blas:   CudaBlas,
}

impl GpuContext {
    pub fn try_init() -> Option<Self> {
        let device = CudaDevice::new(0).ok()?;
        println!("[GPU] CUDA device 0 initialised");
        let blas = CudaBlas::new(device.clone()).ok()?;
        Some(GpuContext { device, blas })
    }
}
