use cudarc::driver::{CudaContext, CudaStream};
use cudarc::cublas::CudaBlas;
use std::sync::Arc;

pub struct GpuContext {
    pub ctx:    Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub blas:   CudaBlas,
}

impl GpuContext {
    pub fn try_init() -> Option<Self> {
        let ctx    = CudaContext::new(0).ok()?;
        let stream = ctx.default_stream();
        println!("[GPU] CUDA device 0 — cuBLAS ready");
        let blas = CudaBlas::new(stream.clone()).ok()?;
        Some(GpuContext { ctx, stream, blas })
    }
}
