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
        // Non-blocking stream is critical for performance. The legacy default stream
        // synchronizes the host after every operation, which kills throughput.
        let stream = ctx.new_stream().ok()?;
        println!("[GPU] CUDA device 0 — non-blocking stream — cuBLAS ready");
        let blas = CudaBlas::new(stream.clone()).ok()?;
        Some(GpuContext { ctx, stream, blas })
    }
}
