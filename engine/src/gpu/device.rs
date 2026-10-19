use std::sync::Arc;

use cudarc::cublas::CudaBlas;
use cudarc::driver::{CudaContext, CudaStream};
use cudarc::nvrtc::Ptx;

use super::kernels::Kernels;
use crate::error::Result;

pub struct Gpu {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub blas: CudaBlas,
    pub kernels: Kernels,
}

impl Gpu {
    pub fn new(ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(ordinal)?;
        // SAFETY: all work is issued on `stream`, so no cross-stream hazards exist.
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.new_stream()?;
        let blas = CudaBlas::new(stream.clone())?;
        let kernels = Kernels::load(&ctx)?;
        Ok(Self {
            ctx,
            stream,
            blas,
            kernels,
        })
    }

    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()?;
        Ok(())
    }
}

pub(crate) fn load_ptx(ctx: &Arc<CudaContext>, src: &'static str) -> Result<Arc<cudarc::driver::CudaModule>> {
    Ok(ctx.load_module(Ptx::from_src(src))?)
}
