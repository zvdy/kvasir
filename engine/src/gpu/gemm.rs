use cudarc::cublas::{result, sys};
use cudarc::driver::{DevicePtr, DevicePtrMut};
use half::f16;

use super::device::Gpu;
use crate::ensure_shape;
use crate::error::Result;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutType {
    F16,
    F32,
}

#[allow(clippy::too_many_arguments)]
// Row-major y = x·wᵀ as column-major C[out×T] = op_T(W[in×out]) · X[in×T]; no transposes materialized.
fn gemm_nt<Y>(
    gpu: &Gpu,
    x: &impl DevicePtr<f16>,
    w: &impl DevicePtr<f16>,
    y: &mut impl DevicePtrMut<Y>,
    t: usize,
    in_dim: usize,
    out_dim: usize,
    out: OutType,
) -> Result<()> {
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let (xp, _gx) = x.device_ptr(&gpu.stream);
    let (wp, _gw) = w.device_ptr(&gpu.stream);
    let (yp, _gy) = y.device_ptr_mut(&gpu.stream);
    let c_type = match out {
        OutType::F16 => sys::cudaDataType_t::CUDA_R_16F,
        OutType::F32 => sys::cudaDataType_t::CUDA_R_32F,
    };
    // SAFETY: callers validate sizes; buffers are live allocations on the handle's stream.
    unsafe {
        result::gemm_ex(
            *gpu.blas.handle(),
            sys::cublasOperation_t::CUBLAS_OP_T,
            sys::cublasOperation_t::CUBLAS_OP_N,
            out_dim as i32,
            t as i32,
            in_dim as i32,
            (&alpha) as *const f32 as *const _,
            wp as *const _,
            sys::cudaDataType_t::CUDA_R_16F,
            in_dim as i32,
            xp as *const _,
            sys::cudaDataType_t::CUDA_R_16F,
            in_dim as i32,
            (&beta) as *const f32 as *const _,
            yp as *mut _,
            c_type,
            out_dim as i32,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )?;
    }
    Ok(())
}

pub fn linear_f16(
    gpu: &Gpu,
    x: &cudarc::driver::CudaView<f16>,
    w: &cudarc::driver::CudaView<f16>,
    y: &mut cudarc::driver::CudaViewMut<f16>,
    t: usize,
    in_dim: usize,
    out_dim: usize,
) -> Result<()> {
    ensure_shape!(
        x.len() == t * in_dim,
        "linear: x has {} elems, want {t}x{in_dim}",
        x.len()
    );
    ensure_shape!(
        w.len() == out_dim * in_dim,
        "linear: w has {} elems, want {out_dim}x{in_dim}",
        w.len()
    );
    ensure_shape!(
        y.len() == t * out_dim,
        "linear: y has {} elems, want {t}x{out_dim}",
        y.len()
    );
    gemm_nt(gpu, x, w, y, t, in_dim, out_dim, OutType::F16)
}

pub fn linear_f32(
    gpu: &Gpu,
    x: &cudarc::driver::CudaView<f16>,
    w: &cudarc::driver::CudaView<f16>,
    y: &mut cudarc::driver::CudaViewMut<f32>,
    t: usize,
    in_dim: usize,
    out_dim: usize,
) -> Result<()> {
    ensure_shape!(
        x.len() == t * in_dim,
        "linear: x has {} elems, want {t}x{in_dim}",
        x.len()
    );
    ensure_shape!(
        w.len() == out_dim * in_dim,
        "linear: w has {} elems, want {out_dim}x{in_dim}",
        w.len()
    );
    ensure_shape!(
        y.len() == t * out_dim,
        "linear: y has {} elems, want {t}x{out_dim}",
        y.len()
    );
    gemm_nt(gpu, x, w, y, t, in_dim, out_dim, OutType::F32)
}
