use cudarc::cublas::{result, sys};
use cudarc::driver::{CudaView, CudaViewMut, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg};
use half::f16;

use super::device::Gpu;
use super::kernels::{check_aligned, i32_of};
use crate::ensure_shape;
use crate::error::{KvasirError, Result};

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GemvKind {
    F16,
    F32,
    Add,
    SiluMul,
}

// Output rows per warp (= GEMV_ROWS / GEMV_SILU_ROWS in gemv.cu) and warps per block; see DESIGN.md §3.
const GEMV_ROWS: usize = 2;
const GEMV_SILU_ROWS: usize = 1;
const GEMV_WARPS: usize = 4;

/// RMSNorm fused into the GEMV prologue: `x` is the raw row, normalised and rounded exactly like `Gpu::rmsnorm`.
#[derive(Clone, Copy)]
pub struct Norm<'a, 'b> {
    pub w: &'a CudaView<'b, f16>,
    pub eps: f32,
}

#[allow(clippy::too_many_arguments)]
fn gemv_launch<Y: cudarc::driver::DeviceRepr>(
    gpu: &Gpu,
    kind: GemvKind,
    x: &CudaView<f16>,
    w: &CudaView<f16>,
    y: &mut CudaViewMut<Y>,
    in_dim: usize,
    rows: usize,
    norm: Option<Norm>,
) -> Result<()> {
    let w_rows = if kind == GemvKind::SiluMul { 2 * rows } else { rows };
    ensure_shape!(
        in_dim > 0 && in_dim.is_multiple_of(8),
        "gemv: in_dim {in_dim} must be a positive multiple of 8"
    );
    ensure_shape!(rows > 0, "gemv: no output rows");
    ensure_shape!(x.len() == in_dim, "gemv: x has {} elems, want {in_dim}", x.len());
    ensure_shape!(
        w.len() == w_rows * in_dim,
        "gemv: w has {} elems, want {w_rows}x{in_dim}",
        w.len()
    );
    ensure_shape!(y.len() == rows, "gemv: y has {} elems, want {rows}", y.len());
    check_aligned(x, &gpu.stream, 16, "gemv: x")?;
    check_aligned(w, &gpu.stream, 16, "gemv: w")?;
    if let Some(nm) = norm {
        ensure_shape!(
            nm.w.len() == in_dim,
            "gemv: norm weight has {} elems, want {in_dim}",
            nm.w.len()
        );
        check_aligned(nm.w, &gpu.stream, 16, "gemv: norm weight")?;
    }
    let k = &gpu.kernels.gemv;
    let (f, r) = match kind {
        GemvKind::F16 => (&k.f16, GEMV_ROWS),
        GemvKind::F32 => (&k.f32, GEMV_ROWS),
        GemvKind::Add => (&k.add, GEMV_ROWS),
        GemvKind::SiluMul => (&k.silu_mul, GEMV_SILU_ROWS),
    };
    let blocks = u32::try_from(rows.div_ceil(r * GEMV_WARPS))
        .map_err(|_| KvasirError::Shape(format!("gemv: {rows} rows need too many blocks")))?;
    let (i, n) = (i32_of(in_dim, "in_dim")?, i32_of(rows, "rows")?);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (32 * GEMV_WARPS as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let null: u64 = 0;
    let eps = norm.map_or(0.0, |nm| nm.eps);
    let mut b = gpu.stream.launch_builder(f);
    b.arg(w).arg(x).arg(y).arg(&i).arg(&n);
    match norm {
        Some(nm) => b.arg(nm.w),
        None => b.arg(&null),
    };
    b.arg(&eps);
    // SAFETY: arguments match `gemv_*` (w, x, y, in, rows, norm weight or null, eps); sizes/alignment checked above.
    unsafe { b.launch(cfg)? };
    Ok(())
}

/// Decode GEMV `y = W[out, in]·x`, FP16 out; `in_dim % 8 == 0`, `x`/`w`/norm weight 16-byte aligned (DESIGN.md §3).
pub fn gemv_f16(
    gpu: &Gpu,
    x: &CudaView<f16>,
    w: &CudaView<f16>,
    y: &mut CudaViewMut<f16>,
    in_dim: usize,
    out_dim: usize,
    norm: Option<Norm>,
) -> Result<()> {
    gemv_launch(gpu, GemvKind::F16, x, w, y, in_dim, out_dim, norm)
}

/// As [`gemv_f16`] with FP32 output (LM head).
pub fn gemv_f32(
    gpu: &Gpu,
    x: &CudaView<f16>,
    w: &CudaView<f16>,
    y: &mut CudaViewMut<f32>,
    in_dim: usize,
    out_dim: usize,
    norm: Option<Norm>,
) -> Result<()> {
    gemv_launch(gpu, GemvKind::F32, x, w, y, in_dim, out_dim, norm)
}

/// Residual GEMV `y += W·x` (W·x rounded to FP16 before the add). `y` must not alias `x`.
pub fn gemv_add(
    gpu: &Gpu,
    x: &CudaView<f16>,
    w: &CudaView<f16>,
    y: &mut CudaViewMut<f16>,
    in_dim: usize,
    out_dim: usize,
) -> Result<()> {
    gemv_launch(gpu, GemvKind::Add, x, w, y, in_dim, out_dim, None)
}

/// Fused gate/up GEMV + SwiGLU: `w_gu` is `[2·inter, in]` (gate rows, then up rows), `act[i] = silu(g_i)·u_i`.
pub fn gemv_silu_mul(
    gpu: &Gpu,
    x: &CudaView<f16>,
    w_gu: &CudaView<f16>,
    act: &mut CudaViewMut<f16>,
    in_dim: usize,
    inter: usize,
    norm: Option<Norm>,
) -> Result<()> {
    gemv_launch(gpu, GemvKind::SiluMul, x, w_gu, act, in_dim, inter, norm)
}
