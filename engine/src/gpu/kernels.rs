use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaView, CudaViewMut, LaunchConfig, PushKernelArg};
use half::{bf16, f16};

use super::device::{load_ptx, Gpu};
use crate::ensure_shape;
use crate::error::{KvasirError, Result};

macro_rules! ptx {
    ($name:literal) => {
        include_str!(concat!(env!("OUT_DIR"), "/", $name, ".ptx"))
    };
}

pub struct Kernels {
    embedding_gather: CudaFunction,
    rmsnorm: CudaFunction,
    rope_inplace: CudaFunction,
    silu_mul: CudaFunction,
    add_inplace: CudaFunction,
    bf16_to_f16: CudaFunction,
    kv_append: CudaFunction,
    attention: CudaFunction,
}

impl Kernels {
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Self> {
        let embedding = load_ptx(ctx, ptx!("embedding"))?;
        let rmsnorm = load_ptx(ctx, ptx!("rmsnorm"))?;
        let rope = load_ptx(ctx, ptx!("rope"))?;
        let silu = load_ptx(ctx, ptx!("silu_mul"))?;
        let elem = load_ptx(ctx, ptx!("elementwise"))?;
        let attn = load_ptx(ctx, ptx!("attention"))?;
        Ok(Self {
            embedding_gather: embedding.load_function("embedding_gather")?,
            rmsnorm: rmsnorm.load_function("rmsnorm")?,
            rope_inplace: rope.load_function("rope_inplace")?,
            silu_mul: silu.load_function("silu_mul")?,
            add_inplace: elem.load_function("add_inplace")?,
            bf16_to_f16: elem.load_function("bf16_to_f16")?,
            kv_append: elem.load_function("kv_append")?,
            attention: attn.load_function("attention")?,
        })
    }
}

fn i32_of(x: usize, what: &str) -> Result<i32> {
    i32::try_from(x).map_err(|_| KvasirError::Shape(format!("{what}={x} does not fit in i32")))
}

fn grid_stride(n: usize) -> u32 {
    (n.div_ceil(256)).clamp(1, 65535 * 4) as u32
}

const MAX_STATIC_SMEM: usize = 48 * 1024;

impl Gpu {
    pub fn embedding_gather(
        &self,
        ids: &CudaView<i32>,
        table: &CudaView<f16>,
        out: &mut CudaViewMut<f16>,
        hidden: usize,
    ) -> Result<()> {
        let t = ids.len();
        ensure_shape!(
            out.len() == t * hidden,
            "embedding: out has {} elems, want {t}x{hidden}",
            out.len()
        );
        ensure_shape!(
            table.len() % hidden == 0,
            "embedding: table len {} not a multiple of hidden {hidden}",
            table.len()
        );
        let h = i32_of(hidden, "hidden")?;
        let cfg = LaunchConfig {
            grid_dim: (t as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = self.stream.launch_builder(&self.kernels.embedding_gather);
        b.arg(ids).arg(table).arg(out).arg(&h);
        // SAFETY: argument order/types match `embedding_gather`; sizes checked above.
        unsafe { b.launch(cfg)? };
        Ok(())
    }

    pub fn rmsnorm(
        &self,
        x: &CudaView<f16>,
        w: &CudaView<f16>,
        out: &mut CudaViewMut<f16>,
        rows: usize,
        hidden: usize,
        eps: f32,
    ) -> Result<()> {
        ensure_shape!(
            x.len() == rows * hidden,
            "rmsnorm: x has {} elems, want {rows}x{hidden}",
            x.len()
        );
        ensure_shape!(
            out.len() == rows * hidden,
            "rmsnorm: out has {} elems, want {rows}x{hidden}",
            out.len()
        );
        ensure_shape!(w.len() == hidden, "rmsnorm: w has {} elems, want {hidden}", w.len());
        let h = i32_of(hidden, "hidden")?;
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = self.stream.launch_builder(&self.kernels.rmsnorm);
        b.arg(x).arg(w).arg(out).arg(&h).arg(&eps);
        // SAFETY: argument order/types match `rmsnorm`; sizes checked above.
        unsafe { b.launch(cfg)? };
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rope_inplace(
        &self,
        qkv: &mut CudaViewMut<f16>,
        cos: &CudaView<f32>,
        sin: &CudaView<f32>,
        t: usize,
        pos0: usize,
        n_q: usize,
        n_kv: usize,
        head_dim: usize,
    ) -> Result<()> {
        let stride = (n_q + 2 * n_kv) * head_dim;
        ensure_shape!(head_dim.is_multiple_of(2), "rope: odd head_dim {head_dim}");
        ensure_shape!(
            qkv.len() == t * stride,
            "rope: qkv has {} elems, want {t}x{stride}",
            qkv.len()
        );
        ensure_shape!(
            cos.len() == sin.len() && cos.len() % (head_dim / 2) == 0,
            "rope: bad cos/sin table sizes"
        );
        let max_pos = cos.len() / (head_dim / 2);
        ensure_shape!(
            pos0 + t <= max_pos,
            "rope: positions up to {} exceed table of {max_pos}",
            pos0 + t
        );
        let (p, nq, nk, hd, rs) = (
            i32_of(pos0, "pos0")?,
            i32_of(n_q, "n_q")?,
            i32_of(n_kv, "n_kv")?,
            i32_of(head_dim, "head_dim")?,
            i32_of(stride, "row_stride")?,
        );
        let cfg = LaunchConfig {
            grid_dim: (t as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = self.stream.launch_builder(&self.kernels.rope_inplace);
        b.arg(qkv).arg(cos).arg(sin).arg(&p).arg(&nq).arg(&nk).arg(&hd).arg(&rs);
        // SAFETY: argument order/types match `rope_inplace`; sizes checked above.
        unsafe { b.launch(cfg)? };
        Ok(())
    }

    pub fn silu_mul(&self, gu: &CudaView<f16>, out: &mut CudaViewMut<f16>, t: usize, inter: usize) -> Result<()> {
        ensure_shape!(
            gu.len() == t * 2 * inter,
            "silu_mul: gu has {} elems, want {t}x{}",
            gu.len(),
            2 * inter
        );
        ensure_shape!(
            out.len() == t * inter,
            "silu_mul: out has {} elems, want {t}x{inter}",
            out.len()
        );
        let (it, n) = (i32_of(inter, "inter")?, (t * inter) as i64);
        let cfg = LaunchConfig {
            grid_dim: (grid_stride(t * inter), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = self.stream.launch_builder(&self.kernels.silu_mul);
        b.arg(gu).arg(out).arg(&it).arg(&n);
        // SAFETY: argument order/types match `silu_mul`; sizes checked above.
        unsafe { b.launch(cfg)? };
        Ok(())
    }

    pub fn add_inplace(&self, x: &mut CudaViewMut<f16>, y: &CudaView<f16>) -> Result<()> {
        ensure_shape!(x.len() == y.len(), "add: lengths differ ({} vs {})", x.len(), y.len());
        let n = x.len() as i64;
        let cfg = LaunchConfig {
            grid_dim: (grid_stride(x.len()), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = self.stream.launch_builder(&self.kernels.add_inplace);
        b.arg(x).arg(y).arg(&n);
        // SAFETY: argument order/types match `add_inplace`; lengths checked above.
        unsafe { b.launch(cfg)? };
        Ok(())
    }

    pub fn bf16_to_f16(
        &self,
        src: &CudaView<bf16>,
        dst: &mut CudaViewMut<f16>,
        bad: &mut CudaViewMut<u32>,
    ) -> Result<()> {
        ensure_shape!(
            src.len() == dst.len(),
            "bf16_to_f16: lengths differ ({} vs {})",
            src.len(),
            dst.len()
        );
        ensure_shape!(bad.len() == 1, "bf16_to_f16: bad counter must hold exactly one u32");
        let n = src.len() as i64;
        let cfg = LaunchConfig {
            grid_dim: (grid_stride(src.len()), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = self.stream.launch_builder(&self.kernels.bf16_to_f16);
        b.arg(src).arg(dst).arg(&n).arg(bad);
        // SAFETY: argument order/types match `bf16_to_f16`; lengths checked above.
        unsafe { b.launch(cfg)? };
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn kv_append(
        &self,
        qkv: &CudaView<f16>,
        kc: &mut CudaViewMut<f16>,
        vc: &mut CudaViewMut<f16>,
        t: usize,
        pos0: usize,
        n_q: usize,
        n_kv: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> Result<()> {
        let stride = (n_q + 2 * n_kv) * head_dim;
        ensure_shape!(
            qkv.len() == t * stride,
            "kv_append: qkv has {} elems, want {t}x{stride}",
            qkv.len()
        );
        ensure_shape!(
            kc.len() == n_kv * max_seq * head_dim && vc.len() == kc.len(),
            "kv_append: bad cache sizes"
        );
        ensure_shape!(
            pos0 + t <= max_seq,
            "kv_append: positions up to {} exceed max_seq {max_seq}",
            pos0 + t
        );
        let (p, nk, hd, ms, qd, rs) = (
            i32_of(pos0, "pos0")?,
            i32_of(n_kv, "n_kv")?,
            i32_of(head_dim, "head_dim")?,
            i32_of(max_seq, "max_seq")?,
            i32_of(n_q * head_dim, "q_dim")?,
            i32_of(stride, "row_stride")?,
        );
        let cfg = LaunchConfig {
            grid_dim: (t as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = self.stream.launch_builder(&self.kernels.kv_append);
        b.arg(qkv)
            .arg(kc)
            .arg(vc)
            .arg(&p)
            .arg(&nk)
            .arg(&hd)
            .arg(&ms)
            .arg(&qd)
            .arg(&rs);
        // SAFETY: argument order/types match `kv_append`; sizes checked above.
        unsafe { b.launch(cfg)? };
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &self,
        qkv: &CudaView<f16>,
        kc: &CudaView<f16>,
        vc: &CudaView<f16>,
        out: &mut CudaViewMut<f16>,
        t: usize,
        pos0: usize,
        n_q: usize,
        n_kv: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> Result<()> {
        let stride = (n_q + 2 * n_kv) * head_dim;
        ensure_shape!(
            n_kv > 0 && n_q.is_multiple_of(n_kv),
            "attention: n_q {n_q} not a multiple of n_kv {n_kv}"
        );
        ensure_shape!(
            head_dim > 0 && 128 % head_dim == 0,
            "attention: head_dim {head_dim} must divide 128"
        );
        ensure_shape!(
            qkv.len() == t * stride,
            "attention: qkv has {} elems, want {t}x{stride}",
            qkv.len()
        );
        ensure_shape!(
            kc.len() == n_kv * max_seq * head_dim && vc.len() == kc.len(),
            "attention: bad cache sizes"
        );
        ensure_shape!(
            out.len() == t * n_q * head_dim,
            "attention: out has {} elems, want {t}x{}",
            out.len(),
            n_q * head_dim
        );
        ensure_shape!(
            pos0 + t <= max_seq,
            "attention: kv length {} exceeds max_seq {max_seq}",
            pos0 + t
        );
        let smem = (pos0 + t + head_dim + 128) * 4;
        ensure_shape!(
            smem <= MAX_STATIC_SMEM,
            "attention: kv length {} needs {smem} B of shared memory (> 48 KiB)",
            pos0 + t
        );
        let (p, nq, nk, hd, ms, rs) = (
            i32_of(pos0, "pos0")?,
            i32_of(n_q, "n_q")?,
            i32_of(n_kv, "n_kv")?,
            i32_of(head_dim, "head_dim")?,
            i32_of(max_seq, "max_seq")?,
            i32_of(stride, "row_stride")?,
        );
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let cfg = LaunchConfig {
            grid_dim: (n_q as u32, t as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: smem as u32,
        };
        let mut b = self.stream.launch_builder(&self.kernels.attention);
        b.arg(qkv)
            .arg(kc)
            .arg(vc)
            .arg(out)
            .arg(&p)
            .arg(&nq)
            .arg(&nk)
            .arg(&hd)
            .arg(&ms)
            .arg(&rs)
            .arg(&scale);
        // SAFETY: argument order/types match `attention`; sizes checked above.
        unsafe { b.launch(cfg)? };
        Ok(())
    }
}
