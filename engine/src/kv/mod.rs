use cudarc::driver::{CudaView, CudaViewMut};
use half::f16;

use crate::ensure_shape;
use crate::error::{KvasirError, Result};
use crate::gpu::{DeviceTensor, Gpu};
use crate::model::{rope, LlamaConfig};

pub struct KvCache {
    k: DeviceTensor<f16>,
    v: DeviceTensor<f16>,
    rope: Option<(DeviceTensor<f32>, DeviceTensor<f32>)>,
    layers: usize,
    n_kv: usize,
    max_seq: usize,
    head_dim: usize,
    len: usize,
}

impl KvCache {
    pub fn new(gpu: &Gpu, layers: usize, n_kv: usize, max_seq: usize, head_dim: usize) -> Result<Self> {
        ensure_shape!(
            layers > 0 && n_kv > 0 && max_seq > 0 && head_dim > 0,
            "kv cache: zero dimension in [{layers}, {n_kv}, {max_seq}, {head_dim}]"
        );
        let shape = [layers, n_kv, max_seq, head_dim];
        Ok(Self {
            k: DeviceTensor::zeros(gpu, &shape)?,
            v: DeviceTensor::zeros(gpu, &shape)?,
            rope: None,
            layers,
            n_kv,
            max_seq,
            head_dim,
            len: 0,
        })
    }

    pub fn from_config(gpu: &Gpu, cfg: &LlamaConfig, max_seq: usize) -> Result<Self> {
        ensure_shape!(
            max_seq <= cfg.max_position_embeddings,
            "kv cache: max_seq {max_seq} exceeds max_position_embeddings {}",
            cfg.max_position_embeddings
        );
        let mut cache = Self::new(gpu, cfg.num_layers, cfg.num_kv_heads, max_seq, cfg.head_dim)?;
        let (cos, sin) = rope::cos_sin_table(cfg, max_seq);
        let shape = [max_seq, cfg.head_dim / 2];
        cache.rope = Some((
            DeviceTensor::from_host(gpu, &cos, &shape)?,
            DeviceTensor::from_host(gpu, &sin, &shape)?,
        ));
        Ok(cache)
    }

    /// RoPE cos/sin tables `[max_seq, head_dim/2]` FP32; present only when built with `from_config`.
    pub fn rope(&self) -> Result<(CudaView<'_, f32>, CudaView<'_, f32>)> {
        let (cos, sin) = self
            .rope
            .as_ref()
            .ok_or_else(|| KvasirError::Config("kv cache built without RoPE tables (use from_config)".into()))?;
        Ok((cos.buf.as_view(), sin.buf.as_view()))
    }

    pub fn layers(&self) -> usize {
        self.layers
    }
    pub fn n_kv(&self) -> usize {
        self.n_kv
    }
    pub fn max_seq(&self) -> usize {
        self.max_seq
    }
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn remaining(&self) -> usize {
        self.max_seq - self.len
    }
    pub fn layer_numel(&self) -> usize {
        self.n_kv * self.max_seq * self.head_dim
    }
    pub fn k(&self) -> &DeviceTensor<f16> {
        &self.k
    }
    pub fn v(&self) -> &DeviceTensor<f16> {
        &self.v
    }

    fn layer_range(&self, layer: usize) -> Result<std::ops::Range<usize>> {
        ensure_shape!(
            layer < self.layers,
            "kv cache: layer {layer} out of range ({} layers)",
            self.layers
        );
        let n = self.layer_numel();
        Ok(layer * n..(layer + 1) * n)
    }

    fn check_fits(&self, t: usize) -> Result<()> {
        ensure_shape!(t > 0, "kv cache: empty append");
        ensure_shape!(
            t <= self.remaining(),
            "kv cache: {t} tokens at position {} overflow max_seq {}",
            self.len,
            self.max_seq
        );
        Ok(())
    }

    pub fn layer(&self, layer: usize) -> Result<(CudaView<'_, f16>, CudaView<'_, f16>)> {
        let r = self.layer_range(layer)?;
        let bad = || KvasirError::Shape(format!("kv cache: layer {layer} slice out of bounds"));
        let k = self.k.buf.try_slice(r.clone()).ok_or_else(bad)?;
        let v = self.v.buf.try_slice(r).ok_or_else(bad)?;
        Ok((k, v))
    }

    pub fn layer_mut(&mut self, layer: usize) -> Result<(CudaViewMut<'_, f16>, CudaViewMut<'_, f16>)> {
        let r = self.layer_range(layer)?;
        let bad = || KvasirError::Shape(format!("kv cache: layer {layer} slice out of bounds"));
        let k = self.k.buf.try_slice_mut(r.clone()).ok_or_else(bad)?;
        let v = self.v.buf.try_slice_mut(r).ok_or_else(bad)?;
        Ok((k, v))
    }

    /// Writes K/V of `t` tokens from `qkv` `[t, (n_q + 2·n_kv)·head_dim]` at positions `len..len+t` of `layer`.
    pub fn append(&mut self, gpu: &Gpu, layer: usize, qkv: &CudaView<f16>, t: usize, n_q: usize) -> Result<()> {
        self.check_fits(t)?;
        let (pos0, n_kv, head_dim, max_seq) = (self.len, self.n_kv, self.head_dim, self.max_seq);
        let (mut k, mut v) = self.layer_mut(layer)?;
        gpu.kv_append(qkv, &mut k, &mut v, t, pos0, n_q, n_kv, head_dim, max_seq)
    }

    /// Causal attention of `t` queries at positions `len..len+t` over keys `0..len+t` of `layer`.
    pub fn attention(
        &self,
        gpu: &Gpu,
        layer: usize,
        qkv: &CudaView<f16>,
        out: &mut CudaViewMut<f16>,
        t: usize,
        n_q: usize,
    ) -> Result<()> {
        self.check_fits(t)?;
        let (k, v) = self.layer(layer)?;
        gpu.attention(
            qkv,
            &k,
            &v,
            out,
            t,
            self.len,
            n_q,
            self.n_kv,
            self.head_dim,
            self.max_seq,
        )
    }

    pub fn advance(&mut self, t: usize) -> Result<()> {
        self.check_fits(t)?;
        self.len += t;
        Ok(())
    }

    pub fn truncate(&mut self, len: usize) -> Result<()> {
        ensure_shape!(
            len <= self.len,
            "kv cache: cannot truncate to {len}, length is {}",
            self.len
        );
        self.len = len;
        Ok(())
    }

    pub fn reset(&mut self) {
        self.len = 0;
    }
}
