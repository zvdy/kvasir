use super::config::LlamaConfig;
use super::weights::Weights;
use super::workspace::{Workspace, WorkspaceDims};
use crate::ensure_shape;
use crate::error::Result;
use crate::gpu::gemm::{linear_f16, linear_f32};
use crate::gpu::Gpu;
use crate::kv::KvCache;

// Mirrors the `attention` wrapper's shared-memory bound: (kv_len + head_dim + 128) FP32 ≤ 48 KiB.
const ATTENTION_SMEM_BYTES: usize = 48 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Logits {
    Last,
    All,
}

pub struct Llama {
    cfg: LlamaConfig,
    weights: Weights,
    cache: KvCache,
    ws: Workspace,
}

impl Llama {
    pub fn new(
        gpu: &Gpu,
        cfg: LlamaConfig,
        weights: Weights,
        max_seq: usize,
        max_tokens: usize,
        max_head_rows: usize,
    ) -> Result<Self> {
        ensure_shape!(
            weights.layers.len() == cfg.num_layers,
            "llama: weights have {} layers, config {}",
            weights.layers.len(),
            cfg.num_layers
        );
        ensure_shape!(
            max_seq <= Self::max_seq_limit(cfg.head_dim),
            "llama: max_seq {max_seq} exceeds the attention kernel's kv length limit {}",
            Self::max_seq_limit(cfg.head_dim)
        );
        ensure_shape!(
            max_tokens <= max_seq,
            "llama: max_tokens {max_tokens} exceeds max_seq {max_seq}"
        );
        let dims = WorkspaceDims::new(&cfg, max_tokens, max_head_rows)?;
        let ws = Workspace::new(gpu, dims)?;
        let cache = KvCache::from_config(gpu, &cfg, max_seq)?;
        Ok(Self {
            cfg,
            weights,
            cache,
            ws,
        })
    }

    pub const fn max_seq_limit(head_dim: usize) -> usize {
        (ATTENTION_SMEM_BYTES / 4).saturating_sub(head_dim + 128)
    }

    pub fn config(&self) -> &LlamaConfig {
        &self.cfg
    }
    pub fn weights(&self) -> &Weights {
        &self.weights
    }
    pub fn cache(&self) -> &KvCache {
        &self.cache
    }
    pub fn workspace(&self) -> &Workspace {
        &self.ws
    }
    pub fn len(&self) -> usize {
        self.cache.len()
    }
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
    pub fn reset(&mut self) {
        self.cache.reset();
    }
    pub fn truncate(&mut self, len: usize) -> Result<()> {
        self.cache.truncate(len)
    }

    /// Logits `[1 | ids.len(), vocab]` FP32 for positions `len..len+ids.len()`; see DESIGN.md §2 "Forward pass".
    pub fn forward(&mut self, gpu: &Gpu, ids: &[u32], logits: Logits) -> Result<&[f32]> {
        let t = ids.len();
        let dims = *self.ws.dims();
        let head_rows = match logits {
            Logits::Last => 1,
            Logits::All => t,
        };
        ensure_shape!(
            (1..=dims.max_tokens).contains(&t),
            "forward: {t} tokens, workspace holds 1..={}",
            dims.max_tokens
        );
        ensure_shape!(
            head_rows <= dims.max_head_rows,
            "forward: {head_rows} logit rows, workspace holds {}",
            dims.max_head_rows
        );
        ensure_shape!(
            t <= self.cache.remaining(),
            "forward: {t} tokens at position {} overflow max_seq {}",
            self.cache.len(),
            self.cache.max_seq()
        );
        self.ws.upload_ids(gpu, ids)?;

        let cfg = &self.cfg;
        let (h, n_q, n_kv, hd) = (cfg.hidden_size, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
        let (qkv_dim, q_dim, inter, eps) = (cfg.qkv_dim(), cfg.q_dim(), cfg.intermediate_size, cfg.rms_norm_eps);
        let pos0 = self.cache.len();
        let ws = &mut self.ws;

        gpu.embedding_gather(
            &ws.ids.row_view(0..t)?,
            &self.weights.embed.buf.as_view(),
            &mut ws.hidden.row_view_mut(0..t)?,
            h,
        )?;
        for (l, lw) in self.weights.layers.iter().enumerate() {
            gpu.rmsnorm(
                &ws.hidden.row_view(0..t)?,
                &lw.attn_norm.buf.as_view(),
                &mut ws.normed.row_view_mut(0..t)?,
                t,
                h,
                eps,
            )?;
            linear_f16(
                gpu,
                &ws.normed.row_view(0..t)?,
                &lw.w_qkv.buf.as_view(),
                &mut ws.qkv.row_view_mut(0..t)?,
                t,
                h,
                qkv_dim,
            )?;
            {
                let (cos, sin) = self.cache.rope()?;
                gpu.rope_inplace(&mut ws.qkv.row_view_mut(0..t)?, &cos, &sin, t, pos0, n_q, n_kv, hd)?;
            }
            self.cache.append(gpu, l, &ws.qkv.row_view(0..t)?, t, n_q)?;
            self.cache.attention(
                gpu,
                l,
                &ws.qkv.row_view(0..t)?,
                &mut ws.attn.row_view_mut(0..t)?,
                t,
                n_q,
            )?;
            linear_f16(
                gpu,
                &ws.attn.row_view(0..t)?,
                &lw.w_o.buf.as_view(),
                &mut ws.proj.row_view_mut(0..t)?,
                t,
                q_dim,
                h,
            )?;
            gpu.add_inplace(&mut ws.hidden.row_view_mut(0..t)?, &ws.proj.row_view(0..t)?)?;

            gpu.rmsnorm(
                &ws.hidden.row_view(0..t)?,
                &lw.mlp_norm.buf.as_view(),
                &mut ws.normed.row_view_mut(0..t)?,
                t,
                h,
                eps,
            )?;
            linear_f16(
                gpu,
                &ws.normed.row_view(0..t)?,
                &lw.w_gu.buf.as_view(),
                &mut ws.gate_up.row_view_mut(0..t)?,
                t,
                h,
                2 * inter,
            )?;
            gpu.silu_mul(&ws.gate_up.row_view(0..t)?, &mut ws.act.row_view_mut(0..t)?, t, inter)?;
            linear_f16(
                gpu,
                &ws.act.row_view(0..t)?,
                &lw.w_down.buf.as_view(),
                &mut ws.proj.row_view_mut(0..t)?,
                t,
                inter,
                h,
            )?;
            gpu.add_inplace(&mut ws.hidden.row_view_mut(0..t)?, &ws.proj.row_view(0..t)?)?;
        }
        self.cache.advance(t)?;

        gpu.rmsnorm(
            &ws.hidden.row_view(t - head_rows..t)?,
            &self.weights.final_norm.buf.as_view(),
            &mut ws.normed.row_view_mut(0..head_rows)?,
            head_rows,
            h,
            eps,
        )?;
        let head = self.weights.head();
        linear_f32(
            gpu,
            &ws.normed.row_view(0..head_rows)?,
            &head.buf.as_view(),
            &mut ws.logits.row_view_mut(0..head_rows)?,
            head_rows,
            h,
            head.rows(),
        )?;
        ws.logits_to_host(gpu, head_rows)
    }
}
