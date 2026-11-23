use half::f16;

use super::config::LlamaConfig;
use crate::ensure_shape;
use crate::error::{KvasirError, Result};
use crate::gpu::kernels::attention_scratch_len;
use crate::gpu::{DeviceTensor, Gpu};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkspaceDims {
    pub max_tokens: usize,
    pub max_head_rows: usize,
    pub hidden: usize,
    pub q_dim: usize,
    pub qkv_dim: usize,
    pub inter: usize,
    pub vocab: usize,
    pub n_q: usize,
    pub head_dim: usize,
}

impl WorkspaceDims {
    pub fn new(cfg: &LlamaConfig, max_tokens: usize, max_head_rows: usize) -> Result<Self> {
        ensure_shape!(max_tokens > 0, "workspace: max_tokens must be > 0");
        ensure_shape!(
            (1..=max_tokens).contains(&max_head_rows),
            "workspace: max_head_rows {max_head_rows} must be in 1..={max_tokens}"
        );
        let d = Self {
            max_tokens,
            max_head_rows,
            hidden: cfg.hidden_size,
            q_dim: cfg.q_dim(),
            qkv_dim: cfg.qkv_dim(),
            inter: cfg.intermediate_size,
            vocab: cfg.vocab_size,
            n_q: cfg.num_heads,
            head_dim: cfg.head_dim,
        };
        let biggest = [
            d.max_tokens.checked_mul(d.hidden),
            d.max_tokens.checked_mul(d.qkv_dim),
            d.max_tokens.checked_mul(d.q_dim),
            d.inter.checked_mul(2).and_then(|n| n.checked_mul(d.max_tokens)),
            d.max_head_rows.checked_mul(d.vocab),
            Some(attention_scratch_len(d.n_q, d.head_dim)),
        ];
        for n in biggest {
            let n = n.ok_or_else(|| KvasirError::Shape("workspace: buffer size overflows usize".into()))?;
            ensure_shape!(
                i32::try_from(n).is_ok(),
                "workspace: buffer of {n} elements exceeds i32 indexing"
            );
        }
        Ok(d)
    }

    pub fn ids_shape(&self) -> [usize; 1] {
        [self.max_tokens]
    }
    pub fn hidden_shape(&self) -> [usize; 2] {
        [self.max_tokens, self.hidden]
    }
    pub fn qkv_shape(&self) -> [usize; 2] {
        [self.max_tokens, self.qkv_dim]
    }
    pub fn attn_shape(&self) -> [usize; 2] {
        [self.max_tokens, self.q_dim]
    }
    pub fn gate_up_shape(&self) -> [usize; 2] {
        [self.max_tokens, 2 * self.inter]
    }
    pub fn act_shape(&self) -> [usize; 2] {
        [self.max_tokens, self.inter]
    }
    pub fn logits_shape(&self) -> [usize; 2] {
        [self.max_head_rows, self.vocab]
    }
    pub fn attn_scratch_shape(&self) -> [usize; 1] {
        [attention_scratch_len(self.n_q, self.head_dim)]
    }

    // hidden, normed and proj share hidden_shape.
    pub fn f16_elems(&self) -> usize {
        let n = |s: [usize; 2]| s[0] * s[1];
        3 * n(self.hidden_shape())
            + n(self.qkv_shape())
            + n(self.attn_shape())
            + n(self.gate_up_shape())
            + n(self.act_shape())
    }

    pub fn device_bytes(&self) -> usize {
        self.f16_elems() * size_of::<f16>()
            + (self.logits_shape().iter().product::<usize>() + self.attn_scratch_shape()[0]) * size_of::<f32>()
            + self.max_tokens * size_of::<i32>()
    }
}

pub struct Workspace {
    dims: WorkspaceDims,
    pub ids: DeviceTensor<i32>,
    pub hidden: DeviceTensor<f16>,
    pub normed: DeviceTensor<f16>,
    pub qkv: DeviceTensor<f16>,
    pub attn: DeviceTensor<f16>,
    pub proj: DeviceTensor<f16>,
    pub gate_up: DeviceTensor<f16>,
    pub act: DeviceTensor<f16>,
    pub logits: DeviceTensor<f32>,
    pub attn_scratch: DeviceTensor<f32>,
    ids_host: Vec<i32>,
    logits_host: Vec<f32>,
}

impl Workspace {
    pub fn new(gpu: &Gpu, dims: WorkspaceDims) -> Result<Self> {
        Ok(Self {
            dims,
            ids: DeviceTensor::zeros(gpu, &dims.ids_shape())?,
            hidden: DeviceTensor::zeros(gpu, &dims.hidden_shape())?,
            normed: DeviceTensor::zeros(gpu, &dims.hidden_shape())?,
            qkv: DeviceTensor::zeros(gpu, &dims.qkv_shape())?,
            attn: DeviceTensor::zeros(gpu, &dims.attn_shape())?,
            proj: DeviceTensor::zeros(gpu, &dims.hidden_shape())?,
            gate_up: DeviceTensor::zeros(gpu, &dims.gate_up_shape())?,
            act: DeviceTensor::zeros(gpu, &dims.act_shape())?,
            logits: DeviceTensor::zeros(gpu, &dims.logits_shape())?,
            attn_scratch: DeviceTensor::zeros(gpu, &dims.attn_scratch_shape())?,
            ids_host: vec![0; dims.max_tokens],
            logits_host: vec![0.0; dims.max_head_rows * dims.vocab],
        })
    }

    pub fn dims(&self) -> &WorkspaceDims {
        &self.dims
    }

    pub fn upload_ids(&mut self, gpu: &Gpu, ids: &[u32]) -> Result<()> {
        let t = ids.len();
        ensure_shape!(
            (1..=self.dims.max_tokens).contains(&t),
            "upload_ids: {t} tokens, workspace holds 1..={}",
            self.dims.max_tokens
        );
        for (dst, &id) in self.ids_host.iter_mut().zip(ids) {
            ensure_shape!(
                (id as usize) < self.dims.vocab,
                "upload_ids: token id {id} >= vocab {}",
                self.dims.vocab
            );
            *dst = id as i32;
        }
        let mut dst = prefix_mut(&mut self.ids, t)?;
        gpu.stream.memcpy_htod(&self.ids_host[..t], &mut dst)?;
        Ok(())
    }

    pub fn logits_to_host(&mut self, gpu: &Gpu, rows: usize) -> Result<&[f32]> {
        ensure_shape!(
            (1..=self.dims.max_head_rows).contains(&rows),
            "logits_to_host: {rows} rows, workspace holds 1..={}",
            self.dims.max_head_rows
        );
        let n = rows * self.dims.vocab;
        let src = self
            .logits
            .buf
            .try_slice(..n)
            .ok_or_else(|| KvasirError::Shape(format!("logits_to_host: {n} exceeds logits buffer")))?;
        gpu.stream.memcpy_dtoh(&src, &mut self.logits_host[..n])?;
        gpu.stream.synchronize()?;
        Ok(&self.logits_host[..n])
    }
}

fn prefix_mut<T>(t: &mut DeviceTensor<T>, n: usize) -> Result<cudarc::driver::CudaViewMut<'_, T>> {
    let len = t.numel();
    t.buf
        .try_slice_mut(..n)
        .ok_or_else(|| KvasirError::Shape(format!("prefix of {n} exceeds buffer of {len}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    const LLAMA_3_2_1B: &str = r#"{
      "head_dim": 64, "hidden_size": 2048, "intermediate_size": 8192, "max_position_embeddings": 131072,
      "num_attention_heads": 32, "num_hidden_layers": 16, "num_key_value_heads": 8, "rms_norm_eps": 1e-05,
      "rope_theta": 500000.0, "tie_word_embeddings": true, "vocab_size": 128256 }"#;

    fn cfg() -> LlamaConfig {
        LlamaConfig::from_json(LLAMA_3_2_1B).unwrap()
    }

    #[test]
    fn workspace_shapes_match_design_table() {
        let d = WorkspaceDims::new(&cfg(), 7, 1).unwrap();
        assert_eq!(d.ids_shape(), [7]);
        assert_eq!(d.hidden_shape(), [7, 2048]);
        assert_eq!(d.qkv_shape(), [7, 3072]);
        assert_eq!(d.attn_shape(), [7, 2048]);
        assert_eq!(d.gate_up_shape(), [7, 16384]);
        assert_eq!(d.act_shape(), [7, 8192]);
        assert_eq!(d.logits_shape(), [1, 128256]);
        assert_eq!(d.attn_scratch_shape(), [32 + 128 * 32 * 66]);
    }

    #[test]
    fn workspace_bytes_exact() {
        let d = WorkspaceDims::new(&cfg(), 512, 1).unwrap();
        let f16_per_tok = 3 * 2048 + 3072 + 2048 + 16384 + 8192;
        assert_eq!(d.f16_elems(), 512 * f16_per_tok);
        let scratch = 32 + 128 * 32 * 66;
        assert_eq!(
            d.device_bytes(),
            512 * f16_per_tok * 2 + (128256 + scratch) * 4 + 512 * 4
        );
    }

    #[test]
    fn workspace_follows_config() {
        let c = LlamaConfig::from_json(
            r#"{"hidden_size":64,"intermediate_size":128,"num_hidden_layers":2,"num_attention_heads":4,
            "num_key_value_heads":2,"head_dim":16,"vocab_size":100,"rms_norm_eps":1e-5,"rope_theta":10000.0,
            "max_position_embeddings":256}"#,
        )
        .unwrap();
        let d = WorkspaceDims::new(&c, 5, 5).unwrap();
        assert_eq!(d.qkv_shape(), [5, 64 + 2 * 32]);
        assert_eq!(d.gate_up_shape(), [5, 256]);
        assert_eq!(d.logits_shape(), [5, 100]);
    }

    #[test]
    fn workspace_rejects_bad_sizes() {
        let c = cfg();
        assert!(WorkspaceDims::new(&c, 0, 0).is_err());
        assert!(WorkspaceDims::new(&c, 4, 0).is_err());
        assert!(WorkspaceDims::new(&c, 4, 5).is_err());
        assert!(WorkspaceDims::new(&c, 1 << 20, 1 << 20).is_err());
        assert!(WorkspaceDims::new(&c, usize::MAX, 1).is_err());
    }
}
