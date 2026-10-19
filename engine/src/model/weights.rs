use std::collections::HashMap;
use std::path::{Path, PathBuf};

use half::{bf16, f16};
use memmap2::Mmap;
use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};

use super::config::LlamaConfig;
use crate::error::{KvasirError, Result};
use crate::gpu::{DeviceTensor, Gpu};

pub struct LayerWeights {
    pub attn_norm: DeviceTensor<f16>, // [hidden]
    pub w_qkv: DeviceTensor<f16>,     // [q_dim + 2*kv_dim, hidden]  rows: q | k | v
    pub w_o: DeviceTensor<f16>,       // [hidden, q_dim]
    pub mlp_norm: DeviceTensor<f16>,  // [hidden]
    pub w_gu: DeviceTensor<f16>,      // [2*inter, hidden]           rows: gate | up
    pub w_down: DeviceTensor<f16>,    // [hidden, inter]
}

pub struct Weights {
    pub embed: DeviceTensor<f16>,           // [vocab, hidden]; also the LM head when tied
    pub lm_head: Option<DeviceTensor<f16>>, // [vocab, hidden], only when not tied
    pub layers: Vec<LayerWeights>,
    pub final_norm: DeviceTensor<f16>, // [hidden]
}

impl Weights {
    pub fn head(&self) -> &DeviceTensor<f16> {
        self.lm_head.as_ref().unwrap_or(&self.embed)
    }

    pub fn decode_bytes(&self) -> usize {
        let mut n = self.final_norm.numel() + self.head().numel();
        for l in &self.layers {
            n += l.attn_norm.numel()
                + l.w_qkv.numel()
                + l.w_o.numel()
                + l.mlp_norm.numel()
                + l.w_gu.numel()
                + l.w_down.numel();
        }
        n * std::mem::size_of::<f16>()
    }

    pub fn load(gpu: &Gpu, dir: &Path, cfg: &LlamaConfig) -> Result<Self> {
        let files = ShardSet::open(dir)?;
        let st = files.tensors()?;
        let mut loader = Loader {
            gpu,
            st: &st,
            bad: gpu.stream.alloc_zeros::<u32>(1)?,
        };
        let (h, q, kv, inter, v) = (
            cfg.hidden_size,
            cfg.q_dim(),
            cfg.kv_dim(),
            cfg.intermediate_size,
            cfg.vocab_size,
        );

        let embed = loader.load_into(&[("model.embed_tokens.weight", &[v, h])])?;
        let lm_head = if cfg.tie_word_embeddings {
            None
        } else {
            Some(loader.load_into(&[("lm_head.weight", &[v, h])])?)
        };
        let final_norm = loader.load_into(&[("model.norm.weight", &[h])])?;

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let p = format!("model.layers.{i}");
            let name = |s: &str| format!("{p}.{s}");
            layers.push(LayerWeights {
                attn_norm: loader.load_into(&[(&name("input_layernorm.weight"), &[h])])?,
                w_qkv: loader.load_into(&[
                    (&name("self_attn.q_proj.weight"), &[q, h]),
                    (&name("self_attn.k_proj.weight"), &[kv, h]),
                    (&name("self_attn.v_proj.weight"), &[kv, h]),
                ])?,
                w_o: loader.load_into(&[(&name("self_attn.o_proj.weight"), &[h, q])])?,
                mlp_norm: loader.load_into(&[(&name("post_attention_layernorm.weight"), &[h])])?,
                w_gu: loader.load_into(&[
                    (&name("mlp.gate_proj.weight"), &[inter, h]),
                    (&name("mlp.up_proj.weight"), &[inter, h]),
                ])?,
                w_down: loader.load_into(&[(&name("mlp.down_proj.weight"), &[h, inter])])?,
            });
        }
        let bad = gpu.stream.clone_dtoh(&loader.bad)?[0];
        if bad != 0 {
            return Err(KvasirError::Config(format!(
                "{bad} weights are non-finite or outside the FP16 range"
            )));
        }
        Ok(Self {
            embed,
            lm_head,
            layers,
            final_norm,
        })
    }
}

struct ShardSet {
    maps: Vec<Mmap>,
}

impl ShardSet {
    fn open(dir: &Path) -> Result<Self> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)?
            .collect::<std::io::Result<Vec<_>>>()?
            .into_iter()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("safetensors"))
            .collect();
        paths.sort();
        if paths.is_empty() {
            return Err(KvasirError::Config(format!(
                "no .safetensors files in {}",
                dir.display()
            )));
        }
        let mut maps = Vec::new();
        for p in paths {
            let f = std::fs::File::open(&p)?;
            // SAFETY: read-only map; the checkpoint must not be truncated while running.
            maps.push(unsafe { Mmap::map(&f)? });
        }
        Ok(Self { maps })
    }

    fn tensors(&self) -> Result<HashMap<String, TensorView<'_>>> {
        let mut all = HashMap::new();
        for m in &self.maps {
            let st = SafeTensors::deserialize(m)?;
            for (name, view) in st.tensors() {
                all.insert(name, view);
            }
        }
        Ok(all)
    }
}

struct Loader<'a> {
    gpu: &'a Gpu,
    st: &'a HashMap<String, TensorView<'a>>,
    bad: cudarc::driver::CudaSlice<u32>,
}

impl Loader<'_> {
    fn load_into(&mut self, parts: &[(&str, &[usize])]) -> Result<DeviceTensor<f16>> {
        let rows: usize = parts.iter().map(|(_, s)| s[0]).sum();
        let mut shape = parts[0].1.to_vec();
        shape[0] = rows;
        let mut out = DeviceTensor::<f16>::zeros(self.gpu, &shape)?;
        let mut offset = 0usize;
        for (name, want) in parts {
            let view = self
                .st
                .get(*name)
                .ok_or_else(|| KvasirError::Config(format!("missing tensor {name}")))?;
            if view.shape() != *want {
                return Err(KvasirError::Shape(format!(
                    "{name}: shape {:?}, expected {want:?}",
                    view.shape()
                )));
            }
            let n: usize = want.iter().product();
            self.convert_to(&mut out.buf.slice_mut(offset..offset + n), view, name)?;
            offset += n;
        }
        Ok(out)
    }

    fn convert_to(
        &mut self,
        dst: &mut cudarc::driver::CudaViewMut<f16>,
        view: &TensorView<'_>,
        name: &str,
    ) -> Result<()> {
        let bytes = view.data();
        match view.dtype() {
            Dtype::BF16 => {
                let host: Vec<bf16> = bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| bf16::from_le_bytes(*c))
                    .collect();
                let staged = self.gpu.stream.clone_htod(&host)?;
                self.gpu
                    .bf16_to_f16(&staged.as_view(), dst, &mut self.bad.as_view_mut())?;
                self.gpu.synchronize()?; // `staged` must outlive the kernel
            }
            Dtype::F16 => {
                let host: Vec<f16> = bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| f16::from_le_bytes(*c))
                    .collect();
                check_finite(&host, name)?;
                self.gpu.stream.memcpy_htod(&host, dst)?;
            }
            Dtype::F32 => {
                let host: Vec<f16> = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f16::from_f32(f32::from_le_bytes(*c)))
                    .collect();
                check_finite(&host, name)?;
                self.gpu.stream.memcpy_htod(&host, dst)?;
            }
            other => return Err(KvasirError::Config(format!("{name}: unsupported dtype {other:?}"))),
        }
        Ok(())
    }
}

fn check_finite(host: &[f16], name: &str) -> Result<()> {
    match host.iter().filter(|x| !x.is_finite()).count() {
        0 => Ok(()),
        n => Err(KvasirError::Config(format!(
            "{name}: {n} values are non-finite or outside the FP16 range"
        ))),
    }
}
