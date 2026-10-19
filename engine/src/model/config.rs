use std::path::Path;

use serde::Deserialize;

use crate::error::{KvasirError, Result};

#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    pub rope_type: String,
    pub factor: f64,
    pub low_freq_factor: f64,
    pub high_freq_factor: f64,
    pub original_max_position_embeddings: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum OneOrMany {
    One(u32),
    Many(Vec<u32>),
}

#[derive(Debug, Deserialize)]
struct RawConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: Option<usize>,
    head_dim: Option<usize>,
    vocab_size: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
    rope_scaling: Option<serde_json::Value>,
    max_position_embeddings: usize,
    #[serde(default)]
    tie_word_embeddings: bool,
    bos_token_id: Option<u32>,
    eos_token_id: Option<OneOrMany>,
}

#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
    pub rope_scaling: Option<RopeScaling>,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    pub bos_token_id: Option<u32>,
    pub eos_token_ids: Vec<u32>,
}

impl LlamaConfig {
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(dir.join("config.json"))?;
        Self::from_json(&text)
    }

    pub fn from_json(text: &str) -> Result<Self> {
        let raw: RawConfig =
            serde_json::from_str(text).map_err(|e| KvasirError::Config(format!("config.json: {e}")))?;
        let num_kv_heads = raw.num_key_value_heads.unwrap_or(raw.num_attention_heads);
        let head_dim = raw.head_dim.unwrap_or(raw.hidden_size / raw.num_attention_heads);
        let rope_scaling = match raw.rope_scaling {
            None | Some(serde_json::Value::Null) => None,
            Some(v) => {
                let rs: RopeScaling =
                    serde_json::from_value(v).map_err(|e| KvasirError::Config(format!("rope_scaling: {e}")))?;
                if rs.rope_type != "llama3" {
                    return Err(KvasirError::Config(format!(
                        "unsupported rope_type {:?} (only \"llama3\")",
                        rs.rope_type
                    )));
                }
                Some(rs)
            }
        };
        let cfg = Self {
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            num_layers: raw.num_hidden_layers,
            num_heads: raw.num_attention_heads,
            num_kv_heads,
            head_dim,
            vocab_size: raw.vocab_size,
            rms_norm_eps: raw.rms_norm_eps as f32,
            rope_theta: raw.rope_theta,
            rope_scaling,
            max_position_embeddings: raw.max_position_embeddings,
            tie_word_embeddings: raw.tie_word_embeddings,
            bos_token_id: raw.bos_token_id,
            eos_token_ids: match raw.eos_token_id {
                None => vec![],
                Some(OneOrMany::One(x)) => vec![x],
                Some(OneOrMany::Many(v)) => v,
            },
        };
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        let bad = |m: String| Err(KvasirError::Config(m));
        if self.num_kv_heads == 0 || !self.num_heads.is_multiple_of(self.num_kv_heads) {
            return bad(format!(
                "num_heads {} not a multiple of num_kv_heads {}",
                self.num_heads, self.num_kv_heads
            ));
        }
        if !self.head_dim.is_multiple_of(2) || 128 % self.head_dim != 0 {
            return bad(format!(
                "head_dim {} unsupported (must be even and divide 128)",
                self.head_dim
            ));
        }
        if self.q_dim() != self.hidden_size {
            return bad(format!(
                "num_heads*head_dim = {} != hidden_size {}",
                self.q_dim(),
                self.hidden_size
            ));
        }
        Ok(())
    }

    pub fn q_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }
    pub fn qkv_dim(&self) -> usize {
        self.q_dim() + 2 * self.kv_dim()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LLAMA_3_2_1B: &str = r#"{
      "bos_token_id": 128000, "eos_token_id": 128001, "head_dim": 64, "hidden_size": 2048,
      "intermediate_size": 8192, "max_position_embeddings": 131072, "num_attention_heads": 32,
      "num_hidden_layers": 16, "num_key_value_heads": 8, "rms_norm_eps": 1e-05,
      "rope_scaling": {"factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192, "rope_type": "llama3"},
      "rope_theta": 500000.0, "tie_word_embeddings": true, "vocab_size": 128256 }"#;

    #[test]
    fn parses_llama_3_2_1b() {
        let c = LlamaConfig::from_json(LLAMA_3_2_1B).unwrap();
        assert_eq!(
            (c.qkv_dim(), c.kv_dim(), c.eos_token_ids.clone()),
            (3072, 512, vec![128001])
        );
        assert!(c.tie_word_embeddings);
        assert_eq!(c.rope_scaling.unwrap().factor, 32.0);
    }

    #[test]
    fn rejects_unknown_rope_type() {
        let s = LLAMA_3_2_1B.replace("llama3", "yarn");
        assert!(LlamaConfig::from_json(&s).is_err());
    }
}
