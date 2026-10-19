use super::config::LlamaConfig;

pub fn inv_freq(cfg: &LlamaConfig) -> Vec<f64> {
    let hd = cfg.head_dim;
    let base: Vec<f64> = (0..hd / 2)
        .map(|j| 1.0 / cfg.rope_theta.powf(2.0 * j as f64 / hd as f64))
        .collect();
    let Some(rs) = &cfg.rope_scaling else { return base };
    let orig = rs.original_max_position_embeddings as f64;
    let low_wavelen = orig / rs.low_freq_factor;
    let high_wavelen = orig / rs.high_freq_factor;
    base.iter()
        .map(|&f| {
            let wavelen = 2.0 * std::f64::consts::PI / f;
            if wavelen < high_wavelen {
                f // high frequency: unchanged
            } else if wavelen > low_wavelen {
                f / rs.factor // low frequency: fully scaled
            } else {
                let smooth = (orig / wavelen - rs.low_freq_factor) / (rs.high_freq_factor - rs.low_freq_factor);
                (1.0 - smooth) * f / rs.factor + smooth * f
            }
        })
        .collect()
}

pub fn cos_sin_table(cfg: &LlamaConfig, max_pos: usize) -> (Vec<f32>, Vec<f32>) {
    let inv = inv_freq(cfg);
    let half = inv.len();
    let (mut cos, mut sin) = (vec![0f32; max_pos * half], vec![0f32; max_pos * half]);
    for p in 0..max_pos {
        for (j, f) in inv.iter().enumerate() {
            let a = p as f64 * f;
            cos[p * half + j] = a.cos() as f32;
            sin[p * half + j] = a.sin() as f32;
        }
    }
    (cos, sin)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> LlamaConfig {
        LlamaConfig::from_json(
            r#"{"hidden_size":2048,"intermediate_size":8192,"num_hidden_layers":16,"num_attention_heads":32,
            "num_key_value_heads":8,"head_dim":64,"vocab_size":128256,"rms_norm_eps":1e-5,"rope_theta":500000.0,
            "max_position_embeddings":131072,"tie_word_embeddings":true,
            "rope_scaling":{"factor":32.0,"high_freq_factor":4.0,"low_freq_factor":1.0,
            "original_max_position_embeddings":8192,"rope_type":"llama3"}}"#,
        )
        .unwrap()
    }

    #[test]
    fn llama3_scaling_regimes() {
        let c = cfg();
        let scaled = inv_freq(&c);
        let mut plain = c.clone();
        plain.rope_scaling = None;
        let base = inv_freq(&plain);
        assert_eq!(scaled[0], base[0]);
        let last = base.len() - 1;
        assert!((scaled[last] - base[last] / 32.0).abs() < 1e-18);
        assert!(scaled.iter().zip(&base).all(|(s, b)| s <= b));
        assert!(scaled.windows(2).all(|w| w[0] > w[1]));
    }

    #[test]
    fn position_zero_is_identity() {
        let (c, s) = cos_sin_table(&cfg(), 4);
        assert!(c[..32].iter().all(|&x| x == 1.0) && s[..32].iter().all(|&x| x == 0.0));
    }
}
