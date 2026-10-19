use std::collections::HashMap;

use half::{bf16, f16};
use kvasir::gpu::Gpu;
use kvasir::model::{LlamaConfig, Weights};
use rand::{rngs::StdRng, Rng, SeedableRng};
use safetensors::tensor::TensorView;
use safetensors::Dtype;

const CFG: &str = r#"{"hidden_size":64,"intermediate_size":128,"num_hidden_layers":2,"num_attention_heads":4,
  "num_key_value_heads":2,"head_dim":16,"vocab_size":100,"rms_norm_eps":1e-5,"rope_theta":10000.0,
  "max_position_embeddings":256,"tie_word_embeddings":true}"#;

fn bytes(v: &[bf16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[test]
fn loads_and_fuses_tiny_checkpoint() {
    let cfg = LlamaConfig::from_json(CFG).unwrap();
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("tiny_ckpt");
    std::fs::create_dir_all(&dir).unwrap();
    let mut rng = StdRng::seed_from_u64(11);

    let (h, q, kv, inter, v) = (64usize, 64usize, 32usize, 128usize, 100usize);
    let mut shapes: Vec<(String, Vec<usize>)> = vec![
        ("model.embed_tokens.weight".into(), vec![v, h]),
        ("model.norm.weight".into(), vec![h]),
    ];
    for i in 0..2 {
        let p = format!("model.layers.{i}");
        for (n, s) in [
            ("input_layernorm.weight", vec![h]),
            ("self_attn.q_proj.weight", vec![q, h]),
            ("self_attn.k_proj.weight", vec![kv, h]),
            ("self_attn.v_proj.weight", vec![kv, h]),
            ("self_attn.o_proj.weight", vec![h, q]),
            ("post_attention_layernorm.weight", vec![h]),
            ("mlp.gate_proj.weight", vec![inter, h]),
            ("mlp.up_proj.weight", vec![inter, h]),
            ("mlp.down_proj.weight", vec![h, inter]),
        ] {
            shapes.push((format!("{p}.{n}"), s));
        }
    }
    let data: HashMap<String, (Vec<usize>, Vec<bf16>)> = shapes
        .into_iter()
        .map(|(n, s)| {
            let cnt = s.iter().product::<usize>();
            let d = (0..cnt).map(|_| bf16::from_f32(rng.random_range(-1.0..1.0))).collect();
            (n, (s, d))
        })
        .collect();
    let raw: HashMap<String, Vec<u8>> = data.iter().map(|(n, (_, d))| (n.clone(), bytes(d))).collect();
    let views: Vec<(String, TensorView)> = data
        .iter()
        .map(|(n, (s, _))| (n.clone(), TensorView::new(Dtype::BF16, s.clone(), &raw[n]).unwrap()))
        .collect();
    safetensors::serialize_to_file(views, None, &dir.join("model.safetensors")).unwrap();

    let gpu = Gpu::new(0).unwrap();
    let w = Weights::load(&gpu, &dir, &cfg).unwrap();

    let host = |t: &kvasir::gpu::DeviceTensor<f16>| t.to_host(&gpu).unwrap();
    let want = |names: &[&str]| -> Vec<f16> {
        names
            .iter()
            .flat_map(|n| data[*n].1.iter().map(|x| f16::from_f32(x.to_f32())))
            .collect()
    };
    assert_eq!(w.embed.shape(), &[v, h]);
    assert!(w.lm_head.is_none());
    assert_eq!(host(&w.embed), want(&["model.embed_tokens.weight"]));
    assert_eq!(host(&w.final_norm), want(&["model.norm.weight"]));
    let l1 = &w.layers[1];
    assert_eq!(l1.w_qkv.shape(), &[q + 2 * kv, h]);
    assert_eq!(
        host(&l1.w_qkv),
        want(&[
            "model.layers.1.self_attn.q_proj.weight",
            "model.layers.1.self_attn.k_proj.weight",
            "model.layers.1.self_attn.v_proj.weight"
        ])
    );
    assert_eq!(l1.w_gu.shape(), &[2 * inter, h]);
    assert_eq!(
        host(&l1.w_gu),
        want(&[
            "model.layers.1.mlp.gate_proj.weight",
            "model.layers.1.mlp.up_proj.weight"
        ])
    );
    assert_eq!(host(&l1.w_down), want(&["model.layers.1.mlp.down_proj.weight"]));
    assert_eq!(
        host(&w.layers[0].w_o),
        want(&["model.layers.0.self_attn.o_proj.weight"])
    );
    let params = v * h + h + 2 * (h + (q + 2 * kv) * h + h * q + h + 2 * inter * h + h * inter);
    assert_eq!(w.decode_bytes() / 2, params); // tied head counted once (embedding == head)
}

#[test]
fn missing_tensor_is_an_error() {
    let cfg = LlamaConfig::from_json(CFG).unwrap();
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("empty_ckpt");
    std::fs::create_dir_all(&dir).unwrap();
    let one = bytes(&[bf16::from_f32(1.0)]);
    let view = TensorView::new(Dtype::BF16, vec![1], &one).unwrap();
    safetensors::serialize_to_file(vec![("model.norm.weight", view)], None, &dir.join("model.safetensors")).unwrap();
    let gpu = Gpu::new(0).unwrap();
    assert!(Weights::load(&gpu, &dir, &cfg).is_err());
}
