use kvasir::gpu::Gpu;
use kvasir::model::{LlamaConfig, Workspace, WorkspaceDims};
use rand::{rngs::StdRng, RngExt, SeedableRng};

const CFG: &str = r#"{"hidden_size":2048,"intermediate_size":8192,"num_hidden_layers":16,"num_attention_heads":32,
  "num_key_value_heads":8,"head_dim":64,"vocab_size":128256,"rms_norm_eps":1e-5,"rope_theta":500000.0,
  "max_position_embeddings":131072,"tie_word_embeddings":true}"#;

#[test]
fn workspace_allocates_every_buffer_at_full_size() {
    let cfg = LlamaConfig::from_json(CFG).unwrap();
    let dims = WorkspaceDims::new(&cfg, 64, 2).unwrap();
    let gpu = Gpu::new(0).unwrap();
    let ws = Workspace::new(&gpu, dims).unwrap();

    assert_eq!(ws.ids.shape(), &dims.ids_shape());
    assert_eq!(ws.hidden.shape(), &dims.hidden_shape());
    assert_eq!(ws.normed.shape(), &dims.hidden_shape());
    assert_eq!(ws.proj.shape(), &dims.hidden_shape());
    assert_eq!(ws.qkv.shape(), &dims.qkv_shape());
    assert_eq!(ws.attn.shape(), &dims.attn_shape());
    assert_eq!(ws.gate_up.shape(), &dims.gate_up_shape());
    assert_eq!(ws.act.shape(), &dims.act_shape());
    assert_eq!(ws.logits.shape(), &dims.logits_shape());
    assert_eq!(ws.attn_scratch.shape(), &dims.attn_scratch_shape());
    assert!(ws.attn_scratch.to_host(&gpu).unwrap().iter().all(|&x| x == 0.0));

    let f16s = [
        &ws.hidden,
        &ws.normed,
        &ws.proj,
        &ws.qkv,
        &ws.attn,
        &ws.gate_up,
        &ws.act,
    ];
    let allocated = f16s.iter().map(|t| t.numel() * 2).sum::<usize>()
        + (ws.logits.numel() + ws.attn_scratch.numel()) * 4
        + ws.ids.numel() * 4;
    assert_eq!(allocated, dims.device_bytes());
}

#[test]
fn workspace_ids_and_logits_round_trip() {
    let cfg = LlamaConfig::from_json(CFG).unwrap();
    let dims = WorkspaceDims::new(&cfg, 16, 3).unwrap();
    let gpu = Gpu::new(0).unwrap();
    let mut ws = Workspace::new(&gpu, dims).unwrap();
    let mut rng = StdRng::seed_from_u64(7);

    let ids: Vec<u32> = (0..5).map(|_| rng.random_range(0..128256u32)).collect();
    ws.upload_ids(&gpu, &ids).unwrap();
    let got = ws.ids.to_host(&gpu).unwrap();
    assert_eq!(&got[..5], ids.iter().map(|&i| i as i32).collect::<Vec<_>>().as_slice());
    assert!(got[5..].iter().all(|&x| x == 0));

    assert!(ws.upload_ids(&gpu, &[0, 128256]).is_err());
    assert!(ws.upload_ids(&gpu, &[]).is_err());
    assert!(ws.upload_ids(&gpu, &[1; 17]).is_err());

    let vals: Vec<f32> = (0..3 * 128256).map(|_| rng.random_range(-30.0..30.0)).collect();
    gpu.stream.memcpy_htod(&vals, &mut ws.logits.buf).unwrap();
    assert_eq!(ws.logits_to_host(&gpu, 2).unwrap(), &vals[..2 * 128256]);
    assert_eq!(ws.logits_to_host(&gpu, 3).unwrap(), vals.as_slice());
    assert!(ws.logits_to_host(&gpu, 0).is_err());
    assert!(ws.logits_to_host(&gpu, 4).is_err());
}
