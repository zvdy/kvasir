use half::f16;
use kvasir::gpu::Gpu;
use kvasir::kv::KvCache;
use kvasir::model::LlamaConfig;
use kvasir::KvasirError;
use rand::{rngs::StdRng, RngExt, SeedableRng};

const LAYERS: usize = 3;
const N_Q: usize = 4;
const N_KV: usize = 2;
const HD: usize = 64;
const MAX_SEQ: usize = 16;
const STRIDE: usize = (N_Q + 2 * N_KV) * HD;

fn gpu() -> Gpu {
    Gpu::new(0).expect("CUDA init")
}

fn rand_f16(rng: &mut StdRng, n: usize, amp: f32) -> Vec<f16> {
    (0..n).map(|_| f16::from_f32(rng.random_range(-amp..amp))).collect()
}

struct Mirror {
    k: Vec<f16>,
    v: Vec<f16>,
}

impl Mirror {
    fn new() -> Self {
        let n = LAYERS * N_KV * MAX_SEQ * HD;
        let z = f16::from_f32(0.0);
        Self {
            k: vec![z; n],
            v: vec![z; n],
        }
    }

    fn idx(layer: usize, h: usize, p: usize, d: usize) -> usize {
        ((layer * N_KV + h) * MAX_SEQ + p) * HD + d
    }

    fn append(&mut self, layer: usize, qkv: &[f16], t: usize, pos0: usize) {
        for i in 0..t {
            for h in 0..N_KV {
                for d in 0..HD {
                    let j = Self::idx(layer, h, pos0 + i, d);
                    self.k[j] = qkv[i * STRIDE + N_Q * HD + h * HD + d];
                    self.v[j] = qkv[i * STRIDE + (N_Q + N_KV) * HD + h * HD + d];
                }
            }
        }
    }
}

fn step(g: &Gpu, cache: &mut KvCache, mirror: &mut Mirror, rng: &mut StdRng, t: usize) {
    let pos0 = cache.len();
    for layer in 0..LAYERS {
        let qkv = rand_f16(rng, t * STRIDE, 2.0);
        let d = g.stream.clone_htod(&qkv).unwrap();
        cache.append(g, layer, &d.as_view(), t, N_Q).unwrap();
        mirror.append(layer, &qkv, t, pos0);
    }
    cache.advance(t).unwrap();
}

fn assert_matches(g: &Gpu, cache: &KvCache, mirror: &Mirror, what: &str) {
    let (k, v) = (cache.k().to_host(g).unwrap(), cache.v().to_host(g).unwrap());
    for (j, (got, want)) in k.iter().zip(&mirror.k).enumerate() {
        assert_eq!(got, want, "{what}: K mismatch at flat index {j}");
    }
    for (j, (got, want)) in v.iter().zip(&mirror.v).enumerate() {
        assert_eq!(got, want, "{what}: V mismatch at flat index {j}");
    }
}

#[test]
fn layout_and_shape() {
    let g = gpu();
    let cache = KvCache::new(&g, LAYERS, N_KV, MAX_SEQ, HD).unwrap();
    assert_eq!(cache.k().shape(), &[LAYERS, N_KV, MAX_SEQ, HD]);
    assert_eq!(cache.v().shape(), &[LAYERS, N_KV, MAX_SEQ, HD]);
    assert_eq!(cache.layer_numel(), N_KV * MAX_SEQ * HD);
    let (k, v) = cache.layer(LAYERS - 1).unwrap();
    assert_eq!((k.len(), v.len()), (cache.layer_numel(), cache.layer_numel()));
    assert!(cache.is_empty());
    assert_eq!(cache.remaining(), MAX_SEQ);
}

#[test]
fn appends_at_several_positions() {
    let g = gpu();
    let mut rng = StdRng::seed_from_u64(21);
    let mut cache = KvCache::new(&g, LAYERS, N_KV, MAX_SEQ, HD).unwrap();
    let mut mirror = Mirror::new();

    for t in [5usize, 1, 3, 1] {
        let pos0 = cache.len();
        step(&g, &mut cache, &mut mirror, &mut rng, t);
        assert_matches(&g, &cache, &mirror, &format!("t={t} pos0={pos0}"));
    }
    assert_eq!(cache.len(), 10);

    cache.advance(4).unwrap();
    step(&g, &mut cache, &mut mirror, &mut rng, 2);
    assert_matches(&g, &cache, &mirror, "fill to max_seq");
    assert_eq!(cache.len(), MAX_SEQ);
    assert_eq!(cache.remaining(), 0);
}

#[test]
fn truncate_then_overwrite() {
    let g = gpu();
    let mut rng = StdRng::seed_from_u64(22);
    let mut cache = KvCache::new(&g, LAYERS, N_KV, MAX_SEQ, HD).unwrap();
    let mut mirror = Mirror::new();
    step(&g, &mut cache, &mut mirror, &mut rng, 6);
    cache.truncate(3).unwrap();
    step(&g, &mut cache, &mut mirror, &mut rng, 2);
    assert_matches(&g, &cache, &mirror, "overwrite after truncate");
    assert_eq!(cache.len(), 5);
    assert!(cache.truncate(6).is_err());
    cache.reset();
    assert!(cache.is_empty());
}

#[test]
fn overflow_and_bad_layer_are_errors() {
    let g = gpu();
    let mut rng = StdRng::seed_from_u64(23);
    let mut cache = KvCache::new(&g, LAYERS, N_KV, MAX_SEQ, HD).unwrap();
    let mut mirror = Mirror::new();
    step(&g, &mut cache, &mut mirror, &mut rng, MAX_SEQ - 1);

    let qkv = g.stream.clone_htod(&rand_f16(&mut rng, 2 * STRIDE, 2.0)).unwrap();
    let mut out = g.stream.alloc_zeros::<f16>(2 * N_Q * HD).unwrap();
    assert!(matches!(
        cache.append(&g, 0, &qkv.as_view(), 2, N_Q),
        Err(KvasirError::Shape(_))
    ));
    assert!(matches!(
        cache.attention(&g, 0, &qkv.as_view(), &mut out.as_view_mut(), 2, N_Q),
        Err(KvasirError::Shape(_))
    ));
    assert!(cache.advance(2).is_err());
    assert!(cache.append(&g, 0, &qkv.as_view(), 0, N_Q).is_err());
    assert!(cache.layer(LAYERS).is_err());
    assert!(cache.layer_mut(LAYERS).is_err());
    assert_eq!(cache.len(), MAX_SEQ - 1);
    assert_matches(&g, &cache, &mirror, "failed calls leave cache untouched");

    assert!(KvCache::new(&g, LAYERS, 0, MAX_SEQ, HD).is_err());
}

#[test]
fn from_config_dims() {
    let g = gpu();
    let cfg = LlamaConfig::from_json(
        r#"{"hidden_size":64,"intermediate_size":128,"num_hidden_layers":2,"num_attention_heads":4,
        "num_key_value_heads":2,"head_dim":16,"vocab_size":100,"rms_norm_eps":1e-5,"rope_theta":10000.0,
        "max_position_embeddings":256,"tie_word_embeddings":true}"#,
    )
    .unwrap();
    let cache = KvCache::from_config(&g, &cfg, 128).unwrap();
    assert_eq!(cache.k().shape(), &[2, 2, 128, 16]);
    assert!(KvCache::from_config(&g, &cfg, 257).is_err());
}

#[test]
fn attention_reads_layer_view() {
    let g = gpu();
    let mut rng = StdRng::seed_from_u64(24);
    let mut cache = KvCache::new(&g, LAYERS, N_KV, MAX_SEQ, HD).unwrap();
    let mut mirror = Mirror::new();
    step(&g, &mut cache, &mut mirror, &mut rng, 7);

    let layer = 1;
    let pos0 = cache.len();
    let qkv = rand_f16(&mut rng, STRIDE, 2.0);
    let dqkv = g.stream.clone_htod(&qkv).unwrap();
    cache.append(&g, layer, &dqkv.as_view(), 1, N_Q).unwrap();
    mirror.append(layer, &qkv, 1, pos0);
    let mut dout = g.stream.alloc_zeros::<f16>(N_Q * HD).unwrap();
    cache
        .attention(&g, layer, &dqkv.as_view(), &mut dout.as_view_mut(), 1, N_Q)
        .unwrap();
    let got: Vec<f32> = g.stream.clone_dtoh(&dout).unwrap().iter().map(|x| x.to_f32()).collect();

    let f = |x: f16| x.to_f32() as f64;
    let scale = 1.0 / (HD as f64).sqrt();
    let kv_len = pos0 + 1;
    for h in 0..N_Q {
        let kvh = h / (N_Q / N_KV);
        let q = &qkv[h * HD..(h + 1) * HD];
        let scores: Vec<f64> = (0..kv_len)
            .map(|j| {
                (0..HD)
                    .map(|d| f(q[d]) * f(mirror.k[Mirror::idx(layer, kvh, j, d)]))
                    .sum::<f64>()
                    * scale
            })
            .collect();
        let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let ps: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
        let denom: f64 = ps.iter().sum();
        for d in 0..HD {
            let want = (0..kv_len)
                .map(|j| ps[j] * f(mirror.v[Mirror::idx(layer, kvh, j, d)]))
                .sum::<f64>()
                / denom;
            let g = got[h * HD + d] as f64;
            assert!(
                (g - want).abs() <= 1e-3 + 2e-3 * want.abs(),
                "attention h{h} d{d}: got {g}, want {want}"
            );
        }
    }
}
