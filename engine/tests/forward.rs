use std::collections::HashMap;
use std::path::PathBuf;

use half::{bf16, f16};
use kvasir::gpu::Gpu;
use kvasir::model::{Llama, LlamaConfig, Logits, Weights};
use rand::{rngs::StdRng, RngExt, SeedableRng};
use safetensors::tensor::TensorView;
use safetensors::Dtype;

const CFG: &str = r#"{"hidden_size":64,"intermediate_size":128,"num_hidden_layers":2,"num_attention_heads":4,
  "num_key_value_heads":2,"head_dim":16,"vocab_size":100,"rms_norm_eps":1e-5,"rope_theta":10000.0,
  "max_position_embeddings":256,"tie_word_embeddings":true}"#;

const MAX_SEQ: usize = 40;
const MAX_TOKENS: usize = 24;
const MAX_HEAD_ROWS: usize = 24;

// Max |logit error| (logit rms ≈ 5). Derivation and measurements: DESIGN.md §6, "Forward-pass test thresholds".
const TOL: f64 = 5e-2;
const TOL_HEAD_ONLY: f64 = 1e-3;

struct HostLayer {
    attn_norm: Vec<f64>,
    wq: Vec<f64>,
    wk: Vec<f64>,
    wv: Vec<f64>,
    wo: Vec<f64>,
    mlp_norm: Vec<f64>,
    wg: Vec<f64>,
    wu: Vec<f64>,
    wd: Vec<f64>,
}

struct HostModel {
    cfg: LlamaConfig,
    embed: Vec<f64>,
    final_norm: Vec<f64>,
    layers: Vec<HostLayer>,
}

fn rand_bf16(rng: &mut StdRng, n: usize, lo: f32, hi: f32) -> Vec<bf16> {
    (0..n).map(|_| bf16::from_f32(rng.random_range(lo..hi))).collect()
}

fn to_f64(v: &[bf16]) -> Vec<f64> {
    v.iter().map(|x| x.to_f64()).collect()
}

// Writes a random BF16 checkpoint for CFG into a per-test directory; returns the exact host copy.
fn write_checkpoint(name: &str, seed: u64) -> (PathBuf, HostModel) {
    let cfg = LlamaConfig::from_json(CFG).unwrap();
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("forward_{name}"));
    std::fs::create_dir_all(&dir).unwrap();
    let mut rng = StdRng::seed_from_u64(seed);
    let (h, q, kv, inter, v) = (
        cfg.hidden_size,
        cfg.q_dim(),
        cfg.kv_dim(),
        cfg.intermediate_size,
        cfg.vocab_size,
    );
    let lin = |rng: &mut StdRng, out: usize, inp: usize| {
        let a = (3.0 / inp as f32).sqrt();
        rand_bf16(rng, out * inp, -a, a)
    };
    let mut tensors: Vec<(String, Vec<usize>, Vec<bf16>)> = vec![
        (
            "model.embed_tokens.weight".into(),
            vec![v, h],
            rand_bf16(&mut rng, v * h, -1.0, 1.0),
        ),
        ("model.norm.weight".into(), vec![h], rand_bf16(&mut rng, h, 0.5, 1.5)),
    ];
    for i in 0..cfg.num_layers {
        let p = format!("model.layers.{i}");
        let parts = [
            ("input_layernorm.weight", vec![h], rand_bf16(&mut rng, h, 0.5, 1.5)),
            ("self_attn.q_proj.weight", vec![q, h], lin(&mut rng, q, h)),
            ("self_attn.k_proj.weight", vec![kv, h], lin(&mut rng, kv, h)),
            ("self_attn.v_proj.weight", vec![kv, h], lin(&mut rng, kv, h)),
            ("self_attn.o_proj.weight", vec![h, q], lin(&mut rng, h, q)),
            (
                "post_attention_layernorm.weight",
                vec![h],
                rand_bf16(&mut rng, h, 0.5, 1.5),
            ),
            ("mlp.gate_proj.weight", vec![inter, h], lin(&mut rng, inter, h)),
            ("mlp.up_proj.weight", vec![inter, h], lin(&mut rng, inter, h)),
            ("mlp.down_proj.weight", vec![h, inter], lin(&mut rng, h, inter)),
        ];
        for (n, s, d) in parts {
            tensors.push((format!("{p}.{n}"), s, d));
        }
    }
    let raw: Vec<Vec<u8>> = tensors
        .iter()
        .map(|(_, _, d)| d.iter().flat_map(|x| x.to_le_bytes()).collect())
        .collect();
    let views: Vec<(String, TensorView)> = tensors
        .iter()
        .zip(&raw)
        .map(|((n, s, _), r)| (n.clone(), TensorView::new(Dtype::BF16, s.clone(), r).unwrap()))
        .collect();
    safetensors::serialize_to_file(views, None, &dir.join("model.safetensors")).unwrap();

    let m: HashMap<String, Vec<f64>> = tensors.iter().map(|(n, _, d)| (n.clone(), to_f64(d))).collect();
    let get = |n: &str| m[n].clone();
    let layers = (0..cfg.num_layers)
        .map(|i| {
            let g = |n: &str| get(&format!("model.layers.{i}.{n}"));
            HostLayer {
                attn_norm: g("input_layernorm.weight"),
                wq: g("self_attn.q_proj.weight"),
                wk: g("self_attn.k_proj.weight"),
                wv: g("self_attn.v_proj.weight"),
                wo: g("self_attn.o_proj.weight"),
                mlp_norm: g("post_attention_layernorm.weight"),
                wg: g("mlp.gate_proj.weight"),
                wu: g("mlp.up_proj.weight"),
                wd: g("mlp.down_proj.weight"),
            }
        })
        .collect();
    let host = HostModel {
        embed: get("model.embed_tokens.weight"),
        final_norm: get("model.norm.weight"),
        layers,
        cfg,
    };
    (dir, host)
}

fn build(gpu: &Gpu, name: &str, seed: u64) -> (Llama, HostModel) {
    let (dir, host) = write_checkpoint(name, seed);
    let cfg = LlamaConfig::from_json(CFG).unwrap();
    let w = Weights::load(gpu, &dir, &cfg).unwrap();
    let model = Llama::new(gpu, cfg, w, MAX_SEQ, MAX_TOKENS, MAX_HEAD_ROWS).unwrap();
    (model, host)
}

// `emulate` rounds to FP16 at every point where the GPU stores an FP16 activation.
fn round(x: f64, emulate: bool) -> f64 {
    if emulate {
        f16::from_f64(x).to_f64()
    } else {
        x
    }
}

fn rmsnorm(x: &[f64], w: &[f64], eps: f64, e: bool) -> Vec<f64> {
    let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + eps).sqrt();
    x.iter().zip(w).map(|(v, w)| round(v * inv * w, e)).collect()
}

// y = W x with W row-major [out, in].
fn matvec(w: &[f64], x: &[f64], e: bool) -> Vec<f64> {
    w.chunks(x.len())
        .map(|row| round(row.iter().zip(x).map(|(a, b)| a * b).sum(), e))
        .collect()
}

fn rope(v: &mut [f64], pos: usize, head_dim: usize, theta: f64, e: bool, interleaved: bool) {
    let half = head_dim / 2;
    for head in v.chunks_mut(head_dim) {
        for j in 0..half {
            let a = pos as f64 * theta.powf(-2.0 * j as f64 / head_dim as f64);
            let (c, s) = (a.cos(), a.sin());
            let (i0, i1) = if interleaved { (2 * j, 2 * j + 1) } else { (j, j + half) };
            let (x0, x1) = (head[i0], head[i1]);
            head[i0] = round(x0 * c - x1 * s, e);
            head[i1] = round(x1 * c + x0 * s, e);
        }
    }
}

// Exact f64, FP16-emulated, or a deliberately wrong variant used to show the tolerance still detects bugs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Ref {
    Exact,
    Fp16,
    GqaModulo,
    CausalLeak,
    RopeInterleaved,
}

// Full causal forward over `ids` from position 0; logits [ids.len(), vocab].
fn reference(m: &HostModel, ids: &[u32], r: Ref) -> Vec<Vec<f64>> {
    let e = r == Ref::Fp16;
    let c = &m.cfg;
    let (h, hd, n_q, n_kv, inter) = (
        c.hidden_size,
        c.head_dim,
        c.num_heads,
        c.num_kv_heads,
        c.intermediate_size,
    );
    let eps = c.rms_norm_eps as f64;
    let mut x: Vec<Vec<f64>> = ids
        .iter()
        .map(|&id| m.embed[id as usize * h..(id as usize + 1) * h].to_vec())
        .collect();
    for l in &m.layers {
        let mut qs = vec![];
        let mut ks = vec![];
        let mut vs = vec![];
        for (pos, xi) in x.iter().enumerate() {
            let n = rmsnorm(xi, &l.attn_norm, eps, e);
            let (mut q, mut k) = (matvec(&l.wq, &n, e), matvec(&l.wk, &n, e));
            let il = r == Ref::RopeInterleaved;
            rope(&mut q, pos, hd, c.rope_theta, e, il);
            rope(&mut k, pos, hd, c.rope_theta, e, il);
            qs.push(q);
            ks.push(k);
            vs.push(matvec(&l.wv, &n, e));
        }
        let n = x.len();
        for (i, xi) in x.iter_mut().enumerate() {
            let last = if r == Ref::CausalLeak { (i + 1).min(n - 1) } else { i };
            let mut attn = vec![0.0; n_q * hd];
            for hq in 0..n_q {
                let kh = if r == Ref::GqaModulo {
                    hq % n_kv
                } else {
                    hq / (n_q / n_kv)
                };
                let q = &qs[i][hq * hd..(hq + 1) * hd];
                let scores: Vec<f64> = (0..=last)
                    .map(|j| {
                        let k = &ks[j][kh * hd..(kh + 1) * hd];
                        q.iter().zip(k).map(|(a, b)| a * b).sum::<f64>() / (hd as f64).sqrt()
                    })
                    .collect();
                let mx = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let p: Vec<f64> = scores.iter().map(|s| (s - mx).exp()).collect();
                let z: f64 = p.iter().sum();
                for d in 0..hd {
                    let o: f64 = (0..=last).map(|j| p[j] * vs[j][kh * hd + d]).sum();
                    attn[hq * hd + d] = round(o / z, e);
                }
            }
            let o = matvec(&l.wo, &attn, e);
            for (a, b) in xi.iter_mut().zip(&o) {
                *a = round(*a + b, e);
            }
            let n = rmsnorm(xi, &l.mlp_norm, eps, e);
            let (g, u) = (matvec(&l.wg, &n, e), matvec(&l.wu, &n, e));
            let act: Vec<f64> = (0..inter)
                .map(|k| round(g[k] / (1.0 + (-g[k]).exp()) * u[k], e))
                .collect();
            let d = matvec(&l.wd, &act, e);
            for (a, b) in xi.iter_mut().zip(&d) {
                *a = round(*a + b, e);
            }
        }
    }
    x.iter()
        .map(|xi| matvec(&m.embed, &rmsnorm(xi, &m.final_norm, eps, e), false))
        .collect()
}

fn random_ids(rng: &mut StdRng, n: usize) -> Vec<u32> {
    (0..n).map(|_| rng.random_range(0..100u32)).collect()
}

fn max_err(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    got.iter()
        .zip(want)
        .map(|(g, w)| (*g as f64 - w).abs())
        .fold(0.0, f64::max)
}

fn argmax<T: Copy + PartialOrd>(v: &[T]) -> usize {
    (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b })
}

fn check(what: &str, got: &[f32], want: &[f64], tol: f64) -> f64 {
    let err = max_err(got, want);
    eprintln!("{what}: max |logit err| {err:.3e} (tol {tol:.1e})");
    assert!(err <= tol, "{what}: max |logit err| {err} > {tol}");
    err
}

#[test]
fn prefill_matches_cpu_reference() {
    let gpu = Gpu::new(0).unwrap();
    let (mut model, host) = build(&gpu, "prefill", 101);
    let ids = random_ids(&mut StdRng::seed_from_u64(1), MAX_TOKENS);
    let got = model.forward(&gpu, &ids, Logits::All).unwrap().to_vec();
    assert_eq!(got.len(), ids.len() * 100);
    assert_eq!(model.len(), ids.len());

    let exact: Vec<f64> = reference(&host, &ids, Ref::Exact).concat();
    let rms = (exact.iter().map(|x| x * x).sum::<f64>() / exact.len() as f64).sqrt();
    eprintln!("logit rms {rms:.3}");
    check("prefill vs exact f64 ref", &got, &exact, TOL);
    check(
        "prefill vs fp16-emulated ref",
        &got,
        &reference(&host, &ids, Ref::Fp16).concat(),
        TOL,
    );
    for (i, (g, w)) in got.chunks(100).zip(exact.chunks(100)).enumerate() {
        let mut sorted = w.to_vec();
        sorted.sort_by(|a, b| b.total_cmp(a));
        if sorted[0] - sorted[1] > 2.0 * TOL {
            assert_eq!(argmax(g), argmax(w), "top-1 differs at position {i}");
        }
    }
    for bug in [Ref::GqaModulo, Ref::CausalLeak, Ref::RopeInterleaved] {
        let err = max_err(&got, &reference(&host, &ids, bug).concat());
        eprintln!("vs {bug:?} ref: max |logit err| {err:.3e}");
        assert!(err > 10.0 * TOL, "{bug:?} is not distinguishable at TOL ({err})");
    }
}

#[test]
fn sequential_decode_matches_prefill() {
    let gpu = Gpu::new(0).unwrap();
    let (mut model, host) = build(&gpu, "decode", 102);
    let ids = random_ids(&mut StdRng::seed_from_u64(2), MAX_TOKENS);
    let prefill = model.forward(&gpu, &ids, Logits::All).unwrap().to_vec();
    model.reset();
    assert!(model.is_empty());

    let mut decoded = Vec::with_capacity(prefill.len());
    for (i, &id) in ids.iter().enumerate() {
        decoded.extend_from_slice(model.forward(&gpu, &[id], Logits::Last).unwrap());
        assert_eq!(model.len(), i + 1);
    }
    let prefill64: Vec<f64> = prefill.iter().map(|&x| x as f64).collect();
    check("decode vs prefill (gpu)", &decoded, &prefill64, TOL);
    check(
        "decode vs fp16-emulated ref",
        &decoded,
        &reference(&host, &ids, Ref::Fp16).concat(),
        TOL,
    );
}

#[test]
fn prefill_then_decode_fills_cache() {
    let gpu = Gpu::new(0).unwrap();
    let (mut model, host) = build(&gpu, "continuity", 103);
    let ids = random_ids(&mut StdRng::seed_from_u64(3), MAX_SEQ);
    let p = 17;
    let mut got = model.forward(&gpu, &ids[..p], Logits::All).unwrap().to_vec();
    got.extend_from_slice(model.forward(&gpu, &ids[p..p + 5], Logits::All).unwrap());
    for &id in &ids[p + 5..] {
        got.extend_from_slice(model.forward(&gpu, &[id], Logits::Last).unwrap());
    }
    assert_eq!(model.len(), MAX_SEQ);
    check(
        "prefill+chunk+decode vs fp16-emulated ref",
        &got,
        &reference(&host, &ids, Ref::Fp16).concat(),
        TOL,
    );
    assert!(model.forward(&gpu, &[0], Logits::Last).is_err());
    assert_eq!(model.len(), MAX_SEQ);
}

#[test]
fn last_row_and_reset_are_consistent() {
    let gpu = Gpu::new(0).unwrap();
    let (mut model, _) = build(&gpu, "last", 104);
    let ids = random_ids(&mut StdRng::seed_from_u64(4), 9);
    let all = model.forward(&gpu, &ids, Logits::All).unwrap().to_vec();
    model.reset();
    let last = model.forward(&gpu, &ids, Logits::Last).unwrap().to_vec();
    let want: Vec<f64> = all[8 * 100..].iter().map(|&x| x as f64).collect();
    check("Last vs last row of All", &last, &want, TOL_HEAD_ONLY);

    model.reset();
    let again = model.forward(&gpu, &ids, Logits::Last).unwrap().to_vec();
    assert_eq!(last, again, "forward is not deterministic after reset");

    model.truncate(4).unwrap();
    let redo = model.forward(&gpu, &ids[4..], Logits::Last).unwrap().to_vec();
    let want: Vec<f64> = last.iter().map(|&x| x as f64).collect();
    check("truncate+recompute vs Last", &redo, &want, TOL);
}

#[test]
fn invalid_inputs_leave_state_unchanged() {
    let gpu = Gpu::new(0).unwrap();
    let (mut model, host) = build(&gpu, "errors", 105);
    let ids = random_ids(&mut StdRng::seed_from_u64(5), 6);
    model.forward(&gpu, &ids[..3], Logits::All).unwrap();

    assert!(model.forward(&gpu, &[], Logits::Last).is_err());
    assert!(model.forward(&gpu, &[0; MAX_TOKENS + 1], Logits::Last).is_err());
    assert!(model.forward(&gpu, &[1, 100], Logits::Last).is_err());
    assert!(model.forward(&gpu, &[u32::MAX], Logits::Last).is_err());
    assert_eq!(model.len(), 3);

    let got = model.forward(&gpu, &ids[3..], Logits::All).unwrap().to_vec();
    let want = reference(&host, &ids, Ref::Fp16)[3..].concat();
    check("after rejected inputs vs fp16-emulated ref", &got, &want, TOL);
}

#[test]
fn rejects_bad_limits() {
    let gpu = Gpu::new(0).unwrap();
    let (dir, _) = write_checkpoint("limits", 106);
    let cfg = LlamaConfig::from_json(CFG).unwrap();
    let load = || Weights::load(&gpu, &dir, &cfg).unwrap();

    assert!(Llama::new(&gpu, cfg.clone(), load(), 20_000, 8, 1).is_err());
    assert!(Llama::new(&gpu, cfg.clone(), load(), 300, 8, 1).is_err());
    assert!(Llama::new(&gpu, cfg.clone(), load(), 16, 32, 1).is_err());
    assert!(Llama::new(&gpu, cfg.clone(), load(), 16, 8, 9).is_err());
    let mut wrong_layers = cfg.clone();
    wrong_layers.num_layers = 3;
    assert!(Llama::new(&gpu, wrong_layers, load(), 16, 8, 1).is_err());

    let mut model = Llama::new(&gpu, cfg.clone(), load(), 16, 8, 2).unwrap();
    assert!(model.forward(&gpu, &[1, 2, 3], Logits::All).is_err());
    assert!(model.is_empty());
    assert_eq!(model.forward(&gpu, &[1, 2, 3], Logits::Last).unwrap().len(), 100);
}
