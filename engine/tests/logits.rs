use std::path::{Path, PathBuf};

use kvasir::gpu::Gpu;
use kvasir::model::{Llama, LlamaConfig, Logits, Weights};
use kvasir::sampling::argmax;
use serde_json::Value;

// Thresholds: DESIGN.md §6, "Real-model logit test".
const MAX_ABS_FLOOR_FACTOR: f64 = 2.0;
const MAX_ABS_CAP: f64 = 1.0;
const MEAN_ABS_MAX: f64 = 0.05;
const PREFILL_TOP1_MIN: f64 = 0.99;
const MARGIN_MIN: f32 = 0.05;
const EXCLUDED_MAX_FRAC: f64 = 0.01;
const GREEDY_PREFIX_MIN: usize = 32;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn read(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| {
        panic!(
            "{}: {e}. Run `make ref` (needs models/Llama-3.2-1B, see backlog ENV-weights)",
            path.display()
        )
    })
}

fn read_u32(path: &Path) -> Vec<u32> {
    let b = read(path);
    assert_eq!(b.len() % 4, 0, "{}: size not a multiple of 4", path.display());
    b.as_chunks::<4>().0.iter().map(|&c| u32::from_le_bytes(c)).collect()
}

fn read_f32(path: &Path, rows: usize, cols: usize) -> Vec<f32> {
    let b = read(path);
    assert_eq!(
        b.len(),
        rows * cols * 4,
        "{}: expected [{rows}, {cols}] f32",
        path.display()
    );
    b.as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c)).collect()
}

fn floor(meta: &Value, part: &str, key: &str) -> f64 {
    meta["hf16_floor"][part][key]
        .as_f64()
        .unwrap_or_else(|| panic!("meta.json: hf16_floor.{part}.{key} missing"))
}

struct Stats {
    max_abs: f64,
    mean_abs: f64,
    top1_agree: usize,
    top1_total: usize,
    excluded: usize,
    rows: usize,
}

fn top2_margin(row: &[f32]) -> f32 {
    let (mut a, mut b) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &x in row {
        if x > a {
            b = a;
            a = x;
        } else if x > b {
            b = x;
        }
    }
    a - b
}

// Rows whose reference top-2 margin is below `margin_min` are left out of the top-1 count.
fn compare(got: &[f32], want: &[f32], vocab: usize, margin_min: f32) -> Stats {
    assert_eq!(got.len(), want.len());
    let (mut max_abs, mut sum_abs) = (0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        let d = (g as f64 - w as f64).abs();
        assert!(d.is_finite(), "non-finite logit: got {g}, want {w}");
        max_abs = max_abs.max(d);
        sum_abs += d;
    }
    let rows = want.len() / vocab;
    let (mut top1_agree, mut top1_total, mut excluded) = (0, 0, 0);
    for (i, (g, w)) in got.chunks_exact(vocab).zip(want.chunks_exact(vocab)).enumerate() {
        let (gi, wi) = (argmax(g).unwrap() as usize, argmax(w).unwrap() as usize);
        let margin = top2_margin(w);
        if gi != wi || margin < margin_min {
            println!(
                "    row {i}: reference top-2 margin {margin:.4}, top-1 HF {wi} kvasir {gi} (kvasir {wi}-{gi} = {:.4}){}",
                g[wi] - g[gi],
                if margin < margin_min { ", excluded" } else { "" }
            );
        }
        if margin < margin_min {
            excluded += 1;
            continue;
        }
        top1_total += 1;
        if gi == wi {
            top1_agree += 1;
        }
    }
    Stats {
        max_abs,
        mean_abs: sum_abs / got.len() as f64,
        top1_agree,
        top1_total,
        excluded,
        rows,
    }
}

struct Checks(Vec<String>);

impl Checks {
    fn check(&mut self, ok: bool, what: String) {
        println!("  [{}] {what}", if ok { "ok" } else { "FAIL" });
        if !ok {
            self.0.push(what);
        }
    }
}

fn check_logits(c: &mut Checks, name: &str, s: &Stats, floor_max: f64, top1_min: f64) {
    let max_tol = (MAX_ABS_FLOOR_FACTOR * floor_max).min(MAX_ABS_CAP);
    let top1 = s.top1_agree as f64 / s.top1_total.max(1) as f64;
    c.check(
        s.max_abs <= max_tol,
        format!(
            "{name} max abs err {:.4} <= {max_tol:.4} (min({MAX_ABS_FLOOR_FACTOR} x HF-FP16 floor {floor_max:.4}, {MAX_ABS_CAP}))",
            s.max_abs
        ),
    );
    c.check(
        s.mean_abs <= MEAN_ABS_MAX,
        format!("{name} mean abs err {:.5} <= {MEAN_ABS_MAX}", s.mean_abs),
    );
    c.check(
        top1 >= top1_min,
        format!(
            "{name} top-1 agreement {}/{} = {:.2}% >= {:.0}%",
            s.top1_agree,
            s.top1_total,
            100.0 * top1,
            100.0 * top1_min
        ),
    );
}

#[test]
fn logits_match_hf_reference() {
    let root = root();
    let out = root.join("ref/out");
    let model_dir = root.join("models/Llama-3.2-1B");
    let meta: Value = serde_json::from_slice(&read(&out.join("meta.json"))).expect("meta.json: invalid JSON");
    assert_eq!(meta["format_version"].as_u64(), Some(1), "meta.json: format_version");
    assert!(
        model_dir.join("model.safetensors").is_file(),
        "{}/model.safetensors not found (backlog ENV-weights)",
        model_dir.display()
    );

    let cfg = LlamaConfig::from_dir(&model_dir).unwrap();
    let vocab = cfg.vocab_size;
    assert_eq!(meta["vocab_size"].as_u64(), Some(vocab as u64), "meta.json: vocab_size");
    let prompt = read_u32(&out.join("prompt_ids.u32"));
    let greedy = read_u32(&out.join("greedy_ids.u32"));
    let t = prompt.len();
    let steps = greedy.len();
    let n_dec = meta["decode_logit_steps"]
        .as_u64()
        .expect("meta.json: decode_logit_steps") as usize;
    assert!(t > 0 && steps >= GREEDY_PREFIX_MIN && n_dec <= steps);
    let ref_prefill = read_f32(&out.join("prefill_logits.f32"), t, vocab);
    let ref_decode = read_f32(&out.join("decode_logits.f32"), n_dec, vocab);

    println!(
        "reference: {} prompt tokens, {steps} greedy, {n_dec} decode logit steps, sha256 {}, torch {}, transformers {}",
        t,
        meta["model_safetensors_sha256"].as_str().unwrap_or("?"),
        meta["versions"]["torch"].as_str().unwrap_or("?"),
        meta["versions"]["transformers"].as_str().unwrap_or("?"),
    );
    for part in ["prefill", "decode"] {
        println!(
            "HF-FP16 floor {part}: max abs {:.4}, mean abs {:.5}, top-1 {:.4}",
            floor(&meta, part, "max_abs_err"),
            floor(&meta, part, "mean_abs_err"),
            floor(&meta, part, "top1_agreement")
        );
    }

    let gpu = Gpu::new(0).unwrap();
    let weights = Weights::load(&gpu, &model_dir, &cfg).unwrap();
    let mut model = Llama::new(&gpu, cfg, weights, t + steps, t, t).unwrap();
    let mut c = Checks(Vec::new());

    let got = model.forward(&gpu, &prompt, Logits::All).unwrap().to_vec();
    let s = compare(&got, &ref_prefill, vocab, MARGIN_MIN);
    println!("prefill ({t} positions):");
    check_logits(
        &mut c,
        "prefill",
        &s,
        floor(&meta, "prefill", "max_abs_err"),
        PREFILL_TOP1_MIN,
    );
    c.check(
        s.excluded as f64 <= EXCLUDED_MAX_FRAC * s.rows as f64,
        format!(
            "prefill positions with reference top-2 margin < {MARGIN_MIN}: {}/{} <= {:.0}%",
            s.excluded,
            s.rows,
            100.0 * EXCLUDED_MAX_FRAC
        ),
    );

    let mut got = Vec::with_capacity(n_dec * vocab);
    for &id in &greedy[..n_dec] {
        got.extend_from_slice(model.forward(&gpu, &[id], Logits::Last).unwrap());
    }
    let s = compare(&got, &ref_decode, vocab, f32::NEG_INFINITY);
    println!("decode, teacher-forced ({n_dec} steps):");
    check_logits(&mut c, "decode", &s, floor(&meta, "decode", "max_abs_err"), 1.0);

    model.reset();
    let mut ids = vec![argmax(model.forward(&gpu, &prompt, Logits::Last).unwrap()).unwrap()];
    while ids.len() < steps {
        let last = ids[ids.len() - 1];
        ids.push(argmax(model.forward(&gpu, &[last], Logits::Last).unwrap()).unwrap());
    }
    let prefix = ids.iter().zip(&greedy).take_while(|(a, b)| a == b).count();
    println!("greedy, free-running ({steps} tokens):");
    c.check(
        prefix >= GREEDY_PREFIX_MIN,
        format!("greedy identical prefix {prefix}/{steps} >= {GREEDY_PREFIX_MIN}"),
    );
    if prefix < steps {
        println!(
            "  first divergence at {prefix}: kvasir {:?} vs HF {:?}",
            &ids[prefix..],
            &greedy[prefix..]
        );
    }

    assert!(c.0.is_empty(), "logit test failed: {:#?}", c.0);
}
