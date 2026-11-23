use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::CudaSlice;
use half::f16;
use kvasir::gpu::{gemm, DeviceTensor, Gpu};
use kvasir::model::weights::LayerWeights;
use kvasir::model::{Llama, LlamaConfig, Logits, Weights};
use rand::{rngs::StdRng, RngExt, SeedableRng};
use serde::{Deserialize, Serialize};

type Res<T> = Result<T, Box<dyn Error>>;

const LLAMA_3_2_1B: &str = r#"{
  "bos_token_id": 128000, "eos_token_id": 128001, "head_dim": 64, "hidden_size": 2048,
  "intermediate_size": 8192, "max_position_embeddings": 131072, "num_attention_heads": 32,
  "num_hidden_layers": 16, "num_key_value_heads": 8, "rms_norm_eps": 1e-05,
  "rope_scaling": {"factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192, "rope_type": "llama3"},
  "rope_theta": 500000.0, "tie_word_embeddings": true, "vocab_size": 128256 }"#;

const MAX_SEQ: usize = 4096;
const PREFILL: usize = 128;
const CTXS: [usize; 3] = [128, 512, 2048];
const WARMUP: usize = 10;
const STEPS: usize = 60;
const PREFILL_RUNS: usize = 21;
const D2H_RUNS: usize = 201;
const ROOFLINE_GBPS: f64 = 504.0;
const REGRESSION: f64 = 0.10;
const POOL: usize = 1 << 22;

#[derive(Serialize, Deserialize, Clone)]
struct Record {
    name: String,
    median_ms: f64,
    p10_ms: f64,
    p90_ms: f64,
    tok_s: f64,
    pct_roofline: f64,
    gated: bool,
}

#[derive(Serialize, Deserialize)]
struct Report {
    gpu: String,
    roofline_gbps: f64,
    decode_bytes: u64,
    roofline_ms: f64,
    results: Vec<Record>,
}

fn upload(gpu: &Gpu, pool: &[f16], rng: &mut StdRng, shape: &[usize]) -> Res<DeviceTensor<f16>> {
    let n: usize = shape.iter().product();
    let mut s = gpu.stream.alloc_zeros::<f16>(n)?;
    let mut off = 0;
    while off < n {
        let start = rng.random_range(0..pool.len() / 2);
        let k = (n - off).min(pool.len() - start);
        gpu.stream
            .memcpy_htod(&pool[start..start + k], &mut s.slice_mut(off..off + k))?;
        off += k;
    }
    Ok(DeviceTensor::from_slice(s, shape)?)
}

fn pool(rng: &mut StdRng, amp: f32) -> Vec<f16> {
    (0..POOL).map(|_| f16::from_f32(rng.random_range(-amp..amp))).collect()
}

// Seeded random weights at the real shapes: linear U(±√(3/in)), norms U(0.5, 1.5), embedding U(±1), tied head.
fn synthetic_weights(gpu: &Gpu, cfg: &LlamaConfig) -> Res<Weights> {
    let mut rng = StdRng::seed_from_u64(0);
    let (h, qkv, q, inter, v) = (
        cfg.hidden_size,
        cfg.qkv_dim(),
        cfg.q_dim(),
        cfg.intermediate_size,
        cfg.vocab_size,
    );
    let embed_pool = pool(&mut rng, 1.0);
    let lin_h = pool(&mut rng, (3.0 / h as f32).sqrt());
    let lin_q = pool(&mut rng, (3.0 / q as f32).sqrt());
    let lin_i = pool(&mut rng, (3.0 / inter as f32).sqrt());
    let norm: Vec<f16> = (0..POOL).map(|_| f16::from_f32(rng.random_range(0.5..1.5))).collect();
    let mut layers = Vec::with_capacity(cfg.num_layers);
    for _ in 0..cfg.num_layers {
        layers.push(LayerWeights {
            attn_norm: upload(gpu, &norm, &mut rng, &[h])?,
            w_qkv: upload(gpu, &lin_h, &mut rng, &[qkv, h])?,
            w_o: upload(gpu, &lin_q, &mut rng, &[h, q])?,
            mlp_norm: upload(gpu, &norm, &mut rng, &[h])?,
            w_gu: upload(gpu, &lin_h, &mut rng, &[2 * inter, h])?,
            w_down: upload(gpu, &lin_i, &mut rng, &[h, inter])?,
        });
    }
    Ok(Weights {
        embed: upload(gpu, &embed_pool, &mut rng, &[v, h])?,
        lm_head: None,
        layers,
        final_norm: upload(gpu, &norm, &mut rng, &[h])?,
    })
}

fn stats(mut v: Vec<f64>) -> (f64, f64, f64) {
    v.sort_by(f64::total_cmp);
    let at = |q: f64| v[((v.len() - 1) as f64 * q).round() as usize];
    (at(0.5), at(0.1), at(0.9))
}

struct Filler {
    x: CudaSlice<f16>,
    w: CudaSlice<f16>,
    y: CudaSlice<f16>,
}

impl Filler {
    const T: usize = 512;
    const IN: usize = 2048;
    const OUT: usize = 8192;

    fn new(gpu: &Gpu) -> Res<Self> {
        Ok(Self {
            x: gpu.stream.alloc_zeros(Self::T * Self::IN)?,
            w: gpu.stream.alloc_zeros(Self::OUT * Self::IN)?,
            y: gpu.stream.alloc_zeros(Self::T * Self::OUT)?,
        })
    }

    fn queue(&mut self, gpu: &Gpu, reps: usize) -> Res<()> {
        for _ in 0..reps {
            gemm::linear_f16(
                gpu,
                &self.x.as_view(),
                &self.w.as_view(),
                &mut self.y.as_view_mut(),
                Self::T,
                Self::IN,
                Self::OUT,
            )?;
        }
        Ok(())
    }
}

struct Ctx<'a> {
    gpu: &'a Gpu,
    model: Llama,
    filler: Filler,
    roofline_ms: f64,
    results: Vec<Record>,
}

impl Ctx<'_> {
    fn push(&mut self, name: String, samples: Vec<f64>, tokens: usize, gated: bool) {
        let (median_ms, p10_ms, p90_ms) = stats(samples);
        let pct_roofline = if tokens == 1 {
            self.roofline_ms / median_ms * 100.0
        } else {
            0.0
        };
        self.results.push(Record {
            name,
            median_ms,
            p10_ms,
            p90_ms,
            tok_s: tokens as f64 / median_ms * 1e3,
            pct_roofline,
            gated,
        });
    }

    fn prefill(&mut self, ids: &[u32]) -> Res<()> {
        let mut wall = Vec::new();
        for r in 0..PREFILL_RUNS + 3 {
            self.model.reset();
            let t0 = Instant::now();
            self.model.forward(self.gpu, ids, Logits::Last)?;
            if r >= 3 {
                wall.push(t0.elapsed().as_secs_f64() * 1e3);
            }
        }
        self.push(format!("prefill/t{}", ids.len()), wall, ids.len(), true);
        Ok(())
    }

    fn decode(&mut self, ctx: usize, ids: &[u32]) -> Res<()> {
        let gpu = self.gpu;
        self.model.reset();
        for chunk in ids[..ctx].chunks(PREFILL) {
            self.model.forward(gpu, chunk, Logits::Last)?;
        }
        let tok = [ids[ctx]];
        let step = |m: &mut Llama| -> Res<()> {
            m.truncate(ctx)?;
            m.forward(gpu, &tok, Logits::Last)?;
            Ok(())
        };
        for _ in 0..WARMUP {
            step(&mut self.model)?;
        }
        let mut wall = Vec::with_capacity(STEPS);
        for _ in 0..STEPS {
            let t0 = Instant::now();
            step(&mut self.model)?;
            wall.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        // Device time: the step is queued behind a filler several times longer than its wall time,
        // so every launch is enqueued before the GPU reaches it and launch latency drops out.
        let wall_med = stats(wall.clone()).0;
        let filler_reps = ((3.0 * wall_med / 0.35).ceil() as usize).clamp(4, 400);
        let start = gpu.ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        let stop = gpu.ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        start.record(&gpu.stream)?;
        self.filler.queue(gpu, filler_reps)?;
        stop.record(&gpu.stream)?;
        gpu.synchronize()?;
        let filler_ms = f64::from(start.elapsed_ms(&stop)?);
        if filler_ms < 2.0 * wall_med {
            eprintln!("warning: ctx {ctx}: filler {filler_ms:.2} ms < 2x step wall {wall_med:.2} ms");
        }
        let mut dev = Vec::with_capacity(STEPS);
        for _ in 0..STEPS {
            self.filler.queue(gpu, filler_reps)?;
            start.record(&gpu.stream)?;
            self.model.truncate(ctx)?;
            self.model.forward(gpu, &tok, Logits::Last)?;
            stop.record(&gpu.stream)?;
            gpu.synchronize()?;
            dev.push(f64::from(start.elapsed_ms(&stop)?));
        }
        self.push(format!("decode/ctx{ctx}/wall"), wall, 1, true);
        self.push(format!("decode/ctx{ctx}/device"), dev, 1, true);
        Ok(())
    }

    fn logits_host(&mut self, vocab: usize) -> Res<()> {
        let gpu = self.gpu;
        let src = gpu.stream.alloc_zeros::<f32>(vocab)?;
        let mut host = vec![0f32; vocab];
        let (mut d2h, mut argmax) = (Vec::new(), Vec::new());
        for _ in 0..D2H_RUNS {
            gpu.synchronize()?;
            let t0 = Instant::now();
            gpu.stream.memcpy_dtoh(&src, &mut host)?;
            gpu.synchronize()?;
            d2h.push(t0.elapsed().as_secs_f64() * 1e3);
            let t0 = Instant::now();
            let best = host
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i);
            std::hint::black_box(best);
            argmax.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        self.push("logits_d2h".into(), d2h, 0, false);
        self.push("host_argmax".into(), argmax, 0, false);
        Ok(())
    }
}

fn print_table(cur: &Report, base: Option<&Report>) -> usize {
    let mut regressions = 0;
    println!(
        "decode roofline: {:.3} GB / {ROOFLINE_GBPS:.0} GB/s = {:.3} ms/token",
        cur.decode_bytes as f64 / 1e9,
        cur.roofline_ms
    );
    println!("| case | base ms | now ms | p10 | p90 | delta | tok/s | % roofline | |");
    println!("|---|---:|---:|---:|---:|---:|---:|---:|---|");
    for r in &cur.results {
        let prev = base.and_then(|b| b.results.iter().find(|p| p.name == r.name));
        let (base_s, delta_s, status) = match prev {
            Some(p) => {
                let d = r.median_ms / p.median_ms - 1.0;
                let st = if !r.gated {
                    "info"
                } else if d > REGRESSION {
                    regressions += 1;
                    "REGRESSION"
                } else if d < -REGRESSION {
                    "faster"
                } else {
                    "ok"
                };
                (format!("{:.3}", p.median_ms), format!("{:+.1}%", d * 100.0), st)
            }
            None if base.is_some() => ("-".into(), "-".into(), "new"),
            None => ("-".into(), "-".into(), ""),
        };
        let tok_s = if r.tok_s > 0.0 {
            format!("{:.1}", r.tok_s)
        } else {
            "-".into()
        };
        let pct = if r.pct_roofline > 0.0 {
            format!("{:.0}%", r.pct_roofline)
        } else {
            "-".into()
        };
        println!(
            "| {} | {base_s} | {:.3} | {:.3} | {:.3} | {delta_s} | {tok_s} | {pct} | {status} |",
            r.name, r.median_ms, r.p10_ms, r.p90_ms
        );
    }
    regressions
}

fn write_json(path: &Path, rep: &Report) -> Res<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    fs::write(path, serde_json::to_string_pretty(rep)? + "\n")?;
    Ok(())
}

fn main() -> Res<()> {
    let (mut out, mut compare) = (None::<PathBuf>, None::<PathBuf>);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--out" => out = args.next().map(PathBuf::from),
            "--compare" => compare = args.next().map(PathBuf::from),
            _ => {}
        }
    }

    let gpu = Gpu::new(0)?;
    let cfg = LlamaConfig::from_json(LLAMA_3_2_1B)?;
    let t0 = Instant::now();
    let weights = synthetic_weights(&gpu, &cfg)?;
    let decode_bytes = weights.decode_bytes();
    let vocab = cfg.vocab_size;
    let model = Llama::new(&gpu, cfg, weights, MAX_SEQ, PREFILL, 1)?;
    gpu.synchronize()?;
    eprintln!(
        "built synthetic Llama 3.2 1B ({:.2} GB) in {:.1} s",
        decode_bytes as f64 / 1e9,
        t0.elapsed().as_secs_f64()
    );
    let roofline_ms = decode_bytes as f64 / (ROOFLINE_GBPS * 1e9) * 1e3;
    let mut c = Ctx {
        gpu: &gpu,
        model,
        filler: Filler::new(&gpu)?,
        roofline_ms,
        results: Vec::new(),
    };
    let mut rng = StdRng::seed_from_u64(1);
    let ids: Vec<u32> = (0..MAX_SEQ).map(|_| rng.random_range(0..vocab as u32)).collect();

    // Clock ramp before anything is timed.
    let ramp = Instant::now();
    while ramp.elapsed().as_secs_f64() < 0.5 {
        c.filler.queue(&gpu, 20)?;
        gpu.synchronize()?;
    }
    c.prefill(&ids[..PREFILL])?;
    for ctx in CTXS {
        c.decode(ctx, &ids)?;
    }
    c.logits_host(vocab)?;

    let rep = Report {
        gpu: gpu.ctx.name()?,
        roofline_gbps: ROOFLINE_GBPS,
        decode_bytes: decode_bytes as u64,
        roofline_ms,
        results: std::mem::take(&mut c.results),
    };
    if let Some(p) = &out {
        write_json(p, &rep)?;
        eprintln!("wrote {}", p.display());
    }
    let base = match &compare {
        Some(p) => Some(serde_json::from_str::<Report>(&fs::read_to_string(p)?)?),
        None => None,
    };
    if let Some(bs) = &base {
        if bs.gpu != rep.gpu {
            eprintln!("warning: baseline is from {}, running on {}", bs.gpu, rep.gpu);
        }
    }
    let n = print_table(&rep, base.as_ref());
    if n > 0 {
        return Err(format!("{n} forward case(s) regressed by more than {:.0}%", REGRESSION * 100.0).into());
    }
    Ok(())
}
