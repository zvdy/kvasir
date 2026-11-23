use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{CudaEvent, CudaSlice};
use half::f16;
use kvasir::gpu::kernels::attention_scratch_len;
use kvasir::gpu::{gemm, Gpu};
use rand::{rngs::StdRng, RngExt, SeedableRng};
use serde::{Deserialize, Serialize};

type Res<T> = Result<T, Box<dyn Error>>;

const HIDDEN: usize = 2048;
const N_Q: usize = 32;
const N_KV: usize = 8;
const HEAD_DIM: usize = 64;
const QKV: usize = (N_Q + 2 * N_KV) * HEAD_DIM;
const INTER: usize = 8192;
const VOCAB: usize = 128256;
const MAX_SEQ: usize = 8192;
const PREFILL: usize = 128;
const EPS: f32 = 1e-5;

const ROOFLINE_GBPS: f64 = 504.0;
// RTX 4070 dense FP16 tensor-core spec peak.
const PEAK_TFLOPS: f64 = 58.3;
const REGRESSION: f64 = 0.10;

const WARMUP: usize = 20;
const SAMPLES: usize = 101;
const TARGET_SAMPLE_US: f64 = 300.0;
const MAX_REPS: usize = 64;
// Weights and KV caches are rotated over this many bytes so repeated launches do not hit the 36 MB L2.
const ROTATE_BYTES: usize = 160 << 20;

#[derive(Serialize, Deserialize, Clone)]
struct Record {
    name: String,
    median_us: f64,
    p10_us: f64,
    p90_us: f64,
    reps: usize,
    bytes: u64,
    flops: u64,
    gbps: f64,
    tflops: f64,
}

#[derive(Serialize, Deserialize)]
struct Report {
    gpu: String,
    roofline_gbps: f64,
    results: Vec<Record>,
}

fn alloc_random(gpu: &Gpu, pool: &[f16], n: usize) -> Res<CudaSlice<f16>> {
    let mut s = gpu.stream.alloc_zeros(n)?;
    let mut off = 0;
    while off < n {
        let k = (n - off).min(pool.len());
        gpu.stream.memcpy_htod(&pool[..k], &mut s.slice_mut(off..off + k))?;
        off += k;
    }
    Ok(s)
}

struct Bench {
    gpu: Gpu,
    pool: Vec<f16>,
    filler: (CudaSlice<f16>, CudaSlice<f16>, CudaSlice<f16>),
    start: CudaEvent,
    stop: CudaEvent,
    starved: usize,
    results: Vec<Record>,
}

impl Bench {
    fn new() -> Res<Self> {
        let gpu = Gpu::new(0)?;
        let mut rng = StdRng::seed_from_u64(0);
        let pool: Vec<f16> = (0..1 << 22)
            .map(|_| f16::from_f32(rng.random_range(-1.0..1.0)))
            .collect();
        Ok(Self {
            start: gpu.ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?,
            stop: gpu.ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?,
            filler: (
                alloc_random(&gpu, &pool, 512 * HIDDEN)?,
                alloc_random(&gpu, &pool, INTER * HIDDEN)?,
                gpu.stream.alloc_zeros(512 * INTER)?,
            ),
            gpu,
            pool,
            starved: 0,
            results: Vec::new(),
        })
    }

    fn alloc(&self, n: usize) -> Res<CudaSlice<f16>> {
        alloc_random(&self.gpu, &self.pool, n)
    }

    fn copies(&self, n: usize, bytes_each: usize) -> Res<Vec<CudaSlice<f16>>> {
        (0..ROTATE_BYTES.div_ceil(bytes_each).max(1))
            .map(|_| self.alloc(n))
            .collect()
    }

    // A ~1 ms GEMM keeps the GPU busy while the timed launches are queued, so launch latency is not measured.
    fn run_filler(&mut self) -> Res<()> {
        let (x, w, y) = &mut self.filler;
        for _ in 0..3 {
            gemm::linear_f16(
                &self.gpu,
                &x.as_view(),
                &w.as_view(),
                &mut y.as_view_mut(),
                512,
                HIDDEN,
                INTER,
            )?;
        }
        Ok(())
    }

    fn sample(&mut self, reps: usize, iter: &mut usize, f: &mut dyn FnMut(&Gpu, usize) -> Res<()>) -> Res<f64> {
        self.run_filler()?;
        self.start.record(&self.gpu.stream)?;
        for _ in 0..reps {
            f(&self.gpu, *iter)?;
            *iter += 1;
        }
        self.stop.record(&self.gpu.stream)?;
        if self.start.is_complete() {
            self.starved += 1;
        }
        Ok(f64::from(self.start.elapsed_ms(&self.stop)?) * 1e3 / reps as f64)
    }

    fn run(&mut self, name: &str, bytes: usize, flops: usize, mut f: impl FnMut(&Gpu, usize) -> Res<()>) -> Res<()> {
        let mut iter = 0;
        for _ in 0..WARMUP {
            f(&self.gpu, iter)?;
            iter += 1;
        }
        self.gpu.synchronize()?;
        let est = self.sample(8, &mut iter, &mut f)?;
        let reps = ((TARGET_SAMPLE_US / est).ceil() as usize).clamp(1, MAX_REPS);
        let starved0 = self.starved;
        let mut v = (0..SAMPLES)
            .map(|_| self.sample(reps, &mut iter, &mut f))
            .collect::<Res<Vec<f64>>>()?;
        v.sort_by(f64::total_cmp);
        let med = v[v.len() / 2];
        if self.starved > starved0 {
            eprintln!("warning: {name}: {} samples drained the queue", self.starved - starved0);
        }
        let r = Record {
            name: name.to_string(),
            median_us: med,
            p10_us: v[v.len() / 10],
            p90_us: v[v.len() * 9 / 10],
            reps,
            bytes: bytes as u64,
            flops: flops as u64,
            gbps: bytes as f64 / med / 1e3,
            tflops: flops as f64 / med / 1e6,
        };
        eprintln!("{name:<24} {med:>10.2} us {:>8.1} GB/s", r.gbps);
        self.results.push(r);
        Ok(())
    }
}

fn gemm_case(b: &mut Bench, name: &str, t: usize, in_dim: usize, out_dim: usize, f32_out: bool) -> Res<()> {
    let wbytes = in_dim * out_dim * 2;
    let ws = b.copies(in_dim * out_dim, wbytes)?;
    let x = b.alloc(t * in_dim)?;
    let osz = if f32_out { 4 } else { 2 };
    let mut y16 = b.gpu.stream.alloc_zeros::<f16>(if f32_out { 0 } else { t * out_dim })?;
    let mut y32 = b.gpu.stream.alloc_zeros::<f32>(if f32_out { t * out_dim } else { 0 })?;
    let bytes = wbytes + t * in_dim * 2 + t * out_dim * osz;
    b.run(name, bytes, 2 * t * in_dim * out_dim, |g, i| {
        let w = ws[i % ws.len()].as_view();
        if f32_out {
            gemm::linear_f32(g, &x.as_view(), &w, &mut y32.as_view_mut(), t, in_dim, out_dim)?;
        } else {
            gemm::linear_f16(g, &x.as_view(), &w, &mut y16.as_view_mut(), t, in_dim, out_dim)?;
        }
        Ok(())
    })
}

fn attention_case(b: &mut Bench, name: &str, t: usize, pos0: usize) -> Res<()> {
    // max_seq = len so the rotated copies hold only what is read and their count really covers ROTATE_BYTES.
    let len = pos0 + t;
    let per_layer = N_KV * len * HEAD_DIM;
    let kcs = b.copies(per_layer, 2 * per_layer * 2)?;
    let vcs = b.copies(per_layer, 2 * per_layer * 2)?;
    let qkv = b.alloc(t * QKV)?;
    let mut out = b.alloc(t * N_Q * HEAD_DIM)?;
    let mut scratch = b.gpu.stream.alloc_zeros::<f32>(attention_scratch_len(N_Q, HEAD_DIM))?;
    let bytes = 2 * N_KV * len * HEAD_DIM * 2 + 2 * t * N_Q * HEAD_DIM * 2;
    let flops = 4 * N_Q * HEAD_DIM * (t * pos0 + t * (t + 1) / 2);
    b.run(name, bytes, flops, |g, i| {
        let l = i % kcs.len();
        Ok(g.attention(
            &qkv.as_view(),
            &kcs[l].as_view(),
            &vcs[l].as_view(),
            &mut out.as_view_mut(),
            &mut scratch.as_view_mut(),
            t,
            pos0,
            N_Q,
            N_KV,
            HEAD_DIM,
            len,
        )?)
    })
}

fn bench_all(b: &mut Bench) -> Res<()> {
    // ~0.5 s of GEMMs so clocks have ramped before the first (smallest) kernels are timed.
    for _ in 0..500 {
        b.run_filler()?;
    }
    b.gpu.synchronize()?;
    let mut rng = StdRng::seed_from_u64(1);
    let (cos, sin) = {
        let n = MAX_SEQ * HEAD_DIM / 2;
        let c: Vec<f32> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
        let s: Vec<f32> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
        (b.gpu.stream.clone_htod(&c)?, b.gpu.stream.clone_htod(&s)?)
    };
    let norm_w = b.alloc(HIDDEN)?;

    for t in [1, PREFILL] {
        {
            let table = b.alloc(VOCAB * HIDDEN)?;
            let ids: Vec<i32> = (0..t * 4096).map(|_| rng.random_range(0..VOCAB as i32)).collect();
            let d_ids = b.gpu.stream.clone_htod(&ids)?;
            let mut out = b.alloc(t * HIDDEN)?;
            b.run(&format!("embedding/t{t}"), 2 * t * HIDDEN * 2 + t * 4, 0, |g, i| {
                let o = (i % 4096) * t;
                Ok(g.embedding_gather(&d_ids.slice(o..o + t), &table.as_view(), &mut out.as_view_mut(), HIDDEN)?)
            })?;
        }
        {
            let x = b.alloc(t * HIDDEN)?;
            let mut out = b.alloc(t * HIDDEN)?;
            b.run(&format!("rmsnorm/t{t}"), 2 * t * HIDDEN * 2 + HIDDEN * 2, 0, |g, _| {
                Ok(g.rmsnorm(&x.as_view(), &norm_w.as_view(), &mut out.as_view_mut(), t, HIDDEN, EPS)?)
            })?;
        }
        {
            let mut qkv = b.alloc(t * QKV)?;
            let pos0 = if t == 1 { 2047 } else { 0 };
            let bytes = t * (N_Q + N_KV) * HEAD_DIM * 2 * 2 + 2 * t * HEAD_DIM / 2 * 4;
            b.run(&format!("rope/t{t}"), bytes, 0, |g, _| {
                Ok(g.rope_inplace(
                    &mut qkv.as_view_mut(),
                    &cos.as_view(),
                    &sin.as_view(),
                    t,
                    pos0,
                    N_Q,
                    N_KV,
                    HEAD_DIM,
                )?)
            })?;
        }
        {
            let qkv = b.alloc(t * QKV)?;
            let mut kc = b.alloc(N_KV * MAX_SEQ * HEAD_DIM)?;
            let mut vc = b.alloc(N_KV * MAX_SEQ * HEAD_DIM)?;
            let pos0 = if t == 1 { 2047 } else { 0 };
            b.run(
                &format!("kv_append/t{t}"),
                2 * t * 2 * N_KV * HEAD_DIM * 2,
                0,
                |g, _| {
                    Ok(g.kv_append(
                        &qkv.as_view(),
                        &mut kc.as_view_mut(),
                        &mut vc.as_view_mut(),
                        t,
                        pos0,
                        N_Q,
                        N_KV,
                        HEAD_DIM,
                        MAX_SEQ,
                    )?)
                },
            )?;
        }
        {
            let gu = b.alloc(t * 2 * INTER)?;
            let mut out = b.alloc(t * INTER)?;
            b.run(&format!("silu_mul/t{t}"), t * 3 * INTER * 2, 0, |g, _| {
                Ok(g.silu_mul(&gu.as_view(), &mut out.as_view_mut(), t, INTER)?)
            })?;
        }
        {
            let mut x = b.alloc(t * HIDDEN)?;
            let y = b.alloc(t * HIDDEN)?;
            b.run(&format!("add/t{t}"), 3 * t * HIDDEN * 2, 0, |g, _| {
                Ok(g.add_inplace(&mut x.as_view_mut(), &y.as_view())?)
            })?;
        }
    }

    for len in [512, 2048, 8192] {
        attention_case(b, &format!("attention/t1/ctx{len}"), 1, len - 1)?;
    }
    attention_case(b, &format!("attention/t{PREFILL}/ctx{PREFILL}"), PREFILL, 0)?;

    for t in [1, PREFILL] {
        gemm_case(b, &format!("gemm_qkv/t{t}"), t, HIDDEN, QKV, false)?;
        gemm_case(b, &format!("gemm_o/t{t}"), t, HIDDEN, HIDDEN, false)?;
        gemm_case(b, &format!("gemm_gate_up/t{t}"), t, HIDDEN, 2 * INTER, false)?;
        gemm_case(b, &format!("gemm_down/t{t}"), t, INTER, HIDDEN, false)?;
        gemm_case(b, &format!("lm_head/t{t}"), t, HIDDEN, VOCAB, true)?;
    }
    Ok(())
}

fn print_table(cur: &Report, base: Option<&Report>) -> usize {
    let mut regressions = 0;
    println!(
        "| kernel | base us | now us | delta | GB/s | % of {ROOFLINE_GBPS:.0} GB/s | TFLOP/s | % of {PEAK_TFLOPS} TFLOP/s | |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|---:|---|");
    for r in &cur.results {
        let prev = base.and_then(|b| b.results.iter().find(|p| p.name == r.name));
        let (base_s, delta_s, status) = match prev {
            Some(p) => {
                let d = r.median_us / p.median_us - 1.0;
                let st = if d > REGRESSION {
                    regressions += 1;
                    "REGRESSION"
                } else if d < -REGRESSION {
                    "faster"
                } else {
                    "ok"
                };
                (format!("{:.2}", p.median_us), format!("{:+.1}%", d * 100.0), st)
            }
            None if base.is_some() => ("-".into(), "-".into(), "new"),
            None => ("-".into(), "-".into(), ""),
        };
        let (tf, tfp) = if r.flops > 0 {
            (
                format!("{:.2}", r.tflops),
                format!("{:.0}%", r.tflops / PEAK_TFLOPS * 100.0),
            )
        } else {
            ("-".into(), "-".into())
        };
        println!(
            "| {} | {base_s} | {:.2} | {delta_s} | {:.1} | {:.0}% | {tf} | {tfp} | {status} |",
            r.name,
            r.median_us,
            r.gbps,
            r.gbps / ROOFLINE_GBPS * 100.0
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

    let mut b = Bench::new()?;
    bench_all(&mut b)?;
    let rep = Report {
        gpu: b.gpu.ctx.name()?,
        roofline_gbps: ROOFLINE_GBPS,
        results: std::mem::take(&mut b.results),
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
        return Err(format!("{n} kernel(s) regressed by more than {:.0}%", REGRESSION * 100.0).into());
    }
    Ok(())
}
