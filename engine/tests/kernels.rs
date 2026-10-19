use half::{bf16, f16};
use kvasir::gpu::{gemm, Gpu};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn gpu() -> Gpu {
    Gpu::new(0).expect("CUDA init")
}

fn rand_f16(rng: &mut StdRng, n: usize, amp: f32) -> Vec<f16> {
    (0..n).map(|_| f16::from_f32(rng.random_range(-amp..amp))).collect()
}

fn f32s(v: &[f16]) -> Vec<f32> {
    v.iter().map(|x| x.to_f32()).collect()
}

fn assert_close(name: &str, got: &[f32], want: &[f32], rtol: f32, atol: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length mismatch");
    let mut worst = 0.0f32;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let err = (g - w).abs();
        worst = worst.max(err);
        assert!(
            err <= atol + rtol * w.abs(),
            "{name}[{i}]: got {g}, want {w} (err {err})"
        );
    }
    eprintln!("{name}: max abs err {worst:.3e} over {} elems", got.len());
}

#[test]
fn embedding_gather_is_exact() {
    let g = gpu();
    let mut rng = StdRng::seed_from_u64(1);
    let (vocab, hidden) = (50, 300);
    let table = rand_f16(&mut rng, vocab * hidden, 2.0);
    let ids: Vec<i32> = vec![7, 0, 49, 7, 23];
    let d_table = g.stream.clone_htod(&table).unwrap();
    let d_ids = g.stream.clone_htod(&ids).unwrap();
    let mut d_out = g.stream.alloc_zeros::<f16>(ids.len() * hidden).unwrap();
    g.embedding_gather(&d_ids.as_view(), &d_table.as_view(), &mut d_out.as_view_mut(), hidden)
        .unwrap();
    let out = g.stream.clone_dtoh(&d_out).unwrap();
    for (t, &id) in ids.iter().enumerate() {
        let want = &table[id as usize * hidden..(id as usize + 1) * hidden];
        assert_eq!(&out[t * hidden..(t + 1) * hidden], want, "row {t}");
    }
}

fn rmsnorm_case(rows: usize, hidden: usize) {
    let g = gpu();
    let mut rng = StdRng::seed_from_u64(2);
    let x = rand_f16(&mut rng, rows * hidden, 3.0);
    let w = rand_f16(&mut rng, hidden, 1.5);
    let eps = 1e-5f32;
    let (dx, dw) = (g.stream.clone_htod(&x).unwrap(), g.stream.clone_htod(&w).unwrap());
    let mut dout = g.stream.alloc_zeros::<f16>(rows * hidden).unwrap();
    g.rmsnorm(&dx.as_view(), &dw.as_view(), &mut dout.as_view_mut(), rows, hidden, eps)
        .unwrap();
    let got = f32s(&g.stream.clone_dtoh(&dout).unwrap());
    let (xf, wf) = (f32s(&x), f32s(&w));
    let mut want = vec![0f32; rows * hidden];
    for r in 0..rows {
        let row = &xf[r * hidden..(r + 1) * hidden];
        let ms: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / hidden as f64;
        let inv = 1.0 / (ms + eps as f64).sqrt();
        for i in 0..hidden {
            want[r * hidden + i] = (row[i] as f64 * inv * wf[i] as f64) as f32;
        }
    }
    assert_close("rmsnorm", &got, &want, 1.5e-3, 1e-3);
}

#[test]
fn rmsnorm_llama_width() {
    rmsnorm_case(5, 2048);
}

#[test]
fn rmsnorm_odd_width() {
    rmsnorm_case(3, 100);
}

fn rope_table(max_pos: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>) {
    let half_d = head_dim / 2;
    let (mut c, mut s) = (vec![0f32; max_pos * half_d], vec![0f32; max_pos * half_d]);
    for p in 0..max_pos {
        for j in 0..half_d {
            let freq = 1.0f64 / 10000f64.powf(2.0 * j as f64 / head_dim as f64);
            let a = p as f64 * freq;
            c[p * half_d + j] = a.cos() as f32;
            s[p * half_d + j] = a.sin() as f32;
        }
    }
    (c, s)
}

#[test]
fn rope_matches_rotate_half() {
    let g = gpu();
    let mut rng = StdRng::seed_from_u64(3);
    let (n_q, n_kv, hd, t, pos0) = (8usize, 2usize, 64usize, 5usize, 3usize);
    let stride = (n_q + 2 * n_kv) * hd;
    let qkv = rand_f16(&mut rng, t * stride, 2.0);
    let (cos, sin) = rope_table(32, hd);
    let (dcos, dsin) = (g.stream.clone_htod(&cos).unwrap(), g.stream.clone_htod(&sin).unwrap());
    let mut dqkv = g.stream.clone_htod(&qkv).unwrap();
    g.rope_inplace(
        &mut dqkv.as_view_mut(),
        &dcos.as_view(),
        &dsin.as_view(),
        t,
        pos0,
        n_q,
        n_kv,
        hd,
    )
    .unwrap();
    let got = f32s(&g.stream.clone_dtoh(&dqkv).unwrap());

    let mut want = f32s(&qkv);
    for tok in 0..t {
        for h in 0..(n_q + n_kv) {
            let base = tok * stride + h * hd;
            for j in 0..hd / 2 {
                let (c, s) = (cos[(pos0 + tok) * hd / 2 + j], sin[(pos0 + tok) * hd / 2 + j]);
                let (x0, x1) = (want[base + j], want[base + j + hd / 2]);
                want[base + j] = x0 * c - x1 * s;
                want[base + j + hd / 2] = x1 * c + x0 * s;
            }
        }
    }
    assert_close("rope", &got, &want, 1e-3, 1e-3);
    for tok in 0..t {
        let v0 = tok * stride + (n_q + n_kv) * hd;
        assert_eq!(&got[v0..tok * stride + stride], &f32s(&qkv)[v0..tok * stride + stride]);
    }
}

#[test]
fn silu_mul_matches_cpu() {
    let g = gpu();
    let mut rng = StdRng::seed_from_u64(4);
    let (t, inter) = (3usize, 1000usize);
    let gu = rand_f16(&mut rng, t * 2 * inter, 6.0);
    let dgu = g.stream.clone_htod(&gu).unwrap();
    let mut dout = g.stream.alloc_zeros::<f16>(t * inter).unwrap();
    g.silu_mul(&dgu.as_view(), &mut dout.as_view_mut(), t, inter).unwrap();
    let got = f32s(&g.stream.clone_dtoh(&dout).unwrap());
    let guf = f32s(&gu);
    let mut want = vec![0f32; t * inter];
    for r in 0..t {
        for i in 0..inter {
            let gate = guf[r * 2 * inter + i] as f64;
            let up = guf[r * 2 * inter + inter + i] as f64;
            want[r * inter + i] = (gate / (1.0 + (-gate).exp()) * up) as f32;
        }
    }
    assert_close("silu_mul", &got, &want, 2e-3, 1e-3);
}

#[test]
fn add_inplace_matches_cpu() {
    let g = gpu();
    let mut rng = StdRng::seed_from_u64(5);
    let n = 10_007;
    let (x, y) = (rand_f16(&mut rng, n, 10.0), rand_f16(&mut rng, n, 10.0));
    let mut dx = g.stream.clone_htod(&x).unwrap();
    let dy = g.stream.clone_htod(&y).unwrap();
    g.add_inplace(&mut dx.as_view_mut(), &dy.as_view()).unwrap();
    let got = g.stream.clone_dtoh(&dx).unwrap();
    for i in 0..n {
        assert_eq!(got[i], f16::from_f32(x[i].to_f32() + y[i].to_f32()), "elem {i}");
    }
}

#[test]
fn bf16_to_f16_converts_and_counts_overflow() {
    let g = gpu();
    let mut rng = StdRng::seed_from_u64(6);
    let mut src: Vec<bf16> = (0..5000).map(|_| bf16::from_f32(rng.random_range(-4.0..4.0))).collect();
    src[10] = bf16::from_f32(1.0e6); // > 65504
    src[20] = bf16::from_f32(-7.0e4);
    let dsrc = g.stream.clone_htod(&src).unwrap();
    let mut ddst = g.stream.alloc_zeros::<f16>(src.len()).unwrap();
    let mut dbad = g.stream.alloc_zeros::<u32>(1).unwrap();
    g.bf16_to_f16(&dsrc.as_view(), &mut ddst.as_view_mut(), &mut dbad.as_view_mut())
        .unwrap();
    let got = g.stream.clone_dtoh(&ddst).unwrap();
    assert_eq!(g.stream.clone_dtoh(&dbad).unwrap(), vec![2]);
    for (i, (s, d)) in src.iter().zip(&got).enumerate() {
        assert_eq!(*d, f16::from_f32(s.to_f32()), "elem {i}");
    }
}

#[test]
fn kv_append_places_k_and_v() {
    let g = gpu();
    let mut rng = StdRng::seed_from_u64(7);
    let (n_q, n_kv, hd, t, pos0, max_seq) = (4usize, 2usize, 64usize, 3usize, 5usize, 16usize);
    let stride = (n_q + 2 * n_kv) * hd;
    let qkv = rand_f16(&mut rng, t * stride, 2.0);
    let dqkv = g.stream.clone_htod(&qkv).unwrap();
    let mut kc = g.stream.alloc_zeros::<f16>(n_kv * max_seq * hd).unwrap();
    let mut vc = g.stream.alloc_zeros::<f16>(n_kv * max_seq * hd).unwrap();
    g.kv_append(
        &dqkv.as_view(),
        &mut kc.as_view_mut(),
        &mut vc.as_view_mut(),
        t,
        pos0,
        n_q,
        n_kv,
        hd,
        max_seq,
    )
    .unwrap();
    let (kh, vh) = (g.stream.clone_dtoh(&kc).unwrap(), g.stream.clone_dtoh(&vc).unwrap());
    let zero = f16::from_f32(0.0);
    for h in 0..n_kv {
        for p in 0..max_seq {
            for d in 0..hd {
                let idx = (h * max_seq + p) * hd + d;
                if p >= pos0 && p < pos0 + t {
                    let row = (p - pos0) * stride;
                    assert_eq!(kh[idx], qkv[row + n_q * hd + h * hd + d]);
                    assert_eq!(vh[idx], qkv[row + (n_q + n_kv) * hd + h * hd + d]);
                } else {
                    assert_eq!((kh[idx], vh[idx]), (zero, zero), "untouched slot written: h{h} p{p}");
                }
            }
        }
    }
}

fn attention_case(n_q: usize, n_kv: usize, hd: usize, t: usize, pos0: usize, max_seq: usize) {
    let g = gpu();
    let mut rng = StdRng::seed_from_u64(100 + (t * 31 + pos0) as u64);
    let stride = (n_q + 2 * n_kv) * hd;
    let qkv = rand_f16(&mut rng, t * stride, 2.0);
    let kc = rand_f16(&mut rng, n_kv * max_seq * hd, 2.0);
    let vc = rand_f16(&mut rng, n_kv * max_seq * hd, 2.0);
    let (dqkv, dk, dv) = (
        g.stream.clone_htod(&qkv).unwrap(),
        g.stream.clone_htod(&kc).unwrap(),
        g.stream.clone_htod(&vc).unwrap(),
    );
    let mut dout = g.stream.alloc_zeros::<f16>(t * n_q * hd).unwrap();
    g.attention(
        &dqkv.as_view(),
        &dk.as_view(),
        &dv.as_view(),
        &mut dout.as_view_mut(),
        t,
        pos0,
        n_q,
        n_kv,
        hd,
        max_seq,
    )
    .unwrap();
    let got = f32s(&g.stream.clone_dtoh(&dout).unwrap());

    let (qf, kf, vf) = (f32s(&qkv), f32s(&kc), f32s(&vc));
    let group = n_q / n_kv;
    let scale = 1.0 / (hd as f64).sqrt();
    let mut want = vec![0f32; t * n_q * hd];
    for tok in 0..t {
        let kv_len = pos0 + tok + 1; // causal: keys 0..=pos0+tok
        for h in 0..n_q {
            let kvh = h / group;
            let q = &qf[tok * stride + h * hd..tok * stride + (h + 1) * hd];
            let scores: Vec<f64> = (0..kv_len)
                .map(|j| {
                    let k = &kf[(kvh * max_seq + j) * hd..(kvh * max_seq + j + 1) * hd];
                    q.iter().zip(k).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() * scale
                })
                .collect();
            let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let ps: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
            let denom: f64 = ps.iter().sum();
            for d in 0..hd {
                let acc: f64 = (0..kv_len)
                    .map(|j| ps[j] * vf[(kvh * max_seq + j) * hd + d] as f64)
                    .sum();
                want[(tok * n_q + h) * hd + d] = (acc / denom) as f32;
            }
        }
    }
    assert_close(&format!("attention(t={t},pos0={pos0})"), &got, &want, 2e-3, 1e-3);
}

#[test]
fn attention_prefill() {
    attention_case(8, 2, 64, 7, 0, 32);
}

#[test]
fn attention_decode_single_token() {
    attention_case(8, 2, 64, 1, 12, 32);
}

#[test]
fn attention_prefill_with_prefix() {
    attention_case(8, 2, 64, 4, 5, 32);
}

#[test]
fn attention_llama_dims_long_context() {
    attention_case(32, 8, 64, 3, 397, 512);
}

#[test]
fn attention_head_dim_128() {
    attention_case(4, 2, 128, 2, 20, 64);
}

fn linear_reference(x: &[f32], w: &[f32], t: usize, i: usize, o: usize) -> Vec<f32> {
    let mut y = vec![0f32; t * o];
    for r in 0..t {
        for c in 0..o {
            y[r * o + c] = (0..i).map(|k| x[r * i + k] as f64 * w[c * i + k] as f64).sum::<f64>() as f32;
        }
    }
    y
}

#[test]
fn linear_f16_matches_cpu() {
    let g = gpu();
    let mut rng = StdRng::seed_from_u64(8);
    let (t, i, o) = (3usize, 96usize, 50usize);
    let (x, w) = (rand_f16(&mut rng, t * i, 1.0), rand_f16(&mut rng, o * i, 1.0));
    let (dx, dw) = (g.stream.clone_htod(&x).unwrap(), g.stream.clone_htod(&w).unwrap());
    let mut dy = g.stream.alloc_zeros::<f16>(t * o).unwrap();
    gemm::linear_f16(&g, &dx.as_view(), &dw.as_view(), &mut dy.as_view_mut(), t, i, o).unwrap();
    let got = f32s(&g.stream.clone_dtoh(&dy).unwrap());
    assert_close(
        "linear_f16",
        &got,
        &linear_reference(&f32s(&x), &f32s(&w), t, i, o),
        2e-3,
        1e-3,
    );
}

#[test]
fn linear_f32_matches_cpu() {
    let g = gpu();
    let mut rng = StdRng::seed_from_u64(9);
    let (t, i, o) = (1usize, 2048usize, 777usize);
    let (x, w) = (rand_f16(&mut rng, t * i, 1.0), rand_f16(&mut rng, o * i, 0.1));
    let (dx, dw) = (g.stream.clone_htod(&x).unwrap(), g.stream.clone_htod(&w).unwrap());
    let mut dy = g.stream.alloc_zeros::<f32>(t * o).unwrap();
    gemm::linear_f32(&g, &dx.as_view(), &dw.as_view(), &mut dy.as_view_mut(), t, i, o).unwrap();
    let got = g.stream.clone_dtoh(&dy).unwrap();
    assert_close(
        "linear_f32",
        &got,
        &linear_reference(&f32s(&x), &f32s(&w), t, i, o),
        1e-4,
        1e-4,
    );
}

#[test]
fn shape_errors_are_reported_not_panics() {
    let g = gpu();
    let x = g.stream.alloc_zeros::<f16>(10).unwrap();
    let w = g.stream.alloc_zeros::<f16>(10).unwrap();
    let mut y = g.stream.alloc_zeros::<f16>(10).unwrap();
    assert!(gemm::linear_f16(&g, &x.as_view(), &w.as_view(), &mut y.as_view_mut(), 2, 4, 3).is_err());
    assert!(g.add_inplace(&mut y.as_view_mut(), &x.slice(0..5)).is_err());
}
