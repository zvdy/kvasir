use std::cmp::Ordering;

use rand::{rngs::StdRng, RngExt, SeedableRng};

use crate::{ensure_shape, KvasirError, Result};

const GREEDY_EPS: f32 = 1e-5;
const NUCLEUS_START: usize = 64;
const NUCLEUS_GROWTH: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
}

impl SamplingParams {
    pub const GREEDY: Self = Self {
        temperature: 0.0,
        top_p: 1.0,
    };

    pub fn new(temperature: f32, top_p: f32) -> Result<Self> {
        let p = Self { temperature, top_p };
        p.validate()?;
        Ok(p)
    }

    pub fn validate(&self) -> Result<()> {
        if !self.temperature.is_finite() {
            return Err(KvasirError::Config(format!(
                "temperature must be finite, got {}",
                self.temperature
            )));
        }
        if !(self.top_p > 0.0 && self.top_p <= 1.0) {
            return Err(KvasirError::Config(format!(
                "top_p must be in (0, 1], got {}",
                self.top_p
            )));
        }
        Ok(())
    }

    pub fn is_greedy(&self) -> bool {
        self.temperature <= GREEDY_EPS
    }
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self::GREEDY
    }
}

/// Lowest index wins ties; NaN entries are ignored.
pub fn argmax(logits: &[f32]) -> Result<u32> {
    let mut best: Option<(usize, f32)> = None;
    for (i, &l) in logits.iter().enumerate() {
        if l.is_nan() {
            continue;
        }
        match best {
            Some((_, b)) if l <= b => {}
            _ => best = Some((i, l)),
        }
    }
    match best {
        Some((i, _)) => Ok(i as u32),
        None => Err(KvasirError::Shape(format!(
            "argmax over {} logits with no non-NaN value",
            logits.len()
        ))),
    }
}

fn by_weight_desc(a: &(f32, u32), b: &(f32, u32)) -> Ordering {
    b.0.total_cmp(&a.0).then(a.1.cmp(&b.1))
}

pub struct Sampler {
    params: SamplingParams,
    vocab: usize,
    rng: StdRng,
    cand: Vec<(f32, u32)>,
}

impl Sampler {
    pub fn new(params: SamplingParams, vocab: usize, seed: u64) -> Result<Self> {
        params.validate()?;
        ensure_shape!(
            vocab > 0 && vocab <= u32::MAX as usize,
            "vocab size {vocab} out of range"
        );
        Ok(Self {
            params,
            vocab,
            rng: StdRng::seed_from_u64(seed),
            cand: Vec::with_capacity(vocab),
        })
    }

    pub fn params(&self) -> SamplingParams {
        self.params
    }

    pub fn set_params(&mut self, params: SamplingParams) -> Result<()> {
        params.validate()?;
        self.params = params;
        Ok(())
    }

    pub fn sample(&mut self, logits: &[f32]) -> Result<u32> {
        ensure_shape!(
            logits.len() == self.vocab,
            "logits length {} != vocab {}",
            logits.len(),
            self.vocab
        );
        let best = argmax(logits)?;
        if self.params.is_greedy() {
            return Ok(best);
        }
        let m = logits[best as usize];
        if m == f32::INFINITY {
            return Ok(best);
        }
        ensure_shape!(m.is_finite(), "all logits are -inf");

        let inv_t = 1.0 / self.params.temperature;
        self.cand.clear();
        let mut total = 0f64;
        for (i, &l) in logits.iter().enumerate() {
            let w = ((l - m) * inv_t).exp();
            if w > 0.0 {
                total += w as f64;
                self.cand.push((w, i as u32));
            }
        }

        let kept = if self.params.top_p < 1.0 {
            self.nucleus(self.params.top_p as f64 * total)
        } else {
            self.cand.len()
        };
        let kept_mass: f64 = self.cand[..kept].iter().map(|c| c.0 as f64).sum();
        let r = self.rng.random::<f64>() * kept_mass;
        let mut acc = 0f64;
        for &(w, i) in &self.cand[..kept] {
            acc += w as f64;
            if r < acc {
                return Ok(i);
            }
        }
        Ok(self.cand[kept - 1].1)
    }

    /// Moves the smallest top-weight prefix with mass >= target to the front of `cand`, sorted; returns its length.
    fn nucleus(&mut self, target: f64) -> usize {
        let n = self.cand.len();
        let mut done = 0;
        let mut k = NUCLEUS_START.min(n);
        let mut acc = 0f64;
        loop {
            let tail = &mut self.cand[done..];
            if k < n {
                tail.select_nth_unstable_by(k - done - 1, by_weight_desc);
            }
            tail[..k - done].sort_unstable_by(by_weight_desc);
            for (j, c) in self.cand[done..k].iter().enumerate() {
                acc += c.0 as f64;
                if acc >= target {
                    return done + j + 1;
                }
            }
            if k == n {
                return n;
            }
            done = k;
            k = k.saturating_mul(NUCLEUS_GROWTH).min(n);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LLAMA_VOCAB: usize = 128_256;

    fn random_logits(rng: &mut StdRng, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| rng.random_range(-1.0f32..1.0) * scale).collect()
    }

    fn counts(s: &mut Sampler, logits: &[f32], draws: usize) -> Vec<usize> {
        let mut c = vec![0usize; logits.len()];
        for _ in 0..draws {
            c[s.sample(logits).unwrap() as usize] += 1;
        }
        c
    }

    fn assert_freqs(c: &[usize], expected: &[f64], draws: usize) {
        for (i, (&ci, &p)) in c.iter().zip(expected).enumerate() {
            let f = ci as f64 / draws as f64;
            // 6 sigma of a binomial proportion, floored for p near 0.
            let bound = 6.0 * (p * (1.0 - p) / draws as f64).sqrt() + 1e-9;
            assert!((f - p).abs() <= bound, "token {i}: freq {f} expected {p} bound {bound}");
        }
    }

    #[test]
    fn greedy_picks_argmax_lowest_index_on_tie() {
        let mut s = Sampler::new(SamplingParams::GREEDY, 6, 0).unwrap();
        assert_eq!(s.sample(&[0.1, 2.0, -1.0, 5.0, 5.0, 3.0]).unwrap(), 3);
        assert_eq!(s.sample(&[f32::NAN, 1.0, 1.0, 0.0, 0.0, 0.0]).unwrap(), 1);
        assert_eq!(s.sample(&[f32::NEG_INFINITY, -3.0, -2.0, -9.0, -2.0, -5.0]).unwrap(), 2);
    }

    #[test]
    fn greedy_on_llama_sized_logits() {
        let mut rng = StdRng::seed_from_u64(7);
        let mut logits = random_logits(&mut rng, LLAMA_VOCAB, 10.0);
        logits[123_456] = 50.0;
        let mut s = Sampler::new(SamplingParams::GREEDY, LLAMA_VOCAB, 0).unwrap();
        assert_eq!(s.sample(&logits).unwrap(), 123_456);
    }

    #[test]
    fn tiny_temperature_is_greedy() {
        let logits = [0.0, 1.0, 0.999, -1.0];
        for t in [0.0, -1.0, 1e-7] {
            let mut s = Sampler::new(SamplingParams::new(t, 1.0).unwrap(), 4, 3).unwrap();
            for _ in 0..100 {
                assert_eq!(s.sample(&logits).unwrap(), 1);
            }
        }
    }

    #[test]
    fn same_seed_same_tokens() {
        let mut rng = StdRng::seed_from_u64(42);
        let steps: Vec<Vec<f32>> = (0..50).map(|_| random_logits(&mut rng, LLAMA_VOCAB, 8.0)).collect();
        let params = SamplingParams::new(0.8, 0.9).unwrap();
        let mut a = Sampler::new(params, LLAMA_VOCAB, 1234).unwrap();
        let mut b = Sampler::new(params, LLAMA_VOCAB, 1234).unwrap();
        let ta: Vec<u32> = steps.iter().map(|l| a.sample(l).unwrap()).collect();
        let tb: Vec<u32> = steps.iter().map(|l| b.sample(l).unwrap()).collect();
        assert_eq!(ta, tb);
        assert!(ta.iter().any(|&t| t != ta[0]), "sampling looks degenerate");
    }

    #[test]
    fn tiny_top_p_is_argmax() {
        let mut rng = StdRng::seed_from_u64(5);
        let mut logits = random_logits(&mut rng, LLAMA_VOCAB, 3.0);
        logits[77] = 3.0;
        logits[78] = 3.0;
        let mut s = Sampler::new(SamplingParams::new(1.5, 1e-6).unwrap(), LLAMA_VOCAB, 9).unwrap();
        for _ in 0..200 {
            assert_eq!(s.sample(&logits).unwrap(), 77);
        }
    }

    #[test]
    fn empirical_distribution_full() {
        let p = [0.5, 0.3, 0.15, 0.05];
        let logits: Vec<f32> = p.iter().map(|&x: &f64| x.ln() as f32 + 4.0).collect();
        let draws = 200_000;
        let mut s = Sampler::new(SamplingParams::new(1.0, 1.0).unwrap(), 4, 11).unwrap();
        assert_freqs(&counts(&mut s, &logits, draws), &p, draws);
    }

    #[test]
    fn empirical_distribution_temperature() {
        let p: [f64; 4] = [0.5, 0.3, 0.15, 0.05];
        let logits: Vec<f32> = p.iter().map(|x| x.ln() as f32).collect();
        let z: f64 = p.iter().map(|x| x.sqrt()).sum();
        let expected: Vec<f64> = p.iter().map(|x| x.sqrt() / z).collect();
        let draws = 200_000;
        let mut s = Sampler::new(SamplingParams::new(2.0, 1.0).unwrap(), 4, 12).unwrap();
        assert_freqs(&counts(&mut s, &logits, draws), &expected, draws);
    }

    #[test]
    fn empirical_distribution_top_p() {
        let p: [f64; 4] = [0.5, 0.3, 0.15, 0.05];
        let logits: Vec<f32> = p.iter().map(|x| x.ln() as f32).collect();
        let draws = 200_000;
        let mut s = Sampler::new(SamplingParams::new(1.0, 0.7).unwrap(), 4, 13).unwrap();
        assert_freqs(&counts(&mut s, &logits, draws), &[0.625, 0.375, 0.0, 0.0], draws);
    }

    #[test]
    fn large_nucleus_matches_full_sort_reference() {
        let mut rng = StdRng::seed_from_u64(21);
        let logits = random_logits(&mut rng, LLAMA_VOCAB, 2.0);
        let (t, top_p) = (1.3f32, 0.5f32);
        let m = logits.iter().copied().fold(f32::MIN, f32::max);
        let mut w: Vec<(f32, u32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &l)| (((l - m) * (1.0 / t)).exp(), i as u32))
            .collect();
        let total: f64 = w.iter().map(|x| x.0 as f64).sum();
        w.sort_by(by_weight_desc);
        let mut acc = 0f64;
        let mut keep = 0;
        while acc < top_p as f64 * total {
            acc += w[keep].0 as f64;
            keep += 1;
        }
        assert!(keep > NUCLEUS_START * NUCLEUS_GROWTH, "test must exercise growth");
        let mut in_nucleus = vec![false; LLAMA_VOCAB];
        for &(_, i) in &w[..keep] {
            in_nucleus[i as usize] = true;
        }

        let mut s = Sampler::new(SamplingParams::new(t, top_p).unwrap(), LLAMA_VOCAB, 4).unwrap();
        for _ in 0..100 {
            let tok = s.sample(&logits).unwrap() as usize;
            assert!(in_nucleus[tok], "token {tok} outside the nucleus");
        }
        s.sample(&logits).unwrap();
        let kept = s.nucleus(top_p as f64 * total);
        let mut got: Vec<u32> = s.cand[..kept].iter().map(|c| c.1).collect();
        let mut want: Vec<u32> = w[..keep].iter().map(|c| c.1).collect();
        got.sort_unstable();
        want.sort_unstable();
        assert_eq!(got, want);
    }

    #[test]
    fn invalid_params_error() {
        for (t, p) in [
            (f32::NAN, 0.9),
            (f32::INFINITY, 0.9),
            (1.0, 0.0),
            (1.0, -0.1),
            (1.0, 1.01),
            (1.0, f32::NAN),
        ] {
            assert!(SamplingParams::new(t, p).is_err(), "t={t} p={p}");
            let raw = SamplingParams {
                temperature: t,
                top_p: p,
            };
            assert!(Sampler::new(raw, 4, 0).is_err(), "t={t} p={p}");
        }
        let mut s = Sampler::new(SamplingParams::GREEDY, 4, 0).unwrap();
        assert!(s
            .set_params(SamplingParams {
                temperature: 1.0,
                top_p: 2.0
            })
            .is_err());
        assert_eq!(s.params(), SamplingParams::GREEDY);
    }

    #[test]
    fn invalid_logits_error() {
        let mut s = Sampler::new(SamplingParams::new(1.0, 0.9).unwrap(), 4, 0).unwrap();
        assert!(s.sample(&[0.0; 3]).is_err());
        assert!(s.sample(&[f32::NAN; 4]).is_err());
        assert!(s.sample(&[f32::NEG_INFINITY; 4]).is_err());
        assert_eq!(s.sample(&[0.0, f32::INFINITY, 1.0, 2.0]).unwrap(), 1);
        assert!(Sampler::new(SamplingParams::GREEDY, 0, 0).is_err());
    }

    #[test]
    fn no_reallocation_across_steps() {
        let mut rng = StdRng::seed_from_u64(8);
        let mut s = Sampler::new(SamplingParams::new(1.0, 0.95).unwrap(), LLAMA_VOCAB, 2).unwrap();
        let cap = s.cand.capacity();
        let ptr = s.cand.as_ptr();
        for _ in 0..5 {
            let logits = random_logits(&mut rng, LLAMA_VOCAB, 5.0);
            s.sample(&logits).unwrap();
        }
        assert_eq!(s.cand.capacity(), cap);
        assert_eq!(s.cand.as_ptr(), ptr);
    }
}
