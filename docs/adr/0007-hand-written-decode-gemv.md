# 7. Hand-written FP16 GEMV for decode (T = 1); cuBLAS for T > 1

Status: accepted (amends 0002)

## Context

ADR 0002 put every linear layer on `cublasGemmEx`. At batch-1 decode every projection is a matrix-vector product
and the step is bound by streaming 2.47 GB of weights. Measured on the RTX 4070 (`make bench`, I-01): cuBLAS at
`T = 1` reaches 84% (qkv), 81% (o), 92% (gate_up), 89% (down) and 94% (LM head) of 504 GB/s. More importantly, a
cuBLAS call cannot absorb the neighbouring element-wise work: at `T = 1` each layer also launched two RMSNorms, a
`silu_mul` and two residual adds (≈ 5 launches of 1.3–5.5 µs each, 81 per token), all latency-bound on 4–48 KB.

## Decision

For `T = 1` the forward pass uses hand-written FP16 GEMV kernels (`engine/kernels/gemv.cu`, DESIGN.md §3): FP32
accumulation, 16-byte streaming weight loads, warp-shuffle reductions, with the RMSNorm fused into the prologue
(qkv, gate/up, LM head), SwiGLU fused into the gate/up epilogue and the residual add fused into the o/down
epilogue. Each variant is validated against a CPU reference in `tests/kernels.rs` at the real shapes and odd ones.
`T > 1` (prefill, `Logits::All`) keeps cuBLAS, as in ADR 0002.

## Consequences

- Per layer at decode: 5 launches fewer (rmsnorm ×2, silu_mul, add ×2), the `gate_up`/`proj`/`normed` round trips
  disappear. Decode is 2.8–3.8% faster (ctx 2048: 5.83 → 5.67 ms wall, 5.91 → 5.69 ms device, 171 → 176 tok/s), see DESIGN.md §8.
- Per matrix the GEMVs match or beat cuBLAS (qkv 86 vs 84%, o 83 vs 81%, gate_up 92 vs 92%, down 89–90 vs 89%,
  LM head 94 vs 94%). The rest is a fixed ≈ 2.1 µs per launch (fit over 4–67 MB matrices; steady state 470 GB/s =
  93% of spec), which no single kernel can remove; overlapping kernel boundaries (PDL / CUDA graphs) is the next step.
- Decode and prefill now use different reduction orders. Decode-vs-prefill logit error over 30 random synthetic
  models rose from 2.51e-2 to 2.82e-2 (TOL 5e-2 unchanged).
- Decode requires `hidden`, `q_dim` and `intermediate` to be multiples of 8 (16-byte loads); `Llama::new` checks it.
- 0002's "custom GEMM only with a benchmark showing parity" is met for `T = 1` by the `gemv_*` entries in
  `bench/kernels-baseline.json`. Prefill GEMM stays on cuBLAS until a tensor-core GEMM shows parity.
