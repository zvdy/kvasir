# kvasir — design notes (Stage 1: Llama 3.2 1B, batch 1)

Target: one RTX 4070 (12 GB, sm_89, ~504 GB/s). Rust + hand-written CUDA kernels, cuBLAS only for GEMMs.

## 1. Module boundaries

```
engine/
  build.rs            nvcc -ptx -arch=sm_89 for kernels/*.cu -> $OUT_DIR/*.ptx (embedded with include_str!)
  kernels/            embedding.cu rmsnorm.cu rope.cu attention.cu silu_mul.cu elementwise.cu common.cuh
  src/error.rs        KvasirError (+ ensure_shape! macro). No unwrap/expect outside tests and main.
  src/gpu/
    device.rs         Gpu { ctx, stream, blas, kernels } — one context, ONE stream (stage 1)
    buffer.rs         DeviceTensor<T>: owned CudaSlice + logical shape, deliberately not Clone
    kernels.rs        one shape-checked launch wrapper per kernel, as methods on Gpu
    gemm.rs           linear_f16 / linear_f32: cublasGemmEx, FP16 in, FP32 accumulate
  src/model/
    config.rs         LlamaConfig from config.json (validated; rope_type must be "llama3" or absent)
    rope.rs           host cos/sin tables incl. llama3 long-context frequency scaling
    weights.rs        mmap safetensors -> GPU, BF16->FP16 on device, QKV and gate/up fused
    llama.rs          Llama { cfg, Weights, KvCache, Workspace }: forward(ids, Logits::Last | All)
    workspace.rs      Workspace: preallocated activations; WorkspaceDims = pure size math
  src/kv/             KvCache: contiguous K/V, per-layer views, length + bounds tracking
  src/sampling.rs     greedy / temperature / top-p on host               [not written yet]
  src/bin/kvasir.rs   `kvasir generate --model <dir> --prompt ...`       [not written yet]
engine/tests/         kernels.rs (per-kernel vs CPU), weights.rs (synthetic checkpoint), workspace.rs, kv.rs (cache placement),
                      forward.rs (tiny model vs CPU f64 reference forward)
engine/benches/       kernels.rs: per-kernel microbenchmarks vs bench/kernels-baseline.json (`make bench`)
tests/ , ref/         logit-matching test and HF reference dumper        [not written yet]
```

Ownership: device memory is owned by exactly three long-lived structs — `Weights`, `KvCache`,
`Workspace`. Kernel wrappers take borrowed `CudaView`/`CudaViewMut`, never allocate. The generation
loop therefore performs no device allocation (the logits D2H copy goes into a reused host `Vec`).

## 2. Data layouts

Conventions: everything is row-major, contiguous, FP16 unless stated. `T` = number of tokens in the call.

| tensor | shape | notes |
|---|---|---|
| hidden / activations | `[T, 2048]` | one token per row |
| `W_qkv` (fused) | `[3072, 2048]` | rows = `q (32×64) \| k (8×64) \| v (8×64)` |
| `W_o` | `[2048, 2048]` | HF layout `[out, in]` |
| `W_gu` (fused) | `[16384, 2048]` | rows = `gate (8192) \| up (8192)` |
| `W_down` | `[2048, 8192]` | |
| `embed` (= LM head, tied) | `[128256, 2048]` | read as a gather by the embedding and as a matrix by the head |
| norm weights | `[2048]` | |
| QKV activation | `[T, 3072]` | same `q \| k \| v` order as the fused weight rows |
| attention out | `[T, 2048]` | head `h` at columns `[64h, 64h+64)` |
| gate/up activation | `[T, 16384]` | `silu_mul` reads it, writes `[T, 8192]` |
| logits | `[T_head, 128256]` FP32 | never rounded to FP16 |
| RoPE cos/sin | `[max_seq, 32]` FP32 | built on host, uploaded once by `KvCache::from_config`, which owns them |

**Workspace.** Allocated once for `max_tokens` (T upper bound) and `max_head_rows` (T_head upper bound, 1 for
generation): `ids [T] i32`, `hidden`/`normed`/`proj [T, hidden]` (`proj` = o_proj / down_proj output before the
residual add), `qkv [T, qkv_dim]`, `attn [T, q_dim]`, `gate_up [T, 2·inter]`, `act [T, inter]`, `logits [T_head, vocab]`
FP32, plus host buffers for the ids and the logits copy. A call with `t` tokens uses the leading `t` rows of each
buffer. Llama 3.2 1B: 70 KiB per token of FP16 activations + 501 KiB per head row.

**Linear layers.** Weights stay in HF `[out, in]` layout. `Y[T,out] = X[T,in] · Wᵀ` is issued to
cuBLAS (column-major) as `C[out×T] = op_T(A) · B` with `A = W` (`[in×out]` col-major, lda = in), `B = X`
(`[in×T]`, ldb = in), ldc = out. The result is already row-major `[T, out]`; nothing is transposed or copied.

**Head layout.** Within a token row, head `h` occupies a contiguous 64-element span. GQA: query head `h` uses
KV head `h / 4` (32 query heads, 8 KV heads).

**RoPE.** HF `rotate_half` (non-interleaved) convention: rotation pairs are `(j, j+32)` inside each head. This
matches the HF checkpoint's q/k row ordering (the converter permutes rows to this form), so weights are used
as-is. Position of token `i` in a call = `pos0 + i`. Frequencies follow HF's `llama3` rope scaling
(factor 32, low/high freq factors 1/4, original max 8192, θ = 500000); verified in `model::rope` unit tests for the
three wavelength regimes. Applied in place to the q and k slices of the QKV buffer, before the KV append.

**KV cache (stage 1, contiguous).** Two tensors (K, V), each `[layers=16, n_kv=8, max_seq, head_dim=64]` FP16.
Per layer/head the keys are contiguous over positions, so the attention kernel streams a head's keys linearly.
`max_seq` is chosen at startup (default 4096 → 2 × 16·8·4096·64·2 B = 128 MiB total). `kv_append` writes the
(already RoPE'd) K and raw V of the new tokens at `pos0..pos0+T`. This maps directly onto stage 3's paged blocks
(`[block, kv_head, block_size, head_dim]`): only the address computation in `attention`/`kv_append` changes.
`KvCache` (`src/kv/`) hands the kernels layer `l` as the contiguous `[n_kv, max_seq, head_dim]` slice at offset
`l·n_kv·max_seq·head_dim`, which is exactly the per-layer layout both wrappers check. Step protocol: during a forward
call `pos0 = cache.len()`; each layer calls `append` then `attention` (both use `pos0`), and `advance(T)` commits
once after the last layer. Every call checks `len + T ≤ max_seq` and returns `KvasirError::Shape` on overflow.

**RoPE table ownership.** The cos/sin tables live in `KvCache` (built by `from_config`, `[max_seq, head_dim/2]` each),
not in `Workspace`: their row count is the cache capacity `max_seq`, not `max_tokens`, and every position a
forward call can rotate is exactly a position the cache can hold, so one bound covers both. `KvCache::new` (generic
dimensions, no config) builds none and `rope()` then returns an error. 2 × 4096 × 32 × 4 B = 1 MiB for Llama 3.2 1B.

**Prefill vs decode.** Same layer code and same kernels; `T = prompt_len, pos0 = 0` vs `T = 1, pos0 = cur_len`.
Query `i` attends to keys `[0, pos0+i]`. Only the *last* row goes through the final norm and LM head during
generation (the logit-matching test asks for all rows).

**Forward pass (`model::Llama`).** `Llama` owns `Weights`, `KvCache` and `Workspace` and holds no other device
memory. `forward(gpu, ids, Logits::Last | Logits::All)` runs `ids` at positions `len..len+T` and returns a host
slice of FP32 logits (`[1, vocab]` or `[T, vocab]`, borrowed from the workspace's reused host buffer). Per layer:
`rmsnorm → W_qkv → rope_inplace → cache.append → cache.attention → W_o → hidden += proj → rmsnorm → W_gu →
silu_mul → W_down → hidden += proj`; then `cache.advance(T)`, the final norm on the last `T_head` rows only (written
to the leading rows of `normed`) and the LM head GEMM into `logits`. All views are leading-row slices
(`DeviceTensor::row_view`); nothing is allocated per call. Inputs (`1 ≤ T ≤ max_tokens`, `T_head ≤ max_head_rows`,
`len + T ≤ max_seq`, ids `< vocab`) are validated before any device work, so a rejected call leaves the sequence
unchanged; `len` only advances after the last layer. `Llama::new` rejects `max_tokens > max_seq` and a `max_seq`
above the attention kernel's shared-memory limit (`(max_seq + head_dim + 128)·4 B ≤ 48 KiB`, 12 096 for
head_dim 64), so the limit surfaces at construction instead of mid-generation. `reset`/`truncate` rewind the cache.

## 3. Kernels (validated against CPU before use)

| kernel | grid × block | notes |
|---|---|---|
| `embedding_gather` | T × 256 | ids range-checked on host before upload |
| `rmsnorm` | rows × 256 | FP32 sum of squares; block reduce via warp shuffles |
| `rope_inplace` | T × 128 | one thread per (head, pair) |
| `kv_append` | T × 128 | |
| `attention_decode_{hd}_{g}` | (8, splits) × 128 | `T = 1`, flash-decoding; see below |
| `attention_prefill_{hd}_{g}` | (⌈T/(64/g)⌉, 8) × 128 | `T > 1`, FA2-style on tensor cores; see below |
| `silu_mul` | grid-stride | |
| `add_inplace` | grid-stride | residual add |
| `bf16_to_f16` | grid-stride | counts values outside FP16 range |

**Attention** (A-01). One wrapper, `Gpu::attention(qkv, kc, vc, out, scratch, t, pos0, n_q, n_kv, head_dim, max_seq)`,
dispatches on `t`. Variants are compiled for `head_dim ∈ {64, 128}` × GQA group `g = n_q/n_kv ∈ {1, 2, 4, 8}`; other
shapes are a `Shape` error. `qkv`, `kc`, `vc` must be 16-byte aligned (true for allocations and per-layer cache
slices). There is no kv-length cap beyond `max_seq` (tested to 40k). Scores are FP32 in the exp2
domain (q·k scaled by `log2 e/√d`), with online (running max/sum) softmax.

- *Decode:* one block per (KV head, context split), so each K/V row is loaded once for all `g` query heads. Eight
  lanes read one 128-byte row with 16-byte `ld.global.cs` loads (a warp covers 4 rows per load, 4 loads in flight per
  lane, next round prefetched); partial dot products are reduced with 3 shuffles. Splits: `chunk = max(256,
  ⌈len/128⌉)` keys, `splits = ⌈len/chunk⌉` (≤ 128). With one split the block writes `out` directly. Otherwise each block
  writes `(m, l, acc[hd])` to `scratch` and bumps a per-KV-head counter; the last block to arrive merges all splits and
  resets the counter to 0 (last-block-reduces, no second launch). Chunk 256 beat 64/128 at every benched context:
  fewer, longer streams per block lose less to ramp/tail than more blocks gain in parallelism.
- *Scratch:* `scratch: &mut CudaViewMut<f32>` with at least `kernels::attention_scratch_len(n_q, head_dim)` =
  `n_q + 128·n_q·(head_dim+2)` floats (1.08 MB for Llama 3.2 1B, one buffer shared by all layers). Layout: `n_q` u32
  counter words, then partials `[split, n_q, head_dim+2]`. It must be **zeroed once at allocation** (`alloc_zeros`);
  every call leaves the counters at zero. Prefill does not touch it. Owned by `Workspace` (or the KV cache) — the
  wrapper does not allocate.
- *Prefill:* block = 64/g query tokens × g heads of one KV head = 64 query rows, 4 warps × 16 rows. Q fragments live
  in registers; K/V tiles (64 keys for hd 64, 32 for hd 128) are double-buffered in smem with `cp.async` (zero-filled
  past the causal end, rows padded by 16 B for conflict-free access). S = QKᵀ and O += PV use
  `mma.sync.m16n8k16` (FP16 in, FP32 accumulate); S accumulators are re-packed in registers as the A operand of PV and
  V fragments come from `ldmatrix.trans`. Causal mask per element (`key ≤ pos0 + token`); query tiles are scheduled
  longest-first. No split over keys, so prefill with a long prefix and few new tokens under-fills the GPU.

Unit-test results (`cargo test --test kernels`, 25 tests): all outputs are within one FP16 output rounding
(≈ 4.9e-4 relative) of an f32/f64 CPU reference; FP32-output GEMM is within 7e-7 absolute. Tolerances in the tests
are `1.5–2e-3 rel + 1e-3 abs`, i.e. ~3–4× the rounding floor, so a wrong index or missing term fails loudly.

## 4. Numerics: deviations from HF

1. **Weights/activations FP16, HF reference FP32.** Checkpoint is BF16; we convert to FP16 on the GPU (BF16 has
   8 exponent bits, FP16 only 5, so a load-time counter rejects any |w| > 65504 or non-finite value; mantissa goes
   8 → 11 bits, so in-range values convert exactly). Expected end-to-end logit error is dominated by FP16
   activations; the integration test calibrates its threshold against HF-FP16-vs-HF-FP32 (see §6).
2. **RMSNorm:** HF rounds `x·rsqrt(..)` to the activation dtype, then multiplies by the weight in that dtype.
   We keep the product in FP32 and round once. Strictly more accurate than HF-FP16, same as HF-FP32 in the limit.
3. **Fused projections:** `q|k|v` and `gate|up` are concatenated at load time (one GEMM each instead of 3/2).
   Mathematically identical; per-output-element accumulation order is cuBLAS's choice either way.
4. **Attention** is softmax(QKᵀ/√d)V with FP32 scores and accumulation, FP16 output, computed with an online
   softmax in base 2 (`exp2f`, ≈2 ulp). Decode is FP32 throughout. Prefill rounds P to FP16 for the tensor-core P·V
   (as FA2 does) and sums the *rounded* P into the denominator so the weights still sum to one. All attention tests
   stay within the unchanged `2e-3 rel + 1e-3 abs` (max error 6.6e-4, same as the old kernel).
5. **RoPE tables** are computed in f64 and rounded once to f32 (HF: f32 throughout). Difference ~1e-7.
6. **LM head** outputs FP32 logits directly from the FP16 GEMM (HF-FP32 has FP32 weights; we cannot).
7. **Tied embeddings:** the embedding matrix doubles as the LM head (`tie_word_embeddings = true` for this model);
   untied checkpoints load a separate `lm_head.weight`.
8. **Sampling** runs on the host over the FP32 logits (513 KB per step over PCIe, ~30 µs).

## 5. Build / dependency notes

- `cudarc 0.19` with `driver, cublas, nvrtc, f16, dynamic-linking`. The `nvrtc` feature is enabled **only** because
  `CudaContext::load_module` takes the `nvrtc::Ptx` type; nothing is compiled by NVRTC — PTX comes from `nvcc` at
  build time. The safe cudarc `Gemm<f16>` only produces FP16 output, so `gpu/gemm.rs` calls `cublas::result::gemm_ex`
  directly (needed for FP32 logits).
- `ctx.disable_event_tracking()` is called: with a single stream cudarc's per-allocation events are pure overhead.
- CUDA toolkit 13.x (nvcc at `/opt/cuda` locally); `build.rs` honours `NVCC`, `CUDA_PATH`, `CUDA_HOME`. cudarc's
  version feature is pinned to `cuda-13040` because 0.19 rejects newer minors (13.5) under
  `cuda-version-from-build-system`; the driver API is backward compatible.
- Rust toolchain pinned in `rust-toolchain.toml`. Clippy runs with `-D warnings` in `make ci`.
- GPU tests (`engine/tests/kernels.rs`, `weights.rs`) need `--features gpu-tests`; plain `cargo test` runs CPU tests only.

## 6. Verification plan

**Forward-pass test thresholds (`tests/forward.rs`, synthetic model).** The tiny config of `tests/weights.rs`
(hidden 64, 2 layers, 4 q / 2 kv heads of 16, inter 128, vocab 100) with random BF16 weights (linear `U(±√(3/in))`,
norms `U(0.5, 1.5)`, embedding `U(±1)`; logit rms ≈ 5) is run on the GPU and compared with an f64 CPU forward
written in the test, in two flavours: exact, and one that rounds to FP16 at every point where the GPU stores an FP16
activation. Metric: max |logit error| over all rows.

- The error is quantised by single FP16 rounding flips. One ulp of a hidden value `x_k` (2⁻¹⁰·|x_k|) moves a logit by
  ≈ 2⁻¹⁰ · (|x_k|/rms(x)) · w_k · |E_vk| ≲ 2⁻¹⁰ · 3 · 1.5 · 1 ≈ 4e-3; a few such flips across 2 layers give ~1–2e-2.
  Any change of accumulation order (cuBLAS picks different algorithms for T = 1 and T = 24, f32 vs f64 sums) flips
  some roundings, so the FP16-emulating reference, GPU decode-vs-prefill and the exact reference all sit at the
  *same* floor; emulation does not buy a tighter bound.
- Measured over 30 random models × 24 tokens: worst vs FP16-emulated 2.14e-2, vs exact 2.40e-2, decode-vs-prefill
  1.99e-2. **`TOL = 5e-2`** (≈ 2× the worst of 30, ≈ 1% of logit rms) for every comparison that crosses an FP16
  rounding: GPU vs either reference, decode vs prefill, chunked prefill + decode vs reference, truncate + recompute.
- Power check, asserted in the test: references with a plausible bug differ from the GPU by ≥ 10×TOL; the smallest
  over the 30 models were 13.6 (GQA `h % n_kv` instead of `h / group`), 10.4 (interleaved instead of rotate_half RoPE)
  and 10.3 (query `i` also sees key `i+1`). A uniform position shift is *not* detectable (RoPE is relative), by design.
- `Logits::Last` vs the last row of `Logits::All` for the same tokens differ only in the FP32-output head GEMM
  (layers run with the same T, final norm is per row): bound 2·γ₆₃·Σ|aᵢbᵢ| ≈ 3e-4, measured 3.3e-6,
  **`TOL_HEAD_ONLY = 1e-3`**. A reset followed by the same call must be bit-identical.
- Top-1 must agree with the exact reference at every position whose reference top-2 margin exceeds 2·TOL.

**Real-model logit test (S1-06/S1-07, not yet built).**

- `ref/dump_logits.py`: HF `LlamaForCausalLM`, FP32, eager attention → prompt ids, prefill logits (all positions),
  greedy tokens (48 steps), decode logits for the first 16 steps (teacher-forced). Also an HF-**FP16** run, giving the
  intrinsic FP16-vs-FP32 error floor.
- `tests/logits.rs` thresholds are *calibrated from that floor* rather than guessed: max abs err ≤ 2× the HF-FP16 floor
  and ≤ 1.0 absolute, mean abs err ≤ 0.05; top-1 agreement 100% on decode steps and ≥ 99% over prefill positions
  (excluding positions whose reference top-2 margin < 0.05, reported and capped at 1%); ≥ 32 greedy tokens identical.
- Roofline: decode bytes = `Weights::decode_bytes()` (all layer matrices + norms + tied head ≈ 2.47 GB) / 504 GB/s
  ≈ 4.9 ms/token ≈ 204 tok/s upper bound (KV reads add < 1% at short context).

## 7. Status

| step | state |
|---|---|
| gpu wrappers, build.rs | done |
| 8 kernels + CPU-reference tests, GEMM test | done, 25/25 passing |
| config, RoPE table, weight loader | done; loader tested on a synthetic BF16 checkpoint (exact fused-layout check) |
| real Llama 3.2 1B weights | **missing**: gated repo, backlog `ENV-weights` |
| workspace (activation buffers, ids upload, logits D2H into a reused host buffer) | done (S1-01); CPU size tests + GPU alloc/round-trip test |
| KV cache (`KvCache`) | done (S1-02); `tests/kv.rs` (6 GPU tests: exact placement at several `pos0`, overflow errors, attention via layer view) |
| forward pass (`Llama::forward`, prefill / chunked prefill / decode) | done (S1-03); `tests/forward.rs` (6 GPU tests vs f64 CPU reference on a synthetic model, thresholds §6) |
| kernel microbenchmarks + stored baseline (`make bench`, H-01) | done; numbers in §8 |
| flash-decoding / FA2 prefill attention (A-01) | done; decode ctx 2048 at 60% of DRAM bandwidth, prefill T=128 25× faster |
| sampling, CLI, ref script, logit test, perf report | ordered in `.agent/backlog.yaml` (S1-04 … S1-08) |

## 8. Kernel microbenchmarks

`make bench` (`engine/benches/kernels.rs`, `harness = false`, needs `gpu-tests`) times every kernel at Llama 3.2 1B
dims, decode (`T=1`) and prefill (`T=128`), and fails if any median is > 10% slower than `bench/kernels-baseline.json`;
`make bench-baseline` rewrites the baseline. No criterion: CUDA events are what we need and it would be a new dependency.

Method: buffers are allocated before timing; each sample queues a ~1 ms filler GEMM, then CUDA events around `reps`
back-to-back launches (sized to ~300 µs), so host launch latency is hidden but inter-kernel gaps are included.
Median of 101 samples after warmup and a 0.5 s clock ramp. Weights and the read KV span rotate over ≥ 160 MiB of copies so
they come from DRAM, as in a real 16-layer pass; activations are deliberately L2-resident (as they are in a pass),
which is why some prefill elementwise kernels exceed the DRAM roofline. Reruns on an idle GPU agree within ~2%.

Initial baseline (RTX 4070, driver 620.70.06), µs; % = of 504 GB/s, or of the 58.3 TFLOP/s FP16 spec peak for
prefill GEMMs:

| kernel | T=1 | % | T=128 | % |
|---|---:|---:|---:|---:|
| embedding | 3.25 | 1 | 4.24 | 49 |
| rmsnorm | 3.17 | 1 | 3.57 | 59 |
| rope | 2.66 | 1 | 3.07 | 87 |
| kv_append | 1.68 | 0 | 1.84 | 56 |
| silu_mul | 1.42 | 7 | 8.72 | L2 |
| add | 1.30 | 2 | 2.46 | L2 |
| attention, ctx 512 / 2048 / 8192 (A-01) | 7.16 / 14.0 / 43.3 | 29 / 60 / 77 | 6.99 (ctx 128) | 37 (17% FP16 peak) |
| gemm qkv / o / gate_up / down | 29.7 / 20.5 / 146 / 74.8 | 84 / 81 / 92 / 89 | 46.2 / 28.3 / 187 / 119 | 60 / 64 / 78 / 61 |
| lm_head (FP32 out) | 1113 | 94 | 1394 | 83 |

Before A-01 attention was 79.5 / 311 / 1231 µs (3% of bandwidth) at decode and 178 µs at prefill: 32 blocks on 46 SMs,
one thread per key walking 64 elements serially, 2 threads per output dim in P·V, and every KV head read by 4 blocks.

Decode at ctx 2048 now sums to ≈ 16 × (271 GEMM + 14 attention + 15 small) µs + 1113 µs ≈ 5.9 ms/token, vs the 4.9 ms
roofline. Decode attention has a ~4 µs fixed cost per call (launch gap, DRAM latency ramp, split merge; ctx 512 is
29% of bandwidth for that reason) and streams at ≈ 430 GB/s marginally. The decode GEMMs are within 6–19% of
bandwidth. The small kernels are launch-bound (~1.3–3.3 µs each, ~10 per layer) and call for fusion and CUDA graphs:
≈ 16 × 15 µs ≈ 0.25 ms/token. Remaining gap to the roofline at ctx 2048: GEMM shortfall ≈ 0.5 ms, small kernels
≈ 0.25 ms, attention ≈ 0.1 ms (its fixed cost).
