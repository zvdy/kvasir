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
    llama.rs          forward pass (prefill / decode)                    [not written yet]
    workspace.rs      preallocated activation buffers                    [not written yet]
  src/kv/             contiguous KV cache                                [not written yet]
  src/sampling.rs     greedy / temperature / top-p on host               [not written yet]
  src/bin/kvasir.rs   `kvasir generate --model <dir> --prompt ...`       [not written yet]
engine/tests/         kernels.rs (per-kernel vs CPU), weights.rs (synthetic checkpoint)
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
| RoPE cos/sin | `[max_seq, 32]` FP32 | built on host, uploaded once |

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

**Prefill vs decode.** Same layer code and same kernels; `T = prompt_len, pos0 = 0` vs `T = 1, pos0 = cur_len`.
Query `i` attends to keys `[0, pos0+i]`. Only the *last* row goes through the final norm and LM head during
generation (the logit-matching test asks for all rows).

## 3. Kernels (naive, correct, validated against CPU before use)

| kernel | grid × block | notes |
|---|---|---|
| `embedding_gather` | T × 256 | ids range-checked on host before upload |
| `rmsnorm` | rows × 256 | FP32 sum of squares; block reduce via warp shuffles |
| `rope_inplace` | T × 128 | one thread per (head, pair) |
| `kv_append` | T × 128 | |
| `attention` | (32, T) × 128 | block per (head, query). Scores in dynamic smem (FP32), 3 passes (max, exp+sum, P·V). Needs `(pos0+T+64+128)·4 B ≤ 48 KiB` → kv length ≲ 11.9k; wrapper errors beyond that |
| `silu_mul` | grid-stride | |
| `add_inplace` | grid-stride | residual add |
| `bf16_to_f16` | grid-stride | counts values outside FP16 range |

Unit-test results (`cargo test --test kernels`, 16 tests): all outputs are within one FP16 output rounding
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
4. **Attention** is a plain softmax(QKᵀ/√d)V with FP32 scores and FP32 accumulation, FP16 output (no
   flash/online softmax yet). Max subtraction is exact; `__expf` is used (≈2 ulp FP32, far below FP16 rounding).
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
- nvcc 13.4 at `/opt/cuda`; `build.rs` honours `NVCC`, `CUDA_PATH`, `CUDA_HOME`.
- Known cosmetic clippy lints (`is_multiple_of`, `chunks_exact`) left as is.

## 6. Verification plan (forward pass + reference, not yet built)

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
| 8 kernels + CPU-reference tests, GEMM test | done, 16/16 passing |
| config, RoPE table, weight loader | done; loader tested on a synthetic BF16 checkpoint (exact fused-layout check) |
| real Llama 3.2 1B weights | **missing** — `hf download` was refused (not logged in / gated repo not approved) |
| forward pass, KV cache, sampling, CLI, ref script, logit test, perf report | waiting for sign-off on this document |
