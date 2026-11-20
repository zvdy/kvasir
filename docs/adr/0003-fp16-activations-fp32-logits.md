# 3. FP16 weights and activations, FP32 logits

Status: accepted

## Context

Llama 3.2 checkpoints are BF16. sm_89 supports both FP16 and BF16 tensor cores. FP16 has 3 more mantissa bits.

## Decision

Convert weights BF16→FP16 on the GPU at load time, rejecting values outside FP16 range. Activations are FP16; reductions
(RMSNorm, softmax, attention) accumulate in FP32; logits are produced in FP32 and never rounded to FP16.

## Consequences

Accuracy is bounded by FP16 activations; the logit test calibrates thresholds against HF-FP16 vs HF-FP32 (DESIGN §6).
A checkpoint with |w| > 65504 fails to load rather than silently overflowing.
