#!/usr/bin/env python3
"""Dump HF reference logits for kvasir's logit-matching test. Output format: DESIGN.md §6."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PROMPT = (
    "The history of the Roman Empire spans more than a thousand years. "
    "In this short summary, we describe its rise, its height under the Antonines, and its decline."
)
FORMAT_VERSION = 1


@dataclass
class Run:
    prefill: "torch.Tensor"
    decode: "torch.Tensor"
    tokens: list[int]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path("models/Llama-3.2-1B"))
    p.add_argument("--out", type=Path, default=Path("ref/out"))
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--steps", type=int, default=48, help="greedy tokens to generate")
    p.add_argument("--decode-logit-steps", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--revision", default="unknown", help="HF revision of the checkpoint, recorded in meta.json")
    a = p.parse_args()
    if a.steps < 1 or not 0 <= a.decode_logit_steps < a.steps:
        p.error("need steps >= 1 and 0 <= decode-logit-steps < steps")
    return a


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def run(model, prompt_ids: list[int], steps: int, n_logits: int, forced: list[int] | None) -> Run:
    import torch

    with torch.inference_mode():
        out = model(torch.tensor([prompt_ids]), use_cache=True)
        prefill = out.logits[0].float()
        tokens = [int(prefill[-1].argmax())]
        decode = []
        past = out.past_key_values
        for i in range(steps - 1):
            feed = forced[i] if forced is not None else tokens[i]
            out = model(torch.tensor([[feed]]), past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[0, -1].float()
            if i < n_logits:
                decode.append(logits)
            tokens.append(int(logits.argmax()))
    if n_logits:
        decode_t = torch.stack(decode)
    else:
        decode_t = torch.empty((0, prefill.shape[1]), dtype=torch.float32)
    return Run(prefill, decode_t, tokens)


def write_f32(path: Path, t) -> dict:
    import numpy as np

    a = np.ascontiguousarray(t.detach().cpu().numpy(), dtype="<f4")
    path.write_bytes(a.tobytes())
    return {"file": path.name, "dtype": "f32le", "shape": list(a.shape)}


def write_u32(path: Path, ids: list[int]) -> dict:
    import numpy as np

    a = np.asarray(ids, dtype="<u4")
    path.write_bytes(a.tobytes())
    return {"file": path.name, "dtype": "u32le", "shape": list(a.shape)}


def err_stats(ref, other) -> dict:
    d = (ref - other).abs()
    top1 = (ref.argmax(-1) == other.argmax(-1)).float()
    return {
        "max_abs_err": float(d.max()) if d.numel() else 0.0,
        "mean_abs_err": float(d.mean()) if d.numel() else 0.0,
        "top1_agreement": float(top1.mean()) if top1.numel() else 1.0,
    }


def main() -> int:
    args = parse_args()
    weights = args.model / "model.safetensors"
    if not weights.is_file():
        print(f"ref: weights absent ({weights}); nothing to dump (backlog ENV-weights)")
        return 0

    import torch
    import transformers
    from transformers import AutoTokenizer, LlamaForCausalLM

    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompt_ids = [int(i) for i in tok(args.prompt, add_special_tokens=True)["input_ids"]]

    def load(dtype):
        m = LlamaForCausalLM.from_pretrained(
            args.model, dtype=dtype, attn_implementation="eager", local_files_only=True
        )
        return m.eval()

    fp32 = run(load(torch.float32), prompt_ids, args.steps, args.decode_logit_steps, None)
    fp16 = run(load(torch.float16), prompt_ids, args.steps, args.decode_logit_steps, fp32.tokens)

    args.out.mkdir(parents=True, exist_ok=True)
    o = args.out
    files = {
        "prompt_ids": write_u32(o / "prompt_ids.u32", prompt_ids),
        "greedy_ids": write_u32(o / "greedy_ids.u32", fp32.tokens),
        "prefill_logits": write_f32(o / "prefill_logits.f32", fp32.prefill),
        "decode_logits": write_f32(o / "decode_logits.f32", fp32.decode),
        "prefill_logits_hf16": write_f32(o / "prefill_logits_hf16.f32", fp16.prefill),
        "decode_logits_hf16": write_f32(o / "decode_logits_hf16.f32", fp16.decode),
    }
    meta = {
        "format_version": FORMAT_VERSION,
        "model": str(args.model),
        "revision": args.revision,
        "model_safetensors_sha256": sha256(weights),
        "prompt": args.prompt,
        "prompt_ids": prompt_ids,
        "greedy_ids": fp32.tokens,
        "greedy_ids_hf16_teacher_forced": fp16.tokens,
        "steps": args.steps,
        "decode_logit_steps": args.decode_logit_steps,
        "vocab_size": int(fp32.prefill.shape[1]),
        "seed": args.seed,
        "attn_implementation": "eager",
        "files": files,
        "hf16_floor": {
            "prefill": err_stats(fp32.prefill, fp16.prefill),
            "decode": err_stats(fp32.decode, fp16.decode),
        },
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }
    (o / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"ref: wrote {o} (prompt {len(prompt_ids)} tokens, {args.steps} greedy steps)")
    print(f"ref: hf16 floor {json.dumps(meta['hf16_floor'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
