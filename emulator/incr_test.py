"""Incremental (cached) mode vs full recompute, from the hardware package.
  1. physics off: logits of every position must match exactly
  2. physics on: next-token loss over 256 tokens must match statistically
  3. speed of incremental generation on the emulated chip"""
import os, time
import mlx.core as mx
import numpy as np
from emulate_hw import load_package
from hw import Chip
from run import encode_set
from sim import AnalogQwen

m, tok, man, cal = load_package(os.environ.get("KLM_PKG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "hw_qwen35")))
ids = encode_set(tok, 1, 600)[0][1]


def incremental_logits(model, ids, n_prefill):
    cache = model.make_cache()
    out = [model(mx.array(ids[:n_prefill])[None], cache=cache)[0]]
    mx.eval(out[-1], cache)
    for t in ids[n_prefill:]:
        out.append(model(mx.array([[t]]), cache=cache)[0])
        mx.eval(out[-1], cache)
    return mx.concatenate(out, axis=0)


exact = AnalogQwen(m, Chip(ideal=True), cal)
full = exact(mx.array(ids[:96])[None])[0]
inc = incremental_logits(exact, ids[:96], 40)
print(f"1. physics off, 96 tokens (40 prefill + 56 one by one): max |logit diff| = {float(mx.abs(full - inc).max()):.2e}"
      f" (logit scale {float(mx.abs(full).max()):.0f})", flush=True)

chip = AnalogQwen(m, Chip(**{**man["design"], "seed": 0}), cal)
seq = ids[:513]
y = np.array(seq[257:513])


def nll(logits):
    lg = logits.astype(mx.float32)
    return np.array(mx.logsumexp(lg, -1) - mx.take_along_axis(lg, mx.array(y)[:, None], -1)[:, 0])


f = nll(chip(mx.array(seq[:512])[None])[0][256:512])
i = nll(incremental_logits(chip, seq[:512], 256)[256:512])
e = nll(exact(mx.array(seq[:512])[None])[0][256:512])
print(f"2. physics on, tokens 256-511: mean loss exact {e.mean():.4f} | chip full recompute {f.mean():.4f} | "
      f"chip incremental {i.mean():.4f} (difference {i.mean() - f.mean():+.4f}, token-level noise {np.std(i - f) / np.sqrt(len(y)):.4f})", flush=True)

cache = chip.make_cache()
mx.eval(chip(mx.array(ids[:512])[None], cache=cache))
t0 = time.time()
nxt = ids[512]
for _ in range(64):
    lg = chip(mx.array([[nxt]]), cache=cache)[0, -1]
    mx.eval(lg, cache)
    nxt = int(mx.argmax(lg))
print(f"3. incremental generation on the emulated chip at ~550-token context: {64 / (time.time() - t0):.1f} tokens/s "
      f"(full recompute was 6.7 tokens/s)", flush=True)
