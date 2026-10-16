"""1k-token context on the emulated chip (hardware package only) vs the exact model.
  1. recall: a 4-digit number hidden early in ~1000 tokens of filler, asked for at the end (20 trials)
  2. bits/byte by position bucket over 1024-token texts
  3. hardware cost of processing a 1024-token prompt (pipelined chain of layer chips)"""
import json, math, os, random, time
import mlx.core as mx
import numpy as np
from emulate_hw import load_package
from hw import Chip
from run import encode_set
from sim import AnalogQwen, t_token

PKG = os.environ.get("KLM_PKG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "hw_qwen35"))
model, tok, man, calib = load_package(PKG)
exact = AnalogQwen(model, Chip(ideal=True), calib)
chip = AnalogQwen(model, Chip(**{**man["design"], "seed": 0}), calib)
texts = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_texts.json")))


def greedy(m, ids, n):
    out = []
    for _ in range(n):
        nxt = int(mx.argmax(m(mx.array(ids + out)[None])[0, -1]))
        out.append(nxt)
    return out


# 1. recall at ~1k context
rng = random.Random(7)
hits = {"exact": 0, "chip": 0}
lens = []
for trial in range(20):
    code = rng.randint(1000, 9999)
    filler = tok.encode(texts["wikipedia"][trial % 40])[:950]
    pos = rng.randint(20, 300)                                   # where the fact is hidden
    fact = tok.encode(f" Note: the secret number is {code}. ")
    question = tok.encode(f"\n\nQuestion: What is the secret number mentioned above?\nAnswer: The secret number is")
    ids = filler[:pos] + fact + filler[pos:] + question
    lens.append(len(ids))
    res = {}
    for name, m in (("exact", exact), ("chip", chip)):
        ans = tok.decode(greedy(m, ids, 6))                     # digits are single tokens
        res[name] = (str(code) in ans.replace(" ", ""), ans.strip())
        hits[name] += res[name][0]
    print(f"trial {trial:2d}: {len(ids)} tokens, fact at token {pos}, code {code} -> "
          f"exact {'ok' if res['exact'][0] else 'MISS'} ({res['exact'][1]!r}), chip {'ok' if res['chip'][0] else 'MISS'} ({res['chip'][1]!r})", flush=True)
print(f"\nrecall at {min(lens)}-{max(lens)} tokens: exact model {hits['exact']}/20, emulated chip {hits['chip']}/20\n", flush=True)
SAVED = {"recall": {"tokens": [min(lens), max(lens)], "exact": hits["exact"], "chip": hits["chip"], "trials": 20}}

import sys
if "--recall-only" in sys.argv:
    sys.exit(0)
# 2. bits/byte by position over 1024-token texts
data = encode_set(tok, 6, 1024)
buckets = [(0, 256), (256, 512), (512, 768), (768, 1023)]
acc = {n: np.zeros(len(buckets)) for n in ("exact", "chip")}
nbytes = np.zeros(len(buckets))
for dom, ids, _ in data:
    x, y = mx.array(ids)[None, :-1], np.array(ids[1:])
    for name, m in (("exact", exact), ("chip", chip)):
        lg = m(x)[0].astype(mx.float32)
        nll = np.array(mx.logsumexp(lg, axis=-1) - mx.take_along_axis(lg, mx.array(y)[:, None], axis=-1)[:, 0])
        for b, (s, e) in enumerate(buckets):
            acc[name][b] += nll[s:e].sum()
    for b, (s, e) in enumerate(buckets):
        nbytes[b] += len(tok.decode(ids[s + 1:e + 1]).encode("utf-8"))
print("bits/byte by position     exact    chip     difference")
SAVED["positions"] = []
for b, (s, e) in enumerate(buckets):
    ex, ch = acc["exact"][b] / math.log(2) / nbytes[b], acc["chip"][b] / math.log(2) / nbytes[b]
    SAVED["positions"].append({"from": s, "to": e, "exact": ex, "chip": ch})
    print(f"  tokens {s:4d}-{e:4d}       {ex:.4f}   {ch:.4f}   {100 * (ch / ex - 1):+.2f}%")

# 3. hardware cost of a 1024-token prompt
n_chips = chip.n_layers
stage = t_token(chip) / n_chips                       # time each layer chip spends per token
t_prefill = (1024 + n_chips - 1) * stage             # pipelined: one token enters every stage time
e_prefill = sum(chip.energy_per_token(c)["total"] for c in range(1, 1025))
e_first, e_last = chip.energy_per_token(1)["total"], chip.energy_per_token(1024)["total"]
print(f"\nhardware: 1024-token prompt in {t_prefill * 1e3:.2f} ms (pipelined, {stage * 1e9:.0f} ns per chip stage), "
      f"{e_prefill * 1e3:.2f} mJ total ({e_first * 1e6:.1f} uJ/token at the start -> {e_last * 1e6:.1f} uJ/token at 1k); "
      f"then {1 / t_token(chip):,.0f} tokens/s generation")
SAVED["prefill_1024"] = {"seconds": t_prefill, "joules": e_prefill}
json.dump(SAVED, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "context_1k.json"), "w"), indent=1)
t0 = time.time(); mx.eval(chip(mx.array(data[0][1][:1024])[None]))
print(f"(emulating that 1024-token prompt on the M3 Max takes {time.time() - t0:.1f} s)")
