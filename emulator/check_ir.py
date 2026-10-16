"""Fit the behavioural IR-drop model to the exact tile solves: gain = exp(-a), a = row + driver + act * column."""
import json, numpy as np, mlx.core as mx
from run import Bench
b = Bench(1, 64)
W = np.array(b.digital.language_model.model.layers[0].mlp.gate_proj.weight.astype(mx.float32))
u_all = W / np.abs(W).max(axis=1, keepdims=True)
exact = {(r["tile"], r["r_seg"]): r for r in json.load(open("runs/circuit_tile.json")) if r["config"].startswith("DC")}

def model(n, R, act):
    u = u_all[:n, :n]
    g = float((np.abs(u) * 73e-6 / 2).mean())
    p = 2 * np.arange(n)
    a_row = R * g * (p * 2 * n - p ** 2 / 2) + 10.0 * g * 2 * n
    q = np.arange(n)
    a_col = (R * g * act * (q * n - q ** 2 / 2)).mean()
    return np.exp(-(a_row[:8].mean() + a_col)), np.exp(-(a_row[-8:].mean() + a_col))

best = None
for act in np.linspace(0.0, 2.0, 81):
    err = sum((model(n, R, act)[0] - exact[(n, R)]["gain_first"]) ** 2 + (model(n, R, act)[1] - exact[(n, R)]["gain_last"]) ** 2
              for n in (128, 256) for R in (0.5, 2.0, 4.0))
    if best is None or err < best[0]:
        best = (err, act)
act = best[1]
print(f"fitted act_ratio {act:.3f}  (rms gain error {np.sqrt(best[0] / 12):.3f})")
for n in (128, 256):
    for R in (0.5, 2.0, 4.0):
        f, l = model(n, R, act)
        e = exact[(n, R)]
        print(f"tile {n} R={R}: model {f:.3f}/{l:.3f}   exact {e['gain_first']:.3f}/{e['gain_last']:.3f}")
