"""Quality when all layers share the same physical capacitor grids (layer-multiplexed engine) vs separate
grids per layer (one chip per layer). Same three chip draws, hardware package only."""
import json, os
from emulate_hw import load_package
from hw import Chip
from run import bits_per_byte, encode_set
from sim import AnalogQwen

ROOT = os.path.dirname(os.path.abspath(__file__))
model, tok, man, cal = load_package(os.environ.get("KLM_PKG", os.path.join(ROOT, "hw_qwen35")))
data = encode_set(tok, 6, 1024)
ref = bits_per_byte(AnalogQwen(model, Chip(ideal=True), cal), data)["mean"]
out = {"reference": ref}
for shared in (False, True):
    losses = []
    for seed in (0, 1, 2):
        r = bits_per_byte(AnalogQwen(model, Chip(**{**man["design"], "seed": seed, "shared_array": shared}), cal), data)["mean"]
        losses.append(100 * (r / ref - 1))
    out["shared" if shared else "separate"] = losses
    print(f"{'shared grids (multiplexed)' if shared else 'separate grids per layer  '}: " + " / ".join(f"{x:+.2f}%" for x in losses), flush=True)
json.dump(out, open(os.path.join(ROOT, "runs", "shared_array.json"), "w"), indent=1)
