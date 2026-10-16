"""Alternative design: programmable switched-capacitor banks, optionally layer-multiplexed (one physical array for
all layers) and program-and-verify calibrated. Quality of Qwen3.5-0.8B from the hardware package, 3 chips each,
next to the fixed-capacitor design. Results -> runs/multiplex_quality.json."""
import json, os, time
from emulate_hw import load_package
from hw import Chip
from run import bits_per_byte, encode_set
from sim import AnalogQwen

ROOT = os.path.dirname(os.path.abspath(__file__))
model, tok, man, cal = load_package(os.environ.get("KLM_PKG", os.path.join(ROOT, "hw_qwen35")))
data = encode_set(tok, 6, 1024)
ref = bits_per_byte(AnalogQwen(model, Chip(ideal=True), cal), data)["mean"]
base = man["design"]
CONFIGS = [
    ("fixed capacitors, one chip per layer (current design)", {}),
    ("programmable bank, one array per layer", {"device": "cap_bank"}),
    ("programmable bank, one array per layer, program-and-verify", {"device": "cap_bank", "pv_calibrate": True}),
    ("programmable bank, shared array (multiplexed)", {"device": "cap_bank", "shared_array": True}),
    ("programmable bank, shared array (multiplexed), program-and-verify", {"device": "cap_bank", "shared_array": True, "pv_calibrate": True}),
]
out = {"reference": ref, "configs": []}
for label, over in CONFIGS:
    t0 = time.time()
    losses = []
    for seed in (0, 1, 2):
        chip = AnalogQwen(model, Chip(**{**base, **over, "seed": seed}), cal)
        losses.append(100 * (bits_per_byte(chip, data)["mean"] / ref - 1))
    out["configs"].append({"label": label, "settings": over, "loss_pct": losses})
    print(f"{label:68s} " + " / ".join(f"{x:+.2f}%" for x in losses) + f"   ({time.time() - t0:.0f}s)", flush=True)
    json.dump(out, open(os.path.join(ROOT, "runs", "multiplex_quality.json"), "w"), indent=1)
