"""Whole chains of per-layer chips: uniform reference, uniform precise, best search point, and
mixed chains with precise chips only on the sensitive layers (from layer_map.jsonl)."""
import json, os
from hw import Chip
from run import Bench, REFERENCE, bits_per_byte, RUNS
from sim import AnalogQwen, t_token
from layer_map import PROFILES

b = Bench(6, 1024)
ideal = b.ideal_bpb()["mean"]
best = json.loads(open(os.path.join(RUNS, "search.jsonl")).read().splitlines()[0])  # placeholder, replaced below
rows = [json.loads(l) for l in open(os.path.join(RUNS, "search.jsonl"))]
best = min(rows, key=lambda r: r["dbpb"])["chip"]
BEST = {k: v for k, v in best.items() if k not in ("layer_over",)}
BEST["link_bits"] = 8
ref, pre = PROFILES["reference"], PROFILES["precise"]

def chain(base, special, layers):
    return Chip(**{**base, "layer_over": {str(i): special for i in layers}})

designs = {
    "all reference": Chip(**ref),
    "all precise": Chip(**pre),
    "precise on 0,1,2,6,23 / reference rest": chain(ref, pre, [0, 1, 2, 6, 23]),
    "precise on 0-2,6,10,15,23 / reference rest": chain(ref, pre, [0, 1, 2, 6, 10, 15, 23]),
    "best search point everywhere": Chip(**BEST),
    "best on 0,1,2,6,23 / precise rest": chain(pre, BEST, [0, 1, 2, 6, 23]),
}
for name, chip in designs.items():
    for seed in (0, 1):
        chip.seed = seed
        m = AnalogQwen(b.digital, chip, b.calib)
        r = bits_per_byte(m, b.data)["mean"]
        e = m.energy_per_token(2048)
        print(f"{name:44s} seed {seed}  loss {100 * (r / ideal - 1):+7.2f}%  E {e['total'] * 1e6:6.2f} uJ/tok "
              f"(arrays {e['arrays+amps'] * 1e6:.2f}, memory+attn {(e['memory'] + e['attention']) * 1e6:.2f}, links {e['links'] * 1e6:.2f})  "
              f"{1 / t_token(m):,.0f} tok/s", flush=True)
