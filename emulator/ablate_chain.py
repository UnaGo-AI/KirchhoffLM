"""Which effects make up the remaining loss of the AC per-layer chip chain?"""
import json, os
import hw
from hw import Chip, EFFECTS
from run import Bench, bits_per_byte, RUNS
from sim import AnalogQwen
from layer_bench import DESIGNS
b = Bench(6, 1024)
ideal = b.ideal_bpb()["mean"]
AC = DESIGNS["AC"]
def loss(**over):
    return sum(100 * (bits_per_byte(AnalogQwen(b.digital, Chip(**{**AC, **over, "seed": s}), b.calib), b.data)["mean"] / ideal - 1)
               for s in (0, 1)) / 2
out = {"all": loss(), "links": {}, "without": {}}
print(f"AC chain, all effects: {out['all']:+.2f}%", flush=True)
for lb in (0, 10, 12):
    out["links"][str(lb)] = loss(link_bits=lb)
    print(f"  chip-to-chip links {lb or 'analog (no ADC)'} bits: {out['links'][str(lb)]:+.2f}%", flush=True)
for e in [x for x in EFFECTS if x not in ("gdn_mem", "kv_mem", "softmax")]:
    out["without"][e] = loss(off=e)
    print(f"  without {e:6s}: {out['without'][e]:+.2f}%", flush=True)
json.dump(out, open(os.path.join(RUNS, "ablate_chain.json"), "w"), indent=1)
