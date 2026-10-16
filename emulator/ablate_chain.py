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
print(f"AC chain, all effects: {loss():+.2f}%", flush=True)
for lb in (0, 10, 12):
    print(f"  chip-to-chip links {lb or 'analog (no ADC)'} bits: {loss(link_bits=lb):+.2f}%", flush=True)
for e in [x for x in EFFECTS if x not in ("gdn_mem", "kv_mem", "softmax")]:
    print(f"  without {e:6s}: {loss(off=e):+.2f}%", flush=True)
