"""Which effects cause the remaining loss of the best design? Each effect off, and each alone."""
import json, os
from hw import Chip, EFFECTS
from run import Bench, bits_per_byte, RUNS
from sim import AnalogQwen

b = Bench(6, 1024)
ideal = b.ideal_bpb()["mean"]
rows = [json.loads(l) for l in open(os.path.join(RUNS, "search.jsonl"))]
best = dict(min(rows, key=lambda r: r["dbpb"])["chip"])

def loss(**over):
    out = []
    for seed in (0, 1):
        m = AnalogQwen(b.digital, Chip(**{**best, **over, "seed": seed}), b.calib)
        out.append(100 * (bits_per_byte(m, b.data)["mean"] / ideal - 1))
    return sum(out) / 2

base = loss()
print(f"best design, all effects on: {base:+.2f}%  (mean of 2 chip draws)", flush=True)
active = [e for e in EFFECTS if e not in ("gdn_mem", "kv_mem", "softmax")]   # digital memory: exact
print(f"{'effect':10s} {'loss without it':>16s} {'loss with only it':>18s}")
for e in active:
    off = loss(off=e)
    only = loss(off=",".join(x for x in EFFECTS if x != e))
    print(f"{e:10s} {off:+15.2f}% {only:+17.2f}%", flush=True)
print(f"none (all off): {loss(off=','.join(EFFECTS)):+.2f}%")
