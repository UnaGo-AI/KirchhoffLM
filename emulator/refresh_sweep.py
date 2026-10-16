"""Level-restoration (refresh) of the analog DeltaNet memory: bits x period x rounding, at
C = 10 fF (noise that breaks the plain analog memory). Only the DeltaNet memory is non-ideal,
so the result isolates the memory; then the same grid on the full reference chip."""
import json, os, sys
from hw import Chip, EFFECTS
from run import Bench, REFERENCE, bits_per_byte, RUNS
from sim import AnalogQwen

b = Bench(6, 1024)
ideal = b.ideal_bpb()["mean"]
only_mem = ",".join(e for e in EFFECTS if e != "gdn_mem")
path = os.path.join(RUNS, "refresh.jsonl")
done = {json.dumps(json.loads(l)["cfg"], sort_keys=True) for l in open(path)} if os.path.exists(path) else set()
grid = [dict(refresh_bits=0)]
grid += [dict(refresh_bits=bt, refresh_every=R, refresh_stoch=st) for bt in (4, 6, 8, 10, 12, 16)
         for R in (1, 8, 64) for st in (True, False)]
for scope in ("memory_only", "full_chip"):
    for over in grid:
        cfg = {**REFERENCE, "memory": "igzo", "c_mem": 10e-15, **over}
        if scope == "memory_only":
            cfg["off"] = only_mem
        key = json.dumps({"scope": scope, **over}, sort_keys=True)
        if key in done:
            continue
        chip = Chip(**cfg).validate()
        m = AnalogQwen(b.digital, chip, b.calib)
        r = bits_per_byte(m, b.data)["mean"]
        e = m.energy_per_token(2048)
        row = {"cfg": {"scope": scope, **over}, "dbpb": r - ideal, "rel": r / ideal - 1, "energy_j": e["total"], "mem_j": e["memory"]}
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{scope:11s} {json.dumps(over):70s} loss {100 * row['rel']:+8.2f}%  mem {e['memory'] * 1e6:8.2f} uJ/tok  total {e['total'] * 1e6:8.2f} uJ/tok", flush=True)
