"""Per-layer precision map, one chip per layer: only layer i runs on a physical chip, every other
layer is exact; repeat for all layers and for a reference and a precise chip profile.
Layers that tolerate the reference chip can use cheap hardware; the rest need precise chips."""
import json, os
from hw import Chip
from run import Bench, REFERENCE, bits_per_byte, RUNS
from sim import AnalogQwen

PROFILES = {
    "reference": dict(REFERENCE, link_bits=8),
    "precise": dict(REFERENCE, device="flash", g_max=2e-6, v_read=0.1, fs_pct=100.0, offset_trim=0.01,
                    p_tia=50e-6, adc_bits=10, link_bits=10),
}

b = Bench(6, 1024)
ideal = b.ideal_bpb()["mean"]
path = os.path.join(RUNS, "layer_map.jsonl")
done = {(r["profile"], r["layer"]) for r in map(json.loads, open(path))} if os.path.exists(path) else set()
n = len(b.digital.language_model.model.layers)
for prof, cfg in PROFILES.items():
    for i in range(n):
        if (prof, i) in done:
            continue
        chip = Chip(ideal=True, layer_over={str(i): {**cfg, "ideal": False}})
        m = AnalogQwen(b.digital, chip, b.calib)
        r = bits_per_byte(m, b.data)["mean"]
        kind = "DeltaNet" if m.layers[i]["linear"] else "attention"
        row = {"profile": prof, "layer": i, "kind": kind, "dbpb": r - ideal, "rel": r / ideal - 1}
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{prof:9s} layer {i:2d} {kind:9s} loss {100 * row['rel']:+7.2f}%", flush=True)
