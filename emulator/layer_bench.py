"""Per-layer performance of one-chip-per-layer designs. For every layer chip:
  fidelity  - error of every block output inside the layer vs the exact layer (same input)
  quality   - bits/byte loss of the whole model when only this layer runs on the chip
  cost      - energy per token, latency, area of that layer chip
Designs: AC capacitor grid (10 fF, 100 MHz, 0.5 V, tile 128, 2 ohm wires, phase calibration) and the
best DC resistor grid (8-bit / 1 % resistors, tile 128, 0.5 ohm wires, 90 % IR compensation).
Results -> runs/layer_bench.jsonl, whole-chain results at the end."""
import json, os
import numpy as np
import mlx.core as mx
import hw
from hw import Chip
from run import Bench, bits_per_byte, RUNS
from sim import AnalogQwen, t_token

hw.DEVICES["resistor"].update(levels=256, prog_rel=0.01)
rows = [json.loads(l) for l in open(os.path.join(RUNS, "search.jsonl"))]
best = {k: v for k, v in min(rows, key=lambda r: r["dbpb"])["chip"].items() if k != "layer_over"}
DESIGNS = {
    "AC": {**best, "device": "cap_ac", "ac_freq": 100e6, "c_cell": 10e-15, "v_read": 0.5, "t_int": 100e-9,
           "tile": 128, "r_wire": 2.0, "phase_comp": 0.9, "link_bits": 8},
    "DC": {**best, "device": "resistor", "tile": 128, "r_wire": 0.5, "ir_comp": 0.9, "link_bits": 8},
}
GDN_NODES = ["qkv", "conv_pre", "gdn_core", "gdn_y", "mix_out", "mlp_gate", "mlp_h", "mlp_out"]
ATT_NODES = ["q", "k", "v", "attn_o", "attn_y", "mix_out", "mlp_gate", "mlp_h", "mlp_out"]


def main():
    b = Bench(6, 1024)
    ideal = b.ideal_bpb()["mean"]
    class Capture:
        def __init__(self, names):
            self.names, self.data = set(names), {}

        def __call__(self, name, x):
            if name in self.names:
                self.data[name] = np.array(x.astype(mx.float32))
            return x


    _, ids, _ = b.data[0]
    x = mx.array(ids)[None]
    ideal_model = AnalogQwen(b.digital, Chip(ideal=True), b.calib)
    n = ideal_model.n_layers
    all_names = ["embed"] + [f"L{i}.{k}" for i in range(n) for k in set(GDN_NODES + ATT_NODES + ["resid"])]
    cap_ideal = Capture(all_names)
    ideal_model(x, rec=cap_ideal)
    rel = lambda a, r: float(np.linalg.norm(a - r) / max(np.linalg.norm(r), 1e-12))

    path = os.path.join(RUNS, "layer_bench.jsonl")
    done = {(r["design"], r["layer"]) for r in map(json.loads, open(path))} if os.path.exists(path) else set()
    for name, cfg in DESIGNS.items():
        full = AnalogQwen(b.digital, Chip(**cfg), b.calib)                       # for per-layer cost
        for i in range(n):
            if (name, i) in done:
                continue
            chip = Chip(ideal=True, layer_over={str(i): {**cfg, "ideal": False}})
            m = AnalogQwen(b.digital, chip, b.calib)
            p = f"L{i}."
            nodes = GDN_NODES if m.layers[i]["linear"] else ATT_NODES
            cap = Capture([p + k for k in nodes + ["resid"]])
            m(x, rec=cap)
            prev = cap_ideal.data["embed" if i == 0 else f"L{i - 1}.resid"]
            fid = {k: rel(cap.data[p + k], cap_ideal.data[p + k]) for k in nodes}
            fid["layer"] = rel(cap.data[p + "resid"] - prev, cap_ideal.data[p + "resid"] - prev)
            q = 100 * (bits_per_byte(m, b.data)["mean"] / ideal - 1)
            e = full.energy_per_token(2048, only_layer=i)
            c = full.layers[i]["chip"]
            lat = c.stages_per_layer * c.t_stage() + (20e-9 + 8 * 1024 / (64 * 10e9) if i + 1 < n else 0.0)
            area = sum(v.area_mm2() for v in full.layers[i].values() if hasattr(v, "area_mm2"))
            row = {"design": name, "layer": i, "kind": "DeltaNet" if m.layers[i]["linear"] else "attention",
                   "fidelity": fid, "loss_pct": q, "energy_uj": e["total"] * 1e6, "energy": e, "latency_ns": lat * 1e9,
                   "area_mm2": area}
            with open(path, "a") as f:
                f.write(json.dumps(row) + "\n")
            worst = max((k for k in nodes), key=lambda k: fid[k])
            print(f"{name} layer {i:2d} {row['kind']:9s} layer-output error {100 * fid['layer']:6.2f}%  worst block {worst} "
                  f"{100 * fid[worst]:6.2f}%  model loss {q:+6.2f}%  E {row['energy_uj']:6.3f} uJ  {lat * 1e9:5.0f} ns  {area:6.1f} mm2", flush=True)
        losses = []
        for seed in (0, 1):
            mm = AnalogQwen(b.digital, Chip(**{**cfg, "seed": seed}), b.calib)
            losses.append(100 * (bits_per_byte(mm, b.data)["mean"] / ideal - 1))
        e = full.energy_per_token(2048)
        print(f"== {name} whole chain: loss {np.mean(losses):+.2f}% (seeds {losses[0]:+.2f}/{losses[1]:+.2f})  "
              f"E {e['total'] * 1e6:.2f} uJ/tok  {1 / t_token(full):,.0f} tok/s  area {full.area_mm2()['total']:.0f} mm2", flush=True)


if __name__ == "__main__":
    main()
