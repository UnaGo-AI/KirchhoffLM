"""Fair AC vs DC comparison of the weight grids. Both sides get their wire effect (DC: data-dependent
IR drop; AC: RC phase delay of the row wires), their compensation (DC: pre-distorted programming;
AC: per-column reference phase), their noise (DC: Johnson + shot; AC: noiseless capacitors, carrier
phase noise), their offsets (DC: auto-zero trim; AC: rejected by synchronous detection) and their
costs (DC: resistive dissipation + trimmed amplifiers; AC: drive C V^2 + modulators + demodulators +
carrier oscillator). Everything else is the best design. Results -> runs/ac_dc.jsonl."""
import json, os
import hw
from hw import Chip
from run import Bench, bits_per_byte, RUNS
from sim import AnalogQwen, t_token

b = Bench(6, 1024)
ideal = b.ideal_bpb()["mean"]
rows = [json.loads(l) for l in open(os.path.join(RUNS, "search.jsonl"))]
best = dict(min(rows, key=lambda r: r["dbpb"])["chip"])
hw.DEVICES["resistor"].update(levels=256, prog_rel=0.01)            # realistic: 8-bit steps, 1%
path = os.path.join(RUNS, "ac_dc.jsonl")
done = {r["label"] for r in map(json.loads, open(path))} if os.path.exists(path) else set()

def run(kind, label, over):
    if label in done:
        return
    losses = []
    for seed in (0, 1):
        chip = Chip(**{**best, **over, "seed": seed}).validate()
        m = AnalogQwen(b.digital, chip, b.calib)
        losses.append(100 * (bits_per_byte(m, b.data)["mean"] / ideal - 1))
    e, a = m.energy_per_token(2048), m.area_mm2()
    phi = max(x.phi_max for x in m.crossbars())
    row = {"kind": kind, "label": label, "over": over, "loss_pct": sum(losses) / 2, "loss_seeds": losses,
           "energy": e, "area": a, "phi_max": phi, "tok_s": 1 / t_token(m)}
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"{kind} {label:46s} loss {row['loss_pct']:+7.2f}%  grid+amps {e['arrays+amps'] * 1e6:5.2f} uJ  carrier {e['carrier'] * 1e6:5.2f} uJ  "
          f"total {e['total'] * 1e6:5.2f} uJ  area/layer {a['layer_max']:7.1f} mm2  total {a['total']:8.0f} mm2  "
          f"phi_max {phi:.2f} rad  {row['tok_s']:>7,.0f} tok/s", flush=True)

for r in (0.5, 2.0, 4.0):
    for tile in (128, 256):
        for comp in (0.0, 0.9):
            run("DC", f"resistor 8b/1%  R={r} tile={tile} comp={comp}", dict(device="resistor", r_wire=r, tile=tile, ir_comp=comp))
run("DC", "rram 25uS v0.2  R=2 tile=256 comp=0.9", dict(device="rram", g_max=25e-6, v_read=0.2, r_wire=2.0, tile=256, ir_comp=0.9))
AC = dict(device="cap_ac", ac_freq=100e6, c_cell=10e-15, v_read=0.5, t_int=100e-9)
for r in (0.5, 2.0, 4.0):
    for tile in (128, 256):
        for comp in (0.0, 0.9):
            run("AC", f"cap 10fF 100MHz  R={r} tile={tile} comp={comp}", dict(AC, r_wire=r, tile=tile, phase_comp=comp))
for tile in (128, 256):
    run("AC", f"cap 2fF 1GHz  R=2 tile={tile} comp=0.9", dict(AC, ac_freq=1e9, c_cell=2e-15, r_wire=2.0, tile=tile, phase_comp=0.9))
run("AC", "cap 50fF 20MHz t200ns  R=2 tile=256 comp=0.9", dict(AC, ac_freq=20e6, c_cell=50e-15, t_int=200e-9, r_wire=2.0, tile=256, phase_comp=0.9))
run("AC", "cap 10fF 100MHz LC Q=10  R=2 tile=128 comp=0.9", dict(AC, r_wire=2.0, tile=128, phase_comp=0.9, q_res=10.0))
