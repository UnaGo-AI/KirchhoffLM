"""Weights as realistic resistors (steps, tolerance, IR compensation) and as an AC capacitive grid
(Y = j omega C) with optional resonant LC drive. Everything else is the best design."""
import json, os
import hw
from hw import Chip
from run import Bench, bits_per_byte, RUNS
from sim import AnalogQwen, t_token

b = Bench(6, 1024)
ideal = b.ideal_bpb()["mean"]
rows = [json.loads(l) for l in open(os.path.join(RUNS, "search.jsonl"))]
best = dict(min(rows, key=lambda r: r["dbpb"])["chip"])
RES0 = dict(hw.DEVICES["resistor"])
CAP0 = dict(hw.DEVICES["cap_ac"])

def run(label, over=None, res=None, cap=None):
    hw.DEVICES["resistor"] = {**RES0, **(res or {})}
    hw.DEVICES["cap_ac"] = {**CAP0, **(cap or {})}
    losses = []
    for seed in (0, 1):
        chip = Chip(**{**best, **(over or {}), "seed": seed}).validate()
        m = AnalogQwen(b.digital, chip, b.calib)
        losses.append(100 * (bits_per_byte(m, b.data)["mean"] / ideal - 1))
    e = m.energy_per_token(2048)
    extra = f" | sigma_C {100 * chip.prog_rel():.2f}%  Y_max {chip.g_eff_max() * 1e6:.2f} uS" if chip.is_ac() else ""
    print(f"{label:52s} loss {sum(losses) / 2:+7.2f}%  arrays+amps {e['arrays+amps'] * 1e6:6.3f} uJ  total {e['total'] * 1e6:6.2f} uJ  "
          f"{1 / t_token(m):>8,.0f} tok/s{extra}", flush=True)

print("== resistors (DC) ==")
run("ideal resistors, continuous, 0.1% (current best)")
run("+ IR drop 90% pre-compensated", {"ir_comp": 0.9})
for bits in (8, 6, 4):
    run(f"{bits}-bit resistor steps, 0.1%, IR comp 90%", {"ir_comp": 0.9}, res={"levels": 2 ** bits})
for tol in (0.01, 0.03):
    run(f"continuous, {100 * tol:.0f}% tolerance, IR comp 90%", {"ir_comp": 0.9}, res={"prog_rel": tol})
run("8-bit steps, 1% tolerance, IR comp 90%", {"ir_comp": 0.9}, res={"levels": 256, "prog_rel": 0.01})

print("== AC capacitive grid (Y = j w C) ==")
base = {"device": "cap_ac", "ac_freq": 100e6, "t_int": 100e-9}
for c in (0.5e-15, 2e-15, 10e-15, 50e-15):
    run(f"100 MHz, C_cell {c * 1e15:g} fF, 8-bit unit caps", {**base, "c_cell": c})
run("100 MHz, 2 fF, drive 0.5 V", {**base, "c_cell": 2e-15, "v_read": 0.5})
run("100 MHz, 10 fF, drive 0.5 V", {**base, "c_cell": 10e-15, "v_read": 0.5})
run("100 MHz, 10 fF, 0.5 V, 6-bit unit caps", {**base, "c_cell": 10e-15, "v_read": 0.5}, cap={"levels": 64})
run("1 GHz, 2 fF, 0.5 V", {**base, "ac_freq": 1e9, "c_cell": 2e-15, "v_read": 0.5})
run("10 MHz, 10 fF, 0.5 V (t_int 500 ns)", {**base, "ac_freq": 10e6, "t_int": 500e-9, "c_cell": 10e-15, "v_read": 0.5})
run("100 MHz, 10 fF, 0.5 V, resonant LC drive Q=10", {**base, "c_cell": 10e-15, "v_read": 0.5, "q_res": 10.0})
run("100 MHz, 10 fF, 0.5 V, phase noise 1e-2 rad", {**base, "c_cell": 10e-15, "v_read": 0.5}, cap={"read_rel": 0.01})
