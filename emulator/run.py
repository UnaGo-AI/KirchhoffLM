"""Analog-hardware simulation of Qwen3.5-0.8B: calibration, validation, evaluation, design search.

Run on the M3 Max server only (MLX, Metal GPU):
  python run.py calibrate            ideal pass on calibration texts -> calib.json (signal ranges)
  python run.py validate             ideal analog model == digital mlx-lm model (logits, bits/byte)
  python run.py eval --chip '{...}'  one design point
  python run.py sweep                one knob at a time around the reference chip
  python run.py search --n 200       random + evolutionary search for the quality/energy front
Results append to runs/*.jsonl (resumable: finished configs are skipped).
"""

import argparse
import hashlib
import json
import math
import os
import random
import time

import mlx.core as mx

from hw import DEVICES, MEMORIES, Chip
from sim import AnalogQwen, Recorder, load_digital, t_token, token_nll

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "runs")
TEXTS = os.path.join(ROOT, "..", "eval_texts.json") if not os.path.exists(os.path.join(ROOT, "eval_texts.json")) else os.path.join(ROOT, "eval_texts.json")
CALIB = os.path.join(ROOT, "calib.json")
os.makedirs(RUNS, exist_ok=True)


def texts():
    return json.load(open(TEXTS))


def encode_set(tok, n, max_tokens, offset=0):
    """First n wiki + n code texts (after offset), truncated to max_tokens; with byte counts."""
    out = []
    t = texts()
    for dom in ("wikipedia", "code"):
        for s in t[dom][offset:offset + n]:
            ids = tok.encode(s)[:max_tokens]
            nbytes = len(tok.decode(ids).encode("utf-8")) - len(tok.decode(ids[:1]).encode("utf-8"))
            out.append((dom, ids, nbytes))
    return out


def bits_per_byte(model, data):
    agg = {}
    for dom, ids, nbytes in data:
        nll = token_nll(model, ids)
        a = agg.setdefault(dom, [0.0, 0])
        a[0] += nll
        a[1] += nbytes
    res = {d: v[0] / math.log(2) / v[1] for d, v in agg.items()}
    res["mean"] = sum(res[d] for d in ("wikipedia", "code")) / 2
    return res


def chip_hash(c: Chip):
    return hashlib.sha1(json.dumps(c.to_dict(), sort_keys=True).encode()).hexdigest()[:12]


def done_hashes(path):
    if not os.path.exists(path):
        return {}
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[r["hash"]] = r
    return out


class Bench:
    def __init__(self, n, max_tokens):
        self.digital, self.tok = load_digital()
        self.digital.set_dtype(mx.float32)
        mx.eval(self.digital.parameters())
        self.calib = json.load(open(CALIB)) if os.path.exists(CALIB) else None
        self.data = encode_set(self.tok, n, max_tokens)
        self.ideal = None

    def ideal_bpb(self):
        if self.ideal is None:
            self.ideal = bits_per_byte(AnalogQwen(self.digital, Chip(ideal=True), self.calib), self.data)
        return self.ideal

    def evaluate(self, chip: Chip, ctx=2048):
        t0 = time.time()
        model = AnalogQwen(self.digital, chip, self.calib)
        t_build = time.time() - t0
        bpb = bits_per_byte(model, self.data)
        ideal = self.ideal_bpb()
        e = model.energy_per_token(ctx)
        return {"hash": chip_hash(chip), "chip": chip.to_dict(), "bpb": bpb,
                "dbpb": bpb["mean"] - ideal["mean"], "rel": bpb["mean"] / ideal["mean"] - 1,
                "energy_j": e["total"], "energy": e, "t_token_s": t_token(model),
                "tokens_per_s": 1.0 / t_token(model),
                "sim_s": round(time.time() - t0, 1), "build_s": round(t_build, 1)}


def cmd_calibrate(a):
    digital, tok = load_digital()
    digital.set_dtype(mx.float32)
    model = AnalogQwen(digital, Chip(ideal=True), None)
    rec = Recorder()
    # calibration texts are the LAST texts of each domain, disjoint from the evaluation subset
    data = encode_set(tok, 3, 512, offset=37)
    for _, ids, _ in data:
        model(mx.array(ids)[None], rec=rec)
    stats = rec.stats()
    json.dump(stats, open(CALIB, "w"), indent=0)
    print(f"calibrated {len(stats)} nodes -> {CALIB}")
    for k in ("L0.norm1", "L0.mlp_h", "L3.attn_y", "L10.state", "final_norm", "logits"):
        print(k, stats[k])


def cmd_validate(a):
    digital, tok = load_digital()
    digital.set_dtype(mx.float32)
    ana = AnalogQwen(digital, Chip(ideal=True), None)
    data = encode_set(tok, 1, 512)
    for dom, ids, _ in data:
        x = mx.array(ids)[None]
        ld, la = digital(x), ana(x)
        diff = float(mx.abs(ld - la).max())
        print(f"{dom}: max |logit diff| ideal-analog vs digital = {diff:.2e} (logit scale {float(mx.abs(ld).max()):.1f})")
    b = Bench(a.n, a.max_tokens)
    dig = bits_per_byte(digital, b.data)
    ide = b.ideal_bpb()
    print("bits/byte digital:", {k: round(v, 4) for k, v in dig.items()})
    print("bits/byte ideal  :", {k: round(v, 4) for k, v in ide.items()})


def cmd_eval(a):
    b = Bench(a.n, a.max_tokens)
    chip = Chip(**json.loads(a.chip)).validate()
    r = b.evaluate(chip)
    print(json.dumps({k: r[k] for k in ("bpb", "dbpb", "rel", "energy_j", "tokens_per_s", "sim_s")}, indent=1))
    print(json.dumps(r["energy"], indent=1))


REFERENCE = dict(device="rram", g_max=25e-6, v_read=0.2, t_int=50e-9, tile=256, p_tia=10e-6,
                 memory="sram", c_mem=10e-15, v_mem=1.0, adc_bits=8, ckpt_every=0, fs_pct=99.99, offset_trim=0.1)

SWEEPS = {
    "ideal_baseline": [dict()],
    "device": [dict(device="resistor", g_max=25e-6, v_read=0.5), dict(device="rram"), dict(device="pcm", v_read=0.2),
               dict(device="fefet", g_max=10e-6, v_read=0.1), dict(device="flash", g_max=2e-6, v_read=0.1)],
    "v_read": [dict(v_read=v) for v in (0.02, 0.05, 0.1, 0.2, 0.3)],
    "g_max": [dict(g_max=g) for g in (5e-6, 10e-6, 25e-6, 50e-6, 100e-6)],
    "t_int": [dict(t_int=t) for t in (10e-9, 25e-9, 50e-9, 200e-9, 1e-6)],
    "tile": [dict(tile=t) for t in (64, 128, 256, 512)],
    "adc_bits": [dict(adc_bits=b) for b in (4, 5, 6, 8, 10)],
    "ckpt_every": [dict(ckpt_every=k) for k in (0, 1, 4, 12)],
    "fs_pct": [dict(fs_pct=p) for p in (99.9, 99.99, 100.0)],
    "offset_trim": [dict(offset_trim=t) for t in (1.0, 0.3, 0.1, 0.03, 0.01)],
    "memory": [dict(memory=m) for m in MEMORIES] + [dict(memory="igzo", c_mem=1e-12, v_mem=3.3), dict(memory="fecap", c_mem=1e-12, v_mem=3.3)],
    "c_mem": [dict(c_mem=c) for c in (1e-15, 10e-15, 100e-15, 1e-12)],
    "v_mem": [dict(v_mem=v) for v in (0.5, 1.0, 2.0, 3.3)],
    "temp": [dict(temp=t) for t in (273.0, 300.0, 330.0, 360.0)],
    "drift_time": [dict(drift_time=t) for t in (1.0, 3600.0, 86400.0, 2.6e6, 3.2e7)],
    "seed": [dict(seed=s) for s in (0, 1, 2)],
}
# ablation: everything on except one effect, then each effect alone
from hw import EFFECTS
SWEEPS["ablate_one_off"] = [dict(off=e) for e in EFFECTS]
SWEEPS["only_one_on"] = [dict(off=",".join(x for x in EFFECTS if x != e)) for e in EFFECTS]


def cmd_sweep(a):
    b = Bench(a.n, a.max_tokens)
    path = os.path.join(RUNS, "sweep.jsonl")
    done = done_hashes(path)
    print("ideal bits/byte:", {k: round(v, 4) for k, v in b.ideal_bpb().items()}, flush=True)
    groups = a.only.split(",") if a.only else list(SWEEPS)
    for g in groups:
        for over in SWEEPS[g]:
            chip = Chip(**{**REFERENCE, **over})
            try:
                chip.validate()
            except ValueError as e:
                print(f"[{g}] skip {over}: {e}")
                continue
            h = chip_hash(chip)
            if h in done:
                r = done[h]
            else:
                r = b.evaluate(chip)
                r["group"] = g
                with open(path, "a") as f:
                    f.write(json.dumps(r) + "\n")
            label = over if g != "only_one_on" else {"only": [e for e in EFFECTS if e not in over["off"].split(",")][0]}
            print(f"[{g}] {json.dumps(label):48s} dbpb {r['dbpb']:+.4f} ({100 * r['rel']:+.2f}%)  "
                  f"E {r['energy_j'] * 1e6:8.2f} uJ/tok  {r['tokens_per_s']:,.0f} tok/s  ({r['sim_s']}s)", flush=True)


SPACE = {
    "device": list(DEVICES),
    "memory": list(MEMORIES),
    "tile": [64, 128, 256, 512],
    "adc_bits": [4, 5, 6, 7, 8, 10],
    "ckpt_every": [0, 1, 2, 3, 4, 6, 12],
    "fs_pct": [99.9, 99.99, 100.0],
    "offset_trim": [1.0, 0.3, 0.1, 0.03, 0.01],
}


def sample(rng: random.Random, base=None, n_mut=None):
    def logu(lo, hi):
        return math.exp(rng.uniform(math.log(lo), math.log(hi)))
    knobs = ["device", "g_max", "v_read", "t_int", "tile", "p_tia", "memory", "c_mem", "adc_bits", "ckpt_every", "fs_pct", "offset_trim", "v_mem"]
    c = dict(base) if base else {}
    change = knobs if base is None else rng.sample(knobs, n_mut or rng.choice([1, 1, 2, 3]))
    for k in change:
        if k in SPACE:
            c[k] = rng.choice(SPACE[k])
        elif k == "g_max":
            c[k] = None                                   # resolved below against the device range
        elif k == "v_read":
            c[k] = None
        elif k == "t_int":
            c[k] = logu(10e-9, 1e-6)
        elif k == "p_tia":
            c[k] = logu(1e-6, 100e-6)
        elif k == "c_mem":
            c[k] = logu(1e-15, 1e-12)
        elif k == "v_mem":
            c[k] = rng.uniform(0.5, 3.3)
    d = DEVICES[c["device"]]
    lo, hi = d["g_max"]
    if c.get("g_max") is None or not lo <= c["g_max"] <= hi:
        c["g_max"] = logu(lo, hi)
    if c.get("v_read") is None or c["v_read"] > d["v_max"]:
        c["v_read"] = rng.uniform(0.02, d["v_max"])
    return c


def pareto(rows, keys=("dbpb", "energy_j")):
    front = []
    for r in rows:
        if not any(all(o[k] <= r[k] for k in keys) and any(o[k] < r[k] for k in keys) for o in rows):
            front.append(r)
    return sorted(front, key=lambda r: r["energy_j"])


def cmd_search(a):
    b = Bench(a.n, a.max_tokens)
    path = os.path.join(RUNS, "search.jsonl")
    done = done_hashes(path)
    rows = list(done.values())
    rng = random.Random(a.seed + len(rows))
    print("ideal bits/byte:", {k: round(v, 4) for k, v in b.ideal_bpb().items()}, f"| {len(rows)} done", flush=True)
    while len(rows) < a.n_configs:
        explore = len(rows) < a.n_random
        if explore:
            knobs = sample(rng)
        else:
            parent = rng.choice(pareto(rows))
            knobs = sample(rng, {k: parent["chip"][k] for k in ("device", "g_max", "v_read", "t_int", "tile", "p_tia",
                                                                 "memory", "c_mem", "adc_bits", "ckpt_every", "fs_pct", "offset_trim", "v_mem")})
        chip = Chip(**knobs).validate()
        h = chip_hash(chip)
        if h in done:
            continue
        r = b.evaluate(chip)
        r["phase"] = "random" if explore else "evolve"
        with open(path, "a") as f:
            f.write(json.dumps(r) + "\n")
        done[h] = r
        rows.append(r)
        print(f"#{len(rows):3d} {r['phase']:6s} {knobs['device']:8s} dbpb {r['dbpb']:+.4f} ({100 * r['rel']:+.2f}%) "
              f"E {r['energy_j'] * 1e6:8.2f} uJ/tok {r['tokens_per_s']:>9,.0f} tok/s ({r['sim_s']}s)", flush=True)
    print("\nPareto front (quality loss vs energy):")
    for r in pareto(rows):
        c = r["chip"]
        print(f"  dbpb {r['dbpb']:+.4f} ({100 * r['rel']:+.2f}%)  E {r['energy_j'] * 1e6:7.2f} uJ  {r['tokens_per_s']:>9,.0f} tok/s  "
              f"{c['device']} g={c['g_max']:.1e} v={c['v_read']:.2f} t={c['t_int']:.0e} tile={c['tile']} "
              f"tia={c['p_tia']:.0e} {c['memory']} C={c['c_mem']:.0e} Vm={c['v_mem']:.1f} adc={c['adc_bits']} ck={c['ckpt_every']} fs={c['fs_pct']} trim={c['offset_trim']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["calibrate", "validate", "eval", "sweep", "search"])
    p.add_argument("--n", type=int, default=6, help="texts per domain")
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--chip", default="{}")
    p.add_argument("--only", default="")
    p.add_argument("--n_configs", type=int, default=200)
    p.add_argument("--n_random", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    {"calibrate": cmd_calibrate, "validate": cmd_validate, "eval": cmd_eval, "sweep": cmd_sweep, "search": cmd_search}[a.cmd](a)


if __name__ == "__main__":
    main()
