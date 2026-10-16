"""Translate trained Qwen3.5-0.8B into a hardware design package (one chip per layer, AC capacitor grids).

Output directory (default ~/analog/hw_qwen35):
  manifest.json          chip settings, physical units, per-chip component counts, area, rails, provenance
  chip_LXX.safetensors   one file per layer chip:
                           <grid>.cap_pos / <grid>.cap_neg   uint8 unit-capacitor counts per cell (0..255)
                           <grid>.r_feedback                 float32 ohm, read-out feedback resistor per column
                           <grid>.gain                       float32 column gain (weight units per full cell)
                           <param>                           norm-gain resistor ratios, conv tap ratios,
                                                             digital memory constants (A_log, dt_bias)
  chip_head.safetensors  the shared embedding / output grid + final norm
  tile_example.cir       SPICE netlist of one real 128 x 128 tile (layer 0 FFN gate, rows 0-127, inputs 0-127)
  config.json, tokenizer files (architecture and text I/O only; no weights)

A weight is recovered as w = (cap_pos - cap_neg) / 255 * gain. The emulator (emulate_hw.py) reads only
this package.
"""
import argparse, json, math, os, shutil

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from hw import Chip
from layer_bench import DESIGNS
from sim import MODEL_ID, load_digital, fs

LEVELS = 255


def grid_codes(w):
    """Row-scaled 8-bit unit-capacitor counts for a differential pair, and the column gain."""
    w = w.astype(mx.float32)
    gain = mx.maximum(mx.abs(w).max(axis=1), 1e-12)
    u = w / gain[:, None]
    pos = mx.round(mx.maximum(u, 0) * LEVELS).astype(mx.uint8)
    neg = mx.round(mx.maximum(-u, 0) * LEVELS).astype(mx.uint8)
    return pos, neg, gain


def spice_tile(path, pos, neg, chip, title):
    """Netlist of one tile: sine-driven rows, wire segments, cell capacitors, stray capacitors,
    columns into virtual grounds (0 V sources that measure the column current)."""
    n_out, n_in = pos.shape
    c_unit = chip.c_cell / LEVELS
    lines = [f"* {title}", f"* AC capacitor grid: {n_in} inputs x {n_out} differential outputs, "
             f"C_unit = {c_unit * 1e18:.2f} aF, carrier {chip.ac_freq / 1e6:g} MHz",
             ".param vin=0.5", f".param f={chip.ac_freq:g}"]
    ncol = 2 * n_out
    for i in range(n_in):
        lines.append(f"VIN{i} drv{i} 0 SIN(0 {{vin}} {{f}})")
        lines.append(f"RDRV{i} drv{i} r{i}_0 {chip.r_drv + chip.r_wire:g}")
        for c in range(ncol - 1):
            lines.append(f"RR{i}_{c} r{i}_{c} r{i}_{c + 1} {chip.r_wire:g}")
    for c in range(ncol):
        for i in range(n_in - 1):
            lines.append(f"RC{i}_{c} k{i}_{c} k{i + 1}_{c} {chip.r_wire:g}")
        lines.append(f"RCE{c} k{n_in - 1}_{c} vg{c} {chip.r_wire:g}")
        lines.append(f"VSENSE{c} vg{c} 0 0")                  # virtual ground: current = column output
    for o in range(n_out):
        for i in range(n_in):
            for c, code in ((2 * o, int(pos[o, i])), (2 * o + 1, int(neg[o, i]))):
                if code:
                    lines.append(f"C{i}_{c} r{i}_{c} k{i}_{c} {code * c_unit:.4e}")
                lines.append(f"CP{i}_{c} r{i}_{c} k{i}_{c} {chip.c_par:.3e}")
    lines += [f".tran {1 / chip.ac_freq / 50:.3e} {chip.t_int:.3e}", ".end"]
    open(path, "w").write("\n".join(lines) + "\n")
    return len(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.environ.get("KLM_PKG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "hw_qwen35")))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    digital, tok = load_digital()
    digital.set_dtype(mx.float32)
    calib = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib.json")))
    design = {k: v for k, v in DESIGNS["AC"].items()}
    design["link_bits"] = 10
    chip = Chip(**design).validate()
    y_max = chip.g_eff_max()
    params = dict(tree_flatten(digital.parameters()))
    linear_names = {name for name, mod in digital.named_modules() if isinstance(mod, nn.Linear)}
    lm = digital.language_model.model
    n_layers = len(lm.layers)
    manifest = {"source_model": os.path.basename(MODEL_ID), "design": chip.to_dict(), "levels": LEVELS,
                "units": {"c_unit_F": chip.c_cell / LEVELS, "carrier_Hz": chip.ac_freq, "drive_V": chip.v_read,
                          "admittance_per_full_cell_S": y_max, "rail_V": 1.0},
                "weight_formula": "w = (cap_pos - cap_neg) / 255 * gain;  r_feedback = gain * x_fs / (v_read * Y_max) * rail_V / y_fs",
                "chips": {}}
    groups = {f"L{i:02d}": f"language_model.model.layers.{i}." for i in range(n_layers)}
    groups["head"] = None
    for chip_name, prefix in groups.items():
        out, counts = {}, {"capacitor_cells": 0, "unit_capacitors": 0, "amplifiers": 0, "weights": 0}
        if prefix is None:
            names = [k for k in params if k.startswith("language_model.model.embed_tokens") or k == "language_model.model.norm.weight"]
        else:
            names = [k for k in params if k.startswith(prefix)]
        for k in names:
            v = params[k]
            mod = k.rsplit(".", 1)[0]
            if (mod in linear_names or "embed_tokens" in k) and v.ndim == 2:
                pos, neg, gain = grid_codes(v)
                nodes = {"mlp.gate_proj": ("norm2", "mlp_gate"), "mlp.up_proj": ("norm2", "mlp_up"),
                         "mlp.down_proj": ("mlp_h", "mlp_out"), "linear_attn.out_proj": ("gdn_y", "mix_out"),
                         "self_attn.o_proj": ("attn_y", "mix_out"), "in_proj_qkv": ("norm1", "qkv"),
                         "in_proj_z": ("norm1", "z"), "in_proj_b": ("norm1", "b"), "in_proj_a": ("norm1", "a"),
                         "q_proj": ("norm1", "q"), "k_proj": ("norm1", "k"), "v_proj": ("norm1", "v")}
                if prefix:
                    short = mod[len(prefix):]
                    i_node, o_node = next(n for s_, n in nodes.items() if short.endswith(s_))
                    lp = f"L{int(chip_name[1:])}."
                    x_fs, y_fs = fs(calib, lp + i_node, chip.fs_pct), fs(calib, lp + o_node, chip.fs_pct)
                else:
                    x_fs, y_fs = fs(calib, "final_norm", chip.fs_pct), fs(calib, "logits", chip.fs_pct)
                out[k + ".cap_pos"], out[k + ".cap_neg"] = pos, neg
                out[k + ".gain"] = gain
                # output voltage = y * (rail / y_fs); column current = v_read * Y_max * sum(xn * u)
                out[k + ".r_feedback"] = gain * x_fs / (chip.v_read * y_max) * 1.0 / y_fs
                mx.eval(pos, neg)
                nz = int((pos > 0).sum()) + int((neg > 0).sum())
                counts["capacitor_cells"] += nz
                counts["unit_capacitors"] += int(pos.astype(mx.uint32).sum()) + int(neg.astype(mx.uint32).sum())
                counts["amplifiers"] += v.shape[0]
                counts["weights"] += v.size
            else:
                out[k] = v                                   # resistor ratios / digital constants
        mx.save_safetensors(os.path.join(a.out, f"chip_{chip_name}.safetensors"), out)
        cells = 2 * counts["weights"]
        counts["area_mm2"] = round((cells * chip.cell_area_um2() + 50.0 * counts["amplifiers"]) * 1e-6, 1)
        counts["cell_fill"] = round(counts["capacitor_cells"] / max(cells, 1), 3)
        manifest["chips"][chip_name] = counts
        print(f"chip {chip_name}: {counts['weights']:>11,} weights  {counts['capacitor_cells']:>11,} capacitors  "
              f"{counts['unit_capacitors']:>14,} unit caps  {counts['amplifiers']:>7,} amplifiers  {counts['area_mm2']:8.1f} mm2", flush=True)
    # one real tile as a SPICE netlist
    pos, neg, _ = grid_codes(lm.layers[0].mlp.gate_proj.weight)
    n_lines = spice_tile(os.path.join(a.out, "tile_example.cir"), np.array(pos[:128, :128]), np.array(neg[:128, :128]),
                         chip, "Qwen3.5 layer 0 FFN gate_proj, outputs 0-127, inputs 0-127")
    manifest["spice_example"] = {"file": "tile_example.cir", "lines": n_lines}
    tot = {k: sum(c[k] for c in manifest["chips"].values()) for k in ("weights", "capacitor_cells", "unit_capacitors", "amplifiers", "area_mm2")}
    manifest["totals"] = tot
    from run import bits_per_byte, encode_set
    from sim import AnalogQwen
    ref = bits_per_byte(AnalogQwen(digital, Chip(ideal=True), calib), encode_set(tok, 6, 1024))
    manifest["reference_bits_per_byte"] = ref                      # original digital model, same eval texts
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    src = MODEL_ID if os.path.isdir(MODEL_ID) else __import__("huggingface_hub").snapshot_download(MODEL_ID)
    for f in os.listdir(src):
        if f == "config.json" or f.startswith("tokenizer") or f in ("vocab.json", "merges.txt", "chat_template.jinja"):
            shutil.copy(os.path.join(src, f), a.out)
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib.json"), a.out)
    print(f"totals: {tot['weights']:,} weights, {tot['capacitor_cells']:,} capacitors, {tot['unit_capacitors']:,} unit caps, "
          f"{tot['amplifiers']:,} amplifiers, {tot['area_mm2']:,.0f} mm2; SPICE tile {n_lines:,} lines")


if __name__ == "__main__":
    main()
