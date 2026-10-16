"""Exact circuit solve of one crossbar tile (linear nodal analysis, like SPICE for linear circuits).

Network of one tile with n inputs and n differential outputs (2n columns):
  - every row wire is a ladder: driver (R_drv) -> R_seg -> crossing node -> R_seg -> ... (2n crossings)
  - every column wire is a ladder of R_seg ending in a virtual ground (the read-out amplifier input)
  - a cell sits between the row node and the column node at every crossing:
      DC: conductance G = |u| * G_max (resistor)       AC: admittance j*omega*C, C = |u| * C_max
  - AC only: stray crossing capacitance C_par (with mismatch), coupling capacitance between
    neighbouring row wires and between neighbouring column wires at every segment.
DC solves G V = I; AC solves the complex phasor system at the carrier frequency. The AC read-out is
a synchronous detector: in-phase part relative to a reference phase that is either the ideal +90 deg
or calibrated per column (drive all rows equally once, measure each column's phase).

Inputs are real Qwen3.5 weights (layer 0 FFN gate_proj) and real activations (norm2 output on text).
Output: error of the exact circuit vs the ideal W x, and vs the behavioural model in hw.py.
"""
import json, math, sys, time

import mlx.core as mx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from hw import Chip
from run import Bench, encode_set
from sim import AnalogQwen, fs

R_DRV = 10.0          # ohm, row driver output resistance


class Capture:
    def __init__(self, names):
        self.names, self.data = set(names), {}

    def __call__(self, name, x):
        if name in self.names:
            self.data[name] = np.array(x.astype(mx.float32))
        return x


def solve_tile(u, V, r_seg, mode, y_max, freq=None, c_par=0.0, c_par_mis=0.0, c_couple=0.0, seed=0):
    """u: (n_out, n_in) normalised weights in [-1, 1]; V: (n_in, T) row voltages.
    Returns column currents (2 n_out, T) (complex for AC)."""
    n_out, n_in = u.shape
    ncol = 2 * n_out
    rng = np.random.default_rng(seed)
    # cell admittances, column 2o = positive, 2o+1 = negative
    cell = np.zeros((n_in, ncol))
    cell[:, 0::2] = np.maximum(u, 0).T * y_max
    cell[:, 1::2] = np.maximum(-u, 0).T * y_max
    ac = mode == "ac"
    dtype = complex if ac else float
    if ac:
        w = 2 * math.pi * freq
        ycell = 1j * cell + 1j * w * c_par * (1 + c_par_mis * rng.standard_normal(cell.shape))  # cell = omega*C
    else:
        ycell = cell
    NR = n_in * ncol
    r = lambda i, c: i * ncol + c
    k = lambda i, c: NR + i * ncol + c
    rows, cols, vals = [], [], []

    def stamp(a, b, y):   # admittance y between node arrays a and b (b = None -> ground)
        a = np.asarray(a).ravel()
        y = np.asarray(y, dtype=dtype)
        y = y.ravel() if y.size == a.size else np.broadcast_to(y, a.shape).ravel()
        rows.append(a); cols.append(a); vals.append(y)
        if b is not None:
            b = np.asarray(b).ravel()
            rows.extend([b, a, b]); cols.extend([b, b, a]); vals.extend([y, -y, -y])

    gw = 1.0 / max(r_seg, 1e-9)
    I, C = np.meshgrid(np.arange(n_in), np.arange(ncol), indexing="ij")
    stamp(r(np.arange(n_in), 0), None, 1.0 / (R_DRV + r_seg))                     # driver -> first node
    stamp(r(I[:, :-1], C[:, :-1]), r(I[:, 1:], C[:, 1:]), gw)                      # row segments
    stamp(r(I, C), k(I, C), ycell)                                                 # cells
    stamp(k(I[:-1], C[:-1]), k(I[1:], C[1:]), gw)                                  # column segments
    stamp(k(n_in - 1, np.arange(ncol)), None, gw)                                  # to virtual ground
    if ac and c_couple > 0:
        yc = 1j * w * c_couple
        stamp(r(I[:-1], C[:-1]), r(I[1:], C[1:]), yc)      # neighbouring row wires
        stamp(k(I[:, :-1], C[:, :-1]), k(I[:, 1:], C[:, 1:]), yc)  # neighbouring column wires
    A = sp.csc_matrix((np.concatenate(vals).astype(dtype), (np.concatenate(rows), np.concatenate(cols))), shape=(2 * NR, 2 * NR))
    lu = spla.splu(A)
    B = np.zeros((2 * NR, V.shape[1]), dtype=dtype)
    B[[r(i, 0) for i in range(n_in)], :] = V / (R_DRV + r_seg)
    X = lu.solve(B)
    i_col = X[[k(n_in - 1, c) for c in range(ncol)], :] * gw                      # current into read-out
    return i_col


def main():
    b = Bench(1, 256)
    best = dict(Chip().to_dict())
    m = AnalogQwen(b.digital, Chip(ideal=True), b.calib)
    cap = Capture({"L0.norm2"})
    _, ids, _ = b.data[0]
    m(mx.array(ids)[None], rec=cap)
    x = cap.data["L0.norm2"][0]                                    # (T, 1024) real activations
    W = np.array(b.digital.language_model.model.layers[0].mlp.gate_proj.weight.astype(mx.float32))
    w_max = np.abs(W).max(axis=1, keepdims=True)
    x_fs = fs(b.calib, "L0.norm2", 100.0)
    T = 64
    xs = x[:T]
    print(f"tile from layer 0 FFN gate_proj, {T} real tokens, x_fs {x_fs:.2f}", flush=True)
    configs = [
        ("DC resistor 73uS, 0.09 V", "dc", dict(y_max=73e-6, v=0.09)),
        ("AC cap 10fF @100MHz, 0.5 V", "ac", dict(y_max=2 * math.pi * 100e6 * 10e-15, v=0.5, freq=100e6)),
        ("AC cap 2fF @1GHz, 0.5 V", "ac", dict(y_max=2 * math.pi * 1e9 * 2e-15, v=0.5, freq=1e9)),
    ]
    out = []
    for n in (128, 256):
        u = W[:n, :n] / w_max[:n]
        V = (np.clip(xs[:, :n] / x_fs, -1, 1)).T                     # (n, T) normalised
        y_ideal = u @ V                                              # in units of v * y_max
        for label, mode, p in configs:
            for r_seg in (0.5, 2.0, 4.0):
                for couple in ((0.0, 0.1e-15) if mode == "ac" else (0.0,)):
                    t0 = time.time()
                    i_col = solve_tile(u, V * p["v"], r_seg, mode, p["y_max"], p.get("freq"),
                                             c_par=0.05e-15 if mode == "ac" else 0.0, c_par_mis=0.1, c_couple=couple)
                    pair = (i_col[0::2] - i_col[1::2]) / (p["v"] * p["y_max"])
                    if mode == "dc":
                        y_raw = pair.real
                        y_cal = y_raw
                        ph = 0.0
                    else:
                        y_raw = (pair * np.exp(-1j * math.pi / 2)).real    # ideal +90 deg reference
                        # per-column calibrated reference: once after programming, drive every row with
                        # the same voltage and measure each column's phase (with the real weights)
                        ical = solve_tile(u, np.ones((n, 1)) * p["v"], r_seg, mode, p["y_max"], p.get("freq"),
                                          c_par=0.05e-15, c_par_mis=0.1, c_couple=couple)
                        theta = np.angle(ical[:, 0])
                        ph = float(np.abs(theta - math.pi / 2).max())
                        rot = i_col * np.exp(-1j * theta)[:, None]
                        y_cal = (rot[0::2] - rot[1::2]).real / (p["v"] * p["y_max"])
                    err_raw = np.linalg.norm(y_raw - y_ideal) / np.linalg.norm(y_ideal)
                    err_cal = np.linalg.norm(y_cal - y_ideal) / np.linalg.norm(y_ideal)
                    # gain by output position (systematic attenuation along the row wire)
                    gain = (y_cal * y_ideal).sum(1) / np.maximum((y_ideal ** 2).sum(1), 1e-30)
                    row = {"tile": n, "config": label, "r_seg": r_seg, "couple_fF": couple * 1e15, "err_raw": err_raw,
                           "err_cal": err_cal, "gain_first": float(gain[:8].mean()), "gain_last": float(gain[-8:].mean()),
                           "phase_max_rad": ph, "solve_s": round(time.time() - t0, 1)}
                    out.append(row)
                    print(f"tile {n:3d} {label:28s} R={r_seg:3.1f} couple={couple * 1e15:.1f}fF  err {100 * err_raw:6.2f}% "
                          f"(calibrated {100 * err_cal:6.2f}%)  gain first/last outputs {row['gain_first']:.3f}/{row['gain_last']:.3f}  "
                          f"max phase lag {ph:.3f} rad  ({row['solve_s']}s)", flush=True)
    json.dump(out, open("runs/circuit_tile.json", "w"), indent=1)


if __name__ == "__main__":
    main()
