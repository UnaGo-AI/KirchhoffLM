"""A TinyStories language model trained directly in hardware parameters (MLX, M3 Max).

Every weight matrix is an AC capacitor grid: the trainable values are the cell capacitances
(C+ = relu(u) * C_max, C- = relu(-u) * C_max, 8-bit unit capacitors) and one feedback resistor per
read-out amplifier (column gain g). Norm gains are trainable resistor ratios. The forward pass runs
through the chip physics:
  capacitor mismatch (Pelgrom, depends on each cell's value), stray crossing capacitance mismatch,
  row-wire RC phase loss (with per-column phase calibration), amplifier gain/offset error and noise,
  rails (inputs and outputs clip), SiLU unit mismatch, norm-unit error, 10-bit ADC/DAC around the
  digital attention, 10-bit chip-to-chip links between layers, an 8-bit peak-referenced output ADC.
Rounding steps (unit capacitors, ADCs) pass gradients straight through.

Modes:
  digital  - ordinary training (exact matrices), then mapped onto the chip for evaluation
  chip     - trained on one fixed chip (its static errors), i.e. per-chip calibration from scratch
  robust   - a new random chip every step, so the model works on any chip

  python tiny_hw.py train --mode chip            (checkpoints every 500 steps, resumable)
  python tiny_hw.py eval                         (all trained modes on chips 0-4, plus samples)
"""
import argparse
import json
import math
import os
import time
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

DATA = os.environ.get("KLM_TINY_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "tinystories"))   # TinyStories: train.bin, val.bin, tokenizer.json
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "tiny_hw")
V, CTX, D, NL, NH, FF = 4096, 256, 256, 6, 8, 1024
K_B = 1.380649e-23


@dataclass
class HW:
    c_max: float = 10e-15       # F, largest cell capacitance
    freq: float = 100e6         # Hz, carrier
    v: float = 0.5              # V, drive amplitude
    t_int: float = 100e-9       # s, integration time per read
    tile: int = 128
    r_wire: float = 2.0         # ohm per cell segment
    phase_comp: float = 0.9
    levels: int = 255           # 8-bit unit-capacitor magnitude per cell of the pair
    pelgrom: float = 0.004      # sigma_C / C of a 1 fF capacitor
    c_par: float = 0.05e-15     # stray crossing capacitance
    c_par_mis: float = 0.1
    i_n: float = 3.2e-12        # A/sqrt(Hz), read-out amplifier input noise
    amp_gain_std: float = 0.005
    amp_off: float = 0.002 * 0.02   # offset (fraction of output rail) after synchronous detection
    fu_gain_std: float = 0.01
    fu_off: float = 0.005 * 0.01    # SiLU unit offset after trimming (fraction of its rail)
    norm_gain_std: float = 0.005
    norm_dyn_std: float = 0.002
    x_fs: float = 5.0           # input rail of a grid (after a norm unit)
    y_fs: float = 12.0          # output rail of a read-out amplifier
    resid_fs: float = 32.0      # range of the chip-to-chip link
    link_bits: int = 10
    adc_bits: int = 10          # around the digital attention
    out_bits: int = 8
    out_range: float = 32.0


hw = HW()
REC = None          # when a dict: CapLinear records the largest |input| and |output| it sees
FREE = False        # ablation: chip physics on, but no rails and no ADC rounding


def ste(x, q):
    """Forward q, backward identity."""
    return x + mx.stop_gradient(q - x)


def quant(x, bits, fs):
    if FREE:
        return x
    lv = 2 ** (bits - 1) - 1
    xc = mx.clip(x, -fs, fs)
    return ste(xc, mx.round(xc / fs * lv) / lv * fs)


class CapLinear(nn.Module):
    """y = W x on an AC capacitor grid. Trainable: u (normalised capacitance difference per cell pair)
    and g (column gain = read-out feedback resistor). W = u_eff * g."""

    def __init__(self, n_in, n_out, init_std=0.02):
        super().__init__()
        self.u = mx.clip(mx.random.normal((n_out, n_in)) * 0.3, -1.0, 1.0)
        self.g = mx.full((n_out,), init_std / 0.3)
        self.x_fs, self.y_fs = hw.x_fs, hw.y_fs
        self._chip = None

    def draw(self, key):
        """Static errors of one manufactured chip (not trainable)."""
        n_out, n_in = self.u.shape
        k = mx.random.split(key, 6)
        s_par = hw.c_par / hw.c_max * hw.c_par_mis
        c_avg = float(mx.abs(mx.clip(self.u, -1, 1)).mean()) / 2 * hw.c_max
        p = (2 * (mx.arange(n_out) % hw.tile)).astype(mx.float32)
        a_row = p * (2 * hw.tile) - p ** 2 / 2
        phi = 2 * math.pi * hw.freq * hw.r_wire * c_avg * a_row * (1 - hw.phase_comp)
        self._chip = {
            "ep": mx.random.normal((n_out, n_in), key=k[0]), "en": mx.random.normal((n_out, n_in), key=k[1]),
            "s": s_par * (mx.random.normal((n_out, n_in), key=k[2]) - mx.random.normal((n_out, n_in), key=k[3])),
            "att": mx.cos(phi),
            "gain": 1 + hw.amp_gain_std * mx.random.normal((n_out,), key=k[4]),
            "off": hw.amp_off * mx.random.normal((n_out,), key=k[5]),
            "key": mx.random.split(key, 7)[6]}
        mx.eval(self._chip)

    def weight(self, physical):
        u = mx.clip(self.u, -1.0, 1.0)
        if not physical:
            return u * self.g[:, None]
        ch = self._chip
        uq = ste(u, mx.round(u * hw.levels) / hw.levels)               # 8-bit unit capacitors
        cp, cn = mx.maximum(uq, 0.0), mx.maximum(-uq, 0.0)
        a = hw.pelgrom * math.sqrt(1e-15 / hw.c_max)                   # Pelgrom: sigma_C = a sqrt(C)
        # mismatch is a fixed manufacturing error whose size depends on the cell value; no gradient
        # flows through sqrt(C) (its slope is infinite at C = 0 and would swamp the real gradient)
        cp = cp + a * mx.stop_gradient(mx.sqrt(cp)) * ch["ep"]
        cn = cn + a * mx.stop_gradient(mx.sqrt(cn)) * ch["en"]
        ueff = (cp - cn + ch["s"]) * ch["att"][:, None]
        return ueff * (self.g * ch["gain"])[:, None]

    def __call__(self, x, physical):
        w = self.weight(physical)
        if not physical:
            y = x @ w.T
            if REC is not None:
                r = REC.setdefault(id(self), [0.0, 0.0])
                r[0], r[1] = max(r[0], float(mx.abs(x).max())), max(r[1], float(mx.abs(y).max()))
            return y
        if not FREE:
            x = mx.clip(x, -self.x_fs, self.x_fs)
        y = x @ w.T
        bw = 1.0 / (2 * hw.t_int)
        rel = hw.i_n * math.sqrt(bw) / (hw.v * 2 * math.pi * hw.freq * hw.c_max)   # vs one full cell
        ch = self._chip
        ch["key"], sub = mx.random.split(ch["key"])
        y = y + rel * self.x_fs * self.g * mx.random.normal(y.shape, key=sub) + ch["off"] * self.y_fs
        return y if FREE else mx.clip(y, -self.y_fs, self.y_fs)


class NormUnit(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = mx.ones((d,))          # trainable gain-resistor ratio per line
        self._chip = None

    def draw(self, key):
        k1, k2 = mx.random.split(key)
        self._chip = {"mism": 1 + hw.norm_gain_std * mx.random.normal(self.w.shape, key=k1), "key": k2}

    def __call__(self, x, physical):
        rms = mx.sqrt((x * x).mean(-1, keepdims=True) + 1e-6)
        if not physical:
            return x / rms * self.w
        ch = self._chip
        ch["key"], sub = mx.random.split(ch["key"])
        rms = rms * (1 + hw.norm_dyn_std * mx.random.normal(rms.shape, key=sub))
        return x / rms * (self.w * ch["mism"])


class SiLUUnit(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n, self.fs, self._chip = n, hw.y_fs, None

    def draw(self, key):
        k1, k2 = mx.random.split(key)
        self._chip = {"gain": 1 + hw.fu_gain_std * mx.random.normal((self.n,), key=k1),
                      "off": hw.fu_off * self.fs * mx.random.normal((self.n,), key=k2)}

    def __call__(self, x, physical):
        if not physical:
            return nn.silu(x)
        ch = self._chip
        x = x * ch["gain"] + ch["off"]
        return nn.silu(x if FREE else mx.clip(x, -self.fs, self.fs))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1, self.n2 = NormUnit(D), NormUnit(D)
        self.qkv = CapLinear(D, 3 * D)
        self.o = CapLinear(D, D, 0.02 / math.sqrt(2 * NL))
        self.up = CapLinear(D, FF)
        self.act = SiLUUnit(FF)
        self.down = CapLinear(FF, D, 0.02 / math.sqrt(2 * NL))
        self.o.x_fs = self.down.x_fs = hw.y_fs            # their inputs come from 12 V-scale rails
        self.rope = nn.RoPE(D // NH)

    def __call__(self, x, mask, physical):
        B, T, _ = x.shape
        h = self.n1(x, physical)
        qkv = self.qkv(h, physical)
        if physical:                                       # ADC into the digital attention unit
            qkv = quant(qkv, hw.adc_bits, hw.y_fs)
        q, k, v = [t.reshape(B, T, NH, D // NH).transpose(0, 2, 1, 3) for t in mx.split(qkv, 3, axis=-1)]
        q, k = self.rope(q), self.rope(k)
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=(D // NH) ** -0.5, mask=mask)
        a = a.transpose(0, 2, 1, 3).reshape(B, T, D)
        if physical:                                       # DAC back to the analog grid
            a = quant(a, hw.adc_bits, hw.y_fs)
        x = x + self.o(a, physical)
        h = self.n2(x, physical)
        return x + self.down(self.act(self.up(h, physical), physical), physical)


class TinyHW(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = CapLinear(D, V)                        # shared embedding / output grid
        self.blocks = [Block() for _ in range(NL)]
        self.nf = NormUnit(D)

    def units(self):
        out = [self.head, self.nf]
        for b in self.blocks:
            out += [b.n1, b.n2, b.qkv, b.o, b.up, b.act, b.down]
        return out

    def draw_chip(self, seed):
        keys = mx.random.split(mx.random.key(seed), len(self.units()))
        for unit, k in zip(self.units(), keys):
            unit.draw(k)

    def __call__(self, ids, physical=False):
        T = ids.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
        x = self.head.weight(physical)[ids]                # embedding: read a row of the grid
        for i, blk in enumerate(self.blocks):
            x = blk(x, mask, physical)
            if physical and i + 1 < NL:                    # chip-to-chip link
                x = quant(x, hw.link_bits, hw.resid_fs)
        h = self.nf(x, physical)
        logits = self.head(h, physical)
        if physical:                                       # output ADC referenced to the peak logit
            peak = mx.stop_gradient(logits.max(-1, keepdims=True))
            r = hw.out_range / 2
            logits = peak - r + quant(logits - peak + r, hw.out_bits, r)
        return logits

    def remap(self):
        """Map a digitally trained model onto the cells: per-column gain = largest |w| in the row."""
        for unit in self.units():
            if isinstance(unit, CapLinear):
                w = mx.clip(unit.u, -1, 1) * unit.g[:, None]
                g = mx.maximum(mx.abs(w).max(axis=1), 1e-8)
                unit.u, unit.g = w / g[:, None], g


def batches(split, bs, seed):
    data = np.memmap(os.path.join(DATA, f"{split}.bin"), dtype=np.uint16, mode="r")
    rng = np.random.default_rng(seed)
    while True:
        ix = rng.integers(0, len(data) - CTX - 1, bs)
        x = np.stack([data[i:i + CTX + 1] for i in ix]).astype(np.int32)
        yield mx.array(x[:, :-1]), mx.array(x[:, 1:])


def loss_fn(model, x, y, physical):
    logits = model(x, physical)
    return nn.losses.cross_entropy(logits, y, reduction="mean")


def calibrate_rails(model, x):
    """For a digitally trained model: set each grid's input/output rails to the largest signal seen
    (the best a designer can do when mapping a model that was not trained for the rails)."""
    global REC
    REC = {}
    model(x, False)
    for unit in model.units():
        if isinstance(unit, CapLinear) and id(unit) in REC:
            unit.x_fs, unit.y_fs = max(REC[id(unit)][0], 1e-3), max(REC[id(unit)][1], 1e-3)
    REC = None


def evaluate(model, physical, n_batches=40, bs=32):
    it = batches("val", bs, 1234)
    tot = 0.0
    for _ in range(n_batches):
        x, y = next(it)
        tot += float(loss_fn(model, x, y, physical))
    return tot / n_batches


def sample(model, physical, prompt="Once upon a time", n=120, temp=0.8, topk=40, seed=0):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(DATA, "tokenizer.json"))
    ids = tok.encode(prompt).ids
    rng = np.random.default_rng(seed)
    for _ in range(n):
        logits = np.array(model(mx.array([ids[-CTX:]]), physical)[0, -1].astype(mx.float32)) / temp
        top = np.argpartition(-logits, topk)[:topk]
        p = np.exp(logits[top] - logits[top].max())
        ids.append(int(rng.choice(top, p=p / p.sum())))
    return tok.decode(ids)


def save(model, path, step):
    mx.save_safetensors(path, dict(tree_flatten(model.trainable_parameters())), metadata={"step": str(step)})


def load(model, path):
    w = mx.load(path, return_metadata=True)
    model.update(tree_unflatten(list(w[0].items())))
    return int(w[1].get("step", "0"))


def cmd_train(a):
    global FREE
    os.makedirs(OUT, exist_ok=True)
    mx.random.seed(a.seed)
    model = TinyHW()
    name = a.name or a.mode
    ckpt = os.path.join(OUT, f"{name}.safetensors")
    FREE = a.free
    step0 = load(model, ckpt) if os.path.exists(ckpt) else 0
    if step0 == 0 and a.init == "digital":          # start from the digital model mapped onto the cells
        load(model, os.path.join(OUT, "digital.safetensors"))
        model.remap()
        calibrate_rails(model, next(batches("train", 16, 99))[0])
        json.dump({str(i): [u.x_fs, u.y_fs] for i, u in enumerate(model.units()) if isinstance(u, CapLinear)},
                  open(os.path.join(OUT, f"{name}.rails.json"), "w"))
    elif a.init == "digital":
        rails = json.load(open(os.path.join(OUT, f"{name}.rails.json")))
        for i, u in enumerate(model.units()):
            if str(i) in rails:
                u.x_fs, u.y_fs = rails[str(i)]
    physical = a.mode != "digital"
    if physical:
        model.draw_chip(0)
    def lr(s):
        s = s.astype(mx.float32)
        return a.lr * mx.minimum(1.0, (s + 1) / 100) * 0.5 * (1 + mx.cos(math.pi * mx.minimum(s / a.steps, 1.0)))
    opt = optim.AdamW(learning_rate=lr, weight_decay=0.0)
    opt.state["step"] = mx.array(step0, dtype=mx.uint64)
    grad_fn = nn.value_and_grad(model, lambda m, x, y: loss_fn(m, x, y, physical))
    it = batches("train", a.batch, a.seed + step0)
    log = open(os.path.join(OUT, f"{name}.log"), "a")
    t0 = time.time()
    for step in range(step0, a.steps):
        if a.mode == "robust":
            model.draw_chip(1000 + step)                # a new manufactured chip every step
        elif physical and step % 250 == 0:
            model.draw_chip(0)                          # same chip; refresh the phase map as weights move
        x, y = next(it)
        loss, grads = grad_fn(model, x, y)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        opt.update(model, grads)
        for unit in model.units():                      # physical limits of the cells / resistors
            if isinstance(unit, CapLinear):
                unit.u = mx.clip(unit.u, -1.0, 1.0)
                unit.g = mx.maximum(unit.g, 1e-5)
        mx.eval(model.parameters(), opt.state)
        if step % 100 == 0 or step + 1 == a.steps:
            msg = f"{name} step {step} loss {float(loss):.4f} {time.time() - t0:.0f}s"
            print(msg, flush=True)
            log.write(msg + "\n"); log.flush()
        if (step + 1) % 500 == 0 or step + 1 == a.steps:
            save(model, ckpt, step + 1)


def cmd_eval(a):
    res = {}
    xcal, _ = next(batches("train", 16, 99))
    global FREE
    for mode in a.models.split(","):
        ckpt = os.path.join(OUT, f"{mode}.safetensors")
        if not os.path.exists(ckpt):
            continue
        model = TinyHW()
        load(model, ckpt)
        FREE = mode.endswith("free")
        r = {"ideal": evaluate(model, False)}
        rails = os.path.join(OUT, f"{mode}.rails.json")
        if mode == "digital":
            model.remap()
            calibrate_rails(model, xcal)
        elif os.path.exists(rails):
            for i, u in enumerate(model.units()):
                if str(i) in json.load(open(rails)):
                    u.x_fs, u.y_fs = json.load(open(rails))[str(i)]
        for chip in range(5):
            model.draw_chip(chip)
            r[f"chip{chip}"] = evaluate(model, True)
        model.draw_chip(0)
        r["sample_chip0"] = sample(model, True)
        res[mode] = r
        chips = [r[f"chip{c}"] for c in range(5)]
        print(f"{mode:10s} exact-math val loss {r['ideal']:.4f} | on chip 0 (training chip) {chips[0]:.4f} | "
              f"unseen chips 1-4 {np.mean(chips[1:]):.4f} +- {np.std(chips[1:]):.4f}", flush=True)
    for mode, r in res.items():
        print(f"\n--- {mode} model, generated on chip 0 ---\n{r['sample_chip0']}")
    json.dump(res, open(os.path.join(OUT, f"eval_{a.models.replace(',', '_')}.json"), "w"), indent=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["train", "eval"])
    p.add_argument("--mode", choices=["digital", "chip", "robust"], default="chip")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default="", help="checkpoint name (default: mode)")
    p.add_argument("--init", choices=["scratch", "digital"], default="scratch")
    p.add_argument("--free", action="store_true", help="ablation: no rails, no ADC rounding")
    p.add_argument("--models", default="digital,chip,robust", help="eval: checkpoint names")
    a = p.parse_args()
    {"train": cmd_train, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
