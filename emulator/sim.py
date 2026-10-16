"""Qwen3.5 text model rebuilt from analog hardware blocks (see hw.py).

The forward pass is written out op by op so every operation is an explicit circuit block:
crossbars for all weight matrices, function units, Gilbert multipliers, norm units, sample-and-
hold short convolutions, a leaky capacitor array for the DeltaNet state, a capacitor KV cache with
a transistor softmax for attention, and an output crossbar read by an ADC.

With Chip(ideal=True) the forward pass must reproduce mlx-lm's digital model (checked by
`run.py validate`).
"""

import math
import os
import zlib

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.gated_delta import compute_g, gated_delta_kernel, normalize_qk

from hw import K_B, T_REF, Chip, Crossbar, FunctionUnit, Multiplier, NormUnit, adc, split_keys
from memkernel import analog_gated_delta

MODEL_ID = os.environ.get("KLM_MODEL", "Qwen/Qwen3.5-0.8B")   # Hugging Face id or a local directory


def load_digital():
    from mlx_lm import load
    model, tok = load(MODEL_ID)
    return model, tok


class Recorder:
    """Collects |signal| samples at named nodes during the calibration pass."""

    def __init__(self, per_node=200_000, seed=0):
        self.samples, self.m2, self.per_node, self.seed = {}, {}, per_node, seed
        self.peak = {}                                    # true maximum over ALL values (rails use it)

    def __call__(self, name, x, fs_of=None):
        a = np.abs(np.array(x.astype(mx.float32))).ravel()
        self.peak[name] = max(self.peak.get(name, 0.0), float(a.max()))
        if a.size > self.per_node:
            # each node gets its own random stream, so adding a node never changes another node's statistics
            calls = len(self.samples.get(name, []))
            rng = np.random.default_rng([self.seed, zlib.crc32(name.encode()), calls])
            a = rng.choice(a, self.per_node, replace=False)
        self.samples.setdefault(name, []).append(a)
        return x

    def stats(self):
        out = {}
        for k, v in self.samples.items():
            a = np.concatenate(v)
            out[k] = {"p99.9": float(np.percentile(a, 99.9)), "p99.99": float(np.percentile(a, 99.99)),
                      "max": self.peak[k], "rms": float(np.sqrt((a ** 2).mean()))}
        return out


def fs(calib, name, pct):
    s = calib[name]
    v = {99.9: s["p99.9"], 99.99: s["p99.99"], 100.0: s["max"]}[pct]
    return max(v, 1e-6)


class AnalogQwen:
    def __init__(self, digital, chip: Chip, calib):
        chip.validate() if not chip.ideal else None
        self.chip, self.calib = chip, calib
        lm = digital.language_model
        self.args = lm.args
        m = lm.model
        pct = chip.fs_pct
        key = mx.random.key(chip.seed)
        keys = iter(split_keys(key, 4096))
        F = lambda name: fs(calib, name, pct) if calib else 1.0
        self.F = F
        def xb(w, i, o, c=chip):
            k = next(keys)
            if c.shared_array:                            # same physical grid for this role in every layer
                role = o.split(".", 1)[-1]
                k = mx.random.key(c.seed * 1000003 + zlib.crc32(role.encode()) % 1000003)
            return Crossbar(w, c, F(i), F(o), k, name=i)
        self.xb = xb
        self.n_layers = len(m.layers)
        # shared embedding / output grid
        E = m.embed_tokens.weight
        self.head = self.xb(E, "final_norm", "logits")
        self.embed_w = self.head.w_eff                       # same conductances, read by rows
        self.final_norm = NormUnit(m.norm.weight, self.args.rms_norm_eps, chip, next(keys))
        self.layers = []
        for i, L in enumerate(m.layers):
            p = f"L{i}."
            ci = chip.for_layer(i)                        # one chip per layer
            if not ci.ideal:
                ci.validate()
            d = {"linear": L.is_linear, "chip": ci,
                 "norm1": NormUnit(L.input_layernorm.weight, self.args.rms_norm_eps, ci, next(keys)),
                 "norm2": NormUnit(L.post_attention_layernorm.weight, self.args.rms_norm_eps, ci, next(keys))}
            mlp = L.mlp
            hid = mlp.gate_proj.weight.shape[0]
            d["gate"] = self.xb(mlp.gate_proj.weight, p + "norm2", p + "mlp_gate", ci)
            d["up"] = self.xb(mlp.up_proj.weight, p + "norm2", p + "mlp_up", ci)
            d["silu"] = FunctionUnit(nn.silu, hid, ci, F(p + "mlp_gate"), next(keys))
            d["mul"] = Multiplier(hid, ci, F(p + "mlp_act"), F(p + "mlp_up"), next(keys))
            d["down"] = self.xb(mlp.down_proj.weight, p + "mlp_h", p + "mlp_out", ci)
            if L.is_linear:
                g = L.linear_attn
                d.update(self._gdn(g, ci, F, p, keys))
            else:
                a = L.self_attn
                d.update(self._attn(a, ci, F, p, keys))
            self.layers.append(d)

    # ---- block builders --------------------------------------------------------------------
    def _gdn(self, g, chip, F, p, keys):
        conv_dim = g.conv_dim
        w = g.conv1d.weight[:, :, 0].astype(mx.float32)          # (C, K) depthwise taps
        k_t = next(keys)
        if chip.on("conv"):
            # taps are resistor ratios; delayed samples sit on sample-and-hold capacitors
            w = w * (1 + 0.002 * mx.random.normal(w.shape, key=k_t))
        return {"g": g,
                "qkv": self.xb(g.in_proj_qkv.weight, p + "norm1", p + "qkv", chip),
                "z": self.xb(g.in_proj_z.weight, p + "norm1", p + "z", chip),
                "b": self.xb(g.in_proj_b.weight, p + "norm1", p + "b", chip),
                "a": self.xb(g.in_proj_a.weight, p + "norm1", p + "a", chip),
                "conv_w": w,
                "conv_silu": FunctionUnit(nn.silu, conv_dim, chip, F(p + "conv_pre"), next(keys)),
                "gnorm": NormUnit(g.norm.weight, g.layer_norm_epsilon, chip, next(keys)),
                "gmul": Multiplier(g.value_dim, chip, F(p + "z_act"), F(p + "gnorm"), next(keys)),
                "zsilu": FunctionUnit(nn.silu, g.value_dim, chip, F(p + "z"), next(keys)),
                "out": self.xb(g.out_proj.weight, p + "gdn_y", p + "mix_out", chip),
                "line_k": None if chip.ideal else 1 + chip.mult_gain_std * mx.random.normal((g.num_k_heads, g.head_k_dim), key=next(keys)),
                "line_q": None if chip.ideal else 1 + chip.mult_gain_std * mx.random.normal((g.num_k_heads, g.head_k_dim), key=next(keys)),
                "line_v": None if chip.ideal else 1 + chip.mult_gain_std * mx.random.normal((g.num_v_heads, g.head_v_dim), key=next(keys)),
                "s_fs": mx.array([F(p + f"state.h{hh}") for hh in range(g.num_v_heads)]) if not chip.ideal else 1.0,
                "y_fs": mx.array([F(p + f"gdn_core.h{hh}") for hh in range(g.num_v_heads)])[:, None] if not chip.ideal else 1.0,
                "mem_fs_head": (mx.array(np.repeat([F(p + f"mem_in.g{j}") for j in range(conv_dim // g.head_k_dim)], g.head_k_dim))
                                if not chip.ideal and self.calib and f"{p}mem_in.g0" in self.calib else None),
                "mism": chip.mem()["write_rel"] * mx.random.normal((g.num_v_heads, g.head_v_dim, g.head_k_dim), key=next(keys)),
                "calls": 0,
                "key": next(keys)}

    def _attn(self, a, chip, F, p, keys):
        H, Hkv, hd = a.num_attention_heads, a.num_key_value_heads, a.head_dim
        d = {"a": a,
             "q": self.xb(a.q_proj.weight, p + "norm1", p + "q", chip),
             "k": self.xb(a.k_proj.weight, p + "norm1", p + "k", chip),
             "v": self.xb(a.v_proj.weight, p + "norm1", p + "v", chip),
             "qn": NormUnit(a.q_norm.weight, a.q_norm.eps, Chip(ideal=True) if chip.mem_digital() else chip, next(keys)),
             "kn": NormUnit(a.k_norm.weight, a.k_norm.eps, Chip(ideal=True) if chip.mem_digital() else chip, next(keys)),
             "sig": FunctionUnit(mx.sigmoid, H * hd, chip, F(p + "attn_gate"), next(keys)),
             "gmul": Multiplier(H * hd, chip, F(p + "attn_o"), 1.0, next(keys)),
             "o": self.xb(a.o_proj.weight, p + "attn_y", p + "mix_out", chip),
             "k_fs": F(p + "k_stored") if not chip.ideal else 1.0,
             "v_fs": F(p + "v_stored") if not chip.ideal else 1.0,
             "sm_eps": None if chip.ideal else chip.softmax_eps_std() * mx.random.normal((1, H, 1, 16384), key=next(keys)),
             "key": next(keys)}
        return d

    # ---- forward -------------------------------------------------------------------------
    def make_cache(self):
        """Per-layer-chip state kept between tokens: short-conv history, DeltaNet state, KV cache."""
        return [dict() for _ in self.layers]

    def __call__(self, ids, rec=None, cache=None):
        """ids: (B, T). Without a cache the whole sequence is processed at once. With a cache (from
        make_cache) the tokens continue from the stored state, exactly as the layer chips run."""
        chip = self.chip
        R = rec if rec is not None else (lambda n, x: x)
        x = R("embed", self.embed_w[ids])
        for i, d in enumerate(self.layers):
            p = f"L{i}."
            c = None if cache is None else cache[i]
            h = R(p + "norm1", d["norm1"](x))
            r = self._gdn_fwd(d, h, R, p, rec, c) if d["linear"] else self._attn_fwd(d, h, R, p, c)
            x = x + r
            h = R(p + "norm2", d["norm2"](x))
            gt = R(p + "mlp_gate", d["gate"](h))
            up = R(p + "mlp_up", d["up"](h))
            act = R(p + "mlp_act", d["silu"](gt))
            hm = R(p + "mlp_h", d["mul"](act, up))
            x = x + R(p + "mlp_out", d["down"](hm))
            R(p + "resid", x)
            ci = d["chip"]
            if i + 1 < self.n_layers and not ci.ideal:
                if ci.link_bits:                          # digital link: ADC -> serial link -> DAC
                    x = adc(x, ci.link_bits, fs(self.calib, p + "resid", ci.fs_pct))
                elif ci.link_noise:                       # analog link between packages
                    d["link_key"], sub = mx.random.split(d.get("link_key", mx.random.key(ci.seed * 31 + i)))
                    x = x + ci.link_noise * fs(self.calib, p + "resid", ci.fs_pct) * mx.random.normal(x.shape, key=sub)
            if chip.ckpt_every and chip.on("adc") and (i + 1) % chip.ckpt_every == 0 and i + 1 < self.n_layers:
                x = adc(x, chip.adc_bits, fs(self.calib, p + "resid", chip.fs_pct))
        h = R("final_norm", self.final_norm(x))
        logits = R("logits", self.head(h))
        if chip.on("adc"):
            peak = logits.max(axis=-1, keepdims=True)
            r = chip.out_range / 2
            logits = peak - r + adc(logits - peak + r, chip.adc_bits, r)
        return logits

    def _gdn_fwd(self, d, h, R, p, rec=None, c=None):
        g, chip = d["g"], d["chip"]
        B, S, _ = h.shape
        qkv = R(p + "qkv", d["qkv"](h))
        z = R(p + "z", d["z"](h)).reshape(B, S, g.num_v_heads, g.head_v_dim)
        b = R(p + "b", d["b"](h))
        a = R(p + "a", d["a"](h))
        # 4-tap causal short conv on sample-and-hold delays
        K = d["conv_w"].shape[1]
        if chip.on("conv"):                               # sample-and-hold: kT/C noise once, when written
            sh = math.sqrt(K_B * chip.temp / chip.c_mem) / chip.v_mem * fs(self.calib, p + "qkv", chip.fs_pct)
            d["key"], sub = mx.random.split(d["key"])
            qkv = qkv + sh * mx.random.normal(qkv.shape, key=sub)
        prev = c["conv"] if c is not None and "conv" in c else mx.zeros((B, K - 1, qkv.shape[-1]), dtype=qkv.dtype)
        xin = mx.concatenate([prev, qkv], axis=1)
        if c is not None:
            c["conv"] = xin[:, -(K - 1):, :]
        conv = sum(xin[:, j:j + S, :] * d["conv_w"][:, j] for j in range(K))
        conv = R(p + "conv_pre", conv)
        conv = d["conv_silu"](conv)
        mem_conv = chip.mem_digital() and chip.on("adc")
        if mem_conv:                                      # ADC into the digital memory unit
            mb, skip = chip.mem_adc_bits, chip.mem_adc_skip.split(",")
            if "qkv" not in skip:
                if chip.mem_adc_per_head and d["mem_fs_head"] is None:
                    raise ValueError("calibration has no per-head memory ranges (mem_in.gN); rerun run.py calibrate")
                rng_ = d["mem_fs_head"] if chip.mem_adc_per_head else fs(self.calib, p + "mem_in", chip.fs_pct)
                conv = adc(conv, mb, rng_)
            if "gates" not in skip:
                a = adc(a, mb, fs(self.calib, p + "a", chip.fs_pct))
                b = adc(b, mb, fs(self.calib, p + "b", chip.fs_pct))
        conv = R(p + "mem_in", conv)
        if rec is not None:                               # per-head ranges for the memory ADC
            hd = g.head_k_dim
            for j in range(conv.shape[-1] // hd):
                rec(p + f"mem_in.g{j}", conv[..., j * hd:(j + 1) * hd])
        q, k, v = [t.reshape(B, S, hh, dd) for t, hh, dd in zip(
            mx.split(conv, [g.key_dim, 2 * g.key_dim], -1),
            [g.num_k_heads, g.num_k_heads, g.num_v_heads], [g.head_k_dim, g.head_k_dim, g.head_v_dim])]
        q, k = normalize_qk(q, k, inv_scale=g.head_k_dim ** -0.5, eps=1e-6)
        if d["line_k"] is not None and chip.on("gdn_mem"):    # row / column driver mismatch
            q, k, v = q * d["line_q"], k * d["line_k"], v * d["line_v"]
        beta = mx.sigmoid(b)
        gdec = compute_g(g.A_log, a, g.dt_bias)
        if chip.on("gdn_mem"):
            gdec = gdec * chip.mem_leak(self.n_layers)    # capacitor leakage per token
        state = c["state"] if c is not None and "state" in c else mx.zeros((B, g.num_v_heads, g.head_v_dim, g.head_k_dim), dtype=mx.float32)
        qf, kf, vf = q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32)
        if chip.on("gdn_mem"):
            d["calls"] += 1
            seed = chip.seed * 1000003 + int(p[1:-1]) * 7919 + d["calls"] * 104729
            sigma = chip.ktc_rel() * d["s_fs"]                    # kT/C charge noise per write
            y, state = analog_gated_delta(qf, kf, vf, gdec, beta, state, d["mism"], sigma, seed,
                                          fs=d["s_fs"], bits=chip.refresh_bits, every=chip.refresh_every,
                                          stochastic=chip.refresh_stoch)
            d["key"], sub = mx.random.split(d["key"])
            y = y + chip.mem_read_rel * d["y_fs"] * mx.random.normal(y.shape, key=sub)
        else:
            y, state = gated_delta_kernel(qf, kf, vf, gdec, beta, state, None)
        if c is not None:
            c["state"] = state
        if mem_conv and "y" not in chip.mem_adc_skip.split(","):   # DAC back to the analog part, per-head range
            y = adc(y, chip.mem_dac_bits, d["y_fs"])
        R(p + "state", state)
        if rec is not None:
            for hh in range(g.num_v_heads):
                rec(p + f"state.h{hh}", state[:, hh])
                rec(p + f"gdn_core.h{hh}", y[:, :, hh])
        y = R(p + "gdn_core", y)
        # gated norm: rms_norm(y) * w  times silu(z)
        n = R(p + "gnorm", d["gnorm"](y))
        za = R(p + "z_act", d["zsilu"](z.reshape(B, S, -1)))
        yg = d["gmul"](za, n.reshape(B, S, -1))
        yg = R(p + "gdn_y", yg)
        return R(p + "mix_out", d["out"](yg))

    def _attn_fwd(self, d, h, R, p, c=None):
        a, chip = d["a"], d["chip"]
        B, L, _ = h.shape
        H, Hkv, hd = a.num_attention_heads, a.num_key_value_heads, a.head_dim
        qo = R(p + "q", d["q"](h))
        queries, gate = mx.split(qo.reshape(B, L, H, -1), 2, axis=-1)
        gate = gate.reshape(B, L, -1)
        keys = R(p + "k", d["k"](h))
        values = R(p + "v", d["v"](h))
        mem_conv = chip.mem_digital() and chip.on("adc")
        if mem_conv and "attn_in" not in chip.mem_adc_skip.split(","):   # ADC into the digital attention unit
            mb = chip.mem_adc_bits
            queries = adc(queries, mb, fs(self.calib, p + "q", chip.fs_pct))
            keys = adc(keys, mb, fs(self.calib, p + "mem_in", chip.fs_pct))
            values = adc(values, mb, fs(self.calib, p + "v", chip.fs_pct))
        keys = R(p + "mem_in", keys)
        queries = d["qn"](queries).transpose(0, 2, 1, 3)
        keys = d["kn"](keys.reshape(B, L, Hkv, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, Hkv, -1).transpose(0, 2, 1, 3)
        p0 = c.get("offset", 0) if c is not None else 0
        queries, keys = a.rope(queries, offset=p0), a.rope(keys, offset=p0)
        R(p + "k_stored", keys)
        R(p + "v_stored", values)
        if chip.on("kv_mem"):
            # KV written as charge on capacitor cells: kT/C + write error, drawn once per write
            wr, ktc = chip.mem()["write_rel"], chip.ktc_rel()
            d["key"], k1, k2, k5, k6 = split_keys(d["key"], 5)
            keys = keys * (1 + wr * mx.random.normal(keys.shape, key=k5)) + ktc * d["k_fs"] * mx.random.normal(keys.shape, key=k1)
            values = values * (1 + wr * mx.random.normal(values.shape, key=k6)) + ktc * d["v_fs"] * mx.random.normal(values.shape, key=k2)
        if c is not None:                                 # the KV cache holds every stored token
            if "k" in c:
                keys, values = mx.concatenate([c["k"], keys], axis=2), mx.concatenate([c["v"], values], axis=2)
            c["k"], c["v"], c["offset"] = keys, values, p0 + L
        T = keys.shape[2]
        rep = H // Hkv
        keys, values = mx.repeat(keys, rep, axis=1), mx.repeat(values, rep, axis=1)
        scores = (queries * a.scale) @ keys.transpose(0, 1, 3, 2)              # (B,H,L,T)
        qpos, kpos = p0 + mx.arange(L), mx.arange(T)
        causal = kpos[None, :] <= qpos[:, None]
        mem_on, sm_on = chip.on("kv_mem"), chip.on("softmax")
        if mem_on:
            age = (qpos[:, None] - kpos[None, :]).astype(mx.float32)
            leak = mx.where(causal, mx.power(chip.mem_leak(self.n_layers), mx.maximum(age, 0)), 0.0)
            scores = scores * leak                                             # stored keys fade
            s_fs = mx.abs(mx.where(causal, scores, 0)).max(axis=(-2, -1), keepdims=True)  # per head
            d["key"], sub = mx.random.split(d["key"])
            scores = scores + chip.mem_read_rel * s_fs * mx.random.normal(scores.shape, key=sub)
        if sm_on:
            scores = scores * (T_REF / chip.temp)                              # V_T = kT/q
            smax = mx.where(causal, scores, -mx.inf).max(axis=-1, keepdims=True)
            eps = d["sm_eps"][..., :T]                                         # static, per branch slot
            cur = mx.exp(mx.maximum(scores - smax, -chip.sm_decades * math.log(10)) + eps)
            cur = mx.where(causal, cur + 10 ** (-chip.sm_decades), 0.0)       # off-branch leakage
            w = cur / cur.sum(axis=-1, keepdims=True)
        else:
            w = mx.softmax(mx.where(causal, scores, -mx.inf), axis=-1)
        if mem_on:
            w = w * leak                                                       # stored values fade
        out = w @ values
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        if mem_conv and "attn_out" not in chip.mem_adc_skip.split(","):   # DAC back to the analog part
            out = adc(out, chip.mem_dac_bits, fs(self.calib, p + "attn_o", chip.fs_pct))
        out = R(p + "attn_o", out)
        sg = R(p + "attn_gate", gate)
        y = R(p + "attn_y", d["gmul"](out, d["sig"](sg)))
        return R(p + "mix_out", d["o"](y))

    # ---- energy / latency model -------------------------------------------------------------
    def crossbars(self):
        for d in self.layers:
            for k, v in d.items():
                if isinstance(v, Crossbar):
                    yield v
        yield self.head

    def area_mm2(self):
        """Weight-array area per layer chip and for the shared embedding/output grid (mm^2)."""
        per_layer = [sum(v.area_mm2() for v in d.values() if isinstance(v, Crossbar)) for d in self.layers]
        return {"layer_max": max(per_layer), "layer_mean": sum(per_layer) / len(per_layer),
                "embed_out": self.head.area_mm2(), "total": sum(per_layer) + self.head.area_mm2()}

    def energy_per_token(self, ctx, only_layer=None):
        """Energy per generated token (J) at context length ctx, by component (one layer chip if only_layer)."""
        c, cal = self.chip, self.calib
        e = {"arrays+amps": 0.0, "function+mult": 0.0, "norms": 0.0, "memory": 0.0, "attention": 0.0, "converters": 0.0}
        xbs = self.crossbars() if only_layer is None else [v for v in self.layers[only_layer].values() if isinstance(v, Crossbar)]
        for xb in xbs:
            m2 = min((cal[xb.name]["rms"] / xb.x_fs) ** 2, 1.0)
            if xb is self.head and c.output == "wta":     # winner-take-all replaces 248k amplifiers
                e["arrays+amps"] += c.t_int * c.v_read ** 2 * m2 * xb.g_total + xb.out_f * c.e_fu
                continue
            e["arrays+amps"] += xb.energy(m2)
        args = self.args
        hid, dm = args.intermediate_size, args.hidden_size
        n_lin = sum(1 for d in self.layers if d["linear"])
        n_att = self.n_layers - n_lin
        e["norms"] += dm * c.e_norm_line
        e["links"] = 0.0
        e["carrier"] = 0.0
        for i, d in enumerate(self.layers):
            if only_layer is not None and i != only_layer:
                continue
            c = d["chip"]
            if c.is_ac() and not c.ideal:   # carrier oscillator runs while the layer chip computes
                e["carrier"] += c.p_carrier * c.stages_per_layer * c.t_stage()
            dig = c.mem_digital()
            e["function+mult"] += hid * (c.e_fu + c.e_mult)
            e["norms"] += 2 * dm * c.e_norm_line
            if i + 1 < self.n_layers and c.link_bits:
                e["links"] += dm * (2 * c.adc_fom * 2 ** c.link_bits + c.link_bits * c.e_link_bit)
            if d["linear"]:
                g = d["g"]
                cells = g.num_v_heads * g.head_v_dim * g.head_k_dim
                if dig:   # decay, read S k, write k delta^T, read S q: ~4 MACs per cell + 2 converters per line
                    e["memory"] += (4 * cells * c.e_dmac + (g.key_dim * 2 + g.value_dim + 2 * g.num_v_heads) * c.adc_fom * 2 ** c.mem_adc_bits
                                    + g.value_dim * c.adc_fom * 2 ** c.mem_dac_bits)
                else:
                    e["memory"] += cells * c.c_mem * c.v_mem ** 2 * 2           # write + read
                    if c.refresh_bits:   # comparator ladder per cell + recharge, every R tokens
                        e["memory"] += cells * (c.adc_fom * 2 ** c.refresh_bits + c.c_mem * c.v_mem ** 2) / c.refresh_every
                e["function+mult"] += g.conv_dim * c.e_fu + 2 * g.value_dim * (c.e_fu + c.e_mult)
                e["norms"] += g.value_dim * c.e_norm_line
            else:
                a = d["a"]
                cells = 2 * a.num_key_value_heads * a.head_dim
                if dig:   # scores + value mix as digital MACs over the context
                    e["memory"] += (cells * c.e_dmac + (a.num_attention_heads * a.head_dim + cells) * c.adc_fom * 2 ** c.mem_adc_bits
                                    + a.num_attention_heads * a.head_dim * c.adc_fom * 2 ** c.mem_dac_bits)
                    e["attention"] += 2 * a.num_attention_heads * ctx * a.head_dim * c.e_dmac
                else:
                    e["memory"] += cells * c.c_mem * c.v_mem ** 2                # write this token's K, V
                    e["attention"] += a.num_attention_heads * ctx * (2 * a.head_dim * c.c_mem * c.v_mem ** 2 * 0.05 + c.e_fu)
                e["function+mult"] += a.num_attention_heads * a.head_dim * (c.e_fu + c.e_mult)
        c = self.chip
        if only_layer is not None:
            e["total"] = sum(e.values())
            return e
        ckpts = 0 if c.ckpt_every == 0 else (self.n_layers - 1) // c.ckpt_every
        e["converters"] += ckpts * dm * c.adc_fom * 2 ** c.adc_bits * 2
        if c.output == "adc":
            e["converters"] += args.vocab_size * c.adc_fom * 2 ** c.adc_bits
        e["total"] = sum(e.values())
        return e


def t_token(model):
    """Latency of one token through the chain of layer chips (s)."""
    t = 0.0
    for i, d in enumerate(model.layers):
        c = d["chip"]
        t += c.stages_per_layer * c.t_stage()
        if i + 1 < model.n_layers and c.link_bits:
            t += 20e-9 + c.link_bits * dm_bits_time(c)
    return t + 2 * model.chip.t_stage()


def dm_bits_time(c, lanes=64, gbps=10.0):
    """Serial-link time per bit-plane of the 1,024-wide residual: 64 lanes at 10 Gbit/s."""
    return 1024 / (lanes * gbps * 1e9)


def token_nll(model, ids, chunk=512):
    """Sum of next-token negative log-likelihood (nats) over ids[1:]."""
    x = mx.array(ids)[None, :-1]
    tgt = mx.array(ids)[None, 1:]
    logits = model(x)
    total = 0.0
    for s in range(0, logits.shape[1], chunk):
        lg = logits[:, s:s + chunk].astype(mx.float32)
        lse = mx.logsumexp(lg, axis=-1)
        pick = mx.take_along_axis(lg, tgt[:, s:s + chunk, None], axis=-1)[..., 0]
        total += float((lse - pick).sum())
    return total
