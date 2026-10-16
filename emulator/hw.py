"""Physical models of the analog hardware that runs the language model.

Units are SI (siemens, volts, amperes, seconds, joules, kelvin). Every model block maps an
ideal math operation to the circuit that computes it and applies the circuit's physics:

  crossbar      weights as differential conductance pairs; Ohm's law multiplies, Kirchhoff's
                law sums. Programming error, quantised levels, drift, temperature spread,
                stuck cells, wire IR drop, I-V nonlinearity (static, drawn once per chip);
                thermal, shot and read noise (dynamic, every read); amplifier noise, gain and
                offset error, output rails.
  function unit SiLU / sigmoid / exp transistors: per-unit gain and offset mismatch, rails.
  multiplier    Gilbert cell: gain and offset mismatch, tanh compression.
  norm unit     RMS detector + divider: static gain error, dynamic RMS error.
  memory        capacitor cells (DeltaNet state, KV cache): kT/C noise, leakage, write error.
  softmax       subthreshold transistors on a shared current: exact softmax plus threshold
                mismatch, finite exponential range (off-leakage), temperature.
  converters    ADC/DAC with Walden figure of merit.

The device numbers below are rough literature-range ASSUMPTIONS for a first exploration, not
measurements of a specific product. The search reports how sensitive the result is to them.
"""

import math
from dataclasses import dataclass, field, asdict

import mlx.core as mx

K_B = 1.380649e-23          # Boltzmann constant, J/K
Q_E = 1.602176634e-19       # electron charge, C
T_REF = 300.0               # reference temperature, K

# Resistive weight devices. g_max: allowed range of the maximum conductance (S).
# on_off: G_max / G_min. levels: programmable levels (0 = continuous). prog_abs: programming
# error sigma as a fraction of G_max (prog_rel: fraction of the cell's own G, for resistors).
# read_rel: read (1/f, RTN) noise sigma relative to the cell current. nu / nu_std: drift
# exponent G(t) = G0 (t/t0)^-nu. v0: sinh I-V nonlinearity scale (None = linear).
# v_max: highest safe read voltage (read disturb). tc_std: per-cell random temperature
# coefficient spread (1/K) left after global compensation.
DEVICES = {
    "resistor": dict(area=2.0, g_max=(1e-7, 1e-4), on_off=math.inf, levels=0, prog_abs=0.0, prog_rel=0.001,
                     read_rel=0.0005, nu=0.0, nu_std=0.0, v0=None, v_max=1.0, stuck=0.0, tc_std=2e-5),
    "rram":     dict(area=0.05, g_max=(5e-6, 1e-4), on_off=20.0, levels=32, prog_abs=0.02, prog_rel=0.0,
                     read_rel=0.01, nu=0.005, nu_std=0.002, v0=0.25, v_max=0.3, stuck=1e-3, tc_std=1e-3),
    "pcm":      dict(area=0.05, g_max=(5e-6, 5e-5), on_off=100.0, levels=16, prog_abs=0.015, prog_rel=0.0,
                     read_rel=0.03, nu=0.05, nu_std=0.02, v0=0.4, v_max=0.3, stuck=5e-4, tc_std=2e-3),
    "fefet":    dict(area=0.08, g_max=(1e-6, 2e-5), on_off=100.0, levels=16, prog_abs=0.03, prog_rel=0.0,
                     read_rel=0.01, nu=0.01, nu_std=0.005, v0=0.15, v_max=0.15, stuck=1e-3, tc_std=2e-3),
    "flash":    dict(area=0.1, g_max=(1e-7, 5e-6), on_off=1000.0, levels=256, prog_abs=0.005, prog_rel=0.0,
                     read_rel=0.01, nu=0.002, nu_std=0.001, v0=0.1, v_max=0.1, stuck=1e-4, tc_std=5e-3),
    # AC capacitive grid: weight = metal-oxide-metal capacitance, current I = j*omega*C*V.
    # Admittance Y = 2*pi*f*C replaces G; capacitors are noiseless (no Johnson noise);
    # mismatch follows Pelgrom: sigma_C/C = 0.4% * sqrt(1 fF / C); read_rel = carrier phase noise.
    # levels: binary-weighted unit capacitors (8 bit). Energy per cycle C V^2 / q_res (resonant
    # LC drive recycles the stored energy; q_res = 1 means plain switched drive).
    "cap_ac":   dict(area=None, g_max=(0.0, math.inf), on_off=math.inf, levels=256, prog_abs=0.0, prog_rel=None,
                     read_rel=0.001, nu=0.0, nu_std=0.0, v0=None, v_max=1.0, stuck=0.0, tc_std=3e-5, ac=True),
}

# Capacitor memories for the DeltaNet state and the KV cache.
# tau: leakage time constant (s). write_rel: write error sigma relative to full scale.
MEMORIES = {
    "si_cap": dict(tau=0.01, write_rel=0.005, digital=False),   # plain capacitor + silicon access transistor
    "igzo":   dict(tau=100.0, write_rel=0.005, digital=False),  # oxide-semiconductor gain cell
    "fecap":  dict(tau=1e7, write_rel=0.03, digital=False),     # ferroelectric capacitor (analog partial switching)
    "sram":   dict(tau=math.inf, write_rel=0.0, digital=True),  # digital: SRAM state + digital MACs (hybrid chip)
}


EFFECTS = ("levels", "prog", "stuck", "drift", "temp", "ir", "iv", "noise", "tia", "rails",
           "fu", "mult", "norm", "conv", "gdn_mem", "kv_mem", "softmax", "adc")


@dataclass
class Chip:
    """One hardware design point. Search knobs first, then fixed technology assumptions."""
    device: str = "rram"
    g_max: float = 25e-6        # S, largest cell conductance
    v_read: float = 0.2         # V, input full-scale read voltage
    t_int: float = 50e-9        # s, integration time per array read
    tile: int = 256             # rows = columns per crossbar tile
    p_tia: float = 10e-6        # W, power of one readout amplifier
    memory: str = "igzo"        # capacitor memory technology
    c_mem: float = 10e-15       # F, memory cell capacitance
    adc_bits: int = 8           # output and checkpoint converters
    mem_adc_bits: int = 10      # ADC into the digital memory (DeltaNet state, attention)
    mem_dac_bits: int = 12      # DAC out of the digital memory (the read-out needs more precision)
    mem_adc_skip: str = ""      # ablation: converter groups left out: qkv, gates, y, attn_in, attn_out
    mem_adc_per_head: bool = True   # one ADC range per head (128 channels) for q/k/v into the DeltaNet memory
    ckpt_every: int = 0         # re-digitise the residual stream every k layers (0 = never)
    fs_pct: float = 99.99       # percentile of calibrated |signal| that maps to the rails
    offset_trim: float = 0.1    # residual offset after auto-zero / trim (1 = untrimmed); costs power
    out_range: float = 32.0     # nats below the peak logit covered by the output ADC
    temp: float = 300.0         # K, operating temperature
    drift_time: float = 86400.0 # s since programming (drift)
    drift_comp: bool = True     # global drift compensation (one calibrated gain per array)
    seed: int = 0               # draw of the static per-chip errors
    ac_freq: float = 100e6      # Hz, carrier of the AC capacitive grid (device cap_ac)
    c_cell: float = 2e-15       # F, largest cell capacitance (device cap_ac)
    q_res: float = 1.0          # resonant energy recovery of the LC drive (1 = none)
    ir_comp: float = 0.0        # fraction of the static IR drop pre-compensated when programming
    # AC grid physics and costs
    phase_comp: float = 0.0     # fraction of the row-wire RC phase delay removed by per-column reference phase
    c_par: float = 0.05e-15     # F, stray row-column capacitance at every crossing
    c_par_mis: float = 0.1      # mismatch of that stray capacitance between the two columns of a pair
    ac_off_rej: float = 0.02    # residual amplifier offset after synchronous detection
    e_mod: float = 10e-15       # J per input row per read (carrier modulator / driver logic)
    e_demod: float = 10e-15     # J per output per read (mixer of the synchronous demodulator)
    p_carrier: float = 1e-3     # W per layer chip, carrier oscillator and distribution
    cap_density: float = 2e-15  # F per um^2 (stacked metal-oxide-metal capacitor)
    # fixed technology assumptions
    r_wire: float = 0.5         # ohm per cell segment of a crossbar wire
    r_drv: float = 10.0         # ohm, row driver output resistance
    act_ratio: float = 1.0      # column-wire IR-drop factor, fitted to exact tile solves (circuit_tile.py)
    i_n0: float = 2e-12         # A/sqrt(Hz), amplifier input noise at 10 uW (scales with 1/sqrt(P))
    tia_gain_std: float = 0.005
    tia_off_std: float = 0.002  # fraction of output full scale
    fu_gain_std: float = 0.01   # function units (SiLU, sigmoid)
    fu_off_std: float = 0.005
    mult_gain_std: float = 0.01
    mult_off_std: float = 0.005
    mult_lin: float = 2.0       # Gilbert linear range in multiples of the calibrated full scale
    norm_gain_std: float = 0.005
    norm_dyn_std: float = 0.002
    v_mem: float = 1.0          # V, memory cell full-scale swing
    refresh_bits: int = 0       # DeltaNet memory level restoration: bits per cell (0 = no refresh)
    refresh_every: int = 1      # tokens between refreshes
    refresh_stoch: bool = True  # stochastic rounding (dither) on refresh
    e_dmac: float = 0.2e-12     # J per digital multiply-accumulate incl. SRAM access (digital memory option)
    mem_read_rel: float = 0.003 # memory-array read noise, fraction of full scale
    sm_vt_std: float = 3e-3     # V, softmax transistor threshold mismatch
    sm_n: float = 1.3           # subthreshold slope factor
    sm_decades: float = 5.0     # usable exponential range (decades of current)
    adc_fom: float = 5e-15      # J per conversion step (Walden)
    e_fu: float = 5e-15         # J per function-unit operation
    e_mult: float = 10e-15      # J per multiplier operation
    e_norm_line: float = 10e-15 # J per norm-unit line
    stages_per_layer: int = 8
    output: str = "wta"         # energy model of the output layer: 'wta' (argmax) or 'adc'
    ideal: bool = False         # all physics off: must reproduce the digital model exactly
    # one chip per layer: per-layer overrides {"3": {"device": "flash", ...}}; the chip-to-chip link
    # after each layer is digital with link_bits (ADC -> serial link -> DAC) or analog (link_bits = 0)
    layer_over: dict = field(default_factory=dict)
    link_bits: int = 0
    link_noise: float = 0.0     # analog link: noise sigma relative to the residual full scale
    e_link_bit: float = 1e-12   # J per bit on a chip-to-chip serial link
    off: str = ""               # comma list of effects to switch off (ablation), see EFFECTS

    def on(self, effect):
        if effect not in EFFECTS:
            raise ValueError(f"unknown effect {effect!r}")
        if effect in ("gdn_mem", "kv_mem", "softmax") and not self.ideal and self.mem_digital():
            return False                       # digital memory and attention: exact
        return not self.ideal and effect not in self.off.split(",")

    def for_layer(self, i):
        """The chip that runs layer i: this chip with the layer's overrides applied."""
        over = self.layer_over.get(str(i), self.layer_over.get(i, {}))
        base = {k: v for k, v in asdict(self).items() if k != "layer_over"}
        return Chip(**{**base, **over})

    def dev(self):
        return DEVICES[self.device]

    def mem(self):
        return MEMORIES[self.memory]

    def is_ac(self):
        return self.dev().get("ac", False)

    def g_eff_max(self):
        """Largest cell conductance (DC) or admittance 2*pi*f*C (AC capacitive grid)."""
        return 2 * math.pi * self.ac_freq * self.c_cell if self.is_ac() else self.g_max

    def cell_area_um2(self):
        a = self.dev()["area"]
        return a if a is not None else self.c_cell / self.cap_density + 0.3

    def prog_rel(self):
        """Programming error relative to a FULL cell. For capacitors (Pelgrom, sigma_C = 0.4 % * sqrt(1 fF * C))
        smaller cells have a larger relative error: sigma_C / C = prog_rel * sqrt(C_full / C)."""
        d = self.dev()
        if d["prog_rel"] is None:
            return 0.004 * math.sqrt(1e-15 / self.c_cell)
        return d["prog_rel"]

    def validate(self):
        d = self.dev()
        lo, hi = d["g_max"]
        if self.is_ac() and self.t_int * self.ac_freq < 3 - 1e-6:
            raise ValueError(f"t_int {self.t_int:g} s holds fewer than 3 carrier cycles at {self.ac_freq:g} Hz")
        if not lo <= self.g_max <= hi:
            raise ValueError(f"g_max {self.g_max:g} outside {self.device} range [{lo:g}, {hi:g}]")
        if self.v_read > d["v_max"]:
            raise ValueError(f"v_read {self.v_read:g} V above {self.device} read limit {d['v_max']:g} V")
        if self.memory not in MEMORIES:
            raise ValueError(f"unknown memory {self.memory!r}")
        if self.refresh_bits and not 2 <= self.refresh_bits <= 16:
            raise ValueError(f"refresh_bits {self.refresh_bits} outside 2-16")
        if not 0.2 <= self.v_mem <= 3.3:
            raise ValueError(f"v_mem {self.v_mem:g} V outside 0.2-3.3 V")
        return self

    # derived quantities -------------------------------------------------------------------
    def trim_overhead(self):
        """Power overhead of offset auto-zeroing: +15% per decade of offset reduction."""
        return 1.0 + 0.15 * math.log10(1.0 / self.offset_trim)

    def bandwidth(self):
        return 1.0 / (2.0 * self.t_int)

    def i_noise(self):
        return self.i_n0 * math.sqrt(10e-6 / self.p_tia)

    def t_settle(self):
        return 20e-9 * math.sqrt(10e-6 / self.p_tia)

    def t_stage(self):
        return self.t_int + self.t_settle()

    def t_token(self, layers=24):
        ckpts = 0 if self.ckpt_every == 0 else layers // self.ckpt_every
        return layers * self.stages_per_layer * self.t_stage() + ckpts * 20e-9 + 2 * self.t_stage()

    def mem_digital(self):
        return self.mem()["digital"]

    def mem_leak(self, layers=24):
        """Fraction of a stored value left after one token period."""
        return math.exp(-self.t_token(layers) / self.mem()["tau"])

    def ktc_rel(self):
        """kT/C noise of one memory cell, relative to the cell full-scale swing."""
        return 0.0 if self.mem_digital() else math.sqrt(K_B * self.temp / self.c_mem) / self.v_mem

    def softmax_eps_std(self):
        """Branch-current mismatch: threshold mismatch over the subthreshold slope."""
        return self.sm_vt_std / (self.sm_n * K_B * self.temp / Q_E)

    def to_dict(self):
        return asdict(self)


def split_keys(key, n):
    return list(mx.random.split(key, n))


class Crossbar:
    """A weight matrix W (out, in) mapped onto differential conductance pairs.

    The static (per-chip) errors are folded into an effective weight matrix once. Dynamic noise
    is added on every read from per-column noise coefficients:
        I_o = v_read * dG * sum_i xn_i * u_oi,     y_o = I_o * x_fs * w_max_o / (v_read * dG)
    """

    def __init__(self, w, chip: Chip, x_fs, y_fs, key, name=""):
        self.name, self.chip = name, chip
        self.out_f, self.in_f = w.shape
        self.x_fs, self.y_fs = float(x_fs), float(y_fs)
        w = w.astype(mx.float32)
        self.n_weights = self.out_f * self.in_f
        if chip.ideal:
            self.w_eff, self.noise = w, None
            return
        d = chip.dev()
        g_max = chip.g_eff_max()
        g_min = 0.0 if math.isinf(d["on_off"]) else g_max / d["on_off"]
        dg = g_max - g_min
        k = split_keys(key, 10)
        self.chip = chip
        w_max = mx.maximum(mx.abs(w).max(axis=1, keepdims=True), 1e-12)        # (out, 1)
        u = w / w_max
        gp = g_min + mx.maximum(u, 0) * dg
        gn = g_min + mx.maximum(-u, 0) * dg
        if d["levels"] and chip.on("levels"):
            step = dg / (d["levels"] - 1)
            gp = g_min + mx.round((gp - g_min) / step) * step
            gn = g_min + mx.round((gn - g_min) / step) * step

        def program(g, kk):
            k1, k2, k3, k4, k5 = split_keys(kk, 5)
            if chip.on("prog"):
                g = g + d["prog_abs"] * g_max * mx.random.normal(g.shape, key=k1)
                if d["prog_rel"] is None:
                    # Pelgrom: sigma_C = a * sqrt(C). Relative to a full cell this is prog_rel(), so a cell
                    # holding the fraction g / g_max of the full capacitance gets sigma = prog_rel * g_max * sqrt(g / g_max)
                    g = g + chip.prog_rel() * g_max * mx.sqrt(mx.maximum(g, 0.0) / g_max) * mx.random.normal(g.shape, key=k2)
                else:
                    g = g * (1 + d["prog_rel"] * mx.random.normal(g.shape, key=k2))
            if d["stuck"] > 0 and chip.on("stuck"):
                r = mx.random.uniform(shape=g.shape, key=k3)
                g = mx.where(r < d["stuck"] / 2, g_min, mx.where(r > 1 - d["stuck"] / 2, g_max, g))
            g = mx.clip(g, 0.0, 1.2 * g_max)
            if d["nu"] > 0 and chip.drift_time > 1.0 and chip.on("drift"):
                nu = mx.maximum(d["nu"] + d["nu_std"] * mx.random.normal(g.shape, key=k4), 0.0)
                g = g * mx.power(chip.drift_time, -nu)
            dt = chip.temp - T_REF
            if dt != 0 and chip.on("temp"):
                g = g * (1 + d["tc_std"] * abs(dt) * mx.random.normal(g.shape, key=k5))
            return g

        gp, gn = program(gp, k[0]), program(gn, k[1])
        ac = chip.is_ac()
        if ac and chip.on("prog"):          # stray crossing capacitance: cancels in the pair except mismatch
            y_par = 2 * math.pi * chip.ac_freq * chip.c_par
            kp1, kp2 = mx.random.split(k[5])
            gp = gp + y_par * (1 + chip.c_par_mis * mx.random.normal(gp.shape, key=kp1))
            gn = gn + y_par * (1 + chip.c_par_mis * mx.random.normal(gn.shape, key=kp2))
        if d["nu"] > 0 and chip.drift_time > 1.0 and chip.drift_comp and chip.on("drift"):
            comp = chip.drift_time ** d["nu"]          # one calibrated gain for the whole array
            gp, gn = gp * comp, gn * comp
        # static IR drop (checked against the exact tile solve in circuit_tile.py).
        # Row wire (input i, 2n cells): it carries only this row's own cell currents, which scale with
        # this row's own voltage, so the relative loss at output position p does NOT depend on the data:
        #   a_row = R_seg * G_avg * (p (2n) - p^2/2) + R_drv * sum_row(G)
        # Column wire (n cells, ends in the virtual ground): it carries the currents of all rows, so its
        # loss relative to one cell scales with the mean input activity:
        #   a_col = R_seg * G_avg * act_ratio * (q n - q^2/2) at input position q
        n = chip.tile
        g_avg = float(((gp + gn) / 2).mean())
        p_row = (2 * (mx.arange(self.out_f) % n)).astype(mx.float32)   # output pair along the row wire
        p_col = (mx.arange(self.in_f) % n).astype(mx.float32)          # input row along the column wire
        a_row = p_row * (2 * n) - p_row ** 2 / 2
        a_col = p_col * n - p_col ** 2 / 2
        atten = (chip.r_wire * g_avg * a_row[:, None] + chip.r_drv * g_avg * 2 * n
                 + chip.r_wire * g_avg * chip.act_ratio * a_col[None, :])
        atten = (1 - mx.exp(-atten)) * (1 - chip.ir_comp) if chip.on("ir") else mx.zeros_like(atten)   # saturating fit
        if ac:
            # capacitive cells draw reactive current: the row wire acts as a distributed RC line. The
            # carrier reaching output position p lags by phi = omega * R * C_cell_avg * (p N - p^2/2)
            # (Elmore delay, independent of the data); the synchronous detector keeps cos(phi).
            # Column wires end in a virtual ground and the high-impedance cells act as current
            # sources, so their resistance is a second-order effect and is neglected.
            c_avg = g_avg / (2 * math.pi * chip.ac_freq)
            phi = 2 * math.pi * chip.ac_freq * chip.r_wire * c_avg * a_row[:, None] * (1 - chip.phase_comp)
            self.phi_max = float(phi.max())
            atten = (1 - mx.cos(phi)) if chip.on("ir") else mx.zeros_like(atten)
            atten = mx.broadcast_to(atten, (self.out_f, self.in_f))
        u_eff = (gp - gn) * (1 - atten) / dg
        self.w_eff = (u_eff * w_max).astype(mx.float32)
        # I-V nonlinearity in normalised input units: xn -> s * sinh(xn / s)
        self.v0n = None if d["v0"] is None or not chip.on("iv") else d["v0"] / chip.v_read
        # dynamic noise coefficients (per output column), in A^2
        bw = chip.bandwidth()
        gsum = (gp + gn).sum(axis=1)                                            # (out,)
        g2sum = (gp ** 2 + gn ** 2).sum(axis=1)
        ac = chip.is_ac()                               # capacitors: no Johnson or shot noise in the grid
        self.th = 0.0 * gsum if ac else 4 * K_B * chip.temp * bw * gsum
        self.sh = 0.0 * gsum if ac else 2 * Q_E * bw * chip.v_read * gsum        # x mean|xn|
        self.rd = (d["read_rel"] ** 2) * chip.v_read ** 2 * g2sum                # x mean(xn^2)
        self.tia = chip.i_noise() ** 2 * bw
        self.conv = (self.x_fs * w_max[:, 0]) / (chip.v_read * dg)               # A -> y units
        t_on = chip.on("tia")
        self.gain = 1 + t_on * chip.tia_gain_std * mx.random.normal((self.out_f,), key=k[2])
        off_scale = chip.ac_off_rej if ac else chip.offset_trim   # synchronous detection rejects DC offset
        self.off = t_on * chip.tia_off_std * off_scale * self.y_fs * mx.random.normal((self.out_f,), key=k[3])
        self.noise_on, self.rails_on = chip.on("noise"), chip.on("rails")
        self.noise = True
        self.phi_max = getattr(self, "phi_max", 0.0)
        self.key = k[4]
        mx.eval(self.w_eff, self.th, self.sh, self.rd, self.conv, self.gain, self.off, gsum)
        self.g_total = float(gsum.sum())

    def __call__(self, x):
        if self.noise is None:
            return x @ self.w_eff.T
        fs = self.x_fs
        if self.rails_on:
            x = mx.clip(x, -fs, fs)
        xn = x / fs
        if self.v0n is not None:
            xc = mx.clip(xn, -1.0, 1.0)
            safe = mx.where(mx.abs(xc) < 1e-6, 1.0, xc)
            xn = xn * mx.where(mx.abs(xc) < 1e-6, 1.0, self.v0n * mx.sinh(safe / self.v0n) / safe)
        y = (xn * fs) @ self.w_eff.T
        if self.noise_on:
            m1 = mx.abs(xn).mean(axis=-1, keepdims=True)
            m2 = (xn * xn).mean(axis=-1, keepdims=True)
            var = self.th + self.sh * m1 + self.rd * m2 + self.tia
            self.key, sub = mx.random.split(self.key)
            y = y + mx.sqrt(var) * self.conv * mx.random.normal(y.shape, key=sub)
        y = y * self.gain + self.off
        return mx.clip(y, -self.y_fs, self.y_fs) if self.rails_on else y

    def energy(self, m2):
        """Energy of one read (J): array dissipation + readout amplifiers."""
        if self.noise is None:
            return 0.0
        c = self.chip
        if c.is_ac():   # g_total is total admittance; C_total = Y / (2 pi f); C V^2 per cycle / q_res
            c_total = self.g_total / (2 * math.pi * c.ac_freq)
            amps = self.out_f * c.p_tia * c.t_stage()                 # no auto-zero needed
            return (c.t_int * c.ac_freq * c_total * c.v_read ** 2 * m2 / c.q_res + amps
                    + self.in_f * c.e_mod + self.out_f * c.e_demod)
        amps = self.out_f * c.p_tia * c.trim_overhead() * c.t_stage()
        return c.t_int * c.v_read ** 2 * m2 * self.g_total + amps

    def area_mm2(self):
        """Cell array (two cells per weight) + one read-out amplifier (~50 um^2) per output."""
        return (2 * self.n_weights * self.chip.cell_area_um2() + 50.0 * self.out_f) * 1e-6


class FunctionUnit:
    """Per-channel analog nonlinearity f(x) with static gain/offset mismatch and rails."""

    def __init__(self, fn, n, chip: Chip, x_fs, key):
        self.fn, self.n, self.x_fs = fn, n, float(x_fs)
        self.ideal = not chip.on("fu")
        if not self.ideal:
            k1, k2 = mx.random.split(key)
            self.gain = 1 + chip.fu_gain_std * mx.random.normal((n,), key=k1)
            self.off = chip.fu_off_std * chip.offset_trim * self.x_fs * mx.random.normal((n,), key=k2)

    def __call__(self, x):
        if self.ideal:
            return self.fn(x)
        x = mx.clip(x * self.gain + self.off, -self.x_fs, self.x_fs)
        return self.fn(x)


class Multiplier:
    """Gilbert-cell multiplier a * b per channel: tanh input compression, gain/offset mismatch."""

    def __init__(self, n, chip: Chip, a_fs, b_fs, key):
        self.ideal = not chip.on("mult")
        self.a_r, self.b_r = chip.mult_lin * float(a_fs), chip.mult_lin * float(b_fs)
        if not self.ideal:
            k1, k2 = mx.random.split(key)
            self.gain = 1 + chip.mult_gain_std * mx.random.normal((n,), key=k1)
            self.off = chip.mult_off_std * chip.offset_trim * mx.random.normal((n,), key=k2)
            self.out_fs = float(a_fs) * float(b_fs)

    def __call__(self, a, b):
        if self.ideal:
            return a * b
        a = self.a_r * mx.tanh(a / self.a_r)
        b = self.b_r * mx.tanh(b / self.b_r)
        return a * b * self.gain + self.off * self.out_fs


class NormUnit:
    """RMS norm: RMS detector + divider + one gain resistor per line."""

    def __init__(self, weight, eps, chip: Chip, key, y_fs=None):
        self.w, self.eps, self.ideal = weight.astype(mx.float32), eps, not chip.on("norm")
        self.y_fs = y_fs
        if not self.ideal:
            k1, k2, self.key = split_keys(key, 3)
            self.w = self.w * (1 + chip.norm_gain_std * mx.random.normal(self.w.shape, key=k1))
            self.dyn = chip.norm_dyn_std

    def __call__(self, x):
        if self.ideal:
            return mx.fast.rms_norm(x, self.w, self.eps)
        rms = mx.sqrt((x * x).mean(axis=-1, keepdims=True) + self.eps)
        self.key, sub = mx.random.split(self.key)
        rms = rms * (1 + self.dyn * mx.random.normal(rms.shape, key=sub))
        y = x / rms * self.w
        return y if self.y_fs is None else mx.clip(y, -self.y_fs, self.y_fs)


def adc(x, bits, fs):
    """Uniform quantiser over [-fs, fs] (ADC followed by an ideal DAC)."""
    levels = 2 ** (bits - 1) - 1
    return mx.round(mx.clip(x / fs, -1, 1) * levels) / levels * fs
