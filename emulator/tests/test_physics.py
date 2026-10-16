"""Unit tests for the device physics, the analog memory kernel and the design-space sampler.
They need MLX (Apple Silicon) but no model weights."""
import math
import random

import mlx.core as mx
import numpy as np
import pytest

from hw import EFFECTS, Chip, Crossbar
from memkernel import analog_gated_delta


@pytest.fixture(scope="module")
def delta_inputs():
    from mlx_lm.models.gated_delta import gated_delta_kernel, normalize_qk
    mx.random.seed(0)
    B, T, H, D = 1, 300, 16, 128
    q, k = normalize_qk(mx.random.normal((B, T, H, D)), mx.random.normal((B, T, H, D)), inv_scale=D ** -0.5, eps=1e-6)
    v = mx.random.normal((B, T, H, D))
    g = mx.random.uniform(0.8, 1.0, (B, T, H))
    beta = mx.random.uniform(0.0, 1.0, (B, T, H))
    s0 = mx.zeros((B, H, D, D))
    y_ref, s_ref = gated_delta_kernel(q, k, v, g, beta, s0, None)
    return q, k, v, g, beta, s0, y_ref, s_ref


def test_kernel_zero_noise_matches_reference(delta_inputs):
    q, k, v, g, beta, s0, y_ref, s_ref = delta_inputs
    H, D = q.shape[2], q.shape[3]
    y, s = analog_gated_delta(q, k, v, g, beta, s0, mx.zeros((H, D, D)), 0.0, 123)
    assert float(mx.abs(y - y_ref).max()) < 1e-5
    assert float(mx.abs(s - s_ref).max()) < 1e-5


def test_kernel_noise_is_seeded(delta_inputs):
    q, k, v, g, beta, s0, *_ = delta_inputs
    H, D = q.shape[2], q.shape[3]
    run = lambda seed: analog_gated_delta(q, k, v, g, beta, s0, mx.zeros((H, D, D)), 1e-3, seed)[0]
    assert float(mx.abs(run(123) - run(123)).max()) == 0.0
    assert float(mx.abs(run(123) - run(124)).max()) > 0.0


def test_kernel_noise_accumulates_like_a_random_walk(delta_inputs):
    q, k, v, g, beta, s0, *_ = delta_inputs
    T, H, D = q.shape[1], q.shape[2], q.shape[3]
    sigma = 1e-3
    _, s = analog_gated_delta(q, k, v, mx.ones_like(g), mx.zeros_like(beta), s0, mx.zeros((H, D, D)), sigma, 7)
    expected = sigma * math.sqrt(T)
    assert abs(float(s.std()) / expected - 1) < 0.05
    assert abs(float(s.mean())) < 5 * expected / math.sqrt(s.size)


def test_refresh_error_falls_with_more_bits(delta_inputs):
    q, k, v, g, beta, s0, y_ref, s_ref = delta_inputs
    H, D = q.shape[2], q.shape[3]
    fs = float(mx.abs(s_ref).max()) * 1.5
    errs = []
    for bits in (4, 8, 12, 16):
        y, _ = analog_gated_delta(q, k, v, g, beta, s0, mx.zeros((H, D, D)), 0.0, 5, fs=fs, bits=bits, every=1, stochastic=False)
        errs.append(float(mx.sqrt(((y - y_ref) ** 2).mean()) / mx.sqrt((y_ref ** 2).mean())))
    assert errs == sorted(errs, reverse=True)
    assert errs[-1] < 1e-3


def _qwen_cell_sigma(u_value, n=200_000, c_cell=10e-15):
    """Effective-weight error of the Qwen crossbar for cells of one value (only programming errors on)."""
    chip = Chip(device="cap_ac", c_cell=c_cell, ac_freq=100e6, v_read=0.5, t_int=100e-9, c_par=0.0,
                off=",".join(e for e in EFFECTS if e not in ("prog",)))
    cols = 1000
    w = mx.full((n // cols, cols), u_value)
    w[:, 0] = 1.0                                     # row scale: largest weight = one full cell
    xb = Crossbar(w, chip, 1.0, 1.0, mx.random.key(1))
    err = np.array(xb.w_eff[:, 1:] - w[:, 1:])
    return float(err.std())


def _tiny_cell_sigma(u_value, n=200_000):
    """Same quantity for the TinyStories CapLinear (stray capacitance, phase and amplifier errors off)."""
    import tiny_hw
    tiny_hw.hw.c_par, tiny_hw.hw.r_wire, tiny_hw.hw.amp_gain_std = 0.0, 0.0, 0.0
    lin = tiny_hw.CapLinear(1000, n // 1000)
    lin.u = mx.full(lin.u.shape, u_value)
    lin.g = mx.ones(lin.g.shape)
    lin.draw(mx.random.key(2))
    return float(np.array(lin.weight(True) - lin.weight(False)).std())


@pytest.mark.parametrize("units", [1, 16, 255])
def test_capacitor_mismatch_matches_between_implementations(units):
    """Both implementations must follow Pelgrom's law: sigma_C = 0.4 % * sqrt(1 fF * C)."""
    u = units / 255
    expected = 0.004 * math.sqrt(1e-15 / 10e-15) * math.sqrt(u)          # in units of a full 10 fF cell
    q, t = _qwen_cell_sigma(u), _tiny_cell_sigma(u)
    assert abs(q / expected - 1) < 0.1, f"Qwen crossbar sigma {q:.3e}, Pelgrom {expected:.3e}"
    assert abs(t / expected - 1) < 0.1, f"TinyStories sigma {t:.3e}, Pelgrom {expected:.3e}"


def test_search_sampler_handles_ac_devices():
    from run import sample
    seen_ac = 0
    for seed in range(400):
        knobs = sample(random.Random(seed))
        chip = Chip(**knobs).validate()
        if chip.is_ac():
            seen_ac += 1
            assert chip.t_int * chip.ac_freq >= 3
    assert seen_ac >= 20


def test_search_mutation_handles_ac_devices():
    from run import sample
    base = sample(random.Random(0), {"device": "cap_ac", "g_max": 1.0, "v_read": 0.5, "t_int": 100e-9, "tile": 128,
                                     "p_tia": 4e-6, "memory": "sram", "c_mem": 1e-14, "adc_bits": 8, "ckpt_every": 0,
                                     "fs_pct": 100.0, "offset_trim": 0.1, "v_mem": 1.0, "c_cell": 1e-14, "ac_freq": 1e8}, n_mut=1)
    for seed in range(200):
        knobs = sample(random.Random(seed), base)
        Chip(**knobs).validate()


def test_calibration_max_is_the_true_maximum():
    """Rails sized to 'the largest signal seen' must use the true peak, not the peak of a random subsample."""
    from sim import Recorder
    rec = Recorder(per_node=1000)
    x = mx.random.normal((1, 64, 4096))
    x[0, 7, 123] = 50.0                                   # one rare peak among 262,144 values
    rec("node", x)
    assert rec.stats()["node"]["max"] == 50.0
