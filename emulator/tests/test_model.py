"""Integration tests on Qwen3.5-0.8B. They need the hardware design package (KLM_PKG, built by export_hw.py)
and, for the digital comparison, the model itself (KLM_MODEL). They are skipped when those are missing."""
import json
import os

import mlx.core as mx
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.environ.get("KLM_PKG", os.path.join(ROOT, "hw_qwen35"))
needs_pkg = pytest.mark.skipif(not os.path.exists(os.path.join(PKG, "manifest.json")), reason="no hardware package")


@pytest.fixture(scope="module")
def pkg():
    from emulate_hw import load_package
    return load_package(PKG)


@pytest.fixture(scope="module")
def ids(pkg):
    _, tok, _, _ = pkg
    return tok.encode("The quick brown fox jumps over the lazy dog. Then it ran to the river and drank some water.")


class Rec:
    def __init__(self):
        self.data = {}

    def __call__(self, name, x):
        self.data[name] = np.array(x.astype(mx.float32))
        return x


def _on_grid(x, fs, bits):
    levels = 2 ** (bits - 1) - 1
    v = np.clip(x / fs, -1, 1) * levels
    return float(np.abs(v - np.round(v)).max())


@needs_pkg
def test_digital_memory_boundaries_are_converted(pkg, ids):
    """With digital memory, signals must pass an ADC into the memory and a DAC out of it."""
    from hw import EFFECTS, Chip
    from sim import AnalogQwen, fs
    model, tok, man, cal = pkg
    design = dict(man["design"])
    chip = Chip(**{**design, "link_bits": 0, "ckpt_every": 0, "off": ",".join(e for e in EFFECTS if e != "adc")})
    assert chip.mem_digital()
    m = AnalogQwen(model, chip, cal)
    rec = Rec()
    mx.eval(m(mx.array(ids)[None], rec=rec))
    pct = chip.fs_pct
    head_fs = np.repeat([fs(cal, f"L0.mem_in.g{j}", pct) for j in range(48)], 128)
    assert "L0.mem_in" in rec.data and "L3.mem_in" in rec.data, "no converter recorded at the memory inputs"
    assert _on_grid(rec.data["L0.mem_in"], head_fs, chip.mem_adc_bits) < 1e-3       # ADC into DeltaNet memory (per head)
    assert _on_grid(rec.data["L3.mem_in"], fs(cal, "L3.mem_in", pct), chip.mem_adc_bits) < 1e-3   # ADC into attention
    y_fs = np.array([fs(cal, f"L0.gdn_core.h{h}", pct) for h in range(16)])[:, None]
    assert _on_grid(rec.data["L0.gdn_core"][0], y_fs, chip.mem_dac_bits) < 1e-3     # DAC out of DeltaNet memory
    assert _on_grid(rec.data["L3.attn_o"], fs(cal, "L3.attn_o", pct), chip.mem_dac_bits) < 1e-3   # DAC out of attention


@needs_pkg
def test_incremental_equals_full_recompute(pkg, ids):
    from hw import Chip
    from sim import AnalogQwen
    model, _, _, cal = pkg
    m = AnalogQwen(model, Chip(ideal=True), cal)
    full = m(mx.array(ids)[None])[0]
    cache = m.make_cache()
    parts = [m(mx.array(ids[:10])[None], cache=cache)[0]]
    for t in ids[10:]:
        parts.append(m(mx.array([[t]]), cache=cache)[0])
        mx.eval(parts[-1], cache)
    inc = mx.concatenate(parts, axis=0)
    assert float(mx.abs(full - inc).max()) < 1e-3


@pytest.mark.skipif(not os.environ.get("KLM_MODEL"), reason="KLM_MODEL not set")
def test_ideal_analog_model_equals_digital_model(ids):
    from hw import Chip
    from sim import AnalogQwen, load_digital
    digital, _ = load_digital()
    digital.set_dtype(mx.float32)
    m = AnalogQwen(digital, Chip(ideal=True), None)
    x = mx.array(ids)[None]
    assert float(mx.abs(digital(x) - m(x)).max()) < 1e-3
