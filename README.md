# KirchhoffLM

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**A modern language model translated into analog chip hardware, and run in a physics emulator.**

Every weight of [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) becomes a pair of capacitors. Every matrix product
is done by sine-wave currents adding up on wires, which is Kirchhoff's current law doing the arithmetic. The model runs through an
emulator of 24 layer chips that includes the hardware's errors: capacitor mismatch, stray capacitance, wire delay, amplifier
noise and offsets, clipping, and converter rounding.

**Website:** https://unago-ai.github.io/KirchhoffLM/: how it works, chip design diagrams, the assumptions, results, and a
step-by-step *Signal journey* that follows one word through the chips.

> This is a simulation study. No chip has been built, and the device values are literature-range assumptions. Real silicon usually
> lands well below simulated efficiency.

## Results (Qwen3.5-0.8B)

| | |
|---|---|
| Translation to component values (8-bit capacitor counts) | lossless: 0.7398 vs 0.7403 bits/byte |
| Quality on emulated chips | **+2.2 – 2.5 %** bits/byte vs the original |
| Long context (~980 tokens) | 20 / 20 recall of a hidden fact, the same as the exact model |
| Emulated speed / energy | ~37,000 tokens/s per user, ~5–15 µJ per token (estimates) |
| Hardware | 747M capacitors, 648k read-out amplifiers, ~8,000 mm² over 25 chips |

Other findings:
- AC capacitor grids are far more robust than DC resistor grids, which suffer from wire voltage drop. This was checked
  with exact circuit solves of real tiles.
- An analog DeltaNet memory fails, because thermal noise piles up on every write, so the memory stays digital.
- With trimming, the remaining loss comes from converters and multipliers, not from the capacitor grids.

## Design in one picture

```
token → embedding → [chip L0] → [chip L1] → … → [chip L23] → output grid + ADC → next token
                     each chip: norm → capacitor grids (AC, 10 fF, 100 MHz) → demodulators → SiLU / multipliers
                                → digital memory (DeltaNet state or KV cache) → 10-bit link to the next chip
```

## Repository

| Path | What |
|---|---|
| `emulator/hw.py` | device physics: capacitor/resistor/memristor grids, amplifiers, converters, error models |
| `emulator/sim.py` | Qwen3.5 forward pass built from circuit blocks; incremental mode (per-chip state) |
| `emulator/memkernel.py` | Metal kernel for an analog DeltaNet memory with in-kernel noise and refresh |
| `emulator/circuit_tile.py` | exact nodal circuit solve (DC and AC phasor) of one tile, used to validate the fast model |
| `emulator/export_hw.py` | translate the model into a hardware design package (capacitor counts, resistor values, SPICE tile) |
| `emulator/emulate_hw.py` | run the model from the design package only |
| `emulator/chat_server.py` | Gradio chat UI with a live processing log, the Signal journey and the documentation pages |
| `emulator/run.py`, `layer_bench.py`, `ac_dc.py`, … | experiments: sweeps, design search, per-layer chips, AC vs DC |
| `emulator/tiny_hw.py` | TinyStories model with hardware-parameter training (exploratory) |
| `emulator/runs/` | result files used by the website |
| `docs/` | the GitHub Pages site (built by `emulator/build_site.py`) |

## Run it (Apple Silicon, MLX)

```bash
cd emulator
pip install -r requirements.txt
python make_eval_texts.py                         # held-out Wikipedia + code texts
python run.py calibrate && python run.py validate  # signal ranges; exact match with physics off
python export_hw.py                               # -> hw_qwen35/ (about 1.4 GB)
python emulate_hw.py --chips 0,1                  # quality on two emulated chips + sample text
python chat_server.py --port 7860                 # chat UI + docs
```

`KLM_MODEL` selects the model (a Hugging Face id or a local directory), and `KLM_PKG` the design package location.

## Related work

[KirchhoffNet](https://arxiv.org/abs/2310.15872) (MIT/CMU) proposes new continuous-time neural networks whose dynamics *are*
a circuit; it was evaluated on small tasks and does not model non-idealities. KirchhoffLM takes the opposite route. It keeps an
existing LLM unchanged, maps it onto realistic hardware, and measures the cost of the hardware's errors.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Copyright 2026 UnaGo.
The Qwen3.5 model is licensed separately by its authors; no model weights are included here.

---
Built by UnaGo.
