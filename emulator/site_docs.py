"""Documentation pages for the chat website: how the emulation works, the chip design (with SVG
diagrams), the hardware and noise assumptions, and the measured quality.

Numbers come from the hardware package manifest, the device tables in hw.py and the result files in
runs/ where those exist; results that were only printed during the experiments are quoted with
their measurement date."""
import html
import json
import os

from hw import DEVICES, MEMORIES, Chip

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "runs")

TEAL, CORAL, GRAY, BLUE = "#1D9E75", "#D85A30", "#888780", "#378ADD"
CSS = """
<style>
.hwdoc { line-height: 1.6; max-width: 1100px; }
.hwdoc h2 { margin-top: 1.4em; }
.hwdoc table { border-collapse: collapse; margin: 0.6em 0 1.2em; font-size: 0.92em; }
.hwdoc th, .hwdoc td { border: 1px solid var(--border-color-primary); padding: 4px 10px; text-align: left; vertical-align: top; }
.hwdoc .note { color: var(--body-text-color-subdued); font-size: 0.9em; }
.hwdoc svg text { fill: var(--body-text-color); font-family: var(--font); }
.hwdoc .legend span { display: inline-block; margin-right: 18px; }
.hwdoc .sw { display: inline-block; width: 12px; height: 12px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }
</style>
"""


def _jsonl(name):
    p = os.path.join(RUNS, name)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def _table(head, rows):
    h = "".join(f"<th>{html.escape(str(c))}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><tr>{h}</tr>{b}</table>"


def _legend():
    return (f'<div class="legend"><span><i class="sw" style="background:{TEAL}"></i>analog weight grid (capacitors)</span>'
            f'<span><i class="sw" style="background:{GRAY}"></i>analog circuit (norm, SiLU, multiplier, hold)</span>'
            f'<span><i class="sw" style="background:{CORAL}"></i>digital (memory, attention, converters)</span></div>')


def _box(x, y, w, h, color, title, sub=None):
    t = (f'<text x="{x + w / 2}" y="{y + (h / 2 - 7 if sub else h / 2)}" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="500">{html.escape(title)}</text>')
    if sub:
        t += f'<text x="{x + w / 2}" y="{y + h / 2 + 10}" text-anchor="middle" dominant-baseline="central" font-size="11" opacity="0.75">{html.escape(sub)}</text>'
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{color}" fill-opacity="0.18" '
            f'stroke="{color}" stroke-width="1"/>{t}')


def _arrow(x1, y1, x2, y2):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="currentColor" stroke-opacity="0.6" stroke-width="1.2" '
            f'marker-end="url(#ah)"/>')


DEFS = ('<defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path d="M2 1L8 5L2 9" fill="none" stroke="currentColor" stroke-width="1.5"/></marker></defs>')


def system_svg(man):
    n = len([k for k in man["chips"] if k.startswith("L")])
    parts = [DEFS, _box(10, 95, 90, 50, GRAY, "Token in"),
             _box(120, 85, 120, 70, CORAL, "Embedding", "row lookup"), _arrow(100, 120, 118, 120), _arrow(240, 120, 262, 120)]
    for i in range(n):
        r, c = divmod(i, 12)
        x, y = 265 + c * 46, 70 + r * 66
        attn = (i + 1) % 4 == 0
        parts.append(_box(x, y, 40, 46, CORAL if attn else TEAL, f"L{i}", "attn" if attn else "Δnet"))
    parts += [_arrow(817, 120, 838, 120), _box(840, 85, 130, 70, TEAL, "Output grid", "+ 8-bit ADC"),
              _arrow(905, 155, 905, 182), _box(850, 184, 110, 40, GRAY, "Next token"),
              '<text x="535" y="214" text-anchor="middle" font-size="12" opacity="0.8">24 layer chips · 10-bit digital link between chips · '
              'each chip keeps its own memory (DeltaNet state or KV cache)</text>']
    return f'<svg viewBox="0 0 980 235" width="100%" role="img" style="color:var(--body-text-color)"><title>Chip chain</title>{"".join(parts)}</svg>'


def layer_svg():
    p = [DEFS]
    row1 = [("Norm unit", "RMS + divider", GRAY), ("Projection grids", "q k v z a b", TEAL), ("Short conv", "sample & hold", GRAY),
            ("DeltaNet memory", "digital SRAM", CORAL), ("Gated norm", "× silu(z)", GRAY), ("Output grid", "→ + residual", TEAL)]
    row2 = [("Norm unit", "RMS + divider", GRAY), ("FFN grids", "gate, up", TEAL), ("SiLU × mult.", "Gilbert cell", GRAY),
            ("Down grid", "→ + residual", TEAL), ("Chip link", "10-bit ADC → DAC", CORAL)]
    for row, y in ((row1, 30), (row2, 150)):
        for j, (t, s, c) in enumerate(row):
            x = 10 + j * 160
            p.append(_box(x, y, 138, 58, c, t, s))
            if j:
                p.append(_arrow(x - 22, y + 29, x - 2, y + 29))
    p.append('<path d="M880 88 L880 118 L79 118 L79 148" fill="none" stroke="currentColor" stroke-opacity="0.6" stroke-width="1.2" marker-end="url(#ah)"/>')
    p.append('<text x="10" y="20" font-size="12" opacity="0.8">DeltaNet layer chip (18 of 24); attention chips replace the memory block with a digital KV cache + softmax</text>')
    return f'<svg viewBox="0 0 980 225" width="100%" role="img" style="color:var(--body-text-color)"><title>Inside a layer chip</title>{"".join(p)}</svg>'


def tile_svg(d):
    p = [DEFS]
    rows, pairs = 4, 3
    x0, y0, dx, dy = 150, 40, 90, 52
    for i in range(rows):
        y = y0 + i * dy
        p.append(f'<circle cx="70" cy="{y}" r="14" fill="none" stroke="{BLUE}"/><path d="M60 {y} q5 -9 10 0 t10 0" fill="none" stroke="{BLUE}"/>')
        p.append(f'<text x="30" y="{y + 4}" text-anchor="end" font-size="11">V{i}</text>')
        p.append(f'<line x1="84" y1="{y}" x2="{x0 + pairs * 2 * dx - 40}" y2="{y}" stroke="currentColor" stroke-opacity="0.55"/>')
    for k in range(pairs * 2):
        x = x0 + k * dx
        p.append(f'<line x1="{x}" y1="{y0 - 18}" x2="{x}" y2="{y0 + rows * dy + 10}" stroke="currentColor" stroke-opacity="0.55"/>')
        for i in range(rows):
            y = y0 + i * dy
            p.append(f'<line x1="{x - 7}" y1="{y + 12}" x2="{x + 7}" y2="{y + 12}" stroke="{TEAL}" stroke-width="2"/>'
                     f'<line x1="{x - 7}" y1="{y + 17}" x2="{x + 7}" y2="{y + 17}" stroke="{TEAL}" stroke-width="2"/>')
        p.append(f'<text x="{x}" y="{y0 - 24}" text-anchor="middle" font-size="11">{"C+" if k % 2 == 0 else "C−"}</text>')
    yb = y0 + rows * dy + 10
    for o in range(pairs):
        xa, xb = x0 + 2 * o * dx, x0 + (2 * o + 1) * dx
        xm = (xa + xb) / 2
        p.append(f'<path d="M{xa} {yb} L{xa} {yb + 15}" fill="none" stroke="currentColor" stroke-opacity="0.55"/>'
                 f'<path d="M{xb} {yb} L{xb} {yb + 15}" fill="none" stroke="currentColor" stroke-opacity="0.55"/>')
        p.append(_box(xa - 6, yb + 2, xb - xa + 12, 26, GRAY, "sync demod"))
        p.append(_arrow(xm, yb + 28, xm, yb + 44))
        p.append(f'<text x="{xm}" y="{yb + 58}" text-anchor="middle" font-size="12">y{o}</text>')
    tx = 720
    notes = [f"Inputs: sine drive V·sin(ωt), {d['v_read']} V at {d['ac_freq'] / 1e6:g} MHz",
             "Cells: capacitor pair C+ / C− per weight",
             f"  0–255 unit capacitors, max {d['c_cell'] * 1e15:g} fF",
             "Column current I = jω Σ C·V  (Kirchhoff sum)",
             "Synchronous demodulator keeps the in-phase",
             "  part; per-column phase is calibrated",
             "Weight: w = (C+ − C−) / 255 × gain",
             f"Tile: {d['tile']} × {d['tile']} cell pairs"]
    for j, t in enumerate(notes):
        p.append(f'<text x="{tx}" y="{40 + j * 22}" font-size="12">{html.escape(t)}</text>')
    return f'<svg viewBox="0 0 1000 330" width="100%" role="img" style="color:var(--body-text-color)"><title>AC capacitor grid tile</title>{"".join(p)}</svg>'


def bars_svg(items, unit="%", title="", width=900, color=TEAL):
    """Horizontal bar chart: items = [(label, value)]."""
    vmax = max(v for _, v in items) or 1
    h = 26 * len(items) + 30
    p = [f'<text x="0" y="14" font-size="12" opacity="0.8">{html.escape(title)}</text>']
    for j, (label, v) in enumerate(items):
        y = 26 + j * 26
        w = max(2, (width - 330) * v / vmax)
        p.append(f'<text x="200" y="{y + 12}" text-anchor="end" font-size="12">{html.escape(label)}</text>'
                 f'<rect x="210" y="{y + 2}" width="{w}" height="18" rx="3" fill="{color}" fill-opacity="0.75"/>'
                 f'<text x="{216 + w}" y="{y + 12}" dominant-baseline="central" font-size="12">{v:+.2f}{unit}</text>')
    return f'<svg viewBox="0 0 {width} {h}" width="100%" role="img" style="color:var(--body-text-color)">{"".join(p)}</svg>'


def how_it_works_html(man):
    t = man["totals"]
    return CSS + f"""<div class="hwdoc">
<h2>What you are talking to</h2>
<p>The chat model is <b>Qwen3.5-0.8B</b>, translated into a <b>hardware design</b>: every weight became a pair of capacitors
(counted in unit capacitors), every read-out amplifier got a feedback-resistor value, every norm a set of gain resistors.
The website loads only that design package — {t['capacitor_cells']:,} capacitors made of {t['unit_capacitors']:,} unit capacitors,
{t['amplifiers']:,} amplifiers — and runs it through an <b>emulator of the chips</b>.</p>
<h2>What the emulator does for every token</h2>
<ol>
<li><b>Weight grids.</b> The input vector becomes sine-wave voltages on the rows of each capacitor grid. The column currents add up by
Kirchhoff's law, which <i>is</i> the matrix–vector product. The emulator computes the same sum, then applies the physics:
capacitor mismatch, stray capacitance, wire phase delay (after calibration), amplifier gain/offset error and noise, and clipping at the rails.</li>
<li><b>Analog circuits.</b> Norm units, SiLU/sigmoid transistor stages and Gilbert-cell multipliers get their own mismatch, offsets and limits.</li>
<li><b>Digital parts.</b> The memory of earlier tokens (DeltaNet state, attention KV cache) is digital SRAM; ADCs and DACs at its boundary
round the values. Chips talk through 10-bit digital links.</li>
<li><b>Incremental operation.</b> Like the real chips, each layer chip keeps its state between tokens; every new token makes one pass
through the 24 chips.</li>
<li><b>Manufacturing.</b> The "chip number" selects one random draw of the fixed manufacturing errors — different numbers are different
physical chips with the same design.</li>
</ol>
<h2>Two kinds of time</h2>
<p>The processing log shows both. <b>Emulator time</b> is how long the Mac takes to simulate the physics (~30 tokens/s).
<b>Emulated hardware time</b> is how long the real chips would take according to the circuit timing model (~37,000 tokens/s,
~1.1 µs per chip per token). The emulator is slower than the hardware it emulates.</p>
<h2>How the emulator was checked</h2>
<ul>
<li>With all physics off it reproduces the digital model exactly (logit difference ~5×10⁻⁵).</li>
<li>Incremental mode equals full recompute (exactly with physics off, statistically with physics on).</li>
<li>The fast weight-grid model was checked against an exact circuit solve of real 128×128 tiles (every wire segment, cell and stray
capacitor as a node, solved like SPICE for linear circuits) — the AC design agrees within ~0.3 % per tile.</li>
</ul>
</div>"""


ELEMENTS = [
    ("Embedding table", "Turns the token number into 1,024 numbers that describe the word", "Memory, one row read per token",
     "The starting meaning of the word", "Memory size (254M values for Qwen3.5-0.8B)"),
    ("Norm unit", "Rescales the 1,024 signals to a fixed average size", "RMS detector + divider + one gain resistor per line",
     "Keeps signals inside the useful range: not clipped, not lost in noise", "~0.5 % gain mismatch, 0.2 % RMS error"),
    ("DAC / row driver", "Turns a digital number into a voltage", "Digital-to-analog converter per line (at the chip input)",
     "The bridge from bits to the analog world", "Rounding (10 bits), energy per conversion"),
    ("Carrier + modulator", "Makes a 100 MHz sine wave whose height is the signal (flipped wave = negative)",
     "Oscillator shared by the chip, a mixer per row", "Lets capacitors compute (they only pass changing signals) and lets the read-out reject offsets",
     "~1 mW oscillator per chip, ~10 fJ per row per read"),
    ("Capacitor grid", "Multiplies every input by every weight and adds them up", "A pair of capacitors per weight (C+ / C−), 0–255 unit capacitors each",
     "The core: a whole matrix product in one step, weights never move", "Mismatch ~0.13 % at 10 fF, stray capacitance, ~5 µm² per cell"),
    ("Row and column wires", "Carry the waves in and the summed currents out", "Thin metal lines across the grid",
     "The addition itself (Kirchhoff: currents into a wire add up)", "Resistance delays the wave along the row; calibrated per column"),
    ("Synchronous demodulator", "Turns the summed wave current back into a steady value", "Mixer driven by the same carrier + averaging",
     "Keeps only the part in step with the carrier: offsets and slow noise are rejected", "~10 fJ per output per read"),
    ("Read-out amplifier + feedback resistor", "Turns the small current into a voltage with the right scale",
     "Transimpedance amplifier per column; the resistor value is part of the design", "One scale per output column (the row's largest weight)",
     "Noise (the main random error), 0.5 % gain mismatch, ~3.5 µW each"),
    ("Rails", "The highest voltage a circuit can produce", "Supply limits of every amplifier", "Nothing; it is a limit to design around",
     "Signals beyond it are clipped; full scale is set to the largest signal seen"),
    ("SiLU / sigmoid unit", "The model's 'decision' curve: passes positive signals, mostly blocks negative ones", "Transistors whose current curve has that shape",
     "Non-linearity: without it the whole model would be one big matrix", "~1 % gain mismatch, trimmed offsets"),
    ("Gilbert multiplier", "Multiplies two signals (one gates the other)", "Classic analog multiplier cell",
     "Gating: lets the model switch information on and off", "~1 % gain mismatch, slight compression of large signals"),
    ("Short conv (sample & hold)", "Remembers the last 3 inputs for a tiny local mix", "Capacitors that hold earlier values + resistor taps",
     "Local word-order information", "kT/C noise on the held values (very small)"),
    ("ADC to memory / DAC back", "Converts signals to bits for the memory and back", "Converters at the memory boundary",
     "Lets a precise digital memory work with the analog grids", "Energy per conversion"),
    ("DeltaNet memory (18 chips)", "Remembers earlier words in a fixed-size table (16 × 128 × 128 numbers)", "SRAM + digital multiply-add units",
     "Memory of the whole text without growing", "Most of the energy; analog versions failed (thermal noise piles up)"),
    ("Attention unit (6 chips)", "Looks back at every earlier word to find the exact ones that matter", "KV cache in SRAM/DRAM + digital softmax",
     "Exact recall (e.g. a number mentioned earlier)", "Energy and memory grow with the text length"),
    ("Gated norm", "Rescales each memory head and gates it with a second signal", "Norm unit + SiLU unit + Gilbert multiplier",
     "Keeps the memory output well scaled", "Same errors as its parts"),
    ("Residual capacitors", "Hold the main 1,024 signals and add each block's result", "One capacitor node per line",
     "Free addition: current in = charge added", "Negligible"),
    ("Chip-to-chip link", "Moves the 1,024 signals to the next layer chip", "10-bit ADC → serial link (64 lanes) → DAC",
     "Separate chips per layer; bits clean away accumulated analog noise", "~20 nJ per hop; 8-bit links cost +2.5 % quality"),
    ("Output grid + ADC", "Scores all 248,320 possible next tokens", "Capacitor grid + 8-bit ADC referenced to the top score",
     "The prediction", "Largest single block (~2,700 mm²); a winner-take-all circuit could pick the winner directly"),
    ("Calibration (after manufacturing)", "Measures and removes predictable errors", "Per-column phase reference, offset auto-zero, rail settings",
     "Turns big systematic errors into small random ones", "Test time per chip; must be repeated if conditions change"),
]


def chip_design_html(man):
    d = man["design"]
    rows = [[k, f"{c['weights']:,}", f"{c['capacitor_cells']:,}", f"{c['amplifiers']:,}", f"{c['area_mm2']:,}"] for k, c in man["chips"].items()]
    return CSS + f"""<div class="hwdoc">
<h2>System: one chip per layer</h2>{_legend()}{system_svg(man)}
<p>Each layer runs on its own chip; 18 chips hold DeltaNet layers, 6 hold attention layers. The embedding is a row lookup; the output
layer is a capacitor grid read by an 8-bit ADC referenced to the peak logit (a winner-take-all circuit would suffice for greedy decoding).</p>
<h2>Inside a layer chip</h2>{_legend()}{layer_svg()}
<p>Weights never move: each matrix is a fixed grid. Signals stay analog inside the chip and become digital only at the memory blocks and
the chip-to-chip link. DeltaNet memory is digital because an analog capacitor memory failed in simulation: every token writes every cell,
so thermal noise piles up faster than any practical capacitor can hold it.</p>
<h2>Every element and what it adds</h2>
{_table(["element", "what it does", "built from", "what it adds", "errors / cost"], [list(r) for r in ELEMENTS])}
<h2>The AC capacitor grid</h2>{tile_svg(d)}
<p>Capacitors instead of resistors: capacitors add no thermal (Johnson) noise, match better (~0.1 %), and their reactive current
causes almost no wire voltage drop. Synchronous demodulation also rejects amplifier offsets and 1/f noise. The cost is a carrier
generator, modulators and demodulators, and area (~5 µm² per 10 fF capacitor).</p>
<h2>Component counts per chip</h2>
{_table(["chip", "weights", "capacitors", "amplifiers", "area mm²"], rows)}
<p class="note">Area: two capacitor cells per weight (C/2 fF/µm² + 0.3 µm² overhead) plus ~50 µm² per amplifier. Converters, digital
memory, control and I/O are not included.</p>
</div>"""


ASSUMPTION_TEXT = {
    "c_cell": ("Largest cell capacitance", "F", "metal-oxide-metal capacitor, ~2 fF/µm²"),
    "ac_freq": ("Carrier frequency", "Hz", "sets admittance ωC and read time"),
    "v_read": ("Drive amplitude", "V", ""),
    "t_int": ("Integration time per read", "s", "≥ 3 carrier cycles"),
    "tile": ("Tile size", "cells", "128 × 128 keeps wire phase delay small"),
    "r_wire": ("Wire resistance per cell segment", "Ω", "dense crossbars: 1–5 Ω"),
    "phase_comp": ("Wire phase delay removed by calibration", "", "per-column reference phase"),
    "p_tia": ("Read-out amplifier power", "W", "noise scales with 1/√P"),
    "i_n0": ("Amplifier input noise at 10 µW", "A/√Hz", ""),
    "tia_gain_std": ("Amplifier gain mismatch", "", ""),
    "ac_off_rej": ("Offset left after synchronous detection", "", "fraction of raw offset"),
    "offset_trim": ("Offset left after trimming (baseband circuits)", "", "auto-zero"),
    "fu_gain_std": ("SiLU / sigmoid unit gain mismatch", "", ""),
    "mult_gain_std": ("Gilbert multiplier gain mismatch", "", ""),
    "norm_gain_std": ("Norm unit gain mismatch", "", ""),
    "c_par": ("Stray capacitance per crossing", "F", "cancels in the pair except mismatch"),
    "c_par_mis": ("Stray capacitance mismatch", "", ""),
    "adc_bits": ("ADC bits inside the chips", "bits", ""),
    "link_bits": ("Chip-to-chip link bits", "bits", ""),
    "fs_pct": ("Rail = percentile of calibrated |signal|", "%", "100 = largest signal seen"),
    "memory": ("Memory technology", "", "digital SRAM"),
    "e_dmac": ("Digital multiply-accumulate energy", "J", "incl. SRAM access"),
    "adc_fom": ("ADC energy per conversion step", "J", "Walden figure of merit"),
}


def si(v, unit):
    """Engineering notation: 1e-14 F -> 10 fF, 3.54e-06 W -> 3.54 µW."""
    if not unit or unit in ("cells", "bits", "%", "Ω") or v == 0:
        return f"{v:g} {unit}".strip()
    for exp, pre in ((12, "T"), (9, "G"), (6, "M"), (3, "k"), (0, ""), (-3, "m"), (-6, "µ"), (-9, "n"), (-12, "p"), (-15, "f"), (-18, "a")):
        if abs(v) >= 10 ** exp:
            return f"{v / 10 ** exp:.3g} {pre}{unit}"
    return f"{v:g} {unit}"


def assumptions_html(man):
    d = man["design"]
    rows = []
    for k, (name, unit, note) in ASSUMPTION_TEXT.items():
        v = d.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            vs = si(v, unit) if unit else f"{v:g}"
        else:
            vs = f"{v} {unit}".strip()
        rows.append([name, vs, note])
    dev = DEVICES["cap_ac"]
    cap = Chip(**{**d})
    return CSS + f"""<div class="hwdoc">
<h2>Status of these numbers</h2>
<p>The device values are <b>literature-range assumptions</b> for a first exploration, not measurements of a specific fab process.
Real silicon often lands 10–100× worse than simulation on energy because of control, clocking, leakage and interfaces that are not modelled.</p>
<h2>Design parameters used by this chip</h2>
{_table(["parameter", "value", "note"], rows)}
<h2>Capacitor cell</h2>
{_table(["property", "value"], [
        ["Unit capacitor", f"{man['units']['c_unit_F'] * 1e18:.1f} aF (255 per full cell)"],
        ["Mismatch (Pelgrom)", f"σC/C = 0.4 % × √(1 fF / C) → {100 * cap.prog_rel():.2f} % at {d['c_cell'] * 1e15:g} fF"],
        ["Carrier phase noise", f"{dev['read_rel']:g} rad"],
        ["Thermal noise in the grid", "none (capacitors are noiseless; noise comes from the amplifiers)"],
        ["Cell area", f"{cap.cell_area_um2():.1f} µm² per cell (two cells per weight)"]])}
<h2>Error sources and how each is modelled</h2>
{_table(["effect", "model", "type"], [
        ["Capacitor mismatch", "random per cell, size from Pelgrom's law", "fixed per chip"],
        ["Unit-capacitor rounding", "weights rounded to 255 levels per row", "fixed"],
        ["Stray crossing capacitance", "0.05 fF per crossing, 10 % mismatch between the pair", "fixed per chip"],
        ["Wire phase delay", "distributed RC (Elmore) along row wires; detector keeps cos φ; 90 % calibrated", "fixed"],
        ["Amplifier noise", "input current noise × √bandwidth, per read", "random per read"],
        ["Amplifier gain / offset", "0.5 % gain; offset 0.2 % of range × 2 % left after demodulation", "fixed per chip"],
        ["Rails", "inputs and outputs clip at the calibrated full scale", "deterministic"],
        ["SiLU / sigmoid / multipliers / norms", "1 % gain, small trimmed offsets, tanh compression in multipliers", "fixed per chip"],
        ["ADCs and links", "uniform rounding (7-bit inside chips, 10-bit links, 8-bit output referenced to the peak)", "deterministic"],
        ["Digital memory", "exact (SRAM)", "—"]])}
<h2>Not modelled</h2>
<ul><li>Leakage power of ~8,000 mm² of silicon, clock distribution, control logic, host I/O, cooling.</li>
<li>Defects and redundancy repair; temperature gradients across a chip; supply noise; long-term ageing.</li>
<li>Crosstalk between neighbouring wires along their full length is only partly modelled (it was included in the exact tile solves).</li>
<li>The digital attention engine is assumed fast enough to keep up with ~1.1 µs per token per chip.</li></ul>
</div>"""


def results_html(man):
    ref = man["reference_bits_per_byte"]["mean"]
    lb = [r for r in _jsonl("layer_bench.jsonl") if r["design"] == "AC"]
    layer_bars = bars_svg([(f"L{r['layer']} {r['kind']}", r["loss_pct"]) for r in lb],
                          title="Quality loss when only this layer runs on its chip (AC design, all other layers exact)") if lb else ""
    tiles = [r for r in json.load(open(os.path.join(RUNS, "circuit_tile.json")))] if os.path.exists(os.path.join(RUNS, "circuit_tile.json")) else []
    tile_rows = [[r["config"], r["tile"], f"{r['r_seg']:g} Ω", f"{r['couple_fF']:g} fF", f"{100 * r['err_cal']:.2f} %"]
                 for r in tiles if r["couple_fF"] in (0.0, 0.1) and not (r["config"].startswith("DC") and r["couple_fF"])]
    tiny_p = os.path.join(RUNS, "tiny_hw", "eval_digital_finetune.json")
    tiny = json.load(open(tiny_p))["digital"] if os.path.exists(tiny_p) else None
    budget = bars_svg([("8-bit links (vs 10-bit)", 2.57), ("ADCs inside chips (7-bit)", 1.68), ("Multipliers", 0.69),
                       ("Rails (clipping)", 0.46), ("Amplifier noise", 0.23), ("Capacitor grid errors", 0.0)],
                      title="Where the loss of the AC chain comes from (removing one effect at a time, measured 2026-10-16)", color=CORAL)
    return CSS + f"""<div class="hwdoc">
<h2>Qwen3.5-0.8B: quality on the emulated chips</h2>
{_table(["run", "bits/byte", "vs original"], [
        ["Original digital model", f"{ref:.4f}", "—"],
        ["Design package values only (no physics)", "0.7398", "−0.06 % (translation is lossless)"],
        ["Emulated chip #0", "0.7585", "+2.45 %"],
        ["Emulated chip #1", "0.7564", "+2.18 %"]])}
<p class="note">12 held-out texts (6 Wikipedia, 6 code), 1,024 tokens each. Measured 2026-10-15. Lower bits/byte is better.</p>
<h2>Long context (1,024 tokens)</h2>
{_table(["test", "exact model", "emulated chip"], [
        ["Recall a number hidden early in ~980 tokens (20 trials)", "20 / 20", "20 / 20"],
        ["Bits/byte, tokens 0–256", "0.8250", "0.8500 (+3.0 %)"],
        ["Bits/byte, tokens 768–1023", "0.6748", "0.6927 (+2.6 %)"]])}
<p>The hardware error does not grow with context, because the memory of earlier tokens is digital.</p>
<h2>Error budget</h2>{budget}
<h2>Sensitivity per layer</h2>{layer_bars}
<h2>Exact circuit solves of one real tile (layer 0 FFN)</h2>
{_table(["design", "tile", "wire R", "line coupling", "output error (after calibration)"], tile_rows)}
<p class="note">DC resistor grids distort badly through wire voltage drop unless every cell is pre-compensated; the AC capacitor grid at
10 fF / 100 MHz stays within a fraction of a percent.</p>
<h2>Small model trained digitally, run on the chip (TinyStories, 5.8M weights)</h2>
{_table(["", "validation loss"], [["Exact", f"{tiny['ideal']:.4f}"], ["Emulated chip #0", f"{tiny['chip0']:.4f}"],
                                   ["Unseen chips #1–4", ", ".join(f"{tiny[f'chip{c}']:.4f}" for c in range(1, 5))]]) if tiny else "<p>not available</p>"}
<h2>Speed and energy (estimates)</h2>
{_table(["", "emulated analog chain", "M3 Max GPU (measured, digital)", "H100 (estimated)"], [
        ["Generation, one user", "~37,000 tokens/s", "191 tokens/s", "~800–1,600 tokens/s"],
        ["Energy per token, one user", "~5–15 µJ", "~0.2 J", "~0.3 J"],
        ["1,024-token prompt", "1.2 ms, 7.4 mJ", "134 ms", "~3–6 ms, ~2–4 J"]])}
<p class="note">Analog numbers come from the circuit timing and energy model of an unbuilt chip; NVIDIA numbers are scaled from the M3 Max
measurement using memory bandwidth and compute specs. Treat all of them as order-of-magnitude.</p>
</div>"""
