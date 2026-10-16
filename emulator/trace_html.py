"""'Signal journey' page: follow one token through the emulated chips, explained for a school student.

build_trace() runs the prompt on the exact model and on the emulated chip, records the signals of the
last token at every stage, and renders step-by-step HTML with the real numbers of this run: text to
tokens, embedding, norm, voltages, AC waves, capacitor currents adding up on a wire, demodulation back to
a number, SiLU and multiplier, residual, digital memory, the 10-bit chip link, all 24 chips, the output
scores and the next word, and the time and energy of the hardware."""
import html
import math

import numpy as np

from site_docs import CSS, TEAL, CORAL, GRAY, BLUE

STEP_CSS = """
<style>
.trace .step { border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 14px 18px; margin: 14px 0; }
.trace .step h3 { margin: 0 0 6px; }
.trace .num { font-family: var(--font-mono); }
.trace .kid { border-left: 3px solid #378ADD; padding-left: 10px; margin: 8px 0; }
.trace .phys { border-left: 3px solid #1D9E75; padding-left: 10px; margin: 8px 0; }
.trace .pill { display: inline-block; border: 1px solid var(--border-color-primary); border-radius: 6px; padding: 1px 7px; margin: 2px; font-family: var(--font-mono); }
.trace .pill.sel { border-color: #D85A30; background: rgba(216,90,48,0.12); }
</style>
"""


class Capture:
    """Recorder: keeps the last token's signal at every named stage."""

    def __init__(self, mx):
        self.mx, self.data = mx, {}

    def __call__(self, name, x):
        if ".state" in name or "gdn_core.h" in name:
            return x
        a = np.array(x.astype(self.mx.float32))
        self.data[name] = a[0, -1].reshape(-1) if a.ndim >= 2 else a.reshape(-1)
        return x


def _fmt(v, n=3):
    return f"{v:+.{n}f}"


def _vec(v, k=8):
    return " ".join(f'<span class="pill">{x:+.3f}</span>' for x in v[:k]) + " …"


def _sine_svg(amps, labels):
    """Small AC waves: amplitude = voltage, a flipped wave = negative number."""
    w, h = 900, 40 + 60 * len(amps)
    p = []
    amax = max(abs(a) for a in amps) or 1
    for j, (a, lab) in enumerate(zip(amps, labels)):
        y0 = 50 + j * 60
        pts = " ".join(f"{220 + t * 1.6:.1f},{y0 - 22 * (a / amax) * math.sin(2 * math.pi * t / 100):.1f}" for t in range(0, 401, 4))
        p.append(f'<text x="10" y="{y0 + 4}" font-size="12">{html.escape(lab)}</text>'
                 f'<line x1="220" y1="{y0}" x2="860" y2="{y0}" stroke="currentColor" stroke-opacity="0.25"/>'
                 f'<polyline points="{pts}" fill="none" stroke="{BLUE}" stroke-width="2"/>')
    return f'<svg viewBox="0 0 {w} {h}" width="100%" style="color:var(--body-text-color)">{"".join(p)}</svg>'


def _column_svg(items, total_ua):
    """Currents from a few cells joining one column wire."""
    w, h = 900, 70 + 46 * len(items)
    p = [f'<line x1="700" y1="20" x2="700" y2="{h - 30}" stroke="{TEAL}" stroke-width="3"/>',
         f'<text x="712" y="{h - 26}" font-size="13">column wire: total {total_ua:+.3f} µA</text>']
    vmax = max(abs(i) for _, i in items) or 1
    for j, (lab, i_ua) in enumerate(items):
        y = 40 + j * 46
        wdt = 260 * abs(i_ua) / vmax
        col = TEAL if i_ua >= 0 else CORAL
        p.append(f'<text x="10" y="{y + 5}" font-size="12">{html.escape(lab)}</text>'
                 f'<rect x="{690 - wdt}" y="{y - 9}" width="{wdt}" height="18" rx="3" fill="{col}" fill-opacity="0.7"/>'
                 f'<text x="{684 - wdt}" y="{y + 5}" text-anchor="end" font-size="12">{i_ua:+.4f} µA</text>')
    return f'<svg viewBox="0 0 {w} {h}" width="100%" style="color:var(--body-text-color)">{"".join(p)}</svg>'


def _layers_svg(errs):
    w, h = 900, 200
    m = max(errs) or 1
    pts = " ".join(f"{60 + i * 34},{170 - 140 * e / m:.1f}" for i, e in enumerate(errs))
    p = [f'<line x1="60" y1="170" x2="{60 + 23 * 34}" y2="170" stroke="currentColor" stroke-opacity="0.4"/>',
         f'<polyline points="{pts}" fill="none" stroke="{CORAL}" stroke-width="2"/>']
    for i, e in enumerate(errs):
        x, y = 60 + i * 34, 170 - 140 * e / m
        p.append(f'<circle cx="{x}" cy="{y:.1f}" r="3" fill="{CORAL}"/>')
        if i % 2 == 0:
            p.append(f'<text x="{x}" y="188" text-anchor="middle" font-size="11">L{i}</text>')
    p.append(f'<text x="10" y="24" font-size="12">difference from the exact model after each chip (max {100 * m:.1f} %)</text>')
    return f'<svg viewBox="0 0 {w} {h}" width="100%" style="color:var(--body-text-color)">{"".join(p)}</svg>'


def _bits_svg(code, bits):
    s = format(code & ((1 << bits) - 1), f"0{bits}b")
    p = []
    for j, ch in enumerate(s):
        x = 20 + j * 46
        on = ch == "1"
        p.append(f'<rect x="{x}" y="20" width="40" height="40" rx="5" fill="{CORAL if on else GRAY}" fill-opacity="{0.75 if on else 0.18}"/>'
                 f'<text x="{x + 20}" y="45" text-anchor="middle" font-size="16">{ch}</text>')
    return f'<svg viewBox="0 0 {40 + bits * 46} 80" width="{40 + bits * 46}" style="color:var(--body-text-color)">{"".join(p)}</svg>'


def step(n, title, kid, phys, body=""):
    return (f'<div class="step"><h3>Step {n}: {html.escape(title)}</h3><div class="kid"><b>In simple words:</b> {kid}</div>'
            f'<div class="phys"><b>In the chip:</b> {phys}</div>{body}</div>')


def build_trace(text, mx, tok, man, cal, exact, chip, model, fs_fn, t_token):
    d = man["design"]
    ids = tok.encode(text)
    if not ids:
        raise ValueError("Enter some text first.")
    ce, cc = Capture(mx), Capture(mx)
    le = exact(mx.array(ids)[None], rec=ce)
    lc = chip(mx.array(ids)[None], rec=cc)
    mx.eval(le, lc)
    E, C = ce.data, cc.data
    pct = d["fs_pct"]
    out = [CSS, STEP_CSS, '<div class="hwdoc trace">']
    pieces = [tok.decode([i]) for i in ids]
    toks = "".join(f'<span class="pill{" sel" if j == len(ids) - 1 else ""}">{html.escape(repr(p_)[1:-1])} <small>#{i}</small></span>'
                   for j, (p_, i) in enumerate(zip(pieces, ids)))
    last = pieces[-1]
    out.append(f"<h2>Follow the word {html.escape(repr(last))} through the chips</h2>"
               f"<p>The model reads the whole text, but here we follow only the <b>last piece</b>, because that is where it "
               f"guesses what comes next. All numbers below come from this run.</p>")

    out.append(step(1, "Text becomes numbers (tokens)",
                    "A computer cannot read letters. The text is cut into pieces called <i>tokens</i>, and each piece has a number, "
                    "like a page number in a dictionary of 248,320 pieces.",
                    "This is ordinary digital work, done before the chips.",
                    f"<p>{toks}</p>"))

    emb = E["embed"]
    out.append(step(2, "The token becomes a list of 1,024 numbers (embedding)",
                    f"Token #{ids[-1]} is looked up in a big table. Its row is a list of 1,024 numbers that describe the word, "
                    "a bit like a fingerprint. Words with similar meaning get similar lists.",
                    "The table is stored once; the row is read out and sent to the first layer chip.",
                    f"<p class='num'>first 8 of 1,024: {_vec(emb)}</p>"))

    n1 = E["L0.norm1"]
    out.append(step(3, "Volume control (norm unit)",
                    "Some lists are loud, some are quiet. The norm unit turns the volume so the list always has the same "
                    "average loudness, so the next parts are never overloaded or too quiet.",
                    "An analog circuit measures the average size of all 1,024 signals (root-mean-square) and divides each signal "
                    "by it; one small resistor per line sets the learned gain.",
                    f"<p class='num'>before: {_vec(emb)}<br>after:&nbsp;&nbsp;{_vec(n1)}</p>"))

    x = E["L0.norm2"]
    x_fs = fs_fn(cal, "L0.norm2", pct)
    volts = np.clip(x / x_fs, -1, 1) * d["v_read"]
    top_in = np.argsort(-np.abs(volts))[:4]
    out.append(step(4, "Numbers become voltages, then waves (DAC + modulator)",
                    "Electronics cannot use numbers directly, so each number becomes an electrical push: a voltage. "
                    "Then the voltage is turned into a wave that wiggles 100 million times per second. A bigger number makes a "
                    "taller wave; a negative number makes a wave that is flipped upside down.",
                    f"A digital-to-analog converter makes the voltage (full scale ±{d['v_read']} V for the largest signal "
                    f"{x_fs:.2f}); a modulator multiplies it by a {d['ac_freq'] / 1e6:g} MHz carrier.",
                    _sine_svg([float(volts[i]) for i in top_in], [f"input {i}: {x[i]:+.3f} → {1000 * volts[i]:+.1f} mV" for i in top_in])))

    W = np.array(model.language_model.model.layers[0].mlp.gate_proj.weight.astype(mx.float32))
    y_ex, y_ch = E["L0.mlp_gate"], C["L0.mlp_gate"]
    j = int(np.argmax(np.abs(y_ex)))
    row = W[j]
    wmax = np.abs(row).max()
    codes = np.round(np.abs(row) / wmax * 255)
    c_unit = man["units"]["c_unit_F"]
    omega = 2 * math.pi * d["ac_freq"]
    v_full = np.clip(x / x_fs, -1, 1) * d["v_read"]
    i_cell = np.sign(row) * codes * c_unit * omega * v_full                     # amperes, in-phase amplitude
    contrib = np.argsort(-np.abs(i_cell))[:5]
    items = [(f"input {i}: {1000 * v_full[i]:+.1f} mV × {'C+' if row[i] > 0 else 'C−'} {codes[i] * c_unit * 1e15:.2f} fF", 1e6 * i_cell[i]) for i in contrib]
    total = 1e6 * i_cell.sum()
    out.append(step(5, "The magic part: capacitors multiply, wires add (capacitor grid)",
                    "Each input wave crosses many tiny capacitors. A capacitor lets a little wave-current through: more if the "
                    "wave is taller (the input) and more if the capacitor is bigger (the learned weight). That is a multiplication, "
                    "done by physics. All those little currents flow into one wire and simply add up, like streams joining a river. "
                    "That is the sum. One wire = one answer, and thousands of wires work at the same time.",
                    f"Current through a capacitor: I = ω·C·V. Each weight is a pair of capacitors (C+ for positive, C− for negative) "
                    f"made of 0–255 unit capacitors of {c_unit * 1e18:.1f} aF. Kirchhoff's law adds all 1,024 currents on the column "
                    f"wire. Here: FFN gate grid of layer 0, output column {j}, the 5 biggest of its 1,024 currents.",
                    _column_svg(items, total)))

    err = (y_ch[j] - y_ex[j]) / (abs(y_ex[j]) + 1e-9)
    out.append(step(6, "Back from a wave to a number (synchronous demodulator + amplifier)",
                    "The wire carries a wave, but the next part needs a plain number. The demodulator listens to the wave in "
                    "perfect step with the original rhythm and measures how tall it is, ignoring noise that does not follow the rhythm. "
                    "An amplifier then turns the small current into a usable voltage.",
                    "The demodulator multiplies the column current by the carrier and averages it (only the in-phase part survives, "
                    "so amplifier offsets and slow noise are rejected). The read-out amplifier's feedback resistor sets the scale. "
                    "Real circuits are not perfect: capacitor mismatch, amplifier noise and wire delays shift the result a little.",
                    f"<p class='num'>exact math: {y_ex[j]:+.4f} &nbsp;|&nbsp; emulated chip: {y_ch[j]:+.4f} &nbsp;|&nbsp; difference {100 * err:+.2f} %</p>"))

    act_e, act_c = E["L0.mlp_act"][j], C["L0.mlp_act"][j]
    h_e, h_c = E["L0.mlp_h"][j], C["L0.mlp_h"][j]
    out.append(step(7, "A smart gate decides what passes (SiLU unit and multiplier)",
                    "Not every signal should pass. A gate lets big positive signals through, almost blocks negative ones, and "
                    "is gentle in between. Then a second signal (from the 'up' grid) is multiplied by the gate's answer: "
                    "it is like a tap controlling how much water flows.",
                    "SiLU is made from transistors whose current curve has the right shape; the multiplication is a Gilbert cell. "
                    "Each unit has its own tiny manufacturing differences (gain about 1 %, trimmed offsets).",
                    f"<p class='num'>gate {y_ex[j]:+.3f} → SiLU {act_e:+.3f} (chip {act_c:+.3f}) → × up → {h_e:+.4f} (chip {h_c:+.4f})</p>"))

    r_prev, r_ex, r_ch = E["L0.norm2"], E["L0.resid"], C["L0.resid"]
    out.append(step(8, "Adding the result to the main signal (residual)",
                    "The layer does not replace the signal; it adds a correction to it, like adding a note to a page instead of "
                    "rewriting the page. The 1,024 main signals carry the meaning from layer to layer.",
                    "The down grid's output currents flow straight into the 1,024 residual capacitors: charge adds to charge, so the "
                    "addition costs nothing.",
                    f"<p class='num'>main signal after layer 0, first 8: {_vec(r_ex)}<br>on the chip: {_vec(r_ch)}</p>"))

    core_e, core_c = E["L0.gdn_core"], C["L0.gdn_core"]
    out.append(step(9, "Remembering earlier words (digital memory)",
                    "To understand a sentence the model must remember the words before. Each layer chip keeps a memory: 16 small "
                    "tables of 128 × 128 numbers that are updated with every word and read to find what matters now.",
                    "Analog memory turned out too noisy, so this memory is digital: signals go through an ADC into SRAM, the "
                    "memory is updated with digital multiply-adds, and the answer goes back through a DAC to the analog part. "
                    "(In this emulator the digital memory is exact.)",
                    f"<p class='num'>memory read-out, first 8 numbers: {_vec(core_e)}<br>on the chip: {_vec(core_c)}</p>"))

    k = int(np.argmax(np.abs(r_ch)))
    fs_l = fs_fn(cal, "L0.resid", pct)
    lv = 2 ** (d["link_bits"] - 1) - 1
    code = int(round(np.clip(r_ch[k] / fs_l, -1, 1) * lv))
    back = code / lv * fs_l
    out.append(step(10, "Jumping to the next chip (ADC → bits → DAC)",
                    "Each layer lives on its own chip. To travel between chips, every signal is turned into a short row of 0s and 1s "
                    "(bits), sent along a fast cable, and turned back into a voltage. Bits are robust: noise on the cable cannot "
                    "change a 0 into a 1 easily, so errors stop piling up.",
                    f"A {d['link_bits']}-bit ADC per line, a serial link, and a DAC on the next chip. Here signal {k}:",
                    f"<p class='num'>{r_ch[k]:+.4f} → code {code} → </p>{_bits_svg(code, d['link_bits'])}"
                    f"<p class='num'>→ back to {back:+.4f} on the next chip</p>"))

    n = chip.n_layers
    errs = []
    for i in range(n):
        a, b = C[f"L{i}.resid"], E[f"L{i}.resid"]
        errs.append(float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-9)))
    out.append(step(11, "The same journey 24 times (all layer chips)",
                    "Steps 3–10 happen on every chip, 24 times. Some chips mix information from earlier words "
                    "(memory chips and attention chips), all of them refine the meaning. Small errors appear on each chip, "
                    "but the norm units and the digital links keep them from growing out of control.",
                    "18 DeltaNet chips and 6 attention chips, each about 1.1 µs per token.",
                    _layers_svg(errs)))

    def top5(lg):
        lg = np.array(lg[0, -1].astype(mx.float32))
        p = np.exp(lg - lg.max())
        p /= p.sum()
        idx = np.argsort(-p)[:5]
        return [(tok.decode([int(i)]), float(p[i])) for i in idx]
    te, tc = top5(le), top5(lc)
    rows = "".join(f"<tr><td>{html.escape(repr(a[0]))}</td><td>{100 * a[1]:.1f} %</td><td>{html.escape(repr(b[0]))}</td><td>{100 * b[1]:.1f} %</td></tr>"
                   for a, b in zip(te, tc))
    out.append(step(12, "Guessing the next word (output grid)",
                    "The final signal is compared with all 248,320 known pieces at once. Each comparison gives a score; the "
                    "highest scores are the best guesses for what comes next.",
                    "The output grid has 248,320 columns (one per token). An 8-bit ADC measures the scores relative to the top one "
                    "(a winner-take-all circuit could pick the winner directly).",
                    f"<table><tr><th>exact model</th><th>chance</th><th>emulated chip</th><th>chance</th></tr>{rows}</table>"))

    t = t_token(chip)
    e = chip.energy_per_token(len(ids))["total"]
    lift_cm = 100 * e / (0.6e-6 * 9.81)                  # a 0.6 mg sugar grain, E = m g h
    out.append(step(13, "How long and how much energy",
                    f"The whole journey for this one word takes about <b>{t * 1e6:.0f} millionths of a second</b>. In the time you "
                    f"blink (0.1 s) the chips could handle about {0.1 / t:,.0f} words. The energy is about {e * 1e6:.1f} millionths of "
                    f"a joule — about what it takes to lift one grain of sugar by {lift_cm:.0f} cm.",
                    f"Emulated hardware: {t * 1e6:.1f} µs and {e * 1e6:.2f} µJ per token at context {len(ids)} (estimates from the "
                    f"circuit model of an unbuilt chip). The Mac needs far longer to simulate it.",
                    ""))
    out.append("</div>")
    return "".join(out)
