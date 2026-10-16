"""Build the static GitHub Pages site (docs/) from the same page generators the chat website uses.

Run on the machine that has the results (runs/), the hardware package and a running chat_server.py
(the Signal journey pages are recorded from its /trace endpoint):
  python build_site.py --out ../docs --server http://127.0.0.1:7860
"""
import argparse, html, json, os

import site_docs

ROOT = os.path.dirname(os.path.abspath(__file__))
PAGES = [("index.html", "Overview"), ("how-it-works.html", "How it works"), ("chip-design.html", "Chip design"),
         ("signal-journey.html", "Signal journey"), ("assumptions.html", "Assumptions"), ("results.html", "Results"),
         ("alternative.html", "Alternative design")]
SHELL = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · KirchhoffLM</title>
<meta name="description" content="KirchhoffLM: a 0.8B-parameter language model translated into analog capacitor-grid hardware and run in an emulator.">
<style>
:root {{ --body-text-color:#1f1f1d; --body-text-color-subdued:#66655f; --border-color-primary:#d9d7cf; --bg:#fbfaf7;
        --accent:#D85A30; --font: system-ui, -apple-system, "Segoe UI", sans-serif; --font-mono: ui-monospace, Menlo, monospace; color-scheme: light; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --body-text-color:#e9e7e0; --body-text-color-subdued:#a9a79f;
        --border-color-primary:#3d3d3a; --bg:#1b1b19; color-scheme: dark; }} }}
:root[data-theme="dark"] {{ --body-text-color:#e9e7e0; --body-text-color-subdued:#a9a79f; --border-color-primary:#3d3d3a; --bg:#1b1b19; color-scheme: dark; }}
body {{ margin:0; background:var(--bg); color:var(--body-text-color); font-family:var(--font); }}
header {{ border-bottom:1px solid var(--border-color-primary); padding:14px 16px; }}
header .wrap, main {{ max-width:1100px; margin:0 auto; }}
header a.brand {{ font-weight:600; font-size:18px; color:var(--body-text-color); text-decoration:none; margin-right:18px; }}
nav {{ display:inline-flex; flex-wrap:wrap; gap:4px 14px; }}
nav a {{ color:var(--body-text-color-subdued); text-decoration:none; font-size:15px; }}
nav a.on {{ color:var(--accent); }}
main {{ padding:8px 16px 60px; }}
main table {{ display:block; overflow-x:auto; max-width:100%; }}
a {{ color:var(--accent); }}
footer {{ max-width:1100px; margin:0 auto; padding:20px 16px 40px; color:var(--body-text-color-subdued); font-size:13px; }}
</style></head><body>
<header><div class="wrap"><a class="brand" href="index.html">KirchhoffLM</a><nav>{nav}</nav></div></header>
<main>{body}</main>
<footer>KirchhoffLM by UnaGo · simulation study, not a fabricated chip · device values are literature-range assumptions ·
<a href="https://github.com/UnaGo-AI/KirchhoffLM">source on GitHub</a></footer></body></html>"""


def page(fname, title, body):
    nav = "".join(f'<a href="{f}"{" class=on" if f == fname else ""}>{t}</a>' for f, t in PAGES)
    return SHELL.format(title=html.escape(title), nav=nav, body=body)


def overview(man):
    t, d = man["totals"], man["design"]
    emu = json.load(open(os.path.join(ROOT, "runs", "emulate.json")))
    ctx = json.load(open(os.path.join(ROOT, "runs", "context_1k.json")))
    ref = emu["reference"]
    chips = [v["bits_per_byte"] for v in emu["chips"].values()]
    lo, hi = 100 * (min(chips) / ref - 1), 100 * (max(chips) / ref - 1)
    tps = list(emu["chips"].values())[0]["tokens_per_s"]
    return site_docs.CSS + f"""<div class="hwdoc">
<h1>KirchhoffLM</h1>
<p style="font-size:1.15em">A modern language model, <b>Qwen3.5-0.8B</b>, translated into an <b>analog chip design</b>: every weight is a pair
of capacitors, every matrix product is done by currents adding up on wires (Kirchhoff's law), and the whole model runs through a
physics emulator of 24 layer chips.</p>
{site_docs._table(["", ""], [
        ["Quality on emulated chips", f"{lo:+.1f} to {hi:+.1f} % bits/byte vs the original model ({len(chips)} chips; {min(chips):.3f}–{max(chips):.3f} vs {ref:.3f})"],
        ["Long context", f"{ctx['recall']['chip']} / {ctx['recall']['trials']} recall of a fact hidden in ~{ctx['recall']['tokens'][0]} tokens (exact model: {ctx['recall']['exact']} / {ctx['recall']['trials']})"],
        ["Emulated speed", f"~{tps:,.0f} tokens/s for one user"],
        ["Emulated energy", f"~{list(emu['chips'].values())[0]['energy_j'] * 1e6:.0f} µJ per token at 2k context (estimate; most of it is the digital memory)"],
        ["Hardware", f"{t['capacitor_cells']:,} capacitors · {t['amplifiers']:,} amplifiers · ~{t['area_mm2']:,.0f} mm² over 25 chips"],
        ["Weight cell", f"AC capacitor grid, up to {d['c_cell'] * 1e15:g} fF in 255 unit steps, {d['ac_freq'] / 1e6:g} MHz carrier"]])}
<h2>Start here</h2>
<ul><li><a href="signal-journey.html">Signal journey</a> — follow one word through the chips, explained for a school student.</li>
<li><a href="chip-design.html">Chip design</a> — diagrams and every element of the chip.</li>
<li><a href="how-it-works.html">How it works</a> and <a href="assumptions.html">Assumptions</a> — what the emulator models and what it does not.</li>
<li><a href="results.html">Results</a> — quality, error budget, per-layer sensitivity, circuit checks.</li>
<li><a href="alternative.html">Alternative design</a> — one programmable engine re-loaded layer by layer instead of one chip per layer.</li></ul>
<p class="note">A simulation study by UnaGo. No chip has been built; device values are literature-range assumptions and real silicon
usually lands well below simulated efficiency. Related but different: KirchhoffNet (MIT/CMU, 2023) builds new continuous-time networks
from circuits; KirchhoffLM maps an existing LLM onto realistic hardware and models its errors.</p></div>"""


def journey(server):
    from gradio_client import Client
    c = Client(server, verbose=False)
    parts = [site_docs.CSS, '<div class="hwdoc"><p>Recorded from the emulator: the chip processed each text below, and the steps show '
             'the real numbers of that run for the last word.</p></div>']
    for text in ("The cat sat on the", "The capital of France is"):
        parts.append(f'<h2 style="margin-top:2em">Text: “{html.escape(text)}”</h2>' + c.predict(text, api_name="/trace"))
    return "".join(parts)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--out", default=os.path.join(ROOT, "..", "docs"))
    a.add_argument("--server", default="http://127.0.0.1:7860")
    a = a.parse_args()
    man = json.load(open(os.path.join(os.environ.get("KLM_PKG", os.path.join(ROOT, "hw_qwen35")), "manifest.json")))
    os.makedirs(a.out, exist_ok=True)
    bodies = {"index.html": overview(man), "how-it-works.html": site_docs.how_it_works_html(man),
              "chip-design.html": site_docs.chip_design_html(man), "signal-journey.html": journey(a.server),
              "assumptions.html": site_docs.assumptions_html(man), "results.html": site_docs.results_html(man),
              "alternative.html": site_docs.alternative_html(man)}
    for fname, title in PAGES:
        open(os.path.join(a.out, fname), "w").write(page(fname, title, bodies[fname]))
    open(os.path.join(a.out, ".nojekyll"), "w").close()
    print("wrote", ", ".join(f for f, _ in PAGES))


if __name__ == "__main__":
    main()
