"""Chat with Qwen3.5-0.8B running on the emulated analog hardware (Gradio web UI).

The model is loaded from the hardware design package only (capacitor counts, feedback resistors,
norm/conv ratios). Replies stream token by token through the emulated chain of layer chips
(incremental mode), and a live log shows what the emulated hardware is doing.

MLX keeps its GPU streams per thread, so one dedicated GPU thread owns the model and runs every
request; the web handlers queue requests to it and stream back its events.

  .venv/bin/python chat_server.py --port 7860        # then open http://<host>:7860
"""
import argparse
import json
import os
import queue
import threading
import time

import gradio as gr
import numpy as np

import site_docs

PKG = os.environ.get("KLM_PKG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "hw_qwen35"))
MAN = json.load(open(os.path.join(PKG, "manifest.json")))
JOBS = queue.Queue()
READY = threading.Event()
DONE = object()


def gpu_worker():
    """Owns the model and the emulated chips; runs one request at a time."""
    import mlx.core as mx
    from emulate_hw import load_package
    from hw import Chip
    from sim import AnalogQwen, fs, t_token

    model, tok, man, cal = load_package(PKG)
    exact = AnalogQwen(model, Chip(ideal=True), cal)
    chips = {}

    def get_chip(seed):
        if seed not in chips:
            chips.clear()                                   # keep one emulated chip in memory
            chips[seed] = AnalogQwen(model, Chip(**{**man["design"], "seed": seed}), cal)
        return chips[seed]

    get_chip(0)                                             # manufacture chip #0 at start-up
    READY.set()

    class LinkMonitor:
        """Per layer chip: share of the newest token's residual values that saturate the chip-to-chip
        link ADC, and the peak as a fraction of its range (reduced on the GPU, read once per token)."""

        def __init__(self, n_layers, pct):
            self.fs = [fs(cal, f"L{i}.resid", pct) for i in range(n_layers)]
            self.stats = {}

        def __call__(self, name, x):
            if name.endswith(".resid"):
                i = int(name[1:].split(".")[0])
                a = mx.abs(x[:, -1:])
                self.stats[i] = (mx.mean(a > self.fs[i]), mx.max(a) / self.fs[i])
            return x

        def read(self):
            out = {i: (float(c), float(u)) for i, (c, u) in self.stats.items()}
            self.stats = {}
            return out

    import trace_html
    while True:
        kind, req, out = JOBS.get()
        try:
            if kind == "chat":
                run_request(req, out, mx, tok, man, exact, get_chip, LinkMonitor, t_token)
            else:
                out.put(("html", trace_html.build_trace(req, mx, tok, man, cal, exact, get_chip(0), model, fs, t_token)))
        except Exception as e:                              # report to the user, keep the worker alive
            out.put(("error", f"{type(e).__name__}: {e}"))
        out.put(DONE)


def run_request(req, out, mx, tok, man, exact, get_chip, LinkMonitor, t_token):
    messages, mode, seed, max_new, temperature, thinking = req
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=thinking)
    ids = tok.encode(prompt)
    emulated = mode.startswith("Emulated")
    hw = get_chip(seed)
    m = hw if emulated else exact
    stage = t_token(hw) / hw.n_layers
    t_tok = t_token(hw)
    e1, e2 = hw.energy_per_token(1)["total"], hw.energy_per_token(4097)["total"]
    energy = lambda ctx: e1 + (e2 - e1) * (ctx - 1) / 4096            # energy is linear in context
    mon = LinkMonitor(m.n_layers, man["design"]["fs_pct"])
    log = lambda s: out.put(("log", s))
    log(f"request: {len(ids)} prompt tokens, run on {mode}" + (f" #{seed}" if emulated else ""))
    if emulated:
        d = man["design"]
        log(f"chip chain: {m.n_layers} layer chips, AC capacitor grids {d['c_cell'] * 1e15:g} fF @ {d['ac_freq'] / 1e6:g} MHz, "
            f"{d['v_read']} V, tile {d['tile']}, {d['link_bits']}-bit chip links, digital memory ({d['memory']})")
    cache = m.make_cache()
    t0 = time.time()
    logits = m(mx.array(ids)[None], cache=cache, rec=mon if emulated else None)[0, -1]
    mx.eval(logits, cache)
    t_pre = time.time() - t0
    e_pre = sum(energy(c) for c in range(1, len(ids) + 1))
    log(f"prompt processed: {'emulator' if emulated else 'digital model'} {t_pre * 1e3:.0f} ms ({len(ids) / t_pre:,.0f} tok/s)"
        + (f" | emulated hardware {(len(ids) + m.n_layers - 1) * stage * 1e6:.1f} us, {e_pre * 1e3:.3f} mJ "
           f"(pipelined, {stage * 1e9:.0f} ns per chip)" if emulated else ""))
    if emulated:
        hot = sorted(mon.read().items(), key=lambda kv: -kv[1][1])[:3]
        log("link use (peak |signal| / ADC range), busiest chips: " + ", ".join(f"L{i} {100 * u:.0f}%" for i, (c, u) in hot))
    rng = np.random.default_rng(seed + len(ids))
    gen, hw_t, hw_e, t_gen0 = [], 0.0, 0.0, time.time()
    eos = set(tok.eos_token_ids)
    for step in range(max_new):
        lg = np.array(logits.astype(mx.float32))
        if temperature <= 0:
            nxt = int(lg.argmax())
        else:
            top = np.argpartition(-lg, 40)[:40]
            p = np.exp((lg[top] - lg[top].max()) / temperature)
            nxt = int(rng.choice(top, p=p / p.sum()))
        if nxt in eos:
            log(f"end of reply (stop token) after {step} tokens")
            break
        gen.append(nxt)
        ctx = len(ids) + len(gen)
        hw_t += t_tok
        hw_e += energy(ctx)
        ts = time.time()
        logits = m(mx.array([[nxt]]), cache=cache, rec=mon if emulated else None)[0, -1]
        mx.eval(logits, cache)
        dt = time.time() - ts
        out.put(("text", tok.decode(gen)))
        if step < 3 or step % 16 == 0:
            line = f"token {step + 1:4d} {tok.decode([nxt])!r:>12}: {'emulator' if emulated else 'digital model'} {dt * 1e3:4.0f} ms"
            if emulated:
                line += f" | hardware {t_tok * 1e6:.1f} us, {energy(ctx) * 1e6:.1f} uJ at context {ctx}"
                sat = [i for i, (c, u) in mon.read().items() if c > 0]
                line += f" | link saturation: {', '.join(f'L{i}' for i in sat) if sat else 'none'}"
            log(line)
        elif emulated:
            mon.read()
    n, wall = max(len(gen), 1), time.time() - t_gen0
    if emulated:
        log(f"done: {len(gen)} tokens | emulator {n / wall:.1f} tok/s | emulated hardware {hw_t * 1e3:.3f} ms = "
            f"{n / max(hw_t, 1e-12):,.0f} tok/s, {hw_e * 1e3:.3f} mJ ({hw_e / n * 1e6:.1f} uJ/token)")
    else:
        log(f"done: {len(gen)} tokens | digital model (float32, this simulator) {n / wall:.1f} tok/s; "
            f"for reference, mlx-lm bf16 on this GPU runs ~191 tok/s")


def text_of(c):
    """Gradio may hand message content back as a list of parts."""
    if isinstance(c, str):
        return c
    if isinstance(c, dict):
        return c.get("text", "")
    return "".join(text_of(x) for x in c)


def chat(message, history, mode, seed, max_new, temperature, thinking):
    if not message.strip():
        raise gr.Error("Enter a message first.")
    messages = [{"role": h["role"], "content": text_of(h["content"])} for h in history] + [{"role": "user", "content": message}]
    history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]
    lines, out = [], queue.Queue()
    waiting = JOBS.qsize()
    JOBS.put(("chat", (messages, mode, int(seed), int(max_new), float(temperature), bool(thinking)), out))
    if waiting:
        lines.append(time.strftime("%H:%M:%S ") + f"queued behind {waiting} request(s)")
        yield history, "\n".join(lines)
    while True:
        ev = out.get()
        if ev is DONE:
            break
        kind, payload = ev
        if kind == "text":
            history[-1]["content"] = payload
        else:
            lines.append(time.strftime("%H:%M:%S ") + ("ERROR " if kind == "error" else "") + payload)
        yield history, "\n".join(lines[-400:])


def trace(text):
    if not text.strip():
        raise gr.Error("Enter some text first.")
    out = queue.Queue()
    JOBS.put(("trace", text, out))
    result = None
    while True:
        ev = out.get()
        if ev is DONE:
            break
        kind, payload = ev
        if kind == "error":
            raise gr.Error(payload)
        result = payload
    return result


def hardware_md():
    d, t = MAN["design"], MAN["totals"]
    rows = "\n".join(f"| {k} | {c['weights']:,} | {c['capacitor_cells']:,} | {c['amplifiers']:,} | {c['area_mm2']:,} |"
                     for k, c in MAN["chips"].items())
    return (f"### Hardware design package\n"
            f"Source model **{MAN['source_model']}**, translated into {len(MAN['chips'])} chips. "
            f"Each weight is a capacitor pair of up to 255 unit capacitors of {MAN['units']['c_unit_F'] * 1e18:.1f} aF "
            f"(max {d['c_cell'] * 1e15:g} fF), read with a {d['ac_freq'] / 1e6:g} MHz carrier at {d['v_read']} V.\n\n"
            f"**Totals:** {t['weights']:,} weights, {t['capacitor_cells']:,} capacitors "
            f"({t['unit_capacitors']:,} unit capacitors), {t['amplifiers']:,} read-out amplifiers, ~{t['area_mm2']:,.0f} mm².\n\n"
            f"Reference quality (original digital model): {MAN['reference_bits_per_byte']['mean']:.4f} bits/byte.\n\n"
            f"| chip | weights | capacitors | amplifiers | area mm² |\n|---|---|---|---|---|\n{rows}")


with gr.Blocks(title="Analog Qwen3.5 emulator") as app:
    gr.Markdown("## Qwen3.5-0.8B on emulated analog hardware\nThe model runs from its hardware design package "
                "(capacitor counts and resistor values) through an emulated chain of 24 layer chips.")
    with gr.Tab("Chat"):
        with gr.Row():
            with gr.Column(scale=3):
                bot = gr.Chatbot(height=520, label="Conversation")
                msg = gr.Textbox(placeholder="Ask something", label="Message", lines=2)
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    clear = gr.Button("Clear")
            with gr.Column(scale=2):
                mode = gr.Radio(["Emulated analog chip", "Exact digital model"], value="Emulated analog chip", label="Run on")
                seed = gr.Number(value=0, precision=0, label="Chip number (manufacturing errors)")
                max_new = gr.Slider(16, 512, value=200, step=16, label="Max new tokens")
                temp = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature (0 = greedy)")
                thinking = gr.Checkbox(value=False, label="Thinking mode")
                logbox = gr.Textbox(label="Processing log", lines=22, max_lines=22, autoscroll=True)
        args = [msg, bot, mode, seed, max_new, temp, thinking]
        send.click(chat, args, [bot, logbox], api_name="chat").then(lambda: "", None, msg)
        msg.submit(chat, args, [bot, logbox]).then(lambda: "", None, msg)
        clear.click(lambda: ([], ""), None, [bot, logbox])
    with gr.Tab("Signal journey"):
        gr.Markdown("Type a short text. The emulated chip processes it, and the page follows the **last word** through "
                    "every step of the hardware, with the real numbers of this run, explained simply.")
        with gr.Row():
            tr_text = gr.Textbox(value="The cat sat on the", label="Text", scale=4)
            tr_btn = gr.Button("Trace the signal", variant="primary", scale=1)
        tr_out = gr.HTML()
        tr_btn.click(trace, tr_text, tr_out, api_name="trace")
        tr_text.submit(trace, tr_text, tr_out)
    with gr.Tab("How it works"):
        gr.HTML(site_docs.how_it_works_html(MAN))
    with gr.Tab("Chip design"):
        gr.HTML(site_docs.chip_design_html(MAN))
    with gr.Tab("Assumptions"):
        gr.HTML(site_docs.assumptions_html(MAN))
    with gr.Tab("Results"):
        gr.HTML(site_docs.results_html(MAN))
    with gr.Tab("Alternative design"):
        gr.HTML(site_docs.alternative_html(MAN))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", default="0.0.0.0")
    a = p.parse_args()
    threading.Thread(target=gpu_worker, daemon=True, name="mlx-gpu").start()
    READY.wait()
    app.queue(default_concurrency_limit=4).launch(server_name=a.host, server_port=a.port, show_error=True)
