"""Run Qwen3.5 from the hardware design package only (no original weights are read).

The architecture comes from config.json; every weight is rebuilt from the package's component
values, w = (cap_pos - cap_neg) / 255 * gain; norm gains, conv taps and the digital memory constants
come from the package too. Then the chip is emulated with its physics (manufacturing errors drawn per
chip seed) and evaluated against the reference quality stored at export time."""
import argparse, glob, json, os, time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.models import qwen3_5
from mlx_lm.utils import load_tokenizer

from hw import Chip
from run import bits_per_byte, encode_set
from sim import AnalogQwen, t_token


def load_package(d):
    d = Path(d)
    man = json.load(open(d / "manifest.json"))
    config = json.load(open(d / "config.json"))
    model = qwen3_5.Model(qwen3_5.ModelArgs.from_dict(config))
    params = {}
    for f in sorted(glob.glob(str(d / "chip_*.safetensors"))):
        arr = mx.load(f)
        for k, v in arr.items():
            if k.endswith(".cap_pos"):
                base = k[: -len(".cap_pos")]
                pos, neg, gain = v.astype(mx.float32), arr[base + ".cap_neg"].astype(mx.float32), arr[base + ".gain"]
                params[base] = (pos - neg) / man["levels"] * gain[:, None]
            elif not k.endswith((".cap_neg", ".gain", ".r_feedback")):
                params[k] = v
    expected = {k for k, _ in tree_flatten(model.parameters())}
    missing, extra = expected - set(params), set(params) - expected
    if missing or extra:
        raise ValueError(f"package does not match the architecture: missing {sorted(missing)[:5]}, extra {sorted(extra)[:5]}")
    model.update(tree_unflatten(list(params.items())))
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    return model, load_tokenizer(d), man, json.load(open(d / "calib.json"))


def generate(model, tok, prompt, n=40, temp=0.7, topk=20, seed=0):
    """Incremental generation, as the layer chips run: the prompt fills each chip's state once,
    then every new token makes one pass through the chain."""
    ids = tok.encode(prompt)
    rng = np.random.default_rng(seed)
    cache = model.make_cache()
    logits = model(mx.array(ids)[None], cache=cache)[0, -1]
    mx.eval(logits, cache)
    for _ in range(n):
        lg = np.array(logits.astype(mx.float32)) / temp
        top = np.argpartition(-lg, topk)[:topk]
        p = np.exp(lg[top] - lg[top].max())
        nxt = int(rng.choice(top, p=p / p.sum()))
        ids.append(nxt)
        logits = model(mx.array([[nxt]]), cache=cache)[0, -1]
        mx.eval(logits, cache)                      # one token through the chain at a time
    return tok.decode(ids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pkg", default=os.environ.get("KLM_PKG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "hw_qwen35")))
    p.add_argument("--chips", default="0,1")
    a = p.parse_args()
    t0 = time.time()
    model, tok, man, calib = load_package(a.pkg)
    print(f"loaded hardware package in {time.time() - t0:.0f}s: {man['totals']['capacitor_cells']:,} capacitors, "
          f"{man['totals']['amplifiers']:,} amplifiers, {len(man['chips'])} chips", flush=True)
    data = encode_set(tok, 6, 1024)
    ref = man["reference_bits_per_byte"]["mean"]
    exact = bits_per_byte(AnalogQwen(model, Chip(ideal=True), calib), data)["mean"]
    print(f"original digital model (export time)      {ref:.4f} bits/byte")
    print(f"package, component values only (no physics) {exact:.4f} bits/byte  ({100 * (exact / ref - 1):+.2f}%)", flush=True)
    design = man["design"]
    saved = {"reference": ref, "package_exact": exact, "chips": {}}
    for seed in map(int, a.chips.split(",")):
        chip = Chip(**{**design, "seed": seed})
        emu = AnalogQwen(model, chip, calib)
        r = bits_per_byte(emu, data)["mean"]
        e = emu.energy_per_token(2048)
        saved["chips"][str(seed)] = {"bits_per_byte": r, "energy_j": e["total"], "tokens_per_s": 1 / t_token(emu)}
        print(f"emulated chip #{seed}                            {r:.4f} bits/byte  ({100 * (r / ref - 1):+.2f}%)  "
              f"{e['total'] * 1e6:.1f} uJ/token  {1 / t_token(emu):,.0f} tokens/s", flush=True)
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"), exist_ok=True)
    json.dump(saved, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "emulate.json"), "w"), indent=1)
    emu = AnalogQwen(model, Chip(**{**design, "seed": 0}), calib)
    for prompt in ("The capital of France is", "def fibonacci(n):"):
        print(f"\n--- generated on emulated chip #0 ---\n{generate(emu, tok, prompt)}", flush=True)


if __name__ == "__main__":
    main()
