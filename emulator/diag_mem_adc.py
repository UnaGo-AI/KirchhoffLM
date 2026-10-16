"""Which memory converter costs quality, and how many bits are needed? Only the converters are non-ideal
(no links, no checkpoints), on the hardware package."""
import json, os
from emulate_hw import load_package
from hw import EFFECTS, Chip
from run import bits_per_byte, encode_set
from sim import AnalogQwen

model, tok, man, cal = load_package(os.environ.get("KLM_PKG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "hw_qwen35")))
data = encode_set(tok, 3, 768)
ref = bits_per_byte(AnalogQwen(model, Chip(ideal=True), cal), data)["mean"]
only_adc = ",".join(e for e in EFFECTS if e != "adc")
base = {**man["design"], "link_bits": 0, "ckpt_every": 0, "off": only_adc}


def loss(**over):
    r = bits_per_byte(AnalogQwen(model, Chip(**{**base, **over}), cal), data)["mean"]
    return 100 * (r / ref - 1)


print(f"exact {ref:.4f}")
for bits in (8, 10, 12):
    print(f"per-head ranges, all converters at {bits} bits: {loss(mem_adc_bits=bits, mem_adc_per_head=True):+.2f}%", flush=True)
print(f"per-head ranges, 10 bits, only 'qkv': {loss(mem_adc_per_head=True, mem_adc_skip='gates,y,attn_in,attn_out'):+.2f}%", flush=True)
base["mem_adc_per_head"] = False                     # the rows below: one range per layer
groups = ["qkv", "gates", "y", "attn_in", "attn_out"]
print(f"all memory converters off: {loss(mem_adc_skip=','.join(groups)):+.2f}%  (output ADC only)", flush=True)
for bits in (8, 10, 12, 14, 16):
    print(f"all converters at {bits} bits: {loss(mem_adc_bits=bits):+.2f}%", flush=True)
for g in groups:
    print(f"10 bits, only '{g}' converted: {loss(mem_adc_skip=','.join(x for x in groups if x != g)):+.2f}%", flush=True)
