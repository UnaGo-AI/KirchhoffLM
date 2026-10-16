"""Layer-multiplexed analog engine: timing, energy and area model.

Instead of one chip per layer (weights fixed in place), a few physical capacitor arrays with SRAM-switched
cells are re-programmed for each layer. A sequencer broadcasts the layer index on a bus; each engine loads
that layer's capacitor codes (16 bits per weight: C+ and C- counts) from a weight store, computes a batch
of tokens, and moves to the next layer. Two banks per engine (ping-pong) let the next layer load while the
current one computes.

All device numbers are rough literature-range assumptions; this is an order-of-magnitude model.
"""
import argparse
from dataclasses import dataclass

BITS_PER_WEIGHT = 16          # two 8-bit capacitor codes


@dataclass
class Store:
    name: str
    read_pj_per_bit: float     # energy to fetch one weight bit from the store
    bandwidth_tbps: float      # TB/s into one engine


STORES = {
    "on-chip SRAM": Store("on-chip SRAM", 0.005, 4.0),
    "3D-stacked SRAM": Store("3D-stacked SRAM", 0.15, 8.0),     # hybrid-bonded on top of the engine
    "HBM": Store("HBM", 4.0, 2.0),
    "DDR": Store("DDR", 15.0, 0.2),
}


@dataclass
class Model:
    name: str
    layers: int
    weights_per_layer: float   # weights that must be loaded per layer (active ones for MoE)
    total_weights: float       # all weights that must be stored
    head_weights: float        # output layer (kept on a fixed engine of its own)


MODELS = {
    "Qwen3.5-0.8B": Model("Qwen3.5-0.8B", 24, 21.5e6, 752e6, 254e6),
    "1T MoE (32B active)": Model("1T MoE (32B active)", 61, 32e9 / 61, 1e12, 1.0e9),
}

CELL_UM2 = 20.0                # SRAM-switched capacitor cell pair, um^2 per weight
FIXED_CELL_UM2 = 10.6          # fixed capacitor pair (the current design), um^2 per weight
SRAM_UM2_PER_BIT = 0.03        # dense SRAM bit incl. periphery (advanced node)
WRITE_FJ_PER_BIT = 5.0         # writing the cell's switch SRAM + charging switch gates
T_STAGE = 140e-9               # one pipelined grid read stage (s); a new token enters every stage
T_LAYER = 8 * T_STAGE          # latency of one token through one layer
E_COMPUTE_PER_WEIGHT = 0.4e-15 # J per weight per token in the analog grid + amplifiers (from the emulator)
E_MEMORY_PER_TOKEN_LAYER = 0.6e-6  # digital memory / attention per token per layer (emulator, 2k context)


def evaluate(m: Model, store: Store, batch: int, engines: int, double_buffer=True):
    """Throughput, latency and energy when each layer's weights are loaded once per batch of tokens."""
    bits = m.weights_per_layer * BITS_PER_WEIGHT
    t_load = bits / 8 / (store.bandwidth_tbps * 1e12)                    # s per layer load per engine
    t_comp = T_LAYER + (batch - 1) * T_STAGE                              # s to push the batch through a layer
    t_layer = max(t_load, t_comp) if double_buffer else t_load + t_comp
    t_pass = m.layers * t_layer                                           # latency of one token through all layers
    # with several engines, several batches are in flight on different layers: more total throughput, same latency
    e_load = bits * (store.read_pj_per_bit * 1e-12 + WRITE_FJ_PER_BIT * 1e-15)  # J per layer load
    e_token = (m.layers * e_load / batch                                  # reprogramming, shared by the batch
               + m.layers * (m.weights_per_layer * E_COMPUTE_PER_WEIGHT + E_MEMORY_PER_TOKEN_LAYER))
    area_engines = engines * (2 if double_buffer else 1) * m.weights_per_layer * CELL_UM2 * 1e-6   # mm^2
    area_store = m.total_weights * BITS_PER_WEIGHT * SRAM_UM2_PER_BIT * 1e-6 if store.name == "on-chip SRAM" else 0.0
    # 3D-stacked SRAM sits on top of the engine (its own dies), HBM / DDR are separate memory chips
    area_head = m.head_weights * FIXED_CELL_UM2 * 1e-6
    return {"t_load_us": t_load * 1e6, "t_comp_us": t_comp * 1e6, "tokens_per_s_total": engines * batch / t_pass,
            "tokens_per_s_user": 1 / t_pass, "uJ_per_token": e_token * 1e6,
            "area_mm2": area_engines + area_store + area_head, "reload_share": e_load * m.layers / batch / e_token}


def stationary(m: Model):
    """Reference: the current design, every layer fixed on its own chip."""
    e = m.layers * (m.weights_per_layer * E_COMPUTE_PER_WEIGHT + E_MEMORY_PER_TOKEN_LAYER)
    area = (m.total_weights - m.head_weights) * FIXED_CELL_UM2 * 1e-6 + m.head_weights * FIXED_CELL_UM2 * 1e-6
    return {"tokens_per_s_user": 1 / (m.layers * T_LAYER), "tokens_per_s_total": 1 / T_STAGE, "uJ_per_token": e * 1e6, "area_mm2": area}


def table():
    """All models x stores x batches, for the website."""
    out = {}
    for mname, m in MODELS.items():
        rows = []
        for store in STORES.values():
            for batch in (1, 16, 128, 1024):
                r = evaluate(m, store, batch, 1)
                rows.append({"store": store.name, "batch": batch, **{k: round(v, 4) for k, v in r.items()}})
        out[mname] = {"stationary": stationary(m), "multiplexed": rows,
                      "stored_weights": m.total_weights, "loaded_per_layer": m.weights_per_layer, "layers": m.layers}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen3.5-0.8B", choices=list(MODELS))
    p.add_argument("--json", help="write the full table for the website to this file")
    a = p.parse_args()
    if a.json:
        import json
        json.dump(table(), open(a.json, "w"), indent=1)
        print("wrote", a.json)
        return
    m = MODELS[a.model]
    s = stationary(m)
    print(f"== {m.name}: {m.layers} layers, {m.weights_per_layer / 1e6:,.0f}M weights loaded per layer, "
          f"{m.total_weights / 1e9:,.2f}B stored")
    print(f"weight-stationary (one chip per layer): {s['tokens_per_s_user']:,.0f} tok/s per user, "
          f"{s['tokens_per_s_total']:,.0f} tok/s total, {s['uJ_per_token']:.1f} uJ/token, {s['area_mm2']:,.0f} mm2")
    for store in STORES.values():
        for engines in (1, 4):
            for batch in (1, 16, 128, 1024):
                r = evaluate(m, store, batch, engines)
                print(f"{store.name:13s} engines {engines} batch {batch:5d}: load {r['t_load_us']:9.1f} us vs compute "
                      f"{r['t_comp_us']:8.1f} us/layer | {r['tokens_per_s_user']:9,.0f} tok/s per user, "
                      f"{r['tokens_per_s_total']:10,.0f} total | {r['uJ_per_token']:10.1f} uJ/token "
                      f"({100 * r['reload_share']:4.0f}% reload) | {r['area_mm2']:10,.0f} mm2")


if __name__ == "__main__":
    main()
