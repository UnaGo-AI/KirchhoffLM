"""Analog DeltaNet memory: the gated delta rule on a capacitor array, simulated exactly.

Each state cell S[h, i, j] is a capacitor. Per token the array
  1. leaks / forgets: S <- g_t * S             (g_t includes the capacitor leakage)
  2. reads:          m = S k_t                  (k drives the rows, currents sum on columns)
  3. writes:         S <- S + (1 + M) * (k_t delta^T) + n_t,   delta = beta (v - m)
  4. reads out:      y = S q_t
M is a static per-cell write-gain mismatch; n_t is fresh kT/C charge noise on every write,
generated in the kernel with a counter-based hash RNG.

Optional level restoration ("refresh"): every R tokens comparators snap each cell to the nearest
of 2^bits - 1 levels spanning the head's full scale [-fs, fs] and recharge it to that level.
Noise then accumulates only between refreshes. Stochastic rounding (dither) rounds up or down
with probability given by the distance, so updates smaller than one step are not lost. Because the write uses the read of the
noisy state, the delta rule's own error correction acts on the noise, as in the real circuit.
"""

import mlx.core as mx

HEADER = """
inline uint pcg_hash(uint v) {
  uint s = v * 747796405u + 2891336453u;
  uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
inline float unif(uint v) {
  return (float(pcg_hash(v) >> 8) + 0.5f) * (1.0f / 16777216.0f);
}
inline float gauss(uint a, uint b) {
  float u1 = unif(a), u2 = unif(b);
  return sqrt(-2.0f * log(u1)) * cos(6.2831853f * u2);
}
"""

SOURCE = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;
        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;
        auto mism_ = mism + (hv_idx * Dv + dv_idx) * Dk;
        float sig = sigma[hv_idx];
        uint seed = uint(seedp[0]);
        float qfs = fs[hv_idx];
        float qlev = qp[0];          // levels per side (2^(bits-1) - 1); 0 = no refresh
        int qevery = int(qp[1]);
        bool qstoch = qp[2] > 0.5f;
        float state[n_per_t];
        float mm[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
          mm[i] = 1.0f + mism_[s_idx];
        }
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;
        uint cell0 = uint((n * Dv + dv_idx) * Dk);
        for (int t = 0; t < T; ++t) {
          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * g_[hv_idx];
            kv_mem += state[i] * k_[s_idx];
          }
          kv_mem = simd_sum(kv_mem);
          auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];
          float out = 0.0f;
          uint base = seed ^ (uint(t) * 2654435761u);
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            uint c = cell0 + uint(s_idx);
            float noise = sig * gauss(base ^ (c * 2u + 1u) * 2246822519u, (base + 0x68bc21ebu) ^ (c * 2u) * 3266489917u);
            state[i] = state[i] + k_[s_idx] * delta * mm[i] + noise;
            if (qlev > 0.0f && ((t + 1) % qevery == 0)) {
              float u = clamp(state[i] / qfs, -1.0f, 1.0f) * qlev;
              float r = qstoch ? floor(u + unif((base + 0x2545F491u) ^ (c * 2654435761u))) : rint(u);
              state[i] = r / qlev * qfs;
            }
            out += state[i] * q_[s_idx];
          }
          out = simd_sum(out);
          if (thread_index_in_simdgroup == 0) {
            y[dv_idx] = static_cast<InT>(out);
          }
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }
"""

_KERNEL = mx.fast.metal_kernel(
    name="analog_gated_delta",
    input_names=["q", "k", "v", "g", "beta", "state_in", "mism", "sigma", "seedp", "fs", "qp", "T"],
    output_names=["y", "state_out"],
    header=HEADER,
    source=SOURCE,
)


def analog_gated_delta(q, k, v, g, beta, state, mism, sigma, seed, fs=1.0, bits=0, every=1, stochastic=True):
    """q, k: [B,T,Hk,Dk]; v: [B,T,Hv,Dv]; g, beta: [B,T,Hv]; state: [B,Hv,Dv,Dk] (float32);
    mism: [Hv,Dv,Dk] static write-gain mismatch; sigma: per-write noise per head [Hv] (state units);
    fs: per-head full scale [Hv]; bits: refresh precision (0 = off); every: refresh period in tokens."""
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    if Dk % 32:
        raise ValueError("Dk must be a multiple of 32")
    sigma = mx.broadcast_to(mx.array(sigma, dtype=mx.float32), (Hv,))
    seedp = mx.array([float(seed % 16777216)], dtype=mx.float32)
    fs = mx.broadcast_to(mx.array(fs, dtype=mx.float32), (Hv,))
    qp = mx.array([float(2 ** (bits - 1) - 1) if bits else 0.0, float(max(int(every), 1)), 1.0 if stochastic else 0.0])
    return _KERNEL(
        inputs=[q, k, v, g, beta, state, mism, sigma, seedp, fs, qp, T],
        template=[("InT", q.dtype), ("StT", state.dtype), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv)],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape],
        output_dtypes=[q.dtype, state.dtype],
    )
