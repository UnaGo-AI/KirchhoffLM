"""The analog DeltaNet kernel must equal mlx-lm's kernel when noise and mismatch are zero."""
import mlx.core as mx
from mlx_lm.models.gated_delta import gated_delta_kernel, normalize_qk
from memkernel import analog_gated_delta

mx.random.seed(0)
B, T, H, D = 1, 300, 16, 128
q, k = normalize_qk(mx.random.normal((B, T, H, D)), mx.random.normal((B, T, H, D)), inv_scale=D ** -0.5, eps=1e-6)
v = mx.random.normal((B, T, H, D))
g = mx.random.uniform(0.8, 1.0, (B, T, H))
beta = mx.random.uniform(0.0, 1.0, (B, T, H))
s0 = mx.zeros((B, H, D, D))
y_ref, s_ref = gated_delta_kernel(q, k, v, g, beta, s0, None)
y, s = analog_gated_delta(q, k, v, g, beta, s0, mx.zeros((H, D, D)), 0.0, 123)
print("zero-noise max |dy| =", float(mx.abs(y - y_ref).max()), " max |dS| =", float(mx.abs(s - s_ref).max()))
y1, _ = analog_gated_delta(q, k, v, g, beta, s0, mx.zeros((H, D, D)), 1e-3, 123)
y2, _ = analog_gated_delta(q, k, v, g, beta, s0, mx.zeros((H, D, D)), 1e-3, 123)
y3, _ = analog_gated_delta(q, k, v, g, beta, s0, mx.zeros((H, D, D)), 1e-3, 124)
print("same seed identical:", float(mx.abs(y1 - y2).max()) == 0.0, "| other seed differs:", float(mx.abs(y1 - y3).max()) > 0)
print("noise 1e-3 -> rms output error", float(mx.sqrt(((y1 - y_ref) ** 2).mean())), "vs rms y", float(mx.sqrt((y_ref ** 2).mean())))
_, sn = analog_gated_delta(q, k, v, mx.ones_like(g), mx.zeros_like(beta), s0, mx.zeros((H, D, D)), 1e-3, 7)
print("pure accumulation (g=1, beta=0): state std", float(sn.std()), "expected", 1e-3 * T ** 0.5, "| mean", float(sn.mean()))

# refresh: with 16-bit nearest rounding every token the result stays close to exact;
# with 4 bits it is coarse; no noise, so the difference is pure rounding error
for bits in (4, 8, 12, 16):
    yq, _ = analog_gated_delta(q, k, v, g, beta, s0, mx.zeros((H, D, D)), 0.0, 5, fs=float(mx.abs(s_ref).max()) * 1.5, bits=bits, every=1, stochastic=False)
    print(f"refresh {bits:2d} bit, nearest: rms output error / rms y = {float(mx.sqrt(((yq - y_ref) ** 2).mean()) / mx.sqrt((y_ref ** 2).mean())):.2e}")
