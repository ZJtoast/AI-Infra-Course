"""TPU Pallas 的 SwiGLU：两次设备调用与单核融合，JAX 0.12.1。"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl


def swiglu(gate, up, *, fused=True, interpret=False):
    rows, cols = gate.shape
    assert rows > 0 and cols > 0 and gate.shape == up.shape
    assert gate.dtype == up.dtype == jnp.float32
    pad = ((0, (-rows) % 8), (0, (-cols) % 128))
    g, u = jnp.pad(gate, pad), jnp.pad(up, pad)
    tile = pl.BlockSpec((8, 128), lambda i, j: (i, j))
    grid = (g.shape[0] // 8, g.shape[1] // 128)
    result = jax.ShapeDtypeStruct(g.shape, jnp.float32)

    def activation(a, y):
        z = a[...]
        y[...] = z * lax.logistic(z)  # 每个特征形成自己的门值

    def multiply(s, b, y):
        y[...] = s[...] * b[...]  # 只配对同一行、同一通道

    def combined(a, b, y):
        z = a[...]
        y[...] = (z * lax.logistic(z)) * b[...]  # 门值直接供当前窗口乘法使用

    if fused:
        run = pl.pallas_call(
            combined,
            grid=grid,
            in_specs=(tile, tile),
            out_specs=tile,
            out_shape=result,
            interpret=interpret,
        )
        y = run(g, u)
    else:
        act = pl.pallas_call(
            activation,
            grid=grid,
            in_specs=(tile,),
            out_specs=tile,
            out_shape=result,
            interpret=interpret,
        )
        mul = pl.pallas_call(
            multiply,
            grid=grid,
            in_specs=(tile, tile),
            out_specs=tile,
            out_shape=result,
            interpret=interpret,
        )
        y = mul(act(g), u)
    return y[:rows, :cols]


if __name__ == "__main__":
    cpu = jax.default_backend() == "cpu"
    rng = np.random.default_rng(43)
    gate = rng.normal(size=(13, 131)).astype(np.float32)
    up = rng.normal(size=gate.shape).astype(np.float32)
    gate[0, :4] = [0.0, np.log(3.0), -np.log(3.0), 2.0]
    up[0, :4] = [5.0, 4.0, 4.0, -1.0]
    reference = up * (gate / (1.0 + np.exp(-gate)))
    for fused in (False, True):
        y = swiglu(jnp.asarray(gate), jnp.asarray(up), fused=fused, interpret=cpu)
        np.testing.assert_allclose(y, reference, rtol=2e-6, atol=1e-7)
    print(jax.default_backend(), "separate and fused SwiGLU passed", np.asarray(y[0, :4]))
