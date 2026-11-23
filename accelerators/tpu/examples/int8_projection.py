"""每行对称 INT8 量化，以及 TPU 上 INT8×INT8→INT32 的矩阵乘。JAX 0.12.1。"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def quantize_rows(x, *, interpret=False):
    rows, width = x.shape
    assert rows % 8 == 0 and width == 128 and x.dtype == jnp.float32

    def body(x_ref, code_ref, scale_ref):
        a = x_ref[...]
        peak = jnp.max(jnp.abs(a), axis=1, keepdims=True)
        scale = jnp.where(peak == 0, 1.0, peak / 127.0)
        code_ref[...] = jnp.clip(jnp.rint(a / scale), -127, 127).astype(jnp.int8)
        scale_ref[...] = scale

    tile = pl.BlockSpec((8, 128), lambda r: (r, 0))
    scale_tile = pl.BlockSpec((8, 1), lambda r: (r, 0))
    return pl.pallas_call(
        body,
        grid=(rows // 8,),
        in_specs=(tile,),
        out_specs=(tile, scale_tile),
        out_shape=(
            jax.ShapeDtypeStruct(x.shape, jnp.int8),
            jax.ShapeDtypeStruct((rows, 1), jnp.float32),
        ),
        interpret=interpret,
    )(x)


def qgemm(aq, wtq, a_scale, wt_scale, *, interpret=False):
    m, k = aq.shape
    n, _ = wtq.shape
    assert k == 128 and m % 8 == 0 and n % 128 == 0

    def body(a, bt, sa, sbt, out):
        dots = lax.dot(a[...], bt[...].T, preferred_element_type=jnp.int32)
        out[...] = dots.astype(jnp.float32) * sa[...] * sbt[...].T

    specs = (
        pl.BlockSpec((8, 128), lambda i, j: (i, 0)),
        pl.BlockSpec((128, 128), lambda i, j: (j, 0)),
        pl.BlockSpec((8, 1), lambda i, j: (i, 0)),
        pl.BlockSpec((128, 1), lambda i, j: (j, 0)),
    )
    dst = pl.BlockSpec((8, 128), lambda i, j: (i, j))
    return pl.pallas_call(
        body,
        grid=(m // 8, n // 128),
        in_specs=specs,
        out_specs=dst,
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
        interpret=interpret,
    )(aq, wtq, a_scale, wt_scale)


def reference_quant(x):
    peak = np.max(np.abs(x), axis=1, keepdims=True)
    scale = np.where(peak == 0, 1.0, peak / 127.0)
    return np.clip(np.rint(x / scale), -127, 127).astype(np.int8), scale


if __name__ == "__main__":
    cpu = jax.default_backend() == "cpu"
    rng = np.random.default_rng(12)
    a = rng.normal(size=(16, 128)).astype(np.float32)
    wt = rng.normal(size=(256, 128)).astype(np.float32)
    a[0:2] = 0
    a[0, :3] = [1.27, -0.64, 0.01]
    wt[0] = 0
    wt[0, :3] = [2.54, 1.28, -0.02]
    ac, sa = quantize_rows(jnp.asarray(a), interpret=cpu)
    bc, sb = quantize_rows(jnp.asarray(wt), interpret=cpu)
    a0, as0 = reference_quant(a)
    b0, bs0 = reference_quant(wt)
    np.testing.assert_array_equal(ac, a0)
    np.testing.assert_array_equal(bc, b0)
    np.testing.assert_array_max_ulp(np.asarray(sa), as0, maxulp=1)
    np.testing.assert_array_max_ulp(np.asarray(sb), bs0, maxulp=1)
    assert float(sa[1, 0]) == 1.0
    gold = (a0.astype(np.int32) @ b0.astype(np.int32).T).astype(np.float32) * as0 * bs0.T
    y = qgemm(ac, bc, sa, sb, interpret=cpu)
    np.testing.assert_allclose(y, gold, rtol=2e-6, atol=1e-6)
    np.testing.assert_allclose(y[0, 0], 2.4064, rtol=1e-5)
    np.testing.assert_array_equal(y[1], np.zeros(256))
    print(
        jax.default_backend(),
        "quantized codes/scales and INT32 contractions passed",
        "max error to float reference:",
        np.max(np.abs(np.asarray(y) - a @ wt.T)),
    )
