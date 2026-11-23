"""分块转置，同时交换输出块号与块内坐标。JAX 0.12.1。"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def transpose(x, *, interpret=False):
    h, w = x.shape
    assert h > 0 and w > 0 and x.dtype == jnp.float32
    rh, cw = (h + 127) // 128, (w + 127) // 128
    tiles = jnp.pad(x, ((0, rh * 128 - h), (0, cw * 128 - w)))

    def body(a, b):
        b[...] = a[...].T

    src = pl.BlockSpec((128, 128), lambda i, j: (i, j))
    dst = pl.BlockSpec((128, 128), lambda i, j: (j, i))
    call = pl.pallas_call(
        body,
        grid=(rh, cw),
        in_specs=(src,),
        out_specs=dst,
        out_shape=jax.ShapeDtypeStruct((cw * 128, rh * 128), jnp.float32),
        interpret=interpret,
    )
    return call(tiles)[:w, :h]


if __name__ == "__main__":
    cpu = jax.default_backend() == "cpu"
    x = np.arange(129 * 257, dtype=np.float32).reshape(129, 257)
    y = transpose(jnp.asarray(x), interpret=cpu)
    np.testing.assert_array_equal(y, x.T)
    np.testing.assert_equal(float(y[129, 2]), float(x[2, 129]))
    print(jax.default_backend(), "edge tiles and all transpose coordinates passed")
