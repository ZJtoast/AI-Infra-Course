"""长行 softmax：第一遍按块合并最大值及分母，第二遍写各块概率。JAX 0.12.1。"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def streaming_softmax(x, *, interpret=False):
    rows, width = x.shape
    assert rows % 8 == 0 and width > 0 and x.dtype == jnp.float32
    blocks = (width + 127) // 128
    pad = jnp.pad(x, ((0, 0), (0, blocks * 128 - width)))
    tile = pl.BlockSpec((8, 128), lambda r, b: (r, b))
    stat = pl.BlockSpec((8, 1), lambda r, b: (r, 0))

    def accumulate(xr, m_out, l_out, m, l):
        b = pl.program_id(1)

        @pl.when(b == 0)
        def clear():
            m[...] = jnp.full((8, 1), -1e30, jnp.float32)
            l[...] = jnp.zeros((8, 1), jnp.float32)

        valid = (b * 128 + jnp.arange(128))[None, :] < width
        a = jnp.where(valid, xr[...], -1e30)
        new = jnp.maximum(m[...], jnp.max(a, axis=1, keepdims=True))
        l[...] = l[...] * jnp.exp(m[...] - new) + jnp.sum(
            jnp.where(valid, jnp.exp(a - new), 0.0), axis=1, keepdims=True
        )
        m[...] = new

        @pl.when(b == blocks - 1)
        def save():
            m_out[...], l_out[...] = m[...], l[...]

    grid = (rows // 8, blocks)
    out_type = jax.ShapeDtypeStruct((rows, 1), jnp.float32)
    m, l = pl.pallas_call(
        accumulate,
        grid=grid,
        in_specs=(tile,),
        out_specs=(stat, stat),
        scratch_shapes=(pltpu.VMEM((8, 1), jnp.float32), pltpu.VMEM((8, 1), jnp.float32)),
        out_shape=(out_type, out_type),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
        interpret=interpret,
    )(pad)

    def normalize(xr, mr, lr, y):
        valid = (pl.program_id(1) * 128 + jnp.arange(128))[None, :] < width
        y[...] = jnp.where(valid, jnp.exp(xr[...] - mr[...]) / lr[...], 0.0)

    run = pl.pallas_call(
        normalize,
        grid=grid,
        in_specs=(tile, stat, stat),
        out_specs=tile,
        out_shape=jax.ShapeDtypeStruct(pad.shape, jnp.float32),
        interpret=interpret,
    )
    return run(pad, m, l)[:, :width]


if __name__ == "__main__":
    cpu = jax.default_backend() == "cpu"
    rng = np.random.default_rng(13)
    for width in (1, 127, 128, 129, 259):
        x = rng.normal(size=(16, width)).astype(np.float32)
        x[0, :] = -5.0
        x[0, -1] = 5.0
        e = np.exp(x - x.max(axis=1, keepdims=True))
        np.testing.assert_allclose(
            streaming_softmax(jnp.asarray(x), interpret=cpu),
            e / e.sum(axis=1, keepdims=True),
            rtol=2e-5,
            atol=1e-6,
        )
    print(jax.default_backend(), "last column and both sides of each block boundary passed")
