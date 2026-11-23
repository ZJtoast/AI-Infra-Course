"""相同 GEMM 的两种设备核：部分积落地，或 VMEM 累加后融合偏置与 ReLU。JAX 0.12.1。"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

TILE = 128


def project(a, b, bias, *, fused=True, buffers=2, interpret=False):
    m, k = a.shape
    n = b.shape[1]
    assert (
        b.shape == (k, n) and bias.shape == (n,) and m % TILE == k % TILE == n % TILE == 0
    )
    grid = (m // TILE, n // TILE, k // TILE)
    tile = (TILE, TILE)
    mode = pl.Buffered(buffer_count=buffers)
    ab = (
        pl.BlockSpec(tile, lambda i, j, s: (i, s), pipeline_mode=mode),
        pl.BlockSpec(tile, lambda i, j, s: (s, j), pipeline_mode=mode),
    )

    def dot(left, right):
        return lax.dot(left[...], right[...], preferred_element_type=jnp.float32)

    if not fused:

        def partial(left, right, part):
            part[...] = dot(left, right)

        out = pl.BlockSpec((None, TILE, TILE), lambda i, j, s: (s, i, j))
        p = pl.pallas_call(
            partial,
            grid=grid,
            in_specs=ab,
            out_specs=out,
            out_shape=jax.ShapeDtypeStruct((grid[2], m, n), jnp.float32),
            interpret=interpret,
        )(a, b)
        return jnp.maximum(jnp.sum(p, axis=0) + bias, 0)

    def finish(left, right, b_ref, y, acc):
        s = pl.program_id(2)

        @pl.when(s == 0)
        def zero():
            acc[...] = jnp.zeros(tile, jnp.float32)

        acc[...] = acc[...] + dot(left, right)

        @pl.when(s == grid[2] - 1)
        def store():
            y[...] = jnp.maximum(acc[...] + b_ref[...][None, :], 0)

    bspec = pl.BlockSpec((TILE,), lambda i, j, s: (j,))
    return pl.pallas_call(
        finish,
        grid=grid,
        in_specs=ab + (bspec,),
        out_specs=pl.BlockSpec(tile, lambda i, j, s: (i, j)),
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
        scratch_shapes=(pltpu.VMEM(tile, jnp.float32),),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        interpret=interpret,
    )(a, b, bias)


if __name__ == "__main__":
    cpu = jax.default_backend() == "cpu"
    x = np.zeros((128, 256), np.float32)
    b = np.zeros((256, 256), np.float32)
    x[:, [0, 128]] = [2.0, 3.0]
    b[0, [0, 128]] = [4.0, -1.0]
    b[128, [0, 128]] = [1.0, 2.0]
    bias = np.zeros(256, np.float32)
    bias[[0, 128]] = [-12.0, 1.0]
    gold = np.maximum(x @ b + bias, 0.0)
    a, bt, offset = (
        jnp.asarray(x, jnp.bfloat16),
        jnp.asarray(b, jnp.bfloat16),
        jnp.asarray(bias),
    )
    for fuse, slots in ((False, 1), (True, 1), (True, 2)):
        np.testing.assert_array_equal(
            project(a, bt, offset, fused=fuse, buffers=slots, interpret=cpu), gold
        )
    print(
        jax.default_backend(),
        "reference, materialized products and both VMEM schedules passed",
    )
