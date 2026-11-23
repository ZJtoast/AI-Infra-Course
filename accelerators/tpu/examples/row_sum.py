import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl


def partial_sum(x_ref, sums_ref):
    sums_ref[...] = jnp.sum(x_ref[...], axis=1)[:, None]  # 八行各自沿列求和


def row_sums(x, interpret=False):
    row_blocks, col_blocks = x.shape[0] // 8, x.shape[1] // 128
    tile = pl.BlockSpec((8, 128), lambda row, col: (row, col))
    # 第二维是每个小块的八行；长度 1 的轴只存一个部分和
    out = pl.BlockSpec((None, 8, 1), lambda row, col: (col, row, 0))
    partial = pl.pallas_call(
        partial_sum,
        grid=(row_blocks, col_blocks),
        in_specs=(tile,),
        out_specs=out,
        out_shape=jax.ShapeDtypeStruct((col_blocks, x.shape[0], 1), x.dtype),
        interpret=interpret,
    )(x)
    return partial, partial.sum(axis=0)[:, 0]  # 对同一行的所有列块求和


if __name__ == "__main__":
    cpu = jax.default_backend() == "cpu"
    x = jnp.broadcast_to(jnp.arange(256, dtype=jnp.float32), (16, 256))
    part, total = row_sums(x, interpret=cpu)
    np.testing.assert_array_equal(np.asarray(part[:, 0, 0]), [8128, 24512])
    np.testing.assert_array_equal(np.asarray(total), np.full(16, 32640))
    print(jax.default_backend(), "all outputs passed")
