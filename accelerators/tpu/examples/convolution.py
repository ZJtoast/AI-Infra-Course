"""2×2 有效卷积。JAX 0.12.1，CPU 用 Pallas 解释器，TPU 默认使用 Mosaic。"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def split_input(x, tile):
    h, w, c = x.shape
    assert h >= 2 and w >= 2 and c == 128 and tile in (8, 16, 32)
    stripes = (w - 1 + tile - 1) // tile
    padded_width = (stripes + 1) * tile  # 最后一条也能安全预取右侧相邻窗
    data = jnp.pad(x.astype(jnp.float32), ((0, 0), (0, padded_width - w), (0, 0)))
    specs = tuple(
        pl.BlockSpec((None, tile, c), lambda y, b, dy=dy, dx=dx: (y + dy, b + dx, 0))
        for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1))
    )
    dst = pl.BlockSpec((None, tile, c), lambda y, b: (y, b, 0))
    return data, specs, dst, stripes


def next_pixel(strip, right):
    # 输出的第 j 项需读输入的第 j+1 个像素；最后一个像素从相邻窗头部补入
    return jnp.concatenate((strip[1:, :], right[:1, :]), axis=0)


def tiled_conv(x, weight, *, tile=8, interpret=False):
    data, xs, dst, stripes = split_input(x, tile)
    h, w, c = x.shape
    assert weight.shape == (2, 2, c, c)
    ws = tuple(
        pl.BlockSpec((None, None, c, c), lambda y, b, i=i, j=j: (i, j, 0, 0))
        for i, j in ((0, 0), (0, 1), (1, 0), (1, 1))
    )

    def body(top, top_halo, bottom, bottom_halo, w00, w01, w10, w11, out):
        top_data, bottom_data = top[...], bottom[...]
        top_next = next_pixel(top_data, top_halo[...])
        bottom_next = next_pixel(bottom_data, bottom_halo[...])

        def product(a, b):
            return lax.dot(a, b[...], preferred_element_type=jnp.float32)

        out[...] = (
            product(top_data, w00)
            + product(top_next, w01)
            + product(bottom_data, w10)
            + product(bottom_next, w11)
        )

    src = data.astype(jnp.bfloat16)
    filt = weight.astype(jnp.bfloat16)
    run = pl.pallas_call(
        body,
        grid=(h - 1, stripes),
        in_specs=xs + ws,
        out_specs=dst,
        out_shape=jax.ShapeDtypeStruct((h - 1, stripes * tile, c), jnp.float32),
        interpret=interpret,
    )
    return run(src, src, src, src, filt, filt, filt, filt)[:, : w - 1, :]


def reference(x, w):
    x, w = np.asarray(x, np.float32), np.asarray(w, np.float32)
    oh, ow = x.shape[0] - 1, x.shape[1] - 1
    return sum(
        x[dy : dy + oh, dx : dx + ow] @ w[dy, dx] for dy in range(2) for dx in range(2)
    )


if __name__ == "__main__":
    cpu = jax.default_backend() == "cpu"
    rng = np.random.default_rng(2)
    x = jnp.asarray(rng.normal(size=(4, 19, 128)), jnp.bfloat16)
    w = jnp.asarray(rng.normal(size=(2, 2, 128, 128)) / 16, jnp.bfloat16)
    gold = reference(x, w)
    for tile in (8, 16):
        np.testing.assert_allclose(
            tiled_conv(x, w, tile=tile, interpret=cpu), gold, atol=2e-4, rtol=2e-4
        )
    x = np.zeros((3, 3, 128), np.float32)
    x[:, :, 0] = [[1, 2, 4], [3, 5, 8], [6, 9, 13]]
    w = np.zeros((2, 2, 128, 128), np.float32)
    w[:, :, 0, 0] = [[-1, 1], [-1, 1]]
    y = tiled_conv(jnp.asarray(x), jnp.asarray(w), interpret=cpu)
    np.testing.assert_array_equal(np.asarray(y)[:, :, 0], [[3, 5], [5, 7]])
    print(
        jax.default_backend(),
        "halo, tail, two tile widths and convolution reference passed",
    )
