"""带右侧相邻窗的最大池化。JAX 0.12.1。"""

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


def tiled_pool(x, *, tile=8, interpret=False):
    data, xs, dst, stripes = split_input(x, tile)
    h, w, c = x.shape

    def body(top, top_halo, bottom, bottom_halo, out):
        a, b = top[...], bottom[...]
        top_pair = jnp.maximum(a, next_pixel(a, top_halo[...]))
        bottom_pair = jnp.maximum(b, next_pixel(b, bottom_halo[...]))
        out[...] = jnp.maximum(top_pair, bottom_pair)

    run = pl.pallas_call(
        body,
        grid=(h - 1, stripes),
        in_specs=xs,
        out_specs=dst,
        out_shape=jax.ShapeDtypeStruct((h - 1, stripes * tile, c), jnp.float32),
        interpret=interpret,
    )
    return run(data, data, data, data)[:, : w - 1, :]


def tiled_pool_choices(x, *, tile=8, interpret=False):
    """为反向传播保存每个窗口的胜者，左上、右上、左下、右下分别编号 0 至 3。"""
    data, specs, dst, stripes = split_input(x, tile)
    h, w, c = x.shape

    def body(top, right_top, bottom, right_bottom, chosen):
        a, c0 = top[...], bottom[...]
        b, d = next_pixel(a, right_top[...]), next_pixel(c0, right_bottom[...])
        peak = jnp.maximum(jnp.maximum(a, b), jnp.maximum(c0, d))
        chosen[...] = jnp.where(
            a == peak, 0, jnp.where(b == peak, 1, jnp.where(c0 == peak, 2, 3))
        )

    run = pl.pallas_call(
        body,
        grid=(h - 1, stripes),
        in_specs=specs,
        out_specs=dst,
        out_shape=jax.ShapeDtypeStruct((h - 1, stripes * tile, c), jnp.int32),
        interpret=interpret,
    )
    return run(data, data, data, data)[:, : w - 1, :]


def tiled_pool_backward(winner, dy, *, tile=8, interpret=False):
    """每个输入像素向最多四个输出窗口收集梯度，每个地址只写一次。"""
    h, w, c = (winner.shape[0] + 1, winner.shape[1] + 1, winner.shape[2])
    assert winner.shape == dy.shape and c == 128
    padding = ((1, 1), (1, 1), (0, 0))
    wi, specs, dst, stripes = split_input(
        jnp.pad(winner, padding, constant_values=-1), tile
    )
    grads, _, _, _ = split_input(jnp.pad(dy, padding), tile)

    def body(it, itr, ib, ibr, gt, gtr, gb, gbr, dx):
        # 以输入像素为中心，取上方与同行各两个可能包含它的输出窗口。
        top, bottom = it[...], ib[...]
        top_grad, bottom_grad = gt[...], gb[...]
        a = jnp.where(top == 3, top_grad, 0.0)
        b = jnp.where(next_pixel(top, itr[...]) == 2, next_pixel(top_grad, gtr[...]), 0.0)
        c0 = jnp.where(bottom == 1, bottom_grad, 0.0)
        d = jnp.where(
            next_pixel(bottom, ibr[...]) == 0, next_pixel(bottom_grad, gbr[...]), 0.0
        )
        dx[...] = (a + b) + (c0 + d)

    run = pl.pallas_call(
        body,
        grid=(h, stripes),
        in_specs=specs + specs,
        out_specs=dst,
        out_shape=jax.ShapeDtypeStruct((h, stripes * tile, c), jnp.float32),
        interpret=interpret,
    )
    return run(wi, wi, wi, wi, grads, grads, grads, grads)[:, :w, :]


if __name__ == "__main__":
    cpu = jax.default_backend() == "cpu"
    rng = np.random.default_rng(3)
    x = -1 - np.abs(rng.normal(size=(5, 19, 128))).astype(np.float32)
    x[2, 8, 0] = 99.0  # 右边相邻窗的首格
    gold = np.maximum.reduce(
        [x[dy : dy + 4, dx : dx + 18] for dy in range(2) for dx in range(2)]
    )
    for tile in (8, 16):
        y = tiled_pool(jnp.asarray(x), tile=tile, interpret=cpu)
        np.testing.assert_array_equal(y, gold)
    np.testing.assert_array_equal(gold[1, 7:9, 0], [99.0, 99.0])
    # 为每个输出位置分配不同的整型梯度，四邻域中读错窗口时可直接定位。
    dy = np.broadcast_to(
        np.arange(4 * 18, dtype=np.float32).reshape(4, 18, 1) + 1, gold.shape
    ).copy()
    x[0:2, 0:2, 1] = 42.0  # 第一列以外另选一个通道，制造四项并列最大
    patch = np.stack(
        [x[yy : yy + 4, xx : xx + 18, :] for yy in range(2) for xx in range(2)]
    )
    index = np.argmax(patch, axis=0).astype(np.int32)
    chosen = tiled_pool_choices(jnp.asarray(x), interpret=cpu)
    np.testing.assert_array_equal(chosen, index)
    dx = tiled_pool_backward(chosen, jnp.asarray(dy), interpret=cpu)
    ref_grad = np.zeros_like(x)
    for idx, (yy, xx) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        ref_grad[yy : yy + 4, xx : xx + 18, :] += np.where(index == idx, dy, 0.0)
    np.testing.assert_array_equal(dx, ref_grad)
    np.testing.assert_array_equal(
        dx[2, 8, 0], dy[1, 7, 0] + dy[1, 8, 0] + dy[2, 7, 0] + dy[2, 8, 0]
    )
    print(jax.default_backend(), "overlap, negative data, halo and last stripe passed")
