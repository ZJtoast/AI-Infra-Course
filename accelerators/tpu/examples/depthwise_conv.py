"""2x2 depthwise cross-correlation, built on convolution.py's Pallas input windows."""
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

from convolution import next_pixel, split_input


def depthwise_conv(x, weights, tile=8, *, interpret=False):
    data, windows, dest, stripes = split_input(x, tile)
    height, width, channels = x.shape
    assert weights.shape == (2, 2, channels)
    filters = tuple(
        pl.BlockSpec((None, None, 1, channels),
                     lambda y, b, dy=dy, dx=dx: (dy, dx, 0, 0))
        for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1))
    )

    def depthwise(top, right_top, bottom, right_bottom, w00, w01, w10, w11, out):
        a = top[...].astype(jnp.float32)
        b = next_pixel(a, right_top[...].astype(jnp.float32))
        c = bottom[...].astype(jnp.float32)
        d = next_pixel(c, right_bottom[...].astype(jnp.float32))
        out[...] = a*w00[...] + b*w01[...] + c*w10[...] + d*w11[...]

    run = pl.pallas_call(depthwise, grid=(height - 1, stripes),
                         in_specs=windows + filters, out_specs=dest,
                         out_shape=jax.ShapeDtypeStruct((height-1, stripes*tile, channels),
                                                        jnp.float32), interpret=interpret)
    filt = jnp.asarray(weights, jnp.float32).reshape(2, 2, 1, channels)
    return run(data, data, data, data, filt, filt, filt, filt)[:, :width-1]


def main():
    cpu = jax.default_backend() == "cpu"
    x = np.zeros((2, 9, 128), np.float32)
    x[:, 7:9, :2] = [[[1, 2], [3, 5]], [[4, 1], [6, 3]]]
    w = np.zeros((2, 2, 128), np.float32)
    w[:, :, 0], w[:, :, 1] = [[-1, 1], [-1, 1]], [[1, 1], [0, 0]]
    y = depthwise_conv(jnp.asarray(x), w, interpret=cpu)
    np.testing.assert_array_equal(y[0, 7, :2], [4, 7])
    rng = np.random.default_rng(11)
    x = rng.normal(size=(4, 19, 128)).astype(np.float32)
    w = rng.normal(size=(2, 2, 128)).astype(np.float32)
    expected = sum(x[dy:dy+3, dx:dx+18] * w[dy, dx] for dy in range(2) for dx in range(2))
    for tile in (8, 16):
        actual = depthwise_conv(jnp.asarray(x), w, tile, interpret=cpu)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    print(jax.default_backend(), "depthwise hand example, halo and channel comparison passed")


if __name__ == "__main__":
    main()
