import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl


def rms_kernel(x_ref, gain_ref, y_ref):
    x = x_ref[...].astype(jnp.float32)  # 一次取八行，每行 128 个输入
    square_sum = jnp.sum(x * x, axis=1)  # 每行产生一个平方和
    inv = lax.rsqrt(square_sum / 128 + 1e-6)  # 一行只求一次平方根倒数
    y_ref[...] = x * inv[:, None] * gain_ref[...][None, :]  # 用本通道的增益缩放输出


def rms(x, gain, interpret=False):
    rows = pl.BlockSpec((8, 128), lambda i: (i, 0))
    weight = pl.BlockSpec((128,), lambda i: (0,))
    return pl.pallas_call(
        rms_kernel,
        grid=(x.shape[0] // 8,),
        in_specs=(rows, weight),
        out_specs=rows,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=interpret,
    )(x, gain)


if __name__ == "__main__":
    cpu = jax.default_backend() == "cpu"
    x = jnp.tile(jnp.array([3.0, 4.0], dtype=jnp.float32), (16, 64))
    gain = jnp.tile(jnp.array([2.0, 1.0]), 64)
    expected = np.asarray(x) / np.sqrt(12.5 + 1e-6) * np.asarray(gain)
    np.testing.assert_allclose(rms(x, gain, interpret=cpu), expected, rtol=1e-5)
    print(jax.default_backend(), "all outputs passed")
