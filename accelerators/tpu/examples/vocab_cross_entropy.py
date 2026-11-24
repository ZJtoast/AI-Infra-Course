"""JAX 0.12.1. Class-index cross entropy for finite FP32 logits on TPU."""
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def vocab_loss(logits, labels, *, interpret=False):
    """Return one loss per row. labels contains one valid vocabulary index per row."""
    batch, vocab = logits.shape
    if not batch or not vocab or labels.shape != (batch,):
        raise ValueError("expected nonempty [batch, vocab] and [batch] inputs")
    if logits.dtype != jnp.float32 or labels.dtype != jnp.int32:
        raise TypeError("expected float32 logits and int32 labels")

    rows, cols = 8, 128
    row_blocks = (batch + rows - 1) // rows
    col_blocks = (vocab + cols - 1) // cols
    row_pad = row_blocks * rows - batch
    x = jnp.pad(logits, ((0, row_pad), (0, col_blocks * cols - vocab)))
    ids = jnp.pad(labels[:, None], ((0, row_pad), (0, 0)))

    def kernel(x_ref, label_ref, loss_ref, peak, denom, target):
        b = pl.program_id(1)

        @pl.when(b == 0)
        def init():
            peak[...] = jnp.full((rows, 1), -jnp.inf, jnp.float32)
            denom[...] = jnp.zeros((rows, 1), jnp.float32)
            target[...] = jnp.zeros((rows, 1), jnp.float32)

        col = (b * cols + jnp.arange(cols))[None, :]
        score = x_ref[...]
        valid = col < vocab
        masked = jnp.where(valid, score, -jnp.inf)
        new_peak = jnp.maximum(peak[...], jnp.max(masked, axis=1, keepdims=True))
        denom[...] = denom[...] * jnp.exp(peak[...] - new_peak) + jnp.sum(
            jnp.exp(masked - new_peak), axis=1, keepdims=True)
        target[...] += jnp.sum(jnp.where(col == label_ref[...], score, 0.0),
                               axis=1, keepdims=True)
        peak[...] = new_peak

        @pl.when(b == col_blocks - 1)
        def emit():
            loss_ref[...] = (peak[...] - target[...]) + jnp.log(denom[...])

    tile = pl.BlockSpec((rows, cols), lambda r, b: (r, b))
    row = pl.BlockSpec((rows, 1), lambda r, b: (r, 0))
    state = pltpu.VMEM((rows, 1), jnp.float32)
    call = pl.pallas_call(kernel, grid=(row_blocks, col_blocks), in_specs=(tile, row),
                          out_specs=row, out_shape=jax.ShapeDtypeStruct(ids.shape, jnp.float32),
                          scratch_shapes=(state, state, state),
                          compiler_params=pltpu.CompilerParams(
                              dimension_semantics=("parallel", "arbitrary")),
                          interpret=interpret)
    return call(x, ids)[:batch, 0]


if __name__ == "__main__":
    interp = jax.default_backend() == "cpu"
    scores = jnp.array([[1.2, -0.8, -0.5], [0.9, -1.2, 1.1]], jnp.float32)
    example = vocab_loss(scores, jnp.array([0, 1], jnp.int32), interpret=interp)
    np.testing.assert_allclose(example, [0.2761, 2.9518], atol=5e-5)
    rng = np.random.default_rng(71)
    for batch, vocab in [(1, 1), (8, 128), (9, 129), (3, 257)]:
        x = rng.normal(size=(batch, vocab)).astype(np.float32) + 1000
        ids = np.arange(batch, dtype=np.int32) % vocab
        ids[-1] = vocab - 1
        y = vocab_loss(jnp.asarray(x), jnp.asarray(ids), interpret=interp)
        shifted = x.astype(np.float64) - x.max(axis=1, keepdims=True)
        ref = np.log(np.exp(shifted).sum(1)) - shifted[np.arange(batch), ids]
        np.testing.assert_allclose(y, ref, atol=2e-6)
    x = jnp.zeros((2, 129), jnp.float32).at[:, 128].set(jnp.log(4.0))
    y = vocab_loss(x, jnp.array([128, 0], jnp.int32), interpret=interp)
    np.testing.assert_allclose(y, np.log([33.0, 132.0]), atol=2e-6)
    print(jax.default_backend(), "four boundary cases and two hand-calculated losses passed")
