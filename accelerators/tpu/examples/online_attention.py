"""分块因果注意力，JAX 0.12.1。直接运行此文件会比较设备核与稠密基准。"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BLOCK = 128
WIDTH = 128


def tiled_attention(q, k, v, *, interpret=False):
    length = q.shape[0]
    assert length > 0 and q.shape == k.shape == v.shape == (length, WIDTH)
    tiles = (length + BLOCK - 1) // BLOCK
    padded = tiles * BLOCK

    def pad(x):
        return jnp.pad(x.astype(jnp.float32), ((0, padded - length), (0, 0))).astype(
            jnp.bfloat16
        )

    def body(q_ref, k_ref, v_ref, out_ref, max_ref, sum_ref, acc_ref):
        qb, kb = pl.program_id(0), pl.program_id(1)

        @pl.when(kb == 0)
        def initialize():
            max_ref[...] = jnp.full((BLOCK, 1), -1e30, jnp.float32)
            sum_ref[...] = jnp.zeros((BLOCK, 1), jnp.float32)
            acc_ref[...] = jnp.zeros((BLOCK, WIDTH), jnp.float32)

        @pl.when(kb <= qb)
        def merge():
            qi = qb * BLOCK + jnp.arange(BLOCK)[:, None]
            ki = kb * BLOCK + jnp.arange(BLOCK)[None, :]
            legal = (qi < length) & (ki < length) & (ki <= qi)
            scores = lax.dot(q_ref[...], k_ref[...].T, preferred_element_type=jnp.float32)
            scores = jnp.where(legal, scores / np.sqrt(WIDTH), -1e30)
            new_max = jnp.maximum(max_ref[...], jnp.max(scores, axis=1, keepdims=True))
            scale = jnp.exp(max_ref[...] - new_max)  # 旧分子分母换用新的指数基准
            weights = jnp.where(legal, jnp.exp(scores - new_max), 0.0)
            sum_ref[...] = sum_ref[...] * scale + jnp.sum(weights, axis=1, keepdims=True)
            chunk = lax.dot(
                weights.astype(jnp.bfloat16), v_ref[...], preferred_element_type=jnp.float32
            )
            acc_ref[...] = acc_ref[...] * scale + chunk  # 旧分子和新块的贡献合并
            max_ref[...] = new_max

        @pl.when(kb == tiles - 1)
        def finish():
            den = sum_ref[...]
            out_ref[...] = acc_ref[...] / jnp.where(den > 0, den, 1.0)

    q_tile = pl.BlockSpec((BLOCK, WIDTH), lambda qb, kb: (qb, 0))
    kv_tile = pl.BlockSpec((BLOCK, WIDTH), lambda qb, kb: (kb, 0))
    spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        grid=(tiles, tiles),
        in_specs=(q_tile, kv_tile, kv_tile),
        out_specs=q_tile,
        scratch_shapes=(
            pltpu.VMEM((BLOCK, 1), jnp.float32),
            pltpu.VMEM((BLOCK, 1), jnp.float32),
            pltpu.VMEM((BLOCK, WIDTH), jnp.float32),
        ),
    )
    run = pl.pallas_call(
        body,
        grid_spec=spec,
        out_shape=jax.ShapeDtypeStruct((padded, WIDTH), jnp.float32),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
        interpret=interpret,
    )
    return run(pad(q), pad(k), pad(v))[:length]


def reference(q, k, v):
    q, k, v = (np.asarray(a, dtype=np.float32) for a in (q, k, v))
    pos = np.arange(len(q))
    logits = (q @ k.T) / np.sqrt(WIDTH)
    logits = np.where(pos[None, :] <= pos[:, None], logits, -np.inf)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    return (probs / probs.sum(axis=1, keepdims=True)) @ v


if __name__ == "__main__":
    cpu = jax.default_backend() == "cpu"
    rng = np.random.default_rng(7)
    for n in (1, 127, 128, 129, 193, 257):
        q, k, v = [jnp.asarray(rng.normal(size=(n, WIDTH)), jnp.bfloat16) for _ in range(3)]
        out = tiled_attention(q, k, v, interpret=cpu)
        np.testing.assert_allclose(out, reference(q, k, v), rtol=0.012, atol=0.012)
    q = jnp.zeros((129, WIDTH), jnp.bfloat16)
    v = jnp.broadcast_to(jnp.asarray(np.arange(129), jnp.bfloat16)[:, None], q.shape)
    out = tiled_attention(q, q, v, interpret=cpu)
    np.testing.assert_allclose(np.asarray(out)[[0, 127, 128], 0], [0.0, 63.5, 64.0])
    q = q.at[128, 0].set(8.0)
    k = jnp.zeros_like(q).at[128, 0].set(2.0)
    v = jnp.full(q.shape, 2.0, jnp.bfloat16).at[128].set(8.0)
    out = tiled_attention(q, k, v, interpret=cpu)
    multiplier = np.exp(16.0 / np.sqrt(WIDTH))
    expected = (128 * 2.0 + multiplier * 8.0) / (128 + multiplier)
    np.testing.assert_allclose(np.asarray(out)[128], expected, rtol=1e-5)
    print(
        jax.default_backend(), "causal boundaries, padded tails and random reference passed"
    )
