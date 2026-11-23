"""Top-2 专家线性层：按专家分块、在 TPU 计算、按原词编号还原。JAX 0.12.1。"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def route_plan(experts, num_experts):
    chosen = np.asarray(experts, np.int32)
    assert (
        chosen.ndim == 2
        and chosen.shape[1] == 2
        and np.all((chosen >= 0) & (chosen < num_experts))
    )
    source = []
    blocks = []
    inverse = np.empty_like(chosen)
    for e in range(num_experts):
        hits = np.argwhere(chosen == e)
        for begin in range(0, len(hits), 8):
            chunk = hits[begin : begin + 8]
            blocks.append(e)
            for token, branch in chunk:
                inverse[token, branch] = len(source)
                source.append(token)
            source.extend([-1] * (8 - len(chunk)))  # 补齐专家自己的最后一个块
    return np.asarray(source, np.int32), np.asarray(blocks, np.int32), inverse.T.copy()


def experts_forward(x, weights, source, blocks, inverse, gate, *, interpret=False):
    indices = jnp.asarray(source)
    packed = jnp.where(indices[:, None] >= 0, x[jnp.maximum(indices, 0)], 0).astype(
        jnp.bfloat16
    )

    def expert(meta, a, w, o):
        o[...] = lax.dot(a[...], w[...], preferred_element_type=jnp.float32)

    tile = pl.BlockSpec((8, 128), lambda i, m: (i, 0))
    wtile = pl.BlockSpec((None, 128, 128), lambda i, m: (m[i], 0, 0))
    spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1, grid=(len(blocks),), in_specs=(tile, wtile), out_specs=tile
    )
    z = pl.pallas_call(
        expert,
        grid_spec=spec,
        out_shape=jax.ShapeDtypeStruct(packed.shape, jnp.float32),
        interpret=interpret,
    )(jnp.asarray(blocks), packed, weights.astype(jnp.bfloat16))

    def merge(rev, g, a, b, o):
        t = pl.program_id(0)
        o[...] = g[0, t] * a[...] + g[1, t] * b[...]  # 两条分支按同一词的原门控相加

    s0 = pl.BlockSpec((None, 1, 128), lambda t, r, g: (r[0, t], 0, 0))
    s1 = pl.BlockSpec((None, 1, 128), lambda t, r, g: (r[1, t], 0, 0))
    dst = pl.BlockSpec((None, 1, 128), lambda t, r, g: (t, 0, 0))
    spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2, grid=(x.shape[0],), in_specs=(s0, s1), out_specs=dst
    )
    return pl.pallas_call(
        merge,
        grid_spec=spec,
        out_shape=jax.ShapeDtypeStruct((x.shape[0], 1, 128), jnp.float32),
        interpret=interpret,
    )(jnp.asarray(inverse), jnp.asarray(gate.T, jnp.float32), z[:, None, :], z[:, None, :])[
        :, 0, :
    ]


if __name__ == "__main__":
    cpu = jax.default_backend() == "cpu"
    e = np.array([[0, 1], [1, 2], [0, 2]], np.int32)
    gate = np.array([[0.25, 0.75], [0.5, 0.5], [0.75, 0.25]], np.float32)
    x = np.broadcast_to(np.array([2, 3, 4], np.float32)[:, None], (3, 128)).copy()
    w = np.stack([a * np.eye(128, dtype=np.float32) for a in [2, 3, 4, 5]])
    plan = route_plan(e, len(w))
    y = experts_forward(jnp.asarray(x), jnp.asarray(w), *plan, gate, interpret=cpu)
    gold = np.stack(
        [sum(gate[t, b] * (x[t] @ w[e[t, b]]) for b in range(2)) for t in range(len(x))]
    )
    np.testing.assert_array_equal(y, gold)
    print(
        jax.default_backend(), np.asarray(y)[:, 0], "empty expert and top-2 routing passed"
    )
