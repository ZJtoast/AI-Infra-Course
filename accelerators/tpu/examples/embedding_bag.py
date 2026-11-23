"""按 CSR 边表读取特征并求和；JAX 0.12.1，目标为 TPU Mosaic。"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def csr_schedule(ptr, ids, table_rows):
    ptr, ids = np.asarray(ptr), np.asarray(ids)
    assert (
        ptr.ndim == ids.ndim == 1 and len(ptr) >= 2 and ptr[0] == 0 and ptr[-1] == len(ids)
    )
    assert (
        np.all(ptr[1:] >= ptr[:-1])
        and table_rows > 0
        and np.all((0 <= ids) & (ids < table_rows))
    )
    slots = []
    for row, (start, stop) in enumerate(zip(ptr[:-1], ptr[1:])):
        if start == stop:
            slots.append((row, 0, 1, 1, 0))  # 空行也要产生一个全零输出
        for edge in range(start, stop):
            slots.append((row, ids[edge], edge == start, edge + 1 == stop, 1))
    return np.asarray(slots, np.int32).T.copy()  # 每列是一次迭代的五个整数


def csr_bag(table, schedule, rows, *, interpret=False):
    assert table.ndim == 2 and table.shape[1] == 128 and schedule.shape[0] == 5

    def body(meta, feature, out, acc):
        e = pl.program_id(0)

        @pl.when(meta[2, e] != 0)
        def start_row():
            acc[...] = jnp.zeros((1, 128), jnp.float32)

        @pl.when(meta[4, e] != 0)
        def add_edge():
            acc[...] = acc[...] + feature[...]  # 重复编号按出现次数计入

        @pl.when(meta[3, e] != 0)
        def finish_row():
            out[...] = acc[...]

    src = pl.BlockSpec((None, 1, 128), lambda e, m: (m[1, e], 0, 0))
    dst = pl.BlockSpec((None, 1, 128), lambda e, m: (m[0, e], 0, 0))
    spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        grid=(schedule.shape[1],),
        in_specs=(src,),
        out_specs=dst,
        scratch_shapes=(pltpu.VMEM((1, 128), jnp.float32),),
    )
    run = pl.pallas_call(
        body,
        grid_spec=spec,
        out_shape=jax.ShapeDtypeStruct((rows, 1, 128), jnp.float32),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        interpret=interpret,
    )
    return run(jnp.asarray(schedule, jnp.int32), table[:, None, :])[:, 0, :]


def transpose_schedule(ptr, ids, source_rows):
    """将同一个源点的每次出现编排到相邻迭代，返回 CSR 聚合核可读的反向计划。"""
    incoming = [[] for _ in range(source_rows)]
    for target, (begin, end) in enumerate(zip(ptr[:-1], ptr[1:])):
        for edge in range(begin, end):
            incoming[ids[edge]].append(target)
    rev_ptr = np.cumsum([0] + [len(v) for v in incoming])
    rev_ids = np.asarray([t for group in incoming for t in group], dtype=np.int32)
    return csr_schedule(rev_ptr, rev_ids, len(ptr) - 1)


if __name__ == "__main__":
    cpu = jax.default_backend() == "cpu"
    table = np.arange(5 * 128, dtype=np.float32).reshape(5, 128) / 8
    table[1, :2], table[2, :2] = [1.0, 0.0], [2.0, 3.0]
    ptr, ids = [0, 3, 3, 5], [2, 1, 2, 4, 0]
    plan = csr_schedule(ptr, ids, len(table))
    out = csr_bag(jnp.asarray(table), plan, len(ptr) - 1, interpret=cpu)
    gold = np.stack(
        [
            table[ids[a:b]].sum(axis=0) if b > a else np.zeros(128)
            for a, b in zip(ptr[:-1], ptr[1:])
        ]
    )
    np.testing.assert_array_equal(out, gold)
    np.testing.assert_array_equal(np.asarray(out)[:2, :2], [[5.0, 6.0], [0.0, 0.0]])
    # 各目标行收到的梯度先取可手算整数，再检查整张反向表。
    grad = np.broadcast_to(np.array([[1.0], [7.0], [3.0]], np.float32), (3, 128)).copy()
    reverse = transpose_schedule(ptr, ids, len(table))
    dx = csr_bag(jnp.asarray(grad), reverse, len(table), interpret=cpu)
    scatter_gold = np.zeros_like(table)
    for target, (start, stop) in enumerate(zip(ptr[:-1], ptr[1:])):
        np.add.at(scatter_gold, ids[start:stop], grad[target])
    np.testing.assert_array_equal(dx, scatter_gold)
    np.testing.assert_array_equal(np.asarray(dx)[:, 0], [3.0, 1.0, 2.0, 0.0, 3.0])
    print(
        jax.default_backend(),
        "CSR duplicate IDs, empty row and all 128 feature channels passed",
    )
