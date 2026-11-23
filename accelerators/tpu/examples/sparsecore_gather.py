"""v6e SparseCore 的索引 DMA；CPU 模式核对十六个窗口的串行数值并检查 TPU 目标 lowering。"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh


def gather(table, ids, *, interpret=False):
    rows, width = table.shape
    window = 128
    assert width == 128 and ids.size == 16 * window and rows > 0
    # 设备把索引块分给同一 SparseCore 的十六个向量子核；解释器串行枚举相同块。
    mesh = plsc.VectorSubcoreMesh(
        core_axis_name="core",
        subcore_axis_name="subcore",
        num_cores=1,
        num_subcores=1 if interpret else 16,
    )

    @pl.kernel(
        mesh=mesh,
        out_type=jax.ShapeDtypeStruct((ids.size, width), table.dtype),
        interpret=interpret,
    )
    def read_by_id(table_hbm, ids_hbm, out_hbm):
        def load_rows(id_window, row_window):
            rows_to_read = id_window[0, :]  # 这个窗口内的 128 个源行编号
            pltpu.sync_copy(table_hbm.at[rows_to_read], row_window)

        pltpu.emit_pipeline(
            load_rows,
            grid=(ids.size // window,),
            in_specs=[pl.BlockSpec((1, window), lambda block: (0, block))],
            out_specs=[pl.BlockSpec((window, width), lambda block: (block, 0))],
            core_axis_name=None if interpret else "subcore",
            dimension_semantics=(pltpu.PARALLEL,),
        )(ids_hbm, out_hbm)

    return read_by_id(table, ids.reshape(1, -1))


if __name__ == "__main__":
    table = np.arange(32 * 128, dtype=np.float32).reshape(32, 128)
    ids = np.resize(np.array([7, 1, 7, 0], np.int32), 16 * 128)
    if jax.default_backend() == "cpu":
        with use_abstract_mesh(
            AbstractMesh((), (), abstract_device=AbstractDevice("TPU v6e", 1, "tpu"))
        ):
            exe = jax.jit(gather).trace(
                jax.ShapeDtypeStruct(table.shape, jnp.float32),
                jax.ShapeDtypeStruct(ids.shape, jnp.int32),
            )
            lowered = exe.lower(lowering_platforms=("tpu",)).as_text()
            assert "tpu_custom_call" in lowered
            print("TPU v6e SparseCore lowering passed")
            out = gather(jnp.asarray(table), jnp.asarray(ids), interpret=True)
    else:
        out = jax.jit(gather)(jnp.asarray(table), jnp.asarray(ids))
    np.testing.assert_array_equal(out, table[ids])
    print("first channel:", np.asarray(out[:4, 0]))
