"""下载本文件和同目录的 pipelined_gemm.py。运行 python profile_gemm.py。"""

from pathlib import Path
from statistics import median
from time import perf_counter
import argparse

import jax
import jax.numpy as jnp
import numpy as np
from pipelined_gemm import project


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--trace", type=Path, help="XProf 的本地轨迹目录")
    args = cli.parse_args()
    device = jax.devices()[0]
    cpu = device.platform == "cpu"
    rng = np.random.default_rng(19)
    a = jnp.asarray(rng.normal(size=(128, 256)), dtype=jnp.bfloat16)
    b = jnp.asarray(rng.normal(size=(256, 256)), dtype=jnp.bfloat16)
    bias = jnp.asarray(rng.normal(size=256), dtype=jnp.float32)
    reference = np.maximum(
        np.asarray(a, np.float32) @ np.asarray(b, np.float32) + np.asarray(bias), 0.0
    )
    cases = {}
    for name, fused, slots in (
        ("partial", False, 1),
        ("single", True, 1),
        ("double", True, 2),
    ):

        def operation(a, b, bias, fused=fused, slots=slots):
            return project(a, b, bias, fused=fused, buffers=slots, interpret=cpu)

        exe = jax.jit(operation).lower(a, b, bias).compile()
        y = exe(a, b, bias).block_until_ready()
        np.testing.assert_allclose(y, reference, rtol=1e-4, atol=1e-4)
        cases[name] = exe

    print(device, "three numerical references passed")
    if cpu:
        return
    if device.platform != "tpu":
        raise RuntimeError("本实验的设备计时需要 TPU。")
    samples = {name: [] for name in cases}
    names = list(cases)
    for rep in range(9):
        for name in names[rep % 3 :] + names[: rep % 3]:
            begin = perf_counter()
            cases[name](a, b, bias).block_until_ready()
            samples[name].append(1e6 * (perf_counter() - begin))
    for name, values in samples.items():
        print(name, "synchronous median (us)", round(median(values), 2))
    if args.trace:
        with jax.profiler.trace(str(args.trace)):
            for name, exe in cases.items():
                with jax.profiler.TraceAnnotation(name):
                    exe(a, b, bias).block_until_ready()


if __name__ == "__main__":
    main()
