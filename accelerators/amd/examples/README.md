# AMD Instinct 算子实验

## CPU 数值对照

```bash
python reference.py
python advanced_reference.py
```

两个脚本仅依赖 Python 标准库。进阶脚本还把每个 MFMA lane 的输入与四个输出寄存器还原到整张矩阵，用普通矩阵乘逐格核验。

## HIP 设备对照

在安装 ROCm 的 Linux 主机解压整个目录。下面以 MI300X 的 gfx942 为例。使用 MI200 系列的 gfx90a 时，将三条命令中的目标都改为 gfx90a。

```bash
hipcc -std=c++17 -O2 --offload-arch=gfx942 run_operators.hip -o run_operators
hipcc -std=c++17 -O2 --offload-arch=gfx942 advanced.hip -o advanced
hipcc -std=c++17 -O2 --offload-arch=gfx942 benchmark.hip -o benchmark
./run_operators
./advanced
```

基础驱动对照转置、矩阵乘、规约、扫描、稀疏乘法、归一化和分页等小算例。进阶驱动在相同输入上比较直接 MFMA 与 LDS MFMA、按值列与按查询行块的 SDPA、两遍与融合量化、稳定路由、三种扫描、三种转置，以及共享邻居索引、残差融合和分页读回。出现不同结果时，程序打印第一个出错的下标、结果与期望值。

## 同输入的计时

```bash
./benchmark gemm
./benchmark attention
./benchmark quant
./benchmark route
./benchmark scan
./benchmark transpose
```

每个命令只测对应家族，使用同一流上的 HIP 事件，对三次预热后的二十轮执行取平均。矩阵乘等以毫秒显示，转置额外报告读写一遍有效矩阵所达到的十进制 GB/s。计数、前缀和散写是路由的三个组成步骤，route 将三次启动一起计时。scan 单独计时可替换的块内阶段。

比较时保留设备型号、ROCm 版本、形状、步长、数据类型、编译目标与输入。修改输入或块形状后，先运行对应的数值对照，再计时。
