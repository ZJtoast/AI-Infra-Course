# AI Infra Course

**AI 基础设施 · 中文课程讲义**

　　一次生成请求经过投影、Attention、集合通信和调度，最终表现为用户可见的首 token 时延。课程从这些可以画出形状、算出字节、量出时间的对象入手，逐步进入分布式训练、长上下文推理与 GPU 性能剖析。

**19 课**　·　**427 个知识点**　·　**143 道随堂练习与推导**

[**在线阅读 →**](https://zjtoast.github.io/AI-Infra-Course/)　 ·　 [学习路线](https://zjtoast.github.io/AI-Infra-Course/roadmap.html)　 ·　 [知识点导航](https://zjtoast.github.io/AI-Infra-Course/handbook.html)　 ·　 [下载 Word 合订本](word/ai-infra-course-handbook.docx)

　　在每道随堂练习的「我的作答」中写下思路，再点「查看答案与推导」逐步核对。在线版会将草稿保存在当前浏览器，同一道题可以在分课页与合订本之间接着写。

## 按问题开始

| 实际问题 | 阅读入口 |
| :--- | :--- |
| 一层 Transformer 要做多少次乘加，KV cache 又占多少显存？ | [GEMM 计算量](https://zjtoast.github.io/AI-Infra-Course/lessons/01-transformer-compute-and-memory.html#l01-s015) · [KV 分页](https://zjtoast.github.io/AI-Infra-Course/lessons/12-llm-serving-metrics-and-kv-paging.html#l12-s007) |
| 一个 GPU 算子被哪项资源限制？ | [Tensor Core](https://zjtoast.github.io/AI-Infra-Course/lessons/02-gpu-microarchitecture-and-tensor-cores.html#l02-s016) · [手写 PTX](https://zjtoast.github.io/AI-Infra-Course/lessons/03-cuda-ptx-and-tiled-kernels.html#l03-s004) · [SOL 利用率](https://zjtoast.github.io/AI-Infra-Course/lessons/18-gpu-profiling-and-performance-analysis.html#l18-s017) · [Roofline](https://zjtoast.github.io/AI-Infra-Course/lessons/18-gpu-profiling-and-performance-analysis.html#l18-s020) |
| 跨卡与跨机的流量怎样核算？ | [NCCL](https://zjtoast.github.io/AI-Infra-Course/lessons/05-collective-communication-and-rdma.html#l05-s006) · [RDMA](https://zjtoast.github.io/AI-Infra-Course/lessons/05-collective-communication-and-rdma.html#l05-s007) · [流水线并行](https://zjtoast.github.io/AI-Infra-Course/lessons/08-multidimensional-parallelism.html#l08-s010) |
| 量化、稀疏注意力和图重放各自改变什么？ | [量化方法](https://zjtoast.github.io/AI-Infra-Course/lessons/10-quantization-and-optimizer-design.html#l10-s014) · [DeepSeek-V4.1-Flash](https://zjtoast.github.io/AI-Infra-Course/lessons/11-long-context-attention-and-state-reuse.html#l11-s008) · [CUDA Graph](https://zjtoast.github.io/AI-Infra-Course/lessons/13-inference-engine-execution.html#l13-s008) |
| 首 token 变慢，先看系统时间线还是算子计数器？ | [Nsight Systems](https://zjtoast.github.io/AI-Infra-Course/lessons/18-gpu-profiling-and-performance-analysis.html#l18-s007) · [Nsight Compute](https://zjtoast.github.io/AI-Infra-Course/lessons/18-gpu-profiling-and-performance-analysis.html#l18-s013) |

## 课程目录

### 计算与 GPU 编程

| 课次 | 主题 | 本课解决的问题 | 讲义 |
| :---: | :--- | :--- | :---: |
| 01 | [Transformer 单层计算量与显存分析](https://zjtoast.github.io/AI-Infra-Course/lessons/01-transformer-compute-and-memory.html) | 从一个 token 的投影追到整层 FLOPs、参数与 KV cache，分别核算 prefill 和 decode。 | [Word](word/01-transformer-compute-and-memory.docx) |
| 02 | [GPU 微架构与 Tensor Core](https://zjtoast.github.io/AI-Infra-Course/lessons/02-gpu-microarchitecture-and-tensor-cores.html) | 沿线程、warp、SM 与存储层次追踪一次访存，比较各代矩阵指令的参与线程与数据布局。 | [Word](word/02-gpu-microarchitecture-and-tensor-cores.docx) |
| 03 | [CUDA、PTX 与分块算子编程](https://zjtoast.github.io/AI-Infra-Course/lessons/03-cuda-ptx-and-tiled-kernels.html) | 从手写 PTX 和反汇编读出寻址与寄存器分配，再用矩阵分块解释复用和异步流水。 | [Word](word/03-cuda-ptx-and-tiled-kernels.docx) |
| 04 | [Attention 算子与编译执行](https://zjtoast.github.io/AI-Infra-Course/lessons/04-attention-kernels-and-compilation.html) | 逐块推导在线 softmax，追踪前向状态、反向重计算和编译边界。 | [Word](word/04-attention-kernels-and-compilation.docx) |

### 通信与分布式训练

| 课次 | 主题 | 本课解决的问题 | 讲义 |
| :---: | :--- | :--- | :---: |
| 05 | [集合通信、NCCL 与 RDMA](https://zjtoast.github.io/AI-Infra-Course/lessons/05-collective-communication-and-rdma.html) | 从四个 rank 的张量归属推导通信量，沿 NCCL、网卡队列和 GPUDirect RDMA 检查传输完成条件。 | [Word](word/05-collective-communication-and-rdma.docx) |
| 06 | [训练显存、激活重计算与卸载](https://zjtoast.github.io/AI-Infra-Course/lessons/06-training-memory-and-recomputation.html) | 画出一次训练更新中参数、梯度、优化器状态和激活的存活区间，找出峰值。 | [Word](word/06-training-memory-and-recomputation.docx) |
| 07 | [数据并行、状态分片与训练恢复](https://zjtoast.github.io/AI-Infra-Course/lessons/07-data-parallelism-and-state-sharding.html) | 从 DDP 梯度归约追到 FSDP 参数收集，再检查分布式检查点能恢复哪些状态。 | [Word](word/07-data-parallelism-and-state-sharding.docx) |
| 08 | [多维并行与张量布局](https://zjtoast.github.io/AI-Infra-Course/lessons/08-multidimensional-parallelism.html) | 在 TP、SP、CP 和流水线并行之间跟踪一个 microbatch，核对每条边界上的形状和字节数。 | [Word](word/08-multidimensional-parallelism.docx) |
| 09 | [MoE 路由与专家并行](https://zjtoast.github.io/AI-Infra-Course/lessons/09-mixture-of-experts-and-expert-parallelism.html) | 沿 token 分派、专家分组计算与加权汇合追踪数据，区分负载偏斜和网络等待。 | [Word](word/09-mixture-of-experts-and-expert-parallelism.docx) |

### 模型架构与数值方法

| 课次 | 主题 | 本课解决的问题 | 讲义 |
| :---: | :--- | :--- | :---: |
| 10 | [量化表示、低精度执行与 Muon](https://zjtoast.github.io/AI-Infra-Course/lessons/10-quantization-and-optimizer-design.html) | 核算 scale、权重和临时缓冲，比较训练与推理的精度边界，推导 Muon 的矩阵更新。 | [Word](word/10-quantization-and-optimizer-design.docx) |
| 11 | [长上下文模型架构与状态复用](https://zjtoast.github.io/AI-Infra-Course/lessons/11-long-context-attention-and-state-reuse.html) | 比较 MLA、压缩稀疏注意力与递归状态，并沿 DeepSeek-V4.1-Flash 的实际配置追踪模型结构。 | [Word](word/11-long-context-attention-and-state-reuse.docx) |

### 推理系统

| 课次 | 主题 | 本课解决的问题 | 讲义 |
| :---: | :--- | :--- | :---: |
| 12 | [推理服务指标与 KV 分页](https://zjtoast.github.io/AI-Infra-Course/lessons/12-llm-serving-metrics-and-kv-paging.html) | 在同一请求中标出排队、首 token 和后续生成的时间边界，再讨论页分配和批调度。 | [Word](word/12-llm-serving-metrics-and-kv-paging.docx) |
| 13 | [推理引擎执行与 CUDA Graph](https://zjtoast.github.io/AI-Infra-Course/lessons/13-inference-engine-execution.html) | 追踪请求槽位、异步缓冲和采样，并检查 CUDA Graph 重放依赖的地址与生命周期。 | [Word](word/13-inference-engine-execution.docx) |
| 14 | [分离式推理与 KV 传输](https://zjtoast.github.io/AI-Infra-Course/lessons/14-disaggregated-inference-and-kv-transfer.html) | 沿 prefill 到 decode 的交接核算有效字节、重排、传输完成和缓存可见性。 | [Word](word/14-disaggregated-inference-and-kv-transfer.docx) |
| 15 | [推测解码与前缀验证](https://zjtoast.github.io/AI-Infra-Course/lessons/15-speculative-decoding.html) | 从草稿生成与目标模型验证推导接受概率，比较候选长度、验证代价与端到端时延。 | [Word](word/15-speculative-decoding.docx) |

### 训练平台与性能工程

| 课次 | 主题 | 本课解决的问题 | 讲义 |
| :---: | :--- | :--- | :---: |
| 16 | [智能体强化学习训练系统](https://zjtoast.github.io/AI-Infra-Course/lessons/16-agentic-rl-training-systems.html) | 沿轨迹采样、奖励、策略更新和权重同步追踪一次训练迭代。 | [Word](word/16-agentic-rl-training-systems.docx) |
| 17 | [训练数据管线、检查点与调度](https://zjtoast.github.io/AI-Infra-Course/lessons/17-training-data-checkpoints-and-scheduling.html) | 把输入字节换算成有效训练 token，核对预取、持久化与恢复中的状态。 | [Word](word/17-training-data-checkpoints-and-scheduling.docx) |
| 18 | [GPU 性能剖析与瓶颈诊断](https://zjtoast.github.io/AI-Infra-Course/lessons/18-gpu-profiling-and-performance-analysis.html) | 先用 Nsight Systems 确定关键路径，再用 Nsight Compute 的 SOL、内存层次和 Roofline 检验算子瓶颈。 | [Word](word/18-gpu-profiling-and-performance-analysis.docx) |
| 19 | [长上下文服务的容量与系统设计](https://zjtoast.github.io/AI-Infra-Course/lessons/19-long-context-system-design.html) | 从请求分布和服务目标推导并发、显存与跨机流量，用可复算的条件比较部署方案。 | [Word](word/19-long-context-system-design.docx) |

## 离线阅读

　　下载仓库后，用浏览器打开根目录的 [index.html](index.html)。网页的目录、术语搜索、公式、插图与代码样式随仓库提供。阅读上游源码和官方文档时需要联网。

　　[Word 学习路线](word/00-learning-roadmap.docx) 适合规划进度。[Word 合订本](word/ai-infra-course-handbook.docx) 可从目录跳至各知识点，每节末尾的按钮返回对应目录项。桌面 Word 通常使用 Ctrl + 单击跟随链接。题目与完整推导放在所属小节。

## 源码与教学示例

　　代码块保留原文件行号与来源链接。阅读时可以先追踪输入形状和地址，再用讲义的数字验算。性能章节给出了采集条件，教学时序、理论上界与实际编译结果在相应小节标明。

| 示例 | 关注点 |
| :--- | :--- |
| [axpy.ptx](examples/axpy.ptx) | PTX 参数、线程索引与地址计算 |
| [warp_mma.ptx](examples/warp_mma.ptx) | warp 级矩阵乘加的指令与寄存器操作数 |
| [descriptor_class.ptx](examples/descriptor_class.ptx) | 描述符相关的 PTX 指令与目标架构条件 |
| [stride_lab.cu](examples/stride_lab.cu) | 用同一内核比较访存步长，按 NVTX 范围区分预热与稳态 |

　　上游仓库、固定版本和许可文本见[源码来源与许可](sources/README.md)。
