# AI Infra Course

**AI 基础设施课程**　·　19 课　·　190 道练习

　　课程涵盖 GPU 编程、分布式训练、大模型推理和性能分析。分课讲义适合按顺序学习，合订本可直接查阅 427 个知识点。每课的普通题、进阶题和情景面试题都放在对应小节，可先独立作答，再展开推导。

[在线阅读](https://zjtoast.github.io/AI-Infra-Course/)　·　[学习路线](https://zjtoast.github.io/AI-Infra-Course/roadmap.html)　·　[知识点合订本](https://zjtoast.github.io/AI-Infra-Course/handbook.html)　·　[下载 Word 合订本](word/ai-infra-course-handbook.docx)

## 课程目录

### 计算与 GPU 编程

| 课次 | 讲义 | 主要内容 | 下载 |
| :---: | :--- | :--- | :---: |
| 01 | [Transformer 单层计算量与显存分析](https://zjtoast.github.io/AI-Infra-Course/lessons/01-transformer-compute-and-memory.html) | QKV 投影、FLOPs、KV cache | [Word](word/01-transformer-compute-and-memory.docx) |
| 02 | [GPU 微架构与 Tensor Core](https://zjtoast.github.io/AI-Infra-Course/lessons/02-gpu-microarchitecture-and-tensor-cores.html) | SM、存储层次、Tensor Core | [Word](word/02-gpu-microarchitecture-and-tensor-cores.docx) |
| 03 | [CUDA、PTX 与分块算子编程](https://zjtoast.github.io/AI-Infra-Course/lessons/03-cuda-ptx-and-tiled-kernels.html) | CUDA、PTX、Triton、TileLang | [Word](word/03-cuda-ptx-and-tiled-kernels.docx) |
| 04 | [Attention 算子与编译执行](https://zjtoast.github.io/AI-Infra-Course/lessons/04-attention-kernels-and-compilation.html) | FlashAttention、图编译与算子融合 | [Word](word/04-attention-kernels-and-compilation.docx) |

### 通信与分布式训练

| 课次 | 讲义 | 主要内容 | 下载 |
| :---: | :--- | :--- | :---: |
| 05 | [集合通信、NCCL 与 RDMA](https://zjtoast.github.io/AI-Infra-Course/lessons/05-collective-communication-and-rdma.html) | Ring AllReduce、NCCL、GPUDirect RDMA | [Word](word/05-collective-communication-and-rdma.docx) |
| 06 | [训练显存、激活重计算与卸载](https://zjtoast.github.io/AI-Infra-Course/lessons/06-training-memory-and-recomputation.html) | Adam 状态、激活重算、CPU 卸载 | [Word](word/06-training-memory-and-recomputation.docx) |
| 07 | [数据并行、状态分片与训练恢复](https://zjtoast.github.io/AI-Infra-Course/lessons/07-data-parallelism-and-state-sharding.html) | DDP、ZeRO、FSDP、分布式检查点 | [Word](word/07-data-parallelism-and-state-sharding.docx) |
| 08 | [多维并行与张量布局](https://zjtoast.github.io/AI-Infra-Course/lessons/08-multidimensional-parallelism.html) | TP、SP、CP、PP 与 DeviceMesh | [Word](word/08-multidimensional-parallelism.docx) |
| 09 | [MoE 路由与专家并行](https://zjtoast.github.io/AI-Infra-Course/lessons/09-mixture-of-experts-and-expert-parallelism.html) | Top-k 路由、AllToAll、Grouped GEMM | [Word](word/09-mixture-of-experts-and-expert-parallelism.docx) |

### 模型架构与数值方法

| 课次 | 讲义 | 主要内容 | 下载 |
| :---: | :--- | :--- | :---: |
| 10 | [量化表示、低精度执行与 Muon](https://zjtoast.github.io/AI-Infra-Course/lessons/10-quantization-and-optimizer-design.html) | INT4、FP8、FP4、PTQ、QAT、Muon | [Word](word/10-quantization-and-optimizer-design.docx) |
| 11 | [长上下文模型架构与状态复用](https://zjtoast.github.io/AI-Infra-Course/lessons/11-long-context-attention-and-state-reuse.html) | MLA、稀疏注意力、DeepSeek-V4.1-Flash | [Word](word/11-long-context-attention-and-state-reuse.docx) |

### 推理系统

| 课次 | 讲义 | 主要内容 | 下载 |
| :---: | :--- | :--- | :---: |
| 12 | [推理服务指标与 KV 分页](https://zjtoast.github.io/AI-Infra-Course/lessons/12-llm-serving-metrics-and-kv-paging.html) | TTFT、TPOT、分页与连续批处理 | [Word](word/12-llm-serving-metrics-and-kv-paging.docx) |
| 13 | [推理引擎执行与 CUDA Graph](https://zjtoast.github.io/AI-Infra-Course/lessons/13-inference-engine-execution.html) | 请求槽位、CUDA Graph、前缀树 | [Word](word/13-inference-engine-execution.docx) |
| 14 | [分离式推理与 KV 传输](https://zjtoast.github.io/AI-Infra-Course/lessons/14-disaggregated-inference-and-kv-transfer.html) | Prefill/decode 分离、KV 重排与传输 | [Word](word/14-disaggregated-inference-and-kv-transfer.docx) |
| 15 | [推测解码与前缀验证](https://zjtoast.github.io/AI-Infra-Course/lessons/15-speculative-decoding.html) | 拒绝采样、草稿长度、变长验证 | [Word](word/15-speculative-decoding.docx) |

### 训练平台与性能工程

| 课次 | 讲义 | 主要内容 | 下载 |
| :---: | :--- | :--- | :---: |
| 16 | [智能体强化学习训练系统](https://zjtoast.github.io/AI-Infra-Course/lessons/16-agentic-rl-training-systems.html) | 轨迹、策略版本、异步 rollout | [Word](word/16-agentic-rl-training-systems.docx) |
| 17 | [训练数据管线、检查点与调度](https://zjtoast.github.io/AI-Infra-Course/lessons/17-training-data-checkpoints-and-scheduling.html) | 有效 token、恢复点、作业放置 | [Word](word/17-training-data-checkpoints-and-scheduling.docx) |
| 18 | [GPU 性能剖析与瓶颈诊断](https://zjtoast.github.io/AI-Infra-Course/lessons/18-gpu-profiling-and-performance-analysis.html) | Nsight Systems、Nsight Compute、SOL、Roofline | [Word](word/18-gpu-profiling-and-performance-analysis.docx) |
| 19 | [长上下文服务的容量与系统设计](https://zjtoast.github.io/AI-Infra-Course/lessons/19-long-context-system-design.html) | 显存预算、跨机流量、服务容量 | [Word](word/19-long-context-system-design.docx) |

## 离线阅读

　　下载仓库后，打开根目录的 [index.html](index.html)。Word 讲义位于 [word](word/) 目录。网页版练习可以直接作答，点「答案」即可查看推导。
