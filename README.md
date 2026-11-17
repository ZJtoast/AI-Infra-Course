# AI Infra Course

**AI 基础设施课程**　·　19 课　·　200 道练习

　　课程涵盖 GPU 编程、分布式训练、大模型推理和性能分析。分课讲义适合按顺序学习，合订本可直接查阅 437 个知识点。每课可从「本课练习顺序」依次做普通题、进阶题和情景面试题，也可在对应知识点下直接作答。

[在线阅读](https://zjtoast.github.io/AI-Infra-Course/)　·　[学习路线](https://zjtoast.github.io/AI-Infra-Course/roadmap.html)　·　[知识点合订本](https://zjtoast.github.io/AI-Infra-Course/handbook.html)　·　[下载 Word 合订本](word/ai-infra-course-handbook.docx)

## 课程目录

### 计算与 GPU 编程

| 课次 | 讲义 | 主要内容 | 下载 |
| :---: | :--- | :--- | :---: |
| 01 | [Transformer 单层计算量与显存分析](https://zjtoast.github.io/AI-Infra-Course/lessons/01-transformer-compute-and-memory.html) | QKV 投影、浮点运算量与 KV 缓存 | [Word](word/01-transformer-compute-and-memory.docx) |
| 02 | [GPU 微架构与 Tensor Core](https://zjtoast.github.io/AI-Infra-Course/lessons/02-gpu-microarchitecture-and-tensor-cores.html) | SM、存储层次、Tensor Core | [Word](word/02-gpu-microarchitecture-and-tensor-cores.docx) |
| 03 | [CUDA、PTX 与分块算子编程](https://zjtoast.github.io/AI-Infra-Course/lessons/03-cuda-ptx-and-tiled-kernels.html) | CUDA、PTX、Triton、TileLang | [Word](word/03-cuda-ptx-and-tiled-kernels.docx) |
| 04 | [Attention 算子与编译执行](https://zjtoast.github.io/AI-Infra-Course/lessons/04-attention-kernels-and-compilation.html) | FlashAttention 1–4、SageAttention 系列与算子融合 | [Word](word/04-attention-kernels-and-compilation.docx) |

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
| 14 | [分离式推理与 KV 传输](https://zjtoast.github.io/AI-Infra-Course/lessons/14-disaggregated-inference-and-kv-transfer.html) | 预填充与解码分离、KV 重排与传输 | [Word](word/14-disaggregated-inference-and-kv-transfer.docx) |
| 15 | [推测解码与前缀验证](https://zjtoast.github.io/AI-Infra-Course/lessons/15-speculative-decoding.html) | 拒绝采样、草稿长度、变长验证 | [Word](word/15-speculative-decoding.docx) |

### 训练平台与性能工程

| 课次 | 讲义 | 主要内容 | 下载 |
| :---: | :--- | :--- | :---: |
| 16 | [智能体强化学习训练系统](https://zjtoast.github.io/AI-Infra-Course/lessons/16-agentic-rl-training-systems.html) | 轨迹、策略版本与异步采样 | [Word](word/16-agentic-rl-training-systems.docx) |
| 17 | [训练数据管线、检查点与调度](https://zjtoast.github.io/AI-Infra-Course/lessons/17-training-data-checkpoints-and-scheduling.html) | 有效 token、恢复点、作业放置 | [Word](word/17-training-data-checkpoints-and-scheduling.docx) |
| 18 | [GPU 性能剖析与瓶颈诊断](https://zjtoast.github.io/AI-Infra-Course/lessons/18-gpu-profiling-and-performance-analysis.html) | Nsight Systems、Nsight Compute、SOL、Roofline | [Word](word/18-gpu-profiling-and-performance-analysis.docx) |
| 19 | [长上下文服务的容量与系统设计](https://zjtoast.github.io/AI-Infra-Course/lessons/19-long-context-system-design.html) | 显存预算、跨机流量、服务容量 | [Word](word/19-long-context-system-design.docx) |

## 离线阅读

　　下载仓库后，打开根目录的 [index.html](index.html)。Word 讲义位于 [word](word/) 目录。网页版练习可以直接作答，点「答案」可展开推导。正文中带点状下划线的术语可以点开，就地查看释义。

## 更新讲义

　　课程正文保存在 [course.xml](authoring/course.xml)。按课名或小节标题搜索即可定位。改完母稿，在仓库根目录执行下面两条命令。两条命令可以独立运行，顺序任意。

```powershell
python authoring/build_html.py
python authoring/build_word.py
```

　　网页脚本生成首页、搜索索引、学习路线、分课页面和合订本。Word 脚本生成 `word/` 下的学习路线、分课讲义和合订本。直接编辑生成文件的改动会在下次构建时被覆盖。要先预览，可在任一命令后加 `--output D:/course-preview`。

| 修改内容 | 母稿中的位置 | 生成结果 |
| --- | --- | --- |
| 小节标题和正文 | 在对应 `section class="topic"` 中修改 `h2`、`h3` 或 `p`，保留已有的 `id` | 单课和合订本同步更新，目录链接继续有效 |
| 公式与上下标 | 公式用带有 MathML 命名空间的 `math`；行文里的简单上下标用 `sub` 和 `sup` | 网页排印数学符号，Word 中的公式可继续编辑 |
| 练习与解析 | 每道 `exercise` 内依次写情境、带 `prompt-` 标识的问题、`answer` 内的推导 | 网页保留笔记输入框和折叠答案；Word 就地排列题解，题干和答案可互跳 |
| 术语释义 | 在本课对应的 `dt` 后修改 `dd` | 网页词条、Word 术语提示与释义同步更新 |
| 插图与演算 | 普通插图引用 `assets/figures/` 下的共享图片；FlashAttention 演算的三个输入矩阵写在 `data-q`、`data-k`、`data-v` 中 | 同一插图进入两个版本；矩阵改值后，网页可重新演算，Word 自动绘制因果掩码下的示例块 |

　　构建环境为 Python 3.10 或更高版本，首次运行可用 `python -m pip install lxml Pillow` 安装依赖。Word 公式转换使用桌面版 Microsoft Office 附带的 `MML2OMML.XSL`，自定义安装位置可通过环境变量 `MML2OMML_XSL` 指定。矩阵插图默认使用 Windows 的微软雅黑与 Times New Roman，其他系统可用 `COURSE_CJK_FONT` 和 `COURSE_LATIN_FONT` 指向相应字体文件。
