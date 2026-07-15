# Mark2 Alibaba: Production-Trace-Driven AI-RAN Offloading Simulation

本项目使用阿里巴巴公开的 GenAI Serving Top-Down Dataset 2026（GenTD26）驱动一个 AI-RAN 二元卸载势博弈，验证异步最佳响应在真实云端请求异质性和拥塞统计校准下的收敛性、均衡效率与容量违规行为。

> **结论边界**：本项目验证的是“真实云端生产轨迹驱动的软件仿真”。它不是完整的真实 AI-RAN 现场实验，不包含真实无线链路、边缘设备功耗或真实 LLM 调用。最佳响应收敛到纯策略纳什均衡（PSNE）也不等于达到系统总代价全局最优。

## 核心结果

| 指标 | 结果 |
|---|---:|
| 抽取请求数 | 100 |
| 异步更新次数 | 836 |
| 实际策略改变次数 | 80 |
| PSNE 卸载请求数 | 74/100 |
| 假设容量 | 80 个请求等价槽位 |
| PSNE 容量违规 | 否 |
| PSNE 系统总代价 | 208.1783 |
| 诊断全局最优卸载数 | 57/100 |
| 诊断全局最优总代价 | 178.5765 |
| PSNE 是否为全局最优 | 否 |



## 实验模型

每个玩家 `i` 选择：

```text
s_i ∈ {0, 1}
```

其中 `s_i = 0` 表示本地执行，`s_i = 1` 表示卸载。卸载请求数为：

```text
K(s) = Σ_i s_i
```

Mark2 保留“本地基础代价 vs. 卸载基础代价 + 共享拥塞”的势博弈结构。节点执行严格最佳响应，算法停止前逐节点检查是否存在可降低自身代价的单边偏离。

### 重要实现修正

原始伪代码直接使用 `K + 1` 估计候选卸载状态。当被选节点当前已经卸载时，这会重复计算该节点。本实现先计算：

```text
K_{-i} = K - s_i
```

再使用 `K_{-i} + 1` 计算候选卸载代价。

此外，“连续 `N` 次随机抽取没有改变”可能重复抽中同一节点，因此本实现停止前还会逐节点执行 PSNE 检查。

## 数据来源

数据来自 [Alibaba Cluster Trace Program](https://github.com/alibaba/clusterdata) 中的 [GenAI Serving Top-Down Dataset 2026](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2026-GenAI)。原始数据由发布方匿名化，并声明源自真实生产环境。

本仓库仅包含本实验实际读取的五张表：

| 文件 | 用途 |
|---|---|
| `lora_request_trace.csv` | 请求时间、执行时间、Prompt 长度、推理步数和 LoRA 数量 |
| `queue_rt_raw_anon.csv` | 校准云端队列时延中位数和 P95 |
| `model_predict_data_anon.csv` | 校准云端纯推理时延中位数 |
| `pod_gpu_memory_used_bytes_anon.csv` | 提供 GPU 显存使用分位数背景 |
| `pod_gpu_duty_cycle_anon.csv` | 提供 GPU 利用率分位数背景 |

请在使用或再分发数据前阅读 [`官方_README.md`](00_原始数据/阿里_GenAI服务_2026/官方_README.md) 以及上游仓库的使用与引用要求。本项目不对上游数据重新授权。

## 真实数据与假设的边界

直接来自轨迹的字段包括：

- 请求创建时间与任务类型；
- 实际执行时间；
- Prompt 长度、推理步数和 LoRA 数量；
- 队列时延、推理时延、GPU 利用率及显存使用统计。

以下参数仍是仿真假设：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `local_slowdown` | 1.8 | 端侧执行相对云端执行时间的假设倍率 |
| `local_energy_addition` | 0.4 | 本地执行的归一化能耗附加项 |
| `tx_base_cost` | 0.15 | 无线传输基础归一化代价 |
| `prompt_tx_weight` | 0.15 | Prompt 长度对传输代价的映射权重 |
| `capacity_fraction` | 0.80 | 中心池允许 80 个请求等价槽位 |
| `memory_alpha` | 1.2 | 容量障碍函数缩放 |
| `memory_beta` | 8.0 | 容量障碍函数陡峭程度 |

因此，系统总代价是归一化指标，不能直接解释为焦耳、人民币或完整端到端实际时延。

## 安装与复现

建议使用 Python 3.10 或更高版本。

```bash
git clone <your-repository-url>
cd <repository-directory>

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

运行仿真：

```bash
python "01_源码与说明/仿真代码_mark2_alibaba.py"
```

重新生成投稿级结果图：

```bash
python "01_源码与说明/投稿级绘图_nature_figure_export.py"
```

仿真使用固定随机种子 `20260713`。默认场景从轨迹峰值小时中抽取 100 个成功请求，因此在相同依赖和数据下可以复现当前结果。

## 输出文件

### 表格数据

| 文件 | 内容 |
|---|---|
| `trace_input_sample.csv` | 100 个实际使用的请求及其映射代价 |
| `equilibrium_strategy_s_star.csv` | 最终均衡策略 `s*` |
| `convergence_trace.csv` | 每轮卸载数、势函数和系统总代价 |
| `algorithm_comparison.csv` | PSNE 与各基准算法的结果 |
| `memory_violation_sweep.csv` | 容量敏感性与违规率 |
| `llm_coordination_overhead.csv` | 数值协调器调用统计 |
| `parameters_and_calibration.json` | 假设参数与真实轨迹校准值 |
| `run_summary.json` | 核心运行结果摘要 |

### 结果图

`03_结果图/` ：


四张图分别为：

1. `convergence`：势函数收敛；
2. `equilibrium_strategy_s_star`：均衡策略分布；
3. `algorithm_comparison`：系统总代价比较；
4. `memory_violation_sweep`：容量违规率敏感性。

## 基准算法

- **All-Local**：全部本地执行；
- **All-Offload**：全部卸载；
- **Random**：以 0.5 概率随机卸载，报告 500 次均值；
- **Greedy**：只比较任务基础代价，忽略共享拥塞；
- **Social Optimum Diagnostic**：固定 `K` 下排序并枚举 `K`，仅用于诊断当前有限实例的系统总代价最优值。

诊断全局最优不是最佳响应算法的输出，也不构成最佳响应具有全局最优保证的证据。

## LLM 协调状态

当前协调器是确定性 Python 数值函数，而不是真实 LLM：

```text
llm_model = not_used
llm_token_count = not_measured
llm_inference_latency_ms = not_measured
```

因此本项目目前验证的是 Algorithm 1 的数学最佳响应机制。自然语言 Intent、Semantic Warning、Token、LLM 推理时延与解析错误仍属于后续实验。

## 仓库结构

```text
.
├── 00_原始数据/
│   └── 阿里_GenAI服务_2026/
├── 01_源码与说明/
│   ├── 仿真代码_mark2_alibaba.py
│   ├── 投稿级绘图_nature_figure_export.py
│   ├── 原始模型_AI_RAN博弈共识.pdf
│   └── 阿里GenAI2026_官方数据说明.md
├── 02_表格数据/
├── 03_结果图/
├── Mark2阿里_框架数据代码结果对应报告.pdf
├── requirements.txt
└── README.md
```

## 当前限制

- 没有真实 UE、Jetson 或其他边缘设备功耗测量；
- 没有真实无线信道、带宽、干扰、丢包和移动性；
- 容量是请求等价槽位，不是物理 GPU 硬容量；
- 没有真实 LLM 调用与边缘-中心通信；
- 一个真实请求被抽象为一个博弈玩家；
- 结果不能直接外推到生产 AI-RAN 部署。

## 引用

使用数据时，请按阿里官方数据说明引用 GenTD26 及其关联工作。使用本实验代码时，请在你的仓库或论文中注明本项目版本、随机种子、参数配置和修改内容。

## License

本仓库当前未附带代码许可证。公开发布前请根据你的使用计划选择并添加明确的开源许可证。阿里原始数据的使用与再分发遵循其上游项目说明，不受未来添加的代码许可证覆盖。
