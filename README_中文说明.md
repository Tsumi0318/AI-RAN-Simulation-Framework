# Mark2 阿里 - 真实生产轨迹驱动的 AI-RAN 卸载仿真

## 1. 实验定位

本实验使用阿里 GenAI Serving Top-Down Dataset 2026（GenTD26）的真实匿名生产轨迹，替换 Mark1 中部分随机输入。

- 真实输入来源：阿里大型 Stable Diffusion 生产服务。
- 场景：选择轨迹中请求数最高的小时 `2024-12-06 00:00-01:00`。
- 该小时成功请求数：436。
- 固定种子抽取：100 个成功请求作为 100 个博弈节点。
- 算法：异步严格最佳响应。
- 结果：收敛到逐节点验证的纯策略纳什均衡（PSNE）。
- 结论边界：这是“真实云端生产轨迹驱动仿真”，不是完整真实 AI-RAN 现场实验，也不是生产部署证明。

## 2. 哪些参数是真实数据

每个节点直接使用阿里请求轨迹中的：

- 请求创建时间 `gmt_create`。
- 任务类型 `predict_type`。
- 实际执行时间 `exec_time_seconds`。
- Prompt 长度 `prompt_length`。
- 推理步数 `num_inference_steps`。
- LoRA 数量 `num_lora`。

系统拥塞校准直接使用：

- 队列时延中位数和 P95：来自 `queue_rt_raw_anon.csv`。
- 推理时延中位数：来自 `model_predict_data_anon.csv`。
- GPU 显存分位数：来自 `pod_gpu_memory_used_bytes_anon.csv`。
- GPU 利用率分位数：来自 `pod_gpu_duty_cycle_anon.csv`。

具体校准值保存在 `02_表格数据/parameters_and_calibration.json`。

## 3. 哪些参数仍是假设

阿里轨迹没有端侧设备本地能耗、无线传输能耗、信道质量和逐请求显存增量，因此以下内容仍是明确假设：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `local_slowdown` | 1.8 | 假设端侧执行相对云端真实执行时间的倍率 |
| `local_energy_addition` | 0.4 | 本地执行的归一化能耗附加项 |
| `tx_base_cost` | 0.15 | 无线传输基础归一化代价 |
| `prompt_tx_weight` | 0.15 | Prompt 长度对传输代价的影响权重 |
| `capacity_fraction` | 0.80 | 假设中心池允许 80 个请求等价槽位 |
| `memory_alpha` | 1.2 | 内存障碍函数缩放 |
| `memory_beta` | 8.0 | 内存障碍函数陡峭程度 |

因此系统总代价仍是归一化指标，不能解释成焦耳、人民币或完整实际时延。

## 4. 核心结果

| 指标 | 结果 |
|---|---:|
| 异步更新次数 | 836 |
| 实际策略改变次数 | 80 |
| PSNE 卸载节点数 | 74/100 |
| 假设容量 | 80 个请求等价槽位 |
| PSNE 内存违规 | 否 |
| PSNE 系统总代价 | 208.1783 |
| 诊断全局最优卸载数 | 57/100 |
| 诊断全局最优总代价 | 178.5765 |
| PSNE 是否为全局最优 | 否 |

本实验验证的是：在真实阿里请求异质性与真实拥塞分位数校准下，所实现的最佳响应过程仍可收敛到 PSNE。它不证明 PSNE 是全局最优。

## 5. 五项核心输出

### 5.1 均衡策略 s*

- CSV：`02_表格数据/equilibrium_strategy_s_star.csv`
- 图片：`03_结果图/equilibrium_strategy_s_star.png`

字段：

| 字段 | 含义 |
|---|---|
| `node` | 从峰值小时抽取的真实请求编号 |
| `s_star` | 最终策略；0 为本地，1 为卸载 |
| `meaning` | `local` 或 `offload` |

### 5.2 收敛曲线

- CSV：`02_表格数据/convergence_trace.csv`
- 图片：`03_结果图/convergence.png`

| 字段 | 含义 |
|---|---|
| `iteration` | 异步更新次数 |
| `k_offload` | 当前卸载节点数 |
| `potential` | 势函数值 |
| `system_total_cost` | 当前归一化系统总代价 |
| `changed` | 本轮是否发生策略改变 |

势函数整体非增，并在 `changed=1` 时严格下降。算法停止前还会逐节点验证 PSNE。PDF 中“连续 N 次随机抽取无改变”的条件可能重复抽中同一节点，因此单独使用它不能保证所有节点都完成检查；Mark2 增加了全节点最终验证。

### 5.3 总代价对比

- CSV：`02_表格数据/algorithm_comparison.csv`
- 图片：`03_结果图/algorithm_comparison.png`

| 算法 | K | 系统总代价 | 内存违规 |
|---|---:|---:|---:|
| 最佳响应 PSNE | 74 | 208.1783 | 否 |
| All-Local | 0 | 255.0000 | 否 |
| All-Offload | 100 | 1030.6029 | 是 |
| Random 均值 | 约 50.09 | 约 202.7327 | 否 |
| 只看真实任务基础代价的 Greedy | 100 | 1030.6029 | 是 |
| 诊断全局最优 | 57 | 178.5765 | 否 |

Random 的均值低于本次 PSNE 不代表单个 Random 状态稳定，也不代表 Random 是更好的在线算法。它是 500 个随机状态的平均值，`is_psne=0`。

### 5.4 Memory Violation Rate

- CSV：`02_表格数据/memory_violation_sweep.csv`
- 图片：`03_结果图/memory_violation_sweep.png`

容量从 `K=50` 扫描到 `K=95`，每个容量点重复 100 次，比较最佳响应、Greedy 和 Random。这里的容量是请求等价槽位假设，不是阿里数据直接提供的物理 GPU 容量。

### 5.5 LLM 协调开销

- CSV：`02_表格数据/llm_coordination_overhead.csv`
- 数值协调调用：836 次。
- 策略改变：80 次。
- 真实 LLM：未调用。
- Token 与真实推理时延：`not_measured`。

不能把 Python 数值函数的运行时间当成 LLM 推理时延。

## 6. 真实轨迹输入表

`02_表格数据/trace_input_sample.csv` 保存本次使用的 100 个请求：

| 字段 | 来源与含义 |
|---|---|
| `gmt_create` | 阿里轨迹：请求创建时间 |
| `predict_type` | 阿里轨迹：生成任务类型 |
| `exec_time_seconds_trace` | 阿里轨迹：实际执行时间 |
| `prompt_length_trace` | 阿里轨迹：Prompt 长度 |
| `num_inference_steps_trace` | 阿里轨迹：推理步数 |
| `num_lora_trace` | 阿里轨迹：LoRA 数量 |
| `e_loc_assumed_mapped` | 根据真实执行时间和假设本地倍率映射的本地代价 |
| `e_tx_assumed_mapped` | 根据 Prompt 长度和假设权重映射的传输代价 |
| `equilibrium_strategy` | PSNE 策略 |
| `social_optimum_strategy` | 诊断全局最优策略 |

字段名中带 `_trace` 的是轨迹直接数据；带 `_assumed_mapped` 的是模型映射结果，不是阿里直接测量值。

## 7. 文件结构

- `01_源码与说明/`：Mark2 代码、原始模型 PDF、阿里官方数据说明。
- `02_表格数据/`：策略、收敛、对比、容量敏感性、输入样本、参数和摘要。
- `03_结果图/`：四张结果图。
- 完整原始数据位于上级目录的 `公开真实数据集/阿里_GenAI服务_2026/`，Mark2 不重复复制几十 MB 原始 CSV。

## 8. 复现

在 `AI RAN` 文件夹运行：

```bash
python3 mark2_alibaba_simulation.py
```

运行会重新生成 `Mark2 阿里/02_表格数据/` 和 `03_结果图/`。

## 9. 投稿级结果图

`03_结果图/` 中的四张图已按投稿级科研绘图规范重制。每张图提供：

- `SVG`：主编辑格式，文字保持可编辑。
- `PDF`：矢量投稿格式。
- `TIFF`：600 dpi 高分辨率格式。
- `PNG`：300 dpi 预览格式。

旧版 PNG 已被新版同名覆盖，不保留旧风格副本。绘图脚本为 `01_源码与说明/投稿级绘图_nature_figure_export.py`。
