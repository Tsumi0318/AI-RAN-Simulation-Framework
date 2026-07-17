# AI-RAN语义协调卸载实验（DeepSeek + GenTD26）

## 1. 实验定位

本实验以《AI RAN博弈共识.pdf》的二元卸载博弈、Algorithm 1和第3节验证指标为主线，并落实导师提出的五点要求：

1. 每个任务携带Prompt长度、Steps、图片数量、任务类型和LoRA等语义特征；
2. 使用阿里GenTD26网关队列长度、队列时延和调度管线延迟校准中心代价；
3. 使用GPU VRAM波动拟合指数障碍函数；
4. DeepSeek读取当前拥塞状态与任务Intent，返回中心代价和语义警告；
5. LLM封装为独立Python类，博弈公式和策略更新不依赖具体API实现。

实验实际调用DeepSeek官方API。请求别名为`deepseek-chat`，服务端返回模型标识为`deepseek-v4-flash`。API Key没有保存在任何Mark4文件中。

## 2. 核心结论

| 指标 | 实际结果 |
|---|---:|
| 主实验节点数 | 100 |
| 异步更新次数 | 543 |
| 实际策略改变 | 56 |
| 最终卸载/本地 | 71 / 29 |
| 全节点PSNE复核 | 通过 |
| 势函数值 | 230.8965 |
| 归一化系统总代价 | 249.1911 |
| 16GB仿真显存负载 | 14.58GB（91.13%） |
| 仿真容量违规 | 否 |
| 模拟平均能耗 | 160380.16mJ/task |
| 卸载任务队列时延 | 214.03ms/task |
| 全部任务平均队列时延 | 151.96ms/task |

这些结果只对应当前有限样本、参数与仿真器。PSNE表示单个节点没有单方面改变策略的动力，不表示全局最优。

## 3. 数据与任务Intent

数据来自阿里GenTD26匿名生产轨迹。Mark4保留了本实验实际使用的官方文件：

- `lora_request_trace.csv`：请求任务特征与执行时间；
- `queue_size_raw_anon.csv`：网关队列长度；
- `queue_rt_raw_anon.csv`：网关队列时延；
- `pipeline_update_latency_anon.csv`：模型/LoRA管线更新时延；
- `model_predict_data_anon.csv`：纯推理时延；
- `pod_gpu_memory_used_bytes_anon.csv`：GPU显存使用；
- `pod_gpu_duty_cycle_anon.csv`：GPU利用率；
- `官方_README.md`：数据发布方说明。

主实验从峰值小时`2024-12-06 00:00:00`的435个成功请求中固定抽取200个任务作为语义池，前100个用于主博弈。Intent示例：

```json
{
  "prompt_length_chars": 67,
  "steps": 30,
  "num_images": 1,
  "lora": false,
  "local_energy_cost_normalized": 2.9,
  "tx_energy_cost_normalized": 0.28
}
```

原始字段是Prompt字符数，不伪装成Token数。

## 4. PDF公式及实现

节点策略与卸载数：

$$
s_i\in\{0,1\},\qquad K(\mathbf{s})=\sum_{i=1}^{N}s_i.
$$

本地和传输能耗项仍采用PDF的二元结构：

$$
E_i(s_i)=(1-s_i)e_{i,\mathrm{loc}}+s_i e_{i,\mathrm{tx}}.
$$

M/M/1拥塞函数：

$$
D_{\mathrm{comp}}(K)=\frac{1}{\mu-\lambda K}.
$$

使用1002个队列长度与队列时延的同时间戳匹配点进行鲁棒拟合：

$$
\mu=4.69351\ \mathrm{s}^{-1},\qquad
\lambda=2.9811\times10^{-4}\ \mathrm{task}^{-1}\mathrm{s}^{-1}.
$$

诊断$R^2=-0.0116$，说明匿名数据中的队列长度与时延关联很弱。这两个参数只能称为仿真代理，不能称为物理识别出的到达率和服务率。

PDF的统一任务显存公式为：

$$
M(K)=\alpha\exp\left[\beta(Kv_{\mathrm{req}}-V_{\max})\right].
$$

为支持任务异质性，本实验使用归一化总显存负载：

$$
B(V)=\alpha\exp\left[\beta\left(\frac{V}{V_{\max}}-1\right)\right].
$$

拟合结果为$\alpha=5.8080$、$\beta=11.0109$、$R^2=0.9918$。其中$\beta$和尾部形状来自真实VRAM波动；16GB容量以及容量点障碍代价5.0属于明确假设。

节点$i$候选卸载时的中心增量为：

$$
\Delta_{\mathrm{center},i}=
C_{i,\mathrm{compute}}+D_{\mathrm{comp}}(K_{-i}+1)
+B(V_{-i}+v_i)-B(V_{-i}).
$$

节点执行严格最佳响应：

$$
s_i=
\begin{cases}
1,&e_{i,\mathrm{tx}}+\Delta_{\mathrm{center},i}<e_{i,\mathrm{loc}},\\
0,&\text{otherwise}.
\end{cases}
$$

扩展势函数为：

$$
\Phi(\mathbf{s})=
\sum_i E_i^{\mathrm{base}}(s_i)
+\sum_{k=1}^{K(\mathbf{s})}D_{\mathrm{comp}}(k)
+B(V(\mathbf{s})).
$$

200次随机状态翻转审计中：

$$
\max|\Delta C_i-\Delta\Phi|=9.19\times10^{-14}.
$$

这证明实现内部的精确势恒等式，不证明物理模型正确，也不证明全局最优。

## 5. DeepSeek协调流程

```text
任务语义池
→ DeepSeek固定解析每个任务的compute/vram multiplier
→ 随机初始化100节点策略
→ 随机选择节点i
→ 发送当前K、VRAM负载与Intent
→ DeepSeek返回中心代价、资源影响与语义警告
→ 公式校验器重新计算并校验中心代价
→ 节点严格最佳响应
→ 连续100轮无变化后进行全节点PSNE复核
```

LLM不直接决定$s_i$。本次有1次LLM原始数值没有满足公式容差，已由确定性公式校验器纠正，语义信息仍保留。

## 6. PDF第3节完整输出

### 6.1 四个基准算法

| 算法 | 卸载数 | 归一化总代价 | 16GB容量违规 |
|---|---:|---:|---:|
| DeepSeek Best Response PSNE | 71 | 249.19 | 否 |
| All-Local | 0 | 280.60 | 否 |
| All-Offload | 100 | 10709.36 | 是 |
| Random（500次均值） | 49.88 | 257.12 | 0% |
| Greedy | 90 | 1384.93 | 是 |

当前PSNE低于四个基准，但这里只是当前实例对比，不是统计显著性或全局最优证明。

### 6.2 收敛与稳定性

势函数仅在策略改变时严格下降，无改变轮次保持不变，最终通过全节点PSNE检查。实际用了543轮，超过PDF预期的$N$到$3N$（100到300轮），因此不声称满足该预期时间范围。

### 6.3 帕累托候选与系统总代价

`pareto_front.csv`按局部卸载收益排序并枚举$K=0\ldots100$，报告模拟能耗与全任务平均排队时延。它是一个可复核的候选前沿，不是遍历$2^{100}$种策略得到的全局帕累托前沿。

模拟能耗由10W边缘功率、本地1.8倍时长、无线每KB能耗等假设换算，不能当成Jetson实测值。

### 6.4 Memory Violation Rate

按PDF要求扫描$N=30$到200，每个$N$运行30个固定种子：

- DeepSeek最佳响应PSNE：全部$N$下仿真违规率为0%；
- All-Offload和Greedy：从$N=80$开始为100%；
- Random：从$N=120$开始出现违规，到$N=180$达到100%。

这里是16GB软件仿真器中的容量代理，不是物理RTX 4080 OOM实验。

### 6.5 LLM协调开销

| 指标 | 结果 |
|---|---:|
| 语义解析逻辑调用 | 200 |
| 博弈协调逻辑调用 | 543 |
| 真实API记录 | 662 |
| 博弈状态复用 | 81 |
| Token总量 | 255008 |
| 平均API时延 | 1.309秒 |
| P95 API时延 | 1.866秒 |

实测为秒级，不支持PDF中“微秒至毫秒级”的预期结论。

## 7. 文件结构

```text
Mark4/
├── 00_原始数据/GenTD26/       官方数据与校验来源
├── 01_源码与说明/             主实验、绘图代码和输出说明
├── 02_表格数据/               Intent、拟合、轨迹、策略、基准和指标CSV
└── 03_结果图/                 PNG/PDF/SVG/TIFF四种格式
```

五组最终结果图：

1. `01_convergence_stability`：势函数、卸载数和容量轨迹；
2. `02_pareto_system_cost`：四基准总代价与能耗-时延候选前沿；
3. `03_memory_violation_rate`：$N=30$到200容量违规率；
4. `04_llm_orchestration_overhead`：时延、调用量和Token；
5. `05_formula_calibration`：M/M/1与VRAM障碍拟合。

## 8. 复现

安装依赖后，将新DeepSeek Key仅设置为环境变量：

```bash
export DEEPSEEK_API_KEY="your-new-key"
python3 "01_源码与说明/mark4_deepseek_experiment.py" --full
python3 "01_源码与说明/plot_mark4_results.py"
```

仅重新拟合数据而不调用API：

```bash
python3 "01_源码与说明/mark4_deepseek_experiment.py" --prepare
```

缓存文件允许在不重复计费的情况下复现相同语义和协调回复。若要形成全新的API实验，应在保留旧版本备份后使用新的空输出目录。

## 9. 结论边界

本实验能够说明：在当前GenTD26任务样本、明确假设和拟合代理下，DeepSeek语义协调的异步最佳响应可以收敛到一个经全节点验证的PSNE，并在16GB仿真容量下避免违规，同时取得低于四个基准的当前实例总代价。

本实验不能证明全局最优、一般规律、真实硬件零OOM、真实Jetson能耗、M/M/1物理参数有效或工程部署可行。
