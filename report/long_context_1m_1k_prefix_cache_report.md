# DSV4 Flash 1M 输入 / 1K 输出场景 Prefix Cache P/D 配置分析报告

生成日期：2026-04-28

## 1. Executive Summary

本文分析当前仓库 DeepSeek V4 Flash 建模配置在 `1,000,000` token 输入、`1,024` token 输出场景下的 P/D 分离部署配置。分析覆盖三种 prefix cache 命中率：`0%`、`90%`、`99%`。搜索方法是先分别选择 Prefill 与 Decode 阶段 `TPS/card` 最优实例，再使用实例级 QPS 做整数 P/D 配平。

核心结论如下：

- Prefix cache 命中率是该场景最关键的部署变量。它只降低 Prefill 计算量，不降低 HBM 估算口径下的完整上下文内存占用。
- 当前默认 HBM reserve 为 `10%`，因此 OOM 过滤使用设备标称 HBM 的 `90%` 作为可用上限：910C 为 `57.6 GB`，H20 为 `86.4 GB`。
- P/D 配比使用整数配平：`N * prefill_qps ~= M * decode_qps`，默认允许 `10%` 相对 QPS imbalance。该方法允许 `N` 和 `M` 同时大于 `1`，不再强制 `*:1D`。
- 在 910C 上，按 `TPS/card` 最优实例选型后的 P/D 配比分别为 `55P:1D`、`3P:2D`、`1P:9D`，对应命中率 `0%`、`90%`、`99%`。
- 在 H20 上，按 `TPS/card` 最优实例选型后的 P/D 配比分别为 `445P:1D`、`12P:1D`、`1P:1D`，对应命中率 `0%`、`90%`、`99%`。
- 无 prefix cache 命中时，1M 输入的 Prefill 成本极高，P/D 配平规模不适合作为常规在线服务路径。该场景必须依赖高 prefix cache 命中率、离线 Prefill、分块 Prefill、请求限流或其他系统级优化。
- 本文中的“配平卡数”不是全局最小卡数优化结果，而是在 P、D 各自选择 `TPS/card` 最优实例后，按照 QPS 整数配平得到的部署单元卡数。

## 2. 关键分析假设与方法论

### 2.1 场景定义

| 项目 | 取值 |
|---|---:|
| 输入长度 | `1,000,000` tokens |
| 输出长度 | `1,024` tokens |
| Prefix cache 命中率 | `0%`、`90%`、`99%` |
| Prefill GPU 搜索规模 | `8`、`16`、`32`、`64` cards |
| Decode GPU 搜索规模 | `8`、`16`、`32`、`64` cards |
| HBM reserve | `10%` |
| P/D 配平目标 | `N * prefill_qps ~= M * decode_qps` |
| P/D 配平误差容忍度 | `10%` |
| P/D 配平口径 | 实例级 QPS，不是单卡 QPS |

### 2.2 Prefix Cache 建模口径

Prefix cache 命中率只影响 Prefill 计算长度：

| 命中率 | 有效 Prefill 输入长度 |
|---:|---:|
| `0%` | `1,000,000` |
| `90%` | `100,000` |
| `99%` | `10,000` |

Decode 侧上下文长度始终保持 `1,000,000`。HBM 估算也始终按完整 1M 上下文计算，因此 prefix cache 命中不会降低本文中的 HBM footprint。

### 2.3 搜索与评价指标

分析流程分为三步：

1. 在 Prefill 阶段搜索所有合法 TP/EP/DP/Batch 组合，过滤不满足可用 HBM 约束的配置，选择逻辑 `prefill_tps_per_gpu` 最高的实例作为 P 实例。
2. 在 Decode 阶段搜索所有合法 TP/EP/DP/Batch 组合，过滤不满足可用 HBM 约束的配置，选择逻辑 `decode_tps_per_gpu` 最高的实例作为 D 实例。
3. 使用实例级 QPS 计算整数 P/D 配比，选择满足误差容忍度的最小整数部署单元；若有多个解，优先选择总实例数更小、误差更小的解。

其中：

- `TPS/card` 用于选择 P、D 各自的最优实例大小。
- `QPS` 用于计算 P/D 配比。
- `Prefill time` 表示所选 P 实例完成一个 batch Prefill 的耗时。
- `Decode total time` 表示所选 D 实例生成完整 `1,024` token 输出的近似总耗时。
- `Decode first step` 表示 1M 上下文下首个 decode step 的耗时。
- `HBM` 表示当前模型估算的单卡权重与 KV cache 总占用，不含工程运行时 overhead。

## 3. 结果总览

### 3.1 P/D 配平结果

| 硬件 | Prefix Cache 命中率 | P 实例卡数 | D 实例卡数 | P/D 配比 | Aggregate P QPS | Aggregate D QPS | QPS imbalance | 配平卡数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 910C | `0%` | 16 | 32 | `55P:1D` | 1.148888 | 1.253546 | 8.349% | 912 |
| 910C | `90%` | 16 | 32 | `3P:2D` | 2.449797 | 2.507093 | 2.285% | 112 |
| 910C | `99%` | 16 | 32 | `1P:9D` | 11.391257 | 11.281918 | 0.960% | 304 |
| H20 | `0%` | 16 | 32 | `445P:1D` | 3.860990 | 4.286600 | 9.929% | 7,152 |
| H20 | `90%` | 16 | 32 | `12P:1D` | 3.867993 | 4.286600 | 9.765% | 224 |
| H20 | `99%` | 16 | 32 | `1P:1D` | 4.423664 | 4.286600 | 3.098% | 48 |

### 3.2 HBM 占用与余量

由于 prefix cache 不改变 HBM 估算，三种命中率下同一硬件、同一阶段的 HBM 占用相同。当前设备配置中的 HBM reserve 为 `10%`，因此下表中的“可用 HBM”已经扣除了预留空间。

| 硬件 | 阶段 | 最优实例配置 | HBM 占用 | 可用 HBM | 理论余量 | 余量比例 | 风险判断 |
|---|---|---|---:|---:|---:|---:|---|
| 910C | Prefill | TP=2, EP=16, DP=8, BS=8, 16 cards | 51.783 GB | 57.6 GB | 5.817 GB | 10.1% | 可用但偏紧 |
| 910C | Decode | TP=4, EP=32, DP=8, BS=32, 32 cards | 51.919 GB | 57.6 GB | 5.681 GB | 9.9% | 可用但偏紧 |
| H20 | Prefill | TP=2, EP=16, DP=8, BS=32, 16 cards | 72.440 GB | 86.4 GB | 13.960 GB | 16.2% | 相对可接受 |
| H20 | Decode | TP=4, EP=32, DP=8, BS=64, 32 cards | 79.461 GB | 86.4 GB | 6.939 GB | 8.0% | 偏紧 |

上述结果说明，加入 `10%` HBM reserve 后，原先贴近标称容量的配置会被 OOM 过滤掉。H20 Prefill 从 8 卡配置切换到 16 卡配置，910C Prefill 的 batch size 从 16 降到 8。理论 HBM fit 仍不等价于工程可部署；若运行时还需要更高安全余量，应继续提高 `hbm_reserved_pct` 后重新搜索。

### 3.3 关键解读

在 `0%` 命中率下，两个硬件平台都受到 Prefill QPS 严重限制。910C 的 P/D 配比为 `55P:1D`，H20 为 `445P:1D`。这说明如果每个请求都需要完整 1M Prefill，该业务形态不适合作为普通在线实时服务路径。

在 `90%` 命中率下，910C 的整数配平结果为 `3P:2D`，aggregate QPS 误差约 `2.285%`，满足 `10%` 容忍度。H20 在该命中率下仍为 `12P:1D`，部署形态仍显著偏向 Prefill。

在 `99%` 命中率下，910C 的 P 实例 QPS 已显著高于 D 实例 QPS，因此配平结果变为 `1P:9D`；H20 配平为 `1P:1D`。高命中率下 Decode 资源占比上升，部署瓶颈从 Prefill-heavy 逐步转向更均衡的 P/D 供给。

## 4. 详细数据与分析

### 4.1 910C 结果

| Prefix Cache 命中率 | 有效 Prefill 长度 | P 实例配置 | P HBM | Prefill 耗时 | P QPS | P TPS/card | D 实例配置 | D HBM | Decode 总耗时 | Decode 首步耗时 | D QPS | D TPS/card | P/D 配比 | QPS imbalance | 配平卡数 |
|---:|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `0%` | 1,000,000 | TP=2, EP=16, DP=8, BS=8, 16 cards | 51.783 GB | 382.979 s | 0.020889 | 1,305.554 | TP=4, EP=32, DP=8, BS=32, 32 cards | 51.919 GB | 25.528 s | 24.928 ms | 1.253546 | 40.113 | `55P:1D` | 8.349% | 912 |
| `90%` | 100,000 | TP=2, EP=16, DP=8, BS=8, 16 cards | 51.783 GB | 9.797 s | 0.816599 | 51,037.430 | TP=4, EP=32, DP=8, BS=32, 32 cards | 51.919 GB | 25.528 s | 24.928 ms | 1.253546 | 40.113 | `3P:2D` | 2.285% | 112 |
| `99%` | 10,000 | TP=2, EP=16, DP=8, BS=8, 16 cards | 51.783 GB | 0.702 s | 11.391257 | 711,953.543 | TP=4, EP=32, DP=8, BS=32, 32 cards | 51.919 GB | 25.528 s | 24.928 ms | 1.253546 | 40.113 | `1P:9D` | 0.960% | 304 |

910C 的最优实例选择在三种命中率下保持稳定。P 实例为 `TP=2, EP=16, DP=8, BS=8`，共 `16` 卡；D 实例为 `TP=4, EP=32, DP=8, BS=32`，共 `32` 卡。prefix cache 命中率提升后，P 实例 HBM 不变，但 Prefill 耗时从 `382.979 s` 降至 `9.797 s` 和 `0.702 s`，实例 QPS 分别提升到 `0.816599` 和 `11.391257`。

### 4.2 H20 结果

| Prefix Cache 命中率 | 有效 Prefill 长度 | P 实例配置 | P HBM | Prefill 耗时 | P QPS | P TPS/card | D 实例配置 | D HBM | Decode 总耗时 | Decode 首步耗时 | D QPS | D TPS/card | P/D 配比 | QPS imbalance | 配平卡数 |
|---:|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `0%` | 1,000,000 | TP=2, EP=16, DP=8, BS=32, 16 cards | 72.440 GB | 3,688.173 s | 0.008676 | 542.274 | TP=4, EP=32, DP=8, BS=64, 32 cards | 79.461 GB | 14.930 s | 14.579 ms | 4.286600 | 137.171 | `445P:1D` | 9.929% | 7,152 |
| `90%` | 100,000 | TP=2, EP=16, DP=8, BS=32, 16 cards | 72.440 GB | 99.276 s | 0.322333 | 20,145.795 | TP=4, EP=32, DP=8, BS=64, 32 cards | 79.461 GB | 14.930 s | 14.579 ms | 4.286600 | 137.171 | `12P:1D` | 9.765% | 224 |
| `99%` | 10,000 | TP=2, EP=16, DP=8, BS=32, 16 cards | 72.440 GB | 7.234 s | 4.423664 | 276,478.973 | TP=4, EP=32, DP=8, BS=64, 32 cards | 79.461 GB | 14.930 s | 14.579 ms | 4.286600 | 137.171 | `1P:1D` | 3.098% | 48 |

H20 的最优 P 实例为 `TP=2, EP=16, DP=8, BS=32`，共 `16` 卡；最优 D 实例为 `TP=4, EP=32, DP=8, BS=64`，共 `32` 卡。加入 `10%` HBM reserve 后，原先 8 卡 P 实例不再满足可用 HBM 约束，P 实例规模上升到 16 卡。H20 的 Decode QPS 较高，但 1M Prefill QPS 在低命中率下过低，导致低命中率下 P/D 配平仍然极端 Prefill-heavy。

### 4.3 跨平台对比

| Prefix Cache 命中率 | 910C P QPS | 910C D QPS | 910C P/D | H20 P QPS | H20 D QPS | H20 P/D | 结论 |
|---:|---:|---:|---:|---:|---:|---:|---|
| `0%` | 0.020889 | 1.253546 | `55P:1D` | 0.008676 | 4.286600 | `445P:1D` | 完整 1M Prefill 成本过高，不宜作为普通在线路径。 |
| `90%` | 0.816599 | 1.253546 | `3P:2D` | 0.322333 | 4.286600 | `12P:1D` | 910C 可用多个 D 实例配平，H20 仍显著 Prefill-heavy。 |
| `99%` | 11.391257 | 1.253546 | `1P:9D` | 4.423664 | 4.286600 | `1P:1D` | 高命中率下 Decode 侧资源占比上升，P/D 更接近均衡部署。 |

### 4.4 部署含义

该场景本质上依赖 prefix cache。若命中率无法稳定保持在较高水平，Prefill 集群规模会迅速膨胀，且端到端首 token 等待时间会被 1M Prefill 主导。

在当前模型假设下，910C 更适合命中率不确定或中等命中率场景，因为其 Prefill QPS 相对更高，P/D 配比更可控。H20 在 Decode 阶段具有更高实例 QPS，但低命中率下会放大 Prefill 与 Decode 的供给不平衡。

## 5. 风险与适用边界

- 本报告基于当前仓库配置文件和 roofline 建模实现，不额外适配外部 DeepSeek V4 Flash 规格。
- Decode 总耗时使用近似估算，不是逐 token 完整仿真。
- 本文选择的是 P、D 各自 `TPS/card` 最优实例，并在此基础上做 QPS 配平；它不等价于全局最小卡数搜索。
- P/D 整数配平使用 `10%` 默认误差容忍度。更严格的容忍度会产生更大的整数部署单元；更宽松的容忍度会产生更小但供给不平衡更大的部署单元。
- HBM reserve 默认值为 `10%`。更高 reserve 会进一步过滤掉贴近 HBM 上限的配置，并可能改变 P/D 最优实例。
- Prefix cache 在本文中只改变 Prefill 计算量，不改变 HBM 估算。这符合当前需求约束，但可能与某些生产系统的 cache eviction、KV materialization 或跨节点传输策略不同。
- 表中 HBM 仅覆盖当前模型实现中的权重和 KV cache 估算，不包含运行时 allocator 碎片、通信 buffer、框架额外 workspace 等工程余量。

## 附录 A：可复现实验命令

以下命令在仓库根目录执行。附录只给出 bash 命令；具体计算逻辑由仓库脚本封装。

```bash
python report/long_context_1m_1k_analysis.py \
  --seq-len 1000000 \
  --output-len 1024 \
  --hit-rates 0 0.9 0.99 \
  --hardware 910C H20 \
  --json-out report/data/long_context_1m_1k_prefix_cache.json
```

校验 JSON 结果文件格式：

```bash
python -m json.tool report/data/long_context_1m_1k_prefix_cache.json >/dev/null
```

查看生成的摘要输出：

```bash
python report/long_context_1m_1k_analysis.py \
  --seq-len 1000000 \
  --output-len 1024 \
  --hit-rates 0 0.9 0.99 \
  --hardware 910C H20
```
