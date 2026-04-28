# 0428 PD 分离推理分析报告

## Executive Summary

本报告基于 `report/0428/data/*.json` 中的当前结果，分析 DeepSeek V4 Flash W8A8 量化模型在 Ascend 910C 上的四类 PD 分离推理场景：Prefill 8K/32K/128K/1M + Decode 1K。核心结论如下：

- Prefill：四个场景在 `batch_size = 1` 下，满足“模型权重 + bs=1 KV cache” HBM 放置的最小实例均为 8 卡；当前最优并行策略均为 `TP=8, EP=8, DP=1`。随着 prefix cache 命中率提升，Prefill HBM 不变，但有效计算长度显著下降，Prefill QPS 显著提升。
- Decode：在 8/16/32/64 卡实例中，四个场景的 No MTP 与 MTP=1 最优 Decode 实例均为 64 卡，最优策略均为 `TP=1, EP=64, DP=64`。MTP=1 在本模型中通过减少 decode forward 次数提升 TPS/card，但不改变 HBM-only 最大 batch。
- P/D 配比：按“选定 Prefill 实例 + 最优 Decode 实例”求 `Prefill QPS = Decode QPS`，MTP=1 的 Decode QPS 更高，因此需要更多 Prefill 实例与其配平；prefix cache 命中率越高，所需 Prefill 实例越少。
- 推荐侧重点：若以 Decode TPS/card 最大化为目标，四个场景均选择 64 卡 Decode + MTP=1；若关注总卡规模，prefix cache 命中率是 P/D 资源需求的主导变量，尤其在 128K 与 1M 场景。

MTP=1 下的 P/D 推荐配比如下，覆盖 `prefix_cache_hit_rate = 0 / 0.9 / 0.99` 三种情况：

| 场景 | h=0 | h=0.9 | h=0.99 |
| --- | --- | --- | --- |
| 8K + 1K | 369P:1D，3,016 卡 | 84P:1D，736 卡 | 58P:1D，528 卡 |
| 32K + 1K | 1054P:1D，8,496 卡 | 129P:1D，1,096 卡 | 48P:1D，448 卡 |
| 128K + 1K | 2535P:1D，20,344 卡 | 187P:1D，1,560 卡 | 34P:1D，336 卡 |
| 1M + 1K | 10352P:1D，82,880 卡 | 285P:1D，2,344 卡 | 24P:1D，256 卡 |

## 1. 分析目标与范围

目标是对四个 PD 分离推理场景做实例 sizing、TPOT 约束下的 Decode 吞吐评估，以及 Prefill/Decode 实例配比求解：

| 场景 | Prefill 输入长度 | Decode 输出长度 |
| --- | ---: | ---: |
| 8K + 1K | 8,192 | 1,024 |
| 32K + 1K | 32,768 | 1,024 |
| 128K + 1K | 131,072 | 1,024 |
| 1M + 1K | 1,000,000 | 1,024 |

报告使用仓库内 Ascend 910C、DeepSeek V4 Flash 与 runtime 配置，分析范围包括：

- Prefill：最小可行实例卡数、HBM 占用、最优并行策略，以及 `prefix_cache_hit_rate` 分别为 0、0.9、0.99 时的性能。
- Decode：8/16/32/64 卡实例下的 HBM-only 最大单卡 batch、TPOT=50ms 约束下最大单卡 batch、TPS/card、QPS 与最优策略；同时比较 No MTP 与 MTP=1。
- P/D 配比：基于每个场景的 Prefill 结果和最佳 Decode 实例，分别计算 No MTP 与 MTP=1 的 P/D 实例比例，覆盖三个 prefix cache 命中率。

## 2. 方法、假设与建模限制

### 2.1 方法

Prefill sizing 按候选卡数 `[1, 2, 4, 8, 16, 32, 64]` 搜索，固定 `batch_size=1`，筛选“模型权重 + 完整输入 KV cache”可放入可用 HBM 的最小实例；在同一最小卡数内选择 Prefill TPS/card 更高的并行策略。

Decode sizing 按实例卡数 `[8, 16, 32, 64]` 搜索。每个实例先求 HBM-only 最大单卡 batch，即只要求模型权重和 Decode context KV cache 可放入 HBM；再在该上限内求 `TPOT <= 50ms` 的最大单卡 batch，并输出对应 TPS/card、QPS 与并行策略。No MTP 与 MTP=1 分开评估。

P/D 配比使用实例级 QPS 求解：

`prefill_instances * prefill_qps ~= decode_instances * decode_qps`

当前结果允许 10% 以内 QPS imbalance。表中 P/D 比例均为实例数比例，不是卡数比例；总卡数按 `Prefill 实例数 * Prefill 卡/实例 + Decode 实例数 * Decode 卡/实例` 计算。

### 2.2 关键假设

| 项目 | 取值 |
| --- | --- |
| 硬件 | Ascend 910C |
| 模型 | DeepSeek V4 Flash |
| 量化 | W8A8 |
| KV cache 量化 | KV8 |
| W8A8 GEMM 吞吐 | 752.0 TFLOPS |
| HBM 容量/预留/可用 | 64 GB / 10.0% / 57.6 GB |
| prefix_cache_hit_rate | 0, 0.9, 0.99 |
| Decode TPOT 目标 | 50 ms |
| MTP accept ratio | 0.9 |

Prefix cache 的建模方式为：

`L_miss = ceil(input_len * (1 - prefix_cache_hit_rate))`

Prefill compute 使用 `L_miss`，但 HBM 仍按完整 input context 计算；因此同一场景的三种 hit rate 有相同 HBM 占用，但 prefill latency/QPS 不同。

MTP=1 的建模方式为：

`tokens_per_forward = 1 + mtp * mtp_accept_ratio = 1.9`

Decode output length 为 1,024，因此 MTP=1 将 decode forward count 从 1,024 降到 539；当前结果未加入 MTP 额外 head 权重或专属计算开销。

### 2.3 建模限制

- 未建模 quant/dequant kernel 时间、runtime 调度开销、allocator fragmentation 以外的额外 HBM 损耗。
- Prefix cache 当前只降低 Prefill compute，不降低 Prefill HBM 与 Decode HBM。
- 未建模 P/D KV transfer、跨实例网络传输、排队、动态 batching、实际请求分布和拓扑放置约束。
- P/D 配比只按稳态 QPS 配平，不代表端到端延迟 SLO、资源碎片和故障冗余策略。

## 3. Prefill 实例 sizing 与性能

Prefill 的最小可行实例由 HBM 约束决定。当前四个场景都在 8 卡实例首次满足 HBM 放置，且最优策略一致为 `TP=8, EP=8, DP=1`。8 卡实例下，模型权重约 37.27-37.36 GB，bs=1 KV cache 随输入长度增长；1M 场景 KV cache 为 3.44 GB，总 HBM 约 40.80 GB，占可用 HBM 约 70.8%。

![Prefill HBM](figure/prefill_hbm.svg)

![Prefill TPS](figure/prefill_tps.svg)

| 场景 | Hit | 最小卡/实例 | 最优策略 | BS | L_miss | Weight GB | KV GB | HBM GB | HBM 占用 | Prefill ms | QPS/实例 | TPS/card |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K + 1K | 0 | 8 | TP=8, EP=8, DP=1 | 1 | 8,192 | 37.27 | 0.03 | 37.30 | 64.8% | 299.81 | 3.34 | 3,415.47 |
| 8K + 1K | 0.9 | 8 | TP=8, EP=8, DP=1 | 1 | 820 | 37.27 | 0.03 | 37.30 | 64.8% | 67.96 | 14.72 | 15,068.29 |
| 8K + 1K | 0.99 | 8 | TP=8, EP=8, DP=1 | 1 | 82 | 37.27 | 0.03 | 37.30 | 64.8% | 46.48 | 21.51 | 22,031.23 |
| 32K + 1K | 0 | 8 | TP=8, EP=8, DP=1 | 1 | 32,768 | 37.27 | 0.12 | 37.39 | 64.9% | 1,174.48 | 0.85 | 3,487.49 |
| 32K + 1K | 0.9 | 8 | TP=8, EP=8, DP=1 | 1 | 3,277 | 37.27 | 0.12 | 37.39 | 64.9% | 143.82 | 6.95 | 28,479.69 |
| 32K + 1K | 0.99 | 8 | TP=8, EP=8, DP=1 | 1 | 328 | 37.27 | 0.12 | 37.39 | 64.9% | 53.09 | 18.84 | 77,155.41 |
| 128K + 1K | 0 | 8 | TP=8, EP=8, DP=1 | 1 | 131,072 | 37.36 | 0.45 | 37.81 | 65.6% | 6,288.22 | 0.16 | 2,605.51 |
| 128K + 1K | 0.9 | 8 | TP=8, EP=8, DP=1 | 1 | 13,108 | 37.36 | 0.45 | 37.81 | 65.6% | 462.22 | 2.16 | 35,445.97 |
| 128K + 1K | 0.99 | 8 | TP=8, EP=8, DP=1 | 1 | 1,311 | 37.36 | 0.45 | 37.81 | 65.6% | 82.99 | 12.05 | 197,417.06 |
| 1M + 1K | 0 | 8 | TP=8, EP=8, DP=1 | 1 | 1,000,000 | 37.36 | 3.44 | 40.80 | 70.8% | 160,290.66 | 0.01 | 779.83 |
| 1M + 1K | 0.9 | 8 | TP=8, EP=8, DP=1 | 1 | 100,000 | 37.36 | 3.44 | 40.80 | 70.8% | 4,399.88 | 0.23 | 28,409.86 |
| 1M + 1K | 0.99 | 8 | TP=8, EP=8, DP=1 | 1 | 10,000 | 37.36 | 3.44 | 40.80 | 70.8% | 358.81 | 2.79 | 348,369.99 |

Prefill 侧的主要现象是：HBM 随完整上下文长度增长，但在 bs=1 下仍由权重占主导；prefix cache 命中率不会改变 HBM sizing，却会强烈影响 Prefill QPS。例如 1M 场景从 h=0 到 h=0.99，Prefill latency 从 160.29s 降到 358.81ms，P/D 配比也随之大幅收敛。

## 4. Decode 实例 sizing、TPOT 约束与吞吐

Decode 表中的 `HBM B/card` 表示仅考虑 HBM 时可达到的最大单卡 batch；`TPOT B/card` 表示同时满足 `TPOT <= 50ms` 时的最大单卡 batch。图中每个点的标签采用 `HBM/TPOT` 简写，例如 `1320/552` 表示该点 HBM-only 上限为 1320 batch/card，而 TPOT=50ms 约束下只能取到 552 batch/card；图内也给出了该标签含义。

当前所有 Decode 表项均可行，因此没有 N/A 项。四个场景的最佳 Decode 实例均为 64 卡；No MTP 与 MTP=1 的最优策略均为 `TP=1, EP=64, DP=64`。

### 4.1 8K + 1K

![Decode 8K + 1K](figure/decode_8k_1k.svg)

8K decode 的 HBM-only batch 余量较高，No MTP 主要被 TPOT 卡住；MTP=1 后 8/16 卡实例可直接达到 HBM 上限，32/64 卡实例仍受 TPOT 约束。

| 模式 | 卡/实例 | HBM B/card | TPOT B/card | TPOT ms | TPS/card | QPS/实例 | 最优策略 | 最佳实例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| No MTP | 8 | 443 | 304 | 49.98 | 6,082.44 | 47.52 | TP=1, EP=8, DP=8 | No |
| No MTP | 16 | 944 | 446 | 49.93 | 8,932.23 | 139.57 | TP=1, EP=16, DP=16 | No |
| No MTP | 32 | 1,195 | 517 | 49.97 | 10,346.15 | 323.32 | TP=1, EP=32, DP=32 | No |
| No MTP | 64 | 1,320 | 552 | 49.96 | 11,048.04 | 690.50 | TP=1, EP=64, DP=64 | Yes |
| MTP=1 | 8 | 443 | 443 | 32.34 | 13,697.29 | 107.01 | TP=1, EP=8, DP=8 | No |
| MTP=1 | 16 | 944 | 944 | 48.06 | 19,643.45 | 306.93 | TP=1, EP=16, DP=16 | No |
| MTP=1 | 32 | 1,195 | 1,057 | 50.00 | 21,141.04 | 660.66 | TP=1, EP=32, DP=32 | No |
| MTP=1 | 64 | 1,320 | 1,091 | 49.99 | 21,823.46 | 1,363.97 | TP=1, EP=64, DP=64 | Yes |

### 4.2 32K + 1K

![Decode 32K + 1K](figure/decode_32k_1k.svg)

32K decode 已基本由 HBM 决定 batch 上限；No MTP 与 MTP=1 在所有实例大小上都可以使用 HBM-only 最大 batch，MTP=1 主要体现为 TPS/card 提升。

| 模式 | 卡/实例 | HBM B/card | TPOT B/card | TPOT ms | TPS/card | QPS/实例 | 最优策略 | 最佳实例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| No MTP | 8 | 128 | 128 | 39.27 | 3,259.67 | 25.47 | TP=1, EP=8, DP=8 | No |
| No MTP | 16 | 273 | 273 | 42.53 | 6,418.94 | 100.30 | TP=1, EP=16, DP=16 | No |
| No MTP | 32 | 346 | 346 | 44.55 | 7,767.14 | 242.72 | TP=1, EP=32, DP=32 | No |
| No MTP | 64 | 382 | 382 | 45.52 | 8,392.45 | 524.53 | TP=1, EP=64, DP=64 | Yes |
| MTP=1 | 8 | 128 | 128 | 20.67 | 6,192.77 | 48.38 | TP=1, EP=8, DP=8 | No |
| MTP=1 | 16 | 273 | 273 | 22.39 | 12,194.80 | 190.54 | TP=1, EP=16, DP=16 | No |
| MTP=1 | 32 | 346 | 346 | 23.45 | 14,756.12 | 461.13 | TP=1, EP=32, DP=32 | No |
| MTP=1 | 64 | 382 | 382 | 23.96 | 15,944.09 | 996.51 | TP=1, EP=64, DP=64 | Yes |

### 4.3 128K + 1K

![Decode 128K + 1K](figure/decode_128k_1k.svg)

128K decode 的 KV cache 显著压低 batch 上限，所有实例大小的 TPOT-constrained batch 都等于 HBM-only batch。MTP=1 继续提升 TPS/card，但资源上限仍由 HBM 主导。

| 模式 | 卡/实例 | HBM B/card | TPOT B/card | TPOT ms | TPS/card | QPS/实例 | 最优策略 | 最佳实例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| No MTP | 8 | 31 | 31 | 34.80 | 890.88 | 6.96 | TP=1, EP=8, DP=8 | No |
| No MTP | 16 | 69 | 69 | 29.68 | 2,324.48 | 36.32 | TP=1, EP=16, DP=16 | No |
| No MTP | 32 | 88 | 88 | 27.15 | 3,241.08 | 101.28 | TP=1, EP=32, DP=32 | No |
| No MTP | 64 | 98 | 98 | 25.98 | 3,772.01 | 235.75 | TP=1, EP=64, DP=64 | Yes |
| MTP=1 | 8 | 31 | 31 | 18.32 | 1,692.50 | 13.22 | TP=1, EP=8, DP=8 | No |
| MTP=1 | 16 | 69 | 69 | 15.62 | 4,416.09 | 69.00 | TP=1, EP=16, DP=16 | No |
| MTP=1 | 32 | 88 | 88 | 14.29 | 6,157.45 | 192.42 | TP=1, EP=32, DP=32 | No |
| MTP=1 | 64 | 98 | 98 | 13.68 | 7,166.11 | 447.88 | TP=1, EP=64, DP=64 | Yes |

### 4.4 1M + 1K

![Decode 1M + 1K](figure/decode_1m_1k.svg)

1M decode 是最典型的 HBM-bound 场景：单卡 batch 上限只有 4/9/11/13。由于 batch 很小，所有点都满足 TPOT=50ms，MTP=1 主要通过减少 forward 次数提高吞吐。

| 模式 | 卡/实例 | HBM B/card | TPOT B/card | TPOT ms | TPS/card | QPS/实例 | 最优策略 | 最佳实例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| No MTP | 8 | 4 | 4 | 33.28 | 120.20 | 0.94 | TP=1, EP=8, DP=8 | No |
| No MTP | 16 | 9 | 9 | 26.41 | 340.84 | 5.33 | TP=1, EP=16, DP=16 | No |
| No MTP | 32 | 11 | 11 | 22.46 | 489.85 | 15.31 | TP=1, EP=32, DP=32 | No |
| No MTP | 64 | 13 | 13 | 21.51 | 604.33 | 37.77 | TP=1, EP=64, DP=64 | Yes |
| MTP=1 | 8 | 4 | 4 | 17.52 | 228.35 | 1.78 | TP=1, EP=8, DP=8 | No |
| MTP=1 | 16 | 9 | 9 | 13.90 | 647.53 | 10.12 | TP=1, EP=16, DP=16 | No |
| MTP=1 | 32 | 11 | 11 | 11.82 | 930.63 | 29.08 | TP=1, EP=32, DP=32 | No |
| MTP=1 | 64 | 13 | 13 | 11.32 | 1,148.11 | 71.76 | TP=1, EP=64, DP=64 | Yes |

## 5. Prefill/Decode 配比

P/D 配比基于第 3 节选出的 Prefill 8 卡实例，以及第 4 节每个场景和 Decode 模式下 TPS/card 最大的 64 卡 Decode 实例。表中 `Prefill QPS` 与 `Decode QPS` 是配比后的聚合 QPS，imbalance 为两侧 QPS 差异比例；所有条目均满足当前 10% imbalance 容忍度。

![P/D Total Cards](figure/pd_ratio_total_cards.svg)

| 场景 | Hit | Decode 模式 | Prefill 卡/实例 | Decode 卡/实例 | P:D | Prefill QPS | Decode QPS | imbalance | 总卡数 | 推荐 |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 8K + 1K | 0 | No MTP | 8 | 64 | 187P:1D | 623.72 | 690.50 | 9.7% | 1,560 | No |
| 8K + 1K | 0 | MTP=1 | 8 | 64 | 369P:1D | 1,230.77 | 1,363.97 | 9.8% | 3,016 | Yes |
| 8K + 1K | 0.9 | No MTP | 8 | 64 | 43P:1D | 632.75 | 690.50 | 8.4% | 408 | No |
| 8K + 1K | 0.9 | MTP=1 | 8 | 64 | 84P:1D | 1,236.07 | 1,363.97 | 9.4% | 736 | Yes |
| 8K + 1K | 0.99 | No MTP | 8 | 64 | 29P:1D | 623.93 | 690.50 | 9.6% | 296 | No |
| 8K + 1K | 0.99 | MTP=1 | 8 | 64 | 58P:1D | 1,247.86 | 1,363.97 | 8.5% | 528 | Yes |
| 32K + 1K | 0 | No MTP | 8 | 64 | 555P:1D | 472.55 | 524.53 | 9.9% | 4,504 | No |
| 32K + 1K | 0 | MTP=1 | 8 | 64 | 1054P:1D | 897.42 | 996.51 | 9.9% | 8,496 | Yes |
| 32K + 1K | 0.9 | No MTP | 8 | 64 | 68P:1D | 472.81 | 524.53 | 9.9% | 608 | No |
| 32K + 1K | 0.9 | MTP=1 | 8 | 64 | 129P:1D | 896.94 | 996.51 | 10.0% | 1,096 | Yes |
| 32K + 1K | 0.99 | No MTP | 8 | 64 | 26P:1D | 489.76 | 524.53 | 6.6% | 272 | No |
| 32K + 1K | 0.99 | MTP=1 | 8 | 64 | 48P:1D | 904.16 | 996.51 | 9.3% | 448 | Yes |
| 128K + 1K | 0 | No MTP | 8 | 64 | 1335P:1D | 212.30 | 235.75 | 9.9% | 10,744 | No |
| 128K + 1K | 0 | MTP=1 | 8 | 64 | 2535P:1D | 403.13 | 447.88 | 10.0% | 20,344 | Yes |
| 128K + 1K | 0.9 | No MTP | 8 | 64 | 99P:1D | 214.18 | 235.75 | 9.1% | 856 | No |
| 128K + 1K | 0.9 | MTP=1 | 8 | 64 | 187P:1D | 404.57 | 447.88 | 9.7% | 1,560 | Yes |
| 128K + 1K | 0.99 | No MTP | 8 | 64 | 18P:1D | 216.89 | 235.75 | 8.0% | 208 | No |
| 128K + 1K | 0.99 | MTP=1 | 8 | 64 | 34P:1D | 409.68 | 447.88 | 8.5% | 336 | Yes |
| 1M + 1K | 0 | No MTP | 8 | 64 | 5449P:1D | 33.99 | 37.77 | 10.0% | 43,656 | No |
| 1M + 1K | 0 | MTP=1 | 8 | 64 | 10352P:1D | 64.58 | 71.76 | 10.0% | 82,880 | Yes |
| 1M + 1K | 0.9 | No MTP | 8 | 64 | 150P:1D | 34.09 | 37.77 | 9.7% | 1,264 | No |
| 1M + 1K | 0.9 | MTP=1 | 8 | 64 | 285P:1D | 64.77 | 71.76 | 9.7% | 2,344 | Yes |
| 1M + 1K | 0.99 | No MTP | 8 | 64 | 13P:1D | 36.23 | 37.77 | 4.1% | 168 | No |
| 1M + 1K | 0.99 | MTP=1 | 8 | 64 | 24P:1D | 66.89 | 71.76 | 6.8% | 256 | Yes |

从配比结果看，prefix cache 命中率对资源规模的影响大于 Decode 是否使用 MTP。MTP=1 提高 Decode 侧能力后，若仍要求两侧 QPS 配平，Prefill 侧实例数会增加；但对同一 Decode 模式，h=0.99 相比 h=0 可显著降低 Prefill 实例数和总卡数。

## 6. 建议与结论

- Prefill 实例建议：四个场景统一以 8 卡作为最小可行 Prefill 实例，策略 `TP=8, EP=8, DP=1`。当前 bs=1 sizing 下无需为 1M prefill 单独提高实例卡数，但实际部署应为 runtime 开销和并发波动预留 HBM 余量。
- Decode 实例建议：若目标是最大 TPS/card，四个场景均选择 64 卡 Decode 实例。8K 场景在 No MTP 下明显受 TPOT 限制，MTP=1 可把可用 batch/card 从 552 提到 1,091；32K/128K/1M 则更多受 HBM 限制。
- P/D 配比建议：最终容量规划必须按 prefix cache 命中率分档。h=0 是保守上限，h=0.9 是中高缓存复用场景，h=0.99 对长上下文场景会把 Prefill 资源需求降低一个到两个数量级。
- MTP 使用建议：在当前建模下 MTP=1 对所有 Decode 场景均提升 TPS/card；若产品目标是吞吐最大化，可使用 MTP=1 配比表作为主方案，同时保留 No MTP 表作为回退容量口径。

## Appendix A. 数据与复现脚本

本报告依据以下已生成数据和图表：

- `report/0428/data/scenario_spec.json`
- `report/0428/data/prefill_results.json`
- `report/0428/data/decode_results.json`
- `report/0428/data/pd_ratio_results.json`
- `report/0428/data/manifest.json`
- `report/0428/figure/prefill_hbm.svg`
- `report/0428/figure/prefill_tps.svg`
- `report/0428/figure/decode_8k_1k.svg`
- `report/0428/figure/decode_32k_1k.svg`
- `report/0428/figure/decode_128k_1k.svg`
- `report/0428/figure/decode_1m_1k.svg`
- `report/0428/figure/pd_ratio_total_cards.svg`

生成脚本入口为 `report/0428/script/generate_report.py`，共享配置与常量在 `report/0428/script/common.py`。脚本读取的仓库配置路径记录在 `manifest.json` 中：

- `configs/device_910C.json`
- `configs/network_910C.json`
- `configs/model_deepseekv4.json`
- `configs/runtime_deepseekv4.json`

从仓库根目录复现当前数据和图表，可运行：

```bash
python report/0428/script/generate_report.py
```

轻量校验 JSON 有效性可运行：

```bash
python -m json.tool report/0428/data/scenario_spec.json >/dev/null
python -m json.tool report/0428/data/prefill_results.json >/dev/null
python -m json.tool report/0428/data/decode_results.json >/dev/null
python -m json.tool report/0428/data/pd_ratio_results.json >/dev/null
python -m json.tool report/0428/data/manifest.json >/dev/null
```
