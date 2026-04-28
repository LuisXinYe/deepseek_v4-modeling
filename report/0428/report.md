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
| 8K + 1K | 362P:1D，2,960 卡 | 82P:1D，720 卡 | 57P:1D，520 卡 |
| 32K + 1K | 1040P:1D，8,384 卡 | 128P:1D，1,088 卡 | 47P:1D，440 卡 |
| 128K + 1K | 2508P:1D，20,128 卡 | 185P:1D，1,544 卡 | 34P:1D，336 卡 |
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
| 8K + 1K | 0 | 8 | TP=8, EP=8, DP=1 | 1 | 8,192 | 37.27 | 0.03 | 37.30 | 64.8% | 299.82 | 3.34 | 3,415.39 |
| 8K + 1K | 0.9 | 8 | TP=8, EP=8, DP=1 | 1 | 820 | 37.27 | 0.03 | 37.30 | 64.8% | 67.96 | 14.72 | 15,068.16 |
| 8K + 1K | 0.99 | 8 | TP=8, EP=8, DP=1 | 1 | 82 | 37.27 | 0.03 | 37.30 | 64.8% | 46.48 | 21.52 | 22,032.17 |
| 32K + 1K | 0 | 8 | TP=8, EP=8, DP=1 | 1 | 32,768 | 37.27 | 0.12 | 37.39 | 64.9% | 1,174.51 | 0.85 | 3,487.42 |
| 32K + 1K | 0.9 | 8 | TP=8, EP=8, DP=1 | 1 | 3,277 | 37.27 | 0.12 | 37.39 | 64.9% | 143.82 | 6.95 | 28,479.21 |
| 32K + 1K | 0.99 | 8 | TP=8, EP=8, DP=1 | 1 | 328 | 37.27 | 0.12 | 37.39 | 64.9% | 53.09 | 18.84 | 77,155.07 |
| 128K + 1K | 0 | 8 | TP=8, EP=8, DP=1 | 1 | 131,072 | 37.36 | 0.45 | 37.81 | 65.6% | 6,288.32 | 0.16 | 2,605.47 |
| 128K + 1K | 0.9 | 8 | TP=8, EP=8, DP=1 | 1 | 13,108 | 37.36 | 0.45 | 37.81 | 65.6% | 462.23 | 2.16 | 35,445.21 |
| 128K + 1K | 0.99 | 8 | TP=8, EP=8, DP=1 | 1 | 1,311 | 37.36 | 0.45 | 37.81 | 65.6% | 82.99 | 12.05 | 197,414.73 |
| 1M + 1K | 0 | 8 | TP=8, EP=8, DP=1 | 1 | 1,000,000 | 37.36 | 3.44 | 40.80 | 70.8% | 160,291.41 | 0.01 | 779.83 |
| 1M + 1K | 0.9 | 8 | TP=8, EP=8, DP=1 | 1 | 100,000 | 37.36 | 3.44 | 40.80 | 70.8% | 4,399.96 | 0.23 | 28,409.38 |
| 1M + 1K | 0.99 | 8 | TP=8, EP=8, DP=1 | 1 | 10,000 | 37.36 | 3.44 | 40.80 | 70.8% | 358.82 | 2.79 | 348,362.68 |

Prefill 侧的主要现象是：HBM 随完整上下文长度增长，但在 bs=1 下仍由权重占主导；prefix cache 命中率不会改变 HBM sizing，却会强烈影响 Prefill QPS。例如 1M 场景从 h=0 到 h=0.99，Prefill latency 从 160.29s 降到 358.82ms，P/D 配比也随之大幅收敛。

## 4. Decode 实例 sizing、TPOT 约束与吞吐

Decode 表中的 `HBM B/card` 表示仅考虑 HBM 时可达到的最大单卡 batch；`TPOT B/card` 表示同时满足 `TPOT <= 50ms` 时的最大单卡 batch。图中每个点的标签采用 `HBM/TPOT` 简写，例如 `1320/542` 表示该点 HBM-only 上限为 1320 batch/card，而 TPOT=50ms 约束下只能取到 542 batch/card；图内也给出了该标签含义。

当前所有 Decode 表项均可行，因此没有 N/A 项。四个场景的最佳 Decode 实例均为 64 卡；No MTP 与 MTP=1 的最优策略均为 `TP=1, EP=64, DP=64`。

### 4.1 8K + 1K

![Decode 8K + 1K](figure/decode_8k_1k.svg)

8K decode 的 HBM-only batch 余量较高，No MTP 主要被 TPOT 卡住；MTP=1 后 8/16 卡实例可直接达到 HBM 上限，32/64 卡实例仍受 TPOT 约束。

| 模式 | 卡/实例 | HBM B/card | TPOT B/card | TPOT ms | TPS/card | QPS/实例 | 最优策略 | 最佳实例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| No MTP | 8 | 443 | 298 | 49.97 | 5,963.30 | 46.59 | TP=1, EP=8, DP=8 | No |
| No MTP | 16 | 944 | 438 | 49.96 | 8,767.77 | 137.00 | TP=1, EP=16, DP=16 | No |
| No MTP | 32 | 1,195 | 507 | 49.92 | 10,155.31 | 317.35 | TP=1, EP=32, DP=32 | No |
| No MTP | 64 | 1,320 | 542 | 49.97 | 10,847.19 | 677.95 | TP=1, EP=64, DP=64 | Yes |
| MTP=1 | 8 | 443 | 443 | 32.71 | 13,543.90 | 105.81 | TP=1, EP=8, DP=8 | No |
| MTP=1 | 16 | 944 | 944 | 48.80 | 19,343.24 | 302.24 | TP=1, EP=16, DP=16 | No |
| MTP=1 | 32 | 1,195 | 1,038 | 49.98 | 20,767.94 | 649.00 | TP=1, EP=32, DP=32 | No |
| MTP=1 | 64 | 1,320 | 1,072 | 50.00 | 21,440.12 | 1,340.01 | TP=1, EP=64, DP=64 | Yes |

### 4.2 32K + 1K

![Decode 32K + 1K](figure/decode_32k_1k.svg)

32K decode 已基本由 HBM 决定 batch 上限；No MTP 与 MTP=1 在所有实例大小上都可以使用 HBM-only 最大 batch，MTP=1 主要体现为 TPS/card 提升。

| 模式 | 卡/实例 | HBM B/card | TPOT B/card | TPOT ms | TPS/card | QPS/实例 | 最优策略 | 最佳实例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| No MTP | 8 | 128 | 128 | 39.52 | 3,238.57 | 25.30 | TP=1, EP=8, DP=8 | No |
| No MTP | 16 | 273 | 273 | 42.98 | 6,351.56 | 99.24 | TP=1, EP=16, DP=16 | No |
| No MTP | 32 | 346 | 346 | 45.10 | 7,671.35 | 239.73 | TP=1, EP=32, DP=32 | No |
| No MTP | 64 | 382 | 382 | 46.13 | 8,281.81 | 517.61 | TP=1, EP=64, DP=64 | Yes |
| MTP=1 | 8 | 128 | 128 | 20.80 | 6,152.68 | 48.07 | TP=1, EP=8, DP=8 | No |
| MTP=1 | 16 | 273 | 273 | 22.62 | 12,066.79 | 188.54 | TP=1, EP=16, DP=16 | No |
| MTP=1 | 32 | 346 | 346 | 23.74 | 14,574.13 | 455.44 | TP=1, EP=32, DP=32 | No |
| MTP=1 | 64 | 382 | 382 | 24.28 | 15,733.90 | 983.37 | TP=1, EP=64, DP=64 | Yes |

### 4.3 128K + 1K

![Decode 128K + 1K](figure/decode_128k_1k.svg)

128K decode 的 KV cache 显著压低 batch 上限，所有实例大小的 TPOT-constrained batch 都等于 HBM-only batch。MTP=1 继续提升 TPS/card，但资源上限仍由 HBM 主导。

| 模式 | 卡/实例 | HBM B/card | TPOT B/card | TPOT ms | TPS/card | QPS/实例 | 最优策略 | 最佳实例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| No MTP | 8 | 31 | 31 | 34.89 | 888.62 | 6.94 | TP=1, EP=8, DP=8 | No |
| No MTP | 16 | 69 | 69 | 29.88 | 2,308.94 | 36.08 | TP=1, EP=16, DP=16 | No |
| No MTP | 32 | 88 | 88 | 27.41 | 3,210.88 | 100.34 | TP=1, EP=32, DP=32 | No |
| No MTP | 64 | 98 | 98 | 26.27 | 3,731.13 | 233.20 | TP=1, EP=64, DP=64 | Yes |
| MTP=1 | 8 | 31 | 31 | 18.36 | 1,688.21 | 13.19 | TP=1, EP=8, DP=8 | No |
| MTP=1 | 16 | 69 | 69 | 15.73 | 4,386.56 | 68.54 | TP=1, EP=16, DP=16 | No |
| MTP=1 | 32 | 88 | 88 | 14.43 | 6,100.07 | 190.63 | TP=1, EP=32, DP=32 | No |
| MTP=1 | 64 | 98 | 98 | 13.83 | 7,088.45 | 443.03 | TP=1, EP=64, DP=64 | Yes |

### 4.4 1M + 1K

![Decode 1M + 1K](figure/decode_1m_1k.svg)

1M decode 是最典型的 HBM-bound 场景：单卡 batch 上限只有 4/9/11/13。由于 batch 很小，所有点都满足 TPOT=50ms，MTP=1 主要通过减少 forward 次数提高吞吐。

| 模式 | 卡/实例 | HBM B/card | TPOT B/card | TPOT ms | TPS/card | QPS/实例 | 最优策略 | 最佳实例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| No MTP | 8 | 4 | 4 | 33.28 | 120.20 | 0.94 | TP=1, EP=8, DP=8 | No |
| No MTP | 16 | 9 | 9 | 26.41 | 340.84 | 5.33 | TP=1, EP=16, DP=16 | No |
| No MTP | 32 | 11 | 11 | 22.46 | 489.84 | 15.31 | TP=1, EP=32, DP=32 | No |
| No MTP | 64 | 13 | 13 | 21.51 | 604.31 | 37.77 | TP=1, EP=64, DP=64 | Yes |
| MTP=1 | 8 | 4 | 4 | 17.52 | 228.36 | 1.78 | TP=1, EP=8, DP=8 | No |
| MTP=1 | 16 | 9 | 9 | 13.90 | 647.53 | 10.12 | TP=1, EP=16, DP=16 | No |
| MTP=1 | 32 | 11 | 11 | 11.82 | 930.61 | 29.08 | TP=1, EP=32, DP=32 | No |
| MTP=1 | 64 | 13 | 13 | 11.32 | 1,148.07 | 71.75 | TP=1, EP=64, DP=64 | Yes |

## 5. Prefill/Decode 配比

P/D 配比基于第 3 节选出的 Prefill 8 卡实例，以及第 4 节每个场景和 Decode 模式下 TPS/card 最大的 64 卡 Decode 实例。表中 `Prefill QPS` 与 `Decode QPS` 是配比后的聚合 QPS，imbalance 为两侧 QPS 差异比例；所有条目均满足当前 10% imbalance 容忍度。

![P/D Total Cards](figure/pd_ratio_total_cards.svg)

| 场景 | Hit | Decode 模式 | Prefill 卡/实例 | Decode 卡/实例 | P:D | Prefill QPS | Decode QPS | imbalance | 总卡数 | 推荐 |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 8K + 1K | 0 | No MTP | 8 | 64 | 183P:1D | 610.37 | 677.95 | 10.0% | 1,528 | No |
| 8K + 1K | 0 | MTP=1 | 8 | 64 | 362P:1D | 1,207.40 | 1,340.01 | 9.9% | 2,960 | Yes |
| 8K + 1K | 0.9 | No MTP | 8 | 64 | 42P:1D | 618.03 | 677.95 | 8.8% | 400 | No |
| 8K + 1K | 0.9 | MTP=1 | 8 | 64 | 82P:1D | 1,206.63 | 1,340.01 | 10.0% | 720 | Yes |
| 8K + 1K | 0.99 | No MTP | 8 | 64 | 29P:1D | 623.96 | 677.95 | 8.0% | 296 | No |
| 8K + 1K | 0.99 | MTP=1 | 8 | 64 | 57P:1D | 1,226.40 | 1,340.01 | 8.5% | 520 | Yes |
| 32K + 1K | 0 | No MTP | 8 | 64 | 548P:1D | 466.58 | 517.61 | 9.9% | 4,448 | No |
| 32K + 1K | 0 | MTP=1 | 8 | 64 | 1040P:1D | 885.48 | 983.37 | 10.0% | 8,384 | Yes |
| 32K + 1K | 0.9 | No MTP | 8 | 64 | 68P:1D | 472.80 | 517.61 | 8.7% | 608 | No |
| 32K + 1K | 0.9 | MTP=1 | 8 | 64 | 128P:1D | 889.98 | 983.37 | 9.5% | 1,088 | Yes |
| 32K + 1K | 0.99 | No MTP | 8 | 64 | 25P:1D | 470.92 | 517.61 | 9.0% | 264 | No |
| 32K + 1K | 0.99 | MTP=1 | 8 | 64 | 47P:1D | 885.32 | 983.37 | 10.0% | 440 | Yes |
| 128K + 1K | 0 | No MTP | 8 | 64 | 1320P:1D | 209.91 | 233.20 | 10.0% | 10,624 | No |
| 128K + 1K | 0 | MTP=1 | 8 | 64 | 2508P:1D | 398.83 | 443.03 | 10.0% | 20,128 | Yes |
| 128K + 1K | 0.9 | No MTP | 8 | 64 | 98P:1D | 212.01 | 233.20 | 9.1% | 848 | No |
| 128K + 1K | 0.9 | MTP=1 | 8 | 64 | 185P:1D | 400.23 | 443.03 | 9.7% | 1,544 | Yes |
| 128K + 1K | 0.99 | No MTP | 8 | 64 | 18P:1D | 216.89 | 233.20 | 7.0% | 208 | No |
| 128K + 1K | 0.99 | MTP=1 | 8 | 64 | 34P:1D | 409.67 | 443.03 | 7.5% | 336 | Yes |
| 1M + 1K | 0 | No MTP | 8 | 64 | 5449P:1D | 33.99 | 37.77 | 10.0% | 43,656 | No |
| 1M + 1K | 0 | MTP=1 | 8 | 64 | 10352P:1D | 64.58 | 71.75 | 10.0% | 82,880 | Yes |
| 1M + 1K | 0.9 | No MTP | 8 | 64 | 150P:1D | 34.09 | 37.77 | 9.7% | 1,264 | No |
| 1M + 1K | 0.9 | MTP=1 | 8 | 64 | 285P:1D | 64.77 | 71.75 | 9.7% | 2,344 | Yes |
| 1M + 1K | 0.99 | No MTP | 8 | 64 | 13P:1D | 36.23 | 37.77 | 4.1% | 168 | No |
| 1M + 1K | 0.99 | MTP=1 | 8 | 64 | 24P:1D | 66.89 | 71.75 | 6.8% | 256 | Yes |

从配比结果看，prefix cache 命中率对资源规模的影响大于 Decode 是否使用 MTP。MTP=1 提高 Decode 侧能力后，若仍要求两侧 QPS 配平，Prefill 侧实例数会增加；但对同一 Decode 模式，h=0.99 相比 h=0 可显著降低 Prefill 实例数和总卡数。

## 6. 建议与结论

- Prefill 实例建议：四个场景统一以 8 卡作为最小可行 Prefill 实例，策略 `TP=8, EP=8, DP=1`。当前 bs=1 sizing 下无需为 1M prefill 单独提高实例卡数，但实际部署应为 runtime 开销和并发波动预留 HBM 余量。
- Decode 实例建议：若目标是最大 TPS/card，四个场景均选择 64 卡 Decode 实例。8K 场景在 No MTP 下明显受 TPOT 限制，MTP=1 可把可用 batch/card 从 542 提到 1,072；32K/128K/1M 则更多受 HBM 限制。
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
