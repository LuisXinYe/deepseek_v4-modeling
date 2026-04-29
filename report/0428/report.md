# 0428 PD 分离推理分析报告

## Executive Summary

本报告基于 `report/0428/data/*.json` 中的当前结果，分析 DeepSeek V4 Flash W8A8 量化模型在 Ascend 910C 上的四类 PD 分离推理场景：Prefill 8K/32K/128K/1M + Decode 1K。核心结论如下：

- Prefill：四个场景在 `batch_size = 1` 下，满足“模型权重 + bs=1 KV cache” HBM 放置的最小实例均为 8 卡；在该 8 卡实例内继续搜索最大 TPS/card 后，当前最优并行策略均为 `TP=1, EP=8, DP=8`。随着 prefix cache 命中率提升，有效计算长度显著下降，Prefill QPS 显著提升。
- Decode：在 8/16/32/64 卡实例中，四个场景的 No MTP 与 MTP=1 最优 Decode 实例均为 64 卡，最优策略均为 `TP=1, EP=64, DP=64`。MTP=1 在本模型中通过减少 decode forward 次数提升 TPS/card，但不改变 HBM-only 最大 batch。
- P/D 配比：按“选定 Prefill 实例 + 最优 Decode 实例”求 `Prefill QPS = Decode QPS`，MTP=1 的 Decode QPS 更高，因此需要更多 Prefill 实例与其配平；prefix cache 命中率越高，所需 Prefill 实例越少。
- 推荐侧重点：若以 Decode TPS/card 最大化为目标，四个场景均选择 64 卡 Decode + MTP=1；若关注总卡规模，prefix cache 命中率是 P/D 资源需求的主导变量，尤其在 128K 与 1M 场景。

MTP=1 下的 P/D 推荐配比如下，覆盖 `prefix_cache_hit_rate = 0 / 0.9 / 0.99` 三种情况：

| 场景 | h=0 | h=0.9 | h=0.99 |
| --- | --- | --- | --- |
| 8K + 1K | 100P:1D，864 卡 | 10P:1D，144 卡 | 1P:1D，72 卡 |
| 32K + 1K | 345P:1D，2,824 卡 | 29P:1D，296 卡 | 3P:1D，88 卡 |
| 128K + 1K | 1019P:1D，8,216 卡 | 57P:1D，520 卡 | 6P:1D，112 卡 |
| 1M + 1K | 5460P:1D，43,744 卡 | 113P:1D，968 卡 | 7P:1D，120 卡 |

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

Prefill sizing 按候选卡数 `[1, 2, 4, 8, 16, 32, 64]` 搜索，固定 `batch_size=1`，筛选“模型权重 + 完整输入 KV cache”可放入可用 HBM 的最小实例；确定最小实例卡数后，在该卡数内搜索所有有效 `{TP, EP, DP, batch_size}`，并选择 Prefill TPS/card 最大的性能配置。

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

Prefill 的最小可行实例由 `bs=1` HBM 约束决定。当前四个场景都在 8 卡实例首次满足 HBM 放置；性能表则是在 8 卡内继续搜索 TPS/card 最大的 `{TP, EP, DP, batch_size}`，当前最优策略一致为 `TP=1, EP=8, DP=8`。该性能配置会把 HBM 尽量用于更大的 prefill batch，因此表中 HBM 占用接近可用容量上限。

![Prefill HBM](figure/prefill_hbm.svg)

![Prefill TPS](figure/prefill_tps.svg)

| 场景 | Hit | 最小卡/实例 | 最优策略 | BS | B/card | L_miss | Weight GB | KV GB | HBM GB | HBM 占用 | Prefill ms | QPS/实例 | TPS/card |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K + 1K | 0 | 8 | TP=1, EP=8, DP=8 | 3,944 | 493.00 | 8,192 | 42.30 | 15.28 | 57.58 | 100.0% | 347,835.74 | 11.34 | 11,610.81 |
| 8K + 1K | 0.9 | 8 | TP=1, EP=8, DP=8 | 3,944 | 493.00 | 820 | 42.30 | 15.28 | 57.58 | 100.0% | 33,454.72 | 117.89 | 120,720.06 |
| 8K + 1K | 0.99 | 8 | TP=1, EP=8, DP=8 | 3,944 | 493.00 | 82 | 42.30 | 15.28 | 57.58 | 100.0% | 3,340.76 | 1,180.57 | 1,208,901.87 |
| 32K + 1K | 0 | 8 | TP=1, EP=8, DP=8 | 1,056 | 132.00 | 32,768 | 42.30 | 15.25 | 57.55 | 99.9% | 433,363.10 | 2.44 | 9,980.95 |
| 32K + 1K | 0.9 | 8 | TP=1, EP=8, DP=8 | 1,056 | 132.00 | 3,277 | 42.30 | 15.25 | 57.55 | 99.9% | 36,283.05 | 29.10 | 119,212.03 |
| 32K + 1K | 0.99 | 8 | TP=1, EP=8, DP=8 | 1,056 | 132.00 | 328 | 42.30 | 15.25 | 57.55 | 99.9% | 3,578.04 | 295.13 | 1,208,866.55 |
| 128K + 1K | 0 | 8 | TP=1, EP=8, DP=8 | 256 | 32.00 | 131,072 | 42.97 | 14.52 | 57.49 | 99.8% | 667,270.12 | 0.38 | 6,285.77 |
| 128K + 1K | 0.9 | 8 | TP=1, EP=8, DP=8 | 256 | 32.00 | 13,108 | 42.97 | 14.52 | 57.49 | 99.8% | 37,082.40 | 6.90 | 113,107.67 |
| 128K + 1K | 0.99 | 8 | TP=1, EP=8, DP=8 | 256 | 32.00 | 1,311 | 42.97 | 14.52 | 57.49 | 99.8% | 3,481.84 | 73.52 | 1,204,622.51 |
| 1M + 1K | 0 | 8 | TP=1, EP=8, DP=8 | 32 | 4.00 | 1,000,000 | 42.97 | 13.77 | 56.74 | 98.5% | 2,718,791.30 | 0.01 | 1,471.24 |
| 1M + 1K | 0.9 | 8 | TP=1, EP=8, DP=8 | 32 | 4.00 | 100,000 | 42.97 | 13.77 | 56.74 | 98.5% | 56,188.05 | 0.57 | 71,189.52 |
| 1M + 1K | 0.99 | 8 | TP=1, EP=8, DP=8 | 32 | 4.00 | 10,000 | 42.97 | 13.77 | 56.74 | 98.5% | 3,478.95 | 9.20 | 1,149,773.81 |

Prefill 侧的主要现象是：`bs=1` 只决定最小卡数，性能配置会使用更大的 batch 来摊薄权重与固定开销；prefix cache 命中率不会改变 HBM sizing，却会强烈影响 Prefill QPS。例如 1M 场景从 h=0 到 h=0.99，Prefill QPS 从 0.01 req/s 提升到 9.20 req/s，P/D 配比也随之大幅收敛。

## 4. Decode 实例 sizing、TPOT 约束与吞吐

Decode 表中的 `HBM B/card` 表示仅考虑 HBM 时可达到的最大单卡 batch；`TPOT B/card` 表示同时满足 `TPOT <= 50ms` 时的最大单卡 batch。图中每个点的标签采用 `HBM/TPOT` 简写，例如 `1320/506` 表示该点 HBM-only 上限为 1320 batch/card，而 TPOT=50ms 约束下只能取到 506 batch/card；图内也给出了该标签含义。

当前所有 Decode 表项均可行，因此没有 N/A 项。四个场景的最佳 Decode 实例均为 64 卡；No MTP 与 MTP=1 的最优策略均为 `TP=1, EP=64, DP=64`。

### 4.1 8K + 1K

![Decode 8K + 1K](figure/decode_8k_1k.svg)

8K decode 的 HBM-only batch 余量较高，No MTP 主要被 TPOT 卡住；MTP=1 后 8 卡实例可直接达到 HBM 上限，16/32/64 卡实例仍受 TPOT 约束。

| 模式 | 卡/实例 | HBM B/card | TPOT B/card | TPOT ms | TPS/card | QPS/实例 | 最优策略 | 最佳实例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| No MTP | 8 | 443 | 278 | 49.94 | 5,567.11 | 43.49 | TP=1, EP=8, DP=8 | No |
| No MTP | 16 | 944 | 409 | 49.95 | 8,188.64 | 127.95 | TP=1, EP=16, DP=16 | No |
| No MTP | 32 | 1,195 | 474 | 49.97 | 9,486.49 | 296.45 | TP=1, EP=32, DP=32 | No |
| No MTP | 64 | 1,320 | 506 | 49.94 | 10,131.35 | 633.21 | TP=1, EP=64, DP=64 | Yes |
| MTP=1 | 8 | 443 | 443 | 34.10 | 12,989.91 | 101.48 | TP=1, EP=8, DP=8 | No |
| MTP=1 | 16 | 944 | 906 | 50.00 | 18,121.02 | 283.14 | TP=1, EP=16, DP=16 | No |
| MTP=1 | 32 | 1,195 | 969 | 49.99 | 19,384.43 | 605.76 | TP=1, EP=32, DP=32 | No |
| MTP=1 | 64 | 1,320 | 1,000 | 49.97 | 20,012.94 | 1,250.81 | TP=1, EP=64, DP=64 | Yes |

### 4.2 32K + 1K

![Decode 32K + 1K](figure/decode_32k_1k.svg)

32K decode 已基本由 HBM 决定 batch 上限；No MTP 与 MTP=1 在所有实例大小上都可以使用 HBM-only 最大 batch，MTP=1 主要体现为 TPS/card 提升。

| 模式 | 卡/实例 | HBM B/card | TPOT B/card | TPOT ms | TPS/card | QPS/实例 | 最优策略 | 最佳实例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| No MTP | 8 | 128 | 128 | 40.30 | 3,176.27 | 24.81 | TP=1, EP=8, DP=8 | No |
| No MTP | 16 | 273 | 273 | 44.73 | 6,103.35 | 95.36 | TP=1, EP=16, DP=16 | No |
| No MTP | 32 | 346 | 346 | 47.33 | 7,309.78 | 228.43 | TP=1, EP=32, DP=32 | No |
| No MTP | 64 | 382 | 382 | 48.59 | 7,861.00 | 491.31 | TP=1, EP=64, DP=64 | Yes |
| MTP=1 | 8 | 128 | 128 | 21.21 | 6,034.32 | 47.14 | TP=1, EP=8, DP=8 | No |
| MTP=1 | 16 | 273 | 273 | 23.54 | 11,595.24 | 181.18 | TP=1, EP=16, DP=16 | No |
| MTP=1 | 32 | 346 | 346 | 24.91 | 13,887.22 | 433.98 | TP=1, EP=32, DP=32 | No |
| MTP=1 | 64 | 382 | 382 | 25.58 | 14,934.45 | 933.40 | TP=1, EP=64, DP=64 | Yes |

### 4.3 128K + 1K

![Decode 128K + 1K](figure/decode_128k_1k.svg)

128K decode 的 KV cache 显著压低 batch 上限，所有实例大小的 TPOT-constrained batch 都等于 HBM-only batch。MTP=1 继续提升 TPS/card，但资源上限仍由 HBM 主导。

| 模式 | 卡/实例 | HBM B/card | TPOT B/card | TPOT ms | TPS/card | QPS/实例 | 最优策略 | 最佳实例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| No MTP | 8 | 31 | 31 | 35.05 | 884.33 | 6.91 | TP=1, EP=8, DP=8 | No |
| No MTP | 16 | 69 | 69 | 30.26 | 2,280.46 | 35.63 | TP=1, EP=16, DP=16 | No |
| No MTP | 32 | 88 | 88 | 27.88 | 3,156.13 | 98.63 | TP=1, EP=32, DP=32 | No |
| No MTP | 64 | 98 | 98 | 26.79 | 3,657.44 | 228.59 | TP=1, EP=64, DP=64 | Yes |
| MTP=1 | 8 | 31 | 31 | 18.45 | 1,680.07 | 13.13 | TP=1, EP=8, DP=8 | No |
| MTP=1 | 16 | 69 | 69 | 15.93 | 4,332.46 | 67.69 | TP=1, EP=16, DP=16 | No |
| MTP=1 | 32 | 88 | 88 | 14.68 | 5,996.06 | 187.38 | TP=1, EP=32, DP=32 | No |
| MTP=1 | 64 | 98 | 98 | 14.10 | 6,948.45 | 434.28 | TP=1, EP=64, DP=64 | Yes |

### 4.4 1M + 1K

![Decode 1M + 1K](figure/decode_1m_1k.svg)

1M decode 是最典型的 HBM-bound 场景：单卡 batch 上限只有 4/9/11/13。由于 batch 很小，所有点都满足 TPOT=50ms，MTP=1 主要通过减少 forward 次数提高吞吐。

| 模式 | 卡/实例 | HBM B/card | TPOT B/card | TPOT ms | TPS/card | QPS/实例 | 最优策略 | 最佳实例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| No MTP | 8 | 4 | 4 | 33.31 | 120.08 | 0.94 | TP=1, EP=8, DP=8 | No |
| No MTP | 16 | 9 | 9 | 26.48 | 339.88 | 5.31 | TP=1, EP=16, DP=16 | No |
| No MTP | 32 | 11 | 11 | 22.55 | 487.87 | 15.25 | TP=1, EP=32, DP=32 | No |
| No MTP | 64 | 13 | 13 | 21.62 | 601.31 | 37.58 | TP=1, EP=64, DP=64 | Yes |
| MTP=1 | 8 | 4 | 4 | 17.53 | 228.12 | 1.78 | TP=1, EP=8, DP=8 | No |
| MTP=1 | 16 | 9 | 9 | 13.94 | 645.71 | 10.09 | TP=1, EP=16, DP=16 | No |
| MTP=1 | 32 | 11 | 11 | 11.87 | 926.86 | 28.96 | TP=1, EP=32, DP=32 | No |
| MTP=1 | 64 | 13 | 13 | 11.38 | 1,142.38 | 71.40 | TP=1, EP=64, DP=64 | Yes |

## 5. Prefill/Decode 配比

P/D 配比基于第 3 节选出的 Prefill 8 卡实例，以及第 4 节每个场景和 Decode 模式下 TPS/card 最大的 64 卡 Decode 实例。表中 `Prefill QPS` 与 `Decode QPS` 是配比后的聚合 QPS，imbalance 为两侧 QPS 差异比例；所有条目均满足当前 10% imbalance 容忍度。

![P/D Total Cards](figure/pd_ratio_total_cards.svg)

| 场景 | Hit | Decode 模式 | Prefill 卡/实例 | Decode 卡/实例 | P:D | Prefill QPS | Decode QPS | imbalance | 总卡数 | 推荐 |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 8K + 1K | 0 | No MTP | 8 | 64 | 51P:1D | 578.27 | 633.21 | 8.7% | 472 | No |
| 8K + 1K | 0 | MTP=1 | 8 | 64 | 100P:1D | 1,133.87 | 1,250.81 | 9.3% | 864 | Yes |
| 8K + 1K | 0.9 | No MTP | 8 | 64 | 5P:1D | 589.45 | 633.21 | 6.9% | 104 | No |
| 8K + 1K | 0.9 | MTP=1 | 8 | 64 | 10P:1D | 1,178.91 | 1,250.81 | 5.7% | 144 | Yes |
| 8K + 1K | 0.99 | No MTP | 8 | 64 | 1P:2D | 1,180.57 | 1,266.42 | 6.8% | 136 | No |
| 8K + 1K | 0.99 | MTP=1 | 8 | 64 | 1P:1D | 1,180.57 | 1,250.81 | 5.6% | 72 | Yes |
| 32K + 1K | 0 | No MTP | 8 | 64 | 182P:1D | 443.49 | 491.31 | 9.7% | 1,520 | No |
| 32K + 1K | 0 | MTP=1 | 8 | 64 | 345P:1D | 840.68 | 933.40 | 9.9% | 2,824 | Yes |
| 32K + 1K | 0.9 | No MTP | 8 | 64 | 16P:1D | 465.67 | 491.31 | 5.2% | 192 | No |
| 32K + 1K | 0.9 | MTP=1 | 8 | 64 | 29P:1D | 844.03 | 933.40 | 9.6% | 296 | Yes |
| 32K + 1K | 0.99 | No MTP | 8 | 64 | 3P:2D | 885.40 | 982.63 | 9.9% | 152 | No |
| 32K + 1K | 0.99 | MTP=1 | 8 | 64 | 3P:1D | 885.40 | 933.40 | 5.1% | 88 | Yes |
| 128K + 1K | 0 | No MTP | 8 | 64 | 537P:1D | 206.02 | 228.59 | 9.9% | 4,360 | No |
| 128K + 1K | 0 | MTP=1 | 8 | 64 | 1019P:1D | 390.94 | 434.28 | 10.0% | 8,216 | Yes |
| 128K + 1K | 0.9 | No MTP | 8 | 64 | 30P:1D | 207.11 | 228.59 | 9.4% | 304 | No |
| 128K + 1K | 0.9 | MTP=1 | 8 | 64 | 57P:1D | 393.50 | 434.28 | 9.4% | 520 | Yes |
| 128K + 1K | 0.99 | No MTP | 8 | 64 | 3P:1D | 220.57 | 228.59 | 3.5% | 88 | No |
| 128K + 1K | 0.99 | MTP=1 | 8 | 64 | 6P:1D | 441.15 | 434.28 | 1.6% | 112 | Yes |
| 1M + 1K | 0 | No MTP | 8 | 64 | 2874P:1D | 33.83 | 37.58 | 10.0% | 23,056 | No |
| 1M + 1K | 0 | MTP=1 | 8 | 64 | 5460P:1D | 64.26 | 71.40 | 10.0% | 43,744 | Yes |
| 1M + 1K | 0.9 | No MTP | 8 | 64 | 60P:1D | 34.17 | 37.58 | 9.1% | 544 | No |
| 1M + 1K | 0.9 | MTP=1 | 8 | 64 | 113P:1D | 64.36 | 71.40 | 9.9% | 968 | Yes |
| 1M + 1K | 0.99 | No MTP | 8 | 64 | 4P:1D | 36.79 | 37.58 | 2.1% | 96 | No |
| 1M + 1K | 0.99 | MTP=1 | 8 | 64 | 7P:1D | 64.39 | 71.40 | 9.8% | 120 | Yes |

从配比结果看，prefix cache 命中率对资源规模的影响大于 Decode 是否使用 MTP。MTP=1 提高 Decode 侧能力后，若仍要求两侧 QPS 配平，Prefill 侧实例数会增加；但对同一 Decode 模式，h=0.99 相比 h=0 可显著降低 Prefill 实例数和总卡数。

## 6. 建议与结论

- Prefill 实例建议：四个场景统一以 8 卡作为最小可行 Prefill 实例；性能配置按最大 TPS/card 选择 `TP=1, EP=8, DP=8`。当前 `bs=1` sizing 下无需为 1M prefill 单独提高实例卡数，但实际部署应为 runtime 开销和并发波动预留 HBM 余量。
- Decode 实例建议：若目标是最大 TPS/card，四个场景均选择 64 卡 Decode 实例。8K 场景在 No MTP 下明显受 TPOT 限制，MTP=1 可把可用 batch/card 从 506 提到 1,000；32K/128K/1M 则更多受 HBM 限制。
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
