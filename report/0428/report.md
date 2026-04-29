# 0428 PD 分离推理分析报告

## 1. Executive Summary（结论摘要）

本报告基于 DeepSeek V4 Flash W8A8 量化模型在 Ascend 910C 上的 roofline 性能模型（`decode_utilization=0.8`），分析四类 PD 分离推理场景（8K/32K/128K/1M + 1K decode）的实例 sizing、Decode 吞吐与 P/D 配比。

核心结论：

- **Prefill**：四个场景最小可行实例均为 8 卡，当前最优策略一致为 `TP=1, EP=8, DP=8`。prefix cache 命中率不改变 HBM sizing，但可使有效计算长度缩短约 100 倍，从而大幅提升 Prefill QPS。

- **Decode**：No MTP 模式在所有场景和所有实例大小（8/16/32/64 卡）下均无法满足 TPOT≤50ms，原因是无 MTP 时单步等效延迟超出 budget。MTP=1 通过将 decode forward 次数从 1,024 降至 539 使 TPOT 满足约束。可行结果：
  - **8K/32K/128K**：MTP=1 在 32 卡和 64 卡均可行，64 卡为最优（`TP=1, EP=64, DP=64`）。
  - **1M**：MTP=1 仅 64 卡可行（`TP=2, EP=64, DP=32`，TPOT=42.1ms，TPS=23.77/card）。

- **P/D 配比**（全部 MTP=1）：四个场景三种 prefix cache 命中率共 12 个可行配比。`h=0` 时 Prefill 是瓶颈，需要大量 Prefill 实例；`h=0.99` 时 Prefill 极快，配比翻转为 Decode-heavy。

MTP=1 各场景 P/D 推荐配比（`prefix_cache_hit_rate = 0 / 0.9 / 0.99`）：

| 场景 | h=0 | h=0.9 | h=0.99 |
| --- | --- | --- | --- |
| 8K + 1K | 15P:1D，184 卡 | 3P:2D，152 卡 | 1P:6D，392 卡 |
| 32K + 1K | 40P:1D，384 卡 | 4P:1D，96 卡 | 2P:5D，336 卡 |
| 128K + 1K | 79P:1D，696 卡 | 6P:1D，112 卡 | 2P:3D，208 卡 |
| 1M + 1K | 212P:1D，1,760 卡 | 6P:1D，112 卡 | 1P:2D，136 卡 |

---

## 2. 分析目标

对四个 PD 分离推理场景进行：

1. **Prefill 实例 sizing**：确定满足 `bs=1` HBM 约束的最小卡数，以及在该卡数内的最大 TPS/card 性能配置。
2. **Decode TPOT 约束分析**：在 8/16/32/64 卡下，分别评估 No MTP 与 MTP=1 的最大单卡 batch 与 TPS/card，并筛选满足 TPOT≤50ms 的可行配置。
3. **P/D 配比求解**：基于选定 Prefill 实例 QPS 与最优 Decode 实例 QPS，按 10% imbalance 容忍度求最接近整数的实例配比与总卡数。

覆盖场景：

| 场景 | Prefill 输入长度 | Decode 输出长度 |
| --- | ---: | ---: |
| 8K + 1K | 8,192 | 1,024 |
| 32K + 1K | 32,768 | 1,024 |
| 128K + 1K | 131,072 | 1,024 |
| 1M + 1K | 1,000,000 | 1,024 |

---

## 3. 方法

### 3.1 Prefill 分析

**Sizing 阶段**：按候选卡数 `[1, 2, 4, 8, 16, 32, 64]` 逐一搜索，固定 `batch_size=1`，找到首个满足"模型权重 + 完整输入 KV cache"可放入可用 HBM（57.6 GB）的最小卡数。

**性能阶段**：在 sizing 所得卡数内，遍历所有有效 `{TP, EP, DP, batch_size}` 组合，选择 Prefill TPS/card 最大的配置。batch_size 通过二分法找到 HBM 上限，再在候选集合中搜索最优。

Prefix cache 建模：`L_miss = ceil(input_len × (1 − prefix_cache_hit_rate))`，Prefill compute 使用 `L_miss`，HBM 仍按完整 input context 计算（命中部分假设已在缓存中）。

### 3.2 Decode 分析

**HBM 上限（`max_batch_per_card_hbm`）**：对每个 `{TP, EP, DP}` 组合，二分搜索满足 HBM 约束的最大单卡 batch。此上限只受内存约束，不考虑延迟。

**TPOT 约束（`max_batch_per_card_tpot`）**：在 HBM 上限内，二分搜索满足 `TPOT = decode_total_time / output_len ≤ 50ms` 的最大单卡 batch。

图中每个可行点的标签为 **`HBM/TPOT`**，分别表示 HBM-only 最大单卡 batch 和 TPOT 约束下的最大单卡 batch（例如 `1320/45` 表示 HBM 上限 1320，TPOT 约束下取 45）。不可行点不绘制。

MTP 建模：`tokens_per_forward = 1 + mtp × mtp_accept_ratio = 1.9`，`decode_forward_count = ceil(1024 / 1.9) = 539`（相比 No MTP 的 1024 步减少约 47%）。

### 3.3 P/D 配比

求整数 `(P, D)` 使得 `P × prefill_qps ≈ D × decode_qps`，|imbalance| ≤ 10%。总卡数 = `P × 8 + D × 64`。

---

## 4. 假设

| 项目 | 取值 |
| --- | --- |
| 硬件 | Ascend 910C |
| 模型 | DeepSeek V4 Flash |
| 量化模式 | W8A8 |
| KV Cache 量化 | KV8 |
| W8A8 GEMM 吞吐 | 752.0 TFLOPS |
| HBM 容量 | 64 GB |
| HBM 预留 | 10.0% |
| HBM 可用 | 57.6 GB |
| `decode_utilization` | **0.8** |
| `prefix_cache_hit_rate` | 0, 0.9, 0.99 |
| MTP accept ratio | 0.9 |
| TPOT 约束 | ≤ 50 ms |
| P/D imbalance 容忍 | ≤ 10% |

`decode_utilization=0.8` 表示 Decode 计算单元平均有效利用率为 80%。相比 0.6 的保守估计，此处每步延迟约缩短 25%，直接带来单卡 batch 容量和 TPS 的显著提升。

---

## 5. 建模限制

- **No MTP 仍无解**：即使 `decode_utilization` 提升至 0.8，No MTP 在所有场景下仍超出 TPOT=50ms 约束。能否部署 No MTP 对 `decode_utilization` 极为敏感，需通过实测校准。
- 未建模 quant/dequant kernel 时间、runtime 调度开销及 allocator fragmentation 以外的额外 HBM 损耗。
- Prefix cache 只降低 Prefill compute，不降低 HBM 占用。高命中率场景的实际 KV cache 占用也可能降低，本模型未体现。
- MTP 只按平均接收 token 数折算 forward 次数，未加入 MTP head 额外权重或专属计算开销。若建模这些开销，可行的 TPOT 区间将进一步收窄。
- 未建模 P/D KV transfer 延迟、排队延迟、动态 batching、拓扑放置约束及故障冗余策略。
- P/D 配比按稳态 QPS 配平，不代表端到端延迟 SLO 保障。

---

## 6. Prefill 结果

四个场景在 `batch_size=1` 下的最小可行实例均为 8 卡。在该 8 卡实例内继续搜索最大 TPS/card，当前最优策略一致为 `TP=1, EP=8, DP=8, SP=True`。性能配置将 batch 填满至 HBM 上限，因此 HBM 占用接近 57.6 GB 上限。

![Prefill HBM](figure/prefill_hbm.svg)

![Prefill TPS](figure/prefill_tps.svg)

| 场景 | Hit | 卡/实例 | 最优策略 | BS | B/card | L_miss | Weight GB | KV GB | HBM GB | Prefill ms | QPS/实例 | TPS/card |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K + 1K | 0 | 8 | TP=1, EP=8, DP=8 | 3,944 | 493.0 | 8,192 | 42.30 | 15.28 | 57.58 | 1,107,818 | 3.560 | 3,645.6 |
| 8K + 1K | 0.9 | 8 | TP=1, EP=8, DP=8 | 3,944 | 493.0 | 820 | 42.30 | 15.28 | 57.58 | 109,180 | 36.124 | 36,990.8 |
| 8K + 1K | 0.99 | 8 | TP=1, EP=8, DP=8 | 3,944 | 493.0 | 82 | 42.30 | 15.28 | 57.58 | 10,920 | 361.156 | 369,824.0 |
| 32K + 1K | 0 | 8 | TP=1, EP=8, DP=8 | 1,056 | 132.0 | 32,768 | 42.30 | 15.25 | 57.55 | 1,267,526 | 0.833 | 3,412.5 |
| 32K + 1K | 0.9 | 8 | TP=1, EP=8, DP=8 | 1,056 | 132.0 | 3,277 | 42.30 | 15.25 | 57.55 | 117,034 | 9.023 | 36,958.4 |
| 32K + 1K | 0.99 | 8 | TP=1, EP=8, DP=8 | 1,056 | 132.0 | 328 | 42.30 | 15.25 | 57.55 | 11,694 | 90.305 | 369,888.1 |
| 128K + 1K | 0 | 8 | TP=1, EP=8, DP=8 | 256 | 32.0 | 131,072 | 42.97 | 14.52 | 57.49 | 1,619,072 | 0.158 | 2,590.6 |
| 128K + 1K | 0.9 | 8 | TP=1, EP=8, DP=8 | 256 | 32.0 | 13,108 | 42.97 | 14.52 | 57.49 | 116,632 | 2.195 | 35,961.8 |
| 128K + 1K | 0.99 | 8 | TP=1, EP=8, DP=8 | 256 | 32.0 | 1,311 | 42.97 | 14.52 | 57.49 | 11,336 | 22.582 | 369,990.2 |
| 1M + 1K | 0 | 8 | TP=1, EP=8, DP=8 | 32 | 4.0 | 1,000,000 | 42.97 | 13.77 | 56.74 | 5,058,232 | 0.00633 | 790.8 |
| 1M + 1K | 0.9 | 8 | TP=1, EP=8, DP=8 | 32 | 4.0 | 100,000 | 42.97 | 13.77 | 56.74 | 141,839 | 0.226 | 28,201.0 |
| 1M + 1K | 0.99 | 8 | TP=1, EP=8, DP=8 | 32 | 4.0 | 10,000 | 42.97 | 13.77 | 56.74 | 11,037 | 2.899 | 362,422.5 |

Prefill 结果与 `decode_utilization` 无关，数值与上一版本相同。主要规律：`bs=1` 只决定最小卡数，性能配置将 batch 放大至 HBM 上限；prefix cache 命中率可使 QPS 提升一到两个数量级。

---

## 7. Decode 结果

表中 `HBM B/card` 为 HBM-only 最大单卡 batch（仅受内存约束，最优并行策略为 `TP=1,EP=N,DP=N`）；`TPOT B/card` 为同时满足 TPOT≤50ms 的最大单卡 batch（最优并行策略下）。"超限"表示在该 GPU count 下任何 batch size 均无法满足 TPOT 约束。

**通用结论**：No MTP 在所有场景和所有 GPU count 下均超限。MTP=1 通过减少 47% 的 forward 次数（1024 → 539）有效降低等效 TPOT，使 32 卡和 64 卡配置成为可行，其中 64 卡始终为最优。

### 7.1 8K + 1K

![Decode 8K + 1K](figure/decode_8k_1k.svg)

8K 上下文 KV cache 小，HBM-only batch 上限高（64 卡 1,320/card）。MTP=1 在 32 卡已可行（TPOT B/card=9），64 卡（TP=1,EP=64,DP=64）的 TPOT B/card 达到 45，TPS=902.25，为最佳实例。

| 模式 | 卡/实例 | HBM B/card | TPOT B/card | TPOT ms | TPS/card | QPS/实例 | 最优策略 | 可行 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| No MTP | 8 | 443 | — | 超限 | — | — | — | No |
| No MTP | 16 | 944 | — | 超限 | — | — | — | No |
| No MTP | 32 | 1,195 | — | 超限 | — | — | — | No |
| No MTP | 64 | 1,320 | — | 超限 | — | — | — | No |
| MTP=1 | 8 | 443 | — | 超限 | — | — | — | No |
| MTP=1 | 16 | 944 | — | 超限 | — | — | — | No |
| MTP=1 | 32 | 1,195 | 9 | 49.92 | 180.30 | 5.63 | TP=2, EP=32, DP=16 | Yes |
| **MTP=1** | **64** | **1,320** | **45** | **49.88** | **902.25** | **56.39** | **TP=1, EP=64, DP=64** | **Yes ★** |

### 7.2 32K + 1K

![Decode 32K + 1K](figure/decode_32k_1k.svg)

32K KV cache 更大，HBM-only batch 上限降至 382/card（64 卡）。MTP=1 在 32 卡可行（B/card=5），64 卡（TP=1,EP=64,DP=64）的 TPOT B/card=29，TPS=581.90，为最佳实例。

| 模式 | 卡/实例 | HBM B/card | TPOT B/card | TPOT ms | TPS/card | QPS/实例 | 最优策略 | 可行 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| No MTP | 8 | 128 | — | 超限 | — | — | — | No |
| No MTP | 16 | 273 | — | 超限 | — | — | — | No |
| No MTP | 32 | 346 | — | 超限 | — | — | — | No |
| No MTP | 64 | 382 | — | 超限 | — | — | — | No |
| MTP=1 | 8 | 128 | — | 超限 | — | — | — | No |
| MTP=1 | 16 | 273 | — | 超限 | — | — | — | No |
| MTP=1 | 32 | 346 | 5 | 49.50 | 101.01 | 3.23 | TP=2, EP=32, DP=16 | Yes |
| **MTP=1** | **64** | **382** | **29** | **49.84** | **581.90** | **36.37** | **TP=1, EP=64, DP=64** | **Yes ★** |

### 7.3 128K + 1K

![Decode 128K + 1K](figure/decode_128k_1k.svg)

128K KV cache 已显著压缩 HBM 余量，64 卡 HBM-only batch 仅 98/card。MTP=1 在 32 卡仅 B/card=1 可行，64 卡（TP=1,EP=64,DP=64）TPOT B/card=11，TPS=221.65，为最佳实例。

| 模式 | 卡/实例 | HBM B/card | TPOT B/card | TPOT ms | TPS/card | QPS/实例 | 最优策略 | 可行 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| No MTP | 8 | 31 | — | 超限 | — | — | — | No |
| No MTP | 16 | 69 | — | 超限 | — | — | — | No |
| No MTP | 32 | 88 | — | 超限 | — | — | — | No |
| No MTP | 64 | 98 | — | 超限 | — | — | — | No |
| MTP=1 | 8 | 31 | — | 超限 | — | — | — | No |
| MTP=1 | 16 | 69 | — | 超限 | — | — | — | No |
| MTP=1 | 32 | 88 | 1 | 47.64 | 20.99 | 0.66 | TP=4, EP=32, DP=8 | Yes |
| **MTP=1** | **64** | **98** | **11** | **49.63** | **221.65** | **13.85** | **TP=1, EP=64, DP=64** | **Yes ★** |

### 7.4 1M + 1K

![Decode 1M + 1K](figure/decode_1m_1k.svg)

1M 上下文 KV cache 极大，64 卡 HBM-only batch 仅 13/card。在 `decode_utilization=0.8` 下，MTP=1 在 64 卡（TP=2,EP=64,DP=32）实现 TPOT B/card=1、TPOT=42.07ms、TPS=23.77，**首次成为可行场景**（0.6 时无解）。

| 模式 | 卡/实例 | HBM B/card | TPOT B/card | TPOT ms | TPS/card | QPS/实例 | 最优策略 | 可行 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| No MTP | 8 | 4 | — | 超限 | — | — | — | No |
| No MTP | 16 | 9 | — | 超限 | — | — | — | No |
| No MTP | 32 | 11 | — | 超限 | — | — | — | No |
| No MTP | 64 | 13 | — | 超限 | — | — | — | No |
| MTP=1 | 8 | 4 | — | 超限 | — | — | — | No |
| MTP=1 | 16 | 9 | — | 超限 | — | — | — | No |
| MTP=1 | 32 | 11 | — | 超限 | — | — | — | No |
| **MTP=1** | **64** | **13** | **1** | **42.07** | **23.77** | **1.49** | **TP=2, EP=64, DP=32** | **Yes ★** |

---

## 8. Prefill/Decode 配比结果

P/D 配比基于第 6 节选定的 8 卡 Prefill 实例，以及第 7 节各场景 MTP=1 的最优 64 卡 Decode 实例（No MTP 全部不可行）。所有 12 个场景×命中率组合均有可行配比（含 1M+1K）。

![P/D Total Cards](figure/pd_ratio_total_cards.svg)

| 场景 | Hit | Decode 模式 | Prefill 卡/实例 | Decode 卡/实例 | P:D | Prefill QPS | Decode QPS | imbalance | 总卡数 | 推荐 |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 8K + 1K | 0 | No MTP | — | — | 不可行 | — | — | — | — | No |
| 8K + 1K | 0 | MTP=1 | 8 | 64 | 15P:1D | 53.40 | 56.39 | 5.3% | 184 | Yes |
| 8K + 1K | 0.9 | No MTP | — | — | 不可行 | — | — | — | — | No |
| 8K + 1K | 0.9 | MTP=1 | 8 | 64 | 3P:2D | 108.37 | 112.78 | 3.9% | 152 | Yes |
| 8K + 1K | 0.99 | No MTP | — | — | 不可行 | — | — | — | — | No |
| 8K + 1K | 0.99 | MTP=1 | 8 | 64 | 1P:6D | 361.16 | 338.34 | 6.3% | 392 | Yes |
| 32K + 1K | 0 | No MTP | — | — | 不可行 | — | — | — | — | No |
| 32K + 1K | 0 | MTP=1 | 8 | 64 | 40P:1D | 33.32 | 36.37 | 8.4% | 384 | Yes |
| 32K + 1K | 0.9 | No MTP | — | — | 不可行 | — | — | — | — | No |
| 32K + 1K | 0.9 | MTP=1 | 8 | 64 | 4P:1D | 36.09 | 36.37 | 0.8% | 96 | Yes |
| 32K + 1K | 0.99 | No MTP | — | — | 不可行 | — | — | — | — | No |
| 32K + 1K | 0.99 | MTP=1 | 8 | 64 | 2P:5D | 180.61 | 181.85 | 0.7% | 336 | Yes |
| 128K + 1K | 0 | No MTP | — | — | 不可行 | — | — | — | — | No |
| 128K + 1K | 0 | MTP=1 | 8 | 64 | 79P:1D | 12.49 | 13.85 | 9.8% | 696 | Yes |
| 128K + 1K | 0.9 | No MTP | — | — | 不可行 | — | — | — | — | No |
| 128K + 1K | 0.9 | MTP=1 | 8 | 64 | 6P:1D | 13.17 | 13.85 | 4.9% | 112 | Yes |
| 128K + 1K | 0.99 | No MTP | — | — | 不可行 | — | — | — | — | No |
| 128K + 1K | 0.99 | MTP=1 | 8 | 64 | 2P:3D | 45.16 | 41.56 | 8.0% | 208 | Yes |
| 1M + 1K | 0 | No MTP | — | — | 不可行 | — | — | — | — | No |
| 1M + 1K | 0 | MTP=1 | 8 | 64 | 212P:1D | 1.341 | 1.486 | 9.7% | 1,760 | Yes |
| 1M + 1K | 0.9 | No MTP | — | — | 不可行 | — | — | — | — | No |
| 1M + 1K | 0.9 | MTP=1 | 8 | 64 | 6P:1D | 1.354 | 1.486 | 8.9% | 112 | Yes |
| 1M + 1K | 0.99 | No MTP | — | — | 不可行 | — | — | — | — | No |
| 1M + 1K | 0.99 | MTP=1 | 8 | 64 | 1P:2D | 2.899 | 2.971 | 2.4% | 136 | Yes |

**关键规律**：

1. **h=0 时 Prefill 是资源瓶颈**：Decode QPS 远高于单个 Prefill 实例的 QPS，需要大量 Prefill 实例（15–212 P per D）。此类场景总卡数由 Prefill 主导。

2. **h=0.9 时趋于平衡**：8K 已翻转为轻度 Decode-heavy（3P:2D），32K/128K/1M 仍 Prefill-heavy 但比例大幅收敛（4:1、6:1、6:1）。

3. **h=0.99 时 Decode 是资源主体**：所有场景均呈 Decode-heavy，总卡数由 Decode 主导（如 8K 1P:6D 总 392 卡，384 卡为 Decode）。

4. **1M+1K h=0 为极端 Prefill-heavy 场景**：212P:1D，总卡数 1,760（1,696 卡为 Prefill）。实际部署需评估 1M 场景的需求量是否足以支撑如此庞大的 Prefill 集群。

5. **imbalance 均≤10%**：所有可行配比均满足约束，接近约束边界的有 128K h=0（9.8%）和 1M h=0（9.7%）。

---

## 9. 结论

### 9.1 主要发现

- **No MTP 仍无解**：`decode_utilization=0.8` 下 No MTP 模式依然超限。与 0.6 相比，TPOT 裕量有所改善，但仍不足以在任何场景满足 50ms 约束。能否支持 No MTP 需要通过实测确认 `decode_utilization` 的实际水平。

- **MTP=1 可行配置大幅扩展**：相比 0.6 时仅 64 卡单一配置可行，0.8 下 8K/32K/128K 的 32 卡实例也已可行，1M 场景首次出现可行解（64 卡 MTP=1，TPS=23.77/card）。

- **TPS/card 显著提升**（0.8 vs 0.6）：8K: 902 vs 343（+163%），32K: 582 vs 204（+185%），128K: 222 vs 63（+252%），1M: 23.77 vs 不可行。

- **最优并行策略**：8K/32K/128K 的最优 Decode 策略从 0.6 的 TP=2 回退至 `TP=1, EP=64, DP=64`；1M 仍为 `TP=2, EP=64, DP=32`（长上下文场景 TP=2 可降低单步计算量，从而满足 TPOT）。

- **P/D 比例更 Prefill-heavy（h=0 时）**：decode 更快 → 每个 Decode 实例 QPS 更高 → 需要更多 Prefill 实例配平（8K: 6P:1D → 15P:1D，32K: 14P:1D → 40P:1D，128K: 23P:1D → 79P:1D）。

### 9.2 部署建议

- **Prefill 实例**：统一 8 卡（TP=1, EP=8, DP=8），覆盖全部上下文长度。实际部署应为并发波动和 runtime 开销预留 HBM 余量。
- **Decode 实例**：优先 64 卡 MTP=1（TP=1, EP=64, DP=64），TPS/card 比 32 卡高 4–11 倍，是首选实例规格。1M 场景需特别部署 TP=2 配置。
- **P/D 容量规划**：按预期 prefix cache 命中率分档。`h=0` 是保守上限，`h=0.9` 是高复用常见场景，`h=0.99` 适用于 system prompt 极长的 agent 应用。1M h=0 场景（212P:1D, 1,760 卡）规模极大，建议按实际需求评估是否需要支持该组合。
- **`decode_utilization` 校准**：当前结论对该参数高度敏感。建议通过真实负载 profiling 校准该值后再做最终容量决策。若实测值高于 0.8，No MTP 可能成为可行选项，P/D 配比将再次发生显著变化。

---

## 附录：数据与复现

本报告基于以下已生成数据和图表（git commit: `44b3c34`，`decode_utilization=0.8`）：

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

生成脚本：

```bash
python report/0428/script/generate_report.py
```

快速 JSON 校验：

```bash
python -m json.tool report/0428/data/prefill_results.json >/dev/null
python -m json.tool report/0428/data/decode_results.json >/dev/null
python -m json.tool report/0428/data/pd_ratio_results.json >/dev/null
```

完整测试：

```bash
python -m unittest test.test_report_0428 test.test_param_search test.test_serving -v
```

依赖配置文件：

- `configs/device_910C.json`（`decode_utilization: 0.8`）
- `configs/network_910C.json`
- `configs/model_deepseekv4.json`
- `configs/runtime_deepseekv4.json`
