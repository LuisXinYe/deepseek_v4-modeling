# 0428 PD 分离推理分析报告

## 1. Executive Summary（结论摘要）

本报告基于 DeepSeek V4 Flash W8A8 量化模型在 Ascend 910C 上的 roofline 性能模型（`decode_utilization=0.9`），分析四类 PD 分离推理场景（8K/32K/128K/1M + 1K decode）的实例 sizing、Decode 吞吐与 P/D 配比。

核心结论：

- **Prefill**：四个场景最小可行实例均为 8 卡，当前最优策略一致为 `TP=1, EP=8, DP=8`。prefix cache 命中率不改变 HBM sizing，但可使有效计算长度缩短约 100 倍，从而大幅提升 Prefill QPS。

- **Decode**：No MTP 在 8K 和 32K 的 64 卡配置下恰好满足 TPOT≤50ms（TPOT 分别为 49.21ms 和 49.16ms），但单卡 batch 极低（分别为 2 和 1），吞吐有限。对 128K 和 1M，No MTP 在所有测试卡数（8/16/32/64）下均超出 TPOT 约束。MTP=1 通过将 decode forward 次数从 1,024 降至 539（减少约 47%）使所有场景在 64 卡下均可行，最优策略统一为 `TP=1, EP=64, DP=64`。

- **P/D 配比**：`h=0` 时 Prefill 是资源瓶颈，需大量 Prefill 实例；`h=0.99` 时 Prefill 极快，配比翻转为 Decode-heavy。128K h=0.99 出现极均衡的 1P:1D（72 卡）配置。

MTP=1 各场景 P/D 推荐配比（`prefix_cache_hit_rate = 0 / 0.9 / 0.99`）：

| 场景 | h=0 | h=0.9 | h=0.99 |
| --- | --- | --- | --- |
| 8K + 1K | 22P:1D，240 卡 | 5P:2D，168 卡 | 1P:4D，264 卡 |
| 32K + 1K | 61P:1D，552 卡 | 6P:1D，112 卡 | 2P:3D，208 卡 |
| 128K + 1K | 129P:1D，1,096 卡 | 10P:1D，144 卡 | **1P:1D，72 卡** |
| 1M + 1K | 535P:1D，4,344 卡 | 15P:1D，184 卡 | 4P:3D，224 卡 |

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

图中每个可行点的标签为 **`HBM/TPOT`**，分别表示 HBM-only 最大单卡 batch 和 TPOT 约束下的最大单卡 batch（例如 `1320/69` 表示 HBM 上限 1320，TPOT 约束下取 69）。不可行点不绘制。

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
| `decode_utilization` | 0.9 |
| `prefix_cache_hit_rate` | 0, 0.9, 0.99 |
| MTP accept ratio | 0.9 |
| TPOT 约束 | ≤ 50 ms |
| P/D imbalance 容忍 | ≤ 10% |

---

## 5. 建模限制

- **No MTP 仅对短上下文（8K/32K）在 64 卡下勉强可行**：TPOT 贴近 50ms 边界，单卡 batch 极低，吞吐有限。能否稳定部署对 `decode_utilization` 实际水平极为敏感，需通过实测校准。
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

主要规律：`bs=1` 只决定最小卡数，性能配置将 batch 放大至 HBM 上限；prefix cache 命中率可使 QPS 提升一到两个数量级，但不影响 sizing 所需卡数。

---

## 7. Decode 结果

表中 `HBM B/card` 为 HBM-only 最大单卡 batch（仅受内存约束）；`TPOT B/card` 为同时满足 TPOT≤50ms 的最大单卡 batch（最优并行策略下）。"超限"表示在该 GPU count 下任何 batch size 均无法满足 TPOT 约束。

**通用结论**：No MTP 仅在 8K 和 32K 的 64 卡下恰好可行，且吞吐极低（TPOT B/card 分别为 2 和 1）。MTP=1 通过减少 47% 的 forward 次数（1024 → 539）大幅降低等效 TPOT，64 卡为所有场景的最优实例，最优并行策略统一为 `TP=1, EP=64, DP=64`。

### 7.1 8K + 1K

![Decode 8K + 1K](figure/decode_8k_1k.svg)

8K 上下文 KV cache 小，HBM-only batch 上限高（64 卡 1,320/card）。No MTP 在 64 卡（TP=4,EP=64,DP=16）以 TPOT B/card=2 满足约束，但 TPS=40.64，吞吐极低。MTP=1 在 32 卡（TPOT B/card=22）和 64 卡（TPOT B/card=69）均可行，64 卡 TPS=1,383.5，为最佳实例。

| 模式 | 卡/实例 | HBM B/card | TPOT B/card | TPOT ms | TPS/card | QPS/实例 | 最优策略 | 可行 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| No MTP | 8 | 443 | — | 超限 | — | — | — | No |
| No MTP | 16 | 944 | — | 超限 | — | — | — | No |
| No MTP | 32 | 1,195 | — | 超限 | — | — | — | No |
| No MTP | 64 | 1,320 | 2 | 49.21 | 40.64 | 2.54 | TP=4, EP=64, DP=16 | Yes |
| MTP=1 | 8 | 443 | — | 超限 | — | — | — | No |
| MTP=1 | 16 | 944 | — | 超限 | — | — | — | No |
| MTP=1 | 32 | 1,195 | 22 | 49.79 | 441.8 | 13.81 | TP=2, EP=32, DP=16 | Yes |
| **MTP=1** | **64** | **1,320** | **69** | **49.87** | **1,383.5** | **86.47** | **TP=1, EP=64, DP=64** | **Yes ★** |

### 7.2 32K + 1K

![Decode 32K + 1K](figure/decode_32k_1k.svg)

32K KV cache 更大，HBM-only batch 上限降至 382/card（64 卡）。No MTP 在 64 卡（TP=4,EP=64,DP=16）以 TPOT B/card=1 勉强可行，TPS=20.34，QPS=1.27。MTP=1 在 32 卡（B/card=13）和 64 卡（B/card=45）均可行，64 卡 TPS=900.3，为最佳实例。

| 模式 | 卡/实例 | HBM B/card | TPOT B/card | TPOT ms | TPS/card | QPS/实例 | 最优策略 | 可行 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| No MTP | 8 | 128 | — | 超限 | — | — | — | No |
| No MTP | 16 | 273 | — | 超限 | — | — | — | No |
| No MTP | 32 | 346 | — | 超限 | — | — | — | No |
| No MTP | 64 | 382 | 1 | 49.16 | 20.34 | 1.27 | TP=4, EP=64, DP=16 | Yes |
| MTP=1 | 8 | 128 | — | 超限 | — | — | — | No |
| MTP=1 | 16 | 273 | — | 超限 | — | — | — | No |
| MTP=1 | 32 | 346 | 13 | 49.37 | 263.3 | 8.23 | TP=2, EP=32, DP=16 | Yes |
| **MTP=1** | **64** | **382** | **45** | **49.98** | **900.3** | **56.27** | **TP=1, EP=64, DP=64** | **Yes ★** |

### 7.3 128K + 1K

![Decode 128K + 1K](figure/decode_128k_1k.svg)

128K KV cache 显著压缩 HBM 余量，64 卡 HBM-only batch 仅 98/card。No MTP 在所有卡数下均超限。MTP=1 在 32 卡（B/card=5）和 64 卡（B/card=18）均可行，64 卡 TPS=360.2，为最佳实例。

| 模式 | 卡/实例 | HBM B/card | TPOT B/card | TPOT ms | TPS/card | QPS/实例 | 最优策略 | 可行 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| No MTP | 8 | 31 | — | 超限 | — | — | — | No |
| No MTP | 16 | 69 | — | 超限 | — | — | — | No |
| No MTP | 32 | 88 | — | 超限 | — | — | — | No |
| No MTP | 64 | 98 | — | 超限 | — | — | — | No |
| MTP=1 | 8 | 31 | — | 超限 | — | — | — | No |
| MTP=1 | 16 | 69 | — | 超限 | — | — | — | No |
| MTP=1 | 32 | 88 | 5 | 49.73 | 100.55 | 3.14 | TP=2, EP=32, DP=16 | Yes |
| **MTP=1** | **64** | **98** | **18** | **49.97** | **360.2** | **22.51** | **TP=1, EP=64, DP=64** | **Yes ★** |

### 7.4 1M + 1K

![Decode 1M + 1K](figure/decode_1m_1k.svg)

1M 上下文 KV cache 极大，64 卡 HBM-only batch 仅 13/card。No MTP 在所有卡数下均超限。MTP=1 在 64 卡（TP=1,EP=64,DP=64）实现 TPOT B/card=3、TPOT=49.94ms、TPS=60.07。

| 模式 | 卡/实例 | HBM B/card | TPOT B/card | TPOT ms | TPS/card | QPS/实例 | 最优策略 | 可行 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| No MTP | 8 | 4 | — | 超限 | — | — | — | No |
| No MTP | 16 | 9 | — | 超限 | — | — | — | No |
| No MTP | 32 | 11 | — | 超限 | — | — | — | No |
| No MTP | 64 | 13 | — | 超限 | — | — | — | No |
| MTP=1 | 8 | 4 | — | 超限 | — | — | — | No |
| MTP=1 | 16 | 9 | — | 超限 | — | — | — | No |
| MTP=1 | 32 | 11 | — | 超限 | — | — | — | No |
| **MTP=1** | **64** | **13** | **3** | **49.94** | **60.07** | **3.75** | **TP=1, EP=64, DP=64** | **Yes ★** |

---

## 8. Prefill/Decode 配比结果

P/D 配比基于第 6 节选定的 8 卡 Prefill 实例，以及第 7 节各场景最优 Decode 实例。8K 和 32K 提供 No MTP 和 MTP=1 两种可行配比，128K 和 1M 仅 MTP=1 可行。

![P/D Total Cards](figure/pd_ratio_total_cards.svg)

| 场景 | Hit | Decode 模式 | Prefill 卡/实例 | Decode 卡/实例 | P:D | Prefill QPS | Decode QPS | imbalance | 总卡数 | 推荐 |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 8K + 1K | 0 | No MTP | 8 | 64 | 2P:3D | 7.12 | 7.62 | 6.6% | 208 | No |
| **8K + 1K** | **0** | **MTP=1** | **8** | **64** | **22P:1D** | **78.32** | **86.47** | **9.4%** | **240** | **Yes ★** |
| 8K + 1K | 0.9 | No MTP | 8 | 64 | 1P:13D | 36.12 | 33.02 | 8.6% | 840 | No |
| **8K + 1K** | **0.9** | **MTP=1** | **8** | **64** | **5P:2D** | **180.62** | **172.94** | **4.3%** | **168** | **Yes ★** |
| 8K + 1K | 0.99 | No MTP | 8 | 64 | 1P:128D | 361.16 | 325.12 | 10.0% | 8,200 | No |
| **8K + 1K** | **0.99** | **MTP=1** | **8** | **64** | **1P:4D** | **361.16** | **345.88** | **4.2%** | **264** | **Yes ★** |
| 32K + 1K | 0 | No MTP | 8 | 64 | 3P:2D | 2.50 | 2.54 | 1.7% | 152 | No |
| **32K + 1K** | **0** | **MTP=1** | **8** | **64** | **61P:1D** | **50.82** | **56.27** | **9.7%** | **552** | **Yes ★** |
| 32K + 1K | 0.9 | No MTP | 8 | 64 | 1P:7D | 9.02 | 8.90 | 1.4% | 456 | No |
| **32K + 1K** | **0.9** | **MTP=1** | **8** | **64** | **6P:1D** | **54.14** | **56.27** | **3.8%** | **112** | **Yes ★** |
| 32K + 1K | 0.99 | No MTP | 8 | 64 | 1P:64D | 90.30 | 81.36 | 9.9% | 4,104 | No |
| **32K + 1K** | **0.99** | **MTP=1** | **8** | **64** | **2P:3D** | **180.61** | **168.81** | **6.5%** | **208** | **Yes ★** |
| 128K + 1K | 0 | No MTP | — | — | 不可行 | — | — | — | — | No |
| **128K + 1K** | **0** | **MTP=1** | **8** | **64** | **129P:1D** | **20.40** | **22.51** | **9.4%** | **1,096** | **Yes ★** |
| 128K + 1K | 0.9 | No MTP | — | — | 不可行 | — | — | — | — | No |
| **128K + 1K** | **0.9** | **MTP=1** | **8** | **64** | **10P:1D** | **21.95** | **22.51** | **2.5%** | **144** | **Yes ★** |
| 128K + 1K | 0.99 | No MTP | — | — | 不可行 | — | — | — | — | No |
| **128K + 1K** | **0.99** | **MTP=1** | **8** | **64** | **1P:1D** | **22.58** | **22.51** | **0.3%** | **72** | **Yes ★** |
| 1M + 1K | 0 | No MTP | — | — | 不可行 | — | — | — | — | No |
| **1M + 1K** | **0** | **MTP=1** | **8** | **64** | **535P:1D** | **3.385** | **3.754** | **9.8%** | **4,344** | **Yes ★** |
| 1M + 1K | 0.9 | No MTP | — | — | 不可行 | — | — | — | — | No |
| **1M + 1K** | **0.9** | **MTP=1** | **8** | **64** | **15P:1D** | **3.384** | **3.754** | **9.9%** | **184** | **Yes ★** |
| 1M + 1K | 0.99 | No MTP | — | — | 不可行 | — | — | — | — | No |
| **1M + 1K** | **0.99** | **MTP=1** | **8** | **64** | **4P:3D** | **11.60** | **11.26** | **2.9%** | **224** | **Yes ★** |

**关键规律**：

1. **h=0 时 Prefill 是资源瓶颈**：MTP=1 Decode 实例 QPS 远高于单个 8 卡 Prefill 实例，需大量 Prefill 实例配平（22–535 P per D），总卡数由 Prefill 主导。

2. **h=0.9 时趋于平衡**：8K 翻转为 Decode-heavy（5P:2D），其余场景仍 Prefill-heavy 但比例大幅收敛（6:1 至 15:1）。

3. **h=0.99 时 Decode 是资源主体**：128K h=0.99 出现极均衡的 **1P:1D（72 卡，imbalance=0.3%）**，是本分析中效率最高的配置。

4. **No MTP（8K/32K）吞吐远低于 MTP=1**：h=0 时 No MTP 总卡数更少（如 32K 152 vs 552 卡），但单卡 QPS 极低，不适合高吞吐场景。h≥0.9 时 No MTP 需要大量 Decode 实例（如 8K h=0.99 需 128 个 64 卡实例共 8,200 卡），不具备部署优势。

5. **imbalance 均≤10%**：所有可行配比均满足约束，接近约束边界的有 1M h=0（9.8%）、1M h=0.9（9.9%）和 8K MTP=1 h=0（9.4%）。

---

## 9. 结论

### 9.1 主要发现

- **MTP=1 是所有场景的可行路径**：所有四个场景在 64 卡（TP=1, EP=64, DP=64）下均可满足 TPOT≤50ms，QPS 从 1M 的 3.75 到 8K 的 86.47 覆盖全范围。

- **No MTP 仅对 8K 和 32K 在 64 卡下恰好可行**，TPOT 贴近 50ms 上限，TPOT B/card 极低（2 和 1），不建议作为主要部署模式。对 128K 和 1M，No MTP 在任何卡数下均无法满足约束。

- **prefix cache 对 P/D 配比影响极大**：h=0 到 h=0.99，Prefill QPS 提升约 100 倍，配比从深度 Prefill-heavy 翻转为 Decode-heavy，总卡数需求分布随之改变。

- **128K h=0.99 出现极优配置**：MTP=1 的 1P:1D（72 卡，imbalance=0.3%），适合以 agent 应用为主的 system prompt 高复用场景。

- **1M h=0 为极端 Prefill-heavy 场景**：535P:1D，总 4,344 卡，规模极大，实际部署需评估需求量。

### 9.2 部署建议

- **Prefill 实例**：统一 8 卡（TP=1, EP=8, DP=8），覆盖全部上下文长度。实际部署应为并发波动和 runtime 开销预留 HBM 余量。
- **Decode 实例**：优先 64 卡 MTP=1（TP=1, EP=64, DP=64），TPS/card 比 32 卡高 3–4 倍，是首选实例规格。
- **P/D 容量规划**：按预期 prefix cache 命中率分档。`h=0` 是保守上限，`h=0.9` 是高复用常见场景，`h=0.99` 适用于 system prompt 极长的 agent 应用。
- **`decode_utilization` 校准**：No MTP 的可行性在当前值附近对该参数极为敏感，TPOT 已贴近 50ms 边界。建议通过真实负载 profiling 校准后再做最终容量决策。

---

## 附录：数据与复现

本报告基于以下已生成数据和图表（git commit: `af218de`，`decode_utilization=0.9`）：

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

- `configs/device_910C.json`（`decode_utilization: 0.9`）
- `configs/network_910C.json`
- `configs/model_deepseekv4.json`
- `configs/runtime_deepseekv4.json`
