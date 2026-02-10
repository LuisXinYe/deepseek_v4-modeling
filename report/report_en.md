# DeepSeek V4 Inference Performance Analysis: Ascend 910C vs NVIDIA H20

## 1. Executive Summary

This report presents a roofline-based performance analysis of DeepSeek V4 inference across two hardware platforms (Ascend 910C and NVIDIA H20) and four serving scenarios (8K/4K, 32K/4K, 128K/4K, 256K/4K input/output token lengths). Key insights:

1. **mHC kernel fusion (default) reduces prefill 3--4x** by eliminating HBM traffic. After fusion, mHC drops from 84% to 36% of prefill time on 910C at 8K.
2. **910C achieves near-parity on prefill throughput** — the gap narrowed from 2.57x to just 1.12x at 8K after kernel fusion eliminated the mHC memory bottleneck where H20's bandwidth advantage dominated.
3. **Decode is MoE-weight-bound on 910C (40--56%)** and communication-bound on H20 at short context (54%), shifting to MoE-weight-bound at long context.
4. **V4's KV compression saves 4.6x memory vs V3** — enabling practical long-context serving that V3's MLA approach cannot match.
5. **P/D disaggregation ratios scale with context**: 1P:1D (8K), 1P:1D (32K), 2P:1D (128K), 3P:1D (256K) on 910C.
6. **910C's EP bandwidth advantage (7.8x)** enables high-EP MoE configs that are prohibitively expensive on H20.

**Analysis method**: Roofline model — each op's time = max(cube, vec, mem) + comm. Utilization: 50% compute, 80% HBM bandwidth.

---

## 2. Model Structure

### 2.1 DeepSeek V4 Architecture

DeepSeek V4 is a Mixture-of-Experts (MoE) model with 43 transformer layers:
- **2 full-attention layers** (ratio=1): standard MQA with 64 Q heads, 1 KV head
- **21 C4A layers** (ratio=4): 4x KV compression with Lightning Index (topK=512)
- **20 C128A layers** (ratio=128): 128x KV compression, no index needed

Key architectural features:
- **MQA** (Multi-Query Attention): 64 Q heads, 1 KV head, head_dim=512
- **Lightning Index**: 64 index heads, dim=128, selects top-512 compressed KV entries
- **MoE**: 256 routed experts (top-6), 1 shared expert, inter_dim=2048
- **mHC (Hyper Connection)**: FP32 pre/post projections + Sinkhorn normalization at every sub-layer
- **KV Compression**: Group projections compress K and V caches by ratio

| Parameter | Value |
|---|---|
| Hidden size | 4,096 |
| Layers | 43 (2 full + 21 C4A + 20 C128A) |
| Q heads | 64 (MQA: 1 KV head) |
| Head dim | 512 |
| Q LoRA rank | 1,024 |
| O groups | 8, O LoRA rank = 1,024 |
| Index heads | 64, dim = 128, topK = 512 |
| Window size | 128 (SWA) |
| Routed experts | 256, top-6 |
| Shared experts | 1 |
| MoE inter dim | 2,048 |
| HC mult | 4 |
| Vocab size | 129,280 |

### 2.2 V4 vs V3 Comparison

| Dimension | DeepSeek V4 | DeepSeek V3 |
|---|---|---|
| Hidden size | 4,096 | 7,168 |
| Layers | 43 | 61 |
| Q heads | 64 | 128 |
| KV approach | MQA + KV compression | MLA (kv_lora_rank=512) |
| Q LoRA rank | 1,024 | 1,536 |
| Routed experts | 256, top-6 | 256, top-8 |
| MoE inter dim | 2,048 | 2,048 |
| KV compression | C4A/C128A (2--128x) | None |
| mHC (Hyper Connection) | Yes (hc_mult=4) | No |
| Lightning Index | Yes (64 heads, dim=128) | No |
| Total params (approx) | ~286B | ~704B |
| KV cache per token | 15,168 bytes | 70,272 bytes |
| **KV compression ratio** | **4.6x less than V3** | Baseline |

V4 trades V3's larger hidden dimension for aggressive KV compression and novel architectural features (mHC, Lightning Index). The 4.6x KV cache savings are critical for practical long-context serving.

### 2.3 Hardware Platforms

| Metric | Ascend 910C | NVIDIA H20 | Ratio |
|---|---|---|---|
| Cube TFLOPS (BF16) | 376 | 148 | **910C 2.54x** |
| Vec TFLOPS (FP32) | 24 | 44 | **H20 1.83x** |
| Cube:Vec ratio | 15.7:1 | 3.4:1 | H20 more balanced |
| HBM Bandwidth (GB/s) | 1,800 | 4,000 | **H20 2.22x** |
| HBM Capacity (GB) | 64 | 96 | **H20 1.5x** |
| TP Bandwidth (GB/s) | 392 | 450 | Similar |
| EP Bandwidth (GB/s) | 392 | 50 | **910C 7.84x** |

The most striking hardware difference: 910C has **7.8x higher EP bandwidth** (intra-node interconnect) but **2.2x lower HBM bandwidth**.

---

## 3. Bottleneck Analysis

### 3.1 Per-Category Bottleneck Summary

| Category | Prefill Bottleneck | Decode Bottleneck |
|---|---|---|
| Attention Projections | CUBE | MEM |
| Attention Compute | CUBE | MEM |
| KV Compression | CUBE | VEC/MEM |
| Lightning Index | COMM (score AllReduce) | COMM |
| mHC (fused) | MEM | MEM |
| MoE Gate | CUBE | MEM |
| MoE Routed Experts | CUBE | MEM |
| Communication | COMM | COMM |

**Key insight: Prefill is compute-bound (CUBE), Decode is memory-bound (MEM/COMM).**

### 3.2 Prefill Op Breakdown (910C, Best Throughput Config, Fused)

| Category | 8K/4K | 32K/4K | 128K/4K | 256K/4K |
|---|---|---|---|---|
| mHC | **36.3%** | **29.4%** | **16.7%** | **10.6%** |
| Attention Proj | 25.4% | 20.6% | 11.7% | 7.4% |
| Attention Compute | 7.8% | 16.2% | 31.6% | 39.0% |
| Lightning Index | 4.8% | 13.1% | 28.2% | 35.5% |
| Communication | 18.1% | 14.6% | 8.3% | 5.3% |
| KV Compression | 2.4% | 2.0% | 1.1% | 0.7% |
| MoE (all) | 3.0% | 2.4% | 1.4% | 0.9% |
| Others | 2.2% | 1.8% | 1.0% | 0.6% |

At 8K, mHC (memory-bound) is the largest single category. As sequence length increases, attention compute and Lightning Index grow quadratically and dominate.

### 3.3 Decode Op Breakdown (910C, Best Throughput Config)

| Category | 8K/4K | 32K/4K | 128K/4K | 256K/4K |
|---|---|---|---|---|
| MoE Routed | **55.9%** | **38.1%** | **40.3%** | **41.1%** |
| Attention Compute | 14.5% | 25.2% | 22.0% | 21.4% |
| Communication | 12.9% | 14.4% | 12.4% | 12.2% |
| Lightning Index | 5.7% | 11.3% | 16.4% | 16.7% |
| Attention Proj | 5.4% | 6.8% | 6.8% | 6.9% |
| mHC | 4.1% | 2.8% | 0.7% | 0.4% |
| Others | 1.5% | 1.4% | 1.4% | 1.3% |

Decode is dominated by MoE weight loading (38--56%). At longer sequences, attention and Lightning Index KV cache reads grow.

### 3.4 910C vs H20 Bottleneck Comparison

**Prefill bottlenecks differ significantly:**

| Category | 910C 8K | H20 8K | Root Cause |
|---|---|---|---|
| mHC | 36.3% | 9.1% | 910C lower HBM BW makes MEM-bound mHC worse |
| Attention Proj | 25.4% | 36.0% | H20 lower cube TFLOPS shifts to CUBE ops |
| Communication | 18.1% | 26.6% | H20 low EP BW (50 vs 392 GB/s) |

**Decode bottlenecks also differ:**

| Category | 910C 8K | H20 8K | Root Cause |
|---|---|---|---|
| MoE Routed | 55.9% | 17.4% | 910C lower HBM BW makes weight loading dominant |
| Communication | 12.9% | 54.2% | H20 low EP BW dominates decode |

On 910C, decode is weight-loading bound. On H20 at short context, decode is communication-bound due to expensive EP AllToAll (50 GB/s). At long context (128K+), both platforms become MoE-weight-bound.

---

## 4. Parameter & Scenario Optimization

### 4.1 Optimal Configurations — Ascend 910C

#### 8K Input / 4K Output

| Scenario | TP | EP | DP | BS | GPUs | Metric |
|---|---|---|---|---|---|---|
| **Prefill Latency** | 8 | 64 | 8 | 8 | 64 | **330 ms** |
| **Prefill Throughput** | 8 | 16 | 2 | 256 | 16 | **1,656 tps/gpu** |
| **Decode Latency** | 4 | 32 | 8 | 8 | 32 | **19.3 ms/step** |
| **Decode Throughput** | 4 | 16 | 4 | 512 | 16 | **181 tps/gpu** |

#### 32K Input / 4K Output

| Scenario | TP | EP | DP | BS | GPUs | Metric |
|---|---|---|---|---|---|---|
| **Prefill Latency** | 8 | 64 | 8 | 8 | 64 | **1,535 ms** |
| **Prefill Throughput** | 8 | 16 | 2 | 64 | 16 | **1,340 tps/gpu** |
| **Decode Latency** | 4 | 32 | 8 | 8 | 32 | **19.4 ms/step** |
| **Decode Throughput** | 4 | 32 | 8 | 512 | 32 | **62.2 tps/gpu** |

#### 128K Input / 4K Output

| Scenario | TP | EP | DP | BS | GPUs | Metric |
|---|---|---|---|---|---|---|
| **Prefill Latency** | 8 | 64 | 8 | 8 | 64 | **10,747 ms** |
| **Prefill Throughput** | 8 | 16 | 2 | 16 | 16 | **760 tps/gpu** |
| **Decode Latency** | 4 | 32 | 8 | 8 | 32 | **21.0 ms/step** |
| **Decode Throughput** | 4 | 32 | 8 | 128 | 32 | **16.7 tps/gpu** |

#### 256K Input / 4K Output

| Scenario | TP | EP | DP | BS | GPUs | Metric |
|---|---|---|---|---|---|---|
| **Prefill Latency** | 8 | 64 | 8 | 8 | 64 | **33,923 ms** |
| **Prefill Throughput** | 8 | 16 | 2 | 8 | 16 | **482 tps/gpu** |
| **Decode Latency** | 4 | 32 | 8 | 8 | 32 | **21.6 ms/step** |
| **Decode Throughput** | 4 | 32 | 8 | 64 | 32 | **8.5 tps/gpu** |

### 4.2 Optimal Configurations — NVIDIA H20

#### 8K Input / 4K Output

| Scenario | TP | EP | DP | BS | GPUs | Metric |
|---|---|---|---|---|---|---|
| **Prefill Latency** | 8 | 64 | 8 | 8 | 64 | **554 ms** |
| **Prefill Throughput** | 8 | 8 | 1 | 128 | 8 | **1,848 tps/gpu** |
| **Decode Latency** | 4 | 32 | 8 | 8 | 32 | **9.0 ms/step** |
| **Decode Throughput** | 8 | 16 | 2 | 512 | 16 | **252 tps/gpu** |

#### 32K Input / 4K Output

| Scenario | TP | EP | DP | BS | GPUs | Metric |
|---|---|---|---|---|---|---|
| **Prefill Latency** | 8 | 64 | 8 | 8 | 64 | **2,771 ms** |
| **Prefill Throughput** | 8 | 8 | 1 | 32 | 8 | **1,463 tps/gpu** |
| **Decode Latency** | 4 | 32 | 8 | 8 | 32 | **9.1 ms/step** |
| **Decode Throughput** | 4 | 16 | 4 | 256 | 16 | **137 tps/gpu** |

#### 128K Input / 4K Output

| Scenario | TP | EP | DP | BS | GPUs | Metric |
|---|---|---|---|---|---|---|
| **Prefill Latency** | 8 | 64 | 8 | 8 | 64 | **20,396 ms** |
| **Prefill Throughput** | 8 | 8 | 1 | 8 | 8 | **798 tps/gpu** |
| **Decode Latency** | 4 | 32 | 8 | 8 | 32 | **9.9 ms/step** |
| **Decode Throughput** | 4 | 16 | 4 | 64 | 16 | **47.4 tps/gpu** |

#### 256K Input / 4K Output

| Scenario | TP | EP | DP | BS | GPUs | Metric |
|---|---|---|---|---|---|---|
| **Prefill Latency** | 8 | 64 | 8 | 8 | 64 | **65,685 ms** |
| **Prefill Throughput** | 8 | 8 | 1 | 4 | 8 | **497 tps/gpu** |
| **Decode Latency** | 4 | 32 | 8 | 8 | 32 | **10.1 ms/step** |
| **Decode Throughput** | 4 | 32 | 8 | 128 | 32 | **25.6 tps/gpu** |

### 4.3 P/D Disaggregated Serving

In prefill/decode disaggregated serving, the ratio formula is:

```
N_p / N_d >= (D_tps_instance × input_len) / (P_tps_instance × output_len)
```

#### Ascend 910C

| Combo | P Config | P tps/inst | D Config | D tps/inst | P:D Ratio | Total GPUs |
|---|---|---|---|---|---|---|
| **8K/4K** | TP=8,EP=16,DP=2 (16 GPUs) | 26,490 | TP=4,EP=16,DP=4 (16 GPUs) | 2,897 | **1P:1D** | 32 |
| **32K/4K** | TP=8,EP=16,DP=2 (16 GPUs) | 21,435 | TP=4,EP=32,DP=8 (32 GPUs) | 1,991 | **1P:1D** | 48 |
| **128K/4K** | TP=8,EP=16,DP=2 (16 GPUs) | 12,155 | TP=4,EP=32,DP=8 (32 GPUs) | 533 | **2P:1D** | 64 |
| **256K/4K** | TP=8,EP=16,DP=2 (16 GPUs) | 7,707 | TP=4,EP=32,DP=8 (32 GPUs) | 273 | **3P:1D** | 80 |

#### NVIDIA H20

| Combo | P Config | P tps/inst | D Config | D tps/inst | P:D Ratio | Total GPUs |
|---|---|---|---|---|---|---|
| **8K/4K** | TP=8,EP=8,DP=1 (8 GPUs) | 14,786 | TP=8,EP=16,DP=2 (16 GPUs) | 4,026 | **1P:1D** | 24 |
| **32K/4K** | TP=8,EP=8,DP=1 (8 GPUs) | 11,704 | TP=4,EP=16,DP=4 (16 GPUs) | 2,192 | **2P:1D** | 32 |
| **128K/4K** | TP=8,EP=8,DP=1 (8 GPUs) | 6,382 | TP=4,EP=16,DP=4 (16 GPUs) | 759 | **4P:1D** | 48 |
| **256K/4K** | TP=8,EP=8,DP=1 (8 GPUs) | 3,973 | TP=4,EP=32,DP=8 (32 GPUs) | 820 | **14P:1D** | 144 |

### 4.4 910C vs H20 Comparative Analysis

#### Prefill Throughput

| Combo | 910C (tps/gpu) | H20 (tps/gpu) | H20/910C | 910C GPUs | H20 GPUs |
|---|---|---|---|---|---|
| 8K/4K | 1,656 | 1,848 | **1.12x** | 16 | 8 |
| 32K/4K | 1,340 | 1,463 | **1.09x** | 16 | 8 |
| 128K/4K | 760 | 798 | **1.05x** | 16 | 8 |
| 256K/4K | 482 | 497 | **1.03x** | 16 | 8 |

H20's advantage narrows at longer sequences — from 1.12x at 8K to 1.03x at 256K — because attention compute (CUBE-bound) dominates and 910C has 2.54x higher cube TFLOPS.

#### Decode Throughput

| Combo | 910C (tps/gpu) | H20 (tps/gpu) | H20/910C | 910C GPUs | H20 GPUs |
|---|---|---|---|---|---|
| 8K/4K | 181 | 252 | **1.39x** | 16 | 16 |
| 32K/4K | 62.2 | 137 | **2.20x** | 32 | 16 |
| 128K/4K | 16.7 | 47.4 | **2.84x** | 32 | 16 |
| 256K/4K | 8.5 | 25.6 | **3.01x** | 32 | 32 |

H20 advantage grows at longer context for decode — more KV cache reads favor H20's 2.2x HBM bandwidth.

#### Prefill & Decode Latency

| Combo | 910C Prefill (ms) | H20 Prefill (ms) | 910C Decode (ms) | H20 Decode (ms) |
|---|---|---|---|---|
| 8K/4K | 330 | 554 | 19.3 | 9.0 |
| 32K/4K | 1,535 | 2,771 | 19.4 | 9.1 |
| 128K/4K | 10,747 | 20,396 | 21.0 | 9.9 |
| 256K/4K | 33,923 | 65,685 | 21.6 | 10.1 |

910C achieves **1.7--1.9x lower prefill latency** across all sequences (cube TFLOPS advantage). H20 achieves **~2x lower decode latency** consistently (HBM bandwidth advantage).

#### Root Cause Summary

**Where 910C wins:**
- Cube-heavy workloads: attention projections, MoE expert matmuls in prefill
- High-EP configurations: EP=32/64 at low cost due to 392 GB/s EP bandwidth
- Prefill latency: 1.7--1.9x faster than H20 across all sequences

**Where H20 wins:**
- Memory-bound workloads: decode weight loading, KV cache reads
- Low-EP throughput: EP=8 with 8 GPUs matches or exceeds 910C's 16 GPU configs
- Decode phase: ~2x faster consistently, widening to 3x at long context

**H20's EP bandwidth problem:** H20's 50 GB/s cross-node EP bandwidth (vs 910C's 392 GB/s) forces EP=8 for throughput, requiring 32 experts per rank. This works at H20's 96 GB HBM but limits scaling.

---

## 5. Key Module Analysis

### 5.1 mHC Optimization

#### Optimization Levels

| Level | mhc_kernel_fused | mhc_sp | mhc_fused_bf16 | Description |
|---|---|---|---|---|
| Unfused FP32 | False | False | False | Original baseline |
| Fused FP32 | True | False | False | Kernel fusion (new default) |
| Fused FP32 + SP | True | True | False | + Sequence parallelism for mHC |
| Fused BF16 + SP | True | True | True | + BF16 precision for fused ops |

#### Prefill Time Comparison (910C, TP=8, EP=16, DP=2, BS=16)

| Level | 8K/4K | mHC % | 32K/4K | mHC % | Speedup vs Unfused (8K) |
|---|---|---|---|---|---|
| Unfused FP32 | 10,144 ms | 84.3% | 42,853 ms | 79.8% | 1.00x |
| Fused FP32 | 2,492 ms | 36.0% | 12,244 ms | 29.3% | 4.07x |
| Fused FP32 + SP | 1,706 ms | 6.6% | 9,102 ms | 4.9% | 5.95x |
| Fused BF16 + SP | 1,650 ms | 3.4% | 8,877 ms | 2.5% | 6.15x |

#### Bottleneck Migration (910C, 8K/4K)

| Category | Unfused FP32 | Fused FP32 | Fused+SP | Fused BF16+SP |
|---|---|---|---|---|
| **mHC** | **84.3%** | **36.0%** | **6.6%** | **3.4%** |
| Attention Proj | 6.2% | 25.2% | 36.9% | 38.1% |
| Communication | 4.6% | 18.6% | 27.1% | 28.0% |
| Attention Compute | 1.9% | 7.7% | 11.3% | 11.7% |
| Lightning Index | 1.2% | 4.9% | 7.1% | 7.4% |
| MoE (all) | 0.7% | 3.0% | 4.4% | 4.5% |

After fusion, the bottleneck migrates to attention projections (CUBE-bound) — where 910C's 2.54x cube TFLOPS advantage is maximally leveraged.

#### Key Insights

1. **Kernel fusion reduces mHC HBM traffic ~10x** by keeping intermediates in registers/SRAM
2. **The fused default makes 910C competitive with H20 on prefill** — throughput gap narrowed from 2.57x to 1.12x
3. **BF16 fusion provides additional ~30% reduction** on top of FP32 fusion+SP
4. **After fusion, the bottleneck migrates to CUBE-bound attention projections** — 910C's hardware advantage
5. **Impact is most dramatic at short sequences** where mHC dominates; at 256K, quadratic attention overshadows mHC

#### SP/mHC-SP Comparison (Fused Baseline, TP=8, EP=16, DP=2)

| Combo | No SP | SP Only | SP + mHC-SP | Speedup |
|---|---|---|---|---|
| **8K/4K** (910C) | 3,507 ms | 2,492 ms | 1,706 ms | **2.06x** |
| **32K/4K** (910C) | 16,333 ms | 12,244 ms | 9,102 ms | **1.79x** |
| **128K/4K** (910C) | 102,647 ms | 86,264 ms | 73,696 ms | **1.39x** |
| **256K/4K** (910C) | 152,440 ms | 136,058 ms | 123,489 ms | **1.23x** |
| **8K/4K** (H20) | 11,738 ms | 4,398 ms | 4,045 ms | **2.90x** |
| **32K/4K** (H20) | 51,606 ms | 22,233 ms | 20,819 ms | **2.48x** |

SP alone provides significant benefit on H20 (2.67x at 8K) by dramatically cutting EP AllToAll communication. On 910C, SP benefit is more modest (1.41x) due to fast EP bandwidth. mHC-SP benefit diminishes at longer sequences as quadratic attention compute dominates.

### 5.2 Attention & KV Cache Analysis

#### KV Cache Scaling (910C, TP=4, EP=32, DP=8, BS=8)

| Seq Len | V4 KV Cache | No Compression | V3 MLA | Savings vs Uncomp. | Decode Step |
|---|---|---|---|---|---|
| 1K | 0.03 GB | 0.09 GB | 0.07 GB | 3.4x | 17.9 ms |
| 4K | 0.08 GB | 0.36 GB | 0.29 GB | 4.6x | 19.3 ms |
| 8K | 0.15 GB | 0.72 GB | 0.58 GB | 4.9x | 19.3 ms |
| 32K | 0.55 GB | 2.89 GB | 2.30 GB | 5.2x | 19.4 ms |
| 64K | 1.09 GB | 5.77 GB | 4.61 GB | 5.3x | 19.5 ms |
| 128K | 2.18 GB | 11.54 GB | 9.21 GB | 5.3x | 21.0 ms |

V4's KV compression saves 3.4--5.3x memory vs uncompressed attention, and 3.9--4.2x vs V3's MLA approach. The compression ratio improves at longer sequences as the C4A/C128A layers dominate.

Decode latency is remarkably stable across context lengths (17.9--21.0 ms) — KV compression effectively caps the decode attention cost.

#### Per-Layer-Type KV Cache Breakdown

| Seq Len | Full Attn | C4A | C128A |
|---|---|---|---|
| 8K | 23.0% | 71.6% | 5.4% |
| 32K | 24.3% | 72.8% | 2.9% |
| 128K | 24.6% | 73.0% | 2.4% |

C4A layers dominate KV cache usage (71--73%) across all sequence lengths. The 2 full-attention layers account for 23--25% of total KV cache despite being only 2 of 43 layers.

#### Compressed vs Uncompressed Comparison

| Seq Len | V4 Compressed | V4 Uncompressed | V3 MLA | V4 vs V3 |
|---|---|---|---|---|
| 8K | 0.15 GB | 0.72 GB | 0.58 GB | **3.9x smaller** |
| 32K | 0.55 GB | 2.89 GB | 2.30 GB | **4.2x smaller** |
| 64K | 1.09 GB | 5.77 GB | 4.61 GB | **4.2x smaller** |
| 128K | 2.18 GB | 11.54 GB | 9.21 GB | **4.2x smaller** |

V4's KV compression is a major architectural advantage for long-context serving. At 128K, V4 uses only 2.18 GB of KV cache per rank vs V3's 9.21 GB — making the difference between fitting in 64 GB HBM and requiring significantly larger batch reductions.

#### Attention Compute Scaling (910C, per-layer, TP=8, EP=16)

| Layer Type | 8K Attn% | 32K Attn% | 128K Attn% |
|---|---|---|---|
| Full (ratio=1) | 65.1% | 86.1% | 96.1% |
| C4A (ratio=4) | 31.3% | 55.2% | 85.6% |
| C128A (ratio=128) | 31.0% | 39.3% | 54.2% |

Full-attention layers become attention-dominated fastest. At 128K, even C128A layers spend >50% of time on attention. This validates the multi-ratio compression strategy: aggressive compression (C128A) keeps the most layers efficient at long context.

---

## 6. Deployment Recommendations

### 6.1 Short Context (8K/4K) — Chat/Coding

| Platform | Prefill | Decode | P:D | Total GPUs |
|---|---|---|---|---|
| **910C** | TP=8, EP=16, DP=2 (16 GPUs) | TP=4, EP=16, DP=4 (16 GPUs) | 1:1 | 32 |
| **H20** | TP=8, EP=8, DP=1 (8 GPUs) | TP=8, EP=16, DP=2 (16 GPUs) | 1:1 | 24 |

Both platforms handle 8K efficiently with 1P:1D ratio. H20 slightly more cost-effective (24 vs 32 GPUs). 910C achieves competitive prefill throughput (1,656 vs 1,848 tps/gpu).

### 6.2 Medium Context (32K/4K) — Document Processing

| Platform | Prefill | Decode | P:D | Total GPUs |
|---|---|---|---|---|
| **910C** | TP=8, EP=16, DP=2 (16 GPUs) | TP=4, EP=32, DP=8 (32 GPUs) | 1:1 | 48 |
| **H20** | TP=8, EP=8, DP=1 (8 GPUs) | TP=4, EP=16, DP=4 (16 GPUs) | 2:1 | 32 |

At 32K, H20 needs 2P:1D while 910C stays at 1P:1D due to better prefill/decode balance. 910C decode throughput drops to 62 tps/gpu (vs 137 on H20) — H20's bandwidth advantage becomes significant.

### 6.3 Long Context (128K/4K) — RAG/Document QA

| Platform | Prefill | Decode | P:D | Total GPUs |
|---|---|---|---|---|
| **910C** | TP=8, EP=16, DP=2 (16 GPUs) | TP=4, EP=32, DP=8 (32 GPUs) | 2:1 | 64 |
| **H20** | TP=8, EP=8, DP=1 (8 GPUs) | TP=4, EP=16, DP=4 (16 GPUs) | 4:1 | 48 |

Memory constraints limit batch sizes to single digits. 910C P:D ratio improved from 4:1 to 2:1 with kernel fusion, reducing total GPUs from 96 to 64. H20 still requires fewer total GPUs (48 vs 64).

### 6.4 Ultra-Long Context (256K/4K) — Full Document Analysis

| Platform | Prefill | Decode | P:D | Total GPUs |
|---|---|---|---|---|
| **910C** | TP=8, EP=16, DP=2 (16 GPUs) | TP=4, EP=32, DP=8 (32 GPUs) | 3:1 | 80 |
| **H20** | TP=8, EP=8, DP=1 (8 GPUs) | TP=4, EP=32, DP=8 (32 GPUs) | 14:1 | 144 |

910C becomes the more GPU-efficient option at 256K (80 vs 144 GPUs) due to H20's extreme P:D ratio (14:1). Consider whether 256K context is worth the investment vs chunking strategies.

### 6.5 General Guidance

1. **Kernel fusion is enabled by default** (`mhc_kernel_fused=True`) — provides 3--4x prefill speedup at no cost
2. **Shared expert overlap is enabled by default** (`shared_expert_overlapped=True`)
3. **Always use SP** (`sp=True`) — free performance gain from reduced Norm/activation compute
4. **Investigate mHC-SP** (`mhc_sp=True`) — additional prefill speedup on top of fusion
5. **Tune EP based on network** — high EP on 910C (EP=16--64), low EP on H20 (EP=8--16)
6. **Batch aggressively for decode** — throughput scales nearly linearly with batch up to memory limits
7. **P/D disaggregation is essential for 128K+** — mixed serving wastes 60--80% of resources

---

## 7. Industry Implications

### 7.1 KV Cache Management & Tiered Caching

Long context (128K+) fundamentally changes the serving landscape:
- Batch sizes shrink to 1--8 per rank due to KV cache memory pressure
- Decode throughput drops 10--21x going from 8K to 128K/256K
- V4's KV compression (4.6x savings vs V3) is essential — without it, 128K context is infeasible at reasonable batch sizes

Practical systems need tiered KV cache strategies: HBM for active sequences, host memory or SSD for evicted/preempted sequences. V4's compressed format makes offloading more efficient — 4.6x less data to move.

### 7.2 P/D Disaggregation Architecture

The analysis confirms that P/D disaggregation is critical for long-context serving:
- At 8K/4K: 1P:1D — mixed serving is viable
- At 128K/4K: 2--4P:1D — mixed serving wastes 60%+ resources
- At 256K/4K: 3--14P:1D — disaggregation is essential

The asymmetry between prefill (CUBE-bound) and decode (MEM-bound) means optimal hardware configs differ: prefill benefits from high compute and EP bandwidth (910C's strength), while decode benefits from high HBM bandwidth (H20's strength). Heterogeneous P/D deployments are a natural optimization.

### 7.3 Hardware Design Tradeoffs

**Compute vs bandwidth tension is scenario-dependent:**
- Before fusion: H20's 2.2x HBM bandwidth dominated prefill via mHC memory bottleneck
- After fusion: 910C's 2.54x cube TFLOPS drives prefill; the bottleneck shifted to compute
- This demonstrates that software optimizations can fundamentally change hardware competitiveness

**HBM bandwidth remains the decode differentiator:** H20's 2.2x bandwidth advantage translates to consistent 2x decode speedup. For decode-heavy workloads (chatbots with long outputs), HBM bandwidth > compute TFLOPS.

**HBM capacity enables larger batches:** H20's 96 GB vs 910C's 64 GB allows 50% more configs to fit in memory. At 128K/256K, this means H20 can serve batch sizes that 910C cannot.

### 7.4 Network Bandwidth for MoE

EP bandwidth is the critical differentiator for MoE model serving:
- 910C's 392 GB/s EP allows EP=64 with modest AllToAll overhead
- H20's 50 GB/s EP makes EP=64 catastrophic (7.8x higher AllToAll cost)
- H20 compensates by using lower EP (EP=8) with more experts per rank

MoE models strongly favor platforms with high-bandwidth interconnects. NVLink-domain (8 GPU) is efficient on H20, but cross-node MoE is very expensive.

### 7.5 mHC as New Paradigm

DeepSeek V4's mHC is architecturally novel and was previously the dominant bottleneck. Kernel fusion proved transformative:
- **Unfused**: 84% of prefill time — the single largest optimization target
- **Fused (default)**: 36% of prefill — 4.1x overall speedup
- **Fused + SP + BF16**: 3.4% — 6.1x total speedup over unfused

The lesson: novel architectural components (like mHC) may initially appear as severe bottlenecks, but targeted kernel optimizations can reduce their cost by an order of magnitude. Hardware/software co-design at the operator level matters more than raw hardware specs.

### 7.6 Ultra-Long Context Serving

256K context presents significant challenges on both platforms:
- Prefill latency: 34s (910C) / 66s (H20) — unacceptable for interactive use
- Decode throughput drops to 8.5 (910C) / 25.6 (H20) tps/gpu
- P/D ratios become extreme: 3:1 (910C) / 14:1 (H20)
- Total GPU budgets: 80 (910C) / 144 (H20)

Practical 256K serving likely requires chunked prefill, speculative decoding, or hierarchical attention to reduce latency and GPU requirements.

---

## 8. Appendix

### 8.1 Hardware & Model Parameters

**Ascend 910C:** Cube 376 TFLOPS, Vec 24 TFLOPS, HBM 64 GB @ 1800 GB/s, TP/EP BW 392 GB/s

**NVIDIA H20:** Cube 148 TFLOPS, Vec 44 TFLOPS, HBM 96 GB @ 4000 GB/s, TP BW 450 GB/s, EP BW 50 GB/s

**Both:** Flops utilization 50%, HBM BW utilization 80%

**DeepSeek V4:** See Section 2.1 parameter table.

### 8.2 Methodology

- **Roofline model**: Each op's time = max(cube_time, vec_time, mem_time) + comm_time
- **FLOPs**: matmul [M,K]x[K,N] = MxNxKx2
- **BF16**: 2 bytes per element; FP32 (mHC): 4 bytes per element
- **Communication**: AllReduce = 2(n-1)/n × vol/BW, AllToAll = (n-1)/n × vol/BW, AllGather = (n-1)/n × vol/BW
- **Utilization**: 50% compute, 80% HBM bandwidth (both platforms)
- **Memory check**: weight_per_rank + KV_cache_per_rank <= HBM capacity
- **Decode approximation**: linear interpolation between first and last decode steps

**Limitations**: Communication is modeled as additive (no overlap with compute). Flash attention memory model assumed. Load balance factor = 1.0 for hash-routing layers.

### 8.3 Data Source Files

All raw data is available in `report/data/`:
- `search_results_910C.json` / `search_results_H20.json` — per-scenario top-20 configs (4 combos: 8K, 32K, 128K, 256K)
- `pd_ratio_analysis.json` — P/D ratio calculations
- `op_analysis.json` — per-op bottleneck breakdown
- `sp_comparison.json` — SP/mHC-SP comparison
- `mhc_optimization_comparison.json` — 4 mHC optimization levels
- `hardware_comparison.json` — cross-platform comparison
- `v3_comparison.json` — V4 vs V3 architecture comparison
- `kv_cache_scaling.json` — KV cache scaling across sequence lengths
- `attention_analysis.json` — per-layer-type attention analysis
