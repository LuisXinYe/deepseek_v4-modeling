# DeepSeek V4 Inference Performance Analysis: Ascend 910C vs NVIDIA H20

## Executive Summary

This report presents a comprehensive roofline-based performance analysis of DeepSeek V4 inference across two hardware platforms (Ascend 910C and NVIDIA H20) and three serving scenarios (8K/4K, 128K/4K, 256K/4K input/output lengths). Key findings:

1. **mHC (Hyper Connection) dominates prefill** — 53–84% of prefill time on 910C, making it the single largest optimization target. mHC is MEM-bound due to FP32 operations.
2. **Decode is MoE-weight-bound** — MoE routed expert weight loading accounts for 38–42% of decode on 910C and 14–53% on H20.
3. **H20 achieves 1.5–2.6× higher prefill throughput** than 910C, driven by 2.2× HBM bandwidth advantage that alleviates the mHC memory bottleneck.
4. **P/D disaggregation ratios scale with input length** — 1P:1D for 8K, 4P:1D for 128K, 5P:1D for 256K on 910C.
5. **SP with mHC parallelization (mHC-SP) yields 2.2–4.2× prefill speedup** over baseline, reducing mHC from 77–84% down to 12–40% of total time.
6. **910C EP bandwidth advantage (7.8×)** enables efficient high-EP MoE configurations that are prohibitively expensive on H20.

---

## 1. Introduction & Methodology

### 1.1 DeepSeek V4 Architecture

DeepSeek V4 is a Mixture-of-Experts (MoE) model with 43 transformer layers:
- **2 full-attention layers** (ratio=1): standard MQA with 64 Q heads, 1 KV head
- **21 C4A layers** (ratio=4): 4× KV compression with Lightning Index (topK=512)
- **20 C128A layers** (ratio=128): 128× KV compression, no index needed

Key architectural features:
- **MQA** (Multi-Query Attention): 64 Q heads, 1 KV head, head_dim=512
- **Lightning Index**: 64 index heads, dim=128, selects top-512 compressed KV entries
- **MoE**: 256 routed experts (top-6), 1 shared expert, inter_dim=2048
- **mHC (Hyper Connection)**: FP32 pre/post projections + Sinkhorn normalization at every sub-layer
- **KV Compression**: Group projections compress K and V caches by ratio

### 1.2 Roofline Model

Each operation is characterized by three resource dimensions:
- **Cube time**: matmul FLOPs / (cube_tflops × utilization)
- **Vec time**: vector FLOPs / (vec_tflops × utilization)
- **Mem time**: HBM bytes / (bandwidth × utilization)

Bottleneck = argmax(cube, vec, mem). Total time = max(cube, vec, mem) + comm.

### 1.3 Hardware Platforms

| Metric | Ascend 910C | NVIDIA H20 | Ratio |
|---|---|---|---|
| Cube TFLOPS (BF16) | 376 | 148 | **910C 2.54×** |
| Vec TFLOPS (FP32) | 24 | 44 | **H20 1.83×** |
| Cube:Vec ratio | 15.7:1 | 3.4:1 | H20 more balanced |
| HBM Bandwidth (GB/s) | 1,800 | 4,000 | **H20 2.22×** |
| HBM Capacity (GB) | 64 | 96 | **H20 1.5×** |
| TP Bandwidth (GB/s) | 392 | 450 | Similar |
| EP Bandwidth (GB/s) | 392 | 50 | **910C 7.84×** |
| Flops Utilization | 50% | 50% | Same |
| HBM BW Utilization | 80% | 80% | Same |

The most striking hardware difference: 910C has **7.8× higher EP bandwidth** (intra-node interconnect) but **2.2× lower HBM bandwidth**. This fundamentally shapes optimal deployment strategies.

### 1.4 Serving Scenarios

| Scenario | Input Length | Output Length | Use Case |
|---|---|---|---|
| **8K/4K** | 8,192 | 4,096 | Chat, coding assistance |
| **128K/4K** | 131,072 | 4,096 | Long-context RAG, document QA |
| **256K/4K** | 262,144 | 4,096 | Full-window document analysis |

### 1.5 Search Constraints

- **Prefill NPU**: 8, 16, 32, 64, 128, 256 GPUs (physical_gpus = TP × DP)
- **Decode NPU**: 16 or 32 GPUs only (practical deployment constraint)
- **Memory**: weight + KV cache must fit in HBM (64 GB for 910C, 96 GB for H20)
- **Parallelism**: TP divides Q heads (64), EP divides experts (256), (TP×DP) % EP == 0

---

## 2. Parameter Search Results — 8K Input / 4K Output

### 2.1 Ascend 910C

| Scenario | TP | EP | DP | BS | GPUs | Metric |
|---|---|---|---|---|---|---|
| **Prefill Latency** | 8 | 64 | 8 | 8 | 64 | **1,286 ms** |
| **Prefill Throughput** | 8 | 16 | 2 | 256 | 16 | **404.5 tps/gpu** |
| **Decode Latency** | 4 | 32 | 8 | 8 | 32 | **19.4 ms/step** |
| **Decode Throughput** | 4 | 16 | 4 | 512 | 16 | **135.3 tps/gpu** |

Key observations:
- Prefill latency minimizes at 64 GPUs with EP=64, leveraging maximum expert parallelism
- Prefill throughput peaks at 16 GPUs (TP=8, EP=16, DP=2) — larger clusters don't improve per-GPU efficiency
- Decode prefers TP=4 over TP=8, because decode is memory-bound and TP=4 already provides sufficient weight splitting
- Decode throughput benefits from large batch (BS=512) to amortize weight loads

### 2.2 NVIDIA H20

| Scenario | TP | EP | DP | BS | GPUs | Metric |
|---|---|---|---|---|---|---|
| **Prefill Latency** | 8 | 64 | 8 | 8 | 64 | **984 ms** |
| **Prefill Throughput** | 8 | 8 | 1 | 128 | 8 | **1,040 tps/gpu** |
| **Decode Latency** | 4 | 32 | 8 | 8 | 32 | **9.1 ms/step** |
| **Decode Throughput** | 8 | 16 | 2 | 512 | 16 | **207.7 tps/gpu** |

Key observations:
- H20 prefill throughput is **2.57× higher** than 910C (1,040 vs 404.5 tps/gpu) — driven by 2.2× HBM bandwidth reducing the mHC memory bottleneck
- H20 achieves best prefill throughput with just **8 GPUs** (EP=8), while 910C needs 16 GPUs — H20's higher HBM BW compensates for fewer expert-parallel ranks
- H20 decode latency is **2.1× better** (9.1 vs 19.4 ms) — primarily from faster weight loading
- H20 selects EP=8 for throughput (not EP=16/64) because EP bandwidth is only 50 GB/s — high-EP configs incur massive AllToAll penalties

### 2.3 Hardware Comparison at 8K/4K

| Metric | 910C | H20 | H20/910C |
|---|---|---|---|
| Prefill Latency (ms) | 1,286 | 984 | **0.77×** (H20 faster) |
| Prefill Throughput (tps/gpu) | 404.5 | 1,040.2 | **2.57×** (H20 higher) |
| Decode Latency (ms/step) | 19.4 | 9.1 | **0.47×** (H20 faster) |
| Decode Throughput (tps/gpu) | 135.3 | 207.7 | **1.53×** (H20 higher) |

---

## 3. Parameter Search Results — 128K & 256K Input

### 3.1 128K Input / 4K Output

#### Ascend 910C

| Scenario | TP | EP | DP | BS | GPUs | Metric |
|---|---|---|---|---|---|---|
| **Prefill Latency** | 8 | 64 | 8 | 8 | 64 | **26,052 ms** |
| **Prefill Throughput** | 8 | 16 | 2 | 16 | 16 | **314.0 tps/gpu** |
| **Decode Latency** | 4 | 32 | 8 | 8 | 32 | **21.2 ms/step** |
| **Decode Throughput** | 4 | 32 | 8 | 128 | 32 | **15.7 tps/gpu** |

#### NVIDIA H20

| Scenario | TP | EP | DP | BS | GPUs | Metric |
|---|---|---|---|---|---|---|
| **Prefill Latency** | 8 | 64 | 8 | 8 | 64 | **27,283 ms** |
| **Prefill Throughput** | 8 | 8 | 1 | 8 | 8 | **597.4 tps/gpu** |
| **Decode Latency** | 4 | 32 | 8 | 8 | 32 | **9.9 ms/step** |
| **Decode Throughput** | 4 | 16 | 4 | 64 | 16 | **45.6 tps/gpu** |

### 3.2 256K Input / 4K Output

#### Ascend 910C

| Scenario | TP | EP | DP | BS | GPUs | Metric |
|---|---|---|---|---|---|---|
| **Prefill Latency** | 8 | 64 | 8 | 8 | 64 | **64,532 ms** |
| **Prefill Throughput** | 8 | 16 | 2 | 8 | 16 | **253.5 tps/gpu** |
| **Decode Latency** | 4 | 32 | 8 | 8 | 32 | **21.7 ms/step** |
| **Decode Throughput** | 4 | 32 | 8 | 64 | 32 | **8.3 tps/gpu** |

#### NVIDIA H20

| Scenario | TP | EP | DP | BS | GPUs | Metric |
|---|---|---|---|---|---|---|
| **Prefill Latency** | 8 | 64 | 8 | 8 | 64 | **79,460 ms** |
| **Prefill Throughput** | 8 | 8 | 1 | 4 | 8 | **410.9 tps/gpu** |
| **Decode Latency** | 4 | 32 | 8 | 8 | 32 | **10.2 ms/step** |
| **Decode Throughput** | 4 | 16 | 4 | 32 | 16 | **24.6 tps/gpu** |

### 3.3 Long-Context Scaling Analysis

**Memory constraints are severe at long context:**
- At 128K, max batch per rank drops to single digits (BS=16 total with DP=2 on 910C)
- At 256K, max batch further halves (BS=8 total on 910C, BS=4 on H20 with EP=8)
- Decode throughput drops dramatically: 135→15.7→8.3 tps/gpu on 910C (8K→128K→256K)

**Attention compute scales quadratically:**
- Prefill latency: 1.3s → 26.1s → 64.5s (8K→128K→256K) on 910C
- The 8K→128K jump is 20× (16² = 256× sequence, but compression mitigates)
- The 128K→256K jump is 2.5× (closer to the theoretical 4× quadratic)

**Decode latency is relatively stable:**
- 19.4→21.2→21.7 ms/step on 910C across all three combos
- Decode processes 1 token per step, so longer context mainly adds KV cache read cost
- The KV compression (C4A/C128A) effectively caps the decode attention cost

---

## 4. P/D Disaggregated Serving Analysis

### 4.1 Methodology

In prefill/decode disaggregated serving:
- **P instances** process input tokens (prefill phase)
- **D instances** generate output tokens (decode phase)
- QPS balance: N_p × P_rps >= N_d × D_rps

Where:
```
P_rps = P_tps_per_instance / input_len
D_rps = D_tps_per_instance / output_len
N_p / N_d >= (D_tps_instance × input_len) / (P_tps_instance × output_len)
```

### 4.2 Results

#### Ascend 910C

| Combo | P Config | P tps/inst | D Config | D tps/inst | P:D Ratio | Total GPUs |
|---|---|---|---|---|---|---|
| **8K/4K** | TP=8,EP=16,DP=2 (16 GPUs) | 6,472 | TP=4,EP=16,DP=4 (16 GPUs) | 2,165 | **1P:1D** | 32 |
| **128K/4K** | TP=8,EP=16,DP=2 (16 GPUs) | 5,024 | TP=4,EP=32,DP=8 (32 GPUs) | 502 | **4P:1D** | 96 |
| **256K/4K** | TP=8,EP=16,DP=2 (16 GPUs) | 4,057 | TP=4,EP=32,DP=8 (32 GPUs) | 264 | **5P:1D** | 112 |

#### NVIDIA H20

| Combo | P Config | P tps/inst | D Config | D tps/inst | P:D Ratio | Total GPUs |
|---|---|---|---|---|---|---|
| **8K/4K** | TP=8,EP=8,DP=1 (8 GPUs) | 8,321 | TP=8,EP=16,DP=2 (16 GPUs) | 3,323 | **1P:1D** | 24 |
| **128K/4K** | TP=8,EP=8,DP=1 (8 GPUs) | 4,779 | TP=4,EP=16,DP=4 (16 GPUs) | 730 | **5P:1D** | 56 |
| **256K/4K** | TP=8,EP=8,DP=1 (8 GPUs) | 3,287 | TP=4,EP=16,DP=4 (16 GPUs) | 394 | **8P:1D** | 80 |

### 4.3 Analysis

**P:D ratio is driven by input/output length imbalance:**
- At 8K/4K (ratio 2:1), P:D ≈ 1:1 — prefill and decode scale proportionally
- At 128K/4K (ratio 32:1), P:D ≈ 4–5:1 — prefill becomes the bottleneck
- At 256K/4K (ratio 64:1), P:D ≈ 5–8:1 — even more prefill-heavy

**910C uses fewer P instances than H20 at long context:**
- 128K: 910C needs 4P:1D vs H20's 5P:1D
- 256K: 910C needs 5P:1D vs H20's 8P:1D
- This is because 910C P instances process fewer tokens per second but D instances are also slower, partially balancing the ratio

**GPU budget scales significantly with context length:**
- 8K/4K: 32 GPUs (910C), 24 GPUs (H20)
- 128K/4K: 96 GPUs (910C), 56 GPUs (H20)
- 256K/4K: 112 GPUs (910C), 80 GPUs (H20)

H20 consistently requires fewer total GPUs due to higher per-GPU throughput and smaller P instance sizes (8 vs 16 GPUs).

---

## 5. SP & mHC-SP Analysis

### 5.1 Three Configurations Tested

Tested with TP=8, EP=16, DP=2, representative batch sizes:

1. **No SP** (`sp=False, mhc_sp=False`) — baseline
2. **SP only** (`sp=True, mhc_sp=False`) — sequence parallelism for RMSNorm/activations; mHC still at T_full
3. **SP + mHC-SP** (`sp=True, mhc_sp=True`) — mHC operations also parallelized across TP dimension

### 5.2 Prefill Time Comparison (ms)

#### Ascend 910C

| Combo | No SP | SP Only | SP + mHC-SP | Speedup (No SP → SP+mHC-SP) |
|---|---|---|---|---|
| **8K/4K** | 11,159 | 10,144 | 2,663 | **4.19×** |
| **128K/4K** | 225,083 | 208,700 | 89,000 | **2.53×** |
| **256K/4K** | 274,876 | 258,494 | 138,793 | **1.98×** |

#### NVIDIA H20

| Combo | No SP | SP Only | SP + mHC-SP | Speedup (No SP → SP+mHC-SP) |
|---|---|---|---|---|
| **8K/4K** | 15,181 | 7,842 | 4,475 | **3.39×** |
| **128K/4K** | 336,214 | 218,712 | 164,846 | **2.04×** |
| **256K/4K** | 871,651 | 636,641 | 528,911 | **1.65×** |

### 5.3 Key Findings

**SP alone provides modest benefit on 910C (9–6%) but significant benefit on H20 (27–48%):**
- On 910C, SP reduces Norm and MoE routed time, but mHC dominates and is unaffected
- On H20, SP additionally reduces the much larger communication cost (EP AllToAll is massive on H20 due to low EP BW)

**mHC-SP is transformative on 910C:**
- At 8K: mHC drops from 84% to 40% of total time (8× reduction in mHC absolute time)
- At 128K: mHC drops from 66% to 19%
- At 256K: mHC drops from 53% to 12%
- The 8× mHC reduction comes from TP=8 parallelism on the memory-bound FP32 operations

**mHC-SP benefit diminishes at longer sequences:**
- At 8K, mHC is the overwhelming bottleneck → mHC-SP gives 4.2× speedup
- At 256K, attention compute (quadratic) grows to dominate → mHC-SP gives 2.0× speedup
- This suggests that for long-context workloads, attention optimization matters more than mHC optimization

### 5.4 Category Breakdown Shift (910C, 8K/4K)

| Category | No SP | SP Only | SP + mHC-SP |
|---|---|---|---|
| mHC | **76.6%** | **84.3%** | **40.1%** |
| Communication | 10.0% | 4.6% | 17.4% |
| Attention Proj | 5.6% | 6.2% | 23.6% |
| Attention Compute | 1.7% | 1.9% | 7.2% |
| MoE (all) | 3.4% | 0.7% | 2.8% |
| Others | 2.6% | 2.4% | 8.9% |

After mHC-SP, the bottleneck shifts to attention projections and communication — indicating that further optimization should target these areas.

---

## 6. Op Boundedness Analysis

### 6.1 Per-Category Bottleneck Summary

| Category | Prefill Bottleneck | Decode Bottleneck |
|---|---|---|
| Attention Projections | CUBE | MEM |
| Attention Compute | CUBE | MEM |
| KV Compression | CUBE | VEC/MEM |
| Lightning Index | COMM (score AllReduce) | COMM |
| mHC Pre/Post | MEM (FP32) | MEM (FP32) |
| Sinkhorn | VEC (FP32) | MEM (FP32) |
| MoE Gate | CUBE | MEM |
| MoE Routed Experts | CUBE | MEM |
| Communication | COMM | COMM |

**Key insight: Prefill is compute-bound (CUBE/VEC), Decode is memory-bound (MEM/COMM).**

This is the classic LLM inference pattern — prefill processes many tokens (compute-intensive matmuls) while decode generates one token per step (weight-loading dominated).

### 6.2 mHC: The Anomalous Component

mHC operations are unique because they use **FP32 throughout** (4 bytes per element instead of BF16's 2 bytes). This doubles memory traffic and changes the compute-to-memory ratio:

- **mHC pre/post**: Memory-bound even during prefill (FP32 doubles memory, while cube FLOPs don't scale)
- **Sinkhorn**: Pure VEC operation (no matmuls), VEC-bound during prefill on 910C
- On H20 with 1.83× higher vec TFLOPS, Sinkhorn is relatively less costly

### 6.3 Prefill Op Breakdown (910C, Best Throughput Config)

| Category | 8K/4K | 128K/4K | 256K/4K |
|---|---|---|---|
| mHC | **84.4%** | **65.6%** | **52.9%** |
| Attention Compute | 1.9% | 13.1% | 20.6% |
| Lightning Index | 1.2% | 11.7% | 18.7% |
| Attention Proj | 6.2% | 4.8% | 3.9% |
| Communication | 4.4% | 3.4% | 2.8% |
| MoE | 0.7% | 0.6% | 0.5% |
| Others | 1.2% | 0.9% | 0.7% |

As sequence length increases, attention compute grows quadratically and erodes mHC's dominance. At 256K, attention + index account for 39.3% combined.

### 6.4 Decode Op Breakdown (910C, Best Throughput Config)

| Category | 8K/4K | 128K/4K | 256K/4K |
|---|---|---|---|
| MoE Routed | **41.6%** | **37.9%** | **39.8%** |
| mHC | 28.8% | 6.6% | 3.5% |
| Attention Compute | 10.8% | 20.7% | 20.8% |
| Communication | 9.6% | 11.7% | 11.8% |
| Lightning Index | 4.3% | 15.4% | 16.2% |
| Attention Proj | 4.0% | 6.4% | 6.7% |
| Others | 1.0% | 1.2% | 1.2% |

Decode is dominated by MoE weight loading (38–42%). At longer sequences, attention and Lightning Index KV cache reads grow.

---

## 7. Time Breakdown by Category — Detailed Tables

### 7.1 Prefill (H20, Best Throughput Config)

| Category | 8K/4K | 128K/4K | 256K/4K |
|---|---|---|---|
| mHC | **48.9%** | **28.1%** | **19.3%** |
| Attention Compute | 6.2% | 31.6% | 42.3% |
| Attention Proj | 20.3% | 11.7% | 8.0% |
| Lightning Index | 2.5% | 16.0% | 21.6% |
| Communication | 15.0% | 8.6% | 5.9% |
| MoE | 3.7% | 2.1% | 1.5% |
| KV Compression | 1.9% | 1.1% | 0.8% |
| Others | 1.5% | 0.9% | 0.6% |

H20 shows lower mHC percentage than 910C because H20's higher HBM bandwidth (4000 vs 1800 GB/s) alleviates the FP32 memory bottleneck.

### 7.2 Decode (H20, Best Throughput Config)

| Category | 8K/4K | 128K/4K | 256K/4K |
|---|---|---|---|
| Communication | **44.6%** | 16.6% | 13.2% |
| mHC | 19.8% | 4.3% | 2.3% |
| MoE Routed | 14.3% | **49.5%** | **53.4%** |
| Attention Compute | 7.3% | 13.5% | 13.9% |
| Attention Proj | 8.2% | 4.2% | 4.5% |
| Lightning Index | 3.7% | 11.0% | 11.9% |
| Others | 2.0% | 0.8% | 0.8% |

On H20 at 8K, decode communication is the dominant factor (44.6%) — this is because H20's low EP bandwidth (50 GB/s) makes AllToAll extremely expensive. At longer sequences, MoE weight loading takes over.

---

## 8. Ascend 910C vs NVIDIA H20 — Detailed Comparison

### 8.1 Prefill Throughput Comparison

| Combo | 910C (tps/gpu) | H20 (tps/gpu) | H20/910C | 910C GPUs | H20 GPUs |
|---|---|---|---|---|---|
| 8K/4K | 404.5 | 1,040.2 | **2.57×** | 16 | 8 |
| 128K/4K | 314.0 | 597.4 | **1.90×** | 16 | 8 |
| 256K/4K | 253.5 | 410.9 | **1.62×** | 16 | 8 |

H20 advantage narrows at longer sequences because:
1. Attention compute (CUBE-bound) grows and 910C has 2.54× higher cube TFLOPS
2. mHC (MEM-bound) shrinks as a fraction, reducing H20's bandwidth advantage

### 8.2 Decode Throughput Comparison

| Combo | 910C (tps/gpu) | H20 (tps/gpu) | H20/910C | 910C GPUs | H20 GPUs |
|---|---|---|---|---|---|
| 8K/4K | 135.3 | 207.7 | **1.53×** | 16 | 16 |
| 128K/4K | 15.7 | 45.6 | **2.91×** | 32 | 16 |
| 256K/4K | 8.3 | 24.6 | **2.98×** | 32 | 16 |

H20 advantage grows at longer sequences for decode because:
1. Longer context → more KV cache → more HBM reads → H20's 2.2× bandwidth advantage matters more
2. 910C needs 32 GPUs (larger EP) while H20 fits in 16 GPUs at long context

### 8.3 Decode Latency Comparison

| Combo | 910C (ms/step) | H20 (ms/step) | H20/910C |
|---|---|---|---|
| 8K/4K | 19.4 | 9.1 | **0.47×** (H20 faster) |
| 128K/4K | 21.2 | 9.9 | **0.47×** |
| 256K/4K | 21.7 | 10.2 | **0.47×** |

H20 decode latency is consistently ~2× better across all context lengths, directly reflecting the HBM bandwidth ratio.

### 8.4 Root Cause Analysis

**Where 910C wins:**
- Cube-heavy workloads: attention projections (matmul-dominant), MoE expert matmuls
- High-EP configurations: EP=32/64 at low cost due to 392 GB/s EP bandwidth
- Prefill latency at EP=64 is similar between platforms (910C cube vs H20 bandwidth trade-off)

**Where H20 wins:**
- Memory-bound workloads: mHC FP32 operations, decode weight loading, KV cache reads
- Low-EP throughput: EP=8 with only 8 GPUs matches or exceeds 910C's 16 GPU configurations
- Decode phase across all scenarios (2× faster consistently)

**H20's EP bandwidth problem:**
- H20 EP bandwidth is only 50 GB/s (IB cross-node) vs 910C's 392 GB/s (on-die interconnect)
- At EP=64, AllToAll time on H20 is ~7.8× higher than 910C
- This forces H20 to use EP=8 for throughput configs, meaning all 256 experts must be split across only 8 ranks (32 experts per rank)
- The weight memory per rank is higher at EP=8, but H20's 96 GB HBM accommodates this

---

## 9. Industry Implications

### 9.1 KV Cache & Long Context

Long context (128K+) fundamentally changes the serving landscape:
- Batch sizes shrink to 1–8 per rank due to KV cache memory
- Decode throughput drops 10–16× going from 8K to 128K/256K
- P/D ratios increase 4–8× requiring proportionally more prefill instances

**Implication**: KV compression (as in DeepSeek V4's C4A/C128A) is essential for practical long-context serving. Without it, 128K context would be infeasible at reasonable batch sizes.

### 9.2 P/D Disaggregation

The analysis confirms that P/D disaggregation is critical for long-context serving:
- At 128K/4K: 4–5 prefill instances per decode instance
- At 256K/4K: 5–8 prefill instances per decode instance
- Mixed (non-disaggregated) serving would waste GPU resources on the non-bottleneck phase

**Practical guidance**: For 128K workloads on 910C, a cluster of 96 GPUs (4×16 GPU prefill + 1×32 GPU decode) provides QPS-balanced serving.

### 9.3 Hardware Design Insights

**Cube:Vec ratio matters for mHC workloads:**
- 910C (15.7:1) leaves vec/mem massively underutilized during mHC Sinkhorn (pure VEC)
- H20 (3.4:1) is better balanced for mixed CUBE/VEC workloads
- Future NPUs should consider higher vec TFLOPS for FP32 normalization layers

**HBM bandwidth is the decode differentiator:**
- H20's 2.2× bandwidth advantage translates to consistent 2× decode speedup
- For decode-heavy workloads (chatbots with long outputs), HBM bandwidth > compute TFLOPS

**HBM capacity enables larger batches:**
- H20's 96 GB vs 910C's 64 GB allows 50% more configs to fit in memory
- At 128K/256K, this means H20 can fit batch sizes that 910C cannot

### 9.4 Network for MoE

EP bandwidth is the critical differentiator for MoE model serving:
- 910C's 392 GB/s EP allows EP=64 with modest AllToAll overhead
- H20's 50 GB/s EP makes EP=64 catastrophic (7.8× higher AllToAll cost)
- H20 compensates by using lower EP (EP=8) with more experts per rank

**Implication**: MoE models strongly favor platforms with high-bandwidth interconnects. NVLink-domain (8 GPU) is efficient on H20, but cross-node MoE is very expensive.

### 9.5 mHC as a New Paradigm

DeepSeek V4's mHC (Hyper Connection) is architecturally novel but computationally expensive:
- Accounts for 49–84% of prefill time depending on hardware and sequence length
- Operates in FP32, doubling memory traffic vs BF16
- Sinkhorn normalization is a pure VEC operation with no matmul parallelism

**Optimization opportunities:**
1. **mHC-SP** (parallelizing mHC across TP): 2–4× prefill speedup (demonstrated in this analysis)
2. **FP16/BF16 mHC**: Halving memory traffic would proportionally reduce mHC time
3. **Fused Sinkhorn kernels**: Custom kernels combining Sinkhorn iterations could reduce memory round-trips

---

## 10. Deployment Recommendations

### 10.1 Short Context (8K/4K) — Chat/Coding

| Platform | Prefill | Decode | P:D | Total GPUs |
|---|---|---|---|---|
| **910C** | TP=8, EP=16, DP=2 (16 GPUs) | TP=4, EP=16, DP=4 (16 GPUs) | 1:1 | 32 |
| **H20** | TP=8, EP=8, DP=1 (8 GPUs) | TP=8, EP=16, DP=2 (16 GPUs) | 1:1 | 24 |

- Both platforms handle 8K efficiently
- H20 more cost-effective (24 vs 32 GPUs for balanced serving)
- Shared expert overlapping recommended for both

### 10.2 Long Context (128K/4K) — RAG/Document QA

| Platform | Prefill | Decode | P:D | Total GPUs |
|---|---|---|---|---|
| **910C** | TP=8, EP=16, DP=2 (16 GPUs) | TP=4, EP=32, DP=8 (32 GPUs) | 4:1 | 96 |
| **H20** | TP=8, EP=8, DP=1 (8 GPUs) | TP=4, EP=16, DP=4 (16 GPUs) | 5:1 | 56 |

- 910C requires significantly more GPUs (96 vs 56) for balanced serving
- H20's compact P instance (8 GPUs) is highly efficient
- Memory is the primary constraint on batch sizes

### 10.3 Ultra-Long Context (256K/4K) — Full Document Analysis

| Platform | Prefill | Decode | P:D | Total GPUs |
|---|---|---|---|---|
| **910C** | TP=8, EP=16, DP=2 (16 GPUs) | TP=4, EP=32, DP=8 (32 GPUs) | 5:1 | 112 |
| **H20** | TP=8, EP=8, DP=1 (8 GPUs) | TP=4, EP=16, DP=4 (16 GPUs) | 8:1 | 80 |

- 256K serving is expensive on both platforms
- Decode throughput drops to 8–25 tps/gpu — a single decode instance can only serve ~2–6 requests/sec
- Consider whether 256K context is worth the GPU investment vs chunking strategies

### 10.4 General Guidance

1. **Always use SP** (`sp=True`) — free performance gain from reduced Norm/activation compute
2. **Investigate mHC-SP** — if mHC parallelization is feasible in the framework, it provides 2–4× prefill speedup
3. **Tune EP based on network** — high EP on 910C (EP=16–64), low EP on H20 (EP=8–16)
4. **Batch aggressively for decode** — decode throughput scales nearly linearly with batch size up to memory limits
5. **P/D disaggregation is essential for 128K+** — mixed serving wastes 60–80% of resources

---

## Appendix A: Methodology Details

- **Roofline model**: Each op's time = max(cube_time, vec_time, mem_time) + comm_time
- **FLOPs**: matmul [M,K]×[K,N] = M×N×K×2
- **BF16**: 2 bytes per element; FP32 (mHC): 4 bytes per element
- **Communication**: AllReduce = 2(n-1)/n × vol/BW, AllToAll = (n-1)/n × vol/BW, AllGather = (n-1)/n × vol/BW
- **Utilization**: 50% compute, 80% HBM bandwidth (both platforms)
- **Memory check**: weight_per_rank + KV_cache_per_rank <= HBM capacity
- **Decode approximation**: linear interpolation between first and last decode steps

## Appendix B: Model Configuration Summary

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

## Appendix C: Data Files

All raw data is available in `report/data/`:
- `search_results_910C.json` / `search_results_H20.json` — per-scenario top-20 configs
- `pd_ratio_analysis.json` — P/D ratio calculations
- `op_analysis.json` — per-op bottleneck breakdown
- `sp_comparison.json` — SP/mHC-SP comparison
- `hardware_comparison.json` — cross-platform comparison
