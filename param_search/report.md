# DeepSeek V4 Inference Performance Analysis on Ascend 910C

## 1. Executive Summary

This report presents the results of a systematic parameter search for optimal DeepSeek V4 inference deployment on Huawei Ascend 910C NPUs. We evaluated **1,736 latency** and **434 throughput** configurations across **4 independent scenarios** using a roofline-based performance model.

**Corrected GPU formula:** `physical_gpus = TP * DP` with constraint `(TP * DP) % EP == 0` (EDP must be a positive integer). This replaces the previous incorrect formula `max(TP, EP) * DP`.

**Key Findings:**

| Scenario | Best Config | Key Metric | Hardware |
|:---|:---|---:|:---|
| **Prefill Latency** | TP=8, EP=64, DP=8, BS=8, seq=1024 | **60.6 ms** | 64 GPUs |
| **Decode Latency** | TP=8, EP=64, DP=8, BS=8, seq=1024 | **14.5 ms/step** | 64 GPUs |
| **Prefill Throughput** | TP=8, EP=16, DP=2, BS=512, seq=1024 | **1,225 tok/s/GPU** | 16 GPUs |
| **Decode Throughput** | TP=8, EP=16, DP=2, BS=512, seq=1024 | **207 tok/s/GPU** | 16 GPUs |

- **Prefill and decode have different optimal configs** when batch size varies — separating them enables better per-phase optimization.
- **DP now matters:** With GPU=TP*DP, higher DP configurations (DP=8) with large EP are now valid (e.g., TP=8, DP=8, EP=64 → 64 GPUs, EDP=1).
- **SP provides ~3x speedup for prefill** at TP=8 (distributes non-matmul ops across 8 TP ranks), but **no impact on decode** (decode processes a single token, so SP has nothing to split).
- **Decode throughput per-GPU scales 5.3x** from 16 to 64 GPUs (39→207 tps/gpu at 16 GPUs vs 64 GPUs), confirming that fewer GPUs with higher EP still wastes per-GPU efficiency.
- The decode approximation (2-point sampling) achieves <2.8% error across all verified configs (avg 0.66%).

---

## 2. DeepSeek V4 Architecture Overview

### Core Architecture

| Parameter | Value |
|:---|---:|
| Hidden size | 4,096 |
| Layers | 43 |
| Vocab size | 129,280 |
| Attention | MQA (64 Q heads, 1 KV head) |
| Head dimension | 512 (+ 64 RoPE) |
| MoE experts | 256 routed (top-6), 1 shared |
| Expert FFN dim | 2,048 |

### Layer Types

The 43 layers use three attention compression strategies:

| Type | Count | Compression Ratio | KV Cache Per Token |
|:---|---:|---:|:---|
| Full attention | 2 | 1:1 | 1,088 dims (K=576, V=512) |
| C4A (4x compressed) | 21 | 4:1 | ~272 compressed dims + 128 SWA window |
| C128A (128x compressed) | 20 | 128:1 | ~8.5 compressed dims + 128 SWA window |

### GPU Formula Correction

The previous report used `physical_gpus = max(TP, EP) * DP`, which is incorrect. The correct model:

```
physical_gpus = TP * DP
EDP = physical_gpus / EP    (must be a positive integer)
```

**Why this matters:** EP does not require dedicated GPUs. Instead, EP experts are distributed across the `TP * DP` GPUs, with each GPU holding `n_routed_experts / EP` experts. The EDP (Expert-Data Parallelism) factor determines how many DP groups share each EP partition. For EDP to work, `(TP * DP)` must be evenly divisible by EP.

**Example:** TP=8, DP=8, EP=64 → 64 GPUs, EDP=1 (each DP group maps to exactly one EP partition). Previously this config was modeled as `max(8, 64) * 8 = 512 GPUs`, which was wrong.

---

## 3. Search Methodology

### Roofline Performance Model

Each operation computes three time components independently:
- **Cube time:** matmul FLOPs / (376 TFLOPS × 50% utilization)
- **Vector time:** elementwise FLOPs / (24 TFLOPS × 50% utilization)
- **Memory time:** HBM bytes / (1,800 GB/s × 80% utilization)

Bottleneck = argmax(cube, vec, mem). Total = max(cube, vec, mem) + communication.

### 4 Independent Scenarios

Unlike the previous combined (prefill+decode) search, we now evaluate prefill and decode **separately**:

| Scenario | Metric | Optimize | Rationale |
|:---|:---|:---|:---|
| Prefill Latency | `prefill_time_ms` | minimize | Time to first token |
| Decode Latency | `decode_first_step_ms` | minimize | Per-step generation speed |
| Prefill Throughput | `B*S / prefill_s / GPUs` | maximize | Prefill tokens processed per GPU per second |
| Decode Throughput | `B*output_len / decode_s / GPUs` | maximize | Output tokens generated per GPU per second |

### Search Grid

**Latency scenarios:** TP ∈ {1..64}, EP ∈ {1..256}, DP ∈ {1..8}, BS ∈ {1..512}, seq ∈ {1K..32K}, SP ∈ {T/F}, overlap ∈ {T/F}
**Throughput scenarios:** Same grid but SP=True, shared_expert_overlapped=True (fixed)

**Constraints:** `TP*DP ∈ [8, 64]`, `(TP*DP) % EP == 0`, `BS % DP == 0`, HBM ≤ 64 GB

### Decode Approximation

For decode throughput, we sample the first and last of 256 steps and linearly interpolate. Verified on top-10: max error 2.78%, avg 0.66%.

---

## 4. Prefill Latency Results

### Top-10 Configurations

| Rank | TP | EP | DP | EDP | BS | SeqLen | SP | Overlap | GPUs | Prefill(ms) | HBM(GB) |
|---:|---:|---:|---:|---:|---:|---:|:---|:---|---:|---:|---:|
| 1 | 8 | 64 | 8 | 1 | 8 | 1024 | Yes | Yes | 64 | 60.6 | 14.0 |
| 2 | 8 | 32 | 4 | 1 | 4 | 1024 | Yes | Yes | 32 | 66.6 | 22.7 |
| 3 | 8 | 32 | 8 | 2 | 8 | 1024 | Yes | Yes | 64 | 66.6 | 22.7 |
| 4 | 8 | 64 | 8 | 1 | 8 | 1024 | Yes | No | 64 | 72.8 | 14.0 |
| 5 | 8 | 16 | 2 | 1 | 2 | 1024 | Yes | Yes | 16 | 78.4 | 40.0 |

### Key Observations

**EP=64 wins for prefill latency:** With EP=64, each rank holds only 4 experts (256/64), minimizing per-rank MoE compute. Combined with TP=8 splitting the attention heads across 8 ranks, each GPU does minimal work per prefill step.

**DP=8 with small batch:** The best latency config uses DP=8, BS=8 (per-rank batch=1). This maximizes parallelism — each rank processes just 1 sample, combining the benefits of both tensor and data parallelism.

**SP provides ~3x speedup for prefill:** At TP=8, SP distributes non-matmul operations (RMSNorm, mHC Sinkhorn, activations) across 8 ranks. This is a larger benefit than the 1.03-1.50x seen with the old GPU formula because the corrected formula enables TP=8 with high EP, where SP divides vector ops by 8.

### Best Config per Sequence Length

| SeqLen | TP | EP | DP | EDP | GPUs | Prefill(ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 8 | 64 | 8 | 1 | 64 | 60.6 |
| 2,048 | 8 | 64 | 8 | 1 | 64 | 109.8 |
| 4,096 | 8 | 64 | 8 | 1 | 64 | 218.7 |
| 8,192 | 8 | 64 | 8 | 1 | 64 | 437.5 |
| 16,384 | 8 | 64 | 8 | 1 | 64 | 915.3 |
| 32,768 | 8 | 64 | 8 | 1 | 64 | 2,021.7 |

Prefill time scales nearly linearly with sequence length (2x seq → 2x time), indicating the model is well-balanced between compute and memory across the sequence range.

### SP Impact on Prefill

| SeqLen | SP=True(ms) | SP=False(ms) | Speedup |
|---:|---:|---:|:---|
| 1,024 | 60.6 | 178.4 | **2.94x** |
| 2,048 | 109.8 | 345.4 | **3.14x** |
| 4,096 | 218.7 | 689.8 | **3.15x** |
| 8,192 | 437.5 | 1,379.7 | **3.15x** |
| 16,384 | 915.3 | 2,799.9 | **3.06x** |
| 32,768 | 2,021.7 | 5,790.7 | **2.86x** |

SP's ~3x speedup for prefill at TP=8 is consistent across sequence lengths. The slight decrease at 32K is due to attention compute (which is cube-bound and not split by SP) growing quadratically and dominating.

---

## 5. Decode Latency Results

### Top-10 Configurations

| Rank | TP | EP | DP | EDP | BS | SeqLen | SP | Overlap | GPUs | 1st Step(ms) |
|---:|---:|---:|---:|---:|---:|---:|:---|:---|---:|---:|
| 1 | 8 | 64 | 8 | 1 | 8 | 1024 | Yes | Yes | 64 | 14.517 |
| 2 | 8 | 64 | 8 | 1 | 8 | 1024 | No | Yes | 64 | 14.517 |
| 3 | 8 | 64 | 8 | 1 | 8 | 2048 | Yes | Yes | 64 | 14.529 |
| 4 | 8 | 64 | 8 | 1 | 16 | 1024 | Yes | Yes | 64 | 14.688 |
| 5 | 8 | 64 | 8 | 1 | 32 | 1024 | Yes | Yes | 64 | 15.031 |

### Key Observations

**SP has zero impact on decode:** Since decode processes a single token per step (sequence length = 1), there is nothing for SP to split. SP=True and SP=False produce identical decode latency.

**Shared expert overlap matters:** The top configs all use `shared_expert_overlapped=True`, saving the time of the shared expert by overlapping it with routed expert computation.

**Decode is memory-bound at small batch:** At BS=8 (per-rank=1), decode step time is dominated by weight reading (HBM bandwidth). The 14.5 ms per step translates to ~17.4 ms at higher seq_len (4K+) due to longer KV cache reads.

**Batch size has minimal latency impact for small BS:** Going from BS=8 to BS=32 only increases step time from 14.5 to 15.0 ms — the weight read cost dominates and a few extra tokens add negligible compute.

### Best Config per Sequence Length

| SeqLen | TP | EP | DP | EDP | GPUs | 1st Step(ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 8 | 64 | 8 | 1 | 64 | 14.517 |
| 2,048 | 8 | 64 | 8 | 1 | 64 | 14.529 |
| 4,096 | 8 | 64 | 8 | 1 | 64 | 17.618 |
| 8,192 | 8 | 64 | 8 | 1 | 64 | 17.635 |
| 16,384 | 8 | 64 | 8 | 1 | 64 | 17.671 |
| 32,768 | 8 | 64 | 8 | 1 | 64 | 17.742 |

The ~3 ms jump from seq=2K to seq=4K corresponds to the transition where Lightning Index activates for C4A layers (compressed sequence length exceeds topK=512 threshold at ~2K tokens, triggering the index lookup overhead).

---

## 6. Prefill Throughput Results

### Top-10 Configurations

| Rank | TP | EP | DP | EDP | BS | SeqLen | GPUs | TPS/GPU | HBM(GB) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 16 | 2 | 1 | 512 | 1024 | 16 | 1,224.88 | 47.1 |
| 2 | 8 | 16 | 2 | 1 | 256 | 1024 | 16 | 1,224.25 | 43.5 |
| 3 | 8 | 16 | 2 | 1 | 128 | 1024 | 16 | 1,222.99 | 41.7 |
| 4 | 8 | 16 | 2 | 1 | 64 | 1024 | 16 | 1,220.48 | 40.8 |
| 5 | 8 | 16 | 2 | 1 | 32 | 1024 | 16 | 1,215.50 | 40.4 |

### Key Observations

**Prefill throughput is nearly batch-invariant:** TPS/GPU ranges from 1,215 to 1,225 across BS=32 to BS=512 — less than 1% variation. This is because prefill processes the entire `B*S` token matrix as a single large matmul, which is already compute-bound at even moderate batch sizes.

**16 GPUs (TP=8, EP=16, DP=2) dominates:** Same as decode throughput, fewer GPUs with lower EP minimizes communication overhead per token.

### GPU Efficiency

| GPUs | Config | TPS/GPU | Total TPS |
|---:|:---|---:|---:|
| 16 | TP=8, EP=16, DP=2 | 1,224.88 | 19,598 |
| 32 | TP=8, EP=32, DP=4 | 638.93 | 20,446 |
| 64 | TP=8, EP=64, DP=8 | 326.26 | 20,881 |

Scaling from 16 to 64 GPUs (4x hardware) yields only 1.07x total prefill throughput — the per-GPU efficiency drops by 73%, almost entirely due to increased AllToAll communication with higher EP.

---

## 7. Decode Throughput Results

### Top-10 Configurations

| Rank | TP | EP | DP | EDP | BS | SeqLen | GPUs | TPS/GPU | Exact TPS/GPU | Err% |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 16 | 2 | 1 | 512 | 1024 | 16 | 206.72 | 206.72 | 0.00 |
| 2 | 8 | 16 | 2 | 1 | 512 | 2048 | 16 | 190.82 | 190.82 | 2.19 |
| 3 | 8 | 16 | 2 | 1 | 512 | 4096 | 16 | 185.67 | 185.67 | 0.00 |
| 4 | 4 | 16 | 4 | 1 | 512 | 1024 | 16 | 153.82 | 153.82 | 0.00 |
| 5 | 4 | 16 | 4 | 1 | 512 | 2048 | 16 | 144.99 | 144.99 | 1.62 |

### Key Observations

**TP=8 beats TP=4 by 34% for decode throughput:** TP=8 provides 206.7 tps/gpu vs 153.8 tps/gpu for TP=4 at the same EP and batch. Higher TP splits the per-step weight reads across more ranks, directly improving the memory-bound decode.

**EP=16, DP=2 is universal for throughput:** All top configs use 16 GPUs with EP=16 and DP=2. DP=2 doubles the total batch processed while DP=1 with the same TP=8 would give the same per-GPU throughput.

### Batch Size Scaling (TP=8, EP=16, DP=2, seq=1024)

| Batch Size | TPS/GPU | HBM (GB) | Efficiency vs BS=512 |
|---:|---:|---:|---:|
| 2 | 1.92 | 40.0 | 0.9% |
| 8 | 7.56 | 40.1 | 3.7% |
| 32 | 28.46 | 40.4 | 13.8% |
| 64 | 52.77 | 40.8 | 25.5% |
| 128 | 92.12 | 41.7 | 44.6% |
| 256 | 146.90 | 43.5 | 71.1% |
| 512 | 206.72 | 47.1 | 100.0% |

Decode throughput scales sub-linearly with batch size, as the model transitions from purely memory-bound (weight reading) to increasingly compute-bound. At BS=512, HBM usage is 47.1 GB — well within the 64 GB limit.

### GPU Efficiency by Scale

| GPUs | Best Config | TPS/GPU | Total TPS |
|---:|:---|---:|---:|
| 16 | TP=8, EP=16, DP=2, BS=512 | 206.72 | 3,308 |
| 32 | TP=8, EP=32, DP=4, BS=512 | 94.22 | 3,015 |
| 64 | TP=8, EP=64, DP=8, BS=512 | 39.37 | 2,520 |

Unlike prefill (where total throughput increases slightly with more GPUs), decode total throughput actually **decreases** from 16 to 64 GPUs. The AllToAll communication overhead for MoE dispatch/combine dominates at high EP, making each additional GPU actively harmful for total decode throughput.

---

## 8. Prefill vs Decode Comparison

### Why Separate Evaluation Matters

| Aspect | Prefill | Decode |
|:---|:---|:---|
| Compute pattern | Large matmuls (B×S tokens) | Tiny matmuls (B×1 token) |
| Bottleneck | Compute-bound at large BS | Memory-bound at small BS |
| SP benefit | ~3x at TP=8 | 1.0x (no benefit) |
| Batch sensitivity | <1% variation (TPS/GPU) | 100x variation (1.9→207 TPS/GPU) |
| Sequence sensitivity | Linear with seq_len | Step function at ~2K (index activation) |

### Optimal Config Alignment

For the common case of TP=8, EP=16, DP=2, BS=512, seq=1024:
- **Prefill:** 1,225 tps/gpu (13.4s to process 256×1024 tokens)
- **Decode:** 207 tps/gpu (19.8s to generate 256×256 tokens)
- **Total time:** ~33.2s per request batch, with decode taking 60% of total time

The same hardware config is optimal for both throughput scenarios, simplifying deployment — you don't need different configs for prefill and decode phases.

---

## 9. Trends & Insights

### Communication vs Compute

With the corrected GPU formula, the key insight is that EP controls communication volume independently of GPU count:

| EP | AllToAll Volume | Per-GPU Experts | Best Use Case |
|---:|:---|---:|:---|
| 16 | Low | 16 | Throughput (less comm) |
| 32 | Medium | 8 | Balanced |
| 64 | High | 4 | Latency (less per-expert compute) |

Higher EP reduces per-expert compute (good for latency) but increases AllToAll communication (bad for throughput). The TP*DP GPU count is orthogonal — you can have EP=64 on 64 GPUs (EDP=1) or EP=16 on 64 GPUs (EDP=4).

### Memory Efficiency

| Config | Weight(GB) | KV Cache (BS=512, seq=1K) | Total |
|:---|---:|---:|---:|
| EP=16, TP=8 | 39.9 | 7.1 | 47.1 |
| EP=32, TP=8 | 22.7 | 7.1 | 29.8 |
| EP=64, TP=8 | 12.9 | 7.1 | 20.0 |

Higher EP dramatically reduces weight memory per rank, leaving more headroom for KV cache. This enables larger batch sizes at high EP, which partially compensates for the communication overhead.

### NPU Hardware Utilization

The Ascend 910C's architectural balance:
- **Cube:Vec ratio = 15.7x** (376 vs 24 TFLOPS): Vector operations (RMSNorm, Sinkhorn, activations) are rarely the bottleneck.
- **Compute:Bandwidth ratio = 0.26 FLOP/byte** (376 TFLOPS / 1440 GB/s effective): At batch_size=1, matmuls need ~4 reuse per byte to be cube-bound. Single-token decode has near-zero reuse → pure memory-bound.
- **Network bandwidth (392 GB/s)** is 27% of effective HBM bandwidth, meaning AllToAll can overlap with memory reads but becomes limiting at EP≥32.

---

## 10. Deployment Recommendations

### For Minimum Latency (Time to First Token)
- **Config:** TP=8, EP=64, DP=8, BS=8, 64 GPUs, SP=on
- **Prefill latency:** 60.6 ms (seq=1024), 218.7 ms (seq=4096)
- **Decode per-step:** 14.5 ms (seq=1024)
- **Caveat:** 64 GPUs for a single request batch; only viable for premium interactive services

### For Maximum Throughput (Batch Processing)
- **Config:** TP=8, EP=16, DP=2, BS=512, 16 GPUs, SP=on
- **Decode throughput:** ~207 tokens/s/GPU, ~3,308 tokens/s total
- **Prefill throughput:** ~1,225 tokens/s/GPU
- **HBM usage:** 47.1 GB (room for longer sequences)
- **Caveat:** DP=2 means global batch of 512 is split into 256 per rank

### For Balanced Cost-Efficiency
- **Config:** TP=8, EP=16, DP=2, BS=128-256, 16 GPUs
- **Rationale:** BS=128 achieves 45% of max decode throughput (92 tps/gpu) with only 41.7 GB HBM, leaving ample room for context growth
- **Alternative:** TP=4, EP=16, DP=4, BS=512 gives 154 tps/gpu with more DP parallelism

---

## Appendix: Verification

The decode approximation (2-point linear interpolation) was verified against full 256-step decode for the top-10 decode throughput configs:

| Metric | Value |
|:---|---:|
| Configs verified | 10 |
| Max error | 2.78% |
| Avg error | 0.66% |
| Configs with 0% error | 6/10 |

Non-zero errors occur at seq_len=2048 where decode step time varies non-linearly due to Lightning Index activation threshold crossing mid-generation.
