# DeepSeek V4 Inference Performance Analysis — PPT Outline

---

## Slide 1: Title Slide

- **Title**: DeepSeek V4 Inference Performance Analysis: Ascend 910C vs NVIDIA H20
- Roofline-based performance modeling across three serving scenarios
- Platforms: Ascend 910C, NVIDIA H20
- Scenarios: 8K/4K, 128K/4K, 256K/4K (input/output token lengths)
- **Visualization**: Title text with platform logos and scenario icons

---

## Slide 2: Executive Summary — Key Findings

- mHC (Hyper Connection) dominates prefill: 53--84% of prefill time on 910C; FP32 memory-bound
- Decode is MoE-weight-bound: routed expert weight loading = 38--42% of decode on 910C
- H20 achieves 1.5--2.6x higher prefill throughput than 910C, driven by 2.2x HBM bandwidth
- P/D disaggregation ratios scale with input length: 1P:1D (8K) to 5--8P:1D (256K)
- mHC-SP yields 2.0--4.2x prefill speedup; 910C EP bandwidth advantage (7.8x) enables high-EP MoE
- **Visualization**: Numbered highlight boxes or icon grid summarizing six key findings

---

## Slide 3: DeepSeek V4 Architecture Overview

- 43 transformer layers: 2 full-attention, 21 C4A (4x compression), 20 C128A (128x compression)
- MQA: 64 Q heads, 1 KV head, head_dim=512 with Q LoRA rank 1,024
- MoE: 256 routed experts (top-6 routing), 1 shared expert, inter_dim=2,048
- mHC (Hyper Connection): FP32 pre/post projections + Sinkhorn normalization at every sub-layer
- Lightning Index: 64 heads, dim=128, topK=512 for compressed KV entry selection
- **Visualization**: Architecture block diagram showing layer types, MoE routing, and mHC data flow

---

## Slide 4: Hardware Comparison — Ascend 910C vs NVIDIA H20

- Cube TFLOPS (BF16): 910C = 376, H20 = 148 (910C 2.54x advantage)
- HBM Bandwidth: 910C = 1,800 GB/s, H20 = 4,000 GB/s (H20 2.22x advantage)
- EP Bandwidth: 910C = 392 GB/s, H20 = 50 GB/s (910C 7.84x advantage)
- HBM Capacity: 910C = 64 GB, H20 = 96 GB (H20 1.5x advantage)
- Cube:Vec ratio: 910C = 15.7:1 (imbalanced), H20 = 3.4:1 (more balanced)
- **Visualization**: Side-by-side comparison table with color-coded advantage indicators (green = winner per metric)

---

## Slide 5: Methodology — Roofline Performance Model

- Each op characterized by three resource dimensions: Cube time, Vec time, Mem time
- Bottleneck = argmax(cube, vec, mem); Total time = max(cube, vec, mem) + comm
- FLOPs convention: matmul [M,K] x [K,N] = M x N x K x 2; BF16 = 2 bytes, FP32 = 4 bytes
- Utilization assumptions: 50% compute, 80% HBM bandwidth (both platforms)
- Search space: TP in {1..64}, EP in {1..256}, DP in {1..8}, BS in {1..512}; constraint: (TP x DP) % EP == 0
- **Visualization**: Roofline diagram showing compute vs memory intensity with example op placements

---

## Slide 6: Parameter Search Results — 8K/4K (Both Platforms)

- 910C Prefill Throughput: 404.5 tps/gpu (TP=8, EP=16, DP=2, BS=256, 16 GPUs)
- H20 Prefill Throughput: 1,040.2 tps/gpu (TP=8, EP=8, DP=1, BS=128, 8 GPUs) — 2.57x higher
- 910C Decode Throughput: 135.3 tps/gpu (TP=4, EP=16, DP=4, BS=512, 16 GPUs)
- H20 Decode Throughput: 207.7 tps/gpu (TP=8, EP=16, DP=2, BS=512, 16 GPUs) — 1.53x higher
- Decode prefers TP=4 on 910C (memory-bound), H20 achieves best prefill with only 8 GPUs
- **Visualization**: Grouped bar chart comparing 910C vs H20 for prefill and decode throughput/latency

---

## Slide 7: Parameter Search Results — 128K/4K (Both Platforms)

- 910C Prefill Throughput: 314.0 tps/gpu (16 GPUs); H20: 597.4 tps/gpu (8 GPUs) — H20 1.90x higher
- 910C Decode Throughput: 15.7 tps/gpu (32 GPUs); H20: 45.6 tps/gpu (16 GPUs) — H20 2.91x higher
- Max batch per rank drops to single digits due to KV cache memory at 128K
- 910C Prefill Latency: 26.1s; H20: 27.3s — nearly equal (910C cube advantage offsets H20 BW)
- 910C Decode Latency: 21.2 ms/step; H20: 9.9 ms/step — H20 2.1x faster
- **Visualization**: Grouped bar chart with 910C vs H20 metrics; annotation for memory constraints

---

## Slide 8: Parameter Search Results — 256K/4K (Both Platforms)

- 910C Prefill Throughput: 253.5 tps/gpu (16 GPUs); H20: 410.9 tps/gpu (8 GPUs) — H20 1.62x higher
- 910C Decode Throughput: 8.3 tps/gpu (32 GPUs); H20: 24.6 tps/gpu (16 GPUs) — H20 2.98x higher
- 910C Prefill Latency: 64.5s; H20: 79.5s — 910C 1.23x faster (cube TFLOPS advantage at quadratic attention)
- Max batch halves again: BS=8 on 910C, BS=4 on H20 at EP=8
- 256K serving is expensive on both platforms; consider chunking strategies
- **Visualization**: Grouped bar chart; callout box highlighting memory-constrained batch sizes

---

## Slide 9: Long-Context Scaling Trends

- Prefill latency scales super-linearly: 1.3s -> 26.1s -> 64.5s (8K -> 128K -> 256K) on 910C
- Decode throughput drops dramatically: 135.3 -> 15.7 -> 8.3 tps/gpu on 910C (8K -> 128K -> 256K)
- Decode latency is relatively stable: 19.4 -> 21.2 -> 21.7 ms/step (compression caps attention cost)
- H20 advantage narrows for prefill at longer sequences (910C cube TFLOPS helps with quadratic attention)
- H20 advantage widens for decode at longer sequences (more KV cache reads favor 2.2x HBM BW)
- **Visualization**: Dual-axis line chart — prefill latency and decode throughput vs sequence length for both platforms

---

## Slide 10: P/D Disaggregation Concept

- Prefill/Decode disaggregated serving: P instances process input tokens, D instances generate output tokens
- QPS balance equation: N_p x P_rps >= N_d x D_rps
- P:D ratio formula: N_p/N_d >= (D_tps_instance x input_len) / (P_tps_instance x output_len)
- Ratio driven by input/output length imbalance: 2:1 at 8K/4K -> 64:1 at 256K/4K
- Mixed (non-disaggregated) serving wastes 60--80% of GPU resources at long context
- **Visualization**: Conceptual diagram showing P instances feeding D instances with QPS flow arrows

---

## Slide 11: P/D Ratio Results Table

- 910C 8K/4K: 1P:1D (P=16 GPUs, D=16 GPUs) — balanced prefill and decode
- 910C 128K/4K: 4P:1D (P=4x16 GPUs, D=1x32 GPUs) — prefill becomes bottleneck
- 910C 256K/4K: 5P:1D (P=5x16 GPUs, D=1x32 GPUs) — even more prefill-heavy
- H20 128K/4K: 5P:1D; 256K/4K: 8P:1D — H20 needs more P instances (faster D, same P throughput)
- 910C uses fewer P instances than H20 at long context (slower D partially balances ratio)
- **Visualization**: Table with color gradient showing P:D ratio escalation across scenarios and platforms

---

## Slide 12: P/D GPU Budget Analysis

- 8K/4K: 32 GPUs (910C) vs 24 GPUs (H20) — H20 33% fewer GPUs
- 128K/4K: 96 GPUs (910C) vs 56 GPUs (H20) — H20 42% fewer GPUs
- 256K/4K: 112 GPUs (910C) vs 80 GPUs (H20) — H20 29% fewer GPUs
- H20 advantage from compact P instances (8 GPUs vs 16 GPUs) and higher per-GPU throughput
- GPU budget scales 3--4x from 8K to 256K — long-context is fundamentally expensive
- **Visualization**: Stacked bar chart showing prefill GPUs + decode GPUs for each scenario/platform

---

## Slide 13: SP and mHC-SP Comparison

- Three configurations tested: No SP (baseline), SP only, SP + mHC-SP
- SP alone: modest benefit on 910C (6--9%), significant on H20 (27--48%) due to EP AllToAll reduction
- mHC-SP on 910C at 8K: mHC drops from 84% to 40% of total time (4.19x overall speedup)
- mHC-SP benefit diminishes at longer sequences: 4.2x at 8K, 2.5x at 128K, 2.0x at 256K
- mHC-SP works by parallelizing FP32 mHC operations across TP dimension (TP=8 -> 8x mHC reduction)
- **Visualization**: Grouped bar chart showing prefill time (ms) for three configs across 8K/128K/256K

---

## Slide 14: mHC-SP Speedup Analysis

- 910C speedups (No SP -> SP+mHC-SP): 4.19x (8K), 2.53x (128K), 1.98x (256K)
- H20 speedups: 3.39x (8K), 2.04x (128K), 1.65x (256K)
- Speedup diminishes because quadratic attention compute grows and is unaffected by mHC-SP
- After mHC-SP, bottleneck shifts to attention projections (23.6%) and communication (17.4%) at 8K
- For long-context workloads, attention optimization becomes more important than mHC optimization
- **Visualization**: Line chart showing speedup factor vs sequence length for both platforms; annotation for bottleneck shift

---

## Slide 15: Op Boundedness Analysis — Prefill

- Attention Projections: CUBE-bound (compute-intensive matmuls)
- Attention Compute: CUBE-bound (quadratic in sequence length)
- mHC Pre/Post: MEM-bound even during prefill (FP32 doubles memory traffic)
- Sinkhorn: VEC-bound on 910C (pure FP32 vector operation, no matmul parallelism)
- MoE Gate and Routed Experts: CUBE-bound during prefill
- **Visualization**: Table with per-category bottleneck indicator (CUBE / VEC / MEM / COMM) with color coding

---

## Slide 16: Op Boundedness Analysis — Decode

- MoE Routed Experts: MEM-bound (weight loading dominates at 38--42% of decode on 910C)
- Attention Projections: MEM-bound (single-token decode -> low arithmetic intensity)
- Attention Compute: MEM-bound (KV cache reads dominate)
- mHC: MEM-bound (FP32, same as prefill but smaller fraction)
- Communication: COMM-bound (AllToAll for MoE, AllReduce for attention)
- **Visualization**: Table with per-category bottleneck indicator; contrasted side-by-side with prefill slide

---

## Slide 17: Prefill Time Breakdown by Category (%)

- 910C 8K: mHC = 84.4%, Attention Proj = 6.2%, Communication = 4.4%, Attention Compute = 1.9%
- 910C 128K: mHC = 65.6%, Attention Compute = 13.1%, Lightning Index = 11.7%
- 910C 256K: mHC = 52.9%, Attention Compute = 20.6%, Lightning Index = 18.7%
- H20 8K: mHC = 48.9%, Attention Proj = 20.3%, Communication = 15.0%
- As sequence length increases, attention compute grows quadratically and erodes mHC dominance
- **Visualization**: Stacked bar chart (or stacked area) showing category % breakdown for each scenario and platform

---

## Slide 18: 910C vs H20 Comparison Summary

- Prefill throughput: H20 wins 1.62--2.57x (HBM bandwidth alleviates mHC memory bottleneck)
- Decode latency: H20 wins consistently at ~2x (directly reflects 2.2x HBM BW ratio)
- 910C advantage in prefill latency at 256K (cube TFLOPS helps quadratic attention)
- 910C EP bandwidth (7.84x) enables EP=32/64 configs that are prohibitive on H20
- H20 consistently needs fewer total GPUs (24--80) vs 910C (32--112) for balanced P/D serving
- **Visualization**: Summary comparison table with green/red arrows indicating platform advantage per metric

---

## Slide 19: Industry Implications and Optimization Opportunities

- KV compression (C4A/C128A) is essential: without it, 128K context is infeasible at reasonable batch sizes
- mHC optimization is the highest-ROI target: mHC-SP (2--4x), FP16/BF16 mHC, fused Sinkhorn kernels
- Cube:Vec ratio matters: 910C's 15.7:1 leaves vec/mem underutilized for FP32 normalization workloads
- HBM bandwidth is the decode differentiator: for chatbot workloads, HBM BW > compute TFLOPS
- MoE models strongly favor high-bandwidth interconnects: NVLink-domain efficient, cross-node MoE expensive
- **Visualization**: Matrix chart mapping optimization opportunities to impact level (high/medium/low) and effort

---

## Slide 20: Deployment Recommendations

- 8K/4K (Chat/Coding): 910C = 32 GPUs (1P:1D), H20 = 24 GPUs (1P:1D); H20 more cost-effective
- 128K/4K (RAG/Doc QA): 910C = 96 GPUs (4P:1D), H20 = 56 GPUs (5P:1D); P/D disaggregation essential
- 256K/4K (Full Document): 910C = 112 GPUs (5P:1D), H20 = 80 GPUs (8P:1D); consider chunking alternatives
- General guidance: always use SP, investigate mHC-SP, tune EP by network (high EP on 910C, low EP on H20)
- Batch aggressively for decode (linear throughput scaling up to memory limit); P/D disaggregation essential for 128K+
- **Visualization**: Decision flowchart or recommendation table by use case with platform-specific configs
