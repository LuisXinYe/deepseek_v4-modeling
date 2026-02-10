# DeepSeek V4 Inference Performance Analysis — PPT Outline

---

## Slide 1: Title Slide

- **Title**: DeepSeek V4 Inference Performance Analysis: Ascend 910C vs NVIDIA H20
- Roofline-based performance modeling across four serving scenarios
- Platforms: Ascend 910C, NVIDIA H20
- Scenarios: 8K/4K, 32K/4K, 128K/4K, 256K/4K (input/output token lengths)
- Structure: Architecture, Bottlenecks, Optimization, Module Deep-Dives, Deployment
- **Visualization**: Title text with platform logos and scenario icons

---

## Slide 2: Executive Summary — Key Findings

- mHC kernel fusion (default) reduces prefill 3--4x; mHC drops from 84% to 36% of prefill on 910C at 8K
- 910C achieves near-parity on prefill throughput — gap narrowed from 2.57x to 1.12x at 8K after fusion
- Decode is MoE-weight-bound on 910C (40--56%) and communication-bound on H20 at short context (54%)
- V4's KV compression saves 4.6x memory vs V3 — enabling practical long-context serving
- P/D disaggregation ratios: 1P:1D (8K), 1P:1D (32K), 2P:1D (128K), 3P:1D (256K) on 910C
- 910C's EP bandwidth advantage (7.84x) enables high-EP MoE configs prohibitive on H20
- **Visualization**: Numbered highlight boxes or icon grid summarizing six key findings

---

## Slide 3: DeepSeek V4 Architecture Overview

- 43 transformer layers: 2 full-attention (ratio=1), 21 C4A (4x compression), 20 C128A (128x compression)
- MQA: 64 Q heads, 1 KV head, head_dim=512, Q LoRA rank=1,024
- MoE: 256 routed experts (top-6 routing), 1 shared expert, inter_dim=2,048
- mHC (Hyper Connection): FP32 pre/post projections + Sinkhorn normalization at every sub-layer
- Lightning Index: 64 heads, dim=128, topK=512 for compressed KV entry selection
- KV Compression: group projections compress K and V caches by ratio (C4A=4x, C128A=128x)
- **Visualization**: Architecture block diagram showing layer types, MoE routing, mHC data flow, and KV compression

---

## Slide 4: V4 vs V3 Comparison

- Total params: V4 ~286B vs V3 ~704B (V4 is 2.5x smaller)
- Hidden size: V4 = 4,096 vs V3 = 7,168; Layers: V4 = 43 vs V3 = 61
- KV approach: V4 = MQA + KV compression (C4A/C128A) vs V3 = MLA (kv_lora_rank=512)
- KV cache per token: V4 = 15,168 bytes vs V3 = 70,272 bytes — **4.6x savings**
- V4 adds mHC (Hyper Connection) and Lightning Index; V3 has neither
- V4 trades larger hidden dim for aggressive compression — enabling 128K+ context at reasonable batch
- **Visualization**: Side-by-side comparison table with arrows highlighting key architectural differences

---

## Slide 5: Hardware Platforms — 910C vs H20

- Cube TFLOPS (BF16): 910C = 376, H20 = 148 (910C 2.54x advantage)
- HBM Bandwidth: 910C = 1,800 GB/s, H20 = 4,000 GB/s (H20 2.22x advantage)
- EP Bandwidth: 910C = 392 GB/s, H20 = 50 GB/s (910C 7.84x advantage)
- HBM Capacity: 910C = 64 GB, H20 = 96 GB (H20 1.5x advantage)
- Cube:Vec ratio: 910C = 15.7:1 (imbalanced), H20 = 3.4:1 (more balanced)
- **Visualization**: Side-by-side comparison table with color-coded advantage indicators (green = winner per metric)

---

## Slide 6: Per-Category Bottleneck Summary

- Prefill is compute-bound (CUBE): attention projections, MoE expert matmuls are CUBE-limited
- Decode is memory-bound (MEM/COMM): MoE weight loading and KV cache reads dominate
- mHC: MEM-bound in both phases (fused kernel reduces absolute time dramatically)
- Lightning Index: COMM-bound (score AllReduce across TP ranks)
- Communication: AllToAll for MoE dispatch/combine, AllReduce for attention, AllGather for SP
- **Visualization**: Table with per-category bottleneck indicator (CUBE / VEC / MEM / COMM) color-coded for prefill and decode

---

## Slide 7: Prefill Op Breakdown — 910C (Fused Baseline)

- 8K: mHC = 36.3%, Attn Proj = 25.4%, Comm = 18.1%, Attn Compute = 7.8%, Lightning Index = 4.8%
- 32K: mHC = 29.4%, Attn Proj = 20.6%, Attn Compute = 16.2%, Comm = 14.6%, Lightning Index = 13.1%
- 128K: Attn Compute = 31.6%, Lightning Index = 28.2%, mHC = 16.7%, Attn Proj = 11.7%
- 256K: Attn Compute = 39.0%, Lightning Index = 35.5%, mHC = 10.6%, Attn Proj = 7.4%
- Trend: mHC dominates at short context; attention compute grows quadratically and dominates at 128K+
- **Visualization**: Stacked bar chart showing category % breakdown across 8K/32K/128K/256K

---

## Slide 8: Decode Op Breakdown — 910C (Fused Baseline)

- 8K: MoE Routed = 55.9%, Attn Compute = 14.5%, Comm = 12.9%, Lightning Index = 5.7%
- 32K: MoE Routed = 38.1%, Attn Compute = 25.2%, Comm = 14.4%, Lightning Index = 11.3%
- 128K: MoE Routed = 40.3%, Attn Compute = 22.0%, Lightning Index = 16.4%, Comm = 12.4%
- 256K: MoE Routed = 41.1%, Attn Compute = 21.4%, Lightning Index = 16.7%, Comm = 12.2%
- Decode dominated by MoE weight loading (38--56%); at longer sequences attention/index KV reads grow
- **Visualization**: Stacked bar chart showing decode category % breakdown across 8K/32K/128K/256K

---

## Slide 9: 910C vs H20 Bottleneck Comparison

- Prefill 8K: 910C dominated by mHC (36.3%) vs H20 by Attn Proj (36.0%) and Comm (26.6%)
- Root cause: 910C lower HBM BW makes MEM-bound mHC worse; H20 lower cube TFLOPS shifts to CUBE ops
- Decode 8K: 910C dominated by MoE Routed (55.9%) vs H20 by Communication (54.2%)
- Root cause: 910C lower HBM BW makes weight loading dominant; H20 50 GB/s EP makes AllToAll dominant
- At 128K+ both platforms converge to MoE-weight-bound decode
- **Visualization**: Paired pie charts or stacked bars comparing 910C vs H20 bottleneck profiles for prefill and decode

---

## Slide 10: Optimal Configs — 8K/4K and 32K/4K

- 910C 8K: Prefill 1,656 tps/gpu (TP=8,EP=16,DP=2,BS=256), Decode 181 tps/gpu (TP=4,EP=16,DP=4,BS=512)
- H20 8K: Prefill 1,848 tps/gpu (TP=8,EP=8,DP=1,BS=128), Decode 252 tps/gpu (TP=8,EP=16,DP=2,BS=512)
- 910C 32K: Prefill 1,340 tps/gpu (TP=8,EP=16,DP=2,BS=64), Decode 62.2 tps/gpu (TP=4,EP=32,DP=8,BS=512)
- H20 32K: Prefill 1,463 tps/gpu (TP=8,EP=8,DP=1,BS=32), Decode 137 tps/gpu (TP=4,EP=16,DP=4,BS=256)
- Prefill throughput gap: H20 only 1.12x (8K), 1.09x (32K) higher — near parity after fusion
- **Visualization**: Grouped bar chart comparing 910C vs H20 prefill and decode throughput for 8K and 32K

---

## Slide 11: Optimal Configs — 128K/4K and 256K/4K

- 910C 128K: Prefill 760 tps/gpu (16 GPUs), Decode 16.7 tps/gpu (32 GPUs); H20: 798 / 47.4 tps/gpu
- 910C 256K: Prefill 482 tps/gpu (16 GPUs), Decode 8.5 tps/gpu (32 GPUs); H20: 497 / 25.6 tps/gpu
- Max batch per rank drops to single digits at 128K+ due to KV cache memory pressure
- H20 decode advantage widens: 2.84x at 128K, 3.01x at 256K (HBM BW dominates decode)
- Prefill throughput nearly equal: H20 only 1.05x (128K), 1.03x (256K) — cube TFLOPS dominates at long context
- **Visualization**: Grouped bar chart with 910C vs H20 metrics for 128K/256K; annotation for memory constraints

---

## Slide 12: Latency Scaling Across Scenarios

- Prefill latency: 330ms -> 1,535ms -> 10,747ms -> 33,923ms (8K -> 32K -> 128K -> 256K) on 910C
- 910C achieves 1.68--1.94x lower prefill latency vs H20 across all sequences (cube TFLOPS advantage)
- Decode latency remarkably stable: 19.3 -> 19.4 -> 21.0 -> 21.6 ms/step on 910C (KV compression caps cost)
- H20 decode latency: 9.0 -> 9.1 -> 9.9 -> 10.1 ms/step — consistently ~2x faster (HBM BW advantage)
- Decode throughput drops dramatically: 181 -> 62.2 -> 16.7 -> 8.5 tps/gpu on 910C (8K -> 256K)
- **Visualization**: Dual-axis line chart — prefill latency and decode throughput vs sequence length for both platforms

---

## Slide 13: P/D Disaggregation Ratios

- Formula: N_p/N_d >= (D_tps_instance x input_len) / (P_tps_instance x output_len)
- 910C: 1P:1D (8K, 32 GPUs), 1P:1D (32K, 48 GPUs), 2P:1D (128K, 64 GPUs), 3P:1D (256K, 80 GPUs)
- H20: 1P:1D (8K, 24 GPUs), 2P:1D (32K, 32 GPUs), 4P:1D (128K, 48 GPUs), 14P:1D (256K, 144 GPUs)
- At 256K, 910C needs 80 GPUs vs H20's 144 GPUs — 910C 44% fewer GPUs due to H20's extreme P:D ratio
- Kernel fusion improved 910C P:D ratios significantly (128K: was 4:1, now 2:1)
- **Visualization**: Stacked bar chart showing prefill GPUs + decode GPUs per scenario/platform; table with P:D ratios

---

## Slide 14: mHC Optimization — Four Levels

- Unfused FP32: 10,144ms prefill, mHC = 84.3% (original baseline)
- Fused FP32 (default): 2,492ms, mHC = 36.0% — 4.07x speedup, fusion eliminates HBM round-trips
- Fused FP32 + SP: 1,706ms, mHC = 6.6% — 5.95x speedup, mHC parallelized across TP ranks
- Fused BF16 + SP: 1,650ms, mHC = 3.4% — 6.15x total speedup
- After full optimization, bottleneck migrates to Attn Proj (38.1%) and Comm (28.0%) — CUBE territory
- **Visualization**: Waterfall chart showing mHC optimization stages with time reduction and mHC % at each level

---

## Slide 15: mHC Bottleneck Migration

- Unfused: mHC = 84.3%, Attn Proj = 6.2%, Comm = 4.6% — mHC overwhelms everything
- Fused: mHC = 36.0%, Attn Proj = 25.2%, Comm = 18.6% — more balanced profile
- Fused+SP: mHC = 6.6%, Attn Proj = 36.9%, Comm = 27.1% — attention projections now dominate
- Fused BF16+SP: mHC = 3.4%, Attn Proj = 38.1%, Comm = 28.0% — mHC effectively eliminated
- Insight: fusion shifts bottleneck from MEM (mHC) to CUBE (attention) — where 910C's 2.54x advantage applies
- **Visualization**: Stacked bar chart showing category % at each optimization level, with arrow indicating bottleneck shift

---

## Slide 16: SP/mHC-SP Comparison (Fused Baseline)

- No SP -> SP+mHC-SP speedup on 910C: 2.06x (8K), 1.79x (32K), 1.39x (128K), 1.23x (256K)
- No SP -> SP+mHC-SP speedup on H20: 2.90x (8K), 2.48x (32K), 1.78x (128K), 1.48x (256K)
- SP alone: H20 benefits more (2.67x at 8K) because it reduces expensive EP AllToAll (50 GB/s)
- SP alone: 910C benefits less (1.41x at 8K) because EP bandwidth already fast (392 GB/s)
- Speedup diminishes at long context because quadratic attention compute is unaffected by SP/mHC-SP
- **Visualization**: Grouped bar chart showing prefill time (ms) for No SP / SP / SP+mHC-SP across 8K/32K/128K/256K

---

## Slide 17: Attention & KV Cache Analysis

- V4 KV cache at 128K: 2.18 GB vs V3 MLA: 9.21 GB vs uncompressed: 11.54 GB — 4.2x savings vs V3
- Per-layer breakdown: C4A layers = 71--73% of KV cache, 2 full-attention layers = 23--25% (despite being only 2/43)
- Decode latency stable across context: 17.9ms (1K) to 21.0ms (128K) — compression caps attention cost
- Attention compute scaling: full-attn layers 65%->96% attn-dominated (8K->128K); C128A stays at 31%->54%
- Multi-ratio compression strategy validated: C128A keeps most layers efficient even at ultra-long context
- **Visualization**: Dual chart — KV cache size comparison (V4 vs V3 vs uncompressed) and decode latency vs sequence length

---

## Slide 18: Deployment Recommendations by Scenario

- 8K/4K (Chat/Coding): 910C = 32 GPUs (1P:1D), H20 = 24 GPUs (1P:1D); both efficient
- 32K/4K (Document): 910C = 48 GPUs (1P:1D), H20 = 32 GPUs (2P:1D); H20 more cost-effective
- 128K/4K (RAG/Doc QA): 910C = 64 GPUs (2P:1D), H20 = 48 GPUs (4P:1D); P/D disaggregation essential
- 256K/4K (Full Document): 910C = 80 GPUs (3P:1D), H20 = 144 GPUs (14P:1D); 910C more GPU-efficient
- General: enable kernel fusion + SP (default), tune EP by network, batch aggressively for decode
- **Visualization**: Decision flowchart or recommendation table by use case with platform-specific configs and GPU counts

---

## Slide 19: Industry Implications

- Software optimization changes hardware competitiveness: mHC fusion moved 910C from 2.57x behind to near parity
- KV compression is essential for long context: without it, 128K is infeasible at reasonable batch sizes
- HBM bandwidth remains the decode differentiator: for chatbot workloads, HBM BW > compute TFLOPS
- EP bandwidth critical for MoE: 910C's 7.84x advantage enables EP=64; H20 forced to EP=8
- P/D disaggregation architecture is required for 128K+ (mixed serving wastes 60--80% of resources)
- 256K serving is expensive on both platforms (34s/66s prefill latency) — chunking strategies needed
- **Visualization**: Matrix chart mapping implications to impact level (high/medium/low) and platform relevance

---

## Slide 20: Appendix — Methodology & Data Sources

- Roofline model: each op's time = max(cube, vec, mem) + comm; utilization: 50% compute, 80% HBM BW
- FLOPs: matmul [M,K]x[K,N] = MxNxKx2; BF16 = 2 bytes, FP32 = 4 bytes
- Search space: TP in {1..64}, EP in {1..256}, DP in {1..8}, BS in {1..512}; constraint (TP x DP) % EP == 0
- Comm models: AllReduce = 2(n-1)/n x vol/BW, AllToAll = (n-1)/n x vol/BW, AllGather = (n-1)/n x vol/BW
- Limitations: comm modeled as additive (no overlap with compute), flash attention memory model assumed
- Raw data: search_results, pd_ratio_analysis, op_analysis, sp_comparison, mhc_optimization, kv_cache_scaling JSONs
- **Visualization**: Roofline diagram showing compute vs memory intensity with example op placements
