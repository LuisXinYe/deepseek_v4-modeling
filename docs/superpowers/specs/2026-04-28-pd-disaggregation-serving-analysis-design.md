# PD Disaggregation Serving Analysis Design

## Purpose and Required Outputs

This design defines the modeling and reporting requirements for a DeepSeek V4 Flash PD-disaggregation analysis on Ascend 910C.

The analysis covers four serving scenarios:

| Scenario ID | Prefill input length | Decode output length |
| --- | ---: | ---: |
| `8k_1k` | 8,192 | 1,024 |
| `32k_1k` | 32,768 | 1,024 |
| `128k_1k` | 131,072 | 1,024 |
| `1m_1k` | 1,000,000 | 1,024 |

The report default model configuration is DeepSeek V4 Flash with W8A8 weights/activation compute and KV8 cache storage on Ascend 910C.

The final report must provide:

- Prefill minimum instance card count, HBM occupancy, and selected parallel strategy for each scenario.
- Decode results for fixed instance sizes of 8, 16, 32, and 64 cards.
- Decode maximum batch under HBM capacity, ignoring TPOT.
- Decode maximum batch satisfying TPOT <= 50 ms without MTP.
- Decode maximum batch satisfying TPOT <= 50 ms with `mtp=1` and `mtp_accept_ratio=0.9`.
- Decode TPS/card curves for no-MTP and MTP=1 modes.
- Best decode instance size by maximum TPOT-constrained TPS/card.
- Prefill/Decode ratio where aggregate Prefill QPS and Decode QPS are balanced.
- A Markdown report with all analysis data, assumptions, formulas, conclusions, and figures.

All non-core report artifacts for this task must be written under `report/0428/`.

## Scope and Non-Goals

### In Scope

- Configurable `prefix_cache_hit_rate`.
- Configurable `mtp` and `mtp_accept_ratio`.
- Configurable `quant_mode` for weight and activation compute.
- Configurable `kv_cache_quant_mode` for KV cache storage and KV-related memory traffic.
- Quantized HBM estimates for weights and KV cache.
- Operation-time adjustment by coarse op kind, not by whole-phase scaling.
- Decode TPOT filtering at 50 ms.
- Fixed decode instance sizes of 8, 16, 32, and 64 cards.
- Prefill minimum-card sizing using model weights plus batch-size-1 KV cache.
- P/D ratio solving with instance-level QPS.

### Non-Goals

- Quantization and dequantization kernel time.
- Full tensor-level memory breakdown per operation.
- Fine-grained prefix-cache prefill with separate `q_len` and `ctx_len`.
- Pipeline parallelism.
- P/D KV transfer time.
- Queueing delay, admission control, and dynamic batching.
- Prefix cache residency, eviction, or sharing policies.
- Topology-aware multi-node placement.
- New communication/compute overlap assumptions beyond the existing model.
- Runtime allocator fragmentation beyond the existing HBM reserve.

## Fixed Inputs and Defaults

### Hardware and Model

Required hardware:

```text
hardware = "910C"
```

Required model:

```text
model = "deepseek-v4-flash"
```

The implementation must use the repository's existing 910C hardware and network configuration files:

- `configs/device_910C.json`
- `configs/network_910C.json`

### Runtime Fields

The runtime model must support these fields:

```python
prefix_cache_hit_rate: float = 0.0
mtp: int = 0
mtp_accept_ratio: float = 1.0
quant_mode: str = "bf16"
kv_cache_quant_mode: str = "bf16"
```

The API defaults stay BF16-compatible so existing configs preserve current behavior.

The report default configuration is:

```text
prefix_cache_hit_rate = 0.0
quant_mode = "w8a8"
kv_cache_quant_mode = "kv8"
mtp_accept_ratio = 0.9
TPOT target = 50 ms
```

Decode analysis must run both:

```text
no MTP: mtp = 0
MTP:    mtp = 1
```

### Batch Terms

`batch_size` means instance-level batch size for the full prefill or decode instance.

Per-card batch is derived as:

```text
batch_per_card = batch_size / physical_gpus
```

Per-rank batch remains the existing DP-local batch:

```text
batch_per_rank = batch_size / dp
```

The user-facing phrase "single-card batch size" in the report means `batch_per_card`. The search may operate over instance `batch_size`, but reported maximum batch values must include both `batch_size` and `batch_per_card`.

### Context Lengths for HBM

Prefill HBM capacity uses the input context:

```text
prefill_hbm_context_len = input_len
```

Decode HBM capacity uses the maximum decode context at the end of the request:

```text
decode_hbm_context_len = input_len + output_len
```

Decode step timing uses a growing context that starts at `input_len`.

## Modeling Semantics

### Prefix Cache

`prefix_cache_hit_rate` is the only canonical field name.

Definitions:

```text
L = input_len
h = prefix_cache_hit_rate
L_miss = ceil(L * (1 - h))
```

Prefill compute:

```text
prefill_time = F_prefill(L_miss)
```

Prefill HBM:

```text
prefill_hbm = quantized_weight_bytes + quantized_KV_bytes(input_len)
```

Decode HBM:

```text
decode_hbm = quantized_weight_bytes + quantized_KV_bytes(input_len + output_len)
```

Prefix cache hit reduces prefill compute only. It does not reduce HBM capacity in this report.

Boundary behavior:

- `prefix_cache_hit_rate = 0` gives `L_miss = input_len`.
- `prefix_cache_hit_rate = 1` gives `L_miss = 0`; prefill time is zero for report calculations.

### MTP

`mtp = N` means each decode forward may draft `N` extra tokens in addition to the normal next token.

`mtp_accept_ratio = a` is the average accepted fraction across MTP tokens. It is not a chain probability.

Expected committed tokens per decode forward:

```text
tokens_per_forward = 1 + mtp * mtp_accept_ratio
```

Decode forward count:

```text
decode_forward_count = ceil(output_len / tokens_per_forward)
```

For MTP timing, context advances by `tokens_per_forward` expected tokens per forward. The total decode time is estimated from the first and last modeled forward:

```text
first_context = input_len
last_context = input_len + max(0, decode_forward_count - 1) * tokens_per_forward

decode_total_time =
  decode_forward_count
  * (F_decode_step(first_context) + F_decode_step(last_context))
  / 2
```

For no-MTP, `mtp = 0`, `tokens_per_forward = 1`, and `decode_forward_count = output_len`.

TPOT:

```text
TPOT = decode_total_time / output_len
```

Decode throughput:

```text
decode_qps_instance = batch_size / decode_total_time
decode_tps_instance = batch_size * output_len / decode_total_time
decode_tps_per_card = decode_tps_instance / physical_gpus
```

MTP extra head weights and MTP-specific compute overhead are zero in this report.

### Quantization and HBM

`quant_mode` controls weight storage and GEMM compute:

```text
bf16
w8a8
```

`kv_cache_quant_mode` controls KV cache storage and KV-related memory traffic:

```text
bf16
kv8
kv4
```

The fields are independent.

HBM bytes:

```text
weight_bytes_quant =
  weight_bytes_bf16 * weight_byte_ratio(quant_mode)
  + weight_scale_overhead_bytes

kv_bytes_quant =
  kv_bytes_bf16 * kv_byte_ratio(kv_cache_quant_mode)
  + kv_scale_overhead_bytes

hbm_total = weight_bytes_quant + kv_bytes_quant
```

Byte ratios:

| Mode | Ratio |
| --- | ---: |
| `quant_mode=bf16` | 1.0 |
| `quant_mode=w8a8` | 0.5 |
| `kv_cache_quant_mode=bf16` | 1.0 |
| `kv_cache_quant_mode=kv8` | 0.5 |
| `kv_cache_quant_mode=kv4` | 0.25 |

Scale overhead defaults:

```text
weight_scale_overhead_bytes = 0
kv_scale_overhead_bytes = 0
```

The model must expose scale-overhead parameters for future sensitivity checks, but the required 0428 report uses zero scale overhead.

### Op-Kind Timing Adjustment

Quantization must not be applied as one blanket multiplier to an entire prefill or decode phase.

Each operation is assigned one coarse op kind:

```text
gemm
attention
vector
comm
other
```

Timing rules:

- `gemm`
  - `quant_mode=bf16`: use BF16 compute throughput and original `mem_bytes`.
  - `quant_mode=w8a8`: use W8A8 GEMM compute throughput and scale `mem_bytes` by `0.5`.
- `attention`
  - Compute remains BF16.
  - `kv_cache_quant_mode` scales attention `mem_bytes` with the KV byte ratio.
- `vector`
  - Compute remains BF16/vector.
  - Memory remains unscaled.
- `other`
  - Compute and memory remain unscaled unless explicitly categorized otherwise.
- `comm`
  - Communication time and bytes remain unchanged.

W8A8 GEMM throughput is configured explicitly as a hardware/runtime parameter. The 0428 report must print the configured W8A8 throughput value in its assumptions table.

Quantization and dequantization kernel time is not modeled.

## Search Semantics and Selection Rules

### Parallelism Constraints

All candidates must satisfy:

```text
physical_gpus = tp * dp
physical_gpus % ep == 0
batch_size % dp == 0
```

Model divisibility constraints:

- TP divides attention heads.
- TP divides output groups.
- EP divides routed experts.

HBM constraint:

```text
hbm_total <= usable_hbm_capacity_gb
```

### Prefill Sizing

For each scenario:

1. Search legal parallel strategies by increasing card count.
2. Use `batch_size = 1` for the minimum-card feasibility check.
3. The first card count with at least one feasible candidate is the required prefill instance size.
4. Within that card count, select the best prefill strategy.

Prefill strategy selection order:

```text
highest prefill_tps_per_card
then lowest prefill_time
then highest prefill_qps_instance
then largest HBM margin
then lower tp
then lower ep
```

Required prefill output fields:

- `scenario_id`
- `input_len`
- `output_len`
- `physical_gpus`
- `tp`
- `ep`
- `dp`
- `edp`
- `sp`
- `batch_size`
- `batch_per_card`
- `batch_per_rank`
- `weight_hbm_gb`
- `kv_hbm_gb`
- `hbm_total_gb`
- `hbm_margin_gb`
- `prefill_time_ms`
- `prefill_qps_instance`
- `prefill_tps_per_card`

### Decode Sizing

For each scenario and each decode instance size in `{8, 16, 32, 64}`:

#### HBM Maximum Batch

Find the maximum `batch_per_card` that fits HBM:

```text
max_batch_per_card_hbm =
  max(batch_size / physical_gpus)
  where hbm_total <= usable_hbm_capacity_gb
```

This ignores TPOT.

#### TPOT-Constrained Batch Without MTP

Find the maximum `batch_per_card` satisfying:

```text
TPOT_no_mtp <= 50 ms
```

If no candidate satisfies TPOT, the no-MTP TPOT-constrained result is `null`.

#### TPOT-Constrained Batch With MTP

Use:

```text
mtp = 1
mtp_accept_ratio = 0.9
tokens_per_forward = 1.9
```

Find the maximum `batch_per_card` satisfying:

```text
TPOT_mtp <= 50 ms
```

If no candidate satisfies TPOT, the MTP-constrained result is `null`.

Decode strategy selection within each fixed instance size:

```text
highest batch_per_card satisfying the relevant constraint
then highest decode_tps_per_card
then lowest TPOT
then largest HBM margin
then lower tp
then lower ep
```

Required decode output fields:

- `scenario_id`
- `input_len`
- `output_len`
- `decode_mode` (`no_mtp` or `mtp`)
- `physical_gpus`
- `tp`
- `ep`
- `dp`
- `edp`
- `batch_size`
- `batch_per_card`
- `batch_per_rank`
- `weight_hbm_gb`
- `kv_hbm_gb`
- `hbm_total_gb`
- `hbm_margin_gb`
- `max_batch_per_card_hbm`
- `max_batch_per_card_tpot`
- `decode_total_time_ms`
- `tpot_ms`
- `decode_qps_instance`
- `decode_tps_per_card`
- `is_tpot_feasible`

### Best Decode Instance

For each scenario and decode mode, choose the best decode instance size from 8, 16, 32, and 64 cards.

Selection order:

```text
highest decode_tps_per_card
then lower physical_gpus
then larger HBM margin
then lower TPOT
```

If all fixed instance sizes are TPOT-infeasible for a mode, the best decode instance for that mode is `null`.

### P/D Balance

P/D balance uses instance-level QPS:

```text
prefill_qps_instance = prefill_batch_size / prefill_time
decode_qps_instance = decode_batch_size / decode_total_time
```

Balance condition:

```text
prefill_instances * prefill_qps_instance
  ~= decode_instances * decode_qps_instance
```

Floating ratio:

```text
prefill_instances / decode_instances =
  decode_qps_instance / prefill_qps_instance
```

Integer ratio:

```text
qps_imbalance_pct <= 10%
```

The report must output P/D balance for both decode modes:

- `no_mtp`
- `mtp`

The headline recommendation uses the best TPOT-feasible MTP decode instance. If MTP is infeasible, the headline recommendation uses the best no-MTP decode instance.

Required P/D output fields:

- `scenario_id`
- `decode_mode`
- `prefill_physical_gpus`
- `decode_physical_gpus`
- `prefill_instances`
- `decode_instances`
- `pd_ratio_float`
- `pd_ratio_actual`
- `prefill_aggregate_qps`
- `decode_aggregate_qps`
- `qps_imbalance_pct`
- `total_cards`
- `is_headline_recommendation`

## Delivery Phases and Review Gates

The work must be delivered in two independently reviewable phases. Each phase has its own review, revision, and commit gate.

### Phase 1: Core Modeling Capability

Phase 1 adds the reusable modeling capabilities required by the report:

- Runtime config support for `prefix_cache_hit_rate`, `mtp`, `mtp_accept_ratio`, `quant_mode`, and `kv_cache_quant_mode`.
- Prefix-cache effective prefill length and full-context HBM semantics.
- MTP decode-time and TPOT calculations.
- Quantized weight and KV HBM calculations.
- Coarse op-kind timing adjustment for W8A8 and KV quantization.
- Shared serving metrics for prefill QPS/TPS/card, decode QPS/TPS/card, TPOT, and P/D ratio.

Phase 1 review gate:

- Verify BF16 defaults preserve existing behavior.
- Verify new modeling functions are covered by unit tests.
- Verify no report-specific generated data is included.
- Revise any review findings before committing.

Phase 1 commit:

```text
Commit 1: core modeling functions and tests
```

### Phase 2: 0428 Analysis Report

Phase 2 uses Phase 1 capabilities to generate the requested analysis:

- `report/0428/data/` artifacts.
- `report/0428/figure/` artifacts.
- `report/0428/report.md`.
- Manifest with configuration, assumptions, and generation metadata.

Phase 2 review gate:

- Verify all four scenarios are present.
- Verify each report part has data and figures.
- Verify decode curves include no-MTP and MTP=1 modes.
- Verify HBM maximum batch and TPOT-constrained batch annotations are present.
- Verify P/D ratio uses instance-level QPS.
- Revise any review findings before committing.

Phase 2 commit:

```text
Commit 2: 0428 analysis data, figures, and report
```

## Output Data Contract

All task-specific outputs live under:

```text
report/0428/
```

Required artifact directories:

```text
report/0428/data/
report/0428/figure/
```

Required final report:

```text
report/0428/report.md
```

Required data files:

```text
report/0428/data/scenario_spec.json
report/0428/data/prefill_results.json
report/0428/data/decode_results.json
report/0428/data/pd_ratio_results.json
report/0428/data/manifest.json
```

`manifest.json` must include:

- generation timestamp
- git commit if available
- hardware config path
- model config path
- runtime defaults
- quantization defaults
- MTP defaults
- TPOT target

Unavailable numeric metrics must be JSON `null`.

Markdown may display unavailable values as `N/A`.

`0` must only represent a real zero-valued metric.

Infeasible rows must include:

```text
is_feasible = false
invalid_reason = "oom" | "tpot_exceeded" | "parallelism_invalid" | "no_candidate"
```

Generated JSON must preserve full numeric precision. Markdown tables and figures may round values for readability.

## Figures and Report Requirements

The report must include at least these figures:

1. Prefill selected card count and HBM occupancy by scenario.
2. Prefill selected strategy performance by scenario.
3. Decode TPS/card versus decode instance size for no-MTP and MTP modes, one chart per scenario.
4. Decode HBM maximum batch and TPOT-constrained maximum batch annotations for each curve point.
5. P/D total-card composition by scenario and decode mode.

The report must include:

- Assumptions table.
- Configuration table.
- Formula section for prefix cache, MTP, quantization, TPOT, QPS, and P/D balance.
- Prefill analysis section.
- Decode analysis section.
- P/D balance section.
- Final conclusions and recommendations.

SVG figures are preferred.

## Validation and Infeasible Results

Validation rules:

```text
0 <= prefix_cache_hit_rate <= 1
mtp >= 0
0 <= mtp_accept_ratio <= 1
quant_mode in {"bf16", "w8a8"}
kv_cache_quant_mode in {"bf16", "kv8", "kv4"}
batch_size % dp == 0
physical_gpus = tp * dp
physical_gpus % ep == 0
```

Invalid candidates are excluded from selection.

For user-facing decode tables, fixed instance sizes with no valid candidate must still appear with `is_feasible=false` and an `invalid_reason`.

For TPOT analysis, HBM-feasible but TPOT-infeasible candidates must not be treated as OOM.

## Verification Requirements

Verification must cover:

- BF16 defaults preserve existing behavior.
- `prefix_cache_hit_rate` changes prefill compute length and does not reduce HBM.
- Prefill HBM uses `input_len`.
- Decode HBM uses `input_len + output_len`.
- `mtp=1`, `mtp_accept_ratio=0.9` gives `tokens_per_forward=1.9`.
- MTP reduces decode forward count and TPOT relative to the same batch without MTP.
- W8A8 changes GEMM compute timing and GEMM memory timing.
- W8A8 does not change communication timing.
- KV8/KV4 changes KV HBM.
- KV8/KV4 changes attention memory timing according to the configured attention approximation.
- Decode TPOT filtering returns `null` when no candidate satisfies 50 ms.
- P/D ratio uses instance QPS, not per-rank QPS.

## Repository Integration Constraints

The design should preserve the repository's existing BF16 model as the default behavior.

Existing modules to reuse:

- `perf_model/config.py` for runtime and hardware configuration.
- `perf_model/roofline.py` for operation timing.
- `perf_model/layers.py` for prefill/decode phase timing.
- `perf_model/memory.py` for baseline weight and KV memory estimates.
- `param_search/` or `report/` utilities for candidate search and report generation where useful.

Core modeling additions may live in `perf_model/` or `param_search/`. Report-specific scripts and generated artifacts must live under `report/0428/`.
