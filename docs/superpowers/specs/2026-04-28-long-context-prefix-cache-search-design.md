# Long-Context Prefix-Cache Search Extension Design

## Summary

Extend the existing search and reporting workflow to support:

- `prefix_cache_hit_rate ∈ {0.0, 0.9, 0.99}` for all existing search scenarios
- long-context `seq_len = 1_000_000` in the existing search matrix
- corrected instance-level throughput semantics for `TPS/card`, QPS, and P/D sizing
- full regeneration of search artifacts and reports after the metric correction

The design intentionally avoids a large architectural rewrite. The implementation stays centered on the current `perf_model/config.py`, `param_search/search.py`, `param_search/analyze.py`, and `report/analyze_scenarios.py` structure.

## Motivation

The repository currently supports fixed-context performance modeling and parameter search, but it cannot answer the required serving-planning question for long context with prefix cache:

- prefill and decode optimal instance size
- `TPS/card`
- per-instance QPS
- P/D ratio under `M * prefill_qps = N * decode_qps`

The current throughput formulas also use per-DP-rank batch instead of instance/global batch. That is acceptable for some per-rank reasoning, but it is wrong for instance throughput, `TPS/card`, and downstream P/D sizing. Historical throughput-heavy conclusions therefore need regeneration after the fix.

## Goals

- Keep the current repo layout and entry points.
- Add `prefix_cache_hit_rate` as a first-class runtime/search dimension.
- Add `1_000_000` to the existing sequence-length search matrix.
- Preserve existing latency modeling.
- Correct throughput and QPS to instance-level semantics.
- Regenerate the full parameter-search and scenario-analysis reports with the new semantics.

## Non-Goals

- No large refactor into a new planner/scenario subsystem.
- No change to core per-op roofline formulas beyond how search constructs runtime inputs.
- No explicit HBM reduction from prefix cache hits.
- No special-case standalone script for 1M-only analysis; the capability is added to the existing project search flow.

## Product Decisions

### Prefix Cache Semantics

- `prefix_cache_hit_rate` affects prefill computation time only.
- It does **not** reduce HBM memory footprint estimation.
- Prefill compute uses an effective input length:
  - `effective_prefill_len = ceil(seq_len * (1 - prefix_cache_hit_rate))`
- Prefill logical throughput continues to use the original request length:
  - `logical_input_len = seq_len`
- Decode is unaffected by `prefix_cache_hit_rate`:
  - decode context length remains the full `seq_len`
  - decode memory estimation remains based on the full `seq_len`

### Search Scope

All existing search scenarios are extended, not replaced.

- Existing sequence lengths remain.
- Add `1_000_000` to the sequence-length grid.
- Add `prefix_cache_hit_rate ∈ {0.0, 0.9, 0.99}` as a search/report dimension for all scenarios.

This means each existing scenario expands from:

- `phase × scenario × seq_len`

to:

- `phase × scenario × seq_len × prefix_cache_hit_rate`

### Throughput Semantics

All throughput metrics are corrected to instance/global semantics:

- `prefill_tps_instance = batch_size * logical_input_len / prefill_time_s`
- `decode_tps_instance = batch_size * output_len / decode_total_time_s`
- `tps_per_gpu = tps_instance / physical_gpus`
- `prefill_qps_instance = batch_size / prefill_time_s`
- `decode_qps_instance = batch_size / decode_total_time_s`

Latency metrics are unchanged:

- `prefill_time_ms`
- `decode_first_step_ms`
- `decode_total_ms_approx`
- `decode_total_ms_exact`

### P/D Ratio Semantics

P/D planning uses instance QPS, not per-rank throughput:

- balance condition: `M * prefill_qps_instance = N * decode_qps_instance`
- therefore: `M / N = decode_qps_instance / prefill_qps_instance`

The output should include:

- floating ratio
- integer-ceiled ratio
- minimal total GPU count using the selected prefill/decode instance sizes

## Design

### 1. `perf_model/config.py`

Extend `RuntimeConfig` with small additive fields while preserving backward compatibility:

- `input_len: int | None = None`
- `decode_context_len: int | None = None`
- `prefix_cache_hit_rate: float = 0.0`

These fields do not replace existing `seq_len`; they add serving-oriented semantics used by search/report code.

Derived runtime helpers:

- `request_input_len = input_len if input_len is not None else seq_len`
- `effective_prefill_len = ceil(request_input_len * (1 - prefix_cache_hit_rate))`
- `decode_context_len_effective = decode_context_len if decode_context_len is not None else request_input_len`

Compatibility rules:

- If the new fields are absent in JSON, the old behavior remains unchanged.
- Existing configs continue to load without modification.

### 2. `param_search/search.py`

#### Search Grid

Extend the search grid:

- keep existing `TP`, `EP`, `DP`, `BATCH_SIZE`, and `seq_len` logic
- extend `SEQ_VALUES` with `1_000_000`
- add `PREFIX_CACHE_HIT_RATE_VALUES = [0.0, 0.9, 0.99]`

Each result row stores:

- `prefix_cache_hit_rate`
- `logical_input_len`
- `effective_prefill_len`
- `decode_context_len`

#### Phase-Aware Evaluation

Search keeps its current split by `phase` and `scenario`, but config construction becomes phase-aware:

- Prefill evaluation config:
  - compute uses `seq_len = effective_prefill_len`
  - logical throughput uses `logical_input_len`
- Decode evaluation config:
  - uses `seq_len = decode_context_len_effective`

Memory check remains conservative:

- weight memory and KV cache are checked using the full context length
- for prefill rows this means memory check uses `decode_context_len_effective`, not `effective_prefill_len`

#### Throughput Fix

Replace all throughput calculations that currently use `batch_size // dp` in the numerator for instance-facing metrics.

New result fields:

- `prefill_tps_instance`
- `prefill_tps_per_gpu`
- `prefill_qps_instance`
- `decode_tps_instance`
- `decode_tps_per_gpu`
- `decode_qps_instance`

Retain `per_rank_batch` only as a diagnostic/memory-related field.

#### Sorting

Scenario ordering remains simple:

- prefill latency: minimize `prefill_time_ms`
- decode latency: minimize `decode_first_step_ms`
- prefill throughput: maximize `prefill_tps_per_gpu`
- decode throughput: maximize `decode_tps_per_gpu`

#### Decode Verification

Keep the current decode exact-verification pass for top decode candidates, but update the exact throughput recomputation to the new instance-level formulas.

### 3. `param_search/analyze.py`

Update analysis and report generation to group by both:

- `seq_len`
- `prefix_cache_hit_rate`

Tables that previously showed only one row per `seq_len` now need one row per `(seq_len, prefix_cache_hit_rate)` pair.

The generated markdown report must:

- show corrected `TPS/card`
- expose `prefix_cache_hit_rate`
- distinguish `logical_input_len` and `effective_prefill_len` for prefill scenarios

### 4. `report/analyze_scenarios.py`

Extend the scenario-analysis matrix in place.

Current combo-style logic remains, but the combo space is expanded to include:

- existing sequence-length scenarios
- new `1M/4K` scenario in the default report matrix, matching the current report convention where scenario combos use `output_len = 4096`
- all three prefix-cache hit rates

For each `(hardware, seq_len, prefix_cache_hit_rate)` tuple:

- run prefill/decode latency and throughput searches
- compute P/D ratios using instance QPS
- store corrected throughput-bearing metrics

The scenario-analysis output JSON must include:

- `prefix_cache_hit_rate`
- `prefill_qps_instance`
- `decode_qps_instance`
- corrected `tps_instance` and `tps_per_gpu`

The `search.py` extension itself should remain generic over `output_len` from runtime/config input, so the earlier `1M/1K` planning use case is still supported by the search machinery. The default regenerated report set, however, stays aligned with the existing project reporting convention and therefore adds `1M/4K`, not a separate default `1M/1K` report track.

### 5. Reports and Documentation

Regenerate all throughput-bearing artifacts together:

- `param_search/results/...`
- `param_search/report.md`
- `param_search/report_zh.md`
- scenario-analysis JSON outputs
- `report/report_en.md`
- `report/report_zh.md`
- any derived PDF outputs

Documentation updates must explicitly state:

- `TPS/card` is instance/global throughput divided by physical GPUs
- prefix cache affects prefill compute only
- memory remains based on full context length

## Data and Schema Changes

### Result Row Additions

Add the following columns where relevant:

- `prefix_cache_hit_rate`
- `logical_input_len`
- `effective_prefill_len`
- `decode_context_len`
- `prefill_tps_instance`
- `prefill_qps_instance`
- `decode_tps_instance`
- `decode_qps_instance`

### Semantic Changes to Existing Columns

These existing fields keep their names but change semantics to corrected instance-level values:

- `prefill_tps_per_gpu`
- `decode_tps_per_gpu`

If retaining legacy field names creates too much ambiguity, rename them now instead of carrying silent semantic drift. The preferred version is to add explicit instance fields and keep `*_tps_per_gpu` as the corrected per-card metric.

## Risks

### 1. Historical Throughput Values Change

Correcting throughput semantics will change historical values and may change throughput rankings, especially across rows with different `DP`.

Implications:

- old throughput tables are not directly comparable with new ones
- some historical “best throughput” winners may change
- P/D recommendations can change materially

### 2. Search Runtime Increase

Adding `1M` and three prefix-cache values expands the search space substantially.

Mitigations:

- keep existing decode approximation logic
- exact-verify only top decode candidates
- continue using aggressive early pruning via GPU-count and memory constraints

### 3. Memory/Compute Semantic Split

Using full-context memory with reduced prefill compute is intentional, but non-obvious.

Mitigations:

- make the distinction explicit in result schemas and docs
- test prefill rows to ensure compute length and memory length are not accidentally coupled

## Testing Strategy

Add or update tests for:

- runtime-config backward compatibility with missing new fields
- `effective_prefill_len` derivation for hit rates `0.0`, `0.9`, `0.99`
- memory check staying tied to full context length even when effective prefill length is smaller
- instance throughput formulas:
  - `tps_instance == batch_size * logical_len / time`
  - `tps_per_gpu == tps_instance / (tp * dp)`
  - `qps_instance == batch_size / time`
- decode exact-verification recomputation using corrected semantics
- search grouping/report generation over `(seq_len, prefix_cache_hit_rate)`
- P/D ratio using QPS rather than per-rank throughput

## Rollout Plan

1. Extend `RuntimeConfig` with additive serving fields.
2. Correct throughput formulas in `param_search/search.py`.
3. Add `prefix_cache_hit_rate` and `1_000_000` to the search matrix.
4. Update `param_search/analyze.py` for the expanded dimension set.
5. Update `report/analyze_scenarios.py` and report generation.
6. Regenerate all throughput-bearing outputs and documents.
7. Review regenerated reports for changed conclusions, especially P/D sizing and hardware comparisons.

## Open Decisions Resolved

The following decisions are fixed by user direction:

- use the current repo’s DeepSeek V4 model/config as the `dsv4 flash` model baseline
- evaluate both `910C` and `H20`
- search instance sizes in the existing project flow rather than creating a new planning subsystem
- apply `prefix_cache_hit_rate` to all existing projects, not just the new 1M scenario
- use logical `TPS/card` for prefill
- keep HBM memory estimation based on full context length
- regenerate the complete reports after the extension
