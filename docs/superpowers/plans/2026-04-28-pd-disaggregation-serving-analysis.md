# PD Disaggregation Serving Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the committed PD-disaggregation serving-analysis spec and generate the `report/0428/` Markdown report, data, and figures.

**Architecture:** Phase 1 adds reusable modeling helpers while preserving BF16 defaults. Phase 2 builds a report-specific pipeline on top of those helpers. Each phase has its own review, revision, and commit gate.

**Tech Stack:** Python stdlib, existing `perf_model` package, existing `param_search`/`report` utilities, `unittest`, JSON, Markdown, SVG.

---

## File Structure

Phase 1 core modeling:

- Modify: `perf_model/config.py`
  - Add runtime fields for MTP and quantization.
  - Add explicit W8A8 throughput support on hardware config.
- Create: `perf_model/quantization.py`
  - Quant mode validation.
  - Quantized weight/KV HBM helpers.
  - Op-kind inference and op/phase timing adjustment.
- Create: `perf_model/serving.py`
  - Prefix-cache, MTP, TPOT, QPS/TPS/card, and P/D helper functions.
- Modify: `perf_model/__init__.py`
  - Re-export stable public serving and quantization helpers used by report scripts.
- Modify: `test/helpers.py`
  - Add new config defaults to test config builder.
- Modify: `test/test_config.py`
  - Test defaults and JSON roundtrip for new fields.
- Create: `test/test_quantization.py`
  - Test HBM ratios and op-kind timing behavior.
- Create: `test/test_serving.py`
  - Test prefix cache, MTP, TPOT, and P/D metrics.

Phase 2 report pipeline:

- Create: `report/0428/script/common.py`
  - Scenario defaults, config builders, candidate utilities, JSON/SVG/Markdown helpers.
- Create: `report/0428/script/generate_report.py`
  - End-to-end generator for data, figures, and report.
- Create generated outputs:
  - `report/0428/data/scenario_spec.json`
  - `report/0428/data/prefill_results.json`
  - `report/0428/data/decode_results.json`
  - `report/0428/data/pd_ratio_results.json`
  - `report/0428/data/manifest.json`
  - `report/0428/figure/*.svg`
  - `report/0428/report.md`
- Create: `test/test_report_0428.py`
  - Smoke tests for schema and generation on a small patched search grid.

---

## Phase 1: Core Modeling Capability

### Task 1: Runtime and Hardware Config Fields

**Files:**
- Modify: `perf_model/config.py`
- Modify: `test/helpers.py`
- Modify: `test/test_config.py`

- [ ] **Step 1: Add failing config tests**

Add tests to `test/test_config.py`:

```python
def test_quant_and_mtp_defaults(self):
    rt = RuntimeConfig()
    self.assertEqual(rt.mtp, 0)
    self.assertEqual(rt.mtp_accept_ratio, 1.0)
    self.assertEqual(rt.quant_mode, "bf16")
    self.assertEqual(rt.kv_cache_quant_mode, "bf16")

def test_hardware_w8a8_default(self):
    hw = HardwareConfig()
    self.assertIsNone(hw.w8a8_tflops)
    self.assertEqual(hw.effective_w8a8_tflops, hw.cube_tflops * 2)
```

Extend `test_roundtrip_custom_json()` runtime JSON to include:

```python
"mtp": 1,
"mtp_accept_ratio": 0.9,
"quant_mode": "w8a8",
"kv_cache_quant_mode": "kv8",
```

and assert these values after load.

- [ ] **Step 2: Run config tests to verify failure**

Run:

```bash
python -m unittest test.test_config -v
```

Expected: fails because `RuntimeConfig` and `HardwareConfig` do not yet expose the new fields.

- [ ] **Step 3: Implement config fields**

Modify `HardwareConfig` in `perf_model/config.py`:

```python
    w8a8_tflops: float | None = None

    @property
    def effective_w8a8_tflops(self) -> float:
        return self.w8a8_tflops if self.w8a8_tflops is not None else self.cube_tflops * 2
```

Modify `RuntimeConfig`:

```python
    mtp: int = 0
    mtp_accept_ratio: float = 1.0
    quant_mode: str = "bf16"
    kv_cache_quant_mode: str = "bf16"
    weight_scale_overhead_bytes: float = 0.0
    kv_scale_overhead_bytes: float = 0.0
```

Add validation helper methods:

```python
    def validate_serving_fields(self) -> None:
        if self.mtp < 0:
            raise ValueError("mtp must be >= 0")
        if not 0 <= self.mtp_accept_ratio <= 1:
            raise ValueError("mtp_accept_ratio must be in [0, 1]")
        if self.quant_mode not in {"bf16", "w8a8"}:
            raise ValueError("quant_mode must be 'bf16' or 'w8a8'")
        if self.kv_cache_quant_mode not in {"bf16", "kv8", "kv4"}:
            raise ValueError("kv_cache_quant_mode must be 'bf16', 'kv8', or 'kv4'")
```

Call this validation from helper-level code in `serving.py`, not from `RuntimeConfig.__post_init__`, so existing tests that construct partial configs remain stable.

- [ ] **Step 4: Update test helper defaults**

Modify `test/helpers.py` runtime defaults:

```python
        mtp=0,
        mtp_accept_ratio=1.0,
        quant_mode="bf16",
        kv_cache_quant_mode="bf16",
        weight_scale_overhead_bytes=0.0,
        kv_scale_overhead_bytes=0.0,
```

Add hardware default:

```python
        w8a8_tflops=None,
```

- [ ] **Step 5: Run config tests**

Run:

```bash
python -m unittest test.test_config -v
```

Expected: PASS.

### Task 2: Quantization Helpers

**Files:**
- Create: `perf_model/quantization.py`
- Create: `test/test_quantization.py`
- Modify: `perf_model/__init__.py`

- [ ] **Step 1: Add failing quantization tests**

Create `test/test_quantization.py`:

```python
import unittest

from test.helpers import make_config
from perf_model.roofline import OpProfile
from perf_model.quantization import (
    infer_op_kind,
    quantized_weight_memory_per_rank,
    quantized_kv_cache_memory,
    quantize_op_profile,
)


class TestQuantization(unittest.TestCase):
    def test_weight_and_kv_ratios(self):
        bf16 = make_config(quant_mode="bf16", kv_cache_quant_mode="bf16")
        w8kv8 = make_config(quant_mode="w8a8", kv_cache_quant_mode="kv8")

        w_bf16 = quantized_weight_memory_per_rank(bf16)["total"]
        w_w8 = quantized_weight_memory_per_rank(w8kv8)["total"]
        kv_bf16 = quantized_kv_cache_memory(bf16)["total_bytes"]
        kv_kv8 = quantized_kv_cache_memory(w8kv8)["total_bytes"]

        self.assertAlmostEqual(w_w8 / w_bf16, 0.5, places=6)
        self.assertAlmostEqual(kv_kv8 / kv_bf16, 0.5, places=6)

    def test_infer_op_kind(self):
        self.assertEqual(infer_op_kind("q_proj_dq"), "gemm")
        self.assertEqual(infer_op_kind("attention_swa"), "attention")
        self.assertEqual(infer_op_kind("attn_tp_allreduce"), "comm")
        self.assertEqual(infer_op_kind("rmsnorm_attn"), "vector")

    def test_w8a8_changes_gemm_not_comm(self):
        bf16 = make_config(quant_mode="bf16")
        w8 = make_config(quant_mode="w8a8")
        op = OpProfile(
            name="q_proj_dq",
            flops=10**12,
            mem_bytes=10**9,
            cube_time_s=1.0,
            mem_time_s=1.0,
            time_s=1.0,
            bottleneck="CUBE",
        )
        q_bf16 = quantize_op_profile(op, bf16)
        q_w8 = quantize_op_profile(op, w8)
        self.assertLess(q_w8.cube_time_s, q_bf16.cube_time_s)
        self.assertLess(q_w8.mem_bytes, q_bf16.mem_bytes)

        comm = OpProfile(name="attn_tp_allreduce", comm_time_s=0.01, time_s=0.01, bottleneck="COMM")
        self.assertEqual(quantize_op_profile(comm, w8).time_s, comm.time_s)
```

- [ ] **Step 2: Run quantization tests to verify failure**

Run:

```bash
python -m unittest test.test_quantization -v
```

Expected: fails because `perf_model.quantization` does not exist.

- [ ] **Step 3: Implement `perf_model/quantization.py`**

Implement constants and validation:

```python
WEIGHT_BYTE_RATIOS = {"bf16": 1.0, "w8a8": 0.5}
KV_BYTE_RATIOS = {"bf16": 1.0, "kv8": 0.5, "kv4": 0.25}
```

Implement HBM helpers by calling existing BF16 memory functions:

```python
def quantized_weight_memory_per_rank(cfg):
    base = weight_memory_per_rank(cfg)
    ratio = WEIGHT_BYTE_RATIOS[cfg.rt.quant_mode]
    result = dict(base)
    for key, value in list(result.items()):
        if isinstance(value, (int, float)) and key not in {"n_swa_layers", "n_comp_layers"}:
            result[key] = value * ratio
    result["total"] = base["total"] * ratio + cfg.rt.weight_scale_overhead_bytes
    result["quant_mode"] = cfg.rt.quant_mode
    return result
```

Implement KV helper similarly:

```python
def quantized_kv_cache_memory(cfg):
    base = kv_cache_memory(cfg)
    ratio = KV_BYTE_RATIOS[cfg.rt.kv_cache_quant_mode]
    layers = {}
    for layer_idx, info in base["layers"].items():
        layers[layer_idx] = {
            key: value * ratio if key.endswith("bytes") or key == "bytes" else value
            for key, value in info.items()
        }
    return {
        "layers": layers,
        "total_bytes": base["total_bytes"] * ratio + cfg.rt.kv_scale_overhead_bytes,
        "kv_cache_quant_mode": cfg.rt.kv_cache_quant_mode,
    }
```

Implement `infer_op_kind()` using current op-name families:

```python
GEMM_NAMES = {
    "q_proj_dq", "q_proj_uq", "kv_proj", "wo_a", "wo_b",
    "index_iq_proj", "moe_gate", "routed_gate_proj", "routed_up_proj",
    "routed_down_proj", "shared_gate_proj", "shared_up_proj",
    "shared_down_proj", "embedding", "lm_head",
}
ATTENTION_NAMES = {"attention_swa", "attention_comp"}
COMM_NAMES = {"attn_tp_allreduce", "moe_ep_dispatch", "moe_ep_combine",
              "sp_ag_before_attn", "sp_ag_before_moe", "sp_ag_after_moe",
              "sp_ag_before_lm_head", "index_score_ar"}
VECTOR_PREFIXES = ("rmsnorm", "mhc_", "sinkhorn", "routed_silu", "shared_silu")
```

Implement `quantize_op_profile(op, cfg)` by recomputing roofline components:

```python
kind = infer_op_kind(op.name)
mem_bytes = op.mem_bytes
cube_tflops = cfg.hw.cube_tflops
if kind == "gemm" and cfg.rt.quant_mode == "w8a8":
    cube_tflops = cfg.hw.effective_w8a8_tflops
    mem_bytes *= WEIGHT_BYTE_RATIOS["w8a8"]
elif kind == "attention":
    mem_bytes *= KV_BYTE_RATIOS[cfg.rt.kv_cache_quant_mode]
elif kind == "comm":
    return replace(op)
```

Use the same roofline equation as `perf_model/roofline.py` to return a new `OpProfile`.

- [ ] **Step 4: Implement phase quantization helper**

Add:

```python
def quantize_phase_profile(phase, cfg):
    # deep-copy layer_profiles and extra_ops with quantized OpProfiles
    # recompute each layer total with sum_ops()
    # recompute phase.total_time_s
```

Preserve `phase.phase` and `phase.total_tokens`.

- [ ] **Step 5: Run quantization tests**

Run:

```bash
python -m unittest test.test_quantization -v
```

Expected: PASS.

### Task 3: Serving Metrics Helpers

**Files:**
- Create: `perf_model/serving.py`
- Create: `test/test_serving.py`
- Modify: `perf_model/__init__.py` if public re-exports are useful.

- [ ] **Step 1: Add failing serving tests**

Create `test/test_serving.py`:

```python
import unittest

from test.helpers import make_config
from perf_model.serving import (
    tokens_per_forward,
    decode_forward_count,
    make_prefill_compute_config,
    make_prefill_memory_config,
    make_decode_memory_config,
    evaluate_prefill_serving,
    evaluate_decode_serving,
    compute_pd_ratio,
)


class TestServingHelpers(unittest.TestCase):
    def test_mtp_tokens_per_forward(self):
        self.assertEqual(tokens_per_forward(0, 0.9), 1.0)
        self.assertAlmostEqual(tokens_per_forward(1, 0.9), 1.9)
        self.assertEqual(decode_forward_count(output_len=1024, mtp=1, mtp_accept_ratio=0.9), 539)

    def test_prefix_cache_compute_and_memory_lengths(self):
        cfg = make_config(seq_len=8192, input_len=8192, output_len=1024, prefix_cache_hit_rate=0.9)
        self.assertEqual(make_prefill_compute_config(cfg).rt.seq_len, 820)
        self.assertEqual(make_prefill_memory_config(cfg).rt.seq_len, 8192)
        self.assertEqual(make_decode_memory_config(cfg).rt.seq_len, 9216)

    def test_decode_mtp_improves_tpot_same_batch(self):
        base = make_config(seq_len=256, input_len=256, output_len=32, batch_size=8, dp=4, tp=2)
        no_mtp = evaluate_decode_serving(base)
        mtp_cfg = make_config(seq_len=256, input_len=256, output_len=32, batch_size=8,
                              dp=4, tp=2, mtp=1, mtp_accept_ratio=0.9)
        mtp = evaluate_decode_serving(mtp_cfg)
        self.assertLess(mtp["tpot_ms"], no_mtp["tpot_ms"])

    def test_pd_ratio_uses_instance_qps(self):
        ratio = compute_pd_ratio(10.0, 25.0, tolerance=0.0)
        self.assertEqual(ratio["prefill_instances"], 5)
        self.assertEqual(ratio["decode_instances"], 2)
```

- [ ] **Step 2: Run serving tests to verify failure**

Run:

```bash
python -m unittest test.test_serving -v
```

Expected: fails because `perf_model.serving` does not exist.

- [ ] **Step 3: Implement prefix-cache config helpers**

Implement in `perf_model/serving.py`:

```python
def make_prefill_compute_config(cfg):
    cfg.rt.validate_serving_fields()
    seq_len = cfg.rt.effective_prefill_len
    return replace(cfg, rt=replace(cfg.rt, seq_len=seq_len))

def make_prefill_memory_config(cfg):
    input_len = cfg.rt.request_input_len
    return replace(cfg, rt=replace(cfg.rt, seq_len=input_len))

def make_decode_compute_config(cfg):
    input_len = cfg.rt.decode_context_len_effective
    return replace(cfg, rt=replace(cfg.rt, seq_len=input_len))

def make_decode_memory_config(cfg):
    context_len = cfg.rt.decode_context_len_effective + cfg.rt.output_len
    return replace(cfg, rt=replace(cfg.rt, seq_len=context_len))
```

- [ ] **Step 4: Implement MTP helpers**

Implement:

```python
def tokens_per_forward(mtp: int, mtp_accept_ratio: float) -> float:
    if mtp < 0:
        raise ValueError("mtp must be >= 0")
    if not 0 <= mtp_accept_ratio <= 1:
        raise ValueError("mtp_accept_ratio must be in [0, 1]")
    return 1.0 + mtp * mtp_accept_ratio

def decode_forward_count(output_len: int, mtp: int, mtp_accept_ratio: float) -> int:
    return math.ceil(output_len / tokens_per_forward(mtp, mtp_accept_ratio))
```

- [ ] **Step 5: Implement serving metrics**

Implement `evaluate_prefill_serving(cfg)`:

```python
compute_cfg = make_prefill_compute_config(cfg)
memory_cfg = make_prefill_memory_config(cfg)
if compute_cfg.rt.seq_len == 0:
    prefill_time_s = 0.0
else:
    phase = quantize_phase_profile(prefill_model(compute_cfg), compute_cfg)
    prefill_time_s = phase.total_time_s
wm = quantized_weight_memory_per_rank(memory_cfg)
kv = quantized_kv_cache_memory(memory_cfg)
physical_gpus = cfg.rt.tp * cfg.rt.dp
```

Return metrics including HBM, `batch_per_card`, `batch_per_rank`, `prefill_time_ms`, `prefill_qps_instance`, and `prefill_tps_per_card`. For zero prefill time, use `None` for QPS/TPS to avoid division by zero.

Implement `evaluate_decode_serving(cfg)`:

```python
compute_cfg = make_decode_compute_config(cfg)
first_context = compute_cfg.rt.seq_len
num_forwards = decode_forward_count(cfg.rt.output_len, cfg.rt.mtp, cfg.rt.mtp_accept_ratio)
tpf = tokens_per_forward(cfg.rt.mtp, cfg.rt.mtp_accept_ratio)
last_context = math.ceil(first_context + max(0, num_forwards - 1) * tpf)
first_phase = quantize_phase_profile(decode_step(first_context, compute_cfg), compute_cfg)
last_phase = quantize_phase_profile(decode_step(last_context, compute_cfg), compute_cfg)
decode_total_s = num_forwards * (first_phase.total_time_s + last_phase.total_time_s) / 2
```

Return metrics including HBM from `make_decode_memory_config(cfg)`, `decode_total_time_ms`, `tpot_ms`, `decode_qps_instance`, `decode_tps_per_card`, and `tokens_per_forward`.

- [ ] **Step 6: Move or copy P/D ratio helper**

Implement `compute_pd_ratio()` in `perf_model/serving.py` with the same semantics as `report/analyze_scenarios.py`, defaulting to `tolerance=0.1`.

Keep the existing function in `report/analyze_scenarios.py` for backward compatibility. Later scripts may import the new helper.

- [ ] **Step 7: Run serving tests**

Run:

```bash
python -m unittest test.test_serving -v
```

Expected: PASS.

### Task 4: Phase 1 Integration Review, Revision, and Commit

**Files:**
- All Phase 1 files.

- [ ] **Step 1: Run focused tests**

Run:

```bash
python -m unittest test.test_config test.test_quantization test.test_serving -v
```

Expected: PASS.

- [ ] **Step 2: Run full test suite**

Run:

```bash
python -m unittest discover -s test -v
```

Expected: PASS.

- [ ] **Step 3: Review Phase 1 diff**

Run:

```bash
git diff -- perf_model test
```

Review checklist:

- BF16 defaults preserve existing behavior.
- No `report/0428/data` or generated report artifacts are included.
- Quantization is op-kind scoped, not phase-level.
- Communication ops remain unchanged under W8A8.
- Decode HBM uses `input_len + output_len`.

- [ ] **Step 4: Revise review findings**

If review finds issues, patch them and rerun:

```bash
python -m unittest discover -s test -v
```

Expected: PASS.

- [ ] **Step 5: Commit Phase 1**

Commit only Phase 1 source and tests:

```bash
git add perf_model test
git commit -m "feat: add serving quantization and mtp modeling"
```

---

## Phase 2: 0428 Analysis Report

### Task 5: Report Scenario and Candidate Pipeline

**Files:**
- Create: `report/0428/script/common.py`
- Create: `report/0428/script/generate_report.py`
- Create: `test/test_report_0428.py`

- [ ] **Step 1: Add failing report smoke tests**

Create `test/test_report_0428.py`:

```python
import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
COMMON_PATH = REPO_ROOT / "report" / "0428" / "script" / "common.py"


def load_common():
    spec = importlib.util.spec_from_file_location("report_0428_common", COMMON_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestReport0428(unittest.TestCase):
    def test_scenarios_match_spec(self):
        common = load_common()
        ids = [s["scenario_id"] for s in common.SCENARIOS]
        self.assertEqual(ids, ["8k_1k", "32k_1k", "128k_1k", "1m_1k"])
        self.assertEqual(common.REPORT_DEFAULTS["quant_mode"], "w8a8")
        self.assertEqual(common.REPORT_DEFAULTS["kv_cache_quant_mode"], "kv8")
        self.assertEqual(common.REPORT_DEFAULTS["mtp_accept_ratio"], 0.9)

    def test_null_not_zero_for_infeasible(self):
        common = load_common()
        row = common.infeasible_row("8k_1k", "no_candidate")
        self.assertFalse(row["is_feasible"])
        self.assertIsNone(row["decode_tps_per_card"])
        self.assertEqual(row["invalid_reason"], "no_candidate")
```

- [ ] **Step 2: Run smoke tests to verify failure**

Run:

```bash
python -m unittest test.test_report_0428 -v
```

Expected: fails because report helper modules do not exist.

- [ ] **Step 3: Implement report constants and helpers**

In `report/0428/script/common.py`, define:

```python
SCENARIOS = [
    {"scenario_id": "8k_1k", "input_len": 8192, "output_len": 1024},
    {"scenario_id": "32k_1k", "input_len": 32768, "output_len": 1024},
    {"scenario_id": "128k_1k", "input_len": 131072, "output_len": 1024},
    {"scenario_id": "1m_1k", "input_len": 1_000_000, "output_len": 1024},
]
DECODE_INSTANCE_SIZES = [8, 16, 32, 64]
REPORT_DEFAULTS = {
    "prefix_cache_hit_rate": 0.0,
    "quant_mode": "w8a8",
    "kv_cache_quant_mode": "kv8",
    "mtp_accept_ratio": 0.9,
    "tpot_target_ms": 50.0,
}
```

Implement `load_910c_config()`, `with_runtime_defaults()`, `is_parallel_valid()`, `write_json()`, `read_json()`, and `infeasible_row()`.

- [ ] **Step 4: Implement candidate enumeration**

In `generate_report.py`, implement reusable generators:

```python
def iter_parallel_candidates(card_count, batch_values):
    for tp in [1, 2, 4, 8, 16, 32, 64]:
        for dp in [1, 2, 4, 8, 16, 32, 64]:
            if tp * dp != card_count:
                continue
            for ep in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                for batch_size in batch_values:
                    yield tp, ep, dp, batch_size
```

Use existing divisibility constraints from `param_search/search.py`.

- [ ] **Step 5: Implement prefill selection**

Implement:

```python
def select_prefill_for_scenario(base_cfg, scenario):
    for cards in [1, 2, 4, 8, 16, 32, 64]:
        candidates = evaluate feasible batch_size=1 candidates
        if candidates:
            return best candidate under spec tie-breakers
    return infeasible row
```

Use `evaluate_prefill_serving()` from `perf_model.serving`.

- [ ] **Step 6: Implement decode selection**

Implement:

```python
def select_decode_for_scenario(base_cfg, scenario, cards, mtp):
    hbm_candidates = all HBM-feasible candidates
    max_batch_hbm = max(batch_per_card)
    tpot_candidates = [c for c in hbm_candidates if c["tpot_ms"] <= 50.0]
    if not tpot_candidates:
        return infeasible TPOT row with max_batch_hbm populated
    return best tpot candidate under spec tie-breakers
```

Run for both:

```python
mtp = 0
mtp = 1
```

- [ ] **Step 7: Implement P/D ratios**

For each scenario:

```python
for decode_mode in ["no_mtp", "mtp"]:
    use selected prefill row and selected best decode row
    ratio = compute_pd_ratio(prefill_qps_instance, decode_qps_instance, tolerance=0.1)
```

Set `is_headline_recommendation=True` for MTP if feasible, else no-MTP if feasible.

- [ ] **Step 8: Run report smoke tests**

Run:

```bash
python -m unittest test.test_report_0428 -v
```

Expected: PASS.

### Task 6: Figures and Markdown Report

**Files:**
- Modify: `report/0428/script/generate_report.py`
- Generate: `report/0428/data/*.json`
- Generate: `report/0428/figure/*.svg`
- Generate: `report/0428/report.md`

- [ ] **Step 1: Implement SVG helpers**

Implement simple stdlib SVG functions:

```python
def write_bar_chart_svg(path, title, labels, values, ylabel):
    # fixed width/height, axes, bars, labels

def write_line_chart_svg(path, title, x_labels, series, ylabel):
    # two series for no_mtp and mtp, point labels for batch annotations
```

Keep figures readable in Markdown and avoid external dependencies.

- [ ] **Step 2: Implement required figures**

Generate at least:

```text
report/0428/figure/prefill_hbm.svg
report/0428/figure/prefill_tps.svg
report/0428/figure/decode_8k_1k.svg
report/0428/figure/decode_32k_1k.svg
report/0428/figure/decode_128k_1k.svg
report/0428/figure/decode_1m_1k.svg
report/0428/figure/pd_ratio_total_cards.svg
```

- [ ] **Step 3: Implement Markdown rendering**

`report/0428/report.md` must include:

- Title and assumptions table.
- Scenario table.
- Formula section.
- Prefill section with table and figures.
- Decode section with one table per scenario and decode curves.
- P/D balance section.
- Final recommendations.

Use relative links to figures:

```markdown
![Decode 8K/1K](figure/decode_8k_1k.svg)
```

- [ ] **Step 4: Generate full report artifacts**

Run:

```bash
python report/0428/script/generate_report.py
```

Expected generated files:

```text
report/0428/data/scenario_spec.json
report/0428/data/prefill_results.json
report/0428/data/decode_results.json
report/0428/data/pd_ratio_results.json
report/0428/data/manifest.json
report/0428/figure/*.svg
report/0428/report.md
```

- [ ] **Step 5: Validate generated JSON**

Run:

```bash
python -m json.tool report/0428/data/prefill_results.json >/dev/null
python -m json.tool report/0428/data/decode_results.json >/dev/null
python -m json.tool report/0428/data/pd_ratio_results.json >/dev/null
python -m json.tool report/0428/data/manifest.json >/dev/null
```

Expected: all commands exit 0.

### Task 7: Phase 2 Review, Revision, and Commit

**Files:**
- All Phase 2 report scripts and generated artifacts.

- [ ] **Step 1: Run report generator from clean state**

Remove only generated `report/0428/data`, `report/0428/figure`, and `report/0428/report.md`, then regenerate.

Do not remove source scripts.

Run:

```bash
python report/0428/script/generate_report.py
```

Expected: all required artifacts regenerate.

- [ ] **Step 2: Run full test suite**

Run:

```bash
python -m unittest discover -s test -v
```

Expected: PASS.

- [ ] **Step 3: Review Phase 2 outputs**

Checklist:

- All four scenarios are present in `report/0428/report.md`.
- Each scenario has prefill results.
- Each scenario has decode results for 8, 16, 32, and 64 cards.
- Decode results include `no_mtp` and `mtp`.
- Decode curves annotate HBM maximum batch and TPOT-constrained batch.
- P/D ratios exist for both decode modes.
- Headline recommendation is set.
- Report states W8A8 throughput, `kv8`, `mtp_accept_ratio=0.9`, and TPOT=50ms assumptions.

- [ ] **Step 4: Revise review findings**

If any checklist item fails, patch scripts or report rendering, regenerate, and rerun:

```bash
python report/0428/script/generate_report.py
python -m unittest discover -s test -v
```

Expected: PASS and checklist complete.

- [ ] **Step 5: Commit Phase 2**

Commit only Phase 2 files:

```bash
git add report/0428 test/test_report_0428.py
git commit -m "docs: add 0428 pd disaggregation analysis report"
```

---

## Final Verification

- [ ] **Step 1: Confirm commit history**

Run:

```bash
git log --oneline -3
```

Expected:

- latest commit is Phase 2 report
- previous implementation commit is Phase 1 modeling
- earlier commit is the spec commit

- [ ] **Step 2: Confirm working tree**

Run:

```bash
git status --short --branch
```

Expected: no unexpected tracked-file changes. Existing unrelated untracked files may remain untracked.

- [ ] **Step 3: Summarize output locations**

Final response should include:

- Phase 1 commit hash.
- Phase 2 commit hash.
- Report path: `report/0428/report.md`.
- Data path: `report/0428/data/`.
- Figure path: `report/0428/figure/`.
- Test command results.
