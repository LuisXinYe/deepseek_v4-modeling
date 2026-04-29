# Roofline Phase Utilization

## Goal Description

Add three new `HardwareConfig` fields (`prefill_utilization`, `decode_utilization`, `vec_static_latency_us`) to the perf model, update the roofline compute formula from `max(cube, vec, mem)` to `max(cube+vec, mem)`, and apply phase-specific utilization multipliers to all compute and HBM timings in prefill and decode paths. The updated `configs/device_910C.json` (which already contains the new fields) must load without error. Quantization recomputation must use the same formula. All existing report columns must remain valid.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: `HardwareConfig` accepts `prefill_utilization`, `decode_utilization`, and `vec_static_latency_us` fields with sensible defaults. `configs/device_910C.json` loads without error. Hardware JSONs that omit the new fields load successfully using the defaults.
  - Positive Tests (expected to PASS):
    - `Config.from_json(device_910C.json, ...)` returns without raising `TypeError: unexpected keyword argument`
    - `HardwareConfig()` constructed without any new fields has `prefill_utilization > 0`, `decode_utilization > 0`, `vec_static_latency_us >= 0`
    - A minimal hardware JSON (only mandatory fields) loads and the new fields resolve to positive defaults
  - Negative Tests (expected to FAIL when criterion is violated):
    - Constructing `HardwareConfig(prefill_utilization=0)` raises `ValueError` (zero is invalid)
    - Constructing `HardwareConfig(decode_utilization=-0.1)` raises `ValueError`

- AC-2: `roofline_time()` uses `compute_time = max(cube_time + vec_time, mem_time)` and adds `vec_static_latency_us` to `vec_time` only when `vec_ops > 0`.
  - Positive Tests (expected to PASS):
    - With known inputs where `cube + vec < mem`: `total_time` equals `mem_time + comm_time`
    - With `vec_ops=0` and `vec_static_latency_us=10.0`: `vec_time == 0.0` (static term not added)
    - With `vec_ops > 0` and `vec_static_latency_us=10.0`: `vec_time` includes `10e-6` seconds
    - A compute-heavy op with `cube + vec > mem` produces `total_time > mem_time`
  - Negative Tests (expected to FAIL when criterion is violated):
    - Old formula `max(cube, vec, mem)` does NOT produce the same result as `max(cube+vec, mem)` when both cube and vec are non-zero and their sum exceeds mem
    - `vec_static_latency_us` is NOT added when `vec_ops == 0`

- AC-3: Prefill and decode model functions apply distinct phase utilization multipliers to all non-communication ops (including embedding, final RMSNorm, LM head, and all layer ops). Communication ops remain unaffected by phase scaling. Direct calls to `roofline_time()` without a phase context are unchanged.
  - Positive Tests (expected to PASS):
    - `prefill_model(cfg)` with `prefill_utilization=0.9` produces longer per-op times than the same cfg with `prefill_utilization=1.0`
    - `decode_step(S, cfg)` with `decode_utilization=0.6` produces longer per-op times than with `decode_utilization=1.0`
    - A communication op within a prefill layer has the same `time_s` regardless of `prefill_utilization`
    - A direct `roofline_time(...)` call (no phase context) produces times identical to `phase_util=1.0`
  - Negative Tests (expected to FAIL when criterion is violated):
    - Prefill and decode total times are NOT equal when `prefill_utilization != decode_utilization` and all other settings are equal
    - `decode_step()` does NOT return unscaled times when `decode_utilization=0.5`

- AC-4: Bottleneck labels use the new `compute_side = cube_time + vec_time` rule in both `roofline_time()` and `sum_ops()`.
  - Positive Tests (expected to PASS):
    - `cube+vec > mem`: bottleneck is `CUBE` when `cube >= vec`, or `VEC` when `vec > cube`
    - `mem > cube+vec`: bottleneck is `MEM`
    - `comm > compute_time`: bottleneck is `COMM`
    - `flops=vec_ops=mem_bytes=comm=0`: bottleneck is `""`
    - `sum_ops()` over a list of ops with `cube_time_s + vec_time_s > mem_time_s` produces `CUBE` or `VEC` aggregate bottleneck
    - `sum_ops()` aggregate bottleneck is `MEM` when aggregate `mem_time_s > cube_time_s + vec_time_s`
  - Negative Tests (expected to FAIL when criterion is violated):
    - Old argmax rule over `{cube, vec, mem}` does NOT correctly classify a case where `cube+vec > mem > cube` (old rule would say MEM; new rule says CUBE/VEC)

- AC-5: `_with_roofline_timings()` in `quantization.py` uses the same `max(cube+vec, mem)` formula as `roofline_time()`, and `quantize_phase_profile()` applies the same phase context as the base timing path.
  - Positive Tests (expected to PASS):
    - A quantized op recomputed via `_with_roofline_timings()` with identical parameters to `roofline_time()` produces the same `time_s`
    - `quantize_phase_profile(prefill_profile, cfg)` applies `prefill_utilization` when recomputing timings
    - `quantize_phase_profile(decode_profile, cfg)` applies `decode_utilization` when recomputing timings
  - Negative Tests (expected to FAIL when criterion is violated):
    - `_with_roofline_timings()` does NOT use old `max(cube, vec, mem)` formula after update

- AC-6: Full test suite passes with no regressions.
  - Positive Tests (expected to PASS):
    - `python -m unittest test.test_config test.test_roofline test.test_quantization` exits 0
    - `python -m unittest` (full suite including test_layers, test_ops, test_integration) exits 0
  - Negative Tests (expected to FAIL when criterion is violated):
    - No test previously asserting old `max(cube, vec, mem)` behavior silently passes due to an unchanged code path

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)

The implementation adds the three HardwareConfig fields, adds a shared `_compute_roofline()` helper used by both `roofline_time()` and `_with_roofline_timings()` to eliminate formula duplication, applies phase context via `Config.for_phase()` at `prefill_model()` and `decode_step()` entry, updates `sum_ops()` aggregate bottleneck, updates `quantize_phase_profile()` with phase-aware context and an explicit override parameter, adds targeted tests for all new behaviors and bottleneck scenarios, and validates that new utilization fields are strictly positive in `HardwareConfig.__post_init__()`.

### Lower Bound (Minimum Acceptable Scope)

The implementation adds the three HardwareConfig fields, updates `roofline_time()` formula inline, applies phase context in `prefill_model()` and `decode_step()`, updates `_with_roofline_timings()` formula, updates `sum_ops()` bottleneck rule, and adds or updates tests in `test_config.py`, `test_roofline.py`, and `test_quantization.py` to verify all AC scenarios.

### Allowed Choices

- Can use: `dataclasses.replace()` for phase-specific config cloning; `Config.for_phase(phase)` method; extracting a shared `_compute_roofline()` helper in `roofline.py`
- Cannot use: adding a `phase` parameter to the 37 op functions in `ops.py` (too large a blast radius); global state for phase; threading phase through `prefill_layer()` / `decode_layer()` instead of model-level entry
- Fixed by draft: formula `max(cube+vec, mem) + comm`; bottleneck precedence COMM > MEM > CUBE/VEC; static latency only when `vec_ops > 0`; `phase=None` → `phase_util=1.0`
- Fixed by user decision: allow `prefill_utilization` and `decode_utilization` in `(0, ∞)` (values > 1.0 are valid for calibration); default values are calibration starting points, not hard-contract values

> **Note on Deterministic Designs**: The compute formula and bottleneck precedence order are fixed by the draft. Path boundaries converge on these points.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

**HardwareConfig fields (config.py)**

```python
@dataclass
class HardwareConfig:
    # ... existing fields ...
    prefill_utilization: float = 1.0
    decode_utilization: float = 0.6
    vec_static_latency_us: float = 10.0

    def __post_init__(self):
        assert self.prefill_utilization > 0, "prefill_utilization must be positive"
        assert self.decode_utilization > 0, "decode_utilization must be positive"
```

**Config.for_phase() (config.py)**

```python
def for_phase(self, phase: str | None) -> "Config":
    """Return a copy with effective utilizations scaled by the phase multiplier."""
    if phase is None:
        return self
    factor = (self.hw.prefill_utilization if phase == "prefill"
              else self.hw.decode_utilization)
    scaled_hw = dataclasses.replace(self.hw,
        cube_utilization=self.hw.effective_cube_utilization * factor,
        vec_utilization=self.hw.effective_vec_utilization * factor,
        hbm_bw_utilization=self.hw.hbm_bw_utilization * factor,
    )
    return dataclasses.replace(self, hw=scaled_hw)
```

Note: `effective_cube_utilization` resolves the `None` fallback to `flops_utilization` before multiplication, ensuring no `None * float` error.

**roofline_time() formula update (roofline.py)**

```python
# vec_time includes static latency only when vec_ops > 0
vec_time = (vec_ops / (hw.vec_tflops * 1e12 * hw.effective_vec_utilization)
            + hw.vec_static_latency_us * 1e-6) if vec_ops > 0 else 0.0

compute_time = max(cube_time + vec_time, mem_time)
total_time = compute_time + comm_time_s

# Bottleneck
if total_time == 0:
    bottleneck = ""
elif comm_time_s > compute_time:
    bottleneck = "COMM"
elif mem_time > cube_time + vec_time:
    bottleneck = "MEM"
elif cube_time >= vec_time:
    bottleneck = "CUBE"
else:
    bottleneck = "VEC"
```

**sum_ops() aggregate bottleneck (roofline.py)**

```python
compute_side = total.cube_time_s + total.vec_time_s
if total.time_s == 0:
    bottleneck = ""
elif total.comm_time_s > max(compute_side, total.mem_time_s):
    bottleneck = "COMM"
elif total.mem_time_s > compute_side:
    bottleneck = "MEM"
elif total.cube_time_s >= total.vec_time_s:
    bottleneck = "CUBE"
else:
    bottleneck = "VEC"
```

**Phase wiring in layers.py**

```python
def prefill_model(cfg: Config) -> PhaseProfile:
    cfg = cfg.for_phase("prefill")   # scales all ops in this call tree
    # ... embedding, layer loop, final ops — all use phase-scaled hw via cfg

def decode_step(S_total: int, cfg: Config) -> PhaseProfile:
    cfg = cfg.for_phase("decode")
    # ... embedding, layer loop, final ops
```

`prefill_layer()` and `decode_layer()` receive the already-scaled `cfg` and need no changes.

**Quantization alignment (quantization.py)**

```python
def _infer_phase(phase_name: str) -> str | None:
    if phase_name == "prefill":
        return "prefill"
    if phase_name.startswith("decode"):
        return "decode"
    return None

def quantize_phase_profile(phase: PhaseProfile, cfg: Config) -> PhaseProfile:
    phase_cfg = cfg.for_phase(_infer_phase(phase.phase))
    # ... use phase_cfg throughout (existing deep-copy loop unchanged in structure)

def _with_roofline_timings(op, cfg, cube_tflops, mem_bytes, comm_time_s):
    # Updated formula: max(cube+vec, mem)
    vec_time = (op.vec_ops / (...) + cfg.hw.vec_static_latency_us * 1e-6) if op.vec_ops > 0 else 0.0
    compute_time = max(cube_time + vec_time, mem_time)
    # bottleneck uses same rule as roofline_time()
```

### Relevant References

- `perf_model/config.py` — HardwareConfig dataclass; `effective_cube_utilization`, `effective_vec_utilization` properties; `Config` container
- `perf_model/roofline.py` — `roofline_time()`, `OpProfile`, `sum_ops()`
- `perf_model/layers.py` — `prefill_model()`, `decode_step()` entry points; `prefill_layer()`, `decode_layer()` called with already-scaled cfg
- `perf_model/quantization.py` — `_with_roofline_timings()`, `quantize_phase_profile()`, `PhaseProfile.phase` string patterns
- `configs/device_910C.json` — already contains `prefill_utilization`, `decode_utilization`, `vec_static_latency_us`
- `test/test_config.py`, `test/test_roofline.py`, `test/test_quantization.py` — test files to update

## Dependencies and Sequence

### Milestones

1. **Foundation**: Config + formula
   - Add three fields to `HardwareConfig` with `__post_init__` validation (> 0 for utilization fields)
   - Add `Config.for_phase()` method using `effective_*` properties
   - Update `roofline_time()` formula: `max(cube+vec, mem)`, static latency conditional on `vec_ops > 0`, new bottleneck logic
   - Update `sum_ops()` aggregate bottleneck rule

2. **Integration**: Phase wiring + quantization alignment
   - Wire `cfg.for_phase("prefill")` in `prefill_model()`; `cfg.for_phase("decode")` in `decode_step()`
   - Update `_with_roofline_timings()` formula to match `roofline_time()`
   - Update `quantize_phase_profile()` with `_infer_phase()` helper and phase context

3. **Verification**: Tests and regression
   - Update `test_config.py`: field defaults (positive, non-zero), `__post_init__` validation, JSON loading
   - Update `test_roofline.py`: new formula, static latency conditions, phase multipliers, all bottleneck scenarios, `sum_ops()` aggregate
   - Update `test_quantization.py`: formula alignment, phase-aware quantization round-trip
   - Run full `python -m unittest` suite; fix any regressions

## Task Breakdown

Each task includes exactly one routing tag.

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | Add `prefill_utilization`, `decode_utilization`, `vec_static_latency_us` to `HardwareConfig`; add `__post_init__` positive-value assertions; verify `configs/device_910C.json` loads | AC-1 | coding | - |
| task2 | Add `Config.for_phase(phase)` using `effective_*` properties; update `roofline_time()` formula and bottleneck; update `sum_ops()` aggregate bottleneck | AC-2, AC-4 | coding | task1 |
| task3 | Wire `cfg = cfg.for_phase("prefill")` in `prefill_model()` and `cfg = cfg.for_phase("decode")` in `decode_step()` | AC-3 | coding | task2 |
| task4 | Update `_with_roofline_timings()` formula; add `_infer_phase()` helper; update `quantize_phase_profile()` phase context | AC-5 | coding | task2, task3 |
| task5 | Update `test_config.py`: defaults (positive, non-zero), validation, JSON loading with and without new fields | AC-1, AC-6 | coding | task1 |
| task6 | Update `test_roofline.py`: new formula, static latency conditions, phase multipliers, all bottleneck scenarios (`CUBE`, `VEC`, `MEM`, `COMM`, empty), `sum_ops()` aggregate bottleneck | AC-2, AC-3, AC-4, AC-6 | coding | task2, task3 |
| task7 | Update `test_quantization.py`: formula alignment, phase-aware quantize_phase_profile round-trip for prefill and decode | AC-5, AC-6 | coding | task4 |
| task8 | Run `python -m unittest` full suite; fix regressions in test_layers, test_ops, test_integration | AC-6 | coding | task5, task6, task7 |

## Claude-Codex Deliberation

### Agreements

- Adding the three fields to HardwareConfig with defaults is the correct entry point; existing hardware JSONs without them load cleanly
- Keeping all 37 op functions in `ops.py` unchanged is a valid constraint; phase context is injected above them
- `vec_static_latency_us` must not be added when `vec_ops == 0`
- Quantization recomputation in `_with_roofline_timings()` must match the new `max(cube+vec, mem)` formula
- Test scope must extend to `test_layers`, `test_ops`, and `test_integration` beyond the three named files

### Resolved Disagreements

- **Phase application scope**: Codex required phase scaling to cover extra ops (embedding, final RMSNorm, LM head) beyond layer ops. Adopted: apply `cfg.for_phase()` in `prefill_model()` and `decode_step()` (model-level entry), NOT in `prefill_layer()` / `decode_layer()`. Confirmed by user.
- **for_phase() with None fields**: Codex identified that directly multiplying `cube_utilization` (which can be `None`) was unsafe. Resolved: use `effective_cube_utilization` and `effective_vec_utilization` (properties that resolve None fallbacks) before multiplying.
- **Phase inference brittleness**: Codex flagged string parsing of `PhaseProfile.phase`. Addressed: isolated in a single `_infer_phase()` helper using `phase_name == "prefill"` and `phase_name.startswith("decode")`.
- **Static latency phase-scaling**: Draft formula shows `vec_static_latency` outside the phase_util factor — it is intentionally phase-independent.

### Convergence Status

- Final Status: `converged`
- Rounds executed: 1 (Codex first-pass + second-pass review)

## Pending User Decisions

- DEC-1: Validation range for `prefill_utilization` and `decode_utilization`
  - Claude Position: Allow (0, ∞) — values above 1.0 are valid for calibration/over-clocking scenarios; only reject zero or negative
  - Codex Position: Constrain to (0, 1] — these model utilization fractions and values > 1.0 are physically meaningless
  - Tradeoff Summary: Upper bound of 1.0 is stricter but may break legitimate calibration workflows; allowing (0, ∞) is looser but more consistent with the draft's "multiplier" framing
  - Decision Status: **Allow (0, ∞)** — validate strictly positive, no upper bound

## Implementation Notes

### Code Style Requirements

- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead
- The word "phase" is a legitimate domain term in this codebase (prefill/decode phases); use it freely in code where appropriate

--- Original Design Draft Start ---

# Roofline Phase Utilization Design

## Context

The hardware config needs phase-specific efficiency knobs for prefill and decode, plus a static vector latency term. The current production `configs/device_910C.json` already demonstrates the desired fields:

- `prefill_utilization`
- `decode_utilization`
- `vec_static_latency_us`

Current code does not yet accept these keys, so loading the updated `device_910C.json` fails with an unexpected keyword error. The existing roofline formula is also still `max(cube_time, vec_time, mem_time) + comm_time`.

## Requirements

- Add hardware config fields:
  - `prefill_utilization`, default `1.0`
  - `decode_utilization`, default `0.6`
  - `vec_static_latency_us`, default `10.0`
- Interpret prefill/decode utilization as an additional effective utilization multiplier for all non-communication operator compute and HBM components in that phase.
- Add `vec_static_latency_us` to `vec_time_s` only when `vec_ops > 0`.
- Change per-op compute time from `max(cube_time, vec_time, mem_time)` to `max(cube_time + vec_time, mem_time)`.
- Keep communication time additive: `time_s = compute_time + comm_time_s`.
- Keep existing bottleneck labels (`CUBE`, `VEC`, `MEM`, `COMM`, empty) for report compatibility.
- Keep non-phase direct calls to `roofline_time()` backward compatible.

## Design

`HardwareConfig` will own the new fields and defaults. Existing hardware configs that omit the new keys remain valid.

`roofline_time()` will accept an optional `phase` argument:

- `phase="prefill"` uses `hw.prefill_utilization`
- `phase="decode"` uses `hw.decode_utilization`
- `phase=None` uses `1.0` for backward-compatible direct calls and tests unless a caller opts in

The effective formulas become:

```text
cube_time = flops / (cube_tflops * 1e12 * cube_util * phase_util)
vec_time = vec_ops / (vec_tflops * 1e12 * vec_util * phase_util) + vec_static_latency
mem_time = mem_bytes / (hbm_bandwidth_gbps * 1e9 * hbm_bw_util * phase_util)
compute_time = max(cube_time + vec_time, mem_time)
total_time = compute_time + comm_time
```

The vector static latency term is only included when `vec_ops > 0`.

`prefill_layer()`, `prefill_model()`, `decode_layer()`, and `decode_step()` will call operator builders in a phase context. The lowest-risk implementation is to thread phase through `roofline_time()` calls by adding a phase parameter to op builders only where needed, or by using a small phase-aware wrapper around `Config`/hardware access. The implementation should prefer the smallest call-surface change that still makes phase explicit.

`perf_model.quantization._with_roofline_timings()` must use the same phase-aware timing formula because quantized serving evaluation recomputes op timings instead of calling `roofline_time()` directly. `quantize_phase_profile()` can infer the phase from `PhaseProfile.phase` (`prefill`, `decode_step@...`, or `decode_total`) and pass it through.

## Bottleneck Semantics

Because cube and vector time are now serial on the compute side, bottleneck selection should compare `compute_side = cube_time + vec_time` against `mem_time`.

- If communication exceeds compute time, bottleneck is `COMM`.
- If memory exceeds `cube_time + vec_time`, bottleneck is `MEM`.
- Otherwise the op is compute-bound; report `CUBE` when `cube_time >= vec_time`, else `VEC`.

This preserves current report labels while reflecting the new compute-vs-memory decision.

`sum_ops()` should aggregate component times as before, but determine aggregate bottleneck using the same `cube + vec` versus `mem` rule.

## Testing

Update or add focused tests for:

- `HardwareConfig` defaults and JSON loading with the updated `device_910C.json`.
- `vec_static_latency_us` applies only when `vec_ops > 0`.
- Per-op total time uses `max(cube_time + vec_time, mem_time) + comm_time`.
- Prefill and decode phases apply different utilization multipliers.
- `sum_ops()` aggregate bottleneck follows the new compute-side rule.
- Quantization recomputation follows the same timing formula and preserves aggregate decode handling.

Run at least:

```bash
python -m unittest test.test_config test.test_roofline test.test_quantization
python -m unittest
```

## Compatibility

The public report fields remain unchanged. Existing CSV columns (`cube_time_ms`, `vec_time_ms`, `mem_time_ms`, `total_time_ms`, `bottleneck`) remain valid, but `total_time_ms` will no longer equal `max(cube, vec, mem) + comm`; it will equal `max(cube + vec, mem) + comm`.

--- Original Design Draft End ---
