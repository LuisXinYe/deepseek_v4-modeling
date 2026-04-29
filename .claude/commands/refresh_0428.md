# Refresh 0428 PD Report

Refresh the 0428 PD disaggregation analysis by regenerating the required data and figures, then using LLM judgment to rewrite `report/0428/report.md` as a coherent technical report.

Use this command when model/config/search logic has changed and the 0428 report needs to be brought back in sync.

---

## Scope

Primary outputs:

- `report/0428/data/*.json`
- `report/0428/figure/*.svg`
- `report/0428/report.md`

Supporting files may be updated only when required to make the refresh reproducible or verified:

- `report/0428/script/generate_report.py`
- `test/test_report_0428.py`

Do not update unrelated reports, README files, or generated PDFs unless the user explicitly asks.

---

## Phase 0: Preflight

1. Check repository status:

```bash
git status --short --branch
```

2. Identify existing local changes before writing anything:

```bash
git diff --stat
```

3. If there are unrelated dirty files, preserve them. Do not revert or overwrite user changes.

4. Confirm the current 0428 generation entry point:

```bash
python report/0428/script/generate_report.py
```

The script should regenerate the data and figure artifacts under `report/0428/`.

---

## Phase 1: Regenerate Data And Figures

Run the 0428 generator:

```bash
python report/0428/script/generate_report.py
```

Expected outputs:

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

After generation, inspect the data shape and high-level diff:

```bash
git diff --stat -- report/0428
```

---

## Phase 2: LLM Rewrite Of `report.md`

Rewrite `report/0428/report.md` with LLM analysis. Do not rely on a script to directly produce the final report prose.

Inputs to read before rewriting:

- `report/0428/data/prefill_results.json`
- `report/0428/data/decode_results.json`
- `report/0428/data/pd_ratio_results.json`
- `report/0428/data/manifest.json`
- all refreshed SVG figures under `report/0428/figure/`
- current `report/0428/report.md`
- relevant modeling code if needed, especially `perf_model/`, `param_search/`, and `report/0428/script/generate_report.py`

Required report structure:

1. Executive Summary
2. Analysis Goals
3. Methodology
4. Assumptions
5. Limitations
6. Prefill Results
7. Decode Results
8. Prefill/Decode Ratio Results
9. Conclusions
10. Appendix: Reproduction

Writing requirements:

- Write a **standalone, self-contained** technical report. A reader who has never seen a previous version should understand all conclusions from the current data alone.
- **Do not reference previous runs, previous parameter values, or version comparisons.** No phrases like "compared to 0.8", "this version brings...", "previously X, now Y". All conclusions must be derived solely from the freshly generated data.
- Use a technical report style: clear claims, data-backed reasoning, and explicit limitations.
- Preserve all required 0428 scenarios: `8K + 1K`, `32K + 1K`, `128K + 1K`, `1M + 1K`.
- Preserve all required prefix cache hit rates: `0`, `0.9`, `0.99`.
- Preserve both decode modes: No MTP and `MTP=1`.
- Explain what `HBM/TPOT` labels mean in decode figures and tables.
- Include the core assumptions for `prefix_cache_hit_rate`, `mtp`, `mtp_accept_ratio`, `quant_mode`, and `kv_cache_quant_mode`.
- Include the distinction between prefill sizing and prefill performance search: `batch_size=1` determines minimum cards, but the performance configuration searches for max TPS/card within that card count.
- Do not leave placeholders, TODOs, or unsupported quantitative claims.

---

## Phase 3: Figure Review

Review every refreshed figure before finalizing the report:

- Ensure labels do not overlap.
- Ensure no text is outside the figure viewport.
- Ensure axis labels, legends, and data labels are understandable.
- Ensure decode point labels explain `HBM/TPOT` either in the figure or nearby report text.

If SVG rendering support is unreliable, either:

- convert SVGs to PNG/JPEG for visual inspection, or
- use a library/tool that can render SVG directly.

Do not sign off on the report until the figures are visually legible.

---

## Phase 4: Verification

Run targeted tests:

```bash
python -m unittest test.test_report_0428 test.test_param_search test.test_serving -v
```

Run a whitespace check:

```bash
git diff --check
```

Inspect the final diff:

```bash
git diff --stat
git diff -- report/0428/report.md
```

If tests fail, debug before reporting completion. If failures are unrelated to the refresh, state the exact failing tests and why they are unrelated.

---

## Phase 5: Final Response

Summarize:

- data and figure regeneration command used
- report rewrite scope
- key result changes
- verification commands and outcomes
- any remaining risks or uncommitted unrelated files

Do not commit or push unless the user explicitly asks.
