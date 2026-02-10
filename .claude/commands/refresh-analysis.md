# Refresh All Analysis Data and Reports

Regenerate all analysis data, parameter search results, and update reports with new numbers.
Run this after changing any config defaults (e.g., mhc_kernel_fused, shared_expert_overlapped) or model parameters.

**Bilingual rule:** Every `*.md` report/doc/README must have a `*_zh.md` Chinese translation with identical data/tables and translated prose/headings.

## Steps

1. **Run scenario analysis** (`report/analyze_scenarios.py`): Regenerates all `report/data/*.json` files (search results, P/D ratios, op analysis, SP comparison, mHC optimization comparison, hardware comparison). This takes ~2-5 minutes.

```bash
python report/analyze_scenarios.py
```

2. **Run parameter search** (`param_search/search.py`): Grid search across TP/EP/DP/BS/seq for 4 scenarios on 910C. Takes ~5 minutes.

```bash
python param_search/search.py
```

3. **Run parameter search analysis** (`param_search/analyze.py`): Generates search_report.md from latest results.

```bash
python param_search/analyze.py
```

Then copy the reports to the standard location:
```bash
cp param_search/results/$(ls -t param_search/results/ | head -1)/search_report.md param_search/report.md
cp param_search/results/$(ls -t param_search/results/ | head -1)/search_report_zh.md param_search/report_zh.md
```

4. **Verify main.py output**: Run main.py to verify ops and configs look correct.

```bash
python main.py configs/device_910C.json configs/network_910C.json configs/model_deepseekv4.json configs/runtime_deepseekv4.json
```

5. **Update reports with new numbers from `report/data/*.json`**:
   - `report/report_en.md` ↔ `report/report_zh.md` — Main analysis report
   - `report/ppt_outline_en.md` ↔ `report/ppt_outline_zh.md` — PPT outline
   - `README.md` ↔ `README_zh.md` — Project README
   - `param_search/report.md` ↔ `param_search/report_zh.md` — Parameter search report
   - `CLAUDE.md` — Parameter search key results (no Chinese mirror, internal instructions)

   Data sources for number updates:
   - `report/data/search_results_910C.json` / `search_results_H20.json` — best configs per scenario
   - `report/data/pd_ratio_analysis.json` — P/D disaggregation ratios
   - `report/data/op_analysis.json` — per-op bottleneck breakdown
   - `report/data/sp_comparison.json` — SP/mHC-SP comparison
   - `report/data/mhc_optimization_comparison.json` — 4 mHC optimization levels
   - `report/data/hardware_comparison.json` — cross-platform comparison

6. **Verification**: Cross-check that report numbers match JSON data files.

## What Changes in Reports

When config defaults change, the following report sections need updating:
- Executive Summary (key findings)
- Sections 2-3 (search result tables for all scenarios)
- Section 4 (P/D ratios)
- Section 5 (SP/mHC-SP comparison)
- Section 6 (mHC Kernel Fusion optimization levels)
- Sections 7-9 (op analysis, hardware comparison)
- Section 11 (deployment recommendations with GPU counts)
- PPT outlines (all slide numbers)
- README key results tables
- CLAUDE.md parameter search results
