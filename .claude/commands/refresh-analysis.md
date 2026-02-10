# Refresh All Analysis Data and Reports

Regenerate all analysis data, parameter search results, and update reports with new numbers.
Run this after changing any config defaults (e.g., mhc_kernel_fused, shared_expert_overlapped) or model parameters.

**Bilingual rule:** Every `*.md` report/doc/README must have a `*_zh.md` Chinese translation with identical data/tables and translated prose/headings.

## Steps

1. **Run scenario analysis** (`report/analyze_scenarios.py`): Regenerates all `report/data/*.json` files (10 files: search results, P/D ratios, op analysis, SP comparison, mHC optimization, hardware comparison, V3 comparison, KV cache scaling, attention analysis). This takes ~30s. Covers 4 combos: 8K, 32K, 128K, 256K.

```bash
python report/analyze_scenarios.py
```

2. **Run parameter search** (optional, `param_search/search.py`): Grid search across TP/EP/DP/BS/seq for 4 scenarios on 910C. Takes ~5 minutes.

```bash
python param_search/search.py
```

3. **Run parameter search analysis** (optional, `param_search/analyze.py`): Generates search_report.md from latest results.

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

### Data-to-Section Mapping

| Section | Data Source |
|---|---|
| **1. Executive Summary** | All JSON (synthesize key findings) |
| **2. Model Structure** | `configs/model_deepseekv4.json`, `v3_comparison.json` |
| **3. Bottleneck Analysis** | `op_analysis.json`, `hardware_comparison.json` |
| **4. Parameter & Scenario Optimization** | `search_results_910C.json`, `search_results_H20.json`, `pd_ratio_analysis.json`, `hardware_comparison.json` |
| **5.1 mHC Analysis** | `mhc_optimization_comparison.json`, `sp_comparison.json` |
| **5.2 Attention & KV Cache** | `kv_cache_scaling.json`, `attention_analysis.json` |
| **6. Deployment Recommendations** | `search_results_*.json`, `pd_ratio_analysis.json` |
| **7. Industry Implications** | **AI-generated narrative** — write from analysis data + web search for current industry context |
| **8. Appendix** | `configs/*.json`, methodology description |

### Files to Update

- `report/report_en.md` ↔ `report/report_zh.md` — Main analysis report (8-section structure)
- `report/ppt_outline_en.md` ↔ `report/ppt_outline_zh.md` — PPT outline (restructured to match 8-section)
- `README.md` ↔ `README_zh.md` — Project README
- `param_search/report.md` ↔ `param_search/report_zh.md` — Parameter search report
- `CLAUDE.md` — Parameter search key results (no Chinese mirror, internal instructions)

### Section 7 Instructions

Section 7 (Industry Implications) is an AI-generated narrative. When refreshing:
1. Read all analysis JSON data files for quantitative context
2. Use web search for current industry trends in LLM inference, MoE serving, KV cache management
3. Write 6 subsections: KV cache management, P/D disaggregation, hardware tradeoffs, network for MoE, mHC lessons, ultra-long context challenges
4. Ground all claims in specific numbers from the analysis

6. **Verification checklist**:
- [ ] `python report/analyze_scenarios.py` completes without errors
- [ ] All 10 JSON files exist in `report/data/` with 4 combos (8K, 32K, 128K, 256K)
- [ ] Report sections reference correct data and numbers match JSON
- [ ] All 4 combos (8K, 32K, 128K, 256K) appear in Sections 4 and 6
- [ ] V3 comparison table present in Section 2 (from `v3_comparison.json`)
- [ ] KV cache scaling data present in Section 5.2 (from `kv_cache_scaling.json`)
- [ ] Attention analysis data present in Section 5.2 (from `attention_analysis.json`)
- [ ] PPT outlines restructured to match 8-section layout
- [ ] Chinese translations match English reports (same data/tables, translated prose)

## What Changes in Reports

When config defaults change, the following report sections need updating:
- Section 1 (executive summary key findings)
- Section 3 (op breakdown tables for all combos)
- Section 4 (search result tables, P/D ratios, hardware comparison)
- Section 5.1 (mHC optimization levels, SP comparison)
- Section 5.2 (KV cache scaling, attention analysis)
- Section 6 (deployment recommendation tables with GPU counts)
- PPT outlines (all slide data)
- README key results tables
- CLAUDE.md parameter search results
