#!/usr/bin/env python3
"""Analyze parameter search results (4 scenarios) and generate markdown report."""

import csv
import json
import os
import sys
from collections import defaultdict


def load_results(csv_path):
    """Parse CSV into list of dicts with numeric conversion."""
    rows = []
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in row:
                val = row[key]
                if val == "" or val is None:
                    continue
                if val in ("True", "False"):
                    row[key] = val == "True"
                    continue
                try:
                    if "." in val:
                        row[key] = float(val)
                    else:
                        row[key] = int(val)
                except (ValueError, TypeError):
                    pass
            rows.append(row)
    return rows


def find_latest_results():
    """Find the most recent timestamp directory in param_search/results/."""
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    if not os.path.isdir(base_dir):
        print(f"No results directory found at {base_dir}")
        sys.exit(1)
    dirs = sorted([d for d in os.listdir(base_dir)
                   if os.path.isdir(os.path.join(base_dir, d))])
    if not dirs:
        print("No result directories found")
        sys.exit(1)
    return os.path.join(base_dir, dirs[-1])


def md_table(headers, rows, alignments=None):
    """Generate a markdown table string."""
    if alignments is None:
        alignments = ["right"] * len(headers)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    sep_parts = []
    for a in alignments:
        if a == "left":
            sep_parts.append(":---")
        elif a == "center":
            sep_parts.append(":---:")
        else:
            sep_parts.append("---:")
    lines.append("| " + " | ".join(sep_parts) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def fmt_num(v, decimals=1):
    """Format a number for table display."""
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


# ---------------------------------------------------------------------------
# Per-scenario analysis
# ---------------------------------------------------------------------------

def analyze_prefill_latency(results):
    """Analyze prefill latency results."""
    analysis = {}
    if not results:
        return analysis

    analysis["best"] = results[0]

    # Best per seq_len
    best_per_seq = {}
    for r in results:
        sl = r["seq_len"]
        metric = r["prefill_time_ms"]
        if sl not in best_per_seq or metric < best_per_seq[sl]["prefill_time_ms"]:
            best_per_seq[sl] = r
    analysis["best_per_seq"] = best_per_seq

    # SP impact
    sp_impact = {}
    for sl in sorted(best_per_seq.keys()):
        sp_true = [r for r in results if r["seq_len"] == sl and r["sp"] is True]
        sp_false = [r for r in results if r["seq_len"] == sl and r["sp"] is False]
        if sp_true and sp_false:
            best_true = min(sp_true, key=lambda r: r["prefill_time_ms"])
            best_false = min(sp_false, key=lambda r: r["prefill_time_ms"])
            sp_impact[sl] = {
                "sp_true_ms": best_true["prefill_time_ms"],
                "sp_false_ms": best_false["prefill_time_ms"],
                "speedup": best_false["prefill_time_ms"] / best_true["prefill_time_ms"]
                           if best_true["prefill_time_ms"] > 0 else 0,
            }
    analysis["sp_impact"] = sp_impact
    return analysis


def analyze_decode_latency(results):
    """Analyze decode latency results."""
    analysis = {}
    if not results:
        return analysis

    analysis["best"] = results[0]

    best_per_seq = {}
    for r in results:
        sl = r["seq_len"]
        metric = r["decode_first_step_ms"]
        if sl not in best_per_seq or metric < best_per_seq[sl]["decode_first_step_ms"]:
            best_per_seq[sl] = r
    analysis["best_per_seq"] = best_per_seq

    # SP impact on decode
    sp_impact = {}
    for sl in sorted(best_per_seq.keys()):
        sp_true = [r for r in results if r["seq_len"] == sl and r["sp"] is True]
        sp_false = [r for r in results if r["seq_len"] == sl and r["sp"] is False]
        if sp_true and sp_false:
            best_true = min(sp_true, key=lambda r: r["decode_first_step_ms"])
            best_false = min(sp_false, key=lambda r: r["decode_first_step_ms"])
            sp_impact[sl] = {
                "sp_true_ms": best_true["decode_first_step_ms"],
                "sp_false_ms": best_false["decode_first_step_ms"],
                "speedup": best_false["decode_first_step_ms"] / best_true["decode_first_step_ms"]
                           if best_true["decode_first_step_ms"] > 0 else 0,
            }
    analysis["sp_impact"] = sp_impact
    return analysis


def analyze_prefill_throughput(results):
    """Analyze prefill throughput results."""
    analysis = {}
    if not results:
        return analysis

    analysis["best"] = results[0]

    # Best per GPU count
    best_per_gpu = {}
    for r in results:
        gpus = r["physical_gpus"]
        metric = r["prefill_tps_per_gpu"]
        if gpus not in best_per_gpu or metric > best_per_gpu[gpus]["prefill_tps_per_gpu"]:
            best_per_gpu[gpus] = r
    analysis["best_per_gpu"] = best_per_gpu

    # Batch scaling for best config
    if results:
        b = results[0]
        batch_scaling = [r for r in results
                         if r["tp"] == b["tp"] and r["ep"] == b["ep"]
                         and r["dp"] == b["dp"] and r["seq_len"] == b["seq_len"]]
        batch_scaling.sort(key=lambda r: r["batch_size"])
        analysis["batch_scaling"] = batch_scaling

    return analysis


def analyze_decode_throughput(results):
    """Analyze decode throughput results."""
    analysis = {}
    if not results:
        return analysis

    analysis["best"] = results[0]

    best_per_gpu = {}
    for r in results:
        gpus = r["physical_gpus"]
        metric = r["decode_tps_per_gpu"]
        if gpus not in best_per_gpu or metric > best_per_gpu[gpus]["decode_tps_per_gpu"]:
            best_per_gpu[gpus] = r
    analysis["best_per_gpu"] = best_per_gpu

    # Batch scaling for best config
    if results:
        b = results[0]
        batch_scaling = [r for r in results
                         if r["tp"] == b["tp"] and r["ep"] == b["ep"]
                         and r["dp"] == b["dp"] and r["seq_len"] == b["seq_len"]]
        batch_scaling.sort(key=lambda r: r["batch_size"])
        analysis["batch_scaling"] = batch_scaling

    # Verification summary
    verified = [r for r in results[:10] if r.get("approx_error_pct", "") != ""]
    if verified:
        errors = [float(r["approx_error_pct"]) for r in verified]
        analysis["verification"] = {
            "count": len(verified),
            "max_error": max(errors),
            "avg_error": sum(errors) / len(errors),
        }

    return analysis


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_markdown_report(results_dir, all_results, all_analyses, meta):
    """Generate a markdown report with 4 scenario sections."""
    lines = []

    lines.append("# DeepSeek V4 Parameter Search Results\n")
    lines.append(f"**Generated:** {meta.get('timestamp', 'N/A')}")
    lines.append(f"**Total search time:** {meta.get('total_time_s', 'N/A')}s")
    lines.append(f"**GPU formula:** `physical_gpus = TP * DP`, constraint `(TP*DP) % EP == 0`")
    valid = meta.get("valid_configs", {})
    for key, count in valid.items():
        lines.append(f"**{key} configs:** {count}")
    lines.append("")

    # --- Section 1: Prefill Latency ---
    pf_lat = all_results.get("prefill_latency", [])
    pf_lat_a = all_analyses.get("prefill_latency", {})
    lines.append("## 1. Prefill Latency (minimize prefill_time_ms)\n")

    if pf_lat:
        lines.append("### Top-10 Configurations\n")
        headers = ["Rank", "TP", "EP", "DP", "EDP", "BS", "SeqLen", "SP", "Overlap",
                    "GPUs", "Prefill(ms)", "HBM(GB)"]
        rows = []
        for i, r in enumerate(pf_lat[:10]):
            rows.append([
                i + 1, r["tp"], r["ep"], r["dp"], r["edp"],
                r["batch_size"], r["seq_len"], r["sp"], r["shared_expert_overlapped"],
                r["physical_gpus"], fmt_num(r["prefill_time_ms"]),
                fmt_num(r["hbm_total_gb"]),
            ])
        lines.append(md_table(headers, rows))
        lines.append("")

    if pf_lat_a.get("best_per_seq"):
        lines.append("### Best Config per Sequence Length\n")
        headers = ["SeqLen", "TP", "EP", "DP", "EDP", "BS", "SP", "GPUs", "Prefill(ms)"]
        rows = []
        for sl in sorted(pf_lat_a["best_per_seq"].keys()):
            r = pf_lat_a["best_per_seq"][sl]
            rows.append([
                sl, r["tp"], r["ep"], r["dp"], r["edp"],
                r["batch_size"], r["sp"], r["physical_gpus"],
                fmt_num(r["prefill_time_ms"]),
            ])
        lines.append(md_table(headers, rows))
        lines.append("")

    if pf_lat_a.get("sp_impact"):
        lines.append("### Sequence Parallelism Impact on Prefill\n")
        headers = ["SeqLen", "SP=True(ms)", "SP=False(ms)", "Speedup"]
        rows = []
        for sl in sorted(pf_lat_a["sp_impact"].keys()):
            imp = pf_lat_a["sp_impact"][sl]
            rows.append([
                sl, fmt_num(imp["sp_true_ms"]),
                fmt_num(imp["sp_false_ms"]),
                f"{imp['speedup']:.2f}x",
            ])
        lines.append(md_table(headers, rows))
        lines.append("")

    # --- Section 2: Decode Latency ---
    dc_lat = all_results.get("decode_latency", [])
    dc_lat_a = all_analyses.get("decode_latency", {})
    lines.append("## 2. Decode Latency (minimize decode_first_step_ms)\n")

    if dc_lat:
        lines.append("### Top-10 Configurations\n")
        headers = ["Rank", "TP", "EP", "DP", "EDP", "BS", "SeqLen", "SP", "Overlap",
                    "GPUs", "1st Step(ms)", "HBM(GB)"]
        rows = []
        for i, r in enumerate(dc_lat[:10]):
            rows.append([
                i + 1, r["tp"], r["ep"], r["dp"], r["edp"],
                r["batch_size"], r["seq_len"], r["sp"], r["shared_expert_overlapped"],
                r["physical_gpus"], fmt_num(r["decode_first_step_ms"], 3),
                fmt_num(r["hbm_total_gb"]),
            ])
        lines.append(md_table(headers, rows))
        lines.append("")

    if dc_lat_a.get("best_per_seq"):
        lines.append("### Best Config per Sequence Length\n")
        headers = ["SeqLen", "TP", "EP", "DP", "EDP", "BS", "SP", "GPUs", "1st Step(ms)"]
        rows = []
        for sl in sorted(dc_lat_a["best_per_seq"].keys()):
            r = dc_lat_a["best_per_seq"][sl]
            rows.append([
                sl, r["tp"], r["ep"], r["dp"], r["edp"],
                r["batch_size"], r["sp"], r["physical_gpus"],
                fmt_num(r["decode_first_step_ms"], 3),
            ])
        lines.append(md_table(headers, rows))
        lines.append("")

    if dc_lat_a.get("sp_impact"):
        lines.append("### Sequence Parallelism Impact on Decode\n")
        headers = ["SeqLen", "SP=True(ms)", "SP=False(ms)", "Speedup"]
        rows = []
        for sl in sorted(dc_lat_a["sp_impact"].keys()):
            imp = dc_lat_a["sp_impact"][sl]
            rows.append([
                sl, fmt_num(imp["sp_true_ms"], 3),
                fmt_num(imp["sp_false_ms"], 3),
                f"{imp['speedup']:.2f}x",
            ])
        lines.append(md_table(headers, rows))
        lines.append("")

    # --- Section 3: Prefill Throughput ---
    pf_thr = all_results.get("prefill_throughput", [])
    pf_thr_a = all_analyses.get("prefill_throughput", {})
    lines.append("## 3. Prefill Throughput (maximize prefill_tps_per_gpu)\n")

    if pf_thr:
        lines.append("### Top-10 Configurations\n")
        headers = ["Rank", "TP", "EP", "DP", "EDP", "BS", "SeqLen",
                    "GPUs", "TPS/GPU", "HBM(GB)"]
        rows = []
        for i, r in enumerate(pf_thr[:10]):
            rows.append([
                i + 1, r["tp"], r["ep"], r["dp"], r["edp"],
                r["batch_size"], r["seq_len"], r["physical_gpus"],
                fmt_num(r["prefill_tps_per_gpu"], 2),
                fmt_num(r["hbm_total_gb"]),
            ])
        lines.append(md_table(headers, rows))
        lines.append("")

    if pf_thr_a.get("best_per_gpu"):
        lines.append("### Best Config per GPU Count\n")
        headers = ["GPUs", "TP", "EP", "DP", "EDP", "BS", "SeqLen", "TPS/GPU"]
        rows = []
        for gpus in sorted(pf_thr_a["best_per_gpu"].keys()):
            r = pf_thr_a["best_per_gpu"][gpus]
            rows.append([
                gpus, r["tp"], r["ep"], r["dp"], r["edp"],
                r["batch_size"], r["seq_len"],
                fmt_num(r["prefill_tps_per_gpu"], 2),
            ])
        lines.append(md_table(headers, rows))
        lines.append("")

    if pf_thr_a.get("batch_scaling"):
        lines.append("### Batch Size Scaling (Best Config)\n")
        headers = ["BS", "TPS/GPU", "HBM(GB)"]
        rows = []
        for r in pf_thr_a["batch_scaling"]:
            rows.append([
                r["batch_size"],
                fmt_num(r["prefill_tps_per_gpu"], 2),
                fmt_num(r["hbm_total_gb"]),
            ])
        lines.append(md_table(headers, rows))
        lines.append("")

    # --- Section 4: Decode Throughput ---
    dc_thr = all_results.get("decode_throughput", [])
    dc_thr_a = all_analyses.get("decode_throughput", {})
    lines.append("## 4. Decode Throughput (maximize decode_tps_per_gpu)\n")

    if dc_thr:
        lines.append("### Top-10 Configurations\n")
        headers = ["Rank", "TP", "EP", "DP", "EDP", "BS", "SeqLen",
                    "GPUs", "TPS/GPU", "Exact TPS/GPU", "Err%", "HBM(GB)"]
        rows = []
        for i, r in enumerate(dc_thr[:10]):
            exact_tps = r.get("decode_tps_per_gpu", "")
            err = r.get("approx_error_pct", "")
            rows.append([
                i + 1, r["tp"], r["ep"], r["dp"], r["edp"],
                r["batch_size"], r["seq_len"], r["physical_gpus"],
                fmt_num(r["decode_tps_per_gpu"], 2) if isinstance(r.get("decode_tps_per_gpu"), (int, float)) else r.get("decode_tps_per_gpu", "-"),
                fmt_num(r.get("decode_tps_per_gpu"), 2) if r.get("approx_error_pct", "") != "" else "-",
                fmt_num(err) if err != "" else "-",
                fmt_num(r["hbm_total_gb"]),
            ])
        lines.append(md_table(headers, rows))
        lines.append("")

    if dc_thr_a.get("best_per_gpu"):
        lines.append("### Best Config per GPU Count\n")
        headers = ["GPUs", "TP", "EP", "DP", "EDP", "BS", "SeqLen", "TPS/GPU"]
        rows = []
        for gpus in sorted(dc_thr_a["best_per_gpu"].keys()):
            r = dc_thr_a["best_per_gpu"][gpus]
            rows.append([
                gpus, r["tp"], r["ep"], r["dp"], r["edp"],
                r["batch_size"], r["seq_len"],
                fmt_num(r["decode_tps_per_gpu"], 2),
            ])
        lines.append(md_table(headers, rows))
        lines.append("")

    if dc_thr_a.get("batch_scaling"):
        lines.append("### Batch Size Scaling (Best Config)\n")
        headers = ["BS", "TPS/GPU", "HBM(GB)"]
        rows = []
        for r in dc_thr_a["batch_scaling"]:
            rows.append([
                r["batch_size"],
                fmt_num(r["decode_tps_per_gpu"], 2),
                fmt_num(r["hbm_total_gb"]),
            ])
        lines.append(md_table(headers, rows))
        lines.append("")

    # --- Verification summary ---
    if dc_thr_a.get("verification"):
        v = dc_thr_a["verification"]
        lines.append("## Verification Summary\n")
        lines.append(f"- **Decode throughput top-10:** {v['count']} verified, "
                      f"max error = {v['max_error']:.2f}%, avg error = {v['avg_error']:.2f}%")
        lines.append("")

    report_path = os.path.join(results_dir, "search_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    return report_path


def generate_markdown_report_zh(results_dir, all_results, all_analyses, meta):
    """Generate a Chinese markdown report with 4 scenario sections."""
    lines = []

    lines.append("# DeepSeek V4 参数搜索结果\n")
    lines.append(f"**生成时间：** {meta.get('timestamp', 'N/A')}")
    lines.append(f"**总搜索时间：** {meta.get('total_time_s', 'N/A')}s")
    lines.append(f"**GPU公式：** `physical_gpus = TP * DP`，约束 `(TP*DP) % EP == 0`")
    valid = meta.get("valid_configs", {})
    for key, count in valid.items():
        lines.append(f"**{key} 配置数：** {count}")
    lines.append("")

    # --- Section 1: Prefill Latency ---
    pf_lat = all_results.get("prefill_latency", [])
    pf_lat_a = all_analyses.get("prefill_latency", {})
    lines.append("## 1. 预填充延迟（最小化 prefill_time_ms）\n")

    if pf_lat:
        lines.append("### 前10最优配置\n")
        headers = ["排名", "TP", "EP", "DP", "EDP", "BS", "序列长度", "SP", "重叠",
                    "GPU数", "预填充(ms)", "HBM(GB)"]
        rows = []
        for i, r in enumerate(pf_lat[:10]):
            rows.append([
                i + 1, r["tp"], r["ep"], r["dp"], r["edp"],
                r["batch_size"], r["seq_len"], r["sp"], r["shared_expert_overlapped"],
                r["physical_gpus"], fmt_num(r["prefill_time_ms"]),
                fmt_num(r["hbm_total_gb"]),
            ])
        lines.append(md_table(headers, rows))
        lines.append("")

    if pf_lat_a.get("best_per_seq"):
        lines.append("### 各序列长度最优配置\n")
        headers = ["序列长度", "TP", "EP", "DP", "EDP", "BS", "SP", "GPU数", "预填充(ms)"]
        rows = []
        for sl in sorted(pf_lat_a["best_per_seq"].keys()):
            r = pf_lat_a["best_per_seq"][sl]
            rows.append([
                sl, r["tp"], r["ep"], r["dp"], r["edp"],
                r["batch_size"], r["sp"], r["physical_gpus"],
                fmt_num(r["prefill_time_ms"]),
            ])
        lines.append(md_table(headers, rows))
        lines.append("")

    if pf_lat_a.get("sp_impact"):
        lines.append("### 序列并行对预填充的影响\n")
        headers = ["序列长度", "SP=True(ms)", "SP=False(ms)", "加速比"]
        rows = []
        for sl in sorted(pf_lat_a["sp_impact"].keys()):
            imp = pf_lat_a["sp_impact"][sl]
            rows.append([
                sl, fmt_num(imp["sp_true_ms"]),
                fmt_num(imp["sp_false_ms"]),
                f"{imp['speedup']:.2f}x",
            ])
        lines.append(md_table(headers, rows))
        lines.append("")

    # --- Section 2: Decode Latency ---
    dc_lat = all_results.get("decode_latency", [])
    dc_lat_a = all_analyses.get("decode_latency", {})
    lines.append("## 2. 解码延迟（最小化 decode_first_step_ms）\n")

    if dc_lat:
        lines.append("### 前10最优配置\n")
        headers = ["排名", "TP", "EP", "DP", "EDP", "BS", "序列长度", "SP", "重叠",
                    "GPU数", "首步(ms)", "HBM(GB)"]
        rows = []
        for i, r in enumerate(dc_lat[:10]):
            rows.append([
                i + 1, r["tp"], r["ep"], r["dp"], r["edp"],
                r["batch_size"], r["seq_len"], r["sp"], r["shared_expert_overlapped"],
                r["physical_gpus"], fmt_num(r["decode_first_step_ms"], 3),
                fmt_num(r["hbm_total_gb"]),
            ])
        lines.append(md_table(headers, rows))
        lines.append("")

    if dc_lat_a.get("best_per_seq"):
        lines.append("### 各序列长度最优配置\n")
        headers = ["序列长度", "TP", "EP", "DP", "EDP", "BS", "SP", "GPU数", "首步(ms)"]
        rows = []
        for sl in sorted(dc_lat_a["best_per_seq"].keys()):
            r = dc_lat_a["best_per_seq"][sl]
            rows.append([
                sl, r["tp"], r["ep"], r["dp"], r["edp"],
                r["batch_size"], r["sp"], r["physical_gpus"],
                fmt_num(r["decode_first_step_ms"], 3),
            ])
        lines.append(md_table(headers, rows))
        lines.append("")

    if dc_lat_a.get("sp_impact"):
        lines.append("### 序列并行对解码的影响\n")
        headers = ["序列长度", "SP=True(ms)", "SP=False(ms)", "加速比"]
        rows = []
        for sl in sorted(dc_lat_a["sp_impact"].keys()):
            imp = dc_lat_a["sp_impact"][sl]
            rows.append([
                sl, fmt_num(imp["sp_true_ms"], 3),
                fmt_num(imp["sp_false_ms"], 3),
                f"{imp['speedup']:.2f}x",
            ])
        lines.append(md_table(headers, rows))
        lines.append("")

    # --- Section 3: Prefill Throughput ---
    pf_thr = all_results.get("prefill_throughput", [])
    pf_thr_a = all_analyses.get("prefill_throughput", {})
    lines.append("## 3. 预填充吞吐量（最大化 prefill_tps_per_gpu）\n")

    if pf_thr:
        lines.append("### 前10最优配置\n")
        headers = ["排名", "TP", "EP", "DP", "EDP", "BS", "序列长度",
                    "GPU数", "TPS/GPU", "HBM(GB)"]
        rows = []
        for i, r in enumerate(pf_thr[:10]):
            rows.append([
                i + 1, r["tp"], r["ep"], r["dp"], r["edp"],
                r["batch_size"], r["seq_len"], r["physical_gpus"],
                fmt_num(r["prefill_tps_per_gpu"], 2),
                fmt_num(r["hbm_total_gb"]),
            ])
        lines.append(md_table(headers, rows))
        lines.append("")

    if pf_thr_a.get("best_per_gpu"):
        lines.append("### 各GPU数量最优配置\n")
        headers = ["GPU数", "TP", "EP", "DP", "EDP", "BS", "序列长度", "TPS/GPU"]
        rows = []
        for gpus in sorted(pf_thr_a["best_per_gpu"].keys()):
            r = pf_thr_a["best_per_gpu"][gpus]
            rows.append([
                gpus, r["tp"], r["ep"], r["dp"], r["edp"],
                r["batch_size"], r["seq_len"],
                fmt_num(r["prefill_tps_per_gpu"], 2),
            ])
        lines.append(md_table(headers, rows))
        lines.append("")

    if pf_thr_a.get("batch_scaling"):
        lines.append("### 批量大小扩展（最优配置）\n")
        headers = ["BS", "TPS/GPU", "HBM(GB)"]
        rows = []
        for r in pf_thr_a["batch_scaling"]:
            rows.append([
                r["batch_size"],
                fmt_num(r["prefill_tps_per_gpu"], 2),
                fmt_num(r["hbm_total_gb"]),
            ])
        lines.append(md_table(headers, rows))
        lines.append("")

    # --- Section 4: Decode Throughput ---
    dc_thr = all_results.get("decode_throughput", [])
    dc_thr_a = all_analyses.get("decode_throughput", {})
    lines.append("## 4. 解码吞吐量（最大化 decode_tps_per_gpu）\n")

    if dc_thr:
        lines.append("### 前10最优配置\n")
        headers = ["排名", "TP", "EP", "DP", "EDP", "BS", "序列长度",
                    "GPU数", "TPS/GPU", "精确TPS/GPU", "误差%", "HBM(GB)"]
        rows = []
        for i, r in enumerate(dc_thr[:10]):
            exact_tps = r.get("decode_tps_per_gpu", "")
            err = r.get("approx_error_pct", "")
            rows.append([
                i + 1, r["tp"], r["ep"], r["dp"], r["edp"],
                r["batch_size"], r["seq_len"], r["physical_gpus"],
                fmt_num(r["decode_tps_per_gpu"], 2) if isinstance(r.get("decode_tps_per_gpu"), (int, float)) else r.get("decode_tps_per_gpu", "-"),
                fmt_num(r.get("decode_tps_per_gpu"), 2) if r.get("approx_error_pct", "") != "" else "-",
                fmt_num(err) if err != "" else "-",
                fmt_num(r["hbm_total_gb"]),
            ])
        lines.append(md_table(headers, rows))
        lines.append("")

    if dc_thr_a.get("best_per_gpu"):
        lines.append("### 各GPU数量最优配置\n")
        headers = ["GPU数", "TP", "EP", "DP", "EDP", "BS", "序列长度", "TPS/GPU"]
        rows = []
        for gpus in sorted(dc_thr_a["best_per_gpu"].keys()):
            r = dc_thr_a["best_per_gpu"][gpus]
            rows.append([
                gpus, r["tp"], r["ep"], r["dp"], r["edp"],
                r["batch_size"], r["seq_len"],
                fmt_num(r["decode_tps_per_gpu"], 2),
            ])
        lines.append(md_table(headers, rows))
        lines.append("")

    if dc_thr_a.get("batch_scaling"):
        lines.append("### 批量大小扩展（最优配置）\n")
        headers = ["BS", "TPS/GPU", "HBM(GB)"]
        rows = []
        for r in dc_thr_a["batch_scaling"]:
            rows.append([
                r["batch_size"],
                fmt_num(r["decode_tps_per_gpu"], 2),
                fmt_num(r["hbm_total_gb"]),
            ])
        lines.append(md_table(headers, rows))
        lines.append("")

    # --- Verification summary ---
    if dc_thr_a.get("verification"):
        v = dc_thr_a["verification"]
        lines.append("## 验证摘要\n")
        lines.append(f"- **解码吞吐量前10：** {v['count']} 个已验证，"
                      f"最大误差 = {v['max_error']:.2f}%，平均误差 = {v['avg_error']:.2f}%")
        lines.append("")

    report_path = os.path.join(results_dir, "search_report_zh.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    return report_path


def main():
    results_dir = find_latest_results()
    print(f"Loading results from: {results_dir}")

    # Load all 4 scenario results
    scenario_keys = [
        "prefill_latency", "decode_latency",
        "prefill_throughput", "decode_throughput",
    ]
    all_results = {}
    for key in scenario_keys:
        path = os.path.join(results_dir, f"{key}_all.csv")
        all_results[key] = load_results(path)
        print(f"  {key}: {len(all_results[key])} configs")

    with open(os.path.join(results_dir, "search_meta.json")) as f:
        meta = json.load(f)
    print()

    # --- Analysis ---
    all_analyses = {
        "prefill_latency": analyze_prefill_latency(all_results["prefill_latency"]),
        "decode_latency": analyze_decode_latency(all_results["decode_latency"]),
        "prefill_throughput": analyze_prefill_throughput(all_results["prefill_throughput"]),
        "decode_throughput": analyze_decode_throughput(all_results["decode_throughput"]),
    }

    # --- Console output ---
    for key in scenario_keys:
        analysis = all_analyses[key]
        data = all_results[key]
        print("=" * 70)
        print(f"{key.upper().replace('_', ' ')} ANALYSIS")
        print("=" * 70)

        if not analysis.get("best"):
            print("  No results.")
            print()
            continue

        b = analysis["best"]
        metric_map = {
            "prefill_latency": ("prefill_time_ms", "ms"),
            "decode_latency": ("decode_first_step_ms", "ms"),
            "prefill_throughput": ("prefill_tps_per_gpu", "tps/gpu"),
            "decode_throughput": ("decode_tps_per_gpu", "tps/gpu"),
        }
        metric_key, metric_label = metric_map[key]

        metric_val = b[metric_key]
        print(f"  Best: TP={b['tp']} EP={b['ep']} DP={b['dp']} EDP={b['edp']} "
              f"BS={b['batch_size']} seq={b['seq_len']} SP={b.get('sp', '-')} "
              f"=> {fmt_num(metric_val, 3)} {metric_label} on {b['physical_gpus']} GPUs")

        # Best per seq_len (latency scenarios)
        if analysis.get("best_per_seq"):
            print("  Best per sequence length:")
            for sl in sorted(analysis["best_per_seq"].keys()):
                r = analysis["best_per_seq"][sl]
                val = r[metric_key]
                print(f"    seq={sl:>6d}: TP={r['tp']:>2d} EP={r['ep']:>3d} DP={r['dp']} "
                      f"BS={r['batch_size']:>3d} => {fmt_num(val, 3)} {metric_label}")

        # Best per GPU count (throughput scenarios)
        if analysis.get("best_per_gpu"):
            print("  Best per GPU count:")
            for gpus in sorted(analysis["best_per_gpu"].keys()):
                r = analysis["best_per_gpu"][gpus]
                val = r[metric_key]
                print(f"    {gpus:>2d} GPUs: TP={r['tp']:>2d} EP={r['ep']:>3d} DP={r['dp']} "
                      f"BS={r['batch_size']:>3d} seq={r['seq_len']:>5d} => {fmt_num(val, 3)} {metric_label}")

        # SP impact
        if analysis.get("sp_impact"):
            print("  SP Impact:")
            for sl in sorted(analysis["sp_impact"].keys()):
                imp = analysis["sp_impact"][sl]
                print(f"    seq={sl:>6d}: SP=True {fmt_num(imp['sp_true_ms'], 3)} vs "
                      f"SP=False {fmt_num(imp['sp_false_ms'], 3)} (speedup={imp['speedup']:.2f}x)")

        # Verification
        if analysis.get("verification"):
            v = analysis["verification"]
            print(f"  Verification: {v['count']} configs, max_err={v['max_error']:.2f}%, "
                  f"avg_err={v['avg_error']:.2f}%")

        print()

    # --- Generate reports ---
    report_path = generate_markdown_report(
        results_dir, all_results, all_analyses, meta,
    )
    print(f"Report saved to: {report_path}")

    report_zh_path = generate_markdown_report_zh(
        results_dir, all_results, all_analyses, meta,
    )
    print(f"Chinese report saved to: {report_zh_path}")


if __name__ == "__main__":
    main()
