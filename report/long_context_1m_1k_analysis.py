#!/usr/bin/env python3
"""Generate focused 1M-input / 1K-output prefix-cache P/D sizing data."""

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from report import analyze_scenarios as scenarios


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate 1M/1K prefix-cache P/D sizing data.",
    )
    parser.add_argument("--seq-len", type=int, default=1_000_000)
    parser.add_argument("--output-len", type=int, default=1024)
    parser.add_argument("--hit-rates", type=float, nargs="+", default=[0.0, 0.9, 0.99])
    parser.add_argument("--hardware", nargs="+", default=["910C", "H20"])
    parser.add_argument(
        "--json-out",
        default="report/data/long_context_1m_1k_prefix_cache.json",
    )
    parser.add_argument(
        "--pd-tolerance",
        type=float,
        default=scenarios.PD_RATIO_TOLERANCE,
        help="Relative QPS imbalance tolerance for integer P/D ratio search.",
    )
    return parser.parse_args()


def best_configs(base_cfg, hw_name, seq_len, output_len, hit_rate, pd_tolerance):
    hbm_limit = base_cfg.hw.usable_hbm_capacity_gb
    with contextlib.redirect_stdout(io.StringIO()):
        prefill = scenarios.sort_results(
            scenarios.run_search(
                base_cfg,
                "prefill",
                seq_len,
                output_len,
                hbm_limit,
                scenarios.PREFILL_GPU_VALUES,
                prefix_cache_hit_rate=hit_rate,
            ),
            "prefill",
            "throughput",
        )
        decode = scenarios.sort_results(
            scenarios.run_search(
                base_cfg,
                "decode",
                seq_len,
                output_len,
                hbm_limit,
                scenarios.DECODE_GPU_VALUES,
                prefix_cache_hit_rate=hit_rate,
            ),
            "decode",
            "throughput",
        )
    if not prefill or not decode:
        return {"error": "no valid configs"}

    p_best = prefill[0]
    d_best = decode[0]
    pd_ratio = scenarios.compute_pd_ratio(
        p_best["prefill_qps_instance"],
        d_best["decode_qps_instance"],
        tolerance=pd_tolerance,
    )
    p_instances = pd_ratio["prefill_instances"]
    d_instances = pd_ratio["decode_instances"]
    return {
        "schema_version": 2,
        "seq_len": seq_len,
        "output_len": output_len,
        "prefix_cache_hit_rate": hit_rate,
        "effective_prefill_len": p_best["effective_prefill_len"],
        "decode_context_len": d_best["decode_context_len"],
        "prefill": p_best,
        "decode": d_best,
        "pd_ratio_float": pd_ratio["pd_ratio_float"],
        "pd_ratio_actual": pd_ratio["pd_ratio_actual"],
        "prefill_instances": p_instances,
        "decode_instances": d_instances,
        "prefill_aggregate_qps": pd_ratio["prefill_aggregate_qps"],
        "decode_aggregate_qps": pd_ratio["decode_aggregate_qps"],
        "qps_imbalance": pd_ratio["qps_imbalance"],
        "qps_imbalance_pct": pd_ratio["qps_imbalance_pct"],
        "balance_tolerance": pd_ratio["balance_tolerance"],
        "pd_label": pd_ratio["label"],
        "balanced_cards_for_selected_instances": (
            p_instances * p_best["physical_gpus"]
            + d_instances * d_best["physical_gpus"]
        ),
    }


def print_summary(data):
    for hw_name, rows in data.items():
        print(f"\n[{hw_name}]")
        print("hit_rate,p_instance,d_instance,p_qps,d_qps,pd_ratio,balanced_cards")
        for hit_key, row in rows.items():
            if "error" in row:
                print(f"{hit_key},ERROR,{row['error']}")
                continue
            p = row["prefill"]
            d = row["decode"]
            p_inst = f"TP={p['tp']}/EP={p['ep']}/DP={p['dp']}/BS={p['batch_size']}/{p['physical_gpus']}cards"
            d_inst = f"TP={d['tp']}/EP={d['ep']}/DP={d['dp']}/BS={d['batch_size']}/{d['physical_gpus']}cards"
            print(
                f"{hit_key},{p_inst},{d_inst},"
                f"{p['prefill_qps_instance']:.6f},{d['decode_qps_instance']:.6f},"
                f"{row['pd_label']},{row['balanced_cards_for_selected_instances']}"
            )


def main():
    args = parse_args()
    data = {}
    for hw_name in args.hardware:
        base_cfg = scenarios.load_base_config(hw_name)
        data[hw_name] = {
            str(hit_rate): best_configs(
                base_cfg,
                hw_name,
                args.seq_len,
                args.output_len,
                hit_rate,
                args.pd_tolerance,
            )
            for hit_rate in args.hit_rates
        }

    out_path = Path(args.json_out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print_summary(data)
    try:
        display_path = out_path.relative_to(REPO_ROOT)
    except ValueError:
        display_path = out_path
    print(f"\nWrote {display_path}")


if __name__ == "__main__":
    main()
