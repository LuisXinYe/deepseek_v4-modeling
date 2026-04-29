#!/usr/bin/env python3
"""Generate the 0428 PD disaggregation analysis report."""

from __future__ import annotations

import html
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import (
    DATA_DIR,
    DECODE_INSTANCE_SIZES,
    EP_VALUES,
    FIGURE_DIR,
    PREFILL_CARD_COUNTS,
    PREFIX_CACHE_HIT_RATES,
    REPORT_DEFAULTS,
    REPORT_ROOT,
    SCENARIOS,
    TP_VALUES,
    infeasible_row,
    is_parallel_valid,
    load_910c_config,
    with_runtime_defaults,
    write_json,
)
from perf_model.serving import (
    compute_pd_ratio,
    evaluate_decode_serving,
    evaluate_prefill_serving,
)


DECODE_MODES = [
    {"decode_mode": "no_mtp", "label": "No MTP", "mtp": 0},
    {"decode_mode": "mtp", "label": "MTP=1", "mtp": 1},
]

MAX_BATCH_PER_CARD_CAP = 1_000_000
MAX_PREFILL_BATCH_UNITS_CAP = 100_000
MAX_EXHAUSTIVE_PREFILL_UNITS = 4_096


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPORT_ROOT.parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def iter_parallel_candidates(card_count: int, batch_size: int | None = None):
    for tp in TP_VALUES:
        if card_count % tp != 0:
            continue
        dp = card_count // tp
        for ep in EP_VALUES:
            if card_count % ep != 0:
                continue
            yield tp, ep, dp, batch_size


def _candidate_cfg(base_cfg, scenario, *, cards: int, tp: int, ep: int, dp: int, batch_size: int, mtp: int):
    return with_runtime_defaults(
        base_cfg,
        input_len=scenario["input_len"],
        output_len=scenario["output_len"],
        batch_size=batch_size,
        tp=tp,
        ep=ep,
        dp=dp,
        mtp=mtp,
        prefix_cache_hit_rate=scenario.get("prefix_cache_hit_rate"),
    )


def _decorate_common(row: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    row["scenario_id"] = scenario["scenario_id"]
    row["scenario_label"] = scenario["label"]
    row["input_len"] = scenario["input_len"]
    row["output_len"] = scenario["output_len"]
    row["prefix_cache_hit_rate"] = scenario.get("prefix_cache_hit_rate", REPORT_DEFAULTS["prefix_cache_hit_rate"])
    row["edp"] = row["physical_gpus"] // row["ep"]
    row["is_feasible"] = row.get("hbm_margin_gb", -1) >= -1e-9
    row["invalid_reason"] = None if row["is_feasible"] else "oom"
    return row


def _prefill_sort_key(row: dict[str, Any]):
    return (
        row["prefill_tps_per_card"] or -1,
        row["prefill_qps_instance"] or -1,
        row["hbm_margin_gb"],
        -(row["prefill_time_ms"] or math.inf),
        -row["tp"],
        -row["ep"],
    )


def with_prefix_hit(scenario: dict[str, Any], prefix_cache_hit_rate: float) -> dict[str, Any]:
    scoped = dict(scenario)
    scoped["prefix_cache_hit_rate"] = prefix_cache_hit_rate
    return scoped


def _evaluate_prefill_row(
    base_cfg,
    scenario: dict[str, Any],
    *,
    cards: int,
    tp: int,
    ep: int,
    dp: int,
    batch_size: int,
) -> dict[str, Any]:
    cfg = _candidate_cfg(
        base_cfg,
        scenario,
        cards=cards,
        tp=tp,
        ep=ep,
        dp=dp,
        batch_size=batch_size,
        mtp=0,
    )
    metrics = evaluate_prefill_serving(cfg)
    row = _decorate_common({**metrics, "tp": tp, "ep": ep, "dp": dp, "sp": cfg.rt.sp}, scenario)
    row["sizing_batch_size"] = 1
    row["sizing_physical_gpus"] = cards
    return row


def _find_min_prefill_cards_by_bs1_hbm(base_cfg, scenario: dict[str, Any]) -> int | None:
    for cards in PREFILL_CARD_COUNTS:
        for tp, ep, dp, _ in iter_parallel_candidates(cards, batch_size=1):
            if not is_parallel_valid(base_cfg, tp=tp, ep=ep, dp=dp, batch_size=1):
                continue
            row = _evaluate_prefill_row(
                base_cfg,
                scenario,
                cards=cards,
                tp=tp,
                ep=ep,
                dp=dp,
                batch_size=1,
            )
            if row["is_feasible"]:
                return cards
    return None


class PrefillEvaluator:
    def __init__(self, base_cfg, scenario: dict[str, Any], cards: int):
        self.base_cfg = base_cfg
        self.scenario = scenario
        self.cards = cards
        self.cache: dict[tuple[int, int, int, int], dict[str, Any]] = {}

    def evaluate(self, tp: int, ep: int, dp: int, batch_size: int) -> dict[str, Any]:
        key = (tp, ep, dp, batch_size)
        if key not in self.cache:
            self.cache[key] = _evaluate_prefill_row(
                self.base_cfg,
                self.scenario,
                cards=self.cards,
                tp=tp,
                ep=ep,
                dp=dp,
                batch_size=batch_size,
            )
        return self.cache[key]


def _find_max_prefill_batch_units(evaluator: PrefillEvaluator, tp: int, ep: int, dp: int) -> int:
    first = evaluator.evaluate(tp, ep, dp, dp)
    if not _fits_hbm(first):
        return 0

    low = 1
    high = 2
    while high <= MAX_PREFILL_BATCH_UNITS_CAP and _fits_hbm(evaluator.evaluate(tp, ep, dp, high * dp)):
        low = high
        high *= 2

    if high > MAX_PREFILL_BATCH_UNITS_CAP:
        high = MAX_PREFILL_BATCH_UNITS_CAP + 1

    while low + 1 < high:
        mid = (low + high) // 2
        if _fits_hbm(evaluator.evaluate(tp, ep, dp, mid * dp)):
            low = mid
        else:
            high = mid
    return low


def _prefill_batch_unit_candidates(max_units: int) -> list[int]:
    if max_units <= MAX_EXHAUSTIVE_PREFILL_UNITS:
        return list(range(1, max_units + 1))

    candidates = {1, max_units}
    power = 1
    while power < max_units:
        candidates.add(power)
        power *= 2
    for i in range(1, 33):
        candidates.add(max(1, round(max_units * i / 32)))
    for unit in range(max(1, max_units - 128), max_units + 1):
        candidates.add(unit)
    return sorted(candidates)


def select_prefill_perf_for_cards(base_cfg, scenario: dict[str, Any], cards: int) -> dict[str, Any] | None:
    evaluator = PrefillEvaluator(base_cfg, scenario, cards)
    best = None

    for tp, ep, dp, _ in iter_parallel_candidates(cards):
        min_batch_size = dp
        if not is_parallel_valid(base_cfg, tp=tp, ep=ep, dp=dp, batch_size=min_batch_size):
            continue

        max_units = _find_max_prefill_batch_units(evaluator, tp, ep, dp)
        if max_units == 0:
            continue

        for units in _prefill_batch_unit_candidates(max_units):
            batch_size = units * dp
            row = evaluator.evaluate(tp, ep, dp, batch_size)
            if not row["is_feasible"]:
                continue
            row["max_batch_size_hbm"] = max_units * dp
            row["max_batch_per_card_hbm"] = row["max_batch_size_hbm"] / cards
            if best is None or _prefill_sort_key(row) > _prefill_sort_key(best):
                best = row

    return best


def select_prefill_for_scenario(
    base_cfg,
    scenario: dict[str, Any],
    prefix_cache_hit_rate: float,
) -> dict[str, Any]:
    scenario = with_prefix_hit(scenario, prefix_cache_hit_rate)
    sizing_cards = _find_min_prefill_cards_by_bs1_hbm(base_cfg, scenario)
    if sizing_cards is not None:
        row = select_prefill_perf_for_cards(base_cfg, scenario, sizing_cards)
        if row is not None:
            row["sizing_physical_gpus"] = sizing_cards
            return row

    return infeasible_row(
        scenario["scenario_id"],
        "oom",
        scenario_label=scenario["label"],
        input_len=scenario["input_len"],
        output_len=scenario["output_len"],
        prefix_cache_hit_rate=prefix_cache_hit_rate,
    )


class DecodeEvaluator:
    def __init__(self, base_cfg, scenario: dict[str, Any], cards: int, mtp: int):
        self.base_cfg = base_cfg
        self.scenario = scenario
        self.cards = cards
        self.mtp = mtp
        self.cache: dict[tuple[int, int, int, int], dict[str, Any]] = {}

    def evaluate(self, tp: int, ep: int, dp: int, batch_per_card: int) -> dict[str, Any]:
        key = (tp, ep, dp, batch_per_card)
        if key in self.cache:
            return self.cache[key]
        batch_size = self.cards * batch_per_card
        cfg = _candidate_cfg(
            self.base_cfg,
            self.scenario,
            cards=self.cards,
            tp=tp,
            ep=ep,
            dp=dp,
            batch_size=batch_size,
            mtp=self.mtp,
        )
        metrics = evaluate_decode_serving(cfg)
        row = _decorate_common({**metrics, "tp": tp, "ep": ep, "dp": dp, "sp": cfg.rt.sp}, self.scenario)
        row["batch_per_card"] = batch_per_card
        row["decode_mode"] = "mtp" if self.mtp else "no_mtp"
        row["mtp"] = self.mtp
        row["is_tpot_feasible"] = (
            row["is_feasible"]
            and row["tpot_ms"] is not None
            and row["tpot_ms"] <= REPORT_DEFAULTS["tpot_target_ms"]
        )
        self.cache[key] = row
        return row


def _fits_hbm(row: dict[str, Any]) -> bool:
    return row["hbm_margin_gb"] >= -1e-9


def _find_max_hbm_bpc(evaluator: DecodeEvaluator, tp: int, ep: int, dp: int) -> tuple[int, dict[str, Any] | None]:
    first = evaluator.evaluate(tp, ep, dp, 1)
    if not _fits_hbm(first):
        return 0, None

    low = 1
    high = 2
    while high <= MAX_BATCH_PER_CARD_CAP and _fits_hbm(evaluator.evaluate(tp, ep, dp, high)):
        low = high
        high *= 2

    if high > MAX_BATCH_PER_CARD_CAP:
        high = MAX_BATCH_PER_CARD_CAP + 1

    while low + 1 < high:
        mid = (low + high) // 2
        if _fits_hbm(evaluator.evaluate(tp, ep, dp, mid)):
            low = mid
        else:
            high = mid
    return low, evaluator.evaluate(tp, ep, dp, low)


def _find_max_tpot_bpc(
    evaluator: DecodeEvaluator,
    tp: int,
    ep: int,
    dp: int,
    max_hbm_bpc: int,
) -> tuple[int, dict[str, Any] | None]:
    low = 0
    high = max_hbm_bpc + 1
    best = None
    while low + 1 < high:
        mid = (low + high) // 2
        row = evaluator.evaluate(tp, ep, dp, mid)
        if row["is_tpot_feasible"]:
            low = mid
            best = row
        else:
            high = mid
    if low == 0:
        return 0, None
    return low, best or evaluator.evaluate(tp, ep, dp, low)


def _hbm_sort_key(row: dict[str, Any]):
    return (
        row["batch_per_card"],
        row["decode_tps_per_card"] or -1,
        row["hbm_margin_gb"],
        -row["tp"],
        -row["ep"],
    )


def _decode_tpot_sort_key(row: dict[str, Any]):
    return (
        row["batch_per_card"],
        row["decode_tps_per_card"] or -1,
        -(row["tpot_ms"] or math.inf),
        row["hbm_margin_gb"],
        -row["tp"],
        -row["ep"],
    )


def _decode_best_instance_key(row: dict[str, Any]):
    return (
        row["decode_tps_per_card"] or -1,
        -row["physical_gpus"],
        row["hbm_margin_gb"],
        -(row["tpot_ms"] or math.inf),
    )


def select_decode_for_scenario(
    base_cfg,
    scenario: dict[str, Any],
    cards: int,
    *,
    mtp: int,
) -> dict[str, Any]:
    evaluator = DecodeEvaluator(base_cfg, scenario, cards, mtp)
    hbm_rows = []
    tpot_rows = []
    valid_parallel = 0

    for tp, ep, dp, _ in iter_parallel_candidates(cards):
        if not is_parallel_valid(base_cfg, tp=tp, ep=ep, dp=dp, batch_size=cards):
            continue
        valid_parallel += 1
        max_hbm_bpc, hbm_row = _find_max_hbm_bpc(evaluator, tp, ep, dp)
        if hbm_row is None:
            continue
        hbm_rows.append(hbm_row)
        max_tpot_bpc, tpot_row = _find_max_tpot_bpc(evaluator, tp, ep, dp, max_hbm_bpc)
        if tpot_row is not None:
            tpot_rows.append(tpot_row)

    mode = "mtp" if mtp else "no_mtp"
    mode_label = "MTP=1" if mtp else "No MTP"
    if not hbm_rows:
        reason = "no_candidate" if valid_parallel == 0 else "oom"
        return infeasible_row(
            scenario["scenario_id"],
            reason,
            scenario_label=scenario["label"],
            input_len=scenario["input_len"],
            output_len=scenario["output_len"],
            decode_mode=mode,
            decode_mode_label=mode_label,
            physical_gpus=cards,
            max_batch_per_card_hbm=None,
            max_batch_per_card_tpot=None,
            is_tpot_feasible=False,
        )

    hbm_best = max(hbm_rows, key=_hbm_sort_key)
    hbm_fields = {
        "max_batch_per_card_hbm": hbm_best["batch_per_card"],
        "max_batch_size_hbm": hbm_best["batch_size"],
        "hbm_tp": hbm_best["tp"],
        "hbm_ep": hbm_best["ep"],
        "hbm_dp": hbm_best["dp"],
        "hbm_total_gb_at_max_batch": hbm_best["hbm_total_gb"],
    }

    if not tpot_rows:
        return infeasible_row(
            scenario["scenario_id"],
            "tpot_exceeded",
            scenario_label=scenario["label"],
            input_len=scenario["input_len"],
            output_len=scenario["output_len"],
            decode_mode=mode,
            decode_mode_label=mode_label,
            physical_gpus=cards,
            is_tpot_feasible=False,
            max_batch_per_card_tpot=None,
            max_batch_size_tpot=None,
            **hbm_fields,
        )

    selected = max(tpot_rows, key=_decode_tpot_sort_key)
    row = dict(selected)
    row["decode_mode_label"] = mode_label
    row["invalid_reason"] = None
    row["max_batch_per_card_tpot"] = row["batch_per_card"]
    row["max_batch_size_tpot"] = row["batch_size"]
    row.update(hbm_fields)
    return row


def generate_prefill_results(base_cfg) -> list[dict[str, Any]]:
    return [
        select_prefill_for_scenario(base_cfg, scenario, hit_rate)
        for scenario in SCENARIOS
        for hit_rate in PREFIX_CACHE_HIT_RATES
    ]


def generate_decode_results(base_cfg) -> dict[str, Any]:
    rows = []
    for scenario in SCENARIOS:
        for mode in DECODE_MODES:
            for cards in DECODE_INSTANCE_SIZES:
                rows.append(select_decode_for_scenario(base_cfg, scenario, cards, mtp=mode["mtp"]))

    best = []
    for scenario in SCENARIOS:
        for mode in DECODE_MODES:
            mode_rows = [
                row for row in rows
                if row["scenario_id"] == scenario["scenario_id"]
                and row["decode_mode"] == mode["decode_mode"]
                and row.get("is_tpot_feasible")
            ]
            for row in mode_rows:
                row["is_best_decode_instance"] = False
            if mode_rows:
                selected = max(mode_rows, key=_decode_best_instance_key)
                selected["is_best_decode_instance"] = True
                best.append(dict(selected))
            else:
                best.append(
                    infeasible_row(
                        scenario["scenario_id"],
                        "tpot_exceeded",
                        scenario_label=scenario["label"],
                        decode_mode=mode["decode_mode"],
                        decode_mode_label=mode["label"],
                    )
                )
    return {"rows": rows, "best_by_scenario_mode": best}


def generate_pd_ratio_results(
    prefill_rows: list[dict[str, Any]],
    decode_best_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prefill_by_scenario_hit = {
        (row["scenario_id"], row["prefix_cache_hit_rate"]): row
        for row in prefill_rows
    }
    decode_by_key = {(row["scenario_id"], row.get("decode_mode")): row for row in decode_best_rows}

    rows = []
    for scenario in SCENARIOS:
        headline_mode = "mtp"
        mtp_best = decode_by_key.get((scenario["scenario_id"], "mtp"))
        if not mtp_best or not mtp_best.get("is_tpot_feasible"):
            headline_mode = "no_mtp"

        for hit_rate in PREFIX_CACHE_HIT_RATES:
            for mode in DECODE_MODES:
                prefill = prefill_by_scenario_hit[(scenario["scenario_id"], hit_rate)]
                decode = decode_by_key.get((scenario["scenario_id"], mode["decode_mode"]))
                if not prefill.get("is_feasible", True) or not decode or not decode.get("is_tpot_feasible"):
                    rows.append(
                        infeasible_row(
                            scenario["scenario_id"],
                            "tpot_exceeded",
                            scenario_label=scenario["label"],
                            prefix_cache_hit_rate=hit_rate,
                            decode_mode=mode["decode_mode"],
                            decode_mode_label=mode["label"],
                            is_headline_recommendation=False,
                        )
                    )
                    continue

                ratio_float = decode["decode_qps_instance"] / prefill["prefill_qps_instance"]
                max_instances = max(10_000, math.ceil(ratio_float / 0.9) + 10)
                ratio = compute_pd_ratio(
                    prefill["prefill_qps_instance"],
                    decode["decode_qps_instance"],
                    tolerance=0.1,
                    max_instances=max_instances,
                )
                row = {
                    "scenario_id": scenario["scenario_id"],
                    "scenario_label": scenario["label"],
                    "prefix_cache_hit_rate": hit_rate,
                    "decode_mode": mode["decode_mode"],
                    "decode_mode_label": mode["label"],
                    "prefill_physical_gpus": prefill["physical_gpus"],
                    "decode_physical_gpus": decode["physical_gpus"],
                    "prefill_instances": ratio["prefill_instances"],
                    "decode_instances": ratio["decode_instances"],
                    "pd_ratio_float": ratio["pd_ratio_float"],
                    "pd_ratio_actual": ratio["pd_ratio_actual"],
                    "prefill_aggregate_qps": ratio["prefill_aggregate_qps"],
                    "decode_aggregate_qps": ratio["decode_aggregate_qps"],
                    "qps_imbalance_pct": ratio["qps_imbalance_pct"],
                    "total_cards": (
                        ratio["prefill_instances"] * prefill["physical_gpus"]
                        + ratio["decode_instances"] * decode["physical_gpus"]
                    ),
                    "is_headline_recommendation": mode["decode_mode"] == headline_mode,
                    "is_feasible": True,
                    "invalid_reason": None,
                    "label": ratio["label"],
                }
                rows.append(row)
    return rows


def _nice(value: Any, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return str(value)


def _hit_label(hit_rate: float) -> str:
    return f"h={hit_rate:g}"


def _scenario_short_label(label: str) -> str:
    return label.split(" + ")[0]


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_nice(value) for value in row) + " |")
    return "\n".join(lines)


def _svg_text(x: float, y: float, text: str, *, size: int = 12, anchor: str = "middle", weight: str = "400") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
        f'text-anchor="{anchor}" fill="#1f2937">{html.escape(text)}</text>'
    )


def _svg_multiline_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 12,
    anchor: str = "middle",
    line_height: int = 14,
) -> list[str]:
    lines = str(text).split("\n")
    return [
        _svg_text(x, y + idx * line_height, line, size=size, anchor=anchor)
        for idx, line in enumerate(lines)
    ]


def write_bar_chart_svg(
    path: Path,
    title: str,
    labels: list[str],
    values: list[float],
    ylabel: str,
    value_labels: list[str] | None = None,
) -> None:
    width, height = 920, 460
    left, right, top, bottom = 90, 40, 60, 86
    chart_w = width - left - right
    chart_h = height - top - bottom
    max_value = max(values) if values else 1
    max_value = max(max_value, 1e-9)
    bar_w = chart_w / max(len(values), 1) * 0.56
    gap = chart_w / max(len(values), 1)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        _svg_text(width / 2, 30, title, size=18, weight="700"),
        _svg_text(left, top - 14, ylabel, size=12, anchor="start"),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#374151"/>',
        f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="#374151"/>',
    ]
    for idx in range(6):
        value = max_value * idx / 5
        y = top + chart_h - chart_h * idx / 5
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        out.append(_svg_text(left - 10, y + 4, _nice(value, 1), size=11, anchor="end"))
    for i, value in enumerate(values):
        x = left + gap * i + gap / 2
        h = chart_h * value / max_value
        y = top + chart_h - h
        out.append(f'<rect x="{x - bar_w / 2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="#2563eb"/>')
        label = value_labels[i] if value_labels else _nice(value, 2)
        out.append(_svg_text(x, y - 8, label, size=11))
        out.extend(_svg_multiline_text(x, top + chart_h + 22, labels[i], size=10, line_height=14))
    out.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n")


def write_line_chart_svg(
    path: Path,
    title: str,
    x_labels: list[str],
    series: list[dict[str, Any]],
    ylabel: str,
) -> None:
    width, height = 980, 540
    left, right, top, bottom = 90, 150, 84, 84
    chart_w = width - left - right
    chart_h = height - top - bottom
    values = [value for item in series for value in item["values"] if value is not None]
    max_value = max(values) if values else 1
    max_value = max(max_value, 1e-9)
    colors = ["#2563eb", "#dc2626", "#059669"]
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        _svg_text(width / 2, 30, title, size=18, weight="700"),
        _svg_text(left, 56, "Point label = HBM / TPOT max B/card", size=11, anchor="start"),
        _svg_text(left, top - 12, ylabel, size=12, anchor="start"),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#374151"/>',
        f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="#374151"/>',
    ]
    for idx in range(6):
        value = max_value * idx / 5
        y = top + chart_h - chart_h * idx / 5
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        out.append(_svg_text(left - 10, y + 4, _nice(value, 1), size=11, anchor="end"))
    x_step = chart_w / (len(x_labels) - 1) if len(x_labels) > 1 else chart_w
    for i, label in enumerate(x_labels):
        x = left + x_step * i
        out.append(_svg_text(x, top + chart_h + 24, label, size=12))
    for s_idx, item in enumerate(series):
        color = colors[s_idx % len(colors)]
        points = []
        for i, value in enumerate(item["values"]):
            if value is None:
                continue
            x = left + x_step * i
            y = top + chart_h - chart_h * value / max_value
            points.append((x, y, value, item.get("point_labels", [""] * len(x_labels))[i]))
        if len(points) > 1:
            out.append(
                '<polyline points="'
                + " ".join(f"{x:.1f},{y:.1f}" for x, y, _, _ in points)
                + f'" fill="none" stroke="{color}" stroke-width="2.5"/>'
            )
        for x, y, _, label in points:
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}"/>')
            if label:
                label_y = y + 18 if s_idx == 0 else y - 12
                out.append(_svg_text(x, label_y, label, size=10))
        legend_y = top + 12 + s_idx * 24
        out.append(f'<rect x="{left + chart_w + 28}" y="{legend_y - 10}" width="14" height="14" fill="{color}"/>')
        out.append(_svg_text(left + chart_w + 50, legend_y + 2, item["name"], size=12, anchor="start"))
    out.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n")


def write_stacked_pd_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    width, height = 1600, 520
    left, right, top, bottom = 90, 60, 62, 112
    chart_w = width - left - right
    chart_h = height - top - bottom
    feasible = [row for row in rows if row.get("is_feasible")]
    max_total = max((row["total_cards"] for row in feasible), default=1)
    bar_count = len(feasible)
    slot = chart_w / max(bar_count, 1)
    bar_w = slot * 0.56
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        _svg_text(width / 2, 30, "P/D Total Card Composition", size=18, weight="700"),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#374151"/>',
        f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="#374151"/>',
    ]
    for idx in range(6):
        value = max_total * idx / 5
        y = top + chart_h - chart_h * idx / 5
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        out.append(_svg_text(left - 10, y + 4, _nice(value, 0), size=11, anchor="end"))
    for i, row in enumerate(feasible):
        x = left + slot * i + slot / 2
        p_cards = row["prefill_instances"] * row["prefill_physical_gpus"]
        d_cards = row["decode_instances"] * row["decode_physical_gpus"]
        p_h = chart_h * p_cards / max_total
        d_h = chart_h * d_cards / max_total
        y_d = top + chart_h - d_h
        y_p = y_d - p_h
        out.append(f'<rect x="{x - bar_w / 2:.1f}" y="{y_p:.1f}" width="{bar_w:.1f}" height="{p_h:.1f}" fill="#2563eb"/>')
        out.append(f'<rect x="{x - bar_w / 2:.1f}" y="{y_d:.1f}" width="{bar_w:.1f}" height="{d_h:.1f}" fill="#dc2626"/>')
        out.append(_svg_text(x, y_p - 8, _nice(row["total_cards"], 0), size=10))
        out.append(_svg_text(x, top + chart_h + 22, row["scenario_label"], size=9))
        out.append(_svg_text(x, top + chart_h + 38, _hit_label(row["prefix_cache_hit_rate"]), size=9))
        out.append(_svg_text(x, top + chart_h + 54, row["decode_mode_label"], size=9))
    out.append(f'<rect x="{width - 170}" y="70" width="14" height="14" fill="#2563eb"/>')
    out.append(_svg_text(width - 150, 82, "Prefill cards", size=12, anchor="start"))
    out.append(f'<rect x="{width - 170}" y="96" width="14" height="14" fill="#dc2626"/>')
    out.append(_svg_text(width - 150, 108, "Decode cards", size=12, anchor="start"))
    out.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n")


def generate_figures(
    prefill_rows: list[dict[str, Any]],
    decode_rows: list[dict[str, Any]],
    pd_rows: list[dict[str, Any]],
) -> None:
    labels = [
        f"{_scenario_short_label(row['scenario_label'])}\n{_hit_label(row['prefix_cache_hit_rate'])}"
        for row in prefill_rows
    ]
    write_bar_chart_svg(
        FIGURE_DIR / "prefill_hbm.svg",
        "Prefill HBM Occupancy by Prefix Cache Hit Rate",
        labels,
        [row["hbm_total_gb"] for row in prefill_rows],
        "GB",
        [f"{row['hbm_total_gb']:.1f} GB" for row in prefill_rows],
    )
    write_bar_chart_svg(
        FIGURE_DIR / "prefill_tps.svg",
        "Prefill TPS per Card by Prefix Cache Hit Rate",
        labels,
        [row["prefill_tps_per_card"] for row in prefill_rows],
        "tokens/s/card",
    )
    for scenario in SCENARIOS:
        scenario_rows = [row for row in decode_rows if row["scenario_id"] == scenario["scenario_id"]]
        series = []
        for mode in DECODE_MODES:
            mode_rows = [row for row in scenario_rows if row["decode_mode"] == mode["decode_mode"]]
            by_cards = {row["physical_gpus"]: row for row in mode_rows}
            values = []
            point_labels = []
            for cards in DECODE_INSTANCE_SIZES:
                row = by_cards[cards]
                values.append(row["decode_tps_per_card"] if row.get("is_tpot_feasible") else None)
                hbm_b = row.get("max_batch_per_card_hbm")
                tpot_b = row.get("max_batch_per_card_tpot")
                point_labels.append(f"{_nice(hbm_b, 0)}/{_nice(tpot_b, 0)}")
            series.append({"name": mode["label"], "values": values, "point_labels": point_labels})
        write_line_chart_svg(
            FIGURE_DIR / f"decode_{scenario['scenario_id']}.svg",
            f"Decode TPS/Card: {scenario['label']}",
            [str(cards) for cards in DECODE_INSTANCE_SIZES],
            series,
            "tokens/s/card",
        )
    write_stacked_pd_svg(FIGURE_DIR / "pd_ratio_total_cards.svg", pd_rows)


def render_report(
    base_cfg,
    prefill_rows: list[dict[str, Any]],
    decode_results: dict[str, Any],
    pd_rows: list[dict[str, Any]],
) -> str:
    decode_rows = decode_results["rows"]
    lines = [
        "# 0428 PD 分离推理分析报告",
        "",
        "## 结论摘要",
        "",
    ]
    headline = [row for row in pd_rows if row.get("is_headline_recommendation") and row.get("is_feasible")]
    for row in headline:
        lines.append(
            f"- {row['scenario_label']}（{_hit_label(row['prefix_cache_hit_rate'])}）："
            f"推荐 Decode {row['decode_mode_label']}，"
            f"P/D={row['prefill_instances']}P:{row['decode_instances']}D，"
            f"总卡数 {row['total_cards']:.0f}。"
        )
    lines.extend([
        "",
        "## 假设与配置",
        "",
        _table(
            ["项目", "取值"],
            [
                ["硬件", "Ascend 910C"],
                ["模型", "DeepSeek V4 Flash"],
                ["量化", REPORT_DEFAULTS["quant_mode"]],
                ["KV Cache 量化", REPORT_DEFAULTS["kv_cache_quant_mode"]],
                ["W8A8 GEMM 吞吐", f"{REPORT_DEFAULTS['w8a8_tflops']} TFLOPS"],
                ["HBM 容量/预留/可用", f"{base_cfg.hw.hbm_capacity_gb} GB / {base_cfg.hw.hbm_reserved_pct}% / {base_cfg.hw.usable_hbm_capacity_gb:.1f} GB"],
                ["prefix_cache_hit_rate", ", ".join(_nice(v) for v in PREFIX_CACHE_HIT_RATES)],
                ["MTP accept ratio", REPORT_DEFAULTS["mtp_accept_ratio"]],
                ["TPOT 约束", f"{REPORT_DEFAULTS['tpot_target_ms']} ms"],
            ],
        ),
        "",
        "## 场景",
        "",
        _table(
            ["场景", "Prefill 输入长度", "Decode 输出长度"],
            [[s["label"], s["input_len"], s["output_len"]] for s in SCENARIOS],
        ),
        "",
        "## 公式",
        "",
        "- Prefix cache：`L_miss = ceil(input_len * (1 - prefix_cache_hit_rate))`，Prefill compute 使用 `F_prefill(L_miss)`，HBM 仍按完整 input context 计算。",
        "- MTP：`tokens_per_forward = 1 + mtp * mtp_accept_ratio`，`decode_forward_count = ceil(output_len / tokens_per_forward)`。",
        "- Decode TPOT：`TPOT = decode_total_time / output_len`，本报告过滤 `TPOT <= 50 ms`。",
        "- HBM：`weight_bytes_quant + kv_bytes_quant`，W8A8 权重按 0.5，KV8 cache 按 0.5。",
        "- P/D 配比：使用 instance QPS，求 `prefill_instances * prefill_qps ~= decode_instances * decode_qps`，容忍 10% imbalance。",
        "",
        "## Prefill 分析",
        "",
        "Prefill 先按 `batch_size=1` 搜索模型权重加完整输入 KV cache 可以放入 HBM 的最小实例卡数；随后在该卡数内搜索 TPS/card 最大的 `{TP, EP, DP, batch_size}` 性能配置。",
        "",
        "![Prefill HBM](figure/prefill_hbm.svg)",
        "",
        "![Prefill TPS](figure/prefill_tps.svg)",
        "",
        _table(
            ["场景", "Hit", "卡数", "TP", "EP", "DP", "BS", "B/card", "L_miss", "Weight GB", "KV GB", "HBM GB", "Prefill ms", "QPS", "TPS/card"],
            [
                [
                    row["scenario_label"],
                    row["prefix_cache_hit_rate"],
                    row["physical_gpus"],
                    row["tp"],
                    row["ep"],
                    row["dp"],
                    row["batch_size"],
                    row["batch_per_card"],
                    row["effective_prefill_len"],
                    row["weight_hbm_gb"],
                    row["kv_hbm_gb"],
                    row["hbm_total_gb"],
                    row["prefill_time_ms"],
                    row["prefill_qps_instance"],
                    row["prefill_tps_per_card"],
                ]
                for row in prefill_rows
            ],
        ),
        "",
        "## Decode 分析",
        "",
        "表中 `HBM B/card` 是不考虑 TPOT、只受 HBM 限制的最大单卡 batch；`TPOT B/card` 是同时满足 TPOT<=50ms 的最大单卡 batch。",
        "按当前建模，prefix cache 只影响 prefill compute，不影响 decode compute/HBM，因此 decode 曲线不按 hit rate 重复。",
        "",
    ])
    for scenario in SCENARIOS:
        rows = [row for row in decode_rows if row["scenario_id"] == scenario["scenario_id"]]
        lines.extend([
            f"### {scenario['label']}",
            "",
            f"![Decode {scenario['label']}](figure/decode_{scenario['scenario_id']}.svg)",
            "",
            _table(
                ["模式", "卡数", "HBM B/card", "TPOT B/card", "TP", "EP", "DP", "TPOT ms", "TPS/card", "QPS", "Best"],
                [
                    [
                        row["decode_mode_label"],
                        row["physical_gpus"],
                        row.get("max_batch_per_card_hbm"),
                        row.get("max_batch_per_card_tpot"),
                        row.get("tp"),
                        row.get("ep"),
                        row.get("dp"),
                        row.get("tpot_ms"),
                        row.get("decode_tps_per_card"),
                        row.get("decode_qps_instance"),
                        row.get("is_best_decode_instance", False),
                    ]
                    for row in rows
                ],
            ),
            "",
        ])
    lines.extend([
        "## P/D 配比",
        "",
        "![P/D Total Cards](figure/pd_ratio_total_cards.svg)",
        "",
        _table(
            ["场景", "Hit", "Decode 模式", "Prefill 卡/实例", "Decode 卡/实例", "P:D", "imbalance", "总卡数", "推荐"],
            [
                [
                    row["scenario_label"],
                    row.get("prefix_cache_hit_rate"),
                    row["decode_mode_label"],
                    row.get("prefill_physical_gpus"),
                    row.get("decode_physical_gpus"),
                    row.get("label"),
                    row.get("qps_imbalance_pct"),
                    row.get("total_cards"),
                    row.get("is_headline_recommendation", False),
                ]
                for row in pd_rows
            ],
        ),
        "",
        "## 建模边界",
        "",
        "- 未建模 quant/dequant kernel 时间。",
        "- Prefix cache 当前只降低 prefill compute，不降低 HBM；q_len/ctx_len 细分 attention 语义列为后续工作。",
        "- MTP 只按平均接收 token 数折算 forward 次数，未加入额外 head 权重或 MTP 专属计算开销。",
        "- 未建模 P/D KV transfer、排队、动态 batching、拓扑放置和 allocator fragmentation 以外的额外 HBM 损耗。",
    ])
    return "\n".join(lines) + "\n"


def generate_all() -> dict[str, Any]:
    base_cfg = load_910c_config()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    prefill_rows = generate_prefill_results(base_cfg)
    decode_results = generate_decode_results(base_cfg)
    pd_rows = generate_pd_ratio_results(prefill_rows, decode_results["best_by_scenario_mode"])
    generate_figures(prefill_rows, decode_results["rows"], pd_rows)

    scenario_spec = {
        "scenarios": SCENARIOS,
        "prefix_cache_hit_rates": PREFIX_CACHE_HIT_RATES,
        "decode_instance_sizes": DECODE_INSTANCE_SIZES,
        "prefill_card_counts": PREFILL_CARD_COUNTS,
        "decode_modes": DECODE_MODES,
        "report_defaults": REPORT_DEFAULTS,
    }
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "hardware_config_path": "configs/device_910C.json",
        "network_config_path": "configs/network_910C.json",
        "model_config_path": "configs/model_deepseekv4.json",
        "runtime_config_path": "configs/runtime_deepseekv4.json",
        "runtime_defaults": {
            "prefix_cache_hit_rates": PREFIX_CACHE_HIT_RATES,
            "quant_mode": REPORT_DEFAULTS["quant_mode"],
            "kv_cache_quant_mode": REPORT_DEFAULTS["kv_cache_quant_mode"],
        },
        "quantization_defaults": {
            "quant_mode": REPORT_DEFAULTS["quant_mode"],
            "kv_cache_quant_mode": REPORT_DEFAULTS["kv_cache_quant_mode"],
            "w8a8_tflops": REPORT_DEFAULTS["w8a8_tflops"],
        },
        "mtp_defaults": {
            "mtp_accept_ratio": REPORT_DEFAULTS["mtp_accept_ratio"],
        },
        "tpot_target_ms": REPORT_DEFAULTS["tpot_target_ms"],
    }

    write_json(DATA_DIR / "scenario_spec.json", scenario_spec)
    write_json(DATA_DIR / "prefill_results.json", prefill_rows)
    write_json(DATA_DIR / "decode_results.json", decode_results)
    write_json(DATA_DIR / "pd_ratio_results.json", pd_rows)
    write_json(DATA_DIR / "manifest.json", manifest)
    return {
        "prefill_rows": prefill_rows,
        "decode_results": decode_results,
        "pd_rows": pd_rows,
        "manifest": manifest,
    }


def main() -> None:
    generate_all()
    print(f"Wrote report data and figures under {REPORT_ROOT}")


if __name__ == "__main__":
    main()
