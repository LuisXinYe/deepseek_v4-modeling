"""Shared helpers for the 2026-04-28 PD disaggregation report."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perf_model.config import Config  # noqa: E402


REPORT_ROOT = REPO_ROOT / "report" / "0428"
DATA_DIR = REPORT_ROOT / "data"
FIGURE_DIR = REPORT_ROOT / "figure"

SCENARIOS = [
    {"scenario_id": "8k_1k", "label": "8K + 1K", "input_len": 8192, "output_len": 1024},
    {"scenario_id": "32k_1k", "label": "32K + 1K", "input_len": 32768, "output_len": 1024},
    {"scenario_id": "128k_1k", "label": "128K + 1K", "input_len": 131072, "output_len": 1024},
    {"scenario_id": "1m_1k", "label": "1M + 1K", "input_len": 1_000_000, "output_len": 1024},
]

DECODE_INSTANCE_SIZES = [8, 16, 32, 64]
PREFILL_CARD_COUNTS = [1, 2, 4, 8, 16, 32, 64]
TP_VALUES = [1, 2, 4, 8, 16, 32, 64]
EP_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
PREFIX_CACHE_HIT_RATES = [0.0, 0.9, 0.99]

REPORT_DEFAULTS = {
    "prefix_cache_hit_rate": 0.0,
    "quant_mode": "w8a8",
    "kv_cache_quant_mode": "kv8",
    "mtp_accept_ratio": 0.9,
    "tpot_target_ms": 50.0,
    "w8a8_tflops": 752.0,
    "weight_scale_overhead_bytes": 0.0,
    "kv_scale_overhead_bytes": 0.0,
}


def load_910c_config() -> Config:
    cfg = Config.from_json(
        str(REPO_ROOT / "configs" / "device_910C.json"),
        str(REPO_ROOT / "configs" / "network_910C.json"),
        str(REPO_ROOT / "configs" / "model_deepseekv4.json"),
        str(REPO_ROOT / "configs" / "runtime_deepseekv4.json"),
    )
    return replace(cfg, hw=replace(cfg.hw, w8a8_tflops=REPORT_DEFAULTS["w8a8_tflops"]))


def with_runtime_defaults(
    cfg: Config,
    *,
    input_len: int,
    output_len: int,
    batch_size: int,
    tp: int,
    ep: int,
    dp: int,
    mtp: int = 0,
    prefix_cache_hit_rate: float | None = None,
) -> Config:
    hit_rate = REPORT_DEFAULTS["prefix_cache_hit_rate"] if prefix_cache_hit_rate is None else prefix_cache_hit_rate
    rt = replace(
        cfg.rt,
        seq_len=input_len,
        input_len=input_len,
        decode_context_len=input_len,
        output_len=output_len,
        batch_size=batch_size,
        tp=tp,
        ep=ep,
        dp=dp,
        prefix_cache_hit_rate=hit_rate,
        mtp=mtp,
        mtp_accept_ratio=REPORT_DEFAULTS["mtp_accept_ratio"],
        quant_mode=REPORT_DEFAULTS["quant_mode"],
        kv_cache_quant_mode=REPORT_DEFAULTS["kv_cache_quant_mode"],
        weight_scale_overhead_bytes=REPORT_DEFAULTS["weight_scale_overhead_bytes"],
        kv_scale_overhead_bytes=REPORT_DEFAULTS["kv_scale_overhead_bytes"],
    )
    return replace(cfg, rt=rt)


def is_parallel_valid(cfg: Config, *, tp: int, ep: int, dp: int, batch_size: int) -> bool:
    if tp <= 0 or ep <= 0 or dp <= 0 or batch_size <= 0:
        return False
    physical_gpus = tp * dp
    if physical_gpus % ep != 0:
        return False
    if batch_size % dp != 0:
        return False
    model = cfg.model
    if model.num_attention_heads % tp != 0:
        return False
    if model.num_attention_heads % model.o_groups != 0:
        return False
    if model.o_groups % tp != 0:
        return False
    if model.index_n_heads % tp != 0:
        return False
    if model.vocab_size % tp != 0:
        return False
    if model.n_routed_experts % ep != 0:
        return False
    return True


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def infeasible_row(scenario_id: str, invalid_reason: str, **fields: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "scenario_id": scenario_id,
        "is_feasible": False,
        "invalid_reason": invalid_reason,
        "decode_tps_per_card": None,
    }
    row.update(fields)
    return row
