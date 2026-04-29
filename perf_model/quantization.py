"""Quantization-aware memory and op timing helpers."""

from dataclasses import replace

from .config import Config
from .layers import LayerProfile, PhaseProfile
from .memory import kv_cache_memory, weight_memory_per_rank
from .roofline import OpProfile, sum_ops


WEIGHT_BYTE_RATIOS = {"bf16": 1.0, "w8a8": 0.5}
KV_BYTE_RATIOS = {"bf16": 1.0, "kv8": 0.5, "kv4": 0.25}

GEMM_NAMES = {
    "q_proj_dq",
    "q_proj_uq",
    "kv_proj",
    "wo_a",
    "wo_b",
    "index_iq_proj",
    "moe_gate",
    "routed_gate_proj",
    "routed_up_proj",
    "routed_down_proj",
    "shared_gate_proj",
    "shared_up_proj",
    "shared_down_proj",
    "embedding",
    "lm_head",
}
ATTENTION_NAMES = {"attention_swa", "attention_comp"}
COMM_NAMES = {
    "attn_tp_allreduce",
    "moe_ep_dispatch",
    "moe_ep_combine",
    "sp_ag_before_attn",
    "sp_ag_before_moe",
    "sp_ag_after_moe",
    "sp_ag_before_lm_head",
    "index_score_ar",
}
VECTOR_PREFIXES = ("rmsnorm", "mhc_", "sinkhorn", "routed_silu", "shared_silu")


def quantized_weight_memory_per_rank(cfg: Config) -> dict:
    """Weight memory per rank after applying the runtime weight quantization mode."""
    cfg.rt.validate_serving_fields()
    base = weight_memory_per_rank(cfg)
    ratio = WEIGHT_BYTE_RATIOS[cfg.rt.quant_mode]
    result = dict(base)
    for key, value in list(result.items()):
        if isinstance(value, (int, float)) and key not in {"n_swa_layers", "n_comp_layers"}:
            result[key] = value * ratio
    result["total"] = base["total"] * ratio + cfg.rt.weight_scale_overhead_bytes
    result["quant_mode"] = cfg.rt.quant_mode
    return result


def quantized_kv_cache_memory(cfg: Config) -> dict:
    """KV cache memory after applying the runtime KV cache quantization mode."""
    cfg.rt.validate_serving_fields()
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


def infer_op_kind(name: str) -> str:
    """Infer quantization scope from the operation name family."""
    if name in GEMM_NAMES:
        return "gemm"
    if name in ATTENTION_NAMES:
        return "attention"
    if name in COMM_NAMES:
        return "comm"
    if name.startswith(VECTOR_PREFIXES):
        return "vector"
    return "other"


def _with_roofline_timings(
    op: OpProfile,
    cfg: Config,
    cube_tflops: float,
    mem_bytes: float,
    comm_time_s: float,
) -> OpProfile:
    flops = op.flops
    vec_ops = op.vec_ops
    cube_time = flops / (cube_tflops * 1e12 * cfg.hw.effective_cube_utilization) if flops > 0 else 0.0
    vec_time = vec_ops / (cfg.hw.vec_tflops * 1e12 * cfg.hw.effective_vec_utilization) if vec_ops > 0 else 0.0
    mem_time = mem_bytes / (cfg.hw.hbm_bandwidth_gbps * 1e9 * cfg.hw.hbm_bw_utilization) if mem_bytes > 0 else 0.0

    compute_time = max(cube_time, vec_time, mem_time)
    if compute_time == 0.0 and comm_time_s == 0.0:
        bottleneck = ""
    elif comm_time_s > compute_time:
        bottleneck = "COMM"
    elif cube_time >= vec_time and cube_time >= mem_time:
        bottleneck = "CUBE"
    elif vec_time >= mem_time:
        bottleneck = "VEC"
    else:
        bottleneck = "MEM"

    return OpProfile(
        name=op.name,
        flops=flops,
        vec_ops=vec_ops,
        mem_bytes=mem_bytes,
        comm_bytes=op.comm_bytes,
        cube_time_s=cube_time,
        vec_time_s=vec_time,
        mem_time_s=mem_time,
        comm_time_s=comm_time_s,
        time_s=compute_time + comm_time_s,
        bottleneck=bottleneck,
    )


def quantize_op_profile(op: OpProfile, cfg: Config) -> OpProfile:
    """Return a quantization-adjusted copy of an op profile."""
    cfg.rt.validate_serving_fields()
    if op.flops == op.vec_ops == op.mem_bytes == op.comm_time_s == 0 and op.time_s > 0:
        return replace(op)

    kind = infer_op_kind(op.name)
    if kind == "comm":
        return replace(op)

    cube_tflops = cfg.hw.cube_tflops
    mem_bytes = op.mem_bytes
    if kind == "gemm" and cfg.rt.quant_mode == "w8a8":
        cube_tflops = cfg.hw.effective_w8a8_tflops
        mem_bytes = op.mem_bytes * WEIGHT_BYTE_RATIOS[cfg.rt.quant_mode]
    elif kind == "attention":
        # Current attention profiles expose only aggregate memory, so the 0428
        # spec scales the full mem_bytes field by the KV cache byte ratio.
        mem_bytes = op.mem_bytes * KV_BYTE_RATIOS[cfg.rt.kv_cache_quant_mode]

    return _with_roofline_timings(op, cfg, cube_tflops, mem_bytes, op.comm_time_s)


def quantize_phase_profile(phase: PhaseProfile, cfg: Config) -> PhaseProfile:
    """Return a quantized copy of a phase profile with recomputed totals."""
    cfg.rt.validate_serving_fields()
    layer_profiles = []
    original_detailed_total = 0.0
    quantized_detailed_total = 0.0

    for layer in phase.layer_profiles:
        ops = [quantize_op_profile(op, cfg) for op in layer.ops]
        total = sum_ops(ops, f"layer_{layer.layer_idx}")
        layer_profiles.append(
            LayerProfile(
                layer_idx=layer.layer_idx,
                ratio=layer.ratio,
                ops=ops,
                total=total,
            )
        )
        if layer.total is not None:
            original_detailed_total += layer.total.time_s
        else:
            original_detailed_total += sum(op.time_s for op in layer.ops)
        quantized_detailed_total += total.time_s

    extra_ops = [quantize_op_profile(op, cfg) for op in phase.extra_ops]
    original_detailed_total += sum(op.time_s for op in phase.extra_ops)
    quantized_detailed_total += sum(op.time_s for op in extra_ops)

    if original_detailed_total > 0 and phase.total_time_s != original_detailed_total:
        total_time_s = phase.total_time_s * quantized_detailed_total / original_detailed_total
    else:
        total_time_s = quantized_detailed_total

    return PhaseProfile(
        phase=phase.phase,
        layer_profiles=layer_profiles,
        extra_ops=extra_ops,
        total_time_s=total_time_s,
        total_tokens=phase.total_tokens,
    )
