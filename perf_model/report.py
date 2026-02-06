"""Formatting and printing functions."""

from typing import List, Optional

from .config import Config
from .roofline import OpProfile
from .layers import LayerProfile, PhaseProfile
from .memory import kv_cache_memory, weight_memory_per_rank


def fmt_bytes(b: float) -> str:
    """Format bytes to human-readable."""
    if b >= 1e9:
        return f"{b / 1e9:.2f} GB"
    elif b >= 1e6:
        return f"{b / 1e6:.2f} MB"
    elif b >= 1e3:
        return f"{b / 1e3:.2f} KB"
    return f"{b:.0f} B"


def fmt_ms(s: float) -> str:
    """Format seconds to ms."""
    return f"{s * 1000:.3f}"


def print_op_table(ops: List[OpProfile], indent: str = "  "):
    """Print per-op table with roofline breakdown."""
    header = f"{indent}{'Operation':<25s} | {'Cube(ms)':>9s} | {'Vec(ms)':>9s} | {'Mem(ms)':>9s} | {'Comm(ms)':>9s} | {'Total(ms)':>10s} | {'Bound':<5s}"
    sep = indent + "-" * (len(header) - len(indent))
    print(header)
    print(sep)
    for op in ops:
        if op.time_s == 0 and op.name in ("kv_compression", "kv_compression_decode",
                                            "index_kv_compress"):
            # Skip zero placeholders in display (still show name)
            print(f"{indent}{op.name:<25s} | {'---':>9s} | {'---':>9s} | {'---':>9s} | {'---':>9s} | {'PLACEHOLDER':>10s} | {'---':<5s}")
            continue
        print(f"{indent}{op.name:<25s} | {fmt_ms(op.cube_time_s):>9s} | {fmt_ms(op.vec_time_s):>9s} | {fmt_ms(op.mem_time_s):>9s} | {fmt_ms(op.comm_time_s):>9s} | {fmt_ms(op.time_s):>10s} | {op.bottleneck:<5s}")


def print_layer_summary(layer_profiles: List[LayerProfile], indent: str = "  "):
    """Print summary table of all layers."""
    header = f"{indent}{'Layer':>5s} | {'Ratio':>5s} | {'Time(ms)':>10s} | {'Bound':<5s}"
    sep = indent + "-" * (len(header) - len(indent))
    print(header)
    print(sep)
    for lp in layer_profiles:
        print(f"{indent}{lp.layer_idx:>5d} | {lp.ratio:>5d} | {fmt_ms(lp.total.time_s):>10s} | {lp.total.bottleneck:<5s}")


def print_config_summary(cfg: Config):
    """Print configuration summary."""
    print("=" * 80)
    print("DeepSeek V4 Inference Performance Model")
    print("=" * 80)
    print()
    print("Hardware (per die):")
    print(f"  BF16 Cube TFLOPS:    {cfg.hw.bf16_tflops}")
    print(f"  BF16 Vector TFLOPS:  {cfg.hw.vec_tflops}")
    print(f"  HBM Capacity:        {cfg.hw.hbm_capacity_gb} GB")
    print(f"  HBM Bandwidth:       {cfg.hw.hbm_bandwidth_gbps} GB/s")
    print(f"  FLOPS Utilization:   {cfg.hw.flops_utilization:.0%}")
    print(f"  HBM BW Utilization:  {cfg.hw.hbm_bw_utilization:.0%}")
    print()
    print("Network:")
    print(f"  TP Bandwidth:        {cfg.net.tp_bandwidth_gbps} GB/s (bidirectional)")
    print(f"  EP Bandwidth:        {cfg.net.ep_bandwidth_gbps} GB/s (bidirectional)")
    print(f"  Latency:             {cfg.net.latency_us} us")
    print(f"  BW Utilization:      {cfg.net.bandwidth_utilization:.0%}")
    print()
    print("Model:")
    print(f"  Hidden size:         {cfg.model.hidden_size}")
    print(f"  Layers:              {cfg.model.num_layers}")
    print(f"  Q heads:             {cfg.model.num_q_heads}, KV heads: {cfg.model.num_kv_heads} (MQA)")
    print(f"  Q content dim:       {cfg.model.q_content_dim}, RoPE dim: {cfg.model.rope_head_dim}")
    print(f"  K dim: {cfg.model.k_dim}, V dim: {cfg.model.v_dim}")
    print(f"  Q LoRA rank:         {cfg.model.q_lora_rank}, O LoRA rank: {cfg.model.o_lora_rank}")
    print(f"  O groups:            {cfg.model.o_num_groups}, O mid dim: {cfg.model.o_mid_dim}")
    print(f"  Index heads:         {cfg.model.index_n_heads}, dim: {cfg.model.index_head_dim}, topK: {cfg.model.index_topk}")
    print(f"  SWA window:          {cfg.model.swa_window}")
    print(f"  Compress C_k:        {cfg.model.compress_c_k}, C_v: {cfg.model.compress_c_v}")
    print(f"  mHC mult:            {cfg.model.mhc_mult}")
    print(f"  MoE: {cfg.model.n_routed_experts} routed experts, top-{cfg.model.num_experts_per_tok}, "
          f"{cfg.model.n_shared_experts} shared, inter={cfg.model.moe_inter_dim}")
    print(f"  Hash routing layers: {cfg.model.n_hash_layers}")
    print()

    # Count layer types
    ratios = cfg.model.compress_ratios
    r1 = sum(1 for r in ratios if r == 1)
    r4 = sum(1 for r in ratios if r == 4)
    r128 = sum(1 for r in ratios if r == 128)
    print(f"  Layer types:         {r1} full-attn (ratio=1), {r4} C4A, {r128} C128A")
    print()

    print("Runtime:")
    print(f"  Seq len:             {cfg.rt.seq_len}")
    print(f"  Batch size:          {cfg.rt.batch_size}")
    print(f"  TP: {cfg.rt.tp}, EP: {cfg.rt.ep}, SP: {cfg.rt.sp}")
    print(f"  MoE load balance:    {cfg.rt.moe_load_balance_factor}")
    print(f"  Output len:          {cfg.rt.output_len}")
    print(f"  Shared expert overlap: {cfg.rt.shared_expert_overlapped}")
    print()


def find_representative_layers(cfg: Config) -> List[int]:
    """Find representative layer indices: one ratio=1, one C4A, one C128A."""
    reps = []
    seen = set()
    for i, r in enumerate(cfg.model.compress_ratios):
        if r not in seen:
            reps.append(i)
            seen.add(r)
        if len(seen) >= 3:
            break
    return reps


def print_phase_report(phase: PhaseProfile, cfg: Config, detailed_step: Optional[PhaseProfile] = None):
    """Print full phase report."""
    is_decode = phase.phase.startswith("decode")
    phase_name = "DECODE" if is_decode else "PREFILL"

    print("=" * 80)
    print(f"  {phase_name} PHASE")
    print("=" * 80)
    print()

    # Use detailed_step for per-op breakdown if provided
    detail_phase = detailed_step if detailed_step else phase

    # Representative layers detailed
    rep_indices = find_representative_layers(cfg)
    for idx in rep_indices:
        if idx >= len(detail_phase.layer_profiles):
            continue
        lp = detail_phase.layer_profiles[idx]
        ratio = lp.ratio
        label = "Full Attention" if ratio == 1 else f"C{ratio}A"
        print(f"  Layer {idx} ({label}, ratio={ratio}):")
        print_op_table(lp.ops, indent="    ")
        print(f"    {'TOTAL':<25s} | {fmt_ms(lp.total.cube_time_s):>9s} | {fmt_ms(lp.total.vec_time_s):>9s} | {fmt_ms(lp.total.mem_time_s):>9s} | {fmt_ms(lp.total.comm_time_s):>9s} | {fmt_ms(lp.total.time_s):>10s} | {lp.total.bottleneck:<5s}")
        print()

    # Layer summary table
    print("  Layer Summary:")
    print_layer_summary(detail_phase.layer_profiles, indent="    ")
    print()

    # Extra ops
    if detail_phase.extra_ops:
        print("  Non-layer ops:")
        print_op_table(detail_phase.extra_ops, indent="    ")
        extra_time = sum(op.time_s for op in detail_phase.extra_ops)
        print(f"    Extra ops total: {fmt_ms(extra_time)} ms")
        print()

    # Totals
    if is_decode and detailed_step:
        step_time = detailed_step.total_time_s
        print(f"  Single decode step (context={cfg.rt.seq_len}): {fmt_ms(step_time)} ms")
        print(f"  Total decode ({cfg.rt.output_len} tokens): {fmt_ms(phase.total_time_s)} ms")
        print(f"  Avg tokens/s: {phase.total_tokens / phase.total_time_s:.1f}")
    else:
        print(f"  Total {phase_name.lower()} time: {fmt_ms(phase.total_time_s)} ms")
        print(f"  Tokens processed: {phase.total_tokens}")
        if phase.total_time_s > 0:
            print(f"  Throughput: {phase.total_tokens / phase.total_time_s:.1f} tokens/s")
    print()


def print_memory_report(cfg: Config):
    """Print memory analysis."""
    print("=" * 80)
    print("  MEMORY ANALYSIS")
    print("=" * 80)
    print()

    # KV Cache
    kv = kv_cache_memory(cfg)
    print(f"  KV Cache (B={cfg.rt.batch_size}, S={cfg.rt.seq_len}):")
    # Show a few representative layers
    for i in find_representative_layers(cfg):
        info = kv["layers"][i]
        print(f"    Layer {i} ({info['type']}): {fmt_bytes(info['bytes'])}", end="")
        if info["type"] != "full":
            parts = f"comp={fmt_bytes(info['comp_bytes'])}, swa={fmt_bytes(info['swa_bytes'])}"
            if "idx_bytes" in info:
                parts += f", idx={fmt_bytes(info['idx_bytes'])}"
            print(f"  ({parts})", end="")
        print()
    print(f"    Total KV cache: {fmt_bytes(kv['total_bytes'])}")
    print()

    # Weight Memory
    wm = weight_memory_per_rank(cfg)
    print("  Weight Memory (per rank):")
    print(f"    Attention (all layers): {fmt_bytes(wm['total_attn'])}")
    print(f"      Per layer (base):    {fmt_bytes(wm['attn_per_layer'])}")
    print(f"      Per layer (index):   {fmt_bytes(wm['index_per_layer'])} (only index layers)")
    print(f"    MoE (all layers):      {fmt_bytes(wm['total_moe'])}")
    print(f"      Per layer:           {fmt_bytes(wm['moe_per_layer'])}")
    print(f"    Other (mHC+norm+emb):  {fmt_bytes(wm['total_other'])}")
    print(f"    Total weights:         {fmt_bytes(wm['total'])}")
    print()

    # Total HBM
    total_hbm = wm["total"] + kv["total_bytes"]
    capacity = cfg.hw.hbm_capacity_gb * 1e9
    print(f"  Total HBM Usage:         {fmt_bytes(total_hbm)}")
    print(f"  HBM Capacity:            {fmt_bytes(capacity)}")
    pct = total_hbm / capacity * 100
    print(f"  Utilization:             {pct:.1f}%")
    if total_hbm > capacity:
        print(f"  WARNING: Exceeds HBM capacity by {fmt_bytes(total_hbm - capacity)}!")
    print()
