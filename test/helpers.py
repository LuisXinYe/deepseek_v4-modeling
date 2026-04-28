"""Shared config builders, assertion helpers, and tolerances for tests."""

import os
import sys

# Ensure project root is on sys.path so `perf_model` can be imported
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from perf_model.config import Config, HardwareConfig, NetworkConfig, ModelConfig, RuntimeConfig

ATOL = 1e-9
RTOL = 1e-6


def make_config(**overrides) -> Config:
    """Build a small deterministic Config for fast unit tests.

    4 layers with compress_ratios [1, 4, 128, 4], hidden=256, 8 heads,
    head_dim=64, tp=2, ep=4, dp=1, bs=2, seq=128, 16 experts, top-2.
    Accepts keyword overrides for any runtime/model field.
    """
    hw_overrides = {}
    net_overrides = {}
    model_overrides = {}
    rt_overrides = {}

    # Map overrides to sub-config by checking field names
    hw_fields = {f.name for f in HardwareConfig.__dataclass_fields__.values()}
    net_fields = {f.name for f in NetworkConfig.__dataclass_fields__.values()}
    model_fields = {f.name for f in ModelConfig.__dataclass_fields__.values()}
    rt_fields = {f.name for f in RuntimeConfig.__dataclass_fields__.values()}

    for k, v in overrides.items():
        if k in hw_fields:
            hw_overrides[k] = v
        elif k in net_fields:
            net_overrides[k] = v
        elif k in model_fields:
            model_overrides[k] = v
        elif k in rt_fields:
            rt_overrides[k] = v
        else:
            raise ValueError(f"Unknown config field: {k}")

    hw_defaults = dict(
        cube_tflops=376,
        vec_tflops=24,
        hbm_capacity_gb=64,
        hbm_reserved_pct=10.0,
        hbm_bandwidth_gbps=1800,
        flops_utilization=0.5,
        hbm_bw_utilization=0.8,
    )
    hw_defaults.update(hw_overrides)

    net_defaults = dict(
        tp_bandwidth_gbps=392,
        ep_bandwidth_gbps=392,
        latency_us=10,
        bandwidth_utilization=0.8,
    )
    net_defaults.update(net_overrides)

    model_defaults = dict(
        hidden_size=256,
        num_hidden_layers=4,
        vocab_size=1024,
        num_attention_heads=8,
        head_dim=64,
        rope_head_dim=16,
        q_lora_rank=128,
        o_groups=2,
        o_lora_rank=64,
        index_n_heads=8,
        index_head_dim=32,
        index_topk=16,
        window_size=16,
        compress_ratios=[1, 4, 128, 4],
        hc_mult=4,
        n_routed_experts=16,
        num_experts_per_tok=2,
        n_shared_experts=1,
        moe_inter_dim=512,
        n_hash_layers=1,
    )
    model_defaults.update(model_overrides)

    rt_defaults = dict(
        seq_len=128,
        batch_size=2,
        dp=1,
        tp=2,
        ep=4,
        sp=True,
        moe_load_balance_factor=1.2,
        output_len=8,
        shared_expert_overlapped=True,
        mhc_sp=False,
        mhc_kernel_fused=True,
        mhc_fused_bf16=False,
        input_len=None,
        decode_context_len=None,
        prefix_cache_hit_rate=0.0,
    )
    rt_defaults.update(rt_overrides)

    return Config(
        hw=HardwareConfig(**hw_defaults),
        net=NetworkConfig(**net_defaults),
        model=ModelConfig(**model_defaults),
        rt=RuntimeConfig(**rt_defaults),
    )


def make_prod_config() -> Config:
    """Load the real production config from JSON files."""
    configs_dir = os.path.join(_PROJECT_ROOT, "configs")
    return Config.from_json(
        os.path.join(configs_dir, "device_910C.json"),
        os.path.join(configs_dir, "network_910C.json"),
        os.path.join(configs_dir, "model_deepseekv4.json"),
        os.path.join(configs_dir, "runtime_deepseekv4.json"),
    )


def assert_op_valid(test_case, op):
    """Assert an OpProfile has valid invariants."""
    test_case.assertGreaterEqual(op.cube_time_s, 0, f"{op.name}: cube_time < 0")
    test_case.assertGreaterEqual(op.vec_time_s, 0, f"{op.name}: vec_time < 0")
    test_case.assertGreaterEqual(op.mem_time_s, 0, f"{op.name}: mem_time < 0")
    test_case.assertGreaterEqual(op.comm_time_s, 0, f"{op.name}: comm_time < 0")
    test_case.assertGreaterEqual(op.time_s, 0, f"{op.name}: time_s < 0")

    compute = max(op.cube_time_s, op.vec_time_s, op.mem_time_s)
    expected_total = compute + op.comm_time_s
    test_case.assertAlmostEqual(
        op.time_s, expected_total, places=12,
        msg=f"{op.name}: time_s={op.time_s} != max(cube,vec,mem)+comm={expected_total}"
    )

    valid_bottlenecks = {"CUBE", "VEC", "MEM", "COMM", ""}
    test_case.assertIn(op.bottleneck, valid_bottlenecks,
                       f"{op.name}: invalid bottleneck '{op.bottleneck}'")
