import unittest

from test.helpers import make_config
from perf_model.config import HardwareConfig
from perf_model.layers import LayerProfile, PhaseProfile, decode_model
from perf_model.roofline import OpProfile, roofline_time, sum_ops
from perf_model.quantization import (
    infer_op_kind,
    quantized_weight_memory_per_rank,
    quantized_kv_cache_memory,
    quantize_op_profile,
    quantize_phase_profile,
    _with_roofline_timings,
)


class TestQuantization(unittest.TestCase):
    def test_weight_and_kv_ratios(self):
        bf16 = make_config(quant_mode="bf16", kv_cache_quant_mode="bf16")
        w8kv8 = make_config(quant_mode="w8a8", kv_cache_quant_mode="kv8")

        w_bf16 = quantized_weight_memory_per_rank(bf16)["total"]
        w_w8 = quantized_weight_memory_per_rank(w8kv8)["total"]
        kv_bf16 = quantized_kv_cache_memory(bf16)["total_bytes"]
        kv_kv8 = quantized_kv_cache_memory(w8kv8)["total_bytes"]

        self.assertAlmostEqual(w_w8 / w_bf16, 0.5, places=6)
        self.assertAlmostEqual(kv_kv8 / kv_bf16, 0.5, places=6)

    def test_infer_op_kind(self):
        self.assertEqual(infer_op_kind("q_proj_dq"), "gemm")
        self.assertEqual(infer_op_kind("attention_swa"), "attention")
        self.assertEqual(infer_op_kind("attn_tp_allreduce"), "comm")
        self.assertEqual(infer_op_kind("rmsnorm_attn"), "vector")

    def test_w8a8_changes_gemm_not_comm(self):
        bf16 = make_config(quant_mode="bf16")
        w8 = make_config(quant_mode="w8a8")
        op = OpProfile(
            name="q_proj_dq",
            flops=10**12,
            mem_bytes=10**9,
            cube_time_s=1.0,
            mem_time_s=1.0,
            time_s=1.0,
            bottleneck="CUBE",
        )
        q_bf16 = quantize_op_profile(op, bf16)
        q_w8 = quantize_op_profile(op, w8)
        self.assertLess(q_w8.cube_time_s, q_bf16.cube_time_s)
        self.assertLess(q_w8.mem_bytes, q_bf16.mem_bytes)
        self.assertLess(q_w8.mem_time_s, q_bf16.mem_time_s)

        comm = OpProfile(name="attn_tp_allreduce", comm_time_s=0.01, time_s=0.01, bottleneck="COMM")
        self.assertEqual(quantize_op_profile(comm, w8).time_s, comm.time_s)

    def test_kv4_and_scale_overheads(self):
        bf16 = make_config(quant_mode="bf16", kv_cache_quant_mode="bf16")
        q = make_config(
            quant_mode="w8a8",
            kv_cache_quant_mode="kv4",
            weight_scale_overhead_bytes=123,
            kv_scale_overhead_bytes=456,
        )

        w_base = quantized_weight_memory_per_rank(bf16)
        w_q = quantized_weight_memory_per_rank(q)
        kv_base = quantized_kv_cache_memory(bf16)
        kv_q = quantized_kv_cache_memory(q)

        self.assertEqual(w_q["quant_mode"], "w8a8")
        self.assertEqual(kv_q["kv_cache_quant_mode"], "kv4")
        self.assertAlmostEqual(w_q["total"], w_base["total"] * 0.5 + 123)
        self.assertAlmostEqual(kv_q["total_bytes"], kv_base["total_bytes"] * 0.25 + 456)

    def test_attention_uses_kv_memory_ratio_without_compute_acceleration(self):
        cfg = make_config(
            quant_mode="w8a8",
            kv_cache_quant_mode="kv4",
            cube_utilization=0.25,
            vec_utilization=0.1,
        )
        op = OpProfile(
            name="attention_comp",
            flops=10**12,
            vec_ops=10**11,
            mem_bytes=10**9,
            comm_time_s=0.02,
            time_s=1.0,
            bottleneck="CUBE",
        )

        q = quantize_op_profile(op, cfg)

        self.assertEqual(q.flops, op.flops)
        self.assertEqual(q.comm_time_s, op.comm_time_s)
        self.assertAlmostEqual(q.mem_bytes, op.mem_bytes * 0.25)
        self.assertAlmostEqual(
            q.cube_time_s,
            op.flops / (cfg.hw.cube_tflops * 1e12 * cfg.hw.effective_cube_utilization),
        )
        expected_vec = (op.vec_ops / (cfg.hw.vec_tflops * 1e12 * cfg.hw.effective_vec_utilization)
                        + cfg.hw.vec_static_latency_us * 1e-6)
        self.assertAlmostEqual(q.vec_time_s, expected_vec)

    def test_phase_quantization_copies_and_recomputes_totals(self):
        cfg = make_config(quant_mode="w8a8", kv_cache_quant_mode="kv8")
        gemm = OpProfile(name="q_proj_dq", flops=10**12, mem_bytes=10**9)
        attn = OpProfile(name="attention_swa", flops=10**11, mem_bytes=10**8)
        comm = OpProfile(name="moe_ep_dispatch", comm_time_s=0.01, time_s=0.01, bottleneck="COMM")
        layer = LayerProfile(layer_idx=0, ratio=1, ops=[gemm, comm])
        layer.total = sum_ops(layer.ops, "layer_0")
        phase = PhaseProfile(
            phase="prefill",
            layer_profiles=[layer],
            extra_ops=[attn],
            total_time_s=layer.total.time_s + attn.time_s,
            total_tokens=2,
        )

        q_phase = quantize_phase_profile(phase, cfg)

        self.assertIsNot(q_phase, phase)
        self.assertIsNot(q_phase.layer_profiles[0], layer)
        self.assertIsNot(q_phase.layer_profiles[0].ops[0], gemm)
        self.assertEqual(phase.layer_profiles[0].ops[0].mem_bytes, 10**9)
        self.assertAlmostEqual(q_phase.layer_profiles[0].ops[0].mem_bytes, 5 * 10**8)
        self.assertAlmostEqual(q_phase.extra_ops[0].mem_bytes, 5 * 10**7)
        expected_total = q_phase.layer_profiles[0].total.time_s + q_phase.extra_ops[0].time_s
        self.assertAlmostEqual(q_phase.total_time_s, expected_total)
        self.assertEqual(q_phase.phase, "prefill")
        self.assertEqual(q_phase.total_tokens, 2)

    def test_bf16_decode_total_time_is_preserved_for_aggregate_phase(self):
        cfg = make_config(quant_mode="bf16", kv_cache_quant_mode="bf16", output_len=300)
        phase = decode_model(cfg)

        q_phase = quantize_phase_profile(phase, cfg)

        self.assertEqual(phase.phase, "decode_total")
        self.assertNotAlmostEqual(
            phase.total_time_s,
            sum(lp.total.time_s for lp in phase.layer_profiles) + sum(op.time_s for op in phase.extra_ops),
        )
        self.assertAlmostEqual(q_phase.total_time_s, phase.total_time_s)

    def test_manually_timed_op_is_preserved(self):
        cfg = make_config(quant_mode="w8a8", kv_cache_quant_mode="kv8")
        op = OpProfile(name="shared_expert_excess", time_s=0.123)

        q_op = quantize_op_profile(op, cfg)

        self.assertEqual(q_op, op)


class TestWithRooflineTimingsFormula(unittest.TestCase):
    """_with_roofline_timings() must match roofline_time() formula exactly."""

    def test_matches_roofline_time_for_cube_op(self):
        hw = HardwareConfig(
            cube_tflops=100, vec_tflops=10,
            hbm_bandwidth_gbps=1000,
            flops_utilization=0.5, hbm_bw_utilization=0.8,
            vec_static_latency_us=5.0,
        )
        cfg = make_config(
            cube_tflops=100, vec_tflops=10,
            hbm_bandwidth_gbps=1000,
            flops_utilization=0.5, hbm_bw_utilization=0.8,
            vec_static_latency_us=5.0,
        )
        flops, vec_ops, mem_bytes = 1e12, 1e11, 1e9
        op = OpProfile(name="test_op", flops=flops, vec_ops=vec_ops, mem_bytes=mem_bytes)
        ref = roofline_time("test_op", flops=flops, vec_ops=vec_ops, mem_bytes=mem_bytes, hw=hw)
        result = _with_roofline_timings(op, cfg, hw.cube_tflops, mem_bytes, 0.0)
        self.assertAlmostEqual(result.time_s, ref.time_s, places=12)
        self.assertAlmostEqual(result.cube_time_s, ref.cube_time_s, places=12)
        self.assertAlmostEqual(result.vec_time_s, ref.vec_time_s, places=12)
        self.assertAlmostEqual(result.mem_time_s, ref.mem_time_s, places=12)
        self.assertEqual(result.bottleneck, ref.bottleneck)

    def test_vec_static_not_added_when_vec_ops_zero(self):
        cfg = make_config(vec_static_latency_us=10.0)
        op = OpProfile(name="no_vec", flops=1e12, vec_ops=0, mem_bytes=1e9)
        result = _with_roofline_timings(op, cfg, cfg.hw.cube_tflops, op.mem_bytes, 0.0)
        self.assertEqual(result.vec_time_s, 0.0)


class TestQuantizePhaseProfilePhaseContext(unittest.TestCase):
    """quantize_phase_profile() applies phase utilization context."""

    def test_prefill_phase_applies_prefill_utilization(self):
        cfg_full = make_config(quant_mode="bf16", prefill_utilization=1.0)
        cfg_low = make_config(quant_mode="bf16", prefill_utilization=0.5)
        gemm = OpProfile(name="q_proj_dq", flops=1e12, mem_bytes=1e9)
        layer = LayerProfile(layer_idx=0, ratio=1, ops=[gemm])
        layer.total = sum_ops(layer.ops, "layer_0")
        phase = PhaseProfile(
            phase="prefill",
            layer_profiles=[layer],
            extra_ops=[],
            total_time_s=layer.total.time_s,
            total_tokens=2,
        )
        q_full = quantize_phase_profile(phase, cfg_full)
        q_low = quantize_phase_profile(phase, cfg_low)
        self.assertGreater(q_low.layer_profiles[0].ops[0].time_s,
                           q_full.layer_profiles[0].ops[0].time_s)

    def test_decode_phase_applies_decode_utilization(self):
        cfg_full = make_config(quant_mode="bf16", decode_utilization=1.0)
        cfg_low = make_config(quant_mode="bf16", decode_utilization=0.5)
        gemm = OpProfile(name="q_proj_dq", flops=1e12, mem_bytes=1e9)
        layer = LayerProfile(layer_idx=0, ratio=1, ops=[gemm])
        layer.total = sum_ops(layer.ops, "layer_0")
        phase = PhaseProfile(
            phase="decode_step@512",
            layer_profiles=[layer],
            extra_ops=[],
            total_time_s=layer.total.time_s,
            total_tokens=1,
        )
        q_full = quantize_phase_profile(phase, cfg_full)
        q_low = quantize_phase_profile(phase, cfg_low)
        self.assertGreater(q_low.layer_profiles[0].ops[0].time_s,
                           q_full.layer_profiles[0].ops[0].time_s)


if __name__ == "__main__":
    unittest.main()
