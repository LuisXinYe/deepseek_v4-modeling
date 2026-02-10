"""Tests for perf_model.layers — layer/phase aggregation."""

import unittest

from test.helpers import make_config, assert_op_valid

from perf_model.roofline import OpProfile, sum_ops
from perf_model.layers import (
    LayerProfile,
    PhaseProfile,
    prefill_layer,
    decode_layer,
    prefill_model,
    decode_step,
    decode_model,
    _compression_period,
)


# ── LayerProfile ──────────────────────────────────────────────────────────

class TestLayerProfile(unittest.TestCase):
    """LayerProfile.compute_total and basic invariants."""

    def test_compute_total_aggregates_ops(self):
        """compute_total sums all op times correctly."""
        ops = [
            OpProfile(name="a", cube_time_s=1.0, vec_time_s=0.5, mem_time_s=0.2,
                      comm_time_s=0.1, time_s=1.1, bottleneck="CUBE"),
            OpProfile(name="b", cube_time_s=0.3, vec_time_s=0.8, mem_time_s=0.1,
                      comm_time_s=0.2, time_s=1.0, bottleneck="VEC"),
        ]
        lp = LayerProfile(layer_idx=5, ratio=4, ops=ops)
        lp.compute_total()
        self.assertIsNotNone(lp.total)
        self.assertAlmostEqual(lp.total.time_s, 1.1 + 1.0, places=10)
        self.assertAlmostEqual(lp.total.cube_time_s, 1.3, places=10)
        self.assertAlmostEqual(lp.total.vec_time_s, 1.3, places=10)
        self.assertAlmostEqual(lp.total.mem_time_s, 0.3, places=10)
        self.assertAlmostEqual(lp.total.comm_time_s, 0.3, places=10)

    def test_name_format(self):
        """compute_total names the aggregate 'layer_{idx}'."""
        lp = LayerProfile(layer_idx=7, ratio=1, ops=[
            OpProfile(name="x", time_s=0.01, cube_time_s=0.01, bottleneck="CUBE"),
        ])
        lp.compute_total()
        self.assertEqual(lp.total.name, "layer_7")

    def test_empty_ops_list(self):
        """Empty ops list produces total with all zeros."""
        lp = LayerProfile(layer_idx=0, ratio=1, ops=[])
        lp.compute_total()
        self.assertEqual(lp.total.time_s, 0.0)
        self.assertEqual(lp.total.cube_time_s, 0.0)
        self.assertEqual(lp.total.vec_time_s, 0.0)
        self.assertEqual(lp.total.mem_time_s, 0.0)
        self.assertEqual(lp.total.comm_time_s, 0.0)
        self.assertEqual(lp.total.bottleneck, "")


# ── Prefill Layer ─────────────────────────────────────────────────────────

class TestPrefillLayer(unittest.TestCase):
    """prefill_layer for various layer types and config options."""

    def test_ratio1_has_swa_attention(self):
        """ratio=1 (layer 0): has attention_swa, no index/compression ops."""
        cfg = make_config()
        lp = prefill_layer(0, cfg)
        self.assertEqual(lp.ratio, 1)
        names = [op.name for op in lp.ops]
        self.assertIn("attention_swa", names)
        self.assertNotIn("attention_comp", names)
        self.assertNotIn("index_iq_proj", names)
        self.assertNotIn("index_kv_compress", names)
        self.assertNotIn("kv_compression", names)

    def test_ratio4_has_index_and_compression(self):
        """ratio=4 (layer 1): has SWA + index + compression + compressed attention."""
        cfg = make_config()
        lp = prefill_layer(1, cfg)
        self.assertEqual(lp.ratio, 4)
        names = [op.name for op in lp.ops]
        self.assertIn("attention_swa", names)
        self.assertIn("index_iq_proj", names)
        self.assertIn("index_kv_compress", names)
        self.assertIn("kv_compression", names)
        self.assertIn("attention_comp", names)

    def test_ratio128_no_index(self):
        """ratio=128 (layer 2): use_index is hardcoded as (ratio==4), so no index ops.
        S=128, ratio=128 => S_comp=1. Regardless, prefill uses ratio==4 check."""
        cfg = make_config()
        lp = prefill_layer(2, cfg)
        self.assertEqual(lp.ratio, 128)
        names = [op.name for op in lp.ops]
        # No index ops (use_index = ratio==4 is False for ratio=128)
        self.assertNotIn("index_iq_proj", names)
        self.assertNotIn("index_kv_compress", names)
        # But SWA + KV compression + compressed attention ARE present
        self.assertIn("attention_swa", names)
        self.assertIn("kv_compression", names)
        self.assertIn("attention_comp", names)

    def test_fused_mhc_ops(self):
        """Fused mHC produces 3 fused ops: pre_attn, post_attn_pre_moe, post_moe."""
        cfg = make_config(mhc_kernel_fused=True)
        lp = prefill_layer(0, cfg)
        names = [op.name for op in lp.ops]
        self.assertIn("mhc_pre_attn", names)
        self.assertIn("mhc_post_attn_pre_moe", names)
        self.assertIn("mhc_post_moe", names)
        # Should NOT have unfused ops
        self.assertNotIn("sinkhorn_attn", names)
        self.assertNotIn("sinkhorn_moe", names)

    def test_unfused_mhc_ops(self):
        """Unfused mHC produces 6 ops: pre_attn, sinkhorn_attn, post_attn,
        pre_moe, sinkhorn_moe, post_moe."""
        cfg = make_config(mhc_kernel_fused=False)
        lp = prefill_layer(0, cfg)
        names = [op.name for op in lp.ops]
        self.assertIn("mhc_pre_attn", names)
        self.assertIn("sinkhorn_attn", names)
        self.assertIn("mhc_post_attn", names)
        self.assertIn("mhc_pre_moe", names)
        self.assertIn("sinkhorn_moe", names)
        self.assertIn("mhc_post_moe", names)

    def test_fused_vs_unfused_op_count_difference(self):
        """Fused has fewer ops than unfused (3 mHC ops vs 6)."""
        cfg_fused = make_config(mhc_kernel_fused=True)
        cfg_unfused = make_config(mhc_kernel_fused=False)
        lp_fused = prefill_layer(0, cfg_fused)
        lp_unfused = prefill_layer(0, cfg_unfused)
        # Unfused has 3 more mHC ops (sinkhorn_attn, mhc_post_attn+mhc_pre_moe replace post_pre_fused)
        self.assertGreater(len(lp_unfused.ops), len(lp_fused.ops))

    def test_sp_allgathers_present(self):
        """SP allgathers present when sp=True and TP>1 (3 of them)."""
        cfg = make_config(sp=True, tp=2)
        lp = prefill_layer(0, cfg)
        names = [op.name for op in lp.ops]
        self.assertIn("sp_ag_before_attn", names)
        self.assertIn("sp_ag_before_moe", names)
        self.assertIn("sp_ag_after_moe", names)

    def test_sp_allgathers_zero_when_tp1(self):
        """SP allgathers have zero comm time when TP=1."""
        cfg = make_config(sp=True, tp=1, ep=4)
        lp = prefill_layer(0, cfg)
        sp_ops = [op for op in lp.ops if op.name.startswith("sp_ag_")]
        for op in sp_ops:
            self.assertEqual(op.comm_time_s, 0.0, f"{op.name} should have 0 comm with TP=1")

    def test_shared_expert_overlapped(self):
        """With overlap, no shared_expert ops in the op list (overlapped with dispatch+combine)."""
        cfg = make_config(shared_expert_overlapped=True)
        lp = prefill_layer(0, cfg)
        names = [op.name for op in lp.ops]
        self.assertNotIn("shared_gate_proj", names)
        self.assertNotIn("shared_up_proj", names)
        self.assertNotIn("shared_silu_mul", names)
        self.assertNotIn("shared_down_proj", names)

    def test_shared_expert_not_overlapped(self):
        """Without overlap, shared expert ops ARE in the list."""
        cfg = make_config(shared_expert_overlapped=False)
        lp = prefill_layer(0, cfg)
        names = [op.name for op in lp.ops]
        self.assertIn("shared_gate_proj", names)
        self.assertIn("shared_up_proj", names)
        self.assertIn("shared_silu_mul", names)
        self.assertIn("shared_down_proj", names)

    def test_all_ops_valid(self):
        """All ops in a prefill layer pass assert_op_valid."""
        cfg = make_config()
        for layer_idx in range(cfg.model.num_hidden_layers):
            lp = prefill_layer(layer_idx, cfg)
            for op in lp.ops:
                assert_op_valid(self, op)

    def test_total_time_positive(self):
        """Total time > 0 for all layers."""
        cfg = make_config()
        for layer_idx in range(cfg.model.num_hidden_layers):
            lp = prefill_layer(layer_idx, cfg)
            self.assertGreater(lp.total.time_s, 0,
                               f"layer {layer_idx} total time should be positive")


# ── Decode Layer ──────────────────────────────────────────────────────────

class TestDecodeLayer(unittest.TestCase):
    """decode_layer for various layer types."""

    def test_ratio1_swa_attention(self):
        """ratio=1: has attention_swa in decode, no attention_comp."""
        cfg = make_config()
        lp = decode_layer(0, 256, cfg)
        self.assertEqual(lp.ratio, 1)
        names = [op.name for op in lp.ops]
        self.assertIn("attention_swa", names)
        self.assertNotIn("attention_comp", names)

    def test_ratio4_use_index_condition(self):
        """Decode uses S_comp > index_topk for use_index.
        S_total=256, ratio=4 => S_comp=64 > topK=16 => use_index=True."""
        cfg = make_config()
        lp = decode_layer(1, 256, cfg)  # layer 1 has ratio=4
        names = [op.name for op in lp.ops]
        self.assertIn("index_iq_proj", names)

    def test_ratio4_no_index_when_s_comp_small(self):
        """S_total=32, ratio=4 => S_comp=8 < topK=16 => use_index=False."""
        cfg = make_config()
        lp = decode_layer(1, 32, cfg)  # layer 1 has ratio=4
        names = [op.name for op in lp.ops]
        self.assertNotIn("index_iq_proj", names)
        # KV compression still present
        self.assertIn("kv_compression_decode", names)

    def test_decode_no_sp_allgather(self):
        """Decode does not use SP allgather ops."""
        cfg = make_config(sp=True, tp=2)
        lp = decode_layer(0, 256, cfg)
        names = [op.name for op in lp.ops]
        sp_names = [n for n in names if n.startswith("sp_ag_")]
        self.assertEqual(len(sp_names), 0, "decode should have no SP allgather ops")

    def test_fused_vs_unfused_decode(self):
        """Fused decode has fewer mHC ops than unfused."""
        cfg_f = make_config(mhc_kernel_fused=True)
        cfg_u = make_config(mhc_kernel_fused=False)
        lp_f = decode_layer(0, 256, cfg_f)
        lp_u = decode_layer(0, 256, cfg_u)
        self.assertGreater(len(lp_u.ops), len(lp_f.ops))

    def test_shared_expert_overlap_decode(self):
        """With overlap, shared expert ops not listed; without, they are."""
        cfg_over = make_config(shared_expert_overlapped=True)
        cfg_no = make_config(shared_expert_overlapped=False)
        lp_over = decode_layer(0, 256, cfg_over)
        lp_no = decode_layer(0, 256, cfg_no)
        names_over = [op.name for op in lp_over.ops]
        names_no = [op.name for op in lp_no.ops]
        self.assertNotIn("shared_gate_proj", names_over)
        self.assertIn("shared_gate_proj", names_no)


# ── Prefill Model ─────────────────────────────────────────────────────────

class TestPrefillModel(unittest.TestCase):
    """prefill_model end-to-end."""

    def test_phase_name(self):
        """Returns PhaseProfile with phase='prefill'."""
        cfg = make_config()
        phase = prefill_model(cfg)
        self.assertIsInstance(phase, PhaseProfile)
        self.assertEqual(phase.phase, "prefill")

    def test_layer_count(self):
        """Correct number of layer profiles."""
        cfg = make_config()
        phase = prefill_model(cfg)
        self.assertEqual(len(phase.layer_profiles), cfg.model.num_hidden_layers)

    def test_extra_ops_present(self):
        """Extra ops include embed, final_rmsnorm, sp_ag, lm_head."""
        cfg = make_config()
        phase = prefill_model(cfg)
        extra_names = [op.name for op in phase.extra_ops]
        self.assertIn("embedding", extra_names)
        self.assertIn("final_rmsnorm", extra_names)
        self.assertIn("sp_ag_before_lm_head", extra_names)
        self.assertIn("lm_head", extra_names)

    def test_total_tokens(self):
        """total_tokens = B * S where B = batch_size // dp."""
        cfg = make_config(batch_size=4, dp=2, seq_len=64)
        phase = prefill_model(cfg)
        B = cfg.rt.batch_size // cfg.rt.dp
        expected = B * cfg.rt.seq_len
        self.assertEqual(phase.total_tokens, expected)


# ── Decode Step ───────────────────────────────────────────────────────────

class TestDecodeStep(unittest.TestCase):
    """decode_step for a single decode step."""

    def test_returns_phase_profile(self):
        cfg = make_config()
        phase = decode_step(128, cfg)
        self.assertIsInstance(phase, PhaseProfile)
        self.assertTrue(phase.phase.startswith("decode_step@"))

    def test_layer_count(self):
        cfg = make_config()
        phase = decode_step(128, cfg)
        self.assertEqual(len(phase.layer_profiles), cfg.model.num_hidden_layers)

    def test_total_tokens(self):
        """total_tokens = B (= batch_size // dp), since 1 token per seq."""
        cfg = make_config(batch_size=4, dp=2)
        phase = decode_step(128, cfg)
        B = cfg.rt.batch_size // cfg.rt.dp
        self.assertEqual(phase.total_tokens, B)


# ── Decode Model ──────────────────────────────────────────────────────────

class TestDecodeModel(unittest.TestCase):
    """decode_model aggregation over output_len steps."""

    def test_total_tokens(self):
        """total_tokens = B * output_len."""
        cfg = make_config(batch_size=4, dp=2, output_len=16)
        phase = decode_model(cfg)
        B = cfg.rt.batch_size // cfg.rt.dp
        self.assertEqual(phase.total_tokens, B * cfg.rt.output_len)

    def test_time_sums_all_steps(self):
        """Total time is sum of all decode steps when output_len <= 2*P (exact path)."""
        cfg = make_config(output_len=4)
        P = _compression_period(cfg)
        # Ensure we're on the exact iteration path
        self.assertLessEqual(cfg.rt.output_len, 2 * P)
        phase = decode_model(cfg)
        # Manually compute sum of individual steps
        total = 0.0
        for step in range(cfg.rt.output_len):
            s_total = cfg.rt.seq_len + step
            step_phase = decode_step(s_total, cfg)
            total += step_phase.total_time_s
        self.assertAlmostEqual(phase.total_time_s, total, places=10)

    def test_phase_name(self):
        """phase name = 'decode_total'."""
        cfg = make_config()
        phase = decode_model(cfg)
        self.assertEqual(phase.phase, "decode_total")

    def test_trapezoidal_interpolation_path(self):
        """When output_len > 2*P, trapezoidal interpolation is used."""
        cfg = make_config(output_len=512)
        P = _compression_period(cfg)
        # Ensure we're on the interpolation path
        self.assertGreater(cfg.rt.output_len, 2 * P)
        phase = decode_model(cfg)
        self.assertGreater(phase.total_time_s, 0)
        # Verify formula: N * (T_first + T_last) / (2*P)
        N = cfg.rt.output_len
        S_base = cfg.rt.seq_len
        T_first = sum(decode_step(S_base + i, cfg).total_time_s for i in range(P))
        T_last = sum(decode_step(S_base + N - P + i, cfg).total_time_s for i in range(P))
        expected = N * (T_first + T_last) / (2 * P)
        self.assertAlmostEqual(phase.total_time_s, expected, places=10)

    def test_trapezoidal_approximation_reasonable(self):
        """Trapezoidal result should be close to exact sum for moderate output_len."""
        # Use a small enough output_len that we can compute exact
        cfg_exact = make_config(output_len=256)
        P = _compression_period(cfg_exact)
        # Compute exact (brute force)
        exact_time = sum(
            decode_step(cfg_exact.rt.seq_len + i, cfg_exact).total_time_s
            for i in range(cfg_exact.rt.output_len)
        )
        # Compute trapezoidal
        phase = decode_model(cfg_exact)
        # Should be within 1% for linear+periodic cost structure
        if exact_time > 0:
            rel_error = abs(phase.total_time_s - exact_time) / exact_time
            self.assertLess(rel_error, 0.01,
                            f"Trapezoidal error {rel_error:.4%} exceeds 1%")


# ── Compression Period ────────────────────────────────────────────────────

class TestCompressionPeriod(unittest.TestCase):
    """Tests for _compression_period LCM helper."""

    def test_default_config_period(self):
        """Default small config has ratios [1,4,128,4] → LCM(1,4,128) = 128."""
        cfg = make_config()
        self.assertEqual(_compression_period(cfg), 128)

    def test_all_ratio_1(self):
        """All ratio=1 → period=1."""
        cfg = make_config(compress_ratios=[1, 1, 1, 1])
        self.assertEqual(_compression_period(cfg), 1)

    def test_single_ratio(self):
        """Single ratio → period = that ratio."""
        cfg = make_config(compress_ratios=[4, 4, 4, 4])
        self.assertEqual(_compression_period(cfg), 4)

    def test_two_ratios(self):
        """Two ratios → period = LCM."""
        cfg = make_config(compress_ratios=[4, 128, 4, 128])
        self.assertEqual(_compression_period(cfg), 128)


if __name__ == "__main__":
    unittest.main()
