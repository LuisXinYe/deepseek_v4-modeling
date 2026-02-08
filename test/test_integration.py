"""Integration tests for the full perf_model pipeline."""

import unittest

from test.helpers import make_config, make_prod_config, assert_op_valid
from perf_model.layers import (
    prefill_model, prefill_layer, decode_step, decode_model, decode_layer,
)
from perf_model.memory import kv_cache_memory, weight_memory_per_rank


class TestEndToEnd(unittest.TestCase):
    """End-to-end tests using production and small configs."""

    def test_prefill_model_prod_nonzero(self):
        cfg = make_prod_config()
        phase = prefill_model(cfg)
        self.assertGreater(phase.total_time_s, 0)

    def test_decode_step_prod_nonzero(self):
        cfg = make_prod_config()
        phase = decode_step(cfg.rt.seq_len, cfg)
        self.assertGreater(phase.total_time_s, 0)

    def test_decode_model_small(self):
        cfg = make_config(output_len=4)
        phase = decode_model(cfg)
        self.assertGreater(phase.total_time_s, 0)
        B = cfg.rt.batch_size // cfg.rt.dp
        self.assertEqual(phase.total_tokens, B * 4)

    def test_memory_fits_hbm_prod(self):
        cfg = make_prod_config()
        kv = kv_cache_memory(cfg)
        wm = weight_memory_per_rank(cfg)
        total_hbm = kv["total_bytes"] + wm["total"]
        capacity = cfg.hw.hbm_capacity_gb * 1e9
        self.assertLessEqual(total_hbm, capacity)

    def test_all_ratio_types_covered(self):
        cfg = make_prod_config()
        phase = prefill_model(cfg)
        ratios_seen = {lp.ratio for lp in phase.layer_profiles}
        self.assertIn(1, ratios_seen)
        self.assertIn(4, ratios_seen)
        self.assertIn(128, ratios_seen)


class TestFusedVsUnfused(unittest.TestCase):
    """Compare fused vs unfused mHC configurations."""

    def test_fused_prefill_faster(self):
        cfg_fused = make_config(mhc_kernel_fused=True)
        cfg_unfused = make_config(mhc_kernel_fused=False)
        fused = prefill_model(cfg_fused)
        unfused = prefill_model(cfg_unfused)
        self.assertLess(fused.total_time_s, unfused.total_time_s)

    def test_shared_expert_overlap_reduces_time(self):
        cfg_overlap = make_config(shared_expert_overlapped=True)
        cfg_no_overlap = make_config(shared_expert_overlapped=False)
        overlap = prefill_model(cfg_overlap)
        no_overlap = prefill_model(cfg_no_overlap)
        self.assertLessEqual(overlap.total_time_s, no_overlap.total_time_s)

    def test_fused_decode_faster(self):
        cfg_fused = make_config(mhc_kernel_fused=True)
        cfg_unfused = make_config(mhc_kernel_fused=False)
        fused = decode_step(cfg_fused.rt.seq_len, cfg_fused)
        unfused = decode_step(cfg_unfused.rt.seq_len, cfg_unfused)
        self.assertLess(fused.total_time_s, unfused.total_time_s)


class TestConsistency(unittest.TestCase):
    """Tests for monotonicity and scaling behavior."""

    def test_decode_time_monotonic_with_context(self):
        cfg = make_config()
        t1 = decode_step(128, cfg).total_time_s
        t2 = decode_step(256, cfg).total_time_s
        t3 = decode_step(512, cfg).total_time_s
        self.assertLess(t1, t2)
        self.assertLess(t2, t3)

    def test_prefill_time_increases_with_batch(self):
        cfg_small = make_config(batch_size=2)
        cfg_large = make_config(batch_size=4)
        small = prefill_model(cfg_small)
        large = prefill_model(cfg_large)
        self.assertLess(small.total_time_s, large.total_time_s)

    def test_tp_scaling_reduces_compute(self):
        # Higher TP -> lower per-rank compute for compute-bound ops
        cfg_tp2 = make_config(tp=2, ep=4)
        cfg_tp4 = make_config(tp=4, ep=4)
        # Compare a single layer's compute time (e.g. layer 0, ratio=1)
        lp_tp2 = prefill_layer(0, cfg_tp2)
        lp_tp4 = prefill_layer(0, cfg_tp4)
        # Compute time = total - comm
        comp_tp2 = lp_tp2.total.time_s - lp_tp2.total.comm_time_s
        comp_tp4 = lp_tp4.total.time_s - lp_tp4.total.comm_time_s
        self.assertLess(comp_tp4, comp_tp2)

    def test_comm_increases_with_tp(self):
        cfg_tp2 = make_config(tp=2, ep=4)
        cfg_tp4 = make_config(tp=4, ep=4)
        lp_tp2 = prefill_layer(0, cfg_tp2)
        lp_tp4 = prefill_layer(0, cfg_tp4)
        self.assertGreater(lp_tp4.total.comm_time_s, lp_tp2.total.comm_time_s)

    def test_ep1_zero_dispatch_combine(self):
        cfg = make_config(tp=2, ep=1)
        lp = prefill_layer(1, cfg)
        for op in lp.ops:
            if op.name in ("moe_ep_dispatch", "moe_ep_combine"):
                self.assertEqual(op.comm_time_s, 0.0,
                                 f"EP=1 but {op.name} has non-zero comm time")


class TestPropertyInvariants(unittest.TestCase):
    """Property-based invariant tests across all ops and layers."""

    def setUp(self):
        self.cfg = make_config()
        self.prefill = prefill_model(self.cfg)
        self.decode_s = decode_step(self.cfg.rt.seq_len, self.cfg)

    def test_all_op_times_nonnegative(self):
        for phase in [self.prefill, self.decode_s]:
            for lp in phase.layer_profiles:
                for op in lp.ops:
                    self.assertGreaterEqual(op.time_s, 0, f"{op.name} has negative time")
                    self.assertGreaterEqual(op.cube_time_s, 0)
                    self.assertGreaterEqual(op.vec_time_s, 0)
                    self.assertGreaterEqual(op.mem_time_s, 0)
                    self.assertGreaterEqual(op.comm_time_s, 0)

    def test_total_geq_max_components(self):
        for phase in [self.prefill, self.decode_s]:
            for lp in phase.layer_profiles:
                for op in lp.ops:
                    assert_op_valid(self, op)

    def test_layer_total_equals_sum_of_ops(self):
        for phase in [self.prefill, self.decode_s]:
            for lp in phase.layer_profiles:
                ops_sum = sum(op.time_s for op in lp.ops)
                self.assertAlmostEqual(
                    lp.total.time_s, ops_sum, places=12,
                    msg=f"Layer {lp.layer_idx}: total != sum of ops"
                )

    def test_no_comm_when_tp1_ep1(self):
        cfg = make_config(tp=1, ep=1, sp=False)
        phase = prefill_model(cfg)
        for lp in phase.layer_profiles:
            for op in lp.ops:
                if op.name in ("attn_tp_allreduce", "moe_ep_dispatch",
                               "moe_ep_combine", "index_score_ar",
                               "sp_allgather", "sp_ag_before_attn",
                               "sp_ag_before_moe", "sp_ag_after_moe",
                               "sp_ag_before_lm_head"):
                    self.assertEqual(
                        op.comm_time_s, 0.0,
                        f"TP=1, EP=1 but {op.name} has comm={op.comm_time_s}"
                    )

    def test_bottleneck_labels_valid(self):
        valid = {"CUBE", "VEC", "MEM", "COMM", ""}
        for phase in [self.prefill, self.decode_s]:
            for lp in phase.layer_profiles:
                for op in lp.ops:
                    self.assertIn(op.bottleneck, valid, f"{op.name}: bad bottleneck")
                self.assertIn(lp.total.bottleneck, valid)

    def test_flops_positive_iff_cube_time_positive(self):
        for phase in [self.prefill, self.decode_s]:
            for lp in phase.layer_profiles:
                for op in lp.ops:
                    if op.flops > 0:
                        self.assertGreater(op.cube_time_s, 0,
                                           f"{op.name}: flops>0 but cube_time=0")
                    if op.cube_time_s > 0:
                        self.assertGreater(op.flops, 0,
                                           f"{op.name}: cube_time>0 but flops=0")

    def test_total_tokens_correct(self):
        cfg = self.cfg
        B = cfg.rt.batch_size // cfg.rt.dp
        S = cfg.rt.seq_len
        # Prefill processes B*S tokens
        self.assertEqual(self.prefill.total_tokens, B * S)
        # Decode step processes B tokens (1 per sequence)
        self.assertEqual(self.decode_s.total_tokens, B)


if __name__ == "__main__":
    unittest.main()
