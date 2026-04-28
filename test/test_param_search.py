"""Tests for parameter-search helpers and metrics."""

import unittest
from unittest.mock import patch

from test.helpers import make_config
from param_search import search


class TestSearchHelpers(unittest.TestCase):
    def test_prefill_eval_uses_logical_input_len_for_instance_throughput(self):
        cfg = make_config(seq_len=100, batch_size=8, dp=4, tp=2, input_len=1000)
        metrics = search.evaluate_prefill(cfg, logical_input_len=cfg.rt.request_input_len)
        self.assertIn("prefill_tps_instance", metrics)
        self.assertIn("prefill_qps_instance", metrics)
        self.assertAlmostEqual(
            metrics["prefill_tps_instance"] / metrics["prefill_qps_instance"],
            cfg.rt.request_input_len,
            places=6,
        )

    def test_decode_eval_uses_global_batch_for_instance_qps(self):
        cfg = make_config(seq_len=256, output_len=8, batch_size=8, dp=4, tp=2)
        metrics = search.evaluate_decode(cfg)
        self.assertIn("decode_tps_instance", metrics)
        self.assertIn("decode_qps_instance", metrics)
        self.assertAlmostEqual(
            metrics["decode_tps_instance"] / metrics["decode_qps_instance"],
            cfg.rt.output_len,
            places=6,
        )

    def test_make_phase_configs_split_compute_and_memory_lengths(self):
        base_cfg = make_config(
            seq_len=8192,
            input_len=8192,
            decode_context_len=8192,
            prefix_cache_hit_rate=0.9,
        )
        mem_cfg, eval_cfg = search.make_phase_configs(
            base_cfg,
            phase="prefill",
            tp=base_cfg.rt.tp,
            ep=base_cfg.rt.ep,
            dp=base_cfg.rt.dp,
            batch_size=base_cfg.rt.batch_size,
            seq_len=base_cfg.rt.seq_len,
            sp=base_cfg.rt.sp,
            shared_expert_overlapped=base_cfg.rt.shared_expert_overlapped,
            prefix_cache_hit_rate=0.9,
        )
        self.assertEqual(mem_cfg.rt.seq_len, 8192)
        self.assertEqual(eval_cfg.rt.seq_len, 820)


class TestSearchMatrix(unittest.TestCase):
    @patch.object(search, "TP_VALUES", [2])
    @patch.object(search, "EP_VALUES", [4])
    @patch.object(search, "DP_VALUES", [4])
    @patch.object(search, "BATCH_VALUES", [8])
    @patch.object(search, "SEQ_VALUES", [1024])
    @patch.object(search, "MIN_GPUS", 8)
    @patch.object(search, "MAX_GPUS", 8)
    @patch.object(search, "PREFIX_CACHE_HIT_RATE_VALUES", [0.0, 0.9, 0.99])
    def test_run_search_emits_one_row_per_prefix_cache_value(self):
        cfg = make_config(tp=2, ep=4, dp=4, batch_size=8, seq_len=1024)
        results = search.run_search(cfg, phase="prefill", scenario="throughput")
        hit_rates = sorted({row["prefix_cache_hit_rate"] for row in results})
        self.assertEqual(hit_rates, [0.0, 0.9, 0.99])
        self.assertEqual(results[0]["logical_input_len"], 1024)
        self.assertIn("effective_prefill_len", results[0])
        self.assertIn("prefill_qps_instance", results[0])

    @patch.object(search, "TP_VALUES", [2])
    @patch.object(search, "EP_VALUES", [4])
    @patch.object(search, "DP_VALUES", [4])
    @patch.object(search, "BATCH_VALUES", [8])
    @patch.object(search, "SEQ_VALUES", [128])
    @patch.object(search, "MIN_GPUS", 8)
    @patch.object(search, "MAX_GPUS", 8)
    @patch.object(search, "PREFIX_CACHE_HIT_RATE_VALUES", [0.0])
    def test_verify_decode_top_n_recomputes_instance_metrics(self):
        cfg = make_config(tp=2, ep=4, dp=4, batch_size=8, seq_len=128, output_len=8)
        results = search.run_search(cfg, phase="decode", scenario="throughput")
        verified = search.verify_decode_top_n(results, cfg, top_n=1)
        self.assertIn("decode_total_ms_exact", verified[0])
        self.assertIn("decode_qps_instance", verified[0])
        self.assertAlmostEqual(
            float(verified[0]["decode_tps_instance"]) / float(verified[0]["decode_qps_instance"]),
            cfg.rt.output_len,
            places=2,
        )


if __name__ == "__main__":
    unittest.main()
