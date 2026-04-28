"""Tests for parameter-search helpers and metrics."""

import unittest
from unittest.mock import patch

from test.helpers import make_config
from param_search import analyze as search_analyze
from param_search import search
from report import analyze_scenarios


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

    def test_search_memory_check_uses_hbm_reserved_pct(self):
        cfg = make_config(hbm_capacity_gb=100, hbm_reserved_pct=10)
        with patch.object(search, "weight_memory_per_rank", return_value={"total": 85e9}), \
             patch.object(search, "kv_cache_memory", return_value={"total_bytes": 6e9}):
            _, _, total_gb, fits = search.check_memory(cfg)

        self.assertEqual(total_gb, 91)
        self.assertFalse(fits)

    def test_scenario_memory_check_uses_hbm_reserved_pct(self):
        cfg = make_config(hbm_capacity_gb=100, hbm_reserved_pct=10)
        with patch.object(analyze_scenarios, "weight_memory_per_rank", return_value={"total": 85e9}), \
             patch.object(analyze_scenarios, "kv_cache_memory", return_value={"total_bytes": 5e9}):
            _, _, total_gb, fits = analyze_scenarios.check_memory(cfg)

        self.assertEqual(total_gb, 90)
        self.assertTrue(fits)


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


class TestSearchAnalyze(unittest.TestCase):
    def test_prefill_latency_best_per_group_uses_hit_rate_dimension(self):
        rows = [
            {"seq_len": 1024, "prefix_cache_hit_rate": 0.0, "prefill_time_ms": 10.0},
            {"seq_len": 1024, "prefix_cache_hit_rate": 0.9, "prefill_time_ms": 2.0},
        ]
        analysis = search_analyze.analyze_prefill_latency(rows)
        self.assertIn((1024, 0.0), analysis["best_per_group"])
        self.assertIn((1024, 0.9), analysis["best_per_group"])

    def test_decode_throughput_batch_scaling_keeps_hit_rate_partition(self):
        rows = [
            {
                "tp": 2, "ep": 4, "dp": 4, "seq_len": 1024,
                "prefix_cache_hit_rate": 0.0, "batch_size": 8,
                "physical_gpus": 8, "decode_tps_per_gpu": 10.0,
            },
            {
                "tp": 2, "ep": 4, "dp": 4, "seq_len": 1024,
                "prefix_cache_hit_rate": 0.9, "batch_size": 8,
                "physical_gpus": 8, "decode_tps_per_gpu": 9.0,
            },
        ]
        analysis = search_analyze.analyze_decode_throughput(rows)
        self.assertEqual(analysis["best"]["prefix_cache_hit_rate"], 0.0)
        self.assertTrue(all("prefix_cache_hit_rate" in row for row in analysis["batch_scaling"]))


class TestScenarioAnalysis(unittest.TestCase):
    def test_compute_pd_ratio_returns_minimal_integer_balance(self):
        ratio = analyze_scenarios.compute_pd_ratio(
            p_qps_instance=100.0,
            d_qps_instance=250.0,
            tolerance=0.0,
        )
        self.assertEqual(ratio["prefill_instances"], 5)
        self.assertEqual(ratio["decode_instances"], 2)
        self.assertEqual(ratio["label"], "5P:2D")
        self.assertEqual(ratio["qps_imbalance_pct"], 0.0)

    def test_compute_pd_ratio_allows_more_decode_instances(self):
        ratio = analyze_scenarios.compute_pd_ratio(
            p_qps_instance=10.0,
            d_qps_instance=1.0,
            tolerance=0.0,
        )
        self.assertEqual(ratio["prefill_instances"], 1)
        self.assertEqual(ratio["decode_instances"], 10)
        self.assertEqual(ratio["label"], "1P:10D")

    def test_compute_pd_ratio_honors_tolerance_for_small_integer_pair(self):
        ratio = analyze_scenarios.compute_pd_ratio(
            p_qps_instance=0.8167422737314898,
            d_qps_instance=1.2535464474056521,
            tolerance=0.02,
        )
        self.assertEqual(ratio["prefill_instances"], 14)
        self.assertEqual(ratio["decode_instances"], 9)
        self.assertEqual(ratio["label"], "14P:9D")
        self.assertLessEqual(ratio["qps_imbalance_pct"], 0.02)

    def test_compute_pd_ratio_uses_lowest_integer_inside_tolerance_band(self):
        ratio = analyze_scenarios.compute_pd_ratio(
            p_qps_instance=1.0,
            d_qps_instance=60.01,
            tolerance=0.02,
        )
        self.assertEqual(ratio["prefill_instances"], 59)
        self.assertEqual(ratio["decode_instances"], 1)
        self.assertEqual(ratio["label"], "59P:1D")
        self.assertLessEqual(ratio["qps_imbalance_pct"], 0.02)

    def test_combos_include_1m_4k(self):
        names = {combo["name"] for combo in analyze_scenarios.COMBOS}
        self.assertIn("1M_4K", names)

    def test_hit_rate_values_are_defined(self):
        self.assertEqual(analyze_scenarios.PREFIX_CACHE_HIT_RATE_VALUES, [0.0, 0.9, 0.99])

    def test_default_pd_ratio_tolerance_is_ten_percent(self):
        self.assertEqual(analyze_scenarios.PD_RATIO_TOLERANCE, 0.1)

    def test_report_gpu_values_are_power_of_two_request_set(self):
        self.assertEqual(analyze_scenarios.PREFILL_GPU_VALUES, [8, 16, 32, 64])
        self.assertEqual(analyze_scenarios.DECODE_GPU_VALUES, [8, 16, 32, 64])

    def test_iter_result_combos_expands_all_prefix_cache_rates(self):
        combos = list(analyze_scenarios.iter_result_combos())
        names = {combo["result_name"] for combo in combos}
        self.assertEqual(
            len(combos),
            len(analyze_scenarios.COMBOS) * len(analyze_scenarios.PREFIX_CACHE_HIT_RATE_VALUES),
        )
        self.assertIn("8K_4K", names)
        self.assertIn("8K_4K_hit90", names)
        self.assertIn("1M_4K_hit99", names)


if __name__ == "__main__":
    unittest.main()
