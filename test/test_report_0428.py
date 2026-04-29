import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
COMMON_PATH = REPO_ROOT / "report" / "0428" / "script" / "common.py"
GENERATOR_PATH = REPO_ROOT / "report" / "0428" / "script" / "generate_report.py"


def load_module(name, path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestReport0428(unittest.TestCase):
    def test_scenarios_match_spec(self):
        common = load_module("report_0428_common", COMMON_PATH)
        ids = [s["scenario_id"] for s in common.SCENARIOS]
        self.assertEqual(ids, ["8k_1k", "32k_1k", "128k_1k", "1m_1k"])
        self.assertEqual(common.REPORT_DEFAULTS["quant_mode"], "w8a8")
        self.assertEqual(common.REPORT_DEFAULTS["kv_cache_quant_mode"], "kv8")
        self.assertEqual(common.REPORT_DEFAULTS["mtp_accept_ratio"], 0.9)
        self.assertEqual(common.PREFIX_CACHE_HIT_RATES, [0.0, 0.9, 0.99])

    def test_null_not_zero_for_infeasible(self):
        common = load_module("report_0428_common", COMMON_PATH)
        row = common.infeasible_row("8k_1k", "no_candidate")
        self.assertFalse(row["is_feasible"])
        self.assertIsNone(row["decode_tps_per_card"])
        self.assertEqual(row["invalid_reason"], "no_candidate")

    def test_decode_candidate_enumeration_has_required_sizes(self):
        generator = load_module("report_0428_generator", GENERATOR_PATH)
        for cards in [8, 16, 32, 64]:
            candidates = list(generator.iter_parallel_candidates(cards))
            self.assertTrue(candidates)
            self.assertTrue(all(tp * dp == cards for tp, _, dp, _ in candidates))

    def test_prefill_perf_search_is_not_limited_to_sizing_batch(self):
        common = load_module("report_0428_common_prefill", COMMON_PATH)
        generator = load_module("report_0428_generator_prefill", GENERATOR_PATH)
        cfg = common.load_910c_config()

        row = generator.select_prefill_for_scenario(cfg, common.SCENARIOS[0], 0.0)

        self.assertGreater(row["batch_size"], 1)
        self.assertGreater(row["prefill_tps_per_card"], 0)

    def test_prefill_sort_key_prioritizes_tps_per_card(self):
        generator = load_module("report_0428_generator_prefill_sort", GENERATOR_PATH)
        lower_qps_higher_tps = {
            "prefill_tps_per_card": 200,
            "prefill_qps_instance": 1,
            "hbm_margin_gb": 0,
            "prefill_time_ms": 10,
            "tp": 1,
            "ep": 1,
        }
        higher_qps_lower_tps = {
            "prefill_tps_per_card": 100,
            "prefill_qps_instance": 10,
            "hbm_margin_gb": 100,
            "prefill_time_ms": 1,
            "tp": 1,
            "ep": 1,
        }

        self.assertGreater(
            generator._prefill_sort_key(lower_qps_higher_tps),
            generator._prefill_sort_key(higher_qps_lower_tps),
        )


if __name__ == "__main__":
    unittest.main()
