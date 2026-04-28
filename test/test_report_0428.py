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


if __name__ == "__main__":
    unittest.main()
