"""Tests for perf_model/report.py: formatting, CSV export, and print functions."""

import unittest
import tempfile
import os
import io
import csv
import json
import contextlib

from test.helpers import make_config, make_prod_config, assert_op_valid
from perf_model.report import (
    fmt_bytes, fmt_ms, find_representative_layers,
    export_ops_csv, export_layer_summary_csv, export_memory_csv,
    export_summary_csv, export_config_json,
    print_config_summary, print_phase_report, print_memory_report,
    print_op_table,
)
from perf_model.layers import prefill_model, decode_step, decode_model


class TestFmtBytes(unittest.TestCase):
    """Tests for fmt_bytes() formatting function."""

    def test_gb_threshold(self):
        self.assertEqual(fmt_bytes(2.5e9), "2.50 GB")

    def test_mb_threshold(self):
        self.assertEqual(fmt_bytes(1.5e6), "1.50 MB")

    def test_kb_threshold(self):
        self.assertEqual(fmt_bytes(1.5e3), "1.50 KB")

    def test_bytes_threshold(self):
        self.assertEqual(fmt_bytes(500), "500 B")

    def test_zero(self):
        self.assertEqual(fmt_bytes(0), "0 B")


class TestFmtMs(unittest.TestCase):
    """Tests for fmt_ms() formatting function."""

    def test_basic(self):
        self.assertEqual(fmt_ms(0.001), "1.000")

    def test_zero(self):
        self.assertEqual(fmt_ms(0), "0.000")

    def test_precision_three_decimal_places(self):
        result = fmt_ms(0.12345)
        # 0.12345 seconds = 123.45 ms -> "123.450"
        self.assertEqual(result, "123.450")
        # Always 3 decimal places
        parts = result.split(".")
        self.assertEqual(len(parts), 2)
        self.assertEqual(len(parts[1]), 3)


class TestFindRepresentativeLayers(unittest.TestCase):
    """Tests for find_representative_layers()."""

    def test_prod_config_returns_three(self):
        cfg = make_prod_config()
        reps = find_representative_layers(cfg)
        self.assertEqual(len(reps), 3)

    def test_first_occurrence_selected(self):
        cfg = make_prod_config()
        reps = find_representative_layers(cfg)
        ratios = cfg.model.compress_ratios
        # Each index should be the first occurrence of its ratio
        seen = set()
        for idx in reps:
            r = ratios[idx]
            self.assertNotIn(r, seen, f"Ratio {r} seen more than once")
            # Verify it is indeed the first occurrence
            self.assertEqual(ratios.index(r), idx)
            seen.add(r)

    def test_small_config(self):
        cfg = make_config()  # compress_ratios=[1, 4, 128, 4]
        reps = find_representative_layers(cfg)
        # 3 unique ratios: 1, 4, 128 -> first of each = [0, 1, 2]
        self.assertEqual(reps, [0, 1, 2])


class TestCSVExport(unittest.TestCase):
    """Tests for CSV and JSON export functions."""

    def setUp(self):
        self.cfg = make_config()
        self.prefill = prefill_model(self.cfg)
        self.decode = decode_step(self.cfg.rt.seq_len, self.cfg)

    def test_ops_csv_header_and_rows(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "ops.csv")
            export_ops_csv(path, self.prefill)
            with open(path) as f:
                reader = csv.reader(f)
                header = next(reader)
                self.assertIn("layer_idx", header)
                self.assertIn("op_name", header)
                self.assertIn("total_time_ms", header)
                rows = list(reader)
                self.assertGreater(len(rows), 0)
                # Should have rows for ops in each layer + extra ops
                layer_rows = [r for r in rows if r[0] != "extra"]
                extra_rows = [r for r in rows if r[0] == "extra"]
                self.assertGreater(len(layer_rows), 0)
                self.assertGreater(len(extra_rows), 0)

    def test_layer_summary_csv(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "layer_summary.csv")
            export_layer_summary_csv(path, self.prefill, self.decode)
            with open(path) as f:
                reader = csv.reader(f)
                header = next(reader)
                self.assertIn("phase", header)
                rows = list(reader)
                prefill_rows = [r for r in rows if r[0] == "prefill"]
                decode_rows = [r for r in rows if r[0] == "decode"]
                self.assertGreater(len(prefill_rows), 0)
                self.assertGreater(len(decode_rows), 0)

    def test_memory_csv(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "memory.csv")
            export_memory_csv(path, self.cfg)
            with open(path) as f:
                reader = csv.reader(f)
                header = next(reader)
                self.assertIn("category", header)
                rows = list(reader)
                categories = {r[0] for r in rows}
                self.assertIn("kv_cache", categories)
                self.assertIn("weights", categories)

    def test_summary_csv(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "summary.csv")
            metrics = {"prefill_time_ms": "10.5", "decode_time_ms": "5.2"}
            export_summary_csv(path, metrics)
            with open(path) as f:
                reader = csv.reader(f)
                header = next(reader)
                self.assertEqual(header, ["metric", "value"])
                rows = list(reader)
                self.assertEqual(len(rows), 2)

    def test_config_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "config.json")
            export_config_json(path, self.cfg)
            with open(path) as f:
                data = json.load(f)
            self.assertIn("hw", data)
            self.assertIn("net", data)
            self.assertIn("model", data)
            self.assertIn("rt", data)
            # Check a few fields round-trip
            self.assertEqual(data["model"]["hidden_size"], self.cfg.model.hidden_size)
            self.assertEqual(data["rt"]["tp"], self.cfg.rt.tp)
            self.assertEqual(data["hw"]["cube_tflops"], self.cfg.hw.cube_tflops)

    def test_ops_csv_covers_all_layers(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "ops.csv")
            export_ops_csv(path, self.prefill)
            with open(path) as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                rows = list(reader)
                layer_indices = {r[0] for r in rows if r[0] != "extra"}
                # Should have entries for all 4 layers
                for i in range(self.cfg.model.num_hidden_layers):
                    self.assertIn(str(i), layer_indices)


class TestPrintFunctions(unittest.TestCase):
    """Smoke tests for print functions -- verify they run without exceptions."""

    def setUp(self):
        self.cfg = make_config()
        self.prefill = prefill_model(self.cfg)
        self.decode_s = decode_step(self.cfg.rt.seq_len, self.cfg)

    def test_print_config_summary(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_config_summary(self.cfg)
        output = buf.getvalue()
        self.assertGreater(len(output), 0)
        self.assertIn("Hardware", output)

    def test_print_phase_report_prefill(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_phase_report(self.prefill, self.cfg)
        output = buf.getvalue()
        self.assertGreater(len(output), 0)
        self.assertIn("PREFILL", output)

    def test_print_phase_report_decode(self):
        decode_total = decode_model(make_config(output_len=4))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_phase_report(decode_total, self.cfg, detailed_step=self.decode_s)
        output = buf.getvalue()
        self.assertGreater(len(output), 0)
        self.assertIn("DECODE", output)

    def test_print_memory_report(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_memory_report(self.cfg)
        output = buf.getvalue()
        self.assertGreater(len(output), 0)
        self.assertIn("MEMORY", output)

    def test_print_op_table(self):
        ops = self.prefill.layer_profiles[0].ops
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_op_table(ops)
        output = buf.getvalue()
        self.assertGreater(len(output), 0)
        self.assertIn("Operation", output)


if __name__ == "__main__":
    unittest.main()
