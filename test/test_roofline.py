"""Tests for perf_model.roofline — OpProfile, roofline engine, and comm helpers."""

import unittest

from test.helpers import make_config, assert_op_valid, ATOL, RTOL

from perf_model.config import HardwareConfig
from perf_model.roofline import (
    OpProfile,
    allgather_time,
    allreduce_time,
    alltoall_time,
    bytes2,
    roofline_time,
    sum_ops,
)


class TestBytes2(unittest.TestCase):
    """bytes2: BF16 byte count = count * 2."""

    def test_basic(self):
        self.assertEqual(bytes2(10), 20)

    def test_zero(self):
        self.assertEqual(bytes2(0), 0)

    def test_large(self):
        self.assertEqual(bytes2(1_000_000_000), 2_000_000_000)


class TestRooflineTime(unittest.TestCase):
    """roofline_time: bottleneck detection and time computation."""

    def setUp(self):
        self.hw = HardwareConfig(
            cube_tflops=100, vec_tflops=10,
            hbm_bandwidth_gbps=1000,
            flops_utilization=0.5, hbm_bw_utilization=0.8,
        )

    def test_cube_bound(self):
        """Large flops, small vec/mem -> CUBE bottleneck."""
        op = roofline_time("cube_test", flops=1e15, vec_ops=0, mem_bytes=0, hw=self.hw)
        assert_op_valid(self, op)
        self.assertEqual(op.bottleneck, "CUBE")
        self.assertGreater(op.cube_time_s, 0)
        self.assertEqual(op.vec_time_s, 0)
        self.assertEqual(op.mem_time_s, 0)

    def test_vec_bound(self):
        """Large vec_ops, small flops/mem -> VEC bottleneck."""
        op = roofline_time("vec_test", flops=0, vec_ops=1e15, mem_bytes=0, hw=self.hw)
        assert_op_valid(self, op)
        self.assertEqual(op.bottleneck, "VEC")

    def test_mem_bound(self):
        """Large mem_bytes, small flops/vec -> MEM bottleneck."""
        op = roofline_time("mem_test", flops=0, vec_ops=0, mem_bytes=1e15, hw=self.hw)
        assert_op_valid(self, op)
        self.assertEqual(op.bottleneck, "MEM")

    def test_comm_bound(self):
        """comm_time_s > compute_time -> COMM bottleneck."""
        op = roofline_time("comm_test", flops=0, vec_ops=0, mem_bytes=0,
                           hw=self.hw, comm_time_s=1.0)
        assert_op_valid(self, op)
        self.assertEqual(op.bottleneck, "COMM")
        self.assertAlmostEqual(op.time_s, 1.0, places=12)

    def test_total_time_formula(self):
        """total = max(cube, vec, mem) + comm."""
        op = roofline_time("formula_test", flops=1e12, vec_ops=1e11,
                           mem_bytes=1e9, hw=self.hw, comm_time_s=0.001)
        assert_op_valid(self, op)
        compute = max(op.cube_time_s, op.vec_time_s, op.mem_time_s)
        self.assertAlmostEqual(op.time_s, compute + op.comm_time_s, places=12)

    def test_all_zero(self):
        """All zeros -> empty bottleneck."""
        op = roofline_time("zero", flops=0, vec_ops=0, mem_bytes=0, hw=self.hw)
        assert_op_valid(self, op)
        self.assertEqual(op.bottleneck, "")
        self.assertEqual(op.time_s, 0.0)

    def test_exact_cube_time(self):
        """cube_time = flops / (tflops * 1e12 * util)."""
        flops = 1e12
        expected = flops / (self.hw.cube_tflops * 1e12 * self.hw.flops_utilization)
        op = roofline_time("exact_cube", flops=flops, vec_ops=0, mem_bytes=0, hw=self.hw)
        self.assertAlmostEqual(op.cube_time_s, expected, places=15)

    def test_exact_vec_time(self):
        """vec_time = vec_ops / (vec_tflops * 1e12 * util)."""
        vec_ops = 1e12
        expected = vec_ops / (self.hw.vec_tflops * 1e12 * self.hw.flops_utilization)
        op = roofline_time("exact_vec", flops=0, vec_ops=vec_ops, mem_bytes=0, hw=self.hw)
        self.assertAlmostEqual(op.vec_time_s, expected, places=15)

    def test_separate_cube_and_vec_utilization(self):
        hw = HardwareConfig(
            cube_tflops=100,
            vec_tflops=10,
            hbm_bandwidth_gbps=1000,
            flops_utilization=0.5,
            cube_utilization=0.25,
            vec_utilization=0.1,
            hbm_bw_utilization=0.8,
        )
        op = roofline_time("split_util", flops=1e12, vec_ops=1e12, mem_bytes=0, hw=hw)
        self.assertAlmostEqual(op.cube_time_s, 1e12 / (100 * 1e12 * 0.25), places=15)
        self.assertAlmostEqual(op.vec_time_s, 1e12 / (10 * 1e12 * 0.1), places=15)

    def test_exact_mem_time(self):
        """mem_time = mem_bytes / (bw_gbps * 1e9 * bw_util)."""
        mem_bytes = 1e12
        expected = mem_bytes / (self.hw.hbm_bandwidth_gbps * 1e9 * self.hw.hbm_bw_utilization)
        op = roofline_time("exact_mem", flops=0, vec_ops=0, mem_bytes=mem_bytes, hw=self.hw)
        self.assertAlmostEqual(op.mem_time_s, expected, places=15)

    def test_name_passthrough(self):
        op = roofline_time("my_op_name", flops=0, vec_ops=0, mem_bytes=0, hw=self.hw)
        self.assertEqual(op.name, "my_op_name")

    def test_comm_bytes_passthrough(self):
        op = roofline_time("cb", flops=0, vec_ops=0, mem_bytes=0,
                           hw=self.hw, comm_bytes=12345)
        self.assertEqual(op.comm_bytes, 12345)

    def test_cube_beats_vec_and_mem(self):
        """When cube > vec > mem, bottleneck is CUBE."""
        # cube_time = 1e14 / (100*1e12*0.5) = 2.0
        # vec_time  = 1e13 / (10*1e12*0.5)  = 2.0 -- tie goes to cube (>= check)
        # Use different values to avoid tie
        op = roofline_time("priority", flops=1e14, vec_ops=1e12, mem_bytes=1e9, hw=self.hw)
        self.assertEqual(op.bottleneck, "CUBE")

    def test_vec_beats_mem_but_not_cube(self):
        """vec_time > mem_time, both > 0, but no cube -> VEC."""
        op = roofline_time("vec_vs_mem", flops=0, vec_ops=1e13, mem_bytes=1e6, hw=self.hw)
        self.assertEqual(op.bottleneck, "VEC")


class TestAllReduceTime(unittest.TestCase):
    """allreduce_time: 2*(n-1)/n * vol / (bw*1e9*util) + 2*(n-1)*lat*1e-6."""

    def test_n1_returns_zero(self):
        self.assertEqual(allreduce_time(1000, 1, 100, 10, 0.8), 0.0)

    def test_n2_formula(self):
        vol, n, bw, lat, util = 1e9, 2, 100, 10, 0.8
        factor = 2.0 * (n - 1) / n  # 1.0
        steps = 2 * (n - 1)          # 2
        expected = factor * vol / (bw * 1e9 * util) + steps * lat * 1e-6
        result = allreduce_time(vol, n, bw, lat, util)
        self.assertAlmostEqual(result, expected, places=15)

    def test_n4_formula(self):
        vol, n, bw, lat, util = 1e9, 4, 200, 5, 0.9
        factor = 2.0 * (n - 1) / n  # 1.5
        steps = 2 * (n - 1)          # 6
        expected = factor * vol / (bw * 1e9 * util) + steps * lat * 1e-6
        result = allreduce_time(vol, n, bw, lat, util)
        self.assertAlmostEqual(result, expected, places=15)

    def test_volume_scaling(self):
        """Doubling volume should roughly double the bandwidth term."""
        t1 = allreduce_time(1e9, 4, 100, 0, 1.0)
        t2 = allreduce_time(2e9, 4, 100, 0, 1.0)
        self.assertAlmostEqual(t2, 2 * t1, places=12)

    def test_zero_volume(self):
        """Zero volume -> only latency term."""
        n, bw, lat, util = 4, 100, 10, 0.8
        steps = 2 * (n - 1)
        expected = steps * lat * 1e-6
        result = allreduce_time(0, n, bw, lat, util)
        self.assertAlmostEqual(result, expected, places=15)


class TestAllToAllTime(unittest.TestCase):
    """alltoall_time: (n-1)/n * vol / (bw*1e9*util) + lat*1e-6."""

    def test_n1_returns_zero(self):
        self.assertEqual(alltoall_time(1000, 1, 100, 10, 0.8), 0.0)

    def test_n2_formula(self):
        vol, n, bw, lat, util = 1e9, 2, 100, 10, 0.8
        factor = (n - 1) / n  # 0.5
        expected = factor * vol / (bw * 1e9 * util) + lat * 1e-6
        result = alltoall_time(vol, n, bw, lat, util)
        self.assertAlmostEqual(result, expected, places=15)

    def test_n4_formula(self):
        vol, n, bw, lat, util = 1e9, 4, 200, 5, 0.9
        factor = (n - 1) / n  # 0.75
        expected = factor * vol / (bw * 1e9 * util) + lat * 1e-6
        result = alltoall_time(vol, n, bw, lat, util)
        self.assertAlmostEqual(result, expected, places=15)


class TestAllGatherTime(unittest.TestCase):
    """allgather_time: (n-1)/n * vol / (bw*1e9*util) + (n-1)*lat*1e-6."""

    def test_n1_returns_zero(self):
        self.assertEqual(allgather_time(1000, 1, 100, 10, 0.8), 0.0)

    def test_n2_formula(self):
        vol, n, bw, lat, util = 1e9, 2, 100, 10, 0.8
        factor = (n - 1) / n  # 0.5
        steps = n - 1          # 1
        expected = factor * vol / (bw * 1e9 * util) + steps * lat * 1e-6
        result = allgather_time(vol, n, bw, lat, util)
        self.assertAlmostEqual(result, expected, places=15)

    def test_allgather_vs_alltoall_latency(self):
        """allgather has (n-1) latency steps vs alltoall has 1 step.
        With zero volume, difference is purely in latency."""
        n, bw, lat, util = 8, 100, 10, 0.8
        ag = allgather_time(0, n, bw, lat, util)
        a2a = alltoall_time(0, n, bw, lat, util)
        # allgather: (n-1) * lat * 1e-6 = 7 * 10 * 1e-6 = 7e-5
        # alltoall:  1 * lat * 1e-6 = 1e-5
        expected_ag = (n - 1) * lat * 1e-6
        expected_a2a = lat * 1e-6
        self.assertAlmostEqual(ag, expected_ag, places=15)
        self.assertAlmostEqual(a2a, expected_a2a, places=15)
        self.assertGreater(ag, a2a)

    def test_bandwidth_term_same_as_alltoall(self):
        """allgather and alltoall share the same bandwidth factor (n-1)/n."""
        vol, n, bw, util = 1e9, 4, 100, 0.8
        # With zero latency, bandwidth terms should be equal
        ag = allgather_time(vol, n, bw, 0, util)
        a2a = alltoall_time(vol, n, bw, 0, util)
        self.assertAlmostEqual(ag, a2a, places=15)


class TestSumOps(unittest.TestCase):
    """sum_ops: aggregate OpProfiles."""

    def setUp(self):
        self.hw = HardwareConfig(
            cube_tflops=100, vec_tflops=10,
            hbm_bandwidth_gbps=1000,
            flops_utilization=0.5, hbm_bw_utilization=0.8,
        )

    def test_empty_list(self):
        total = sum_ops([], "empty")
        self.assertEqual(total.name, "empty")
        self.assertEqual(total.time_s, 0.0)
        self.assertEqual(total.bottleneck, "")

    def test_single_op(self):
        op = roofline_time("single", flops=1e12, vec_ops=0, mem_bytes=0, hw=self.hw)
        total = sum_ops([op], "agg")
        self.assertEqual(total.name, "agg")
        self.assertAlmostEqual(total.flops, op.flops, places=12)
        self.assertAlmostEqual(total.time_s, op.time_s, places=12)
        self.assertAlmostEqual(total.cube_time_s, op.cube_time_s, places=12)

    def test_two_ops_aggregation(self):
        op1 = roofline_time("a", flops=1e12, vec_ops=0, mem_bytes=0, hw=self.hw)
        op2 = roofline_time("b", flops=0, vec_ops=0, mem_bytes=1e12, hw=self.hw)
        total = sum_ops([op1, op2], "both")
        self.assertAlmostEqual(total.flops, op1.flops + op2.flops, places=12)
        self.assertAlmostEqual(total.mem_bytes, op1.mem_bytes + op2.mem_bytes, places=12)
        self.assertAlmostEqual(total.cube_time_s, op1.cube_time_s + op2.cube_time_s, places=12)
        self.assertAlmostEqual(total.mem_time_s, op1.mem_time_s + op2.mem_time_s, places=12)
        self.assertAlmostEqual(total.time_s, op1.time_s + op2.time_s, places=12)

    def test_bottleneck_determination(self):
        """Bottleneck = component with largest aggregate time."""
        # Make a cube-heavy op and a mem-heavy op where cube dominates
        op_cube = roofline_time("c", flops=1e15, vec_ops=0, mem_bytes=0, hw=self.hw)
        op_mem = roofline_time("m", flops=0, vec_ops=0, mem_bytes=1e9, hw=self.hw)
        total = sum_ops([op_cube, op_mem], "cube_dom")
        self.assertEqual(total.bottleneck, "CUBE")

    def test_comm_dominant(self):
        """When comm_time_s dominates, bottleneck is COMM."""
        op = OpProfile(name="comm_heavy", comm_time_s=10.0, time_s=10.0,
                       cube_time_s=0.001, vec_time_s=0.001, mem_time_s=0.001)
        total = sum_ops([op], "comm_dom")
        self.assertEqual(total.bottleneck, "COMM")

    def test_comm_bytes_aggregated(self):
        op1 = OpProfile(name="a", comm_bytes=100)
        op2 = OpProfile(name="b", comm_bytes=200)
        total = sum_ops([op1, op2], "comm_bytes")
        self.assertEqual(total.comm_bytes, 300)

    def test_vec_ops_aggregated(self):
        op1 = roofline_time("v1", flops=0, vec_ops=1e12, mem_bytes=0, hw=self.hw)
        op2 = roofline_time("v2", flops=0, vec_ops=2e12, mem_bytes=0, hw=self.hw)
        total = sum_ops([op1, op2], "vec_agg")
        self.assertAlmostEqual(total.vec_ops, 3e12, places=6)


if __name__ == "__main__":
    unittest.main()
