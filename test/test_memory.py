"""Tests for perf_model.memory — KV cache and weight memory analysis."""

import unittest

from test.helpers import make_config

from perf_model.roofline import bytes2
from perf_model.memory import kv_cache_memory, weight_memory_per_rank


# ── KV Cache Memory ──────────────────────────────────────────────────────

class TestKVCacheMemory(unittest.TestCase):
    """kv_cache_memory per-layer and total KV cache sizing."""

    def test_ratio1_swa_cache(self):
        """ratio=1 layer: SWA cache = B * W * kv_dim * 2 (K=V shared, window only)."""
        cfg = make_config()
        result = kv_cache_memory(cfg)
        B = cfg.rt.batch_size // cfg.rt.dp
        W = cfg.model.window_size
        kv_dim = cfg.model.kv_dim
        expected = B * W * kv_dim * 2
        layer0 = result["layers"][0]
        self.assertEqual(layer0["type"], "SWA")
        self.assertEqual(layer0["bytes"], expected)

    def test_ratio4_compressed_with_index(self):
        """ratio=4 layer: compressed + SWA + index (S//4=32 > topK=16). K=V shared."""
        cfg = make_config()
        result = kv_cache_memory(cfg)
        B = cfg.rt.batch_size // cfg.rt.dp
        S = cfg.rt.seq_len
        m = cfg.model
        ratio = 4
        S_comp = S // ratio

        # Verify use_index = S_comp > index_topk => 32 > 16 => True
        self.assertGreater(S_comp, m.index_topk)

        layer1 = result["layers"][1]
        self.assertEqual(layer1["type"], "C4A")

        comp_bytes = B * S_comp * m.compress_c_kv * 2
        swa_bytes = B * m.window_size * m.kv_dim * 2
        idx_bytes = B * S_comp * m.index_head_dim * 2

        self.assertEqual(layer1["comp_bytes"], comp_bytes)
        self.assertEqual(layer1["swa_bytes"], swa_bytes)
        self.assertIn("idx_bytes", layer1)
        self.assertEqual(layer1["idx_bytes"], idx_bytes)
        self.assertEqual(layer1["bytes"], comp_bytes + swa_bytes + idx_bytes)

    def test_ratio128_no_index(self):
        """ratio=128 layer: S//128=1, not > topK=16, so NO index cache. K=V shared."""
        cfg = make_config()
        result = kv_cache_memory(cfg)
        B = cfg.rt.batch_size // cfg.rt.dp
        S = cfg.rt.seq_len
        m = cfg.model
        ratio = 128
        S_comp = S // ratio

        # Verify use_index = S_comp > index_topk => 1 > 16 => False
        self.assertFalse(S_comp > m.index_topk)

        layer2 = result["layers"][2]
        self.assertEqual(layer2["type"], "C128A")

        comp_bytes = B * S_comp * m.compress_c_kv * 2
        swa_bytes = B * m.window_size * m.kv_dim * 2

        self.assertEqual(layer2["comp_bytes"], comp_bytes)
        self.assertEqual(layer2["swa_bytes"], swa_bytes)
        self.assertNotIn("idx_bytes", layer2)
        self.assertEqual(layer2["bytes"], comp_bytes + swa_bytes)

    def test_total_sums_all_layers(self):
        """Total bytes is sum across all layers."""
        cfg = make_config()
        result = kv_cache_memory(cfg)
        expected_total = sum(result["layers"][i]["bytes"]
                            for i in range(cfg.model.num_hidden_layers))
        self.assertEqual(result["total_bytes"], expected_total)

    def test_dp_splits_batch(self):
        """dp splits batch: B = batch_size // dp."""
        cfg_dp1 = make_config(batch_size=4, dp=1)
        cfg_dp2 = make_config(batch_size=4, dp=2)
        r1 = kv_cache_memory(cfg_dp1)
        r2 = kv_cache_memory(cfg_dp2)
        # dp=2 halves effective batch, so total should be half
        self.assertEqual(r1["total_bytes"], 2 * r2["total_bytes"])

    def test_all_layers_present(self):
        """All layers present in result."""
        cfg = make_config()
        result = kv_cache_memory(cfg)
        for i in range(cfg.model.num_hidden_layers):
            self.assertIn(i, result["layers"])

    def test_layer_type_labels(self):
        """Layer type labels correct: 'SWA', 'C4A', 'C128A'."""
        cfg = make_config()
        result = kv_cache_memory(cfg)
        self.assertEqual(result["layers"][0]["type"], "SWA")
        self.assertEqual(result["layers"][1]["type"], "C4A")
        self.assertEqual(result["layers"][2]["type"], "C128A")
        self.assertEqual(result["layers"][3]["type"], "C4A")

    def test_all_bytes_positive(self):
        """All per-layer bytes are positive."""
        cfg = make_config()
        result = kv_cache_memory(cfg)
        for i in range(cfg.model.num_hidden_layers):
            self.assertGreater(result["layers"][i]["bytes"], 0)


# ── Weight Memory ─────────────────────────────────────────────────────────

class TestWeightMemory(unittest.TestCase):
    """weight_memory_per_rank formula verification."""

    def test_attn_per_layer_formula(self):
        """attn_per_layer = bytes2(w_dq + w_uq + w_kv + w_wo_a + w_wo_b) with TP splitting."""
        cfg = make_config()
        result = weight_memory_per_rank(cfg)
        m = cfg.model
        TP = cfg.rt.tp
        H = m.hidden_size

        w_dq = H * m.q_lora_rank
        w_uq = m.q_lora_rank * (m.num_attention_heads // TP) * (m.head_dim + m.rope_head_dim)
        w_kv = H * m.kv_dim
        Ng = m.o_groups
        w_wo_a = (Ng // TP) * (m.num_attention_heads // Ng) * m.head_dim * m.o_lora_rank
        w_wo_b = (m.o_mid_dim // TP) * H

        expected = bytes2(w_dq + w_uq + w_kv + w_wo_a + w_wo_b)
        self.assertEqual(result["attn_per_layer"], expected)

    def test_moe_per_layer_formula(self):
        """moe_per_layer = bytes2(gate + routed/EP + shared)."""
        cfg = make_config()
        result = weight_memory_per_rank(cfg)
        m = cfg.model
        H = m.hidden_size
        EP = cfg.rt.ep

        w_gate = H * m.n_routed_experts
        experts_per_rank = m.n_routed_experts // EP
        w_routed = experts_per_rank * 3 * H * m.moe_inter_dim
        w_shared = m.n_shared_experts * 3 * H * m.moe_inter_dim

        expected = bytes2(w_gate + w_routed + w_shared)
        self.assertEqual(result["moe_per_layer"], expected)

    def test_tp_scaling_attn(self):
        """Higher TP reduces attn_per_layer (Q proj and wo split by TP)."""
        cfg_tp2 = make_config(tp=2, ep=4)
        cfg_tp4 = make_config(tp=4, ep=4)
        r2 = weight_memory_per_rank(cfg_tp2)
        r4 = weight_memory_per_rank(cfg_tp4)
        self.assertGreater(r2["attn_per_layer"], r4["attn_per_layer"])

    def test_ep_scaling_moe(self):
        """Higher EP reduces moe_per_layer (routed experts split by EP)."""
        cfg_ep2 = make_config(tp=2, ep=2)
        cfg_ep4 = make_config(tp=2, ep=4)
        r2 = weight_memory_per_rank(cfg_ep2)
        r4 = weight_memory_per_rank(cfg_ep4)
        self.assertGreater(r2["moe_per_layer"], r4["moe_per_layer"])

    def test_total_includes_all_components(self):
        """Total includes attn + moe + mhc + norm + embed + lm_head + final_norm."""
        cfg = make_config()
        result = weight_memory_per_rank(cfg)
        m = cfg.model
        H = m.hidden_size
        TP = cfg.rt.tp

        total_attn = result["total_attn"]
        total_moe = result["total_moe"]
        total_other = result["total_other"]

        self.assertEqual(result["total"], total_attn + total_moe + total_other)
        self.assertGreater(result["total"], 0)

    def test_embedding_and_lm_head(self):
        """embedding = bytes2(vocab * H), lm_head = bytes2(H * vocab//TP)."""
        cfg = make_config()
        result = weight_memory_per_rank(cfg)
        m = cfg.model
        H = m.hidden_size
        TP = cfg.rt.tp

        self.assertEqual(result["embedding"], bytes2(m.vocab_size * H))
        self.assertEqual(result["lm_head"], bytes2(H * (m.vocab_size // TP)))

    def test_layer_type_counts(self):
        """n_swa_layers and n_comp_layers counts correct for [1,4,128,4]."""
        cfg = make_config()
        result = weight_memory_per_rank(cfg)
        # compress_ratios = [1, 4, 128, 4]
        self.assertEqual(result["n_swa_layers"], 1)
        self.assertEqual(result["n_comp_layers"], 3)


if __name__ == "__main__":
    unittest.main()
