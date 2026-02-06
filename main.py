#!/usr/bin/env python3
"""CLI entry point for DeepSeek V4 Performance Model."""

import sys

from perf_model import Config, prefill_model, decode_step, decode_model
from perf_model.report import (
    fmt_ms, print_config_summary, print_phase_report, print_memory_report,
)


def main():
    if len(sys.argv) < 5:
        print("Usage: python main.py <device.json> <network.json> <model.json> <runtime.json>")
        sys.exit(1)

    cfg = Config.from_json(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

    # Validate compress_ratios length
    if len(cfg.model.compress_ratios) != cfg.model.num_layers:
        print(f"ERROR: compress_ratios length ({len(cfg.model.compress_ratios)}) "
              f"!= num_layers ({cfg.model.num_layers})")
        sys.exit(1)

    print_config_summary(cfg)

    # --- Prefill ---
    prefill = prefill_model(cfg)
    print_phase_report(prefill, cfg)

    # --- Decode ---
    # Get detailed single-step profile for reporting
    decode_first_step = decode_step(cfg.rt.seq_len, cfg)
    decode_total = decode_model(cfg)
    print_phase_report(decode_total, cfg, detailed_step=decode_first_step)

    # --- Memory ---
    print_memory_report(cfg)

    # --- End-to-end summary ---
    print("=" * 80)
    print("  END-TO-END SUMMARY")
    print("=" * 80)
    print()
    print(f"  Prefill ({cfg.rt.batch_size}×{cfg.rt.seq_len} tokens):")
    print(f"    Time:       {fmt_ms(prefill.total_time_s)} ms")
    if prefill.total_time_s > 0:
        print(f"    Throughput: {prefill.total_tokens / prefill.total_time_s:.1f} tokens/s")
    print()
    print(f"  Decode ({cfg.rt.output_len} tokens):")
    print(f"    Total time: {fmt_ms(decode_total.total_time_s)} ms")
    print(f"    First step: {fmt_ms(decode_first_step.total_time_s)} ms")
    if decode_total.total_time_s > 0:
        print(f"    Avg tok/s:  {decode_total.total_tokens / decode_total.total_time_s:.1f}")
    print()
    total_time = prefill.total_time_s + decode_total.total_time_s
    total_output_tokens = cfg.rt.batch_size * cfg.rt.output_len
    print(f"  Total (prefill + decode): {fmt_ms(total_time)} ms")
    if total_time > 0:
        print(f"  Output tokens/s:          {total_output_tokens / total_time:.1f}")
    print()


if __name__ == "__main__":
    main()
