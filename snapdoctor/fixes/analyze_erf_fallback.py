"""
Inspect a profile_data execution_detail for compute_unit placement and
per-op cost (using execution_cycles, since execution_time is zeroed in
this export format).
Usage: python -m snapdoctor.fixes.analyze_erf_fallback <profile_json_path>
"""

import argparse
import json
from collections import Counter, defaultdict


def analyze(profile_data: dict, top_n: int = 20):
    detail = profile_data.get("execution_detail", [])
    if not detail:
        print("No execution_detail found in profile_data.")
        print("Top-level keys present:", list(profile_data.keys()))
        return

    watch_types = ("erf", "layernorm", "softmax", "gelu", "matmul", "gemm")
    unit_counts = Counter()
    unit_cycles = Counter()
    type_cycles = defaultdict(int)
    type_counts = Counter()
    flagged = []

    for op in detail:
        name = op.get("name", "")
        unit = op.get("compute_unit", "UNKNOWN")
        cycles = op.get("execution_cycles", 0) or 0
        op_type = op.get("type", "?")

        unit_counts[unit] += 1
        unit_cycles[unit] += cycles
        type_cycles[op_type] += cycles
        type_counts[op_type] += 1

        if any(w in name.lower() for w in watch_types):
            flagged.append((name, op_type, unit, cycles))

    total_cycles = sum(unit_cycles.values())

    print("=== Overall compute_unit distribution (op count) ===")
    total_ops = sum(unit_counts.values())
    for unit, count in unit_counts.most_common():
        print(f"  {unit:>6}: {count:4d} ops ({100*count/total_ops:.1f}%)")

    print("\n=== Overall compute_unit distribution (cycles) ===")
    for unit, c in unit_cycles.most_common():
        pct = 100 * c / total_cycles if total_cycles else 0
        print(f"  {unit:>6}: {c:12,d} cycles ({pct:.1f}% of total)")

    print(f"\n=== Cost by node 'type', top {top_n} (this is where the real 72ms is going) ===")
    for op_type, cycles in sorted(type_cycles.items(), key=lambda x: -x[1])[:top_n]:
        pct = 100 * cycles / total_cycles if total_cycles else 0
        count = type_counts[op_type]
        print(f"  {op_type:<20} {cycles:12,d} cycles ({pct:5.1f}%)  across {count:4d} nodes  (~{cycles//max(count,1):,} cycles/node avg)")

    print(f"\n=== Individual most expensive nodes, top {top_n} ===")
    top_nodes = sorted(detail, key=lambda op: op.get("execution_cycles", 0) or 0, reverse=True)[:top_n]
    for op in top_nodes:
        print(f"  {op.get('name','?'):<55} [{op.get('type','?'):<10}] -> {op.get('compute_unit','?'):>4}  {op.get('execution_cycles',0):>12,d} cycles")

    print("\n=== Attention/GELU/Norm-related ops (compute_unit + cost) ===")
    if not flagged:
        print("  None matched.")
    else:
        flagged_sorted = sorted(flagged, key=lambda x: -x[3])[:top_n]
        for name, op_type, unit, cycles in flagged_sorted:
            flag = "  <-- NOT NPU" if unit != "NPU" else ""
            print(f"  {name:<50} [{op_type:<10}] -> {unit:>4}  {cycles:>10,d} cycles{flag}")

    npu_flagged = sum(1 for *_, unit, _ in flagged if unit == "NPU")
    print(f"\n{npu_flagged}/{len(flagged)} watched ops ran on NPU.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("profile_json", help="Path to a saved profile_data JSON")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    with open(args.profile_json) as f:
        data = json.load(f)
    analyze(data, top_n=args.top)