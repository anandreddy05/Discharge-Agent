"""
evaluate.py

Reads learning_results.json and produces:
  1. A before/after table (first iteration vs latest).
  2. An improvement curve (overall score per iteration).
  3. Per-section improvement breakdown.

Usage:
    python evaluate.py [--results learning_results.json]
"""

import argparse
import json
import os

from learning.reward import SCORED_SECTIONS

RESULTS_LOG = os.path.join(os.getcwd(), "learning_results.json")


def load_results(path: str) -> list:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Results file not found: {path}. Run learning_loop.py first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_improvement_curve(successful: list) -> None:
    print("\n=== IMPROVEMENT CURVE (overall reward per iteration) ===")
    print(f"  {'Iter':<6} {'Overall':>8}  {'Bar'}")
    print(f"  {'-' * 6} {'-' * 8}  {'-' * 22}")

    baseline = successful[0]["scores"]["overall"]

    for r in successful:
        it = r["iteration"]
        score = r["scores"]["overall"]
        delta = score - baseline
        bar = "█" * int(score * 20)
        sign = "+" if delta >= 0 else ""
        delta_str = (
            f"({sign}{delta:.4f})" if it > successful[0]["iteration"] else "(baseline)"
        )
        print(f"  {it:<6} {score:>8.4f}  {bar:<22} {delta_str}")


def print_section_table(successful: list) -> None:
    if len(successful) < 2:
        print("\n[Not enough iterations for before/after comparison]")
        return

    first = successful[0]["scores"]
    last = successful[-1]["scores"]

    print("\n=== SECTION-LEVEL BEFORE vs AFTER ===")
    print(f"  {'Section':<35} {'Before':>8} {'After':>8} {'Delta':>8}")
    print(f"  {'-' * 35} {'-' * 8} {'-' * 8} {'-' * 8}")

    improved = 0
    degraded = 0

    for section in SCORED_SECTIONS + ["overall"]:
        before = first.get(section, 0.0)
        after = last.get(section, 0.0)
        delta = after - before
        sign = "▲" if delta > 0.001 else ("▼" if delta < -0.001 else " ")
        if delta > 0.001:
            improved += 1
        elif delta < -0.001:
            degraded += 1
        print(f"  {section:<35} {before:>8.4f} {after:>8.4f} {sign}{delta:>+7.4f}")

    print(f"\n  Sections improved : {improved}")
    print(f"  Sections degraded : {degraded}")
    print(f"  Sections unchanged: {len(SCORED_SECTIONS) - improved - degraded}")


def print_memory_growth(results: list) -> None:
    print("\n=== MEMORY BANK GROWTH ===")
    print(f"  {'Iter':<6} {'Total entries':>14}")
    print(f"  {'-' * 6} {'-' * 14}")
    for r in results:
        if r.get("status") != "success":
            continue
        total = r.get("memory_stats_after", {}).get("total_entries", "?")
        print(f"  {r['iteration']:<6} {str(total):>14}")


def run_evaluation(results_path: str) -> None:
    results = load_results(results_path)
    successful = [r for r in results if r.get("status") == "success"]

    if not successful:
        print("No successful iterations found in results.")
        return

    print(f"\nLoaded {len(successful)} successful iteration(s) from {results_path}")

    print_improvement_curve(successful)
    print_section_table(successful)
    print_memory_growth(results)

    # Highlight weakest sections in latest run
    latest_scores = successful[-1]["scores"]
    weak = sorted([(s, latest_scores[s]) for s in SCORED_SECTIONS], key=lambda x: x[1])[
        :5
    ]
    print("\n=== TOP 5 WEAKEST SECTIONS (latest iteration) ===")
    for section, score in weak:
        print(f"  {section:<35} {score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Part 2 learning results")
    parser.add_argument(
        "--results", default=RESULTS_LOG, help="Path to learning_results.json"
    )
    args = parser.parse_args()
    run_evaluation(args.results)
