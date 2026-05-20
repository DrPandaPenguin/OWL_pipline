#!/usr/bin/env python3
"""
Cross-pipeline node comparison: is `direct`'s extra extraction signal or noise?

Categorises every node label produced by any of the three pipelines
(slide_anchored / multi_stage / direct) into:

  A. Validated by all three sources (other pipelines + slide GT)
  B. Validated by SA or MS (cross-pipeline agreement, but not in GT)
  C. Validated by GT only (the LLM saw it; the other pipeline did not)
  D. Validated by nothing — over-extraction candidate

Per-condition categorisation answers:
  - How many of `direct`'s 115 nodes/run sit in (D)?
  - Are SA / MS less prone to (D) than direct?

Uses lexical SequenceMatcher @ 0.45 (no API). Optional --semantic flag
re-runs with cached embeddings if available.

Usage:
    python -m experiments.analyse_direct_overextraction
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from experiments.precision_recall import parse_slide_ground_truth  # noqa

ROOT = os.path.join(_PROJECT_ROOT, "experiments", "stability_test")
LECTURE = os.path.join(_PROJECT_ROOT, "input", "ecommerce", "lec4")

LEX_THRESHOLD = 0.45  # SequenceMatcher cutoff (matches §4.2 metric)


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _fuzzy(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _best_match(label: str, pool: list[str], threshold: float = LEX_THRESHOLD) -> bool:
    """Does `label` have any best-match >= threshold in `pool`?"""
    if not label or not pool:
        return False
    n = _norm(label)
    for p in pool:
        if _fuzzy(n, p) >= threshold:
            return True
    return False


def _load_labels(condition: str) -> tuple[list[str], list[list[str]]]:
    """Return (flat_pool, per_run_lists) of labels for a condition."""
    flat: list[str] = []
    per_run: list[list[str]] = []
    for path in sorted(glob.glob(os.path.join(ROOT, condition, "run*.json"))):
        if path.endswith("_FAILED.json"):
            continue
        d = json.load(open(path, encoding="utf-8"))
        labs = [n.get("label", "") for n in d["graph"]["nodes"] if n.get("label")]
        flat.extend(labs)
        per_run.append(labs)
    return flat, per_run


def _categorise(label: str,
                sa_pool: list[str],
                ms_pool: list[str],
                gt_pool: list[str]) -> str:
    in_sa = _best_match(label, sa_pool)
    in_ms = _best_match(label, ms_pool)
    in_gt = _best_match(label, gt_pool)
    other_pipe = in_sa or in_ms

    if other_pipe and in_gt:
        return "A_validated_all"
    if other_pipe and not in_gt:
        return "B_pipelines_agree"
    if not other_pipe and in_gt:
        return "C_gt_only"
    return "D_no_validation"  # overextraction candidate


def main():
    slide_text = open(os.path.join(LECTURE, "slides.txt"), encoding="utf-8").read()
    gt_pool = [g["concept"] for g in parse_slide_ground_truth(slide_text)]

 # Pool all labels from all 15 runs per condition.
    sa_flat, sa_runs = _load_labels("slide_anchored")
    ms_flat, ms_runs = _load_labels("multi_stage")
    di_flat, di_runs = _load_labels("direct")

    print(f"Pools: SA={len(sa_flat)} labels (across {len(sa_runs)} runs), "
          f"MS={len(ms_flat)} ({len(ms_runs)} runs), "
          f"DI={len(di_flat)} ({len(di_runs)} runs), GT={len(gt_pool)} concepts")
    print()

 # Categorise every label in every run, against the OTHER two pipelines + GT.
    summary: dict[str, dict[str, list[float]]] = {
        cond: {"A": [], "B": [], "C": [], "D": []}
        for cond in ("slide_anchored", "multi_stage", "direct")
    }
    sample_labels: dict[str, dict[str, list[str]]] = {
        cond: {"A": [], "B": [], "C": [], "D": []}
        for cond in summary
    }

    def _process(cond: str, runs: list[list[str]],
                 other_a: list[str], other_b: list[str]):
        for run_labels in runs:
            counts = Counter()
            for lab in run_labels:
                cat = _categorise(lab, other_a, other_b, gt_pool)
                key = cat[0]  # 'A'/'B'/'C'/'D'
                counts[key] += 1
                if len(sample_labels[cond][key]) < 12:
                    sample_labels[cond][key].append(lab)
            n = max(1, len(run_labels))
            for k in "ABCD":
                summary[cond][k].append(counts[k] / n)

    _process("slide_anchored", sa_runs, ms_flat, di_flat)
    _process("multi_stage",    ms_runs, sa_flat, di_flat)
    _process("direct",         di_runs, sa_flat, ms_flat)

 # Print headline table
    print(f"{'condition':18s}  "
          f"{'A: all 3':>10s}  {'B: pipes':>10s}  "
          f"{'C: GT-only':>11s}  {'D: nothing':>11s}")
    print("-" * 68)
    for cond, cats in summary.items():
        line = f"{cond:18s}  "
        for k in "ABCD":
            arr = cats[k]
            mean = sum(arr) / max(1, len(arr))
            line += f"{mean*100:>9.1f}%  " if k != "D" else f"{mean*100:>9.1f}%"
        print(line)

    print()
    print("Interpretation:")
    print("  A = node validated by other pipeline AND slide GT (core concept)")
    print("  B = node validated by other pipeline but not in slide GT  (cross-pipeline agreement, lecturer omitted from slides)")
    print("  C = node in slide GT but not in any other pipeline       (this pipeline's unique find)")
    print("  D = node not validated by anyone                          (likely over-extraction)")
    print()

 # Sample dump for direct's "D" the over-extraction candidates
    print("=" * 68)
    print("SAMPLES — direct's D-category (over-extraction candidates):")
    for s in sample_labels["direct"]["D"]:
        print(f"  • {s}")
    print()
    print("SAMPLES — direct's C-category (concepts other pipelines missed):")
    for s in sample_labels["direct"]["C"]:
        print(f"  • {s}")
    print()
    print("SAMPLES — slide_anchored's D-category (for comparison):")
    for s in sample_labels["slide_anchored"]["D"]:
        print(f"  • {s}")

 # Save full results
    out = {
        "lexical_threshold": LEX_THRESHOLD,
        "pools": {"sa": len(sa_flat), "ms": len(ms_flat),
                  "di": len(di_flat), "gt": len(gt_pool)},
        "per_condition_means": {
            cond: {k: round(sum(vs)/max(1,len(vs)), 4) for k, vs in cats.items()}
            for cond, cats in summary.items()
        },
        "per_condition_per_run": {
            cond: {k: [round(v, 4) for v in vs] for k, vs in cats.items()}
            for cond, cats in summary.items()
        },
        "samples": sample_labels,
    }
    out_path = os.path.join(ROOT, "direct_overextraction_analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
