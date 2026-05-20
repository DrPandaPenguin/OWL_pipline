#!/usr/bin/env python3
"""
Generic per-experiment aggregator. Reads runs from a given experiment dir
(structure: <root>/<condition>/runNN.json), computes per-condition stats,
between-condition Welch t / F-tests on F1, and pairwise overlap.

Used for §4.3 and §4.4. Compatible with the per-run save format from
run_stability_test.py.

Usage:
    python -m experiments.aggregate_experiment experiments/exp1_pipeline_architecture
    python -m experiments.aggregate_experiment experiments/exp3_strict_vs_soft
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import sys
from itertools import combinations
from typing import Any, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

# Reuse the same helpers used by aggregate_stability where possible.
from experiments.aggregate_stability import (  # noqa
    _summarise, _welch_t, _f_test_variances,
    _node_labels, _edge_keys, _jaccard, _pairwise, _full_intersection,
    METRICS_FIELDS,
)


def _load_runs(exp_root: str, condition: str) -> list[dict]:
    pattern = os.path.join(exp_root, condition, "run*.json")
    out = []
    for path in sorted(glob.glob(pattern)):
        if path.endswith("_FAILED.json"):
            continue
        with open(path, encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def _list_failed(exp_root: str, condition: str) -> list[str]:
    return sorted(glob.glob(os.path.join(exp_root, condition, "run*_FAILED.json")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_root", help="experiments/<expN_name>")
    ap.add_argument("--no-semantic", action="store_true")
    args = ap.parse_args()

    exp_root = args.exp_root
    if not os.path.isdir(exp_root):
        print(f"Error: not a directory: {exp_root}")
        sys.exit(1)

    do_semantic = False
    if not args.no_semantic:
        try:
            from experiments.semantic_matching import semantic_jaccard, SEM_THRESHOLDS
            from experiments.aggregate_stability import (
                _pairwise_semantic, SEM_THRESHOLDS, SEM_MODEL,
            )
            do_semantic = True
        except Exception:
            try:
                from experiments.aggregate_stability import (
                    _pairwise_semantic, SEM_THRESHOLDS, SEM_MODEL,
                )
                do_semantic = True
            except Exception:
                pass

    conditions = sorted([
        d for d in os.listdir(exp_root)
        if os.path.isdir(os.path.join(exp_root, d)) and not d.startswith(".")
    ])

    per_condition: dict[str, Any] = {}
    f1_by_cond: dict[str, list[float]] = {}

    for cond in conditions:
        runs = _load_runs(exp_root, cond)
        failed = _list_failed(exp_root, cond)
        if not runs:
            per_condition[cond] = {
                "n_runs": 0,
                "n_failed": len(failed),
                "failed_files": [os.path.basename(f) for f in failed],
            }
            continue

        graphs = [r["graph"] for r in runs]
        metrics_lists = {fld: [r["metrics"].get(fld) for r in runs]
                         for fld in METRICS_FIELDS}
        per_run_summary = {fld: _summarise(vs)
                           for fld, vs in metrics_lists.items()}
        f1_by_cond[cond] = [v for v in metrics_lists["f1_at_45"] if v is not None]

        pw = {
            "node_overlap":     _pairwise(graphs, lambda g: _node_labels(g)),
            "backbone_overlap": _pairwise(graphs, lambda g: _node_labels(g, backbone_only=True)),
            "edge_overlap":     _pairwise(graphs, lambda g: _edge_keys(g, "all")),
            "explicit_overlap": _pairwise(graphs, lambda g: _edge_keys(g, "explicit")),
            "inferred_overlap": _pairwise(graphs, lambda g: _edge_keys(g, "inferred")),
        }

        nwi = {
            "node":     _full_intersection(graphs, lambda g: _node_labels(g)),
            "backbone": _full_intersection(graphs, lambda g: _node_labels(g, backbone_only=True)),
            "edge":     _full_intersection(graphs, lambda g: _edge_keys(g, "all")),
        }

        runtimes = [r.get("runtime", {}) for r in runs]
        wall_summary = _summarise([r.get("wall_time_s") for r in runs])
        cost_summary = _summarise([rt.get("cost_usd") for rt in runtimes])

        sem_block: dict[str, Any] = {}
        if do_semantic:
            try:
                sem_block["pairwise_semantic_node_overlap"] = {
                    f"{t:.2f}": _pairwise_semantic(graphs, t, backbone_only=False)
                    for t in SEM_THRESHOLDS
                }
                sem_block["pairwise_semantic_backbone_overlap"] = {
                    f"{t:.2f}": _pairwise_semantic(graphs, t, backbone_only=True)
                    for t in SEM_THRESHOLDS
                }
            except Exception as e:
                sem_block["error"] = str(e)

        per_condition[cond] = {
            "n_runs": len(runs),
            "n_failed": len(failed),
            "metrics": per_run_summary,
            "pairwise": pw,
            "n_way_intersection": nwi,
            "wall_time_s": wall_summary,
            "cost_usd": cost_summary,
            **({"semantic": sem_block} if do_semantic else {}),
        }

 # Between-condition tests
    between = {}
    if len(f1_by_cond) >= 2:
        names = list(f1_by_cond.keys())
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                key = f"{a}_vs_{b}"
                fa, fb = f1_by_cond[a], f1_by_cond[b]
                between[key] = {
                    "welch_t_F1": _welch_t(fa, fb),
                    "f_test_F1_variances": _f_test_variances(fa, fb),
                    "mean_F1_a": round(statistics.fmean(fa), 4) if fa else None,
                    "mean_F1_b": round(statistics.fmean(fb), 4) if fb else None,
                    "sd_F1_a": round(statistics.stdev(fa), 4) if len(fa) > 1 else None,
                    "sd_F1_b": round(statistics.stdev(fb), 4) if len(fb) > 1 else None,
                }

    payload = {
        "experiment_root": exp_root,
        "lecture": "ecommerce/lec4",
        "per_condition": per_condition,
        "between": between,
    }

    out_path = os.path.join(exp_root, "metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"wrote {out_path}\n")

 # Summary print
    for cond, info in per_condition.items():
        if info.get("n_runs", 0) == 0:
            print(f"[{cond}] n=0  (failed: {info.get('n_failed', 0)})")
            continue
        print(f"[{cond}] n={info['n_runs']}  failed={info.get('n_failed', 0)}")
        for fld in ("n_nodes", "n_edges", "n_components", "f1_at_45",
                    "backbone_recall", "edge_type_diversity"):
            s = info["metrics"][fld]
            if s["mean"] is None:
                continue
            print(f"  {fld:20s} mean={s['mean']:>8.4f}  sd={s['sd']:>7.4f}  range=[{s['min']:>7.4f}, {s['max']:>7.4f}]")

    if between:
        print()
        for k, v in between.items():
            wt = v["welch_t_F1"]
            ft = v["f_test_F1_variances"]
            print(f"  {k}:")
            print(f"    F1 means: {v['mean_F1_a']} vs {v['mean_F1_b']}")
            print(f"    Welch t = {wt['t']}  df={wt['df']}  p={wt['p_two_sided']}")
            print(f"    F-test  F={ft['F']}  df=({ft['df1']},{ft['df2']})  p={ft['p_two_sided']}")


if __name__ == "__main__":
    main()
