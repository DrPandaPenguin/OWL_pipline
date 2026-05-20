#!/usr/bin/env python3
"""
For each direct-pipeline node, classify into A/B/C/D categories (using same
rules as analyse_direct_overextraction.py), then check whether category
membership predicts:

  - degree (number of incident edges)
  - orphan-rate
  - which connected component the node sits in (largest / mid / singleton)
  - edge type / edge_source breakdown for the node's incident edges

Hypothesis: are B-category nodes (the 'extra 70' that other pipelines
validated) peripheral and disconnected, or are they integrated into the
main graph at the cost of more components?
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from experiments.precision_recall import parse_slide_ground_truth  # noqa

ROOT = os.path.join(_PROJECT_ROOT, "experiments", "stability_test")
LECTURE = os.path.join(_PROJECT_ROOT, "input", "ecommerce", "lec4")
LEX = 0.45


def _norm(s): return (s or "").strip().lower()
def _fuzzy(a, b): return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _best(label, pool, thr=LEX):
    if not label or not pool:
        return False
    n = _norm(label)
    for p in pool:
        if _fuzzy(n, p) >= thr:
            return True
    return False


def _categorise(label, sa, ms, gt):
    in_other = _best(label, sa) or _best(label, ms)
    in_gt = _best(label, gt)
    if in_other and in_gt: return "A"
    if in_other and not in_gt: return "B"
    if not in_other and in_gt: return "C"
    return "D"


def _load_pool(condition):
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, condition, "run*.json"))):
        if p.endswith("_FAILED.json"): continue
        d = json.load(open(p, encoding="utf-8"))
        out.extend(n.get("label", "") for n in d["graph"]["nodes"] if n.get("label"))
    return out


def _components(nodes_ids, edges):
    adj = {i: set() for i in nodes_ids}
    for e in edges:
        s = e.get("from") or e.get("source") or e.get("source_id")
        t = e.get("to") or e.get("target") or e.get("target_id")
        if s in adj and t in adj:
            adj[s].add(t); adj[t].add(s)
    seen = set(); comps = []
    for nid in nodes_ids:
        if nid in seen: continue
        stack = [nid]; comp = []
        while stack:
            cur = stack.pop()
            if cur in seen: continue
            seen.add(cur); comp.append(cur)
            stack.extend(adj.get(cur, ()))
        comps.append(comp)
 # node -> component idx (sorted by size desc, so 0 = largest)
    comps_sorted = sorted(comps, key=len, reverse=True)
    nid_to_comp = {}
    for ci, comp in enumerate(comps_sorted):
        for nid in comp:
            nid_to_comp[nid] = ci
    return adj, comps_sorted, nid_to_comp


def _summarise(values):
    if not values: return None
    return {"mean": round(statistics.fmean(values), 3),
            "sd": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values), "n": len(values)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="run01.json")
    ap.add_argument("--all-runs", action="store_true",
                    help="Aggregate across all 15 direct runs")
    args = ap.parse_args()

    sa_pool = _load_pool("slide_anchored")
    ms_pool = _load_pool("multi_stage")
    gt = [g["concept"] for g in parse_slide_ground_truth(
        open(os.path.join(LECTURE, "slides.txt"), encoding="utf-8").read())]

 # Which runs to analyse
    if args.all_runs:
        run_paths = sorted(glob.glob(os.path.join(ROOT, "direct", "run*.json")))
        run_paths = [p for p in run_paths if not p.endswith("_FAILED.json")]
    else:
        run_paths = [os.path.join(ROOT, "direct", args.run)]

 # Aggregators
    cat_degrees = defaultdict(list)        # cat -> list of node degrees
    cat_orphans = defaultdict(list)        # cat -> list of 0/1 orphan flags
    cat_comp_sizes = defaultdict(list)     # cat -> list of component sizes containing this node
    cat_in_largest = defaultdict(list)     # cat -> list of 0/1 flags
    cat_in_singleton = defaultdict(list)
    edge_type_by_cat = defaultdict(Counter)  # cat -> Counter of edge_types on incident edges
    edge_source_by_cat = defaultdict(Counter)  # cat -> Counter of edge_source

    n_runs_used = 0
    for run_path in run_paths:
        d = json.load(open(run_path, encoding="utf-8"))
        nodes = d["graph"]["nodes"]
        edges = d["graph"]["edges"]
        if not nodes:
            continue
        n_runs_used += 1
        ids = [n.get("id") for n in nodes]
        adj, comps_sorted, nid_to_comp = _components(ids, edges)
        largest_size = len(comps_sorted[0]) if comps_sorted else 0
        comp_size_by_node = {nid: len(comps_sorted[nid_to_comp[nid]])
                             for nid in ids}

        for n in nodes:
            label = n.get("label", "")
            nid = n.get("id")
            cat = _categorise(label, sa_pool, ms_pool, gt)
            deg = len(adj.get(nid, set()))
            comp_idx = nid_to_comp.get(nid, -1)
            comp_size = comp_size_by_node.get(nid, 0)

            cat_degrees[cat].append(deg)
            cat_orphans[cat].append(1 if deg == 0 else 0)
            cat_comp_sizes[cat].append(comp_size)
            cat_in_largest[cat].append(1 if comp_idx == 0 else 0)
            cat_in_singleton[cat].append(1 if comp_size == 1 else 0)

 # incident edges
            for e in edges:
                s = e.get("from") or e.get("source") or e.get("source_id")
                t = e.get("to") or e.get("target") or e.get("target_id")
                if s == nid or t == nid:
                    et = e.get("edge_type") or e.get("relation_type") or "?"
                    es = e.get("edge_source") or "?"
                    edge_type_by_cat[cat][et] += 1
                    edge_source_by_cat[cat][es] += 1

 # Print
    print(f"Analysed {n_runs_used} direct run(s)")
    print()
    print(f"{'Category':>10} {'n_nodes':>8} {'mean_deg':>9} {'sd_deg':>8} "
          f"{'orphan%':>8} {'in_largest%':>13} {'in_singleton%':>15} {'mean_comp_size':>16}")
    print("-" * 100)
    for cat in "ABCD":
        n = len(cat_degrees[cat])
        if not n:
            print(f"  {cat}    no nodes")
            continue
        print(f"  {cat}    {n:>6}  "
              f"{statistics.fmean(cat_degrees[cat]):>9.2f}  "
              f"{statistics.stdev(cat_degrees[cat]) if n>1 else 0:>7.2f}  "
              f"{100*statistics.fmean(cat_orphans[cat]):>7.1f}  "
              f"{100*statistics.fmean(cat_in_largest[cat]):>12.1f}  "
              f"{100*statistics.fmean(cat_in_singleton[cat]):>14.1f}  "
              f"{statistics.fmean(cat_comp_sizes[cat]):>15.2f}")

    print()
    print("Edge_type distribution per category (incident edges):")
    for cat in "ABCD":
        if not edge_type_by_cat[cat]:
            continue
        total = sum(edge_type_by_cat[cat].values())
        items = sorted(edge_type_by_cat[cat].items(), key=lambda x: -x[1])
        print(f"  {cat}: " + ", ".join(f"{t}={c}" for t, c in items[:8])
              + f"  (total incident = {total})")
    print()
    print("Edge_source per category:")
    for cat in "ABCD":
        if not edge_source_by_cat[cat]:
            continue
        total = sum(edge_source_by_cat[cat].values())
        print(f"  {cat}: " + dict(edge_source_by_cat[cat]).__repr__()
              + f"  (total incident = {total})")

    print()
    print("CONCLUSIONS:")
    a_orph = 100*statistics.fmean(cat_orphans["A"]) if cat_orphans["A"] else 0
    b_orph = 100*statistics.fmean(cat_orphans["B"]) if cat_orphans["B"] else 0
    d_orph = 100*statistics.fmean(cat_orphans["D"]) if cat_orphans["D"] else 0
    a_deg = statistics.fmean(cat_degrees["A"]) if cat_degrees["A"] else 0
    b_deg = statistics.fmean(cat_degrees["B"]) if cat_degrees["B"] else 0
    d_deg = statistics.fmean(cat_degrees["D"]) if cat_degrees["D"] else 0
    a_lg = 100*statistics.fmean(cat_in_largest["A"]) if cat_in_largest["A"] else 0
    b_lg = 100*statistics.fmean(cat_in_largest["B"]) if cat_in_largest["B"] else 0

    print(f"  A degree {a_deg:.2f} vs B degree {b_deg:.2f} vs D degree {d_deg:.2f}")
    print(f"  A orphan {a_orph:.0f}% vs B orphan {b_orph:.0f}% vs D orphan {d_orph:.0f}%")
    print(f"  A in_largest {a_lg:.0f}% vs B in_largest {b_lg:.0f}%")


if __name__ == "__main__":
    main()
