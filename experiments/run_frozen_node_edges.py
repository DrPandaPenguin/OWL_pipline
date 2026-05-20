#!/usr/bin/env python3
"""
Frozen-node edge experiments — covers §4.4 and §4.7.

§4.4 (Strict vs Soft, no KU): 3 conditions × 3 runs on a frozen node list.
§4.7 (Frozen edges):          1 condition (both passes) × 15 runs.

All runs use:
  - Same frozen node list (from a canonical slide_anchored run)
  - Same lecture (lec4)
  - Same model (gpt-5.2)
  - extract_edges() with knowledge_units=[] → strict path uses
    extract_edges_grounded_transcript() (transcript verbatim, no KU).

Usage:
  # §4.4: ablation on edge passes
  python -m experiments.run_frozen_node_edges --mode ablation --runs 3
  # §4.7: stability of full edge step
  python -m experiments.run_frozen_node_edges --mode stability --runs 15
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from itertools import combinations

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from experiments.config import DEFAULT_CONFIG  # noqa
from src.extract_edges import extract_edges  # noqa
from src.usage_tracker import install as install_usage, reset as reset_usage, snapshot as usage_snapshot  # noqa

install_usage()

LECTURE_DIR = os.path.join(_PROJECT_ROOT, "input", "ecommerce", "lec4")
CANON_PATH = os.path.join(_PROJECT_ROOT, "experiments", "frozen_canonical_nodes.json")


# ---------------------------------------------------------------------------
# Canonical node list — one frozen list, used by both §4.4 and §4.7
# ---------------------------------------------------------------------------

def _bootstrap_canonical_nodes() -> list:
    """Load a canonical node list from a saved slide_anchored stability run."""
    if os.path.isfile(CANON_PATH):
        return json.load(open(CANON_PATH, encoding="utf-8"))["nodes"]

    # Use slide_anchored run01 from existing stability test
    src = os.path.join(_PROJECT_ROOT, "experiments", "stability_test",
                       "slide_anchored", "run01.json")
    d = json.load(open(src, encoding="utf-8"))
    nodes = d["graph"]["nodes"]
    # Strip enrichment / KU references — these are not relevant to edge extraction
    cleaned = []
    for n in nodes:
        cleaned.append({
            "id": n.get("id") or n.get("node_id"),
            "label": n.get("label"),
            "node_type": n.get("node_type", "concept"),
            "is_backbone": n.get("is_backbone", False),
            "source_sentence": n.get("source_sentence", ""),
        })
    payload = {
        "source": "experiments/stability_test/slide_anchored/run01.json",
        "n_nodes": len(cleaned),
        "nodes": cleaned,
    }
    with open(CANON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Bootstrapped canonical: {len(cleaned)} nodes "
          f"(source: slide_anchored/run01.json)")
    return cleaned


# ---------------------------------------------------------------------------
# Per-run runner
# ---------------------------------------------------------------------------

def _run_once(transcript: str, nodes: list, condition: dict,
              run_idx: int) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    cfg["enrich_graph"] = False
    cfg["include_strict"] = condition["include_strict"]
    cfg["include_soft"] = condition["include_soft"]

    reset_usage()
    t0 = time.time()
    edges = extract_edges(
        transcript=transcript,
        nodes=nodes,
        knowledge_units=[],            # ← force no-KU path; strict uses transcript
        include_soft=condition["include_soft"],
        config=cfg,
    )
    elapsed = time.time() - t0

    n_strict = sum(1 for e in edges if e.get("edge_source") == "explicit")
    n_soft = sum(1 for e in edges if e.get("edge_source") == "inferred")

    usage = usage_snapshot()
    return {
        "condition": condition["name"],
        "run_index": run_idx,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "wall_time_s": round(elapsed, 2),
        "n_strict": n_strict,
        "n_soft": n_soft,
        "n_edges_total": len(edges),
        "edges": edges,
        "usage": usage,
        "config": {
            "include_strict": condition["include_strict"],
            "include_soft": condition["include_soft"],
            "knowledge_units_used": False,
            "model": cfg.get("strict_model", "gpt-5.2"),
            "strict_temperature": cfg.get("strict_temperature"),
            "soft_temperature": cfg.get("soft_temperature"),
        },
    }


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def _norm(s): return (s or "").strip().lower()


def _edges_with_labels(edges, id2lab):
    out = []
    for e in edges:
        s = e.get("from") or e.get("source") or e.get("source_id")
        t = e.get("to") or e.get("target") or e.get("target_id")
        et = e.get("edge_type") or e.get("relation_type") or "?"
        sl = id2lab.get(s, _norm(s) if isinstance(s, str) else "")
        tl = id2lab.get(t, _norm(t) if isinstance(t, str) else "")
        if sl and tl:
            out.append((sl, tl, et))
    return out


def _summarise(values):
    vs = [v for v in values if v is not None]
    if not vs: return {"n": 0}
    return {"n": len(vs),
            "mean": round(statistics.fmean(vs), 4),
            "sd": round(statistics.stdev(vs), 4) if len(vs) > 1 else 0.0,
            "min": round(min(vs), 4),
            "max": round(max(vs), 4)}


def _aggregate(out_root: str, conditions: list, canonical_nodes: list):
    id2lab = {n["id"]: _norm(n["label"]) for n in canonical_nodes if n.get("label")}

    out = {"frozen_n_nodes": len(canonical_nodes), "per_condition": {}}

    for cond in conditions:
        cond_dir = os.path.join(out_root, cond["name"])
        runs = []
        for p in sorted(glob.glob(os.path.join(cond_dir, "run*.json"))):
            if p.endswith("_FAILED.json"): continue
            runs.append(json.load(open(p, encoding="utf-8")))

        if not runs:
            out["per_condition"][cond["name"]] = {"n_runs": 0}
            continue

        edges_per_run = [_edges_with_labels(r["edges"], id2lab) for r in runs]
        n_strict = [r["n_strict"] for r in runs]
        n_soft = [r["n_soft"] for r in runs]
        walls = [r["wall_time_s"] for r in runs]
        costs = [r["usage"]["total_cost_usd"] for r in runs]

        # Pairwise edge overlap (3 modes)
        pw = {}
        for mode in ["exact", "type_agnostic_dir", "type_agnostic_undir"]:
            sets = []
            for elist in edges_per_run:
                s = set()
                for ss, tt, et in elist:
                    if mode == "exact": s.add((ss, tt, et))
                    elif mode == "type_agnostic_dir": s.add((ss, tt))
                    else: s.add(frozenset({ss, tt}))
                sets.append(s)
            js = []
            for i, j in combinations(range(len(sets)), 2):
                a, b = sets[i], sets[j]
                if not a and not b: js.append(1.0); continue
                js.append(len(a & b) / max(1, len(a | b)))
            pw[mode] = _summarise(js)

        # Components (per run)
        n_components_list = []
        for r in runs:
            adj = {n["id"]: set() for n in canonical_nodes}
            for e in r["edges"]:
                ss = e.get("from"); tt = e.get("to")
                if ss in adj and tt in adj:
                    adj[ss].add(tt); adj[tt].add(ss)
            seen = set(); comp = 0
            for nid in adj:
                if nid in seen: continue
                comp += 1
                stack = [nid]
                while stack:
                    cur = stack.pop()
                    if cur in seen: continue
                    seen.add(cur); stack.extend(adj[cur])
            n_components_list.append(comp)

        out["per_condition"][cond["name"]] = {
            "n_runs": len(runs),
            "edge_counts": {
                "strict": _summarise(n_strict),
                "soft": _summarise(n_soft),
                "total": _summarise([s + t for s, t in zip(n_strict, n_soft)]),
            },
            "n_components": _summarise(n_components_list),
            "pairwise_edge_overlap": pw,
            "wall_time_s": _summarise(walls),
            "cost_usd": _summarise(costs),
        }

    out_path = os.path.join(out_root, "metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

CONDITIONS = {
    "ablation": [
        {"name": "strict_only_no_ku", "include_strict": True,  "include_soft": False},
        {"name": "soft_only_no_ku",   "include_strict": False, "include_soft": True},
        {"name": "both_no_ku",        "include_strict": True,  "include_soft": True},
    ],
    "stability": [
        {"name": "both_no_ku",        "include_strict": True,  "include_soft": True},
    ],
}

OUT_ROOTS = {
    "ablation":  os.path.join(_PROJECT_ROOT, "experiments", "exp3_strict_vs_soft_no_ku"),
    "stability": os.path.join(_PROJECT_ROOT, "experiments", "exp_frozen_edges_no_ku"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ablation", "stability"], required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--start-from", type=int, default=1)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    transcript = open(os.path.join(LECTURE_DIR, "transcript.txt"), encoding="utf-8").read()
    canonical = _bootstrap_canonical_nodes()
    print(f"Frozen nodes: {len(canonical)}  |  Mode: {args.mode}  |  Runs/cond: {args.runs}")

    out_root = OUT_ROOTS[args.mode]
    conditions = CONDITIONS[args.mode]
    os.makedirs(out_root, exist_ok=True)

    if not args.aggregate_only:
        for cond in conditions:
            cond_dir = os.path.join(out_root, cond["name"])
            os.makedirs(cond_dir, exist_ok=True)
            print(f"\n── {cond['name']} ─────────────")
            for i in range(args.start_from, args.runs + 1):
                out_path = os.path.join(cond_dir, f"run{i:02d}.json")
                if os.path.isfile(out_path):
                    print(f"  ◦ run{i:02d} exists, skip")
                    continue
                # 3-attempt retry
                last_err = None
                for attempt in range(1, 4):
                    try:
                        result = _run_once(transcript, canonical, cond, i)
                        if result["n_edges_total"] == 0 and cond["include_strict"]:
                            raise RuntimeError("0 edges (likely API failure)")
                        with open(out_path, "w", encoding="utf-8") as f:
                            json.dump(result, f, indent=2, ensure_ascii=False)
                        u = result["usage"]
                        print(f"  ✓ run{i:02d}  strict={result['n_strict']} "
                              f"soft={result['n_soft']}  "
                              f"${u['total_cost_usd']:.3f}  {result['wall_time_s']:.0f}s")
                        last_err = None
                        break
                    except Exception as e:
                        last_err = e
                        print(f"  ⚠ run{i:02d} attempt {attempt} failed: {e}")
                        time.sleep(2 * attempt)
                if last_err is not None:
                    fail = out_path.replace(".json", "_FAILED.json")
                    with open(fail, "w") as f:
                        json.dump({"condition": cond["name"], "run_index": i,
                                   "error": str(last_err),
                                   "error_type": type(last_err).__name__,
                                   "timestamp": datetime.utcnow().isoformat() + "Z"},
                                  f, indent=2)
                    print(f"  ✗ run{i:02d} failed all retries")

    print("\nAggregating...")
    out = _aggregate(out_root, conditions, canonical)
    print(f"Saved {os.path.join(out_root, 'metrics.json')}\n")
    for cond, info in out["per_condition"].items():
        if info.get("n_runs", 0) == 0:
            print(f"[{cond}] n=0")
            continue
        e = info["edge_counts"]
        comp = info["n_components"]
        pw = info["pairwise_edge_overlap"]
        print(f"[{cond}] n={info['n_runs']}")
        print(f"  edges: strict={e['strict'].get('mean')} ± {e['strict'].get('sd')}, "
              f"soft={e['soft'].get('mean')} ± {e['soft'].get('sd')}, "
              f"total={e['total'].get('mean')}")
        print(f"  components: mean={comp.get('mean')} sd={comp.get('sd')} "
              f"range=[{comp.get('min')}, {comp.get('max')}]")
        print(f"  pairwise overlap: exact={pw['exact'].get('mean')} "
              f"undir={pw['type_agnostic_undir'].get('mean')}")
        print(f"  cost: ${info['cost_usd'].get('mean')}/run, "
              f"wall: {info['wall_time_s'].get('mean')}s/run")


if __name__ == "__main__":
    main()
