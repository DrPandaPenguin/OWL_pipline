#!/usr/bin/env python3
"""
Aggregate stability test runs into experiments/stability_test/metrics.json.

Reads experiments/stability_test/{condition}/runNN.json for each condition,
computes per-condition descriptive stats, between-condition tests, and
within-condition pairwise overlaps.

Output (metrics.json):
  per_condition: {
      <condition>: {
          n_runs, mean/sd/min/max/range for each metric,
          pairwise_node_overlap, pairwise_backbone_overlap,
          pairwise_edge_overlap, pairwise_explicit_overlap,
          pairwise_inferred_overlap,
          15_way_intersection: { node, backbone, edge }
      }
  }
  between: {
      welch_t_F1: {t, p, df},
      f_test_F1_variances: {F, p, df1, df2}
  }
"""
from __future__ import annotations

import glob
import json
import math
import os
import statistics
from itertools import combinations
from typing import Any

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(_PROJECT_ROOT, "experiments", "stability_test")

# Semantic matching is optional skip cleanly if no API key or module fails.
try:
    from experiments.semantic_matching import (
        semantic_pr_f1, semantic_jaccard, embed_texts,
        DEFAULT_THRESHOLDS as SEM_THRESHOLDS, DEFAULT_MODEL as SEM_MODEL,
    )
    _SEM_OK = True
except Exception:
    _SEM_OK = False
    SEM_THRESHOLDS = (0.65, 0.70, 0.75, 0.80)
    SEM_MODEL = "text-embedding-3-small"


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _summarise(values: list[float]) -> dict[str, float]:
    vs = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vs:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None, "range": None}
    mean = statistics.fmean(vs)
    sd = statistics.stdev(vs) if len(vs) > 1 else 0.0
    return {
        "n": len(vs),
        "mean": round(mean, 4),
        "sd": round(sd, 4),
        "min": round(min(vs), 4),
        "max": round(max(vs), 4),
        "range": round(max(vs) - min(vs), 4),
    }


def _welch_t(a: list[float], b: list[float]) -> dict[str, float]:
    """Welch's t-test for unequal variances. Returns t, df, p (two-sided).
    p computed via SciPy if available, else via a Student-t approximation
    that is accurate for df ≥ 5."""
    a = [x for x in a if x is not None]
    b = [x for x in b if x is not None]
    if len(a) < 2 or len(b) < 2:
        return {"t": None, "df": None, "p_two_sided": None,
                "note": "insufficient data"}
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    na, nb = len(a), len(b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return {"t": None, "df": None, "p_two_sided": None,
                "note": "zero variance"}
    t = (ma - mb) / se
    df = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    )
    p = _t_pvalue_two_sided(t, df)
    return {"t": round(t, 4), "df": round(df, 2), "p_two_sided": round(p, 5)}


def _f_test_variances(a: list[float], b: list[float]) -> dict[str, float]:
    """Two-sided F-test for equality of variances."""
    a = [x for x in a if x is not None]
    b = [x for x in b if x is not None]
    if len(a) < 2 or len(b) < 2:
        return {"F": None, "df1": None, "df2": None, "p_two_sided": None,
                "note": "insufficient data"}
    va = statistics.variance(a)
    vb = statistics.variance(b)
    if va == 0 or vb == 0:
        return {"F": None, "df1": len(a) - 1, "df2": len(b) - 1,
                "p_two_sided": None, "note": "zero variance"}
    if va >= vb:
        F = va / vb
        df1, df2 = len(a) - 1, len(b) - 1
    else:
        F = vb / va
        df1, df2 = len(b) - 1, len(a) - 1
    p = _f_pvalue_two_sided(F, df1, df2)
    return {"F": round(F, 4), "df1": df1, "df2": df2,
            "p_two_sided": round(p, 5),
            "var_a": round(va, 6), "var_b": round(vb, 6)}


def _t_pvalue_two_sided(t: float, df: float) -> float:
    """Approximate two-sided p-value for Student-t.
    Uses scipy.stats.t.sf if available, else a continuous-fraction approx."""
    try:
        from scipy import stats  # type: ignore
        return 2.0 * float(stats.t.sf(abs(t), df))
    except Exception:
 # Fallback: regularised incomplete beta via Lentz's algorithm
        x = df / (df + t * t)
        return _betainc(df / 2.0, 0.5, x)


def _f_pvalue_two_sided(F: float, df1: int, df2: int) -> float:
    try:
        from scipy import stats  # type: ignore
        p_one = float(stats.f.sf(F, df1, df2))
        return min(1.0, 2 * p_one)
    except Exception:
 # Fallback via beta-incomplete
        x = df2 / (df2 + df1 * F)
        p_one = _betainc(df2 / 2.0, df1 / 2.0, x)
        return min(1.0, 2 * p_one)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b) — minimal implementation."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
 # continued fraction
    f, c, d = 1.0, 1.0, 0.0
    for m in range(1, 200):
 # even step
        num = -(a + m - 1) * (a + b + m - 1) * x / ((a + 2 * m - 2) * (a + 2 * m - 1))
        d = 1 + num * d
        if abs(d) < 1e-300: d = 1e-300
        c = 1 + num / c
        if abs(c) < 1e-300: c = 1e-300
        d = 1 / d
        f *= d * c
 # odd step
        num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1 + num * d
        if abs(d) < 1e-300: d = 1e-300
        c = 1 + num / c
        if abs(c) < 1e-300: c = 1e-300
        d = 1 / d
        delta = d * c
        f *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return front * (f - 1)


# ---------------------------------------------------------------------------
# Overlap calculations
# ---------------------------------------------------------------------------

def _node_labels(graph: dict, backbone_only: bool = False) -> set[str]:
    out: set[str] = set()
    for n in graph.get("nodes", []):
        if backbone_only and not n.get("is_backbone"):
            continue
        lab = (n.get("label") or "").strip().lower()
        if lab:
            out.add(lab)
    return out


def _edge_keys(graph: dict, kind: str = "all") -> set[tuple]:
    """kind: 'all' / 'explicit' / 'inferred'."""
    by_id = {}
    for n in graph.get("nodes", []):
        nid = n.get("id") or n.get("node_id")
        lab = (n.get("label") or "").strip().lower()
        if nid and lab:
            by_id[nid] = lab

    def _is_explicit(e):
        src = e.get("edge_source")
        if src == "explicit":
            return True
        if src == "inferred":
            return False
        return "confidence_score" not in e

    out: set[tuple] = set()
    for e in graph.get("edges", []):
        if kind == "explicit" and not _is_explicit(e):
            continue
        if kind == "inferred" and _is_explicit(e):
            continue
        s = e.get("from") or e.get("source") or e.get("source_id")
        t = e.get("to") or e.get("target") or e.get("target_id")
        et = e.get("edge_type") or e.get("relation_type") or e.get("relation") or "?"
        s_lab = by_id.get(s, s)
        t_lab = by_id.get(t, t)
        out.add((s_lab, t_lab, et))
    return out


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def _pairwise(graphs: list[dict], extractor) -> dict[str, float]:
    sets = [extractor(g) for g in graphs]
    js = []
    for i, j in combinations(range(len(sets)), 2):
        js.append(_jaccard(sets[i], sets[j]))
    if not js:
        return {"n_pairs": 0, "mean": None, "sd": None,
                "min": None, "max": None}
    return {
        "n_pairs": len(js),
        "mean": round(statistics.fmean(js), 4),
        "sd": round(statistics.stdev(js), 4) if len(js) > 1 else 0.0,
        "min": round(min(js), 4),
        "max": round(max(js), 4),
    }


def _full_intersection(graphs: list[dict], extractor) -> dict[str, Any]:
    if not graphs:
        return {"size": 0, "items": []}
    inter = extractor(graphs[0])
    for g in graphs[1:]:
        inter = inter & extractor(g)
    return {"size": len(inter), "items": sorted(inter)[:50]}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

METRICS_FIELDS = [
    "n_nodes", "n_edges", "n_components", "n_orphans", "orphan_pct",
    "p_at_45", "r_at_45", "f1_at_45", "backbone_recall", "backbone_f1",
    "edge_type_diversity",
]


def _load_runs(condition: str) -> list[dict]:
    pattern = os.path.join(ROOT, condition, "run*.json")
    out = []
    for path in sorted(glob.glob(pattern)):
        if path.endswith("_FAILED.json"):
            continue
        with open(path, encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def _semantic_per_run(graph: dict, slide_text: str) -> dict:
    """Compute semantic P/R/F1 at each threshold for a single graph."""
    from experiments.precision_recall import parse_slide_ground_truth
    gt = parse_slide_ground_truth(slide_text)
    gt_concepts = [g["concept"] for g in gt]
    labels = [n.get("label", "") for n in graph.get("nodes", []) if n.get("label")]
    return semantic_pr_f1(labels, gt_concepts, thresholds=SEM_THRESHOLDS, model=SEM_MODEL)


def _pairwise_semantic(graphs: list[dict], threshold: float, backbone_only: bool = False) -> dict:
    """Pairwise semantic Jaccard across all run pairs at one threshold."""
    label_sets = []
    for g in graphs:
        labs = []
        for n in g.get("nodes", []):
            if backbone_only and not n.get("is_backbone"):
                continue
            lab = (n.get("label") or "").strip()
            if lab:
                labs.append(lab)
        label_sets.append(labs)
    js = []
    for i, j in combinations(range(len(label_sets)), 2):
        js.append(semantic_jaccard(label_sets[i], label_sets[j], threshold, model=SEM_MODEL))
    if not js:
        return {"n_pairs": 0, "mean": None, "sd": None}
    return {
        "n_pairs": len(js),
        "mean": round(statistics.fmean(js), 4),
        "sd": round(statistics.stdev(js), 4) if len(js) > 1 else 0.0,
        "min": round(min(js), 4),
        "max": round(max(js), 4),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-semantic", action="store_true",
                    help="Skip semantic matching (faster, but loses semantic metrics).")
    args = ap.parse_args()
    do_semantic = _SEM_OK and not args.no_semantic

    conditions = [d for d in os.listdir(ROOT)
                  if os.path.isdir(os.path.join(ROOT, d)) and not d.startswith(".")]
    conditions.sort()

 # Need slide_text once for semantic eval against GT
    slide_path = os.path.join(_PROJECT_ROOT, "input", "ecommerce", "lec4", "slides.txt")
    slide_text = open(slide_path, encoding="utf-8").read() if do_semantic else ""

    per_condition: dict[str, Any] = {}

    f1_by_cond: dict[str, list[float]] = {}
    sem_f1_by_cond: dict[str, dict[str, list[float]]] = {}  # cond -> threshold -> list

    for cond in conditions:
        runs = _load_runs(cond)
        if not runs:
            continue
        graphs = [r["graph"] for r in runs]
        metrics_lists = {fld: [r["metrics"].get(fld) for r in runs]
                         for fld in METRICS_FIELDS}

        per_run_summary = {fld: _summarise(vals) for fld, vals in metrics_lists.items()}
        f1_by_cond[cond] = [v for v in metrics_lists["f1_at_45"] if v is not None]

 # Pairwise overlaps
        pw = {
            "node_overlap":     _pairwise(graphs, lambda g: _node_labels(g)),
            "backbone_overlap": _pairwise(graphs, lambda g: _node_labels(g, backbone_only=True)),
            "edge_overlap":     _pairwise(graphs, lambda g: _edge_keys(g, "all")),
            "explicit_overlap": _pairwise(graphs, lambda g: _edge_keys(g, "explicit")),
            "inferred_overlap": _pairwise(graphs, lambda g: _edge_keys(g, "inferred")),
        }

 # N-way intersection (stable core)
        nwi = {
            "node":     _full_intersection(graphs, lambda g: _node_labels(g)),
            "backbone": _full_intersection(graphs, lambda g: _node_labels(g, backbone_only=True)),
            "edge":     _full_intersection(graphs, lambda g: _edge_keys(g, "all")),
        }

 # Cost/time
        runtimes = [r.get("runtime", {}) for r in runs]
        wall_summary = _summarise([r.get("wall_time_s") for r in runs])
        cost_summary = _summarise([rt.get("cost_usd") for rt in runtimes])
        token_in_summary = _summarise([rt.get("input_tokens") for rt in runtimes])
        token_out_summary = _summarise([rt.get("output_tokens") for rt in runtimes])

 # ---- Semantic matching (optional) ----
        sem_block: dict[str, Any] = {}
        if do_semantic:
 # 1) Per-run semantic P/R/F1 against slide GT, summarised
            sem_per_threshold: dict[str, dict] = {f"{t:.2f}": {"p": [], "r": [], "f1": []}
                                                  for t in SEM_THRESHOLDS}
            for g in graphs:
                pr = _semantic_per_run(g, slide_text)
                for thr_key, vals in pr.get("by_threshold", {}).items():
                    sem_per_threshold[thr_key]["p"].append(vals["precision"])
                    sem_per_threshold[thr_key]["r"].append(vals["recall"])
                    sem_per_threshold[thr_key]["f1"].append(vals["f1"])

            sem_block["semantic_vs_gt"] = {
                thr_key: {
                    "precision": _summarise(d["p"]),
                    "recall":    _summarise(d["r"]),
                    "f1":        _summarise(d["f1"]),
                }
                for thr_key, d in sem_per_threshold.items()
            }
            sem_f1_by_cond[cond] = {
                thr_key: d["f1"] for thr_key, d in sem_per_threshold.items()
            }

 # 2) Pairwise semantic Jaccard at each threshold
            sem_block["pairwise_semantic_node_overlap"] = {
                f"{t:.2f}": _pairwise_semantic(graphs, t, backbone_only=False)
                for t in SEM_THRESHOLDS
            }
            sem_block["pairwise_semantic_backbone_overlap"] = {
                f"{t:.2f}": _pairwise_semantic(graphs, t, backbone_only=True)
                for t in SEM_THRESHOLDS
            }

        per_condition[cond] = {
            "n_runs": len(runs),
            "metrics": per_run_summary,
            "pairwise": pw,
            "n_way_intersection": nwi,
            "wall_time_s": wall_summary,
            "cost_usd": cost_summary,
            "tokens_in": token_in_summary,
            "tokens_out": token_out_summary,
            **({"semantic": sem_block} if do_semantic else {}),
        }

 # Between-condition tests (F1)
    between = {}
    if len(f1_by_cond) >= 2:
        names = list(f1_by_cond.keys())
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                key = f"{a}_vs_{b}"
                between[key] = {
                    "welch_t_F1": _welch_t(f1_by_cond[a], f1_by_cond[b]),
                    "f_test_F1_variances": _f_test_variances(f1_by_cond[a], f1_by_cond[b]),
                    "mean_F1_a": round(statistics.fmean(f1_by_cond[a]), 4) if f1_by_cond[a] else None,
                    "mean_F1_b": round(statistics.fmean(f1_by_cond[b]), 4) if f1_by_cond[b] else None,
                    "sd_F1_a": round(statistics.stdev(f1_by_cond[a]), 4) if len(f1_by_cond[a]) > 1 else None,
                    "sd_F1_b": round(statistics.stdev(f1_by_cond[b]), 4) if len(f1_by_cond[b]) > 1 else None,
                }

 # Between-condition semantic F1 tests (one per threshold)
    between_semantic = {}
    if do_semantic and len(sem_f1_by_cond) >= 2:
        names = list(sem_f1_by_cond.keys())
        for thr_key in [f"{t:.2f}" for t in SEM_THRESHOLDS]:
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    key = f"{a}_vs_{b}__sem{thr_key}"
                    fa = sem_f1_by_cond[a][thr_key]
                    fb = sem_f1_by_cond[b][thr_key]
                    between_semantic[key] = {
                        "welch_t_F1": _welch_t(fa, fb),
                        "f_test_F1_variances": _f_test_variances(fa, fb),
                        "mean_F1_a": round(statistics.fmean(fa), 4) if fa else None,
                        "mean_F1_b": round(statistics.fmean(fb), 4) if fb else None,
                    }

 # Lexical-vs-semantic agreement: for each run, do the two metrics agree
 # on which GT concepts are matched?
    agreement: dict[str, Any] = {}
    if do_semantic:
        for cond in conditions:
            runs = _load_runs(cond)
            if not runs:
                continue
            agree_runs = []
            for r in runs:
                lex_matched = set(r["metrics"].get("matched_concepts", []))
 # Recompute semantic match at the conventional 0.75 threshold
                sem = _semantic_per_run(r["graph"], slide_text)
                sem_matched = set(sem["by_threshold"].get("0.75", {}).get("matched_gt", []))
                if not (lex_matched | sem_matched):
                    agree_runs.append(1.0)
                    continue
                agree_runs.append(
                    len(lex_matched & sem_matched) / max(1, len(lex_matched | sem_matched))
                )
            agreement[cond] = _summarise(agree_runs)

    payload = {
        "lecture": "ecommerce/lec4",
        "per_condition": per_condition,
        "between": between,
        "between_semantic": between_semantic,
        "lex_sem_agreement_at_0.75": agreement,
        "semantic_model": SEM_MODEL if do_semantic else None,
        "semantic_thresholds": list(SEM_THRESHOLDS) if do_semantic else None,
    }

    out_path = os.path.join(ROOT, "metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"wrote {out_path}")
 # Print a compact summary
    for cond, info in per_condition.items():
        print(f"\n[{cond}] n={info['n_runs']}")
        for fld in ("n_nodes", "f1_at_45", "n_components", "backbone_recall"):
            s = info["metrics"][fld]
            print(f"  {fld:18s} mean={s['mean']}  sd={s['sd']}  range=[{s['min']}, {s['max']}]")
        for k, v in info["pairwise"].items():
            print(f"  pw_{k:18s} mean={v['mean']}  sd={v['sd']}  (n={v['n_pairs']})")
    if between:
        print("\n[between]")
        for k, v in between.items():
            print(f"  {k}: F1 mean {v['mean_F1_a']} vs {v['mean_F1_b']}  "
                  f"Welch t={v['welch_t_F1']['t']} (p={v['welch_t_F1']['p_two_sided']});  "
                  f"F-var={v['f_test_F1_variances']['F']} (p={v['f_test_F1_variances']['p_two_sided']})")


if __name__ == "__main__":
    main()
