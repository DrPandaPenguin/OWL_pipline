#!/usr/bin/env python3
"""
Semantic matching evaluator — embedding-based counterpart to the
SequenceMatcher lexical metric in `experiments/precision_recall.py`.

Used to validate the lexical proxy: if lexical and semantic matching
agree on most pairs, the cheaper SequenceMatcher metric is defensible.
Where they disagree, the semantic metric reveals the cases where
paraphrased-but-equivalent concepts are missed by SequenceMatcher.

Typical use:
    from experiments.semantic_matching import (
        embed_texts, semantic_pr_f1, semantic_jaccard,
    )

    p_r_f1 = semantic_pr_f1(node_labels, gt_concepts, thresholds=[0.65, 0.70, 0.75, 0.80])

Embeddings are cached to disk under experiments/embedding_cache/<sha>.json
so reruns on the same labels are free.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Iterable

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE_DIR = os.path.join(_PROJECT_ROOT, "experiments", "embedding_cache")

DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_THRESHOLDS = (0.65, 0.70, 0.75, 0.80)


# ---------------------------------------------------------------------------
# Embedding (with on-disk cache)
# ---------------------------------------------------------------------------

def _cache_key(text: str, model: str) -> str:
    return hashlib.sha256(f"{model}::{text}".encode("utf-8")).hexdigest()[:32]


def _load_cache(model: str) -> dict:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = os.path.join(_CACHE_DIR, f"{model}.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(cache: dict, model: str) -> None:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = os.path.join(_CACHE_DIR, f"{model}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f)


def embed_texts(texts: list[str], model: str = DEFAULT_MODEL) -> list[list[float]]:
    """Embed a list of strings with caching. Empty strings get a zero vector
    of the right shape (computed from the first non-empty embedding)."""
    if not texts:
        return []
    cache = _load_cache(model)

    needed = [t for t in texts if t and _cache_key(t, model) not in cache]
    if needed:
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set; cannot embed")
        client = OpenAI(api_key=api_key)

 # Batch embeddings (OpenAI supports up to 2048 inputs per call)
        BATCH = 256
        for i in range(0, len(needed), BATCH):
            batch = needed[i:i + BATCH]
            resp = client.embeddings.create(model=model, input=batch)
            for txt, item in zip(batch, resp.data):
                cache[_cache_key(txt, model)] = item.embedding
        _save_cache(cache, model)

    out: list[list[float]] = []
    zero_dim = None
    for t in texts:
        if not t:
            out.append(None)  # type: ignore
            continue
        vec = cache[_cache_key(t, model)]
        zero_dim = len(vec)
        out.append(vec)
    if zero_dim is not None:
        out = [(v if v is not None else [0.0] * zero_dim) for v in out]
    return out


# ---------------------------------------------------------------------------
# Cosine + match logic
# ---------------------------------------------------------------------------

def _cos(a: list[float], b: list[float]) -> float:
    s = 0.0; aa = 0.0; bb = 0.0
    for x, y in zip(a, b):
        s += x * y
        aa += x * x
        bb += y * y
    if aa == 0 or bb == 0:
        return 0.0
    return s / (aa ** 0.5 * bb ** 0.5)


def _sim_matrix(rows: list[list[float]], cols: list[list[float]]) -> list[list[float]]:
    return [[_cos(r, c) for c in cols] for r in rows]


def semantic_pr_f1(
    pred_labels: list[str],
    gt_concepts: list[str],
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    model: str = DEFAULT_MODEL,
) -> dict:
    """
    Greedy 1-1 matching between predicted labels and GT concepts.
    For each threshold, returns precision/recall/F1 plus matched/unmatched
    GT lists for downstream analysis.
    """
    pred_labels = [str(l).strip() for l in pred_labels if l and str(l).strip()]
    gt_concepts = [str(g).strip() for g in gt_concepts if g and str(g).strip()]

    if not pred_labels or not gt_concepts:
        return {
            "model": model,
            "n_pred": len(pred_labels),
            "n_gt": len(gt_concepts),
            "by_threshold": {},
            "note": "empty inputs",
        }

    pred_vecs = embed_texts(pred_labels, model)
    gt_vecs = embed_texts(gt_concepts, model)
    sim = _sim_matrix(pred_vecs, gt_vecs)

    out: dict = {
        "model": model,
        "n_pred": len(pred_labels),
        "n_gt": len(gt_concepts),
        "by_threshold": {},
    }
    for thr in thresholds:
 # Greedy: sort all (i, j, sim) descending, take match if both unused
        pairs = sorted(
            ((sim[i][j], i, j) for i in range(len(pred_labels))
             for j in range(len(gt_concepts))),
            key=lambda t: -t[0],
        )
        used_pred: set[int] = set()
        used_gt: set[int] = set()
        matched_gt_idx: list[int] = []
        for s, i, j in pairs:
            if s < thr:
                break
            if i in used_pred or j in used_gt:
                continue
            used_pred.add(i)
            used_gt.add(j)
            matched_gt_idx.append(j)

        tp = len(matched_gt_idx)
        precision = tp / max(1, len(pred_labels))
        recall = tp / max(1, len(gt_concepts))
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        out["by_threshold"][f"{thr:.2f}"] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "true_positives": tp,
            "matched_gt": [gt_concepts[j] for j in sorted(matched_gt_idx)],
            "unmatched_gt": [g for k, g in enumerate(gt_concepts) if k not in used_gt],
        }
    return out


def semantic_jaccard(
    set_a_labels: list[str],
    set_b_labels: list[str],
    threshold: float,
    model: str = DEFAULT_MODEL,
) -> float:
    """
    Semantic Jaccard between two label sets.
    Two labels are "the same" if cosine ≥ threshold.
    Performs greedy 1-1 matching, then |intersection| / |union|.
    """
    a = [str(l).strip() for l in set_a_labels if l and str(l).strip()]
    b = [str(l).strip() for l in set_b_labels if l and str(l).strip()]
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    av = embed_texts(a, model)
    bv = embed_texts(b, model)
    sim = _sim_matrix(av, bv)
    pairs = sorted(
        ((sim[i][j], i, j) for i in range(len(a)) for j in range(len(b))),
        key=lambda t: -t[0],
    )
    used_a: set[int] = set(); used_b: set[int] = set()
    matched = 0
    for s, i, j in pairs:
        if s < threshold:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i); used_b.add(j); matched += 1
    union = len(a) + len(b) - matched
    return matched / max(1, union)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pred = ["Lamport Clocks", "Logical Time", "Vector Clocks"]
    gt = ["Lamport timestamps", "Logical clocks for ordering events",
          "Vector clock algorithm", "Causal consistency"]
    print(json.dumps(semantic_pr_f1(pred, gt), indent=2))
    print("Jaccard same-set:", semantic_jaccard(pred, pred, 0.75))
    print("Jaccard pred-vs-gt:", semantic_jaccard(pred, gt, 0.75))
