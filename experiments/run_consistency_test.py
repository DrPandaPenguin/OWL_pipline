#!/usr/bin/env python3
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass

from experiments.pipelines import pipeline_slide_anchored


def load_inputs():
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    with open(os.path.join(base, "Test_transcript_lec4_Ecom"), "r") as f:
        transcript = f.read()
    with open(os.path.join(base, "Test_Slide_strcuture_lec4_Ecoms"), "r") as f:
        slide = f.read()
    return transcript, slide


def run_once(transcript, slide_text, run_id):
    print(f"\n=== RUN {run_id} ===")
    t0 = time.time()
    cfg = {
        "slide_text": slide_text,
        "node_model": "gpt-4.1",
        "soft_model": "gpt-4.1",
        "node_temperature": 0.1,
        "enrich_graph": True,
    }
    res = pipeline_slide_anchored(transcript, cfg)
    print(f"  {time.time() - t0:.1f}s, nodes={len(res.get('nodes', []))}, edges={len(res.get('edges', []))}")
    return res


def _edge_key(e, nodes):
    nmap = {n["id"]: n["label"].lower().strip() for n in nodes}
    return (
        nmap.get(e.get("source", e.get("from", "")), "?"),
        nmap.get(e.get("target", e.get("to", "")), "?"),
        e.get("relation", ""),
    )


def _overlap_pct(a, b):
    return len(a & b) / max(len(a | b), 1) * 100


def compare(r1, r2):
    print("\n=== CONSISTENCY ===")

    n1 = {n["label"].lower().strip() for n in r1.get("nodes", [])}
    n2 = {n["label"].lower().strip() for n in r2.get("nodes", [])}
    print(f"\nNodes  R1={len(n1)} R2={len(n2)} overlap={_overlap_pct(n1, n2):.0f}%")
    only1, only2 = n1 - n2, n2 - n1
    if only1:
        print(f"  Only R1 ({len(only1)}):")
        for l in sorted(only1)[:10]:
            print(f"    - {l}")
    if only2:
        print(f"  Only R2 ({len(only2)}):")
        for l in sorted(only2)[:10]:
            print(f"    - {l}")

    bb1 = {n["label"].lower().strip() for n in r1.get("nodes", []) if n.get("is_backbone")}
    bb2 = {n["label"].lower().strip() for n in r2.get("nodes", []) if n.get("is_backbone")}
    print(f"\nBackbone  R1={len(bb1)} R2={len(bb2)} overlap={_overlap_pct(bb1, bb2):.0f}%")

    e1 = {_edge_key(e, r1["nodes"]) for e in r1.get("edges", [])}
    e2 = {_edge_key(e, r2["nodes"]) for e in r2.get("edges", [])}
    print(f"\nEdges  R1={len(e1)} R2={len(e2)} overlap={_overlap_pct(e1, e2):.0f}%")

    ex1 = {_edge_key(e, r1["nodes"]) for e in r1.get("edges", []) if e.get("edge_source") == "explicit"}
    ex2 = {_edge_key(e, r2["nodes"]) for e in r2.get("edges", []) if e.get("edge_source") == "explicit"}
    print(f"\nExplicit  R1={len(ex1)} R2={len(ex2)} overlap={_overlap_pct(ex1, ex2):.0f}%")

    in1 = {_edge_key(e, r1["nodes"]) for e in r1.get("edges", []) if e.get("edge_source") == "inferred"}
    in2 = {_edge_key(e, r2["nodes"]) for e in r2.get("edges", []) if e.get("edge_source") == "inferred"}
    print(f"\nInferred  R1={len(in1)} R2={len(in2)} overlap={_overlap_pct(in1, in2):.0f}%")

    return {
        "node_overlap_pct": _overlap_pct(n1, n2),
        "backbone_overlap_pct": _overlap_pct(bb1, bb2),
        "edge_overlap_pct": _overlap_pct(e1, e2),
        "explicit_edge_overlap_pct": _overlap_pct(ex1, ex2),
        "inferred_edge_overlap_pct": _overlap_pct(in1, in2),
    }


def main():
    transcript, slide_text = load_inputs()
    print(f"Transcript: {len(transcript)} chars")
    print(f"Slides: {len(slide_text)} chars")

    r1 = run_once(transcript, slide_text, 1)
    r2 = run_once(transcript, slide_text, 2)
    metrics = compare(r1, r2)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "consistency_test")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "run1.json"), "w") as f:
        json.dump(r1, f, indent=2)
    with open(os.path.join(out, "run2.json"), "w") as f:
        json.dump(r2, f, indent=2)
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved to {out}/")
    print("\n=== SUMMARY ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.1f}%")


if __name__ == "__main__":
    main()
