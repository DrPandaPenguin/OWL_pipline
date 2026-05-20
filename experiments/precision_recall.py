import json
import re
import sys
from difflib import SequenceMatcher


def parse_slide_ground_truth(slide_text):
    """Pull Key Ideas bullets out of a slide structure file."""
    gt = []
    current_part = 0
    current_title = ""
    in_key_ideas = False

    for line in slide_text.split("\n"):
        s = line.strip()

        m = re.match(r"PART\s+(\d+)\s*[—–-]\s*(.+)", s)
        if m:
            current_part = int(m.group(1))
            current_title = m.group(2).strip()
            in_key_ideas = False
            continue

        if s == "Key Ideas":
            in_key_ideas = True
            continue

        if s in ("Core Focus", "Why This Part Exists", "Visual Notes",
                 "Structural Flow", "Overall Theme", "Lecture Title") or s.startswith("PART "):
            if s.startswith("PART "):
                m2 = re.match(r"PART\s+(\d+)\s*[—–-]\s*(.+)", s)
                if m2:
                    current_part = int(m2.group(1))
                    current_title = m2.group(2).strip()
            in_key_ideas = False
            continue

        if s in ("⸻", "---"):
            in_key_ideas = False
            continue

        if in_key_ideas and s:
            concept = re.sub(r"^[•\-\*\t\s]+", "", s).strip()
            if concept and len(concept) > 5:
                gt.append({
                    "part": current_part,
                    "part_title": current_title,
                    "concept": concept,
                })
    return gt


def fuzzy_similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def compute_best_matches(predicted_labels, ground_truth_concepts, threshold=0.45):
    pred_norm = [p.lower().strip() for p in predicted_labels]
    gt_norm = [g.lower().strip() for g in ground_truth_concepts]

    matched_pred = []
    unmatched_pred = []
    for i, p in enumerate(pred_norm):
        best_score = 0.0
        best_j = -1
        for j, g in enumerate(gt_norm):
            sc = fuzzy_similarity(p, g)
            if sc > best_score:
                best_score = sc
                best_j = j
        if best_score >= threshold:
            matched_pred.append((predicted_labels[i], ground_truth_concepts[best_j], round(best_score, 3)))
        else:
            unmatched_pred.append((predicted_labels[i], round(best_score, 3)))

    matched_gt = []
    unmatched_gt = []
    for j, g in enumerate(gt_norm):
        best_score = 0.0
        best_i = -1
        for i, p in enumerate(pred_norm):
            sc = fuzzy_similarity(p, g)
            if sc > best_score:
                best_score = sc
                best_i = i
        if best_score >= threshold:
            matched_gt.append((ground_truth_concepts[j], predicted_labels[best_i], round(best_score, 3)))
        else:
            unmatched_gt.append((ground_truth_concepts[j], round(best_score, 3)))

    precision = len(matched_pred) / len(predicted_labels) if predicted_labels else 0.0
    recall = len(matched_gt) / len(ground_truth_concepts) if ground_truth_concepts else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "threshold": threshold,
        "num_predicted": len(predicted_labels),
        "num_ground_truth": len(ground_truth_concepts),
        "num_matched_pred": len(matched_pred),
        "num_matched_gt": len(matched_gt),
        "matched_pairs": matched_pred,
        "unmatched_predicted": unmatched_pred,
        "unmatched_gt": unmatched_gt,
    }


def compute_backbone_recall(predicted_nodes, ground_truth, threshold=0.45):
    backbone_labels = [n.get("label", "") for n in predicted_nodes if n.get("is_backbone")]
    parts = sorted({gt["part"] for gt in ground_truth})

    per_part = {}
    for pn in parts:
        part_gt = [gt["concept"] for gt in ground_truth if gt["part"] == pn]
        r = compute_best_matches(backbone_labels, part_gt, threshold)
        per_part[f"part_{pn}"] = {
            "recall": r["recall"],
            "matched": r["num_matched_gt"],
            "total_gt": len(part_gt),
        }

    overall = compute_best_matches(
        backbone_labels, [gt["concept"] for gt in ground_truth], threshold
    )
    return {
        "backbone_recall": overall["recall"],
        "backbone_precision": overall["precision"],
        "backbone_f1": overall["f1"],
        "num_backbone_nodes": len(backbone_labels),
        "per_part": per_part,
    }


def evaluate_pipeline_result(result_path, slide_path, threshold=0.45):
    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "outputs" in data and "node_ids" in data.get("outputs", {}):
        return {"error": "Experiment result format doesn't contain full node labels. Use raw KG JSON."}

    nodes = data.get("nodes", [])
    if not nodes:
        return {"error": f"No nodes found in {result_path}"}

    with open(slide_path, "r", encoding="utf-8") as f:
        slide_text = f.read()

    gt = parse_slide_ground_truth(slide_text)
    if not gt:
        return {"error": f"No ground truth concepts extracted from {slide_path}"}

    all_labels = [n.get("label", "") for n in nodes if n.get("label")]
    gt_concepts = [g["concept"] for g in gt]

    return {
        "source": result_path,
        "slide_source": slide_path,
        "all_nodes": compute_best_matches(all_labels, gt_concepts, threshold),
        "backbone": compute_backbone_recall(nodes, gt, threshold),
        "ground_truth_count": len(gt),
        "ground_truth_parts": len({g["part"] for g in gt}),
    }


def main():
    import argparse

    p = argparse.ArgumentParser(description="Precision/Recall vs slide ground truth")
    p.add_argument("result_path", nargs="?", default="data/demo_kg.json")
    p.add_argument("--slides", default="data/Test_Slide_strcuture_lec4_Ecoms")
    p.add_argument("--threshold", type=float, default=0.45)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    result = evaluate_pipeline_result(args.result_path, args.slides, args.threshold)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if "error" in result:
        print(f"Error: {result['error']}")
        sys.exit(1)

    an = result["all_nodes"]
    bb = result["backbone"]

    print(f"\n{'='*60}")
    print(f"  Precision / Recall")
    print(f"  Source: {result['source']}")
    print(f"  GT: {result['ground_truth_count']} concepts / {result['ground_truth_parts']} PARTs")
    print(f"  Threshold: {args.threshold}")
    print(f"{'='*60}")

    print(f"\n  ALL NODES ({an['num_predicted']} pred vs {an['num_ground_truth']} GT)")
    print(f"    Precision: {an['precision']:.1%}  ({an['num_matched_pred']}/{an['num_predicted']})")
    print(f"    Recall:    {an['recall']:.1%}  ({an['num_matched_gt']}/{an['num_ground_truth']})")
    print(f"    F1:        {an['f1']:.1%}")

    print(f"\n  BACKBONE ({bb['num_backbone_nodes']})")
    print(f"    Precision: {bb['backbone_precision']:.1%}")
    print(f"    Recall:    {bb['backbone_recall']:.1%}")
    print(f"    F1:        {bb['backbone_f1']:.1%}")

    print(f"\n  Per-PART Backbone Recall:")
    for k in sorted(bb["per_part"].keys()):
        v = bb["per_part"][k]
        print(f"    {k}: {v['recall']:.0%} ({v['matched']}/{v['total_gt']})")

    if an["unmatched_gt"]:
        print(f"\n  MISSED GT ({len(an['unmatched_gt'])}):")
        for concept, sc in an["unmatched_gt"][:10]:
            print(f"    - {concept[:70]}  (best: {sc:.2f})")

    if an["unmatched_predicted"]:
        print(f"\n  EXTRA predicted ({len(an['unmatched_predicted'])}):")
        for lab, sc in an["unmatched_predicted"][:10]:
            print(f"    + {lab[:70]}  (best: {sc:.2f})")

    print()


if __name__ == "__main__":
    main()
