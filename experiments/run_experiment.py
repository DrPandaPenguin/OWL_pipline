import argparse
import json
import os
import sys
import time
from datetime import datetime

from experiments.config import merge_config, list_experiments, list_groups, EXPERIMENTS
from experiments.metrics import compute_all_metrics
from experiments.pipelines import get_pipeline


RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)


def _save_result(result, experiment_name, run_num):
    group = result.get("config", {}).get("group", "ungrouped")
    out_dir = os.path.join(RESULTS_DIR, group)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"{experiment_name}_run{run_num}_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return path


def run_single(experiment_name, transcript_path, run_num=1, save=True):
    config = merge_config(experiment_name)
    pipeline_fn = get_pipeline(config.get("pipeline", "multi_stage"))

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = f.read()

    result = {
        "experiment": experiment_name,
        "transcript": transcript_path,
        "run": run_num,
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }

    if config.get("slide_text") is None and config.get("slide_text_path"):
        slide_path = config["slide_text_path"]
        if os.path.isfile(slide_path):
            with open(slide_path, "r", encoding="utf-8") as f:
                config["slide_text"] = f.read()

    t0 = time.time()
    try:
        output = pipeline_fn(transcript, config)
        result["status"] = "success"
    except NotImplementedError as e:
        result.update({"status": "not_implemented", "error": str(e),
                       "wall_time": round(time.time() - t0, 2)})
        if save:
            result["saved_to"] = _save_result(result, experiment_name, run_num)
        return result
    except Exception as e:
        result.update({"status": "error", "error": str(e),
                       "wall_time": round(time.time() - t0, 2)})
        if save:
            result["saved_to"] = _save_result(result, experiment_name, run_num)
        return result

    result["wall_time"] = round(time.time() - t0, 2)
    result["timing"] = output.get("timing", {})

    nodes = output.get("nodes", [])
    edges = output.get("edges", [])
    kus = output.get("kus", [])
    result["metrics"] = compute_all_metrics(nodes, edges, kus, config)

    slide_path = config.get("slide_text_path")
    if slide_path and os.path.isfile(slide_path):
        try:
            from experiments.precision_recall import (
                parse_slide_ground_truth, compute_best_matches, compute_backbone_recall
            )
            with open(slide_path, "r", encoding="utf-8") as sf:
                gt = parse_slide_ground_truth(sf.read())
            if gt:
                all_labels = [n.get("label", "") for n in nodes if n.get("label")]
                gt_concepts = [g["concept"] for g in gt]
                for th in (0.40, 0.45, 0.50):
                    pr = compute_best_matches(all_labels, gt_concepts, th)
                    key = f"pr_t{int(th*100)}"
                    result["metrics"][f"{key}_precision"] = pr["precision"]
                    result["metrics"][f"{key}_recall"] = pr["recall"]
                    result["metrics"][f"{key}_f1"] = pr["f1"]
                bb = compute_backbone_recall(nodes, gt, 0.45)
                result["metrics"]["backbone_recall"] = bb["backbone_recall"]
                result["metrics"]["backbone_f1"] = bb["backbone_f1"]
        except Exception:
            pass

    result["outputs"] = {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "ku_count": len(kus),
        "node_ids": [n.get("id") for n in nodes],
        "edge_summary": [
            {"from": e.get("from"), "to": e.get("to"), "type": e.get("edge_type")}
            for e in edges
        ],
    }

    if save:
        result["saved_to"] = _save_result(result, experiment_name, run_num)
    return result


def run_all(experiments=None, num_runs=None, save=True):
    if experiments is None:
        experiments = list_experiments()

    out = []
    total = len(experiments)
    for idx, exp_name in enumerate(experiments, 1):
        cfg = merge_config(exp_name)
        runs = num_runs if num_runs is not None else cfg.get("num_runs", 3)
        transcripts = cfg.get("transcript_paths", [])

        print(f"\n=== [{idx}/{total}] {exp_name} ===")
        print(f"  pipeline={cfg.get('pipeline', 'multi_stage')}, runs={runs}, transcripts={len(transcripts)}")

        for t in transcripts:
            for run in range(1, runs + 1):
                print(f"  run {run}/{runs} on {os.path.basename(t)}...", end="", flush=True)
                r = run_single(exp_name, t, run, save=save)
                m = r.get("metrics", {})
                print(f" {r.get('status', '?')} "
                      f"({r.get('wall_time', 0):.1f}s, "
                      f"{m.get('node_count', '?')} nodes, "
                      f"{m.get('edge_count', '?')} edges)")
                if r.get("saved_to"):
                    print(f"    -> {r['saved_to']}")
                out.append(r)
    return out


def _print_list():
    groups = list_groups()
    print(f"\nAvailable experiments ({len(EXPERIMENTS)}):\n")
    for g in groups:
        print(f"  [{g}]")
        for exp in list_experiments(g):
            overrides = {k: v for k, v in EXPERIMENTS[exp].items() if k != "group"}
            print(f"    {exp:30s} {overrides}")
        print()
    print("Usage:")
    print("  python -m experiments.run_experiment <name>")
    print("  python -m experiments.run_experiment --group <name>")
    print("  python -m experiments.run_experiment --list")


def main():
    p = argparse.ArgumentParser(description="OWL pipeline experiment runner")
    p.add_argument("experiment", nargs="?")
    p.add_argument("--group")
    p.add_argument("--runs", type=int)
    p.add_argument("--list", action="store_true")
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args()

    if args.list:
        _print_list()
        return

    save = not args.no_save
    if args.experiment:
        experiments = [args.experiment]
    elif args.group:
        experiments = list_experiments(args.group)
        if not experiments:
            print(f"Error: no experiments in group '{args.group}'. Groups: {list_groups()}")
            sys.exit(1)
    else:
        experiments = None

    print(f"\nstarted: {datetime.now().isoformat()}")
    results = run_all(experiments=experiments, num_runs=args.runs, save=save)

    success = sum(1 for r in results if r.get("status") == "success")
    errors = sum(1 for r in results if r.get("status") == "error")
    ni = sum(1 for r in results if r.get("status") == "not_implemented")
    print(f"\ndone: {success} success, {errors} errors, {ni} not_impl")
    print(f"results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
