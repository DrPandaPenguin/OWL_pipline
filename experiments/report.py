import argparse
import json
import math
import os
from collections import defaultdict


RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)

_KEY_METRICS = [
    "node_count", "edge_count", "ku_count",
    "strict_count", "soft_count",
    "valid_edge_ratio", "hallucination_count",
    "orphan_node_ratio", "orphan_ku_ratio",
    "num_connected_components", "graph_density",
    "avg_soft_confidence", "avg_kus_per_node", "self_loop_count",
]


def load_results(group=None):
    out = defaultdict(list)
    if not os.path.isdir(RESULTS_DIR):
        return dict(out)

    for dirpath, _, filenames in os.walk(RESULTS_DIR):
        for fn in sorted(filenames):
            if not fn.endswith(".json"):
                continue
            try:
                with open(os.path.join(dirpath, fn), "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if data.get("status") != "success":
                continue
            exp_group = data.get("config", {}).get("group", "ungrouped")
            if group and exp_group != group:
                continue
            out[data.get("experiment", "unknown")].append(data)

    return dict(out)


def _mean_std(values):
    if not values:
        return "-"
    m = sum(values) / len(values)
    if len(values) == 1:
        return f"{m:.2f}"
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return f"{m:.2f} ± {math.sqrt(var):.2f}"


def generate_group_table(group_name, results, metrics=None):
    metrics = metrics or _KEY_METRICS
    group_exps = {
        name: runs for name, runs in results.items()
        if runs and runs[0].get("config", {}).get("group") == group_name
    }
    if not group_exps:
        return f"### {group_name}\n\nNo results found.\n"

    exp_names = sorted(group_exps.keys())
    lines = [f"### {group_name}\n"]
    lines.append("| Metric | " + " | ".join(exp_names) + " |")
    lines.append("|---|" + "|".join(["---"] * len(exp_names)) + "|")

    for metric in metrics:
        row = f"| {metric} |"
        for exp in exp_names:
            vals = []
            for r in group_exps[exp]:
                v = r.get("metrics", {}).get(metric)
                if v is not None:
                    try:
                        vals.append(float(v))
                    except (TypeError, ValueError):
                        pass
            row += f" {_mean_std(vals)} |"
        lines.append(row)

    row = "| wall_time (s) |"
    for exp in exp_names:
        row += f" {_mean_std([r.get('wall_time', 0) for r in group_exps[exp]])} |"
    lines.append(row)

    lines.append("")
    return "\n".join(lines)


def generate_full_report(results=None, group=None):
    if results is None:
        results = load_results(group=group)

    if not results:
        return ("# Experiment Report\n\nNo results found. Run experiments first:\n"
                "```\npython -m experiments.run_experiment\n```\n")

    groups = defaultdict(set)
    for exp_name, runs in results.items():
        if runs:
            g = runs[0].get("config", {}).get("group", "ungrouped")
            groups[g].add(exp_name)

    lines = ["# OWL Pipeline Experiment Report\n"]
    total_runs = sum(len(r) for r in results.values())
    lines.append(f"**{len(results)} experiments, {total_runs} runs, {len(groups)} groups**\n")

    for g in sorted(groups.keys()):
        lines.append(generate_group_table(g, results))

    lines.append("### Cross-Experiment Summary\n")
    lines.append("| Experiment | Pipeline | Nodes | Edges | Valid% | Wall(s) |")
    lines.append("|---|---|---|---|---|---|")

    for exp_name in sorted(results.keys()):
        runs = results[exp_name]
        if not runs:
            continue
        pipeline = runs[0].get("config", {}).get("pipeline", "?")
        nodes = _mean_std([r.get("metrics", {}).get("node_count", 0) for r in runs])
        edges = _mean_std([r.get("metrics", {}).get("edge_count", 0) for r in runs])
        valid = _mean_std([r.get("metrics", {}).get("valid_edge_ratio", 0) * 100 for r in runs])
        wall = _mean_std([r.get("wall_time", 0) for r in runs])
        lines.append(f"| {exp_name} | {pipeline} | {nodes} | {edges} | {valid} | {wall} |")

    lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="OWL experiment report generator")
    p.add_argument("--output", "-o")
    p.add_argument("--group")
    args = p.parse_args()

    report = generate_full_report(group=args.group)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Saved: {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
