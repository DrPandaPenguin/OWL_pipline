#!/usr/bin/env python3
# CLI 진입점: input/<course>/<lecture>/ 폴더의 transcript(+slides)을 읽어
# 파이프라인 돌려서 graphs/<course>_<lecture>.json 으로 저장.
# viewer (dash_app)에서 ?graph=<name> 쿼리로 바로 띄움.
import argparse
import json
import os
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)

from src.pipelines import get_pipeline, list_pipelines, DEFAULT_CONFIG
from src.usage_tracker import install as install_usage_tracker, reset as reset_usage, snapshot as usage_snapshot, report as usage_report

install_usage_tracker()

INPUT_DIR = os.path.join(_PROJECT_ROOT, "input")
GRAPHS_DIR = os.path.join(_PROJECT_ROOT, "graphs")
OUTPUTS_DIR = os.path.join(_PROJECT_ROOT, "outputs")


def _find_file(directory, prefix):
    if not os.path.isdir(directory):
        return None
    for fname in os.listdir(directory):
        if os.path.splitext(fname.lower())[0] == prefix.lower():
            return os.path.join(directory, fname)
    return None


def list_inputs():
    if not os.path.isdir(INPUT_DIR):
        print("No input/ directory found.")
        return
    print(f"\nAvailable lectures:\n")
    for course in sorted(os.listdir(INPUT_DIR)):
        course_dir = os.path.join(INPUT_DIR, course)
        if not os.path.isdir(course_dir) or course.startswith("."):
            continue
        for lec in sorted(os.listdir(course_dir)):
            lec_dir = os.path.join(course_dir, lec)
            if not os.path.isdir(lec_dir) or lec.startswith("."):
                continue
            has_transcript = _find_file(lec_dir, "transcript") is not None
            has_slides = _find_file(lec_dir, "slides") is not None
            parts = []
            if has_transcript:
                parts.append("transcript")
            if has_slides:
                parts.append("slides")
            status = "ready" if has_transcript else "no transcript!"
            path = f"input/{course}/{lec}"
            print(f"  {path:<35} [{', '.join(parts)}] {status}")
    print()


# 한 강의에 대해 파이프라인 한 번 돌리고 graphs/, outputs/ 둘 다 저장
def run(lec_dir, pipeline_name, enrich, model=""):
    lec_dir = os.path.abspath(lec_dir)
    if not os.path.isdir(lec_dir):
        print(f"Error: {lec_dir} not found")
        sys.exit(1)

    rel = os.path.relpath(lec_dir, INPUT_DIR)
    output_name = rel.replace(os.sep, "_")

    transcript_path = _find_file(lec_dir, "transcript")
    if not transcript_path:
        print(f"Error: no 'transcript' file in {lec_dir}")
        sys.exit(1)

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = f.read()
    if not transcript.strip():
        print(f"Error: transcript is empty")
        sys.exit(1)
    print(f"Transcript: {len(transcript):,} chars ({os.path.basename(transcript_path)})")

    config = dict(DEFAULT_CONFIG)
    config["pipeline"] = pipeline_name
    config["enrich_graph"] = enrich

    if model:
        for k in ("ku_model", "node_model", "strict_model", "soft_model", "refine_model", "model"):
            config[k] = model

    slides_path = _find_file(lec_dir, "slides")
    if slides_path:
        with open(slides_path, "r", encoding="utf-8") as f:
            config["slide_text"] = f.read()
        print(f"Slides: {len(config['slide_text']):,} chars ({os.path.basename(slides_path)})")
    elif pipeline_name.startswith("slide"):
        print(f"Warning: {pipeline_name} needs slides — falling back to multi_stage")
        pipeline_name = "multi_stage"
        config["pipeline"] = pipeline_name

    print(f"\npipeline={pipeline_name}, enrich={enrich}" + (f", model={model}" if model else ""))
    print("Running...\n")

    reset_usage()
    t0 = time.time()
    result = get_pipeline(pipeline_name)(transcript, config)
    elapsed = time.time() - t0

    kg = {"nodes": result.get("nodes", []), "edges": result.get("edges", [])}
    print(f"\n{elapsed:.1f}s — {len(kg['nodes'])} nodes, {len(kg['edges'])} edges")
    print(usage_report())

    os.makedirs(GRAPHS_DIR, exist_ok=True)
    graph_path = os.path.join(GRAPHS_DIR, f"{output_name}.json")
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(kg, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: graphs/{output_name}.json")
    print(f"View:  http://localhost:8050/?graph={output_name}")

    out_dir = os.path.join(OUTPUTS_DIR, output_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "graph.json"), "w", encoding="utf-8") as f:
        json.dump(kg, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "transcript.txt"), "w", encoding="utf-8") as f:
        f.write(transcript)

    usage = usage_snapshot()
    usage["wall_time_s"] = round(elapsed, 2)
    usage["pipeline"] = pipeline_name
    usage["model_override"] = model or None
    with open(os.path.join(out_dir, "usage.json"), "w", encoding="utf-8") as f:
        json.dump(usage, f, indent=2, ensure_ascii=False)
    if result.get("kus"):
        with open(os.path.join(out_dir, "knowledge_units.json"), "w", encoding="utf-8") as f:
            json.dump(result["kus"], f, indent=2, ensure_ascii=False)
    print(f"Saved: outputs/{output_name}/")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Knowledge Graph from lecture input",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_graph.py --list
  python run_graph.py input/ecommerce/lec4
  python run_graph.py input/ecommerce/lec5 -p slide_anchored --enrich
  python run_graph.py input/ecommerce/lec3 -p multi_stage
        """,
    )
    parser.add_argument("lecture", nargs="?", help="Path to lecture folder (e.g. input/ecommerce/lec4)")
    parser.add_argument("--list", action="store_true", help="List available inputs")
    parser.add_argument(
        "--pipeline", "-p",
        default="slide_anchored",
        help=f"Pipeline to use (default: slide_anchored). Available: {', '.join(list_pipelines())}",
    )
    parser.add_argument("--enrich", "-e", action=argparse.BooleanOptionalAction, default=True, help="Add descriptions (extra LLM call). Default: True. Use --no-enrich to skip.")
    parser.add_argument("--model", "-m", default="", help="Override model for all stages (e.g. gpt-5.2, gpt-5.4, gpt-4.5). Empty = use config default.")
    parser.add_argument("--list-pipelines", action="store_true", help="List available pipelines")

    args = parser.parse_args()

    if args.list:
        list_inputs()
        return

    if args.list_pipelines:
        print("\nAvailable pipelines:")
        for p in list_pipelines():
            print(f"  {p}")
        print()
        return

    if not args.lecture:
        parser.print_help()
        return

    run(args.lecture, args.pipeline, args.enrich, args.model)


if __name__ == "__main__":
    main()
