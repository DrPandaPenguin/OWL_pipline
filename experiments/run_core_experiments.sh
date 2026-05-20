#!/usr/bin/env bash
# Run the core experiments needed for the dissertation.
# Usage: OPENAI_API_KEY="sk-..." bash experiments/run_core_experiments.sh
#
# Runs 3 experiment groups (1 run each for quick test, 3 for final):
#   1. exp6: slide_anchored vs multi_stage (main comparison)
#   2. exp1: single_pass vs multi_stage vs refined (pipeline architecture)
#   3. exp3: strict vs soft vs both (edge extraction strategy)

set -e

if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY not set"
    echo "Usage: OPENAI_API_KEY='sk-...' bash experiments/run_core_experiments.sh"
    exit 1
fi

RUNS=${1:-1}  # default 1 run; pass 3 for final

echo "=== OWL Pipeline Core Experiments ==="
echo "Runs per condition: $RUNS"
echo ""

echo "--- Experiment 6: Slide vs KU Grounding (main comparison) ---"
python3 -m experiments.run_experiment exp6_multi_stage --runs "$RUNS"
python3 -m experiments.run_experiment exp6_slide_anchored --runs "$RUNS"

echo ""
echo "--- Experiment 1: Pipeline Architecture ---"
python3 -m experiments.run_experiment --group exp1_single_vs_multi --runs "$RUNS"

echo ""
echo "--- Experiment 3: Edge Extraction Strategy ---"
python3 -m experiments.run_experiment --group exp3_strict_soft --runs "$RUNS"

echo ""
echo "=== All core experiments done. Results in results/ ==="
echo "Run precision/recall: python3 -m experiments.precision_recall"
