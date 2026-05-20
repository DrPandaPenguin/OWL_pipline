#!/usr/bin/env bash
# Compare segmentation methods and merge strategies.
#
# Usage: source setup_env.sh && bash experiments/run_segmentation_experiment.sh
#
# Exp 7: Segmentation comparison (baseline + 4 methods, merge C fixed)
# Exp 8: Merge strategy comparison (TF-IDF fixed, 3 merge strategies)

set -e

if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY not set"
    echo "Usage: source setup_env.sh && bash experiments/run_segmentation_experiment.sh"
    exit 1
fi

RUNS=${1:-1}

echo "=== Segmentation Experiment ==="
echo "Runs per condition: $RUNS"
echo ""

echo "--- Exp 7: Segmentation Method Comparison ---"
echo "  (baseline, tfidf, fuzzy, embedding, llm) × merge C"
python3 -m experiments.run_experiment --group exp7_segmentation --runs "$RUNS"

echo ""
echo "--- Exp 8: Merge Strategy Comparison ---"
echo "  (merge A, B, C) × TF-IDF segmentation"
python3 -m experiments.run_experiment --group exp8_merge_strategy --runs "$RUNS"

echo ""
echo "=== Done. Results in results/ ==="
echo "Run P/R: python3 -m experiments.precision_recall"
