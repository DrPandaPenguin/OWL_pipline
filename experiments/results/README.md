# Experiment results (cited in dissertation §4)

Aggregated metrics and raw per-run outputs for the cited experiments.

| Directory | Dissertation section |
|---|---|
| `exp1_pipeline_architecture/` | §4.3 Pipeline Architecture |
| `exp3_strict_vs_soft_no_ku/` | §4.5 Grounded vs Soft Edges |
| `exp9_model_sweep/lec3,lec4/` | §4.7 Model Comparison (metrics only) |
| `stability_test/` | §4.8 Consistency (metrics + over-extraction analysis) |
| `metrics.json`, `run1/2.json` (root) | consistency test runs |

## Note on redaction

Raw `runNN.json` files retain node labels, edge structure, and all numeric
metrics, but **verbatim transcript quotes** (`source_sentence`,
`justification_sentence`, `evidence`) are replaced with `[redacted: lecture
transcript]`. The source lectures are third-party copyright and are not
redistributed (see top-level note on `data/`). Aggregated `metrics.json` files
contain only statistics and are unredacted.

`exp9_model_sweep/` includes only `metrics.json` (raw runs omitted — they
contained verbatim slide-anchored quotes).
