# Repository Overview

This section describes the structure of the submitted source code, as required by the
Cambridge Part II project guidelines. The repository separates the **production system**
(the final pipeline described in this dissertation) from the **evaluation harness** used
to produce the comparison results in Chapter 4.

## Top-level layout

```
owl-pipeline/
├── src/              Production pipeline (the final V3 system)
├── experiments/      Evaluation harness (all pipeline variants + runners)
├── prompts/          LLM prompt templates (.txt, loaded at runtime)
├── dash_app.py       Interactive web visualisation (Dash + Cytoscape)
├── run_graph.py      Command-line entry point
└── requirements.txt  Python dependencies
```

## `src/` — production pipeline

The final slide-anchored pipeline and its supporting modules. Each file has a single
responsibility:

| Module | Responsibility |
|---|---|
| `pipelines.py` | Orchestrates the six pipeline stages (`slide_anchored`, `direct`) |
| `extract_edges.py` | Soft-edge extraction from transcript |
| `build_graph.py` | Assembles validated nodes and edges into the output graph |
| `hallucination_checker.py` | Gemini-based factual audit of the generated graph |
| `usage_tracker.py` | Tracks OpenAI token usage and cost |
| `prompt_loader.py` | Loads prompt templates from `prompts/` |
| `visualize_graph.py` | Renders a static PNG of a graph (command-line utility) |

## `experiments/` — evaluation harness

A self-contained package reproducing the empirical evaluation (Chapter 4). It includes
its own copies of the extraction modules so that the earlier pipeline variants (V1/V2),
which depend on knowledge-unit and strict-edge logic not present in the production
`src/`, remain runnable.

| Module | Responsibility |
|---|---|
| `pipelines.py` | Cited pipeline variants (single_pass, multi_stage, direct, slide_anchored) |
| `extract_nodes.py` | Knowledge-unit node extraction (used by V1/V2 variants) |
| `extract_edges.py` | Full edge extraction (grounded, soft, refinement) |
| `config.py` | Experiment matrix — which conditions to run |
| `metrics.py` | Graph structural metrics (F1, connected components, orphans) |
| `precision_recall.py` | Precision/recall against slide-derived ground truth |
| `report.py` | Generates result tables |
| `run_experiment.py` | Runs the experiment matrix |
| `run_consistency_test.py` | Repeated-run consistency test |
| `results/` | Saved consistency-test output (JSON) |

## Authorship and tools

All pipeline logic, prompts, validation, and evaluation code in `src/` and
`experiments/` were designed and written by the candidate. GitHub Copilot was used for
routine autocompletion during development. The interactive viewer (`dash_app.py`) was
developed with AI assistance for the front-end layout; the underlying data
transformations (`normalize_kg`, Cytoscape element construction) are the candidate's.

Third-party libraries (OpenAI SDK, Google Gemini SDK, Dash, Cytoscape, NetworkX,
scikit-learn) are used as documented in `requirements.txt` and are excluded from the
reported line count.

## Building and running

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...          # required for graph generation

# Command-line: generate a graph from a lecture folder
python run_graph.py input/<course>/<lecture>

# Web viewer (also deployed on HuggingFace Spaces)
python dash_app.py                 # http://localhost:8050

# Reproduce the evaluation
python -m experiments.run_experiment --list
python -m experiments.run_experiment <experiment_name>
```
