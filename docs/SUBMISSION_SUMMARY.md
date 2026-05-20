# OWL Pipeline — Submission Summary

A standalone overview of the submitted codebase: what each part does, line counts,
and how the code is structured. Intended for self-review and external verification.

---

## 1. What the system does (one paragraph)

OWL Pipeline converts a university lecture (transcript + slide deck) into a typed
knowledge graph. The production pipeline (`slide_anchored`) uses the slide deck as
a structural anchor, extracts concept nodes and typed edges via LLM calls, validates
them deterministically in Python, and optionally enriches them with student-facing
explanations. A separate Gemini-based auditor can fact-check the output. The graph
is viewed in an interactive Dash + Cytoscape web application, deployed on HuggingFace
Spaces.

---

## 2. Directory structure

```
owl-pipeline/
├── src/                  production system (the final V3 pipeline)
│   ├── pipelines.py            slide_anchored + direct pipelines
│   ├── extract_edges.py        soft-edge extraction
│   ├── build_graph.py          KG assembly
│   ├── hallucination_checker.py Gemini fact-check
│   ├── usage_tracker.py        OpenAI token/cost tracking
│   ├── prompt_loader.py        loads prompts from prompts/
│   └── visualize_graph.py      static PNG (matplotlib CLI)
├── experiments/         evaluation / comparison study (self-contained)
│   ├── pipelines.py            20 pipeline variants (8 main + 12 segmentation)
│   ├── extract_nodes.py        KU extraction (used by V1/V2 variants)
│   ├── extract_edges.py        full edge extraction (strict + soft + refine)
│   ├── config.py               experiment matrix
│   ├── metrics.py              F1, components, orphan metrics
│   ├── precision_recall.py     P/R vs slide ground truth
│   ├── report.py               markdown result tables
│   ├── run_experiment.py       180-run experiment runner
│   ├── run_consistency_test.py 15-replicate consistency test
│   └── results/                consistency test output (JSON)
├── prompts/             26 LLM prompt files (.txt)
├── dash_app.py          web UI (builder + JSON loader)
├── run_graph.py         CLI entry point
└── viewer.py            read-only viewer (excluded from formal submission)
```

---

## 3. Line counts

| Group | Code | Comment | Blank | Total |
|---|---:|---:|---:|---:|
| `src/` (production) | 1,364 | 54 | 281 | 1,699 |
| `experiments/` (evaluation) | 4,078 | 279 | 809 | 5,166 |
| `dash_app.py` + `run_graph.py` | 1,256 | 89 | 194 | 1,539 |
| **Total (submission)** | **6,698** | — | — | **8,404** |
| `viewer.py` (excluded) | 682 | 32 | 103 | 817 |

Excluded from line count: libraries, generated outputs, dead code already removed.

---

## 4. Production pipeline (slide_anchored) — 6 stages

| Stage | Type | What it does |
|---|---|---|
| 1. Slide parsing | Python | Regex extracts PART sections from slide text |
| 2. Node extraction | LLM × 1 | Extract concept nodes anchored to slide sections |
| 3. Anchor validation | Python | Verify slide IDs, assign node IDs, fuzzy-match sentence index, compute ordering |
| 4. Edge extraction | LLM × 2 | Pass 1 grounded (verbatim evidence), Pass 2 soft (interpretive + confidence) |
| 5. Edge validation | Python | Drop duplicates/self-loops/invalid types, validate evidence section |
| 6. Enrichment | LLM × N | Add description, why_it_matters, edge explanation (optional) |

Output: typed knowledge graph (JSON). Optional post-pipeline Gemini hallucination check.

---

## 5. Edge schema (key design — three orthogonal fields)

| Field | Meaning | Source |
|---|---|---|
| `edge_source` | extraction tag: `explicit` or `inferred` | Python |
| `evidence` | transcript text supporting the relation (verbatim or paraphrase) | LLM |
| `explanation` | why the relation makes sense (pedagogical rationale) | LLM (enrichment) |

---

## 6. Pipeline evolution (research narrative)

| Version | Approach | Result | Why superseded |
|---|---|---|---|
| V1 single_pass | 1 LLM call → KG | F1 28.6%, 148 nodes (unusable) | extraction + structuring conflated |
| V2 multi_stage | KU → Node → Edge | F1 41–53%, high variance | global meaning lost |
| **V3 slide_anchored** | slide deck as anchor | stable, backbone visible | **adopted (production)** |

The experiments/ folder contains all variants so the comparison numbers cited in
the dissertation (§4) are reproducible.

---

## 7. Libraries used

| Library | Where | Purpose |
|---|---|---|
| openai | src + experiments | LLM calls (GPT) |
| google-genai | src | Gemini hallucination check |
| dash, dash-cytoscape | dash_app, viewer | interactive graph UI |
| networkx, matplotlib | visualize_graph | static PNG (CLI) |
| scikit-learn, sentence-transformers, numpy | experiments | segmentation variants only |
| Python stdlib (re, difflib, json, collections) | everywhere | parsing, fuzzy match |

No LLM framework (langchain, llama-index, instructor) — pipeline is plain Python.

---

## 8. Deployment

- Production builder UI deployed to HuggingFace Spaces (Docker).
- Read-only: paste a graph JSON into "Load existing KG from JSON" to inspect without
  running the pipeline or providing an API key.
