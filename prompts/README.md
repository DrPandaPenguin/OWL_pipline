# Prompts Directory

Each `.txt` file contains one LLM prompt (system or user).
Edit freely — changes take effect on the next pipeline run, no code change needed.

## Template syntax

User prompts use Python's `.format()` substitution:
- `{variable}` → replaced at runtime (e.g. `{transcript}`, `{nodes_text}`)
- `{{` and `}}` → literal `{` and `}` in the output (used for JSON examples)

## Prompt inventory

| File | Used in | Role |
|------|---------|------|
| `ku_extraction_system.txt` | `extract_nodes.py` | Phase 1 system: KU extraction |
| `ku_extraction_user.txt` | `extract_nodes.py` | Phase 1 user: KU extraction template |
| `node_construction_system.txt` | `extract_nodes.py` | Phase 3 system: KU → Node |
| `node_construction_user.txt` | `extract_nodes.py` | Phase 3 user: KU → Node template |
| `edge_refine_system.txt` | `extract_edges.py` | Edge refinement system |
| `edge_refine_user.txt` | `extract_edges.py` | Edge refinement user template |
| `single_pass_system.txt` | `pipelines.py` / `single_pass` | Single-pass pipeline system |
| `single_pass_user.txt` | `pipelines.py` / `single_pass` | Single-pass pipeline user template |
| `slide_parse_system.txt` | `pipelines.py` / `slide_*` | Slide structure parser system |
| `slide_parse_user.txt` | `pipelines.py` / `slide_*` | Slide structure parser user template |
| `slide_segment_system.txt` | `pipelines.py` / `slide_no_ku` | Transcript→section segmentation system |
| `slide_segment_user.txt` | `pipelines.py` / `slide_no_ku` | Transcript→section segmentation user |
| `slide_grounded_edge_system.txt` | `pipelines.py` / `slide_no_ku` | Grounded edge extraction system |
| `slide_grounded_edge_user.txt` | `pipelines.py` / `slide_no_ku` | Grounded edge extraction user |
| `node_anchored_system.txt` | `pipelines.py` / `slide_anchored*` | Combined slides+transcript → nodes system |
| `node_anchored_user.txt` | `pipelines.py` / `slide_anchored*` | Combined slides+transcript → nodes user |
| `edge_anchored_full_system.txt` | `pipelines.py` / `slide_anchored_full` | Node list + transcript → edges system |
| `edge_anchored_full_user.txt` | `pipelines.py` / `slide_anchored_full` | Node list + transcript → edges user |

## Skip slide parse

Set `"skip_slide_parse": true` in config to bypass the LLM slide-parse step.
The pipeline will use a Python markdown parser instead (splits on `#` headers).
Useful when slides are already well-structured markdown — saves one LLM call.
