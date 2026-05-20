import sys
import os
import time
import json

# Add project root to path so we can import src modules
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.prompt_loader import load_prompt
from experiments.extract_nodes import (
    extract_knowledge_units,
    add_ids_and_timestamps,
    construct_nodes,
    process_nodes,
    compute_orphan_kus,
)
from experiments.extract_edges import extract_edges, refine_edges

# Registry
_PIPELINES = {}


def register_pipeline(name: str):
    """Decorator to register a pipeline function"""
    def decorator(fn):
        _PIPELINES[name] = fn
        return fn
    return decorator


def get_pipeline(name: str):
    """Look up a pipeline by name. Raises ValueError if not found"""
    if name not in _PIPELINES:
        raise ValueError(f"Unknown pipeline: {name}. Available: {list(_PIPELINES.keys())}")
    return _PIPELINES[name]


def list_pipelines():
    return list(_PIPELINES.keys())


# Shared helper: enrich_graph
#
# Post-extraction enrichment step. Adds student-facing explanations to
# every node and edge in the graph:
#
# Node fields added:
# description 2 3 sentences explaining the concept for revision
# why_it_matters 1 2 sentences on why the concept matters in the lecture
#
# Edge fields added:
# reason 1 2 sentences explaining the logical / pedagogical link
#
# Called by any pipeline that has config["enrich_graph"] = True.
# Silently returns the original graph unchanged on any error.

_GRAPH_ENRICH_SYSTEM = load_prompt("enrich_graph_system", """\
You are improving a lecture knowledge graph for student readability.

This graph has already been extracted from a lecture transcript.
Your job is NOT to rebuild, relabel, or validate the graph.
Your job is only to improve explanatory text fields so that the graph
is easier for a student to understand.

You may add or improve:
- node.description
- node.why_it_matters
- edge.reason

You must NOT:
- add or remove nodes or edges
- modify node labels, IDs, edge endpoints, edge_type, or justification
- modify graph topology
- perform hallucination checking

Definitions:
  node.description     A short explanation of what the node is.
  node.why_it_matters  A short explanation of why this node matters in the lecture.
  edge.reason          A short explanation of why the two connected nodes are related.

Important distinction:
- justification = evidence / where the connection comes from (must remain unchanged)
- reason = explanation / why the connection makes sense (may be added/refined)

Write concise, student-friendly text. Do not introduce new concepts
that are not supported by the graph context.

Return strict JSON only.\
""")

_GRAPH_ENRICH_USER = load_prompt("enrich_graph_user", """\
You are given an extracted lecture graph.

Your task is to enrich the graph by improving explanatory text only.

For each node:
- add or improve `description`
- add or improve `why_it_matters`

For each edge:
- add or improve `reason`

Do not change:
- labels
- IDs
- node or edge existence
- edge_type
- justification
- graph structure

GRAPH:
{graph_json}

OUTPUT FORMAT:
{{
  "nodes": [
    {{
      "id": "node_001",
      "description": "...",
      "why_it_matters": "..."
    }}
  ],
  "edges": [
    {{
      "edge_id": "edge_001",
      "reason": "..."
    }}
  ]
}}\
""")


def enrich_graph(
    nodes,
    edges,
    transcript: str,
    config,
):
    if not nodes:
        return {"nodes": nodes, "edges": edges}

    try:
        from openai import OpenAI
    except ImportError:
        return {"nodes": nodes, "edges": edges}

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"nodes": nodes, "edges": edges}

    client = OpenAI(api_key=api_key)
    model = config.get("node_model", "gpt-5.2")
    temperature = config.get("node_temperature", 0.2)

 # Batch size: max items per LLM call to avoid context overload
    BATCH_SIZE = 20

    def _edge_from(e):
        """Resolve 'from' regardless of whether the edge uses from/to or source/target"""
        return e.get("from") or e.get("source", "")

    def _edge_to(e):
        """Resolve 'to' regardless of whether the edge uses from/to or source/target"""
        return e.get("to") or e.get("target", "")

    def _edge_type(e):
        """Resolve edge_type regardless of whether it's edge_type or relation"""
        return e.get("edge_type") or e.get("relation", "")

    def _enrich_batch(batch_nodes, batch_edges):
        """Run one enrichment LLM call for a batch of nodes and/or edges"""
        node_summaries = [
            {"id": n.get("id", ""), "label": n.get("label", ""),
             "node_type": n.get("node_type", "concept"),
             "source_sentence": n.get("source_sentence", "")}
            for n in batch_nodes
        ]
        edge_summaries = [
            {"edge_id": e.get("edge_id", ""), "from": _edge_from(e),
             "to": _edge_to(e), "edge_type": _edge_type(e),
             "justification": e.get("justification", "")}
            for e in batch_edges
        ]
        graph_json = json.dumps(
            {"nodes": node_summaries, "edges": edge_summaries}, indent=2
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _GRAPH_ENRICH_SYSTEM},
                    {"role": "user", "content": _GRAPH_ENRICH_USER.format(
                        graph_json=graph_json,
                    )},
                ],
                response_format={"type": "json_object"},
                temperature=temperature,
            )
            return json.loads(resp.choices[0].message.content)
        except Exception:
            return {"nodes": [], "edges": []}

 # --- Split into batches and collect enrichments ---
    node_enrich_map = {}
    edge_enrich_map = {}

 # Batch nodes
    for i in range(0, max(len(nodes), 1), BATCH_SIZE):
        batch_n = nodes[i:i + BATCH_SIZE]
 # Find edges connected to this batch of nodes
        batch_node_ids = {n.get("id") for n in batch_n}
        batch_e = [e for e in edges if _edge_from(e) in batch_node_ids or _edge_to(e) in batch_node_ids]
 # Cap edges per batch too
        batch_e = batch_e[:BATCH_SIZE]

        result = _enrich_batch(batch_n, batch_e)

        for item in result.get("nodes", []):
            if isinstance(item, dict) and "id" in item:
                node_enrich_map[item["id"]] = item
        for item in result.get("edges", []):
            if not isinstance(item, dict):
                continue
            eid = item.get("edge_id", "")
            if eid:  # only store if edge_id is non-empty
                edge_enrich_map[eid] = item

 # Handle any edges not covered by the node batches
    enriched_edge_ids = set(edge_enrich_map.keys())
    remaining_edges = [e for e in edges if e.get("edge_id", "") not in enriched_edge_ids]
    if remaining_edges:
        for i in range(0, len(remaining_edges), BATCH_SIZE):
            batch_e = remaining_edges[i:i + BATCH_SIZE]
            result = _enrich_batch([], batch_e)
            for item in result.get("edges", []):
                if not isinstance(item, dict):
                    continue
                eid = item.get("edge_id", "")
                if eid:
                    edge_enrich_map[eid] = item

 # --- Merge enrichments ---
    enriched_nodes = []
    for n in nodes:
        node_copy = dict(n)
        enrich = node_enrich_map.get(n.get("id", ""), {})
        if "description" in enrich:
            node_copy["description"] = enrich["description"]
        if "why_it_matters" in enrich:
            node_copy["why_it_matters"] = enrich["why_it_matters"]
        enriched_nodes.append(node_copy)

    enriched_edges = []
    for e in edges:
        edge_copy = dict(e)
 # Use edge_id as primary key; fall back to (from, to, edge_type) tuple
        eid = e.get("edge_id", "")
        if eid and eid in edge_enrich_map:
            enrich = edge_enrich_map[eid]
        else:
 # Fallback: match by endpoint+type for edges that lost their edge_id
            fallback_key = (_edge_from(e), _edge_to(e), _edge_type(e))
            enrich = next(
                (v for v in edge_enrich_map.values()
                 if (v.get("from"), v.get("to"), v.get("edge_type")) == fallback_key),
                {}
            )
        if "reason" in enrich:
            edge_copy["reason"] = enrich["reason"]
        enriched_edges.append(edge_copy)

    return {"nodes": enriched_nodes, "edges": enriched_edges}


# Post-processing helpers (slide_anchored_full pipeline)

def _split_transcript_sentences(transcript: str):
    """Split transcript into sentences for fuzzy matching"""
    import re
    parts = re.split(r'(?<=[.!?])\s+', transcript.strip())
    return [s.strip() for s in parts if s.strip()]


def _find_sentence_index(source_sentence: str, transcript_sentences) -> int:
    """Find the 0-based index of source_sentence in transcript_sentences"""
    from difflib import SequenceMatcher
    if not source_sentence or not transcript_sentences:
        return 0
    src = source_sentence.lower().strip()
    best_idx, best_score = 0, 0.0
    for i, sent in enumerate(transcript_sentences):
        score = SequenceMatcher(None, src, sent.lower().strip()).ratio()
        if score > best_score:
            best_score, best_idx = score, i
    return best_idx


def _resolve_parent_id(
    parent_label: str,
    nodes,
    threshold: float = 0.85,
) -> Any:
    """Resolve a parent_id label (LLM free text) to a node ID using fuzzy matching"""
    from difflib import SequenceMatcher
    if not parent_label:
        return None
    norm = parent_label.lower().strip()
    best_id, best_score = None, 0.0
    for n in nodes:
        score = SequenceMatcher(None, norm, n["label"].lower().strip()).ratio()
        if score > best_score:
            best_score, best_id = score, n["id"]
    return best_id if best_score >= threshold else None


def _compute_ordering(nodes):
    """Compute section_order and lecture_order for each node in-place"""
    from collections import defaultdict

 # --- section_order: rank within section by sentence_index ---
    by_section = defaultdict(list)
    for n in nodes:
        by_section[n.get("slide_anchor_id", "")].append(n)

    for section_nodes in by_section.values():
        sorted_nodes = sorted(section_nodes, key=lambda n: n.get("sentence_index", 0))
        for rank, n in enumerate(sorted_nodes, 1):
            n["section_order"] = rank

 # --- lecture_order: backbone nodes sorted by (PART number, sentence_index) ---
    def _part_num(n) -> int:
        anchor = n.get("slide_anchor_id", "PART_0")
        import re
        m = re.search(r'\d+', anchor)
        return int(m.group()) if m else 0

    backbone = [n for n in nodes if n.get("is_backbone", False)]
    backbone_sorted = sorted(backbone, key=lambda n: (_part_num(n), n.get("sentence_index", 0)))
    for rank, n in enumerate(backbone_sorted, 1):
        n["lecture_order"] = rank

 # Non-backbone nodes get None
    for n in nodes:
        if "lecture_order" not in n:
            n["lecture_order"] = None

    return nodes


# Pipeline 1: multi_stage (current production pipeline)
@register_pipeline("multi_stage")
def pipeline_multi_stage(transcript: str, config):
    """Current production pipeline: KU → Node → Strict Edge → Soft Edge"""
    timing = {}

 # Phase 1: KU extraction
    t0 = time.time()
    raw_kus = extract_knowledge_units(transcript, config)
    timing["phase1_ku_extraction"] = round(time.time() - t0, 2)

    if not raw_kus:
        return {"nodes": [], "edges": [], "kus": [], "timing": timing}

 # Glue 1: IDs + timestamps
    t1 = time.time()
    kus = add_ids_and_timestamps(raw_kus, transcript, config)
    timing["glue1_ids_timestamps"] = round(time.time() - t1, 2)

 # Phase 3: Node construction
    t2 = time.time()
    raw_nodes = construct_nodes(kus, config)
    timing["phase3_node_construction"] = round(time.time() - t2, 2)

 # Glue 2: Node IDs + timestamps
    t3 = time.time()
    nodes = process_nodes(raw_nodes, kus)
    for n in nodes:
        n["node_id"] = n["id"]
    timing["glue2_node_ids"] = round(time.time() - t3, 2)

    if not nodes or len(nodes) < 2:
        return {"nodes": nodes, "edges": [], "kus": kus, "timing": timing}

 # Phase 4+5: Edge extraction
    t4 = time.time()
    edges = extract_edges(transcript, nodes, knowledge_units=kus, include_soft=True, config=config)
    timing["phase4_5_edge_extraction"] = round(time.time() - t4, 2)

    timing["total"] = round(sum(timing.values()), 2)
    return {"nodes": nodes, "edges": edges, "kus": kus, "timing": timing}


# Pipeline 2: single_pass (baseline no KU, one LLM call)

_SINGLE_PASS_SYSTEM = load_prompt("single_pass_system", """\
You are a knowledge graph extraction system.
Given a lecture transcript, extract nodes (key concepts) and edges (relationships) in a single pass.

Return strict JSON only.\
""")

_SINGLE_PASS_USER = load_prompt("single_pass_user", """\
Extract a knowledge graph from this lecture transcript.

Transcript:
---
{transcript}
---

Instructions:
- Nodes: key concepts, definitions, examples from the lecture. Short noun-phrase labels.
- Edges: relationships between nodes.
- Edge types allowed: {edge_types_str}
- Each edge needs a justification_sentence (short quote from transcript).

Output strict JSON:
{{
  "nodes": [
    {{ "label": "Concept Name" }}
  ],
  "edges": [
    {{
      "from_label": "Source Concept",
      "to_label": "Target Concept",
      "edge_type": "requires",
      "justification_sentence": "quote from transcript"
    }}
  ]
}}\
""")


@register_pipeline("single_pass")
def pipeline_single_pass(transcript: str, config):
    """Baseline: single LLM call produces nodes + edges together. No KU step"""
    timing = {}

 # Import OpenAI client
    try:
        from openai import OpenAI
    except ImportError:
        return {"nodes": [], "edges": [], "kus": [], "timing": timing, "error": "openai not installed"}

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"nodes": [], "edges": [], "kus": [], "timing": timing, "error": "OPENAI_API_KEY not set"}

    client = OpenAI(api_key=api_key)
    model = config.get("ku_model", "gpt-5.2")
    temperature = config.get("ku_temperature", 0.1)
    edge_types = config.get("edge_types", ["defines", "requires", "explains", "details", "example_of", "contrasts", "drives"])
    edge_types_str = ", ".join(edge_types)

    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SINGLE_PASS_SYSTEM},
                {"role": "user", "content": _SINGLE_PASS_USER.format(
                    transcript=transcript,
                    edge_types_str=edge_types_str,
                )},
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
        )
        data = json.loads(response.choices[0].message.content)
    except Exception as e:
        timing["single_pass_llm"] = round(time.time() - t0, 2)
        return {"nodes": [], "edges": [], "kus": [], "timing": timing, "error": str(e)}

    timing["single_pass_llm"] = round(time.time() - t0, 2)

 # Python: assign node IDs, map label→ID
    raw_nodes = data.get("nodes", [])
    label_to_id = {}
    nodes = []
    for i, n in enumerate(raw_nodes):
        label = (n.get("label") or "").strip()
        if not label:
            continue
        nid = f"node_{i + 1:03d}"
        label_to_id[label.lower()] = nid
        nodes.append({
            "id": nid,
            "node_id": nid,
            "label": label,
            "supporting_ku_ids": [],
            "timestamp": {"method": "single_pass", "sentence_index": 0},
        })

 # Python: assign edge IDs, resolve labels to IDs
    raw_edges = data.get("edges", [])
    edges = []
    valid_types = set(edge_types)
    for i, e in enumerate(raw_edges):
        from_label = (e.get("from_label") or "").strip().lower()
        to_label = (e.get("to_label") or "").strip().lower()
        fr = label_to_id.get(from_label)
        to = label_to_id.get(to_label)
        if not fr or not to:
            continue
        etype = e.get("edge_type", "")
        if etype not in valid_types:
            continue
        edges.append({
            "edge_id": f"edge_{i + 1:03d}",
            "from": fr,
            "to": to,
            "edge_type": etype,
            "justification_sentence": (e.get("justification_sentence") or "").strip(),
        })

    timing["total"] = round(sum(timing.values()), 2)
    return {"nodes": nodes, "edges": edges, "kus": [], "timing": timing}


# Pipeline 2b: direct (no KU, no slides Node extraction → 2-pass Edge extraction)

_DIRECT_NODE_SYSTEM = """\
You are a knowledge graph extraction system specialising in university lectures.
Given a lecture transcript, extract the key concepts as nodes.

Each node should represent a retained knowledge unit — an academic concept, formal definition,
principle, mental model, analogy, or key example — that a student needs after the lecture.

Return strict JSON only.\
"""

_DIRECT_NODE_USER = """\
LECTURE TRANSCRIPT:
---
{transcript}
---

Extract all key concepts from this lecture as nodes.

Rules:
1. Each node needs a short noun-phrase label (2-6 words).
2. Include a source_sentence: a verbatim quote from the transcript that best supports this concept.
3. Mark is_backbone: true for core concepts essential to following the lecture, false for supporting details.
4. Aim for 30-80 nodes depending on lecture density.

Output strict JSON:
{{
  "nodes": [
    {{
      "label": "Concept Name",
      "source_sentence": "verbatim quote from transcript",
      "is_backbone": true
    }}
  ]
}}
"""


@register_pipeline("direct")
def pipeline_direct(transcript: str, config):
    """Direct pipeline: Transcript → Node extraction (1 LLM call) → 2-pass Edge extraction"""
    timing = {}

    try:
        from openai import OpenAI
    except ImportError:
        return {"nodes": [], "edges": [], "kus": [], "timing": timing, "error": "openai not installed"}

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"nodes": [], "edges": [], "kus": [], "timing": timing, "error": "OPENAI_API_KEY not set"}

    client = OpenAI(api_key=api_key)
    model = config.get("node_model", "gpt-5.2")
    temperature = config.get("node_temperature", 0.2)

 # Step 1: Node extraction (single LLM call)
    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _DIRECT_NODE_SYSTEM},
                {"role": "user", "content": _DIRECT_NODE_USER.format(transcript=transcript)},
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
        )
        data = json.loads(response.choices[0].message.content)
    except Exception as e:
        timing["node_extraction"] = round(time.time() - t0, 2)
        return {"nodes": [], "edges": [], "kus": [], "timing": timing, "error": str(e)}

    timing["node_extraction"] = round(time.time() - t0, 2)

 # Python: assign IDs
    raw_nodes = data.get("nodes", [])
    nodes = []
    for i, n in enumerate(raw_nodes):
        label = (n.get("label") or "").strip()
        if not label:
            continue
        nid = f"node_{i + 1:03d}"
        nodes.append({
            "id": nid,
            "node_id": nid,
            "label": label,
            "source_sentence": (n.get("source_sentence") or "").strip(),
            "is_backbone": n.get("is_backbone", False),
            "supporting_ku_ids": [],
            "timestamp": {"method": "direct", "sentence_index": 0},
        })

    if not nodes or len(nodes) < 2:
        timing["total"] = round(sum(timing.values()), 2)
        return {"nodes": nodes, "edges": [], "kus": [], "timing": timing}

 # Step 2: 2-pass edge extraction (reuses extract_edges from src)
    t1 = time.time()
    edges = extract_edges(transcript, nodes, knowledge_units=[], include_soft=True, config=config)
    timing["edge_extraction_2pass"] = round(time.time() - t1, 2)

    timing["total"] = round(sum(timing.values()), 2)
    return {"nodes": nodes, "edges": edges, "kus": [], "timing": timing}


# Slide parser helpers (shared by slide_structure, slide_no_ku, slide_anchored*)

def _extract_part_ids(slide_text: str):
    """Lightweight parser — returns only {slide_id, title} for each PART section"""
    import re as _re
    PART_RE = _re.compile(
        r"^PART\s+([IVXLCDM]+|\d+)\s*[—–\-]+\s*(.+)$",
        _re.MULTILINE,
    )
    results = []
    for i, m in enumerate(PART_RE.finditer(slide_text)):
        results.append({
            "slide_id": f"part_{i + 1:03d}",
            "title": m.group(2).strip(),
        })
    return results


def _extract_part_sections(slide_text: str):
    """Rich Python parser for structured slide notes in the format:"""
    import re as _re

    PART_RE = _re.compile(
        r"^PART\s+([IVXLCDM]+|\d+)\s*[—–\-]+\s*(.+)$",
        _re.MULTILINE,
    )
    BULLET_RE = _re.compile(r"^[•\-\*]\s*(.+)$")

    part_matches = list(PART_RE.finditer(slide_text))
    if not part_matches:
        return []

 # Determine text boundaries between consecutive PART headers
    boundaries = [m.start() for m in part_matches] + [len(slide_text)]

    sections = []
    for i, m in enumerate(part_matches):
        section_id = f"part_{i + 1:03d}"
        title = m.group(2).strip()
        section_text = slide_text[boundaries[i]:boundaries[i + 1]]

 # ---- Core Focus ----
        core_focus = ""
        cf = _re.search(
            r"Core Focus[:\s]+(.+?)(?=\n(?:Key Ideas|Why This Part|PART\s|[⸻—–]{2})|$)",
            section_text, _re.DOTALL | _re.IGNORECASE,
        )
        if cf:
            core_focus = " ".join(cf.group(1).split())  # collapse whitespace

 # ---- Key Ideas (bullet lines after "Key Ideas" heading) ----
        key_ideas = []
        ki = _re.search(
            r"Key Ideas\s*\n((?:.+\n?)*?)(?=\n(?:Why This Part|PART\s|[⸻—–]{2})|$)",
            section_text, _re.IGNORECASE,
        )
        if ki:
            for line in ki.group(1).splitlines():
                bm = BULLET_RE.match(line.strip())
                if bm:
                    key_ideas.append(bm.group(1).strip())

 # ---- Why This Part Exists ----
        why = ""
        wy = _re.search(
            r"Why This Part Exists\s*\n(.+?)(?=[⸻—–]{2}|PART\s|$)",
            section_text, _re.DOTALL | _re.IGNORECASE,
        )
        if wy:
            why = " ".join(wy.group(1).split())

        sections.append({
            "section_id": section_id,
            "title": title,
            "core_focus": core_focus,
            "key_ideas": key_ideas,
            "why_this_part_exists": why,
        })

    return sections


def _format_slide_sections_text(sections) -> str:
    """Format a list of _extract_part_sections() dicts into the structured"""
    blocks = []
    for sec in sections:
        lines = [f"[{sec['section_id']}] {sec['title']}"]
        if sec.get("core_focus"):
            lines.append(f"  Core Focus: {sec['core_focus']}")
        if sec.get("key_ideas"):
            lines.append("  Key Ideas:")
            for idea in sec["key_ideas"]:
                lines.append(f"    • {idea}")
        if sec.get("why_this_part_exists"):
            lines.append(f"  Why This Part Exists: {sec['why_this_part_exists']}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _parse_slides_python(slide_text: str):
    """Pure-Python slide parser — no LLM call"""
    import re as _re

    ADMIN_KEYWORDS = {"outline", "agenda", "logistics", "overview", "today",
                      "welcome", "introduction", "recap", "summary", "plan"}
    HEADER_RE = _re.compile(r"^#{1,4}\s+(.+)$")
    BULLET_RE = _re.compile(r"^[-*•]|\d+\.\s")

    sections = []
    current_title = None
    current_bullets = []

    def _flush():
        nonlocal current_title, current_bullets
        if current_title:
            title_words = current_title.lower().split()
            is_admin = (
                len(title_words) <= 3
                and any(w in ADMIN_KEYWORDS for w in title_words)
            )
            if not is_admin:
                sections.append({
                    "title": current_title,
                    "bullets": list(current_bullets),
                })
        current_title = None
        current_bullets = []

    for raw_line in slide_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = HEADER_RE.match(line)
        if m:
            _flush()
            current_title = m.group(1).strip()
        elif BULLET_RE.match(line) and current_title:
            bullet = _re.sub(r"^[-*•\d.]+\s*", "", line).strip()
            if bullet:
                current_bullets.append(bullet)
 # non-header, non-bullet lines are ignored (slide body text / page numbers)

    _flush()

 # Fallback: no headers found treat each non-empty line as a section title
    if not sections:
        for raw_line in slide_text.splitlines():
            line = raw_line.strip()
            if line and not BULLET_RE.match(line):
                sections.append({"title": line, "bullets": []})

    return sections


def _get_slide_sections(
    slide_text: str,
    client,
    model: str,
    temperature: float,
    config,
):
    """Parse slide_text into a list of {title, bullets} sections"""
    if config.get("skip_slide_parse", False):
        return _parse_slides_python(slide_text)

 # LLM parse path
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SLIDE_PARSE_SYSTEM},
                {"role": "user", "content": _SLIDE_PARSE_USER.format(slides=slide_text)},
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
        )
        data = json.loads(resp.choices[0].message.content)
        return data.get("sections", [])
    except Exception as e:
        print(f"[slide_parse] LLM error: {e}", file=__import__("sys").stderr)
        return []


# Pipeline 3: slide_structure (lecture slides as node backbone)

_SLIDE_PARSE_SYSTEM = load_prompt("slide_parse_system", (
    "You are a lecture slide parser. Extract slide section titles and their key bullet points. "
    "Return strict JSON only."
))

_SLIDE_PARSE_USER = load_prompt("slide_parse_user", """\
Parse the following lecture slides and extract their structure.

Slides:
---
{slides}
---

Return strict JSON:
{{
  "sections": [
    {{
      "title": "Slide or section title",
      "bullets": ["key bullet point", "another point"]
    }}
  ]
}}

Instructions:
- Each "title" becomes a candidate node label — make it a short noun phrase.
- Include only slides/sections with conceptual content (skip pure admin/logistics slides).
- Merge near-duplicate slides into one section.
- If no structured slides are found, return {{"sections": []}}\
""")


@register_pipeline("slide_structure")
def pipeline_slide_structure(transcript: str, config):
    """Slide-anchored pipeline: uses lecture slide titles as the node backbone"""
    timing = {}
    slide_text = (config.get("slide_text") or "").strip()

    if not slide_text:
 # No slides provided fall back gracefully
        import warnings
        warnings.warn("slide_structure: no slide_text in config, falling back to multi_stage.")
        return pipeline_multi_stage(transcript, config)

    try:
        from openai import OpenAI
    except ImportError:
        return {"nodes": [], "edges": [], "kus": [], "timing": timing, "error": "openai not installed"}

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"nodes": [], "edges": [], "kus": [], "timing": timing, "error": "OPENAI_API_KEY not set"}

    client = OpenAI(api_key=api_key)
    model = config.get("node_model", "gpt-5.2")
    temperature = config.get("node_temperature", 0.1)

 # Step 1: Parse slides → section titles (LLM or Python depending on config)
    t0 = time.time()
    sections = _get_slide_sections(slide_text, client, model, temperature, config)
    timing["slide_parse"] = round(time.time() - t0, 2)

    if not sections:
 # Slides parsed but empty fall back
        return pipeline_multi_stage(transcript, config)

 # Step 2: Python seed nodes from slide sections
    nodes = []
    for i, sec in enumerate(sections):
        title = (sec.get("title") or "").strip()
        if not title:
            continue
        nid = f"node_{i + 1:03d}"
        nodes.append({
            "id": nid,
            "node_id": nid,
            "label": title,
            "supporting_ku_ids": [],
            "slide_bullets": sec.get("bullets", []),
            "timestamp": {"method": "slide_structure", "sentence_index": i},
        })

    if not nodes:
        return pipeline_multi_stage(transcript, config)

 # Step 3: KU extraction from transcript (Phase 1)
    t1 = time.time()
    raw_kus = extract_knowledge_units(transcript, config)
    timing["phase1_ku_extraction"] = round(time.time() - t1, 2)

    kus = []
    if raw_kus:
        t2 = time.time()
        kus = add_ids_and_timestamps(raw_kus, transcript, config)
        timing["glue1_ids_timestamps"] = round(time.time() - t2, 2)

 # Step 4: Edge extraction using slide-seeded nodes
    if len(nodes) < 2:
        timing["total"] = round(sum(timing.values()), 2)
        return {"nodes": nodes, "edges": [], "kus": kus, "timing": timing}

    t3 = time.time()
    edges = extract_edges(
        transcript, nodes,
        knowledge_units=kus if kus else None,
        include_soft=True,
        config=config,
    )
    timing["phase4_5_edge_extraction"] = round(time.time() - t3, 2)

    timing["total"] = round(sum(timing.values()), 2)
    return {"nodes": nodes, "edges": edges, "kus": kus, "timing": timing}


# Pipeline 4: multi_stage_refined (multi_stage + edge refinement pass)
@register_pipeline("multi_stage_refined")
def pipeline_multi_stage_refined(transcript: str, config):
    """multi_stage + one additional LLM pass that reviews all extracted edges"""
    timing = {}

 # Run base multi_stage pipeline
    base = pipeline_multi_stage(transcript, config)
    timing.update(base.get("timing", {}))

    nodes = base["nodes"]
    edges = base["edges"]
    kus = base["kus"]

    if not nodes or not edges:
        return base

 # Refinement pass
    t_refine = time.time()
    refined_edges = refine_edges(transcript, nodes, edges, config)
    timing["refine_pass"] = round(time.time() - t_refine, 2)
    timing["total"] = round(sum(timing.values()), 2)

    return {"nodes": nodes, "edges": refined_edges, "kus": kus, "timing": timing}


# Pipeline 5: slide_no_ku
#
# Design question this pipeline answers:
# "Can slide sections replace KUs as the grounding unit for edge extraction?"
#
# KU role in multi_stage:
# 1. Content segmentation breaks transcript into atomic evidence units
# 2. Evidence grounding strict edge justification_sentence comes from KU text
#
# This pipeline replaces both roles with slide-aligned transcript segments:
# 1. Slide sections divide the transcript into topically coherent chunks
# (preserves lecture flow within each section; doesn't shatter sentences)
# 2. Segment text is used as evidence for grounded edges (same strictness
# as KU-grounded edges but with paragraph-level rather than sentence-level context)
#
# Flow:
# Step 1 (LLM) Parse slide_text → section titles → node list
# Step 2 (Python) Assign node IDs
# Step 3 (LLM) Segment transcript by slide sections → {node_id, segment_text}
# Step 4 (LLM) Extract grounded edges from segments (replaces strict pass)
# Step 5 (LLM) Extract soft edges from full transcript
# Step 6 (LLM) Refine pass (optional, enabled by default)
#
# Requires: config["slide_text"] falls back to multi_stage_refined if empty.

_SEGMENT_TRANSCRIPT_SYSTEM = load_prompt("slide_segment_system", (
    "You are a lecture transcript segmentation assistant. "
    "Given slide section titles and a lecture transcript, assign transcript text "
    "to the most relevant slide section. "
    "Return strict JSON only."
))

_SEGMENT_TRANSCRIPT_USER = load_prompt("slide_segment_user", """\
Segment this lecture transcript by slide sections.

SLIDE SECTIONS (node_id → title):
{sections_text}

TRANSCRIPT:
---
{transcript}
---

For each slide section, extract the portion of the transcript that most directly
covers that section's topic. Copy sentences verbatim from the transcript.

Rules:
- segment_text should be verbatim transcript text (do NOT paraphrase)
- A sentence belongs to the section it is most topically relevant to
- If a section has no matching content set segment_text to ""
- Sections may have varying length — quality over coverage

Return strict JSON:
{{
  "segments": [
    {{
      "node_id": "node_001",
      "title": "Section title",
      "segment_text": "Verbatim transcript sentences for this section..."
    }}
  ]
}}\
""")


_SLIDE_GROUNDED_EDGE_SYSTEM = load_prompt("slide_grounded_edge_system", (
    "You are a lecture knowledge graph extractor. "
    "Extract relationships between lecture concepts. "
    "Each edge must be grounded in the transcript segment of its source or target node. "
    "Return strict JSON only."
))

_SLIDE_GROUNDED_EDGE_USER = load_prompt("slide_grounded_edge_user", """\
Extract GROUNDED edges between lecture nodes.

Each edge must be justified by a verbatim or near-verbatim quote from the
segment text of either the source or target node shown below.

NODES and their transcript segments:
{segments_text}

EDGE TYPES (use only these {num_types}):
{types_section}

RULES:
1. Use only node IDs listed above (exact match for "from" and "to").
2. justification_sentence MUST be a verbatim or near-verbatim copy of text
   from the segment of the source OR target node.
3. One justification sentence per edge. Do NOT invent or paraphrase.
4. Do NOT create self-loops (from == to).
5. Only extract edges where the relationship is explicitly supported by the text.

Return strict JSON:
{{
  "edges": [
    {{
      "from": "node_001",
      "to": "node_002",
      "edge_type": "requires",
      "justification_sentence": "verbatim text from the relevant segment"
    }}
  ]
}}\
""")


@register_pipeline("slide_no_ku")
def pipeline_slide_no_ku(transcript: str, config):
    """Slide-anchored pipeline WITHOUT KU extraction"""
    timing = {}
    slide_text = (config.get("slide_text") or "").strip()

    if not slide_text:
        import warnings
        warnings.warn("slide_no_ku: no slide_text in config, falling back to multi_stage_refined.")
        return pipeline_multi_stage_refined(transcript, config)

    try:
        from openai import OpenAI
    except ImportError:
        return {"nodes": [], "edges": [], "kus": [], "timing": timing, "error": "openai not installed"}

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"nodes": [], "edges": [], "kus": [], "timing": timing, "error": "OPENAI_API_KEY not set"}

    client = OpenAI(api_key=api_key)
    model = config.get("node_model", "gpt-5.2")
    strict_model = config.get("strict_model", model)
    soft_model = config.get("soft_model", model)
    node_temp = config.get("node_temperature", 0.1)
    strict_temp = config.get("strict_temperature", 0.1)

 # ---- Step 1: Parse slides → section titles (LLM or Python) ----
    t0 = time.time()
    sections = _get_slide_sections(slide_text, client, model, node_temp, config)
    timing["slide_parse"] = round(time.time() - t0, 2)

    if not sections:
        import warnings
        warnings.warn("slide_no_ku: slide parse produced no sections, falling back to multi_stage_refined.")
        return pipeline_multi_stage_refined(transcript, config)

 # ---- Step 2: Python seed nodes from slide sections ----
    nodes = []
    for i, sec in enumerate(sections):
        title = (sec.get("title") or "").strip()
        if not title:
            continue
        nid = f"node_{i + 1:03d}"
        nodes.append({
            "id": nid,
            "node_id": nid,
            "label": title,
            "supporting_ku_ids": [],            # empty — no KUs in this pipeline
            "slide_bullets": sec.get("bullets", []),
            "source_sentence": "; ".join(sec.get("bullets", []))[:120] or title,
            "timestamp": {"method": "slide_no_ku", "sentence_index": i},
        })

    if len(nodes) < 2:
        import warnings
        warnings.warn("slide_no_ku: fewer than 2 nodes from slides, falling back to multi_stage_refined.")
        return pipeline_multi_stage_refined(transcript, config)

 # ---- Step 3: LLM segments transcript by slide section ----
    sections_text = "\n".join(f"  {n['id']}: {n['label']}" for n in nodes)
    t1 = time.time()
    try:
        resp2 = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SEGMENT_TRANSCRIPT_SYSTEM},
                {"role": "user", "content": _SEGMENT_TRANSCRIPT_USER.format(
                    sections_text=sections_text,
                    transcript=transcript,
                )},
            ],
            response_format={"type": "json_object"},
            temperature=node_temp,
        )
        seg_data = json.loads(resp2.choices[0].message.content)
    except Exception as e:
        timing["transcript_segmentation"] = round(time.time() - t1, 2)
        return {"nodes": nodes, "edges": [], "kus": [], "timing": timing, "error": f"Segmentation failed: {e}"}
    timing["transcript_segmentation"] = round(time.time() - t1, 2)

 # Build segment index: node_id → segment_text
    raw_segments = seg_data.get("segments", [])
    segment_map = {
        seg["node_id"]: (seg.get("segment_text") or "").strip()
        for seg in raw_segments
        if seg.get("node_id") and seg.get("segment_text", "").strip()
    }

 # ---- Step 4: Grounded edge extraction from slide segments ----
    from experiments.extract_edges import _DEFAULT_EDGE_TYPES, _build_edge_types_section, _normalize_edge_from_to

    edge_types = config.get("edge_types", _DEFAULT_EDGE_TYPES)
    types_section = _build_edge_types_section(edge_types)

 # Build the segments_text block: each node with its transcript segment
    seg_lines = []
    for n in nodes:
        nid = n["id"]
        seg_text = segment_map.get(nid, "")
        seg_lines.append(
            f"[{nid}] {n['label']}\n"
            f"  Segment: {seg_text[:600] if seg_text else '(no segment text)'}"
        )
    segments_text = "\n\n".join(seg_lines)

    t2 = time.time()
    grounded_edges = []
    try:
        resp3 = client.chat.completions.create(
            model=strict_model,
            messages=[
                {"role": "system", "content": _SLIDE_GROUNDED_EDGE_SYSTEM},
                {"role": "user", "content": _SLIDE_GROUNDED_EDGE_USER.format(
                    segments_text=segments_text,
                    num_types=len(edge_types),
                    types_section=types_section,
                )},
            ],
            response_format={"type": "json_object"},
            temperature=strict_temp,
        )
        edge_data = json.loads(resp3.choices[0].message.content)
        node_ids = {n["id"] for n in nodes}
        for e in edge_data.get("edges", []):
            e = _normalize_edge_from_to(e)
            fr, to = e.get("from"), e.get("to")
            if fr and to and fr in node_ids and to in node_ids and fr != to:
                if e.get("edge_type") in set(edge_types):
                    grounded_edges.append(e)
    except Exception as e:
        timing["grounded_edge_extraction"] = round(time.time() - t2, 2)
 # Non-fatal continue with soft edges
    timing["grounded_edge_extraction"] = round(time.time() - t2, 2)

 # ---- Step 5: Soft edge extraction from full transcript ----
    from experiments.extract_edges import extract_edges_soft

    t3 = time.time()
    soft_edges = extract_edges_soft(transcript, nodes, grounded_edges, config)
 # Deduplicate: don't add soft edges that duplicate grounded ones
    grounded_keys = {
        (e.get("from"), e.get("to"), e.get("edge_type")) for e in grounded_edges
    }
    soft_edges = [
        e for e in soft_edges
        if (e.get("from"), e.get("to"), e.get("edge_type")) not in grounded_keys
    ]
    timing["soft_edge_extraction"] = round(time.time() - t3, 2)

 # Combine + assign edge IDs (Python, never LLM)
    all_edges = grounded_edges + soft_edges
    for i, e in enumerate(all_edges):
        e["edge_id"] = f"edge_{i + 1:03d}"

 # ---- Step 6: Refinement pass ----
    t4 = time.time()
    all_edges = refine_edges(transcript, nodes, all_edges, config)
    timing["refine_pass"] = round(time.time() - t4, 2)

    timing["total"] = round(sum(timing.values()), 2)
 # kus=[] this pipeline intentionally produces no KUs (that's the point)
    return {"nodes": nodes, "edges": all_edges, "kus": [], "timing": timing}


# Pipeline 6: slide_anchored
#
# Core design principle:
# Slides = ground truth minimum coverage guarantee.
# Every slide section must map to at least one node the slide is the
# traceability anchor and hallucination guard. If a concept doesn't appear
# on a slide, it cannot be a node (prevents transcript hallucination).
#
# Transcript = source of memorable content.
# The *label* of each node, its source_sentence, and its memorable insight
# all come from what the lecturer actually *said* not from the slide header.
# "Introduction to Objects" is NOT a node. The principle the lecturer
# expressed about objects IN the transcript is the node.
#
# KU problem avoided:
# KUs fragment the transcript into isolated sentences with no ground truth.
# Making KUs bigger (2-3 sentences) is just LLM segmentation with no anchor.
# Slides give free, structured, ground-truth segmentation instead.
#
# Flow:
# Step 1 (LLM) Parse slides → section list with IDs [slide_001, …]
# Step 2 (LLM) COMBINED call: full slide sections + full transcript →
# nodes with {label, slide_anchor_id, source_sentence}
# Step 3 (Python) Assign node_ids, validate anchors, attach slide_bullets
# Step 4 (LLM) Soft edge extraction from full transcript
# Step 5 (LLM) Refinement pass
#
# Node data structure includes slide_anchor_id for full traceability.
# Falls back to multi_stage_refined if no slide_text provided.

_SLIDE_ANCHORED_NODE_SYSTEM = load_prompt("node_anchored_system", """\
You are a Lecture Knowledge Graph builder.
Your task is to extract knowledge nodes from a lecture transcript using slide sections as anchors.

A NODE represents a reusable concept or explanatory idea introduced in the lecture.
Nodes capture ideas that help a student understand the conceptual structure
of the lecture — not transient dialogue.

DO NOT create nodes for:
    greetings, logistics, course administration, filler conversation,
    passing remarks or one-off examples with no conceptual reuse.

NODE TYPES:
    concept   — Any idea, term, principle, rule, theorem, mental model, analogy,
                or framework introduced in the lecture.
    example   — A canonical example used to illustrate a concept.

LABEL RULES:
• Prefer short noun phrases (e.g. "Type Safety", "Progress Theorem")
• Short statements are allowed when they capture a principle
• Avoid overly generic labels, punctuation like ⇒ + :

BACKBONE vs SUPPORT:
    is_backbone: true  — main conceptual argument, essential for following the lecture.
    is_backbone: false — supporting detail, sub-concept, or secondary illustration.
    Any node_type can be backbone.

HIERARCHY:
    parent_id = exact label of parent node if sub-concept, otherwise null.

GROUNDING:
- Every node anchored to one slide section (slide_anchor_id).
- source_sentence MUST be verbatim or near-verbatim from the transcript.
- Typically 1–6 nodes per section depending on density. Quality over quantity.

Return strict JSON only.\
""")

_SLIDE_ANCHORED_NODE_USER = load_prompt("node_anchored_user", """\
SLIDE SECTIONS (ground truth):
{slide_sections_text}

Each section contains:
  section_id           — use this as slide_anchor_id
  title                — the topic of the section
  core_focus           — the essential idea the lecturer wants students to grasp
  key_ideas            — specific concepts and facts to cover
  why_this_part_exists — pedagogical motivation for the section

------------------------------------
LECTURE TRANSCRIPT:
{transcript}
------------------------------------

TASK
Extract knowledge nodes grounded in the transcript.
Use the slide sections to determine where concepts belong.
The key_ideas are a coverage checklist, but nodes must be grounded in the transcript.

------------------------------------
OUTPUT FORMAT
Return JSON:
{{
  "nodes": [
    {{
      "label": "...",
      "node_type": "concept | example",
      "slide_anchor_id": "...",
      "source_sentence": "...",
      "is_backbone": true,
      "parent_id": null
    }}
  ]
}}

------------------------------------
RULES
1. slide_anchor_id MUST match a section_id listed above.
2. source_sentence MUST come verbatim or near-verbatim from the transcript.
3. Labels must follow the label rules in the system prompt.
4. node_type must be "concept" or "example".
5. is_backbone: true for main conceptual thread nodes, false for supporting nodes.
6. parent_id: exact label of parent node if sub-concept, otherwise null.
7. Do not invent concepts not mentioned in the transcript.
8. Prefer nodes that represent ideas the lecturer emphasises or repeats.

------------------------------------
EXTRACTION STRATEGY
1. Identify the core concept of the section (core_focus) — this is likely backbone.
2. Extract key technical terms introduced or explained in the transcript.
3. Capture reasoning, guarantees, or properties stated by the lecturer.
4. Capture mental models or analogies if present — these are concept nodes.
5. Capture canonical examples if they illustrate a concept — these are example nodes.
6. For each node: does this belong under another node? If yes, set parent_id.

Avoid duplicate nodes across sections unless the concept is newly explained.\
""")


@register_pipeline("slide_anchored")
def pipeline_slide_anchored(transcript: str, config):
    """Slide-anchored pipeline: slides as ground truth + transcript as full content source"""
    timing = {}
    slide_text = (config.get("slide_text") or "").strip()

    if not slide_text:
        import warnings
        warnings.warn("slide_anchored: no slide_text in config, falling back to multi_stage_refined.")
        return pipeline_multi_stage_refined(transcript, config)

    try:
        from openai import OpenAI
    except ImportError:
        return {"nodes": [], "edges": [], "kus": [], "timing": timing, "error": "openai not installed"}

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"nodes": [], "edges": [], "kus": [], "timing": timing, "error": "OPENAI_API_KEY not set"}

    client = OpenAI(api_key=api_key)
    model = config.get("node_model", "gpt-5.2")
    soft_model = config.get("soft_model", model)
    node_temp = config.get("node_temperature", 0.1)

 # ---- Step 1: Python extract PART IDs from slide notes (no LLM call) ----
 # Parses "PART I Title" lines to get stable slide_anchor_ids.
 # The FULL raw slide_text (Core Focus, Key Ideas, Why This Part Exists) goes
 # directly to the node extraction LLM nothing is parsed away.
    t0 = time.time()
    part_ids = _extract_part_ids(slide_text)
    timing["slide_parse"] = round(time.time() - t0, 3)   # near-zero, Python only

    if not part_ids:
        import warnings
        warnings.warn("slide_anchored: no PART headers found in slide_text, falling back to multi_stage_refined.")
        return pipeline_multi_stage_refined(transcript, config)

 # Upgrade to full section parse: gets core_focus, key_ideas, why_this_part_exists
    sections = _extract_part_sections(slide_text)
    if not sections:
 # _extract_part_ids found some IDs but _extract_part_sections returned nothing
 # treat part_ids as minimal sections
        sections = [{"section_id": p["slide_id"], "title": p["title"],
                     "core_focus": "", "key_ideas": [], "why_this_part_exists": ""}
                    for p in part_ids]

    valid_slide_ids = {s["section_id"] for s in sections}
    slide_title_map = {s["section_id"]: s["title"] for s in sections}

 # Build structured {slide_sections_text} for the prompt
    slide_sections_text = _format_slide_sections_text(sections)

 # ---- Step 2: COMBINED structured slide sections + transcript → nodes ----
    t1 = time.time()
    try:
        resp2 = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SLIDE_ANCHORED_NODE_SYSTEM},
                {"role": "user", "content": _SLIDE_ANCHORED_NODE_USER.format(
                    slide_sections_text=slide_sections_text,
                    transcript=transcript,
                )},
            ],
            response_format={"type": "json_object"},
            temperature=node_temp,
        )
        node_data = json.loads(resp2.choices[0].message.content)
    except Exception as e:
        timing["node_extraction"] = round(time.time() - t1, 2)
        return {"nodes": [], "edges": [], "kus": [], "timing": timing, "error": f"Node extraction failed: {e}"}
    timing["node_extraction"] = round(time.time() - t1, 2)

    raw_node_list = node_data.get("nodes", [])

 # ---- Step 3: Python validate anchors, assign node IDs, build node dicts ----
 # Also fuzzy-match sentence_index for each node
    transcript_sentences = _split_transcript_sentences(transcript)

    nodes = []
    for i, raw in enumerate(raw_node_list):
        label = (raw.get("label") or "").strip()
        anchor_id = (raw.get("slide_anchor_id") or "").strip()
        source_sentence = (raw.get("source_sentence") or "").strip()
        node_type = (raw.get("node_type") or "concept").strip()
        is_backbone = bool(raw.get("is_backbone", False))
        parent_label = (raw.get("parent_id") or "")  # LLM sends label text

        if not label:
            continue
 # Reject any node whose slide_anchor_id doesn't match hallucination guard
        if anchor_id not in valid_slide_ids:
            continue

        nid = f"node_{i + 1:03d}"
        nodes.append({
            "id": nid,
            "node_id": nid,
            "label": label,
            "node_type": node_type,
            "is_backbone": is_backbone,
            "_parent_label": parent_label,   # temp field, resolved below
            "slide_anchor_id": anchor_id,
            "slide_anchor_title": slide_title_map.get(anchor_id, ""),
            "source_sentence": source_sentence,
            "supporting_ku_ids": [],
        })

    if len(nodes) < 2:
        import warnings
        warnings.warn("slide_anchored: fewer than 2 valid nodes extracted, falling back to multi_stage_refined.")
        return pipeline_multi_stage_refined(transcript, config)

 # Fuzzy-compute sentence_index per node
    for n in nodes:
        n["sentence_index"] = _find_sentence_index(n["source_sentence"], transcript_sentences)

 # Resolve parent_id: label → node ID
    for n in nodes:
        parent_label = n.pop("_parent_label", "")
        if parent_label:
            resolved = _resolve_parent_id(parent_label, nodes)
            n["parent_id"] = resolved  # None if below threshold or no match
        else:
            n["parent_id"] = None

 # Compute section_order and lecture_order
    nodes = _compute_ordering(nodes)

 # ---- Step 4a: Edge extraction Pass 1 Grounded/Explicit ----
    from experiments.extract_edges import (
        _DEFAULT_EDGE_TYPES, _build_edge_types_section, _normalize_edge_from_to,
    )
    edge_model = config.get("strict_model", model)
    edge_temp = config.get("edge_temperature", 0.1)
    edge_types = config.get("edge_types", _DEFAULT_EDGE_TYPES)
    types_section = _build_edge_types_section(edge_types)
    edge_type_set = set(edge_types)

 # Include slide_anchor_id so LLM knows which section each node belongs to
    nodes_text = "\n".join(
        f"  {n['id']}: {n['label']}  [section: {n.get('slide_anchor_id', '?')}]"
        for n in nodes
    )
    section_ids_text = "\n".join(f"  {s['section_id']}" for s in sections)

    t2 = time.time()
    raw_pass1 = []
    try:
        resp_p1 = client.chat.completions.create(
            model=edge_model,
            messages=[
                {"role": "system", "content": _EDGE_PASS1_SYSTEM},
                {"role": "user", "content": _EDGE_PASS1_USER.format(
                    nodes_text=nodes_text,
                    section_ids=section_ids_text,
                    transcript=transcript,
                    num_types=len(edge_types),
                    types_section=types_section,
                )},
            ],
            response_format={"type": "json_object"},
            temperature=edge_temp,
        )
        raw_pass1 = json.loads(resp_p1.choices[0].message.content).get("edges", [])
    except Exception as e:
        timing["edge_pass1"] = round(time.time() - t2, 2)
        return {"nodes": nodes, "edges": [], "kus": [], "timing": timing, "error": f"Edge Pass 1 failed: {e}"}
    timing["edge_pass1"] = round(time.time() - t2, 2)

 # ---- Step 4b: Edge extraction Pass 2 Soft/Inferred ----
    pass1_edges_text = "\n".join(
        f"  {e.get('from')} → {e.get('edge_type')} → {e.get('to')}: {e.get('justification', '')[:80]}"
        for e in raw_pass1
    ) or "  (none)"

    t2b = time.time()
    raw_pass2 = []
    try:
        resp_p2 = client.chat.completions.create(
            model=config.get("soft_model", model),
            messages=[
                {"role": "system", "content": _EDGE_PASS2_SYSTEM},
                {"role": "user", "content": _EDGE_PASS2_USER.format(
                    nodes_text=nodes_text,
                    section_ids=section_ids_text,
                    pass1_edges_text=pass1_edges_text,
                    transcript=transcript,
                    num_types=len(edge_types),
                    types_section=types_section,
                )},
            ],
            response_format={"type": "json_object"},
            temperature=edge_temp,
        )
        raw_pass2 = json.loads(resp_p2.choices[0].message.content).get("edges", [])
    except Exception:
        raw_pass2 = []  # Pass 2 failure is non-fatal
    timing["edge_pass2"] = round(time.time() - t2b, 2)

 # ---- Step 5: Python validate, assign edge_source, build edge dicts ----
    node_ids = {n["id"] for n in nodes}
    edges = []

    def _build_edge_sa(raw_e, edge_source: str):
        """Validate and build a single edge dict. Returns None if invalid"""
        raw_e = _normalize_edge_from_to(raw_e)
        fr, to = raw_e.get("from"), raw_e.get("to")
        if not fr or not to or fr not in node_ids or to not in node_ids or fr == to:
            return None
        if raw_e.get("edge_type") not in edge_type_set:
            return None

        ev_section = raw_e.get("evidence_section", "")
        if ev_section not in valid_slide_ids:
            ev_section = None

        out = {
            "from": fr,
            "to": to,
            "edge_type": raw_e["edge_type"],
            "justification": raw_e.get("justification") or "",
            "reason": raw_e.get("reason") or "",
            "evidence_section": ev_section,
            "edge_source": edge_source,
            "confidence_score": None,
        }
        conf = raw_e.get("confidence_score")
        if conf is not None:
            try:
                out["confidence_score"] = max(0.0, min(1.0, float(conf)))
            except (TypeError, ValueError):
                pass
        return out

    seen_edges: set = set()
    for raw_e in raw_pass1:
        e = _build_edge_sa(raw_e, "explicit")
        if e:
            key = (e["from"], e["to"], e["edge_type"])
            rev_key = (e["to"], e["from"], e["edge_type"])
            if key not in seen_edges and rev_key not in seen_edges:
                seen_edges.add(key)
                edges.append(e)

    for raw_e in raw_pass2:
        e = _build_edge_sa(raw_e, "inferred")
        if e:
            key = (e["from"], e["to"], e["edge_type"])
            rev_key = (e["to"], e["from"], e["edge_type"])
            if key not in seen_edges and rev_key not in seen_edges:
                seen_edges.add(key)
                edges.append(e)

 # ---- Step 5b: Cross-PART direction validation ----
 # "drives" edges where source PART > target PART are likely reversed.
 # The LLM often confuses "A drives B" with "A is driven by B".
 # Fix: auto-swap direction for suspicious backward "drives" edges.
    node_part_num = {}
    for n in nodes:
        sid = n.get("slide_anchor_id", "")
        try:
            node_part_num[n["id"]] = int(sid.split("_")[-1]) if sid else 0
        except (ValueError, IndexError):
            node_part_num[n["id"]] = 0

    DIRECTION_SENSITIVE_RELS = {"drives"}  # relations where "later→earlier" is suspicious
    for e in edges:
        if e["edge_type"] not in DIRECTION_SENSITIVE_RELS:
            continue
        src_part = node_part_num.get(e["from"], 0)
        tgt_part = node_part_num.get(e["to"], 0)
        if src_part > 0 and tgt_part > 0 and src_part > tgt_part:
 # Swap direction: "A drives B" where A is later → flip to B drives A
            e["from"], e["to"] = e["to"], e["from"]
            e["_direction_swapped"] = True

    for i, e in enumerate(edges):
        e["edge_id"] = f"edge_{i + 1:03d}"

 # ---- Step 6 (optional): Enrichment pass ----
 # Adds description + why_it_matters to nodes, reason to edges.
 # Only runs when config["enrich_graph"] = True.
    if config.get("enrich_graph", False):
        t4 = time.time()
        enriched = enrich_graph(nodes, edges, transcript, config)
        nodes = enriched["nodes"]
        edges = enriched["edges"]
        timing["enrich_pass"] = round(time.time() - t4, 2)

    timing["total"] = round(sum(timing.values()), 2)
    return {"nodes": nodes, "edges": edges, "kus": [], "timing": timing}


# Pipeline 7: slide_anchored_full
#
# Extension of slide_anchored: BOTH nodes AND edges are extracted using
# slide structure + transcript together as dual evidence sources.
#
# Why slides for edges too?
# Slide structure encodes relationships explicitly:
# - Bullets within a section: sub-concepts of the section heading
# - Section ordering: pedagogical prerequisite / motivation flow
# - Co-occurrence in a section: concepts the lecturer grouped together
# These structural relationships are MORE reliable than transcript inference
# because they represent the lecturer's deliberately prepared connections.
# Transcript ALSO adds verbal explanations not visible from slides alone.
#
# Evidence source field on each edge:
# evidence_source="slide" → no confidence_score → solid line (strict-like)
# evidence_source="transcript" → confidence_score → dashed line (soft-like)
#
# Flow:
# Step 1 (LLM) Parse slides → section list
# Step 2 (LLM) COMBINED: slides + transcript → nodes (same as slide_anchored)
# Step 3 (LLM) COMBINED: nodes (with anchor bullets) + transcript → edges
# Step 4 (Python) Validate, assign edge IDs, split strict-like/soft-like
# Step 5 (LLM) Refine pass

_SLIDE_ANCHORED_FULL_EDGE_SYSTEM = load_prompt("edge_anchored_full_system", """\
You are a lecture knowledge graph edge extractor.

You are given a node list and the full lecture transcript.
Extract relationships between nodes. Evidence comes ONLY from the transcript.

Two edge types to extract in one pass:

  GROUNDED edge — justification is verbatim or near-verbatim from the transcript.
                  The relationship is explicitly stated. Do NOT include confidence_score.
                  These appear as solid lines in the knowledge graph.

  SOFT edge     — justification is interpretive: the relationship follows from the
                  lecture's logic and flow but is not stated word-for-word.
                  MUST include confidence_score 0.0–1.0.
                  These appear as dashed lines in the knowledge graph.

Return strict JSON only.\
""")

# Two-pass edge extraction prompts (Pass 1 = grounded, Pass 2 = soft)
_EDGE_PASS1_SYSTEM = load_prompt("edge_pass1_system", """\
You are a lecture knowledge graph edge extractor — Pass 1 (Grounded).

Your task is to extract ONLY relationships that are explicitly stated in the lecture transcript.
A relationship is explicit when the lecturer directly describes, connects, or asserts it in words.

GROUNDED edge rules:
• justification MUST be verbatim or near-verbatim from the transcript.
• Do NOT include confidence_score.
• Do NOT extract inferred or implied relationships — those belong in Pass 2.
• When in doubt whether a relationship is stated or inferred, leave it out.

Each edge MUST include:
    from, to, edge_type, justification, reason, evidence_section.

Return strict JSON only.\
""")

_EDGE_PASS1_USER = load_prompt("edge_pass1_user", """\
NODES:
{nodes_text}

VALID SECTION IDs (use one of these for evidence_section):
{section_ids}

FULL LECTURE TRANSCRIPT:
---
{transcript}
---

EDGE TYPES (use only these {num_types}):
{types_section}

------------------------------------
TASK — PASS 1: GROUNDED EDGES ONLY

Extract relationships that are EXPLICITLY STATED in the transcript.
The lecturer must have directly said, described, or asserted the relationship.

Rules:
1. Only use node IDs from the list above.
2. No self-loops (from ≠ to). No duplicate edges.
3. justification MUST be verbatim or near-verbatim from the transcript.
4. Do NOT include confidence_score — if you are uncertain, do not include the edge.
5. evidence_section MUST be one of the valid section IDs listed above.
6. reason explains WHY the relationship exists — write this yourself, do not quote the transcript.

------------------------------------
Return strict JSON:
{{
  "edges": [
    {{
      "from": "node_001",
      "to": "node_002",
      "edge_type": "requires",
      "justification": "near-verbatim sentence from transcript",
      "reason": "conceptual explanation of why this relationship holds",
      "evidence_section": "PART_3"
    }}
  ]
}}\
""")

_EDGE_PASS2_SYSTEM = load_prompt("edge_pass2_system", """\
You are a lecture knowledge graph edge extractor — Pass 2 (Soft/Inferred).

You are given a node list, the lecture transcript, and the edges already extracted in Pass 1.
Your task is to extract ONLY relationships that are implied by the lecture logic but NOT
explicitly stated in the transcript.

SOFT edge rules:
• justification is an interpretive paraphrase — it summarises the lecture reasoning.
• MUST include confidence_score (0.0–1.0) reflecting your certainty.
• Do NOT duplicate any edge already in the Pass 1 edges list.
• Do NOT extract relationships that are explicitly stated — those were handled in Pass 1.
• Only extract relationships where the conceptual connection adds meaningful insight.

Each edge MUST include:
    from, to, edge_type, justification, reason, evidence_section, confidence_score.

Return strict JSON only.\
""")

_EDGE_PASS2_USER = load_prompt("edge_pass2_user", """\
NODES:
{nodes_text}

VALID SECTION IDs (use one of these for evidence_section):
{section_ids}

EDGES ALREADY EXTRACTED (Pass 1 — do NOT duplicate these):
{pass1_edges_text}

FULL LECTURE TRANSCRIPT:
---
{transcript}
---

EDGE TYPES (use only these {num_types}):
{types_section}

------------------------------------
TASK — PASS 2: SOFT / INFERRED EDGES ONLY

Extract relationships that are IMPLIED by the lecture logic but not directly stated.
The relationship should follow naturally from how the lecture is structured or argued,
even if the lecturer never said it explicitly.

Rules:
1. Only use node IDs from the list above.
2. No self-loops (from ≠ to). No duplicate edges.
3. Do NOT duplicate any edge from the Pass 1 list above.
4. justification is an interpretive paraphrase — not a direct quote.
5. MUST include confidence_score for every edge (0.0–1.0).
   Use lower scores (0.5–0.7) for weaker inferences, higher (0.8–0.95) for strong ones.
6. evidence_section MUST be one of the valid section IDs listed above.
7. reason explains WHY the relationship exists — write this yourself.
8. Only extract edges that add meaningful conceptual insight. Prefer quality over quantity.

------------------------------------
Return strict JSON:
{{
  "edges": [
    {{
      "from": "node_001",
      "to": "node_003",
      "edge_type": "drives",
      "justification": "interpretive paraphrase from lecture reasoning",
      "reason": "conceptual explanation of why this relationship holds",
      "evidence_section": "PART_1",
      "confidence_score": 0.75
    }}
  ]
}}\
""")

_SLIDE_ANCHORED_FULL_EDGE_USER = load_prompt("edge_anchored_full_user", """\
NODES:
{nodes_text}

FULL LECTURE TRANSCRIPT:
---
{transcript}
---

EDGE TYPES (use only these {num_types}):
{types_section}

Extract ALL relationships between the nodes above.
Evidence must come from the TRANSCRIPT only.

Rules:
1. Only use node IDs from the list above.
2. No self-loops (from ≠ to). No duplicate edges.
3. GROUNDED edge: justification is verbatim or near-verbatim from transcript.
   Do NOT include confidence_score. Use when the relationship is explicitly stated.
4. SOFT edge: justification is an interpretive paraphrase. MUST include confidence_score.
   Use when the relationship is implied by the lecture's logic but not stated directly.
5. Prefer GROUNDED when possible. Add SOFT only for additional meaningful relationships.

Return strict JSON:
{{
  "edges": [
    {{
      "from": "node_001",
      "to": "node_002",
      "edge_type": "requires",
      "justification": "verbatim or near-verbatim from transcript"
    }},
    {{
      "from": "node_002",
      "to": "node_003",
      "edge_type": "explains",
      "justification": "interpretive paraphrase grounded in lecture flow",
      "confidence_score": 0.85
    }}
  ]
}}\
""")


@register_pipeline("slide_anchored_full")
def pipeline_slide_anchored_full(transcript: str, config):
    """Full slide+transcript pipeline: nodes AND edges both use slides + transcript"""
    timing = {}
    slide_text = (config.get("slide_text") or "").strip()

    if not slide_text:
        import warnings
        warnings.warn("slide_anchored_full: no slide_text, falling back to slide_anchored.")
        return pipeline_slide_anchored(transcript, config)

    try:
        from openai import OpenAI
    except ImportError:
        return {"nodes": [], "edges": [], "kus": [], "timing": timing, "error": "openai not installed"}

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"nodes": [], "edges": [], "kus": [], "timing": timing, "error": "OPENAI_API_KEY not set"}

    client = OpenAI(api_key=api_key)
    model = config.get("node_model", "gpt-5.2")
    edge_model = config.get("soft_model", model)
    node_temp = config.get("node_temperature", 0.1)
    edge_temp = config.get("soft_temperature", 0.2)

 # ---- Step 1: Python parse slide notes into structured sections (no LLM) ----
 # _extract_part_sections() pulls section_id, title, core_focus,
 # key_ideas, why_this_part_exists for each PART nothing lost.
    t0 = time.time()
    sections = _extract_part_sections(slide_text)
    timing["slide_parse"] = round(time.time() - t0, 3)   # near-zero, Python only

    if not sections:
        import warnings
        warnings.warn("slide_anchored_full: no PART headers found, falling back to slide_anchored.")
        return pipeline_slide_anchored(transcript, config)

    valid_slide_ids = {s["section_id"] for s in sections}
    slide_title_map = {s["section_id"]: s["title"] for s in sections}

 # Build structured slide_sections_text for the prompt
    slide_sections_text = _format_slide_sections_text(sections)

 # ---- Step 2: COMBINED nodes structured slide sections + transcript ----
    t1 = time.time()
    try:
        resp2 = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SLIDE_ANCHORED_NODE_SYSTEM},
                {"role": "user", "content": _SLIDE_ANCHORED_NODE_USER.format(
                    slide_sections_text=slide_sections_text,
                    transcript=transcript,
                )},
            ],
            response_format={"type": "json_object"},
            temperature=node_temp,
        )
        node_data = json.loads(resp2.choices[0].message.content)
    except Exception as e:
        timing["node_extraction"] = round(time.time() - t1, 2)
        return {"nodes": [], "edges": [], "kus": [], "timing": timing, "error": f"Node extraction failed: {e}"}
    timing["node_extraction"] = round(time.time() - t1, 2)

 # ---- Step 2b: Python assign IDs, new fields, post-process nodes ----
    transcript_sentences = _split_transcript_sentences(transcript)

    nodes = []
    raw_node_list = node_data.get("nodes", [])
    for i, raw in enumerate(raw_node_list):
        label = (raw.get("label") or "").strip()
        anchor_id = (raw.get("slide_anchor_id") or "").strip()
        source_sentence = (raw.get("source_sentence") or "").strip()
        node_type = (raw.get("node_type") or "concept").strip()
        is_backbone = bool(raw.get("is_backbone", False))
        parent_label = (raw.get("parent_id") or "")  # LLM sends label text

        if not label or anchor_id not in valid_slide_ids:
            continue
        nid = f"node_{i + 1:03d}"
        nodes.append({
            "id": nid,
            "node_id": nid,
            "label": label,
            "node_type": node_type,
            "slide_anchor_id": anchor_id,
            "slide_anchor_title": slide_title_map.get(anchor_id, ""),
            "source_sentence": source_sentence,
            "is_backbone": is_backbone,
            "_parent_label": parent_label,   # temp field, resolved below
            "supporting_ku_ids": [],
        })

    if len(nodes) < 2:
        import warnings
        warnings.warn("slide_anchored_full: fewer than 2 nodes, falling back to slide_anchored.")
        return pipeline_slide_anchored(transcript, config)

 # Fuzzy-compute sentence_index per node
    for n in nodes:
        n["sentence_index"] = _find_sentence_index(n["source_sentence"], transcript_sentences)

 # Resolve parent_id: label → node ID
    for n in nodes:
        parent_label = n.pop("_parent_label", "")
        if parent_label:
            resolved = _resolve_parent_id(parent_label, nodes)
            n["parent_id"] = resolved  # None if below threshold or no match
        else:
            n["parent_id"] = None

 # Compute section_order and lecture_order
    nodes = _compute_ordering(nodes)

 # ---- Step 3a: Edge extraction Pass 1 Grounded/Strict ----
    from experiments.extract_edges import (
        _DEFAULT_EDGE_TYPES, _build_edge_types_section, _normalize_edge_from_to,
    )
    edge_types = config.get("edge_types", _DEFAULT_EDGE_TYPES)
    types_section = _build_edge_types_section(edge_types)
    edge_type_set = set(edge_types)

 # Include slide_anchor_id so LLM knows which section each node belongs to
    nodes_text = "\n".join(
        f"  {n['id']}: {n['label']}  [section: {n.get('slide_anchor_id', '?')}]"
        for n in nodes
    )
    section_ids_text = "\n".join(f"  {s['section_id']}" for s in sections)

    t2 = time.time()
    raw_pass1 = []
    try:
        resp_p1 = client.chat.completions.create(
            model=edge_model,
            messages=[
                {"role": "system", "content": _EDGE_PASS1_SYSTEM},
                {"role": "user", "content": _EDGE_PASS1_USER.format(
                    nodes_text=nodes_text,
                    section_ids=section_ids_text,
                    transcript=transcript,
                    num_types=len(edge_types),
                    types_section=types_section,
                )},
            ],
            response_format={"type": "json_object"},
            temperature=edge_temp,
        )
        raw_pass1 = json.loads(resp_p1.choices[0].message.content).get("edges", [])
    except Exception as e:
        timing["edge_pass1"] = round(time.time() - t2, 2)
        return {"nodes": nodes, "edges": [], "kus": [], "timing": timing, "error": f"Edge Pass 1 failed: {e}"}
    timing["edge_pass1"] = round(time.time() - t2, 2)

 # ---- Step 3b: Edge extraction Pass 2 Soft/Inferred ----
 # Pass 1 edge summary for deduplication
    pass1_edges_text = "\n".join(
        f"  {e.get('from')} → {e.get('edge_type')} → {e.get('to')}: {e.get('justification', '')[:80]}"
        for e in raw_pass1
    ) or "  (none)"

    t2b = time.time()
    raw_pass2 = []
    try:
        resp_p2 = client.chat.completions.create(
            model=edge_model,
            messages=[
                {"role": "system", "content": _EDGE_PASS2_SYSTEM},
                {"role": "user", "content": _EDGE_PASS2_USER.format(
                    nodes_text=nodes_text,
                    section_ids=section_ids_text,
                    pass1_edges_text=pass1_edges_text,
                    transcript=transcript,
                    num_types=len(edge_types),
                    types_section=types_section,
                )},
            ],
            response_format={"type": "json_object"},
            temperature=edge_temp,
        )
        raw_pass2 = json.loads(resp_p2.choices[0].message.content).get("edges", [])
    except Exception as e:
        timing["edge_pass2"] = round(time.time() - t2b, 2)
 # Pass 2 failure is non-fatal continue with Pass 1 edges only
        raw_pass2 = []
    timing["edge_pass2"] = round(time.time() - t2b, 2)

 # ---- Step 4: Python validate, assign edge_source, build edge dicts ----
    node_ids = {n["id"] for n in nodes}
    valid_section_ids = {s["section_id"] for s in sections}
    edges = []

    def _build_edge(raw_e, edge_source: str) -> Any:
        """Validate and build a single edge dict. Returns None if invalid"""
        raw_e = _normalize_edge_from_to(raw_e)
        fr, to = raw_e.get("from"), raw_e.get("to")
        if not fr or not to or fr not in node_ids or to not in node_ids or fr == to:
            return None
        if raw_e.get("edge_type") not in edge_type_set:
            return None

        ev_section = raw_e.get("evidence_section", "")
        if ev_section not in valid_section_ids:
            ev_section = None  # invalid section ID → discard but keep edge (P2 mitigation)

        out = {
            "from": fr,
            "to": to,
            "edge_type": raw_e["edge_type"],
            "justification": raw_e.get("justification") or "",
            "reason": raw_e.get("reason") or "",
            "evidence_section": ev_section,
            "edge_source": edge_source,
            "confidence_score": None,
        }
        conf = raw_e.get("confidence_score")
        if conf is not None:
            try:
                out["confidence_score"] = max(0.0, min(1.0, float(conf)))
            except (TypeError, ValueError):
                pass
        return out

 # Dedup key: (from, to, edge_type)
    seen_edges: set = set()

    for raw_e in raw_pass1:
        e = _build_edge(raw_e, "explicit")
        if e:
            key = (e["from"], e["to"], e["edge_type"])
            if key not in seen_edges:
                seen_edges.add(key)
                edges.append(e)

    for raw_e in raw_pass2:
        e = _build_edge(raw_e, "inferred")
        if e:
            key = (e["from"], e["to"], e["edge_type"])
            if key not in seen_edges:
                seen_edges.add(key)
                edges.append(e)

    for i, e in enumerate(edges):
        e["edge_id"] = f"edge_{i + 1:03d}"

 # ---- Step 5 (optional): Enrichment pass ----
    if config.get("enrich_graph", False):
        t4 = time.time()
        enriched = enrich_graph(nodes, edges, transcript, config)
        nodes = enriched["nodes"]
        edges = enriched["edges"]
        timing["enrich_pass"] = round(time.time() - t4, 2)

    timing["total"] = round(sum(timing.values()), 2)
    return {"nodes": nodes, "edges": edges, "kus": [], "timing": timing}


# Segmented pipelines: extract per-PART mini-graphs, then merge

# Transcript segmentation helpers

def _segment_transcript_tfidf(
    transcript: str,
    sections,
):
    import re
    from collections import defaultdict

    lines = transcript.split("\n")
    paragraphs = []
    current_text_lines = []
    current_start = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r'^\d+:\d{2}(:\d{2})?$', stripped):
            if current_text_lines:
                paragraphs.append({
                    "text": " ".join(current_text_lines),
                    "line_start": current_start,
                    "line_end": i - 1,
                })
                current_text_lines = []
            current_start = i + 1
        elif stripped:
            if not current_text_lines:
                current_start = i
            current_text_lines.append(stripped)

    if current_text_lines:
        paragraphs.append({
            "text": " ".join(current_text_lines),
            "line_start": current_start,
            "line_end": len(lines) - 1,
        })

    if not paragraphs:
        return {sections[0]["section_id"]: transcript} if sections else {}

    part_docs = []
    for sec in sections:
        doc_parts = [sec.get("title", ""), sec.get("core_focus", "")]
        doc_parts.extend(sec.get("key_ideas", []))
        part_docs.append(" ".join(doc_parts))

    para_texts = [p["text"] for p in paragraphs]

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        all_docs = part_docs + para_texts
        tfidf_matrix = vectorizer.fit_transform(all_docs)
        part_vectors = tfidf_matrix[:len(part_docs)]
        para_vectors = tfidf_matrix[len(part_docs):]

        sim_matrix = cosine_similarity(para_vectors, part_vectors)
        assignments = np.argmax(sim_matrix, axis=1)
    except ImportError:
        assignments = _segment_by_keyword_overlap(paragraphs, sections)

    part_paragraphs = defaultdict(list)
    for i, part_idx in enumerate(assignments):
        sid = sections[int(part_idx)]["section_id"]
        part_paragraphs[sid].append(paragraphs[i]["text"])

    return {sid: "\n".join(texts) for sid, texts in part_paragraphs.items()}


def _segment_by_keyword_overlap(
    paragraphs,
    sections,
):
    """Fallback segmentation using keyword overlap (no sklearn)"""
    import re
    stopwords = {"the", "and", "for", "that", "this", "with", "are", "was", "from",
                 "have", "has", "been", "not", "but", "they", "their", "which", "how",
                 "can", "will", "may", "also", "into", "than", "more", "most", "such"}

    part_keywords = []
    for sec in sections:
        words = set()
        for text in [sec.get("title", ""), sec.get("core_focus", "")] + sec.get("key_ideas", []):
            words.update(re.findall(r'\b[a-zA-Z]{3,}\b', text.lower()))
        words -= stopwords
        part_keywords.append(words)

    assignments = []
    for para in paragraphs:
        para_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', para["text"].lower()))
        best_idx, best_score = 0, 0
        for j, kw in enumerate(part_keywords):
            score = len(para_words & kw)
            if score > best_score:
                best_score, best_idx = score, j
        assignments.append(best_idx)
    return assignments


def _segment_transcript_fuzzy(
    transcript: str,
    sections,
):
    import re
    from difflib import SequenceMatcher
    from collections import defaultdict

 # Parse paragraphs (same logic as TF-IDF)
    lines = transcript.split("\n")
    paragraphs = []
    current_text_lines = []
    current_start = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r'^\d+:\d{2}(:\d{2})?$', stripped):
            if current_text_lines:
                paragraphs.append({"text": " ".join(current_text_lines),
                                   "line_start": current_start, "line_end": i - 1})
                current_text_lines = []
            current_start = i + 1
        elif stripped:
            if not current_text_lines:
                current_start = i
            current_text_lines.append(stripped)

    if current_text_lines:
        paragraphs.append({"text": " ".join(current_text_lines),
                           "line_start": current_start, "line_end": len(lines) - 1})

    if not paragraphs:
        return {sections[0]["section_id"]: transcript} if sections else {}

 # Build reference texts per PART from key_ideas + core_focus
    part_refs = []
    for sec in sections:
        ref = " ".join([sec.get("title", ""), sec.get("core_focus", "")] + sec.get("key_ideas", []))
        part_refs.append(ref.lower())

 # Assign each paragraph to best-matching PART
    assignments = []
    for para in paragraphs:
        para_lower = para["text"].lower()
        best_idx, best_score = 0, 0.0
        for j, ref in enumerate(part_refs):
            score = SequenceMatcher(None, para_lower[:500], ref[:500]).ratio()
            if score > best_score:
                best_score, best_idx = score, j
        assignments.append(best_idx)

    part_paragraphs = defaultdict(list)
    for i, part_idx in enumerate(assignments):
        sid = sections[part_idx]["section_id"]
        part_paragraphs[sid].append(paragraphs[i]["text"])

    return {sid: "\n".join(texts) for sid, texts in part_paragraphs.items()}


def _segment_transcript_embedding(
    transcript: str,
    sections,
):
    import re
    from collections import defaultdict

 # Parse paragraphs
    lines = transcript.split("\n")
    paragraphs = []
    current_text_lines = []
    current_start = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r'^\d+:\d{2}(:\d{2})?$', stripped):
            if current_text_lines:
                paragraphs.append({"text": " ".join(current_text_lines),
                                   "line_start": current_start, "line_end": i - 1})
                current_text_lines = []
            current_start = i + 1
        elif stripped:
            if not current_text_lines:
                current_start = i
            current_text_lines.append(stripped)

    if current_text_lines:
        paragraphs.append({"text": " ".join(current_text_lines),
                           "line_start": current_start, "line_end": len(lines) - 1})

    if not paragraphs:
        return {sections[0]["section_id"]: transcript} if sections else {}

 # Build PART reference texts
    part_refs = []
    for sec in sections:
        ref = " ".join([sec.get("title", ""), sec.get("core_focus", "")] + sec.get("key_ideas", []))
        part_refs.append(ref)

    para_texts = [p["text"] for p in paragraphs]

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        model = SentenceTransformer("all-MiniLM-L6-v2")
        part_embeddings = model.encode(part_refs, show_progress_bar=False)
        para_embeddings = model.encode(para_texts, show_progress_bar=False)

 # Cosine similarity
 # Normalize
        part_norm = part_embeddings / np.linalg.norm(part_embeddings, axis=1, keepdims=True)
        para_norm = para_embeddings / np.linalg.norm(para_embeddings, axis=1, keepdims=True)
        sim_matrix = para_norm @ part_norm.T  # (n_para, n_parts)
        assignments = np.argmax(sim_matrix, axis=1)

    except ImportError:
 # Fallback to TF-IDF
        return _segment_transcript_tfidf(transcript, sections)

    part_paragraphs = defaultdict(list)
    for i, part_idx in enumerate(assignments):
        sid = sections[int(part_idx)]["section_id"]
        part_paragraphs[sid].append(paragraphs[i]["text"])

    return {sid: "\n".join(texts) for sid, texts in part_paragraphs.items()}


def _segment_transcript_llm(
    transcript: str,
    sections,
    client,
    model: str,
):
    """Segment transcript into PART regions using an LLM call"""
    import re

    sections_desc = "\n".join(f"  {s['section_id']}: {s['title']}" for s in sections)

    lines = transcript.split("\n")
    numbered_lines = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not re.match(r'^\d+:\d{2}(:\d{2})?$', stripped):
            numbered_lines.append(f"L{i}: {stripped}")

    if len(numbered_lines) > 300:
        step = max(1, len(numbered_lines) // 300)
        numbered_lines = numbered_lines[::step]

    transcript_sample = "\n".join(numbered_lines)

    prompt = f"""You are given a lecture transcript with line numbers and a list of slide sections.
Assign each section a START line number and END line number from the transcript.

SECTIONS:
{sections_desc}

TRANSCRIPT (with line numbers):
{transcript_sample}

Return JSON:
{{
  "segments": [
    {{"section_id": "part_001", "start_line": 50, "end_line": 120}},
    {{"section_id": "part_002", "start_line": 121, "end_line": 200}}
  ]
}}

Rules:
1. Segments must cover the entire transcript — no gaps, no overlaps.
2. Order must follow the lecture flow (sections appear in order).
3. Use the content topics to determine where each section starts and ends.
4. Return ONLY JSON."""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You segment lecture transcripts by topic. Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        result = json.loads(resp.choices[0].message.content)
        segments = result.get("segments", [])
    except Exception:
        segments = []

    if not segments:
        chunk_size = len(lines) // max(len(sections), 1)
        return {
            s["section_id"]: "\n".join(lines[i * chunk_size:(i + 1) * chunk_size])
            for i, s in enumerate(sections)
        }

    result_map = {}
    for seg in segments:
        sid = seg.get("section_id", "")
        start = seg.get("start_line", 0)
        end = seg.get("end_line", len(lines) - 1)
        result_map[sid] = "\n".join(lines[start:end + 1])
    return result_map


# Shared: extract mini-graph for a single PART

def _extract_mini_graph(
    section,
    transcript_chunk: str,
    client,
    config,
):
    """Extract nodes and within-PART edges for a single section"""
    model = config.get("node_model", "gpt-4.1")
    edge_model = config.get("strict_model", model)
    node_temp = config.get("node_temperature", 0.1)
    edge_temp = config.get("edge_temperature", 0.1)

    section_text = _format_slide_sections_text([section])
    section_id = section["section_id"]
    valid_slide_ids = {section_id}
    slide_title_map = {section_id: section["title"]}

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SLIDE_ANCHORED_NODE_SYSTEM},
                {"role": "user", "content": _SLIDE_ANCHORED_NODE_USER.format(
                    slide_sections_text=section_text,
                    transcript=transcript_chunk,
                )},
            ],
            response_format={"type": "json_object"},
            temperature=node_temp,
        )
        node_data = json.loads(resp.choices[0].message.content)
    except Exception:
        return {"nodes": [], "edges": []}

    raw_nodes = node_data.get("nodes", [])
    transcript_sentences = _split_transcript_sentences(transcript_chunk)

    nodes = []
    for i, raw in enumerate(raw_nodes):
        label = (raw.get("label") or "").strip()
        anchor_id = (raw.get("slide_anchor_id") or "").strip()
        if not label or anchor_id not in valid_slide_ids:
            continue
        nid = f"{section_id}_node_{i + 1:03d}"
        nodes.append({
            "id": nid, "node_id": nid, "label": label,
            "node_type": (raw.get("node_type") or "concept").strip(),
            "is_backbone": bool(raw.get("is_backbone", False)),
            "_parent_label": raw.get("parent_id") or "",
            "slide_anchor_id": anchor_id,
            "slide_anchor_title": slide_title_map.get(anchor_id, ""),
            "source_sentence": (raw.get("source_sentence") or "").strip(),
            "supporting_ku_ids": [],
        })

    if not nodes:
        return {"nodes": [], "edges": []}

    for n in nodes:
        n["sentence_index"] = _find_sentence_index(n["source_sentence"], transcript_sentences)
    for n in nodes:
        parent_label = n.pop("_parent_label", "")
        n["parent_id"] = _resolve_parent_id(parent_label, nodes) if parent_label else None

    from experiments.extract_edges import _DEFAULT_EDGE_TYPES, _build_edge_types_section, _normalize_edge_from_to

    edge_types = config.get("edge_types", _DEFAULT_EDGE_TYPES)
    types_section = _build_edge_types_section(edge_types)
    edge_type_set = set(edge_types)
    node_ids = {n["id"] for n in nodes}

    nodes_text = "\n".join(
        f"  {n['id']}: {n['label']}  [section: {n.get('slide_anchor_id', '?')}]"
        for n in nodes
    )

    try:
        resp_e = client.chat.completions.create(
            model=edge_model,
            messages=[
                {"role": "system", "content": _EDGE_PASS1_SYSTEM},
                {"role": "user", "content": _EDGE_PASS1_USER.format(
                    nodes_text=nodes_text,
                    section_ids=f"  {section_id}",
                    transcript=transcript_chunk,
                    num_types=len(edge_types),
                    types_section=types_section,
                )},
            ],
            response_format={"type": "json_object"},
            temperature=edge_temp,
        )
        raw_edges = json.loads(resp_e.choices[0].message.content).get("edges", [])
    except Exception:
        raw_edges = []

    edges = []
    for raw_e in raw_edges:
        raw_e = _normalize_edge_from_to(raw_e)
        fr, to = raw_e.get("from"), raw_e.get("to")
        if not fr or not to or fr not in node_ids or to not in node_ids or fr == to:
            continue
        if raw_e.get("edge_type") not in edge_type_set:
            continue
        edges.append({
            "from": fr, "to": to,
            "edge_type": raw_e["edge_type"],
            "justification": raw_e.get("justification", ""),
            "reason": raw_e.get("reason", ""),
            "evidence_section": section_id,
            "edge_source": "explicit",
            "confidence_score": None,
        })

    return {"nodes": nodes, "edges": edges}


# Merge strategies

def _merge_python_dedup(
    mini_graphs,
    sections,
):
    """Merge: concat + Python dedup (nodes with similar labels merged)"""
    from difflib import SequenceMatcher

    all_nodes = []
    all_edges = []
    for mg in mini_graphs:
        all_nodes.extend(mg.get("nodes", []))
        all_edges.extend(mg.get("edges", []))

    merged_nodes = []
    label_to_id = {}
    id_remap = {}

    for n in all_nodes:
        norm_label = n["label"].lower().strip()
        best_match_id, best_score = None, 0.0
        for existing_label, existing_id in label_to_id.items():
            score = SequenceMatcher(None, norm_label, existing_label).ratio()
            if score > best_score:
                best_score, best_match_id = score, existing_id

        if best_score >= 0.85 and best_match_id:
            id_remap[n["id"]] = best_match_id
        else:
            label_to_id[norm_label] = n["id"]
            id_remap[n["id"]] = n["id"]
            merged_nodes.append(n)

    merged_edges = []
    seen_edge_keys: set = set()
    for e in all_edges:
        fr = id_remap.get(e["from"], e["from"])
        to = id_remap.get(e["to"], e["to"])
        if fr == to:
            continue
        key = (fr, to, e["edge_type"])
        if key not in seen_edge_keys:
            seen_edge_keys.add(key)
            e_copy = dict(e)
            e_copy["from"], e_copy["to"] = fr, to
            merged_edges.append(e_copy)

    return {"nodes": merged_nodes, "edges": merged_edges}


def _add_cross_part_edges_llm(
    nodes, edges,
    transcript,  # str or None
    sections,
    client, config,
):
    """Add cross-PART edges via LLM. transcript=None means nodes-only"""
    from experiments.extract_edges import _DEFAULT_EDGE_TYPES, _build_edge_types_section

    model = config.get("soft_model", config.get("node_model", "gpt-4.1"))
    edge_types = config.get("edge_types", _DEFAULT_EDGE_TYPES)
    types_section = _build_edge_types_section(edge_types)
    edge_type_set = set(edge_types)
    node_ids = {n["id"] for n in nodes}

    nodes_by_part = {}
    for n in nodes:
        sid = n.get("slide_anchor_id", "unknown")
        nodes_by_part.setdefault(sid, []).append(f"{n['id']}: {n['label']}")

    nodes_grouped = "\n".join(
        f"\n[{sid}]\n" + "\n".join(f"  {item}" for item in items)
        for sid, items in sorted(nodes_by_part.items())
    )

    existing_edges_text = "\n".join(
        f"  {e['from']} -> {e['edge_type']} -> {e['to']}" for e in edges
    ) or "  (none)"

    transcript_block = f"\nFULL LECTURE TRANSCRIPT:\n---\n{transcript}\n---\n" if transcript else ""

    prompt = f"""You are a lecture knowledge graph edge extractor.

NODES (grouped by section):
{nodes_grouped}

EXISTING WITHIN-SECTION EDGES (do NOT duplicate):
{existing_edges_text}
{transcript_block}
EDGE TYPES ({len(edge_types)}):
{types_section}

TASK: Extract CROSS-SECTION edges — relationships between nodes in DIFFERENT sections.

Rules:
1. Only edges between nodes in DIFFERENT sections.
2. Do NOT duplicate existing edges.
3. Each edge: from, to, edge_type, justification, reason, confidence_score (0.0-1.0).
4. Focus on meaningful pedagogical connections.

Return JSON: {{"edges": [...]}}"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You extract cross-section edges for lecture knowledge graphs. Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        raw_cross = json.loads(resp.choices[0].message.content).get("edges", [])
    except Exception:
        raw_cross = []

    seen = {(e["from"], e["to"], e["edge_type"]) for e in edges}
    for raw_e in raw_cross:
        fr = raw_e.get("from") or raw_e.get("source", "")
        to = raw_e.get("to") or raw_e.get("target", "")
        etype = raw_e.get("edge_type", "")
        if not fr or not to or fr not in node_ids or to not in node_ids or fr == to:
            continue
        if etype not in edge_type_set:
            continue
        key = (fr, to, etype)
        if key in seen:
            continue
        seen.add(key)
        conf = raw_e.get("confidence_score")
        try:
            conf = max(0.0, min(1.0, float(conf))) if conf is not None else 0.7
        except (TypeError, ValueError):
            conf = 0.7
        edges.append({
            "from": fr, "to": to, "edge_type": etype,
            "justification": raw_e.get("justification", ""),
            "reason": raw_e.get("reason", ""),
            "evidence_section": "", "edge_source": "inferred",
            "confidence_score": conf,
        })

    return {"nodes": nodes, "edges": edges}


# segmented_* 변종 등록: seg_method × merge_strategy = 12개
_SEG_METHODS = {"tfidf": "tfidf", "llm": "llm", "fuzzy": "fuzzy", "emb": "embedding"}
for _alias, _method in _SEG_METHODS.items():
    for _merge in ("a", "b", "c"):
        def _make(m, mg):
            def _fn(transcript, config):
                return _pipeline_segmented(transcript, config, seg_method=m, merge_strategy=mg)
            return _fn
        register_pipeline(f"segmented_{_alias}_{_merge}")(_make(_method, _merge))


def _pipeline_segmented(
    transcript: str, config,
    seg_method: str = "tfidf", merge_strategy: str = "c",
):
    """Segmented pipeline: split transcript by PART -> mini-graphs -> merge"""
    timing = {}
    slide_text = (config.get("slide_text") or "").strip()
    if not slide_text:
        return {"nodes": [], "edges": [], "kus": [], "timing": timing, "error": "No slide_text"}

    try:
        from openai import OpenAI
    except ImportError:
        return {"nodes": [], "edges": [], "kus": [], "timing": timing, "error": "openai not installed"}

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"nodes": [], "edges": [], "kus": [], "timing": timing, "error": "OPENAI_API_KEY not set"}

    client = OpenAI(api_key=api_key)
    model = config.get("node_model", "gpt-4.1")

    sections = _extract_part_sections(slide_text)
    if not sections:
        return {"nodes": [], "edges": [], "kus": [], "timing": timing, "error": "No PART sections"}

 # ---- Step 2: Segment transcript ----
    t0 = time.time()
    if seg_method == "llm":
        transcript_segments = _segment_transcript_llm(transcript, sections, client, model)
    elif seg_method == "fuzzy":
        transcript_segments = _segment_transcript_fuzzy(transcript, sections)
    elif seg_method == "embedding":
        transcript_segments = _segment_transcript_embedding(transcript, sections)
    else:  # "tfidf"
        transcript_segments = _segment_transcript_tfidf(transcript, sections)
    timing["segmentation"] = round(time.time() - t0, 2)

 # ---- Step 3: Extract mini-graphs per PART ----
    t1 = time.time()
    mini_graphs = []
    for sec in sections:
        sid = sec["section_id"]
        chunk = transcript_segments.get(sid, "")
        if not chunk.strip():
            continue
        mg = _extract_mini_graph(sec, chunk, client, config)
        if mg["nodes"]:
            mini_graphs.append(mg)
    timing["mini_graph_extraction"] = round(time.time() - t1, 2)

    if not mini_graphs:
        return {"nodes": [], "edges": [], "kus": [], "timing": timing, "error": "No mini-graphs"}

 # ---- Step 4: Merge ----
    t2 = time.time()
    base = _merge_python_dedup(mini_graphs, sections)
    if merge_strategy == "a":
 # A: cross-PART edges via LLM WITH full transcript
        merged = _add_cross_part_edges_llm(base["nodes"], base["edges"], transcript, sections, client, config)
    elif merge_strategy == "b":
 # B: cross-PART edges via LLM WITHOUT transcript (nodes only)
        merged = _add_cross_part_edges_llm(base["nodes"], base["edges"], None, sections, client, config)
    else:
 # C: no cross-PART LLM call Python dedup only, cheapest
        merged = base
    timing["merge"] = round(time.time() - t2, 2)

    nodes = merged["nodes"]
    edges = merged["edges"]

 # ---- Step 5: Post-processing ----
    transcript_sentences = _split_transcript_sentences(transcript)
    for n in nodes:
        n["sentence_index"] = _find_sentence_index(n["source_sentence"], transcript_sentences)
    nodes = _compute_ordering(nodes)

    node_part_num = {}
    for n in nodes:
        sid = n.get("slide_anchor_id", "")
        try:
            node_part_num[n["id"]] = int(sid.split("_")[-1]) if sid else 0
        except (ValueError, IndexError):
            node_part_num[n["id"]] = 0

    for e in edges:
        if e.get("edge_type") == "drives":
            src_p = node_part_num.get(e["from"], 0)
            tgt_p = node_part_num.get(e["to"], 0)
            if src_p > 0 and tgt_p > 0 and src_p > tgt_p:
                e["from"], e["to"] = e["to"], e["from"]
                e["_direction_swapped"] = True

    for i, e in enumerate(edges):
        e["edge_id"] = f"edge_{i + 1:03d}"

    if config.get("enrich_graph", False):
        t3 = time.time()
        enriched = enrich_graph(nodes, edges, transcript, config)
        nodes, edges = enriched["nodes"], enriched["edges"]
        timing["enrich_pass"] = round(time.time() - t3, 2)

    timing["total"] = round(sum(timing.values()), 2)
    return {
        "nodes": nodes, "edges": edges, "kus": [], "timing": timing,
        "meta": {"seg_method": seg_method, "merge_strategy": merge_strategy,
                 "num_mini_graphs": len(mini_graphs), "num_sections": len(sections)},
    }
