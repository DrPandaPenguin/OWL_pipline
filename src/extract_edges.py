# edge 추출 (soft pass). slide_anchored은 자체 edge 로직 사용, direct가 이걸 호출
import json
import sys
import os

try:
    from src.prompt_loader import load_prompt as _load_prompt
except ImportError:
    try:
        from prompt_loader import load_prompt as _load_prompt
    except ImportError:
        def _load_prompt(name, default=""):
            return default

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai library not installed.", file=sys.stderr)


def _get_openai_client():
    if not OPENAI_AVAILABLE:
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def _node_id(node):
    return node.get("id") or node.get("node_id") or ""


# 7개 atomic edge type (논문에서 정의한 분류)
_EDGE_TYPE_DESCRIPTIONS = {
    "defines":    "A gives the definition or formal meaning of B",
    "requires":   "A cannot be understood or used without B",
    "explains":   "A provides a causal or mechanistic explanation for B",
    "details":    "A adds detail, refinement, or clarification to B",
    "example_of": "A is presented as an example or instance of B",
    "contrasts":  "A is explicitly compared against B to highlight differences",
    "drives":     "A provides a reason, purpose, or rationale for introducing B",
    "precedes":   "A is presented before B in the lecture narrative",
    "summarizes": "A provides a summary or recap of B",
}

_DEFAULT_EDGE_TYPES = ["defines", "requires", "explains", "details",
                       "example_of", "contrasts", "drives"]


def _build_edge_types_section(edge_types=None):
    types = edge_types or _DEFAULT_EDGE_TYPES
    return "\n".join(
        f"{i+1}. **{t}** - {_EDGE_TYPE_DESCRIPTIONS.get(t, t)}"
        for i, t in enumerate(types)
    )


def _create_soft_edge_prompt(transcript, nodes, edge_types=None):
    nodes_text = "\n".join(
        f"  - {_node_id(n)}: {n.get('label', '')}" for n in nodes if _node_id(n)
    )
    types_section = _build_edge_types_section(edge_types)
    num_types = len(edge_types) if edge_types else len(_DEFAULT_EDGE_TYPES)

    return f"""You are analyzing a lecture transcript to extract edges based on logic and flow.
Use ONLY the Node IDs listed below. Justification may span multiple sentences or a short paragraph.

Available nodes (use these Node IDs only):
{nodes_text}

Lecture transcript:
---
{transcript}
---

Task: Extract edges (interpretive, multi-sentence reasoning).

ATOMIC EDGE TYPES (use only these {num_types})

{types_section}

RULES
1. ONLY create edges between nodes in the list above. Use exact Node IDs (from, to).
2. Use justification_span. May span multiple sentences.
3. Assign confidence_score: a float 0.0 to 1.0 (1.0 = high confidence).

OUTPUT — strict JSON only:

{{
  "edges": [
    {{
      "from": "node_001",
      "to": "node_002",
      "edge_type": "explains",
      "justification_span": "Description of evidence from transcript...",
      "confidence_score": 0.85
    }}
  ]
}}

Return ONLY valid JSON. Use "from" and "to" with Node IDs. confidence_score must be 0.0~1.0."""


def _validate_soft_edge(edge, node_ids, transcript, config=None):
    cfg = config or {}
    fr = edge.get("from") or edge.get("source_node")
    to = edge.get("to") or edge.get("target_node")
    if not fr or not to or fr not in node_ids or to not in node_ids:
        return False
    if edge.get("edge_type") not in set(cfg.get("edge_types", _DEFAULT_EDGE_TYPES)):
        return False
    span = edge.get("justification_span", "").strip()
    if not span:
        return False
    conf = edge.get("confidence_score")
    if conf is not None:
        try:
            c = float(conf)
            if not (0 <= c <= 1):
                return False
        except (TypeError, ValueError):
            pass
    if any(k in span.lower() for k in
           ["sentence", "sentences", "paragraph", "describes", "discusses", "explains"]):
        return True
    sw = set(span.lower().split())
    tw = set(transcript.lower().split())
    return len(sw) > 0 and len(sw & tw) >= 3


def _call_llm_for_edges(prompt, system_prompt, model, temperature):
    """Single LLM call returning parsed list of edges. Returns [] on failure."""
    client = _get_openai_client()
    if not client:
        return []
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
        )
        return json.loads(response.choices[0].message.content).get("edges", [])
    except json.JSONDecodeError as e:
        print(f"LLM JSON parse error: {e}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"OpenAI error: {e}", file=sys.stderr)
        return []


def _extract_soft_edges_with_llm(transcript, nodes, config=None):
    cfg = config or {}
    if not nodes or len(nodes) < 2:
        return []
    if not _get_openai_client():
        return []

    edge_types = cfg.get("edge_types", _DEFAULT_EDGE_TYPES)
    default_conf = cfg.get("soft_default_confidence", 0.7)

    edges = _call_llm_for_edges(
        _create_soft_edge_prompt(transcript, nodes, edge_types),
        cfg.get("soft_system_prompt") or
        "You are a knowledge extraction system that identifies interpretive pedagogical relationships. Extract relationships that may require multi-sentence reasoning. Return strict JSON only.",
        cfg.get("soft_model", "gpt-5.2"),
        cfg.get("soft_temperature", 0.2),
    )

    node_ids = {_node_id(n) for n in nodes if _node_id(n)}
    out = []
    for i, raw in enumerate(edges):
        e = dict(raw)
        if "from" not in e and "source_node" in e:
            e["from"] = e["source_node"]
        if "to" not in e and "target_node" in e:
            e["to"] = e["target_node"]
        if not _validate_soft_edge(e, node_ids, transcript, cfg):
            continue
        e.setdefault("edge_id", f"edge_{i + 1:03d}")
        if e.get("confidence_score") is None:
            e["confidence_score"] = default_conf
        else:
            try:
                e["confidence_score"] = float(e["confidence_score"])
            except (TypeError, ValueError):
                e["confidence_score"] = default_conf
        out.append(e)
    return out


def _dedupe_edges(edges):
    seen = set()
    out = []
    for e in edges:
        fr = e.get("from") or e.get("source_node")
        to = e.get("to") or e.get("target_node")
        key = (fr, to, e.get("edge_type"))
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


def extract_edges_soft(transcript, nodes, config=None):
    if not transcript or not transcript.strip() or not nodes or len(nodes) < 2:
        return []
    return _dedupe_edges(_extract_soft_edges_with_llm(transcript, nodes, config))


# 메인 진입점 (direct 파이프라인이 호출)
def extract_edges(transcript, nodes, knowledge_units=None, include_soft=True, config=None):
    if not nodes or len(nodes) < 2:
        return []
    edges = extract_edges_soft(transcript, nodes, config) if include_soft else []
    for i, e in enumerate(edges):
        e["edge_id"] = f"edge_{i + 1:03d}"
    return edges
