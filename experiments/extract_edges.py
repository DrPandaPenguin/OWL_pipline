# Edge 추출 (2-pass 구조)
# - Strict: KU에 있는 문장을 근거로 한 grounded edge
# - Soft: transcript 전체를 보고 추론한 interpretive edge (confidence_score 0~1)
# - Refine: 위 결과를 LLM이 한 번 더 review해서 add/remove
import json
import sys
import os
import re
import time

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

try:
    from extract_nodes import extract_nodes_with_ku_metadata
    EXTRACT_NODES_AVAILABLE = True
except ImportError:
    EXTRACT_NODES_AVAILABLE = False


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


def _create_strict_edge_prompt(knowledge_units, nodes, edge_types=None):
    ku_text = "\n".join(
        f"  [{ku.get('id', '?')}] {(ku.get('text') or '').strip()}"
        for ku in knowledge_units
        if ku.get("id") and (ku.get("text") or "").strip()
    )
    nodes_text = "\n".join(
        f"  - {_node_id(n)}: {n.get('label', '')}" for n in nodes if _node_id(n)
    )
    types_section = _build_edge_types_section(edge_types)
    num_types = len(edge_types) if edge_types else len(_DEFAULT_EDGE_TYPES)

    return f"""You are extracting STRICT edges from Knowledge Units (KUs) and nodes.
Each edge MUST be justified by one of the KU texts below. Use ONLY the Node IDs listed.

Knowledge Units (use these texts as evidence only; do NOT output KU IDs):
---
{ku_text}
---

Available nodes (use these Node IDs: from, to):
{nodes_text}

Task: Extract STRICT edges. Each justification_sentence MUST be an EXACT or NEAR-EXACT copy of one of the KU texts above.

ATOMIC EDGE TYPES (use only these {num_types})

{types_section}

RULES
1. ONLY create edges between nodes in the list above. Use exact Node IDs.
2. justification_sentence MUST be copied from one of the KU texts (exact or near-exact).
3. One KU text per edge. Do NOT invent sentences.
4. Do NOT hallucinate nodes or relationships.

OUTPUT — strict JSON only:

{{
  "edges": [
    {{
      "from": "node_001",
      "to": "node_002",
      "edge_type": "requires",
      "justification_sentence": "exact or near-exact copy of a KU text above"
    }}
  ]
}}

Return ONLY valid JSON. Use "from" and "to" with Node IDs."""


def _create_soft_edge_prompt(transcript, nodes, strict_edges, edge_types=None):
    nodes_text = "\n".join(
        f"  - {_node_id(n)}: {n.get('label', '')}" for n in nodes if _node_id(n)
    )

    strict_text = ""
    if strict_edges:
        strict_text = "\n\nAlready extracted STRICT edges (do NOT duplicate):\n"
        for e in strict_edges[:10]:
            fr = e.get("from") or e.get("source_node")
            to = e.get("to") or e.get("target_node")
            strict_text += f"  - {fr} --{e.get('edge_type')}--> {to}\n"
        if len(strict_edges) > 10:
            strict_text += f"  ... and {len(strict_edges) - 10} more\n"

    types_section = _build_edge_types_section(edge_types)
    num_types = len(edge_types) if edge_types else len(_DEFAULT_EDGE_TYPES)

    return f"""You are analyzing a lecture transcript to extract SOFT edges based on logic and flow.
Use ONLY the Node IDs listed below. Justification may span multiple sentences or a short paragraph.

Available nodes (use these Node IDs only):
{nodes_text}
{strict_text}

Lecture transcript:
---
{transcript}
---

Task: Extract SOFT edges (interpretive, multi-sentence reasoning). Do NOT duplicate strict edges.

ATOMIC EDGE TYPES (use only these {num_types})

{types_section}

RULES
1. ONLY create edges between nodes in the list above. Use exact Node IDs (from, to).
2. Use justification_span (not justification_sentence). May span multiple sentences.
3. Assign confidence_score: a float 0.0 to 1.0 (1.0 = high confidence).
4. Do NOT duplicate edges already listed as STRICT above.

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


def _split_into_sentences(text):
    sents = re.split(r'(?<=[.!?])\s+', text)
    return [(i, s.strip()) for i, s in enumerate(sents) if s.strip()]


def _normalize_edge_from_to(edge):
    e = dict(edge)
    if "from" not in e and "source_node" in e:
        e["from"] = e["source_node"]
    if "to" not in e and "target_node" in e:
        e["to"] = e["target_node"]
    return e


def _sentence_similarity(a, b):
    aw = set(a.lower().split())
    bw = set(b.lower().split())
    if not aw:
        return 0.0
    union = aw | bw
    if not union:
        return 0.0
    return len(aw & bw) / len(union)


# Strict edge 검증: justification_sentence가 실제 KU text에 (거의) 그대로 있어야 통과
def _validate_strict_edge(edge, node_ids, ku_texts, config=None):
    cfg = config or {}
    fr = edge.get("from") or edge.get("source_node")
    to = edge.get("to") or edge.get("target_node")
    if not fr or not to or fr not in node_ids or to not in node_ids:
        return False
    if edge.get("edge_type") not in set(cfg.get("edge_types", _DEFAULT_EDGE_TYPES)):
        return False
    threshold = cfg.get("strict_similarity_threshold", 0.8)
    just = edge.get("justification_sentence", "").strip()
    if not just:
        return False
    just_l = just.lower()
    for kt in ku_texts:
        if not kt.strip():
            continue
        kt_l = kt.lower()
        if just_l in kt_l or kt_l in just_l:
            return True
        if _sentence_similarity(just, kt) > threshold:
            return True
    return False


# Soft edge 검증: justification_span이 transcript에 어느 정도 단어 겹치면 통과
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


def _extract_strict_edges_with_llm(knowledge_units, nodes, config=None):
    cfg = config or {}
    if not nodes or len(nodes) < 2 or not knowledge_units:
        return []
    if not _get_openai_client():
        print("Error: OpenAI client unavailable. Set OPENAI_API_KEY.", file=sys.stderr)
        return []

    ku_texts = [(ku.get("text") or "").strip() for ku in knowledge_units
                if (ku.get("text") or "").strip()]
    if not ku_texts:
        return []

    edge_types = cfg.get("edge_types", _DEFAULT_EDGE_TYPES)
    edges = _call_llm_for_edges(
        _create_strict_edge_prompt(knowledge_units, nodes, edge_types),
        cfg.get("strict_system_prompt") or
        "You extract strict edges from Knowledge Units and nodes. justification_sentence MUST be an exact or near-exact copy of a KU text. Return strict JSON only.",
        cfg.get("strict_model", "gpt-5.2"),
        cfg.get("strict_temperature", 0.1),
    )

    node_ids = {_node_id(n) for n in nodes if _node_id(n)}
    out = []
    for i, raw in enumerate(edges):
        e = _normalize_edge_from_to(raw)
        if not _validate_strict_edge(e, node_ids, ku_texts, cfg):
            continue
        e.setdefault("edge_id", f"edge_{i + 1:03d}")
        e.setdefault("from", e.get("source_node"))
        e.setdefault("to", e.get("target_node"))
        out.append(e)
    return out


def _extract_soft_edges_with_llm(transcript, nodes, strict_edges, config=None):
    cfg = config or {}
    if not nodes or len(nodes) < 2:
        return []
    if not _get_openai_client():
        return []

    edge_types = cfg.get("edge_types", _DEFAULT_EDGE_TYPES)
    default_conf = cfg.get("soft_default_confidence", 0.7)

    edges = _call_llm_for_edges(
        _create_soft_edge_prompt(transcript, nodes, strict_edges, edge_types),
        cfg.get("soft_system_prompt") or
        "You are a knowledge extraction system that identifies interpretive pedagogical relationships. Extract relationships that may require multi-sentence reasoning. Return strict JSON only.",
        cfg.get("soft_model", "gpt-5.2"),
        cfg.get("soft_temperature", 0.2),
    )

    node_ids = {_node_id(n) for n in nodes if _node_id(n)}
    out = []
    for i, raw in enumerate(edges):
        e = _normalize_edge_from_to(raw)
        if not _validate_soft_edge(e, node_ids, transcript, cfg):
            continue
        e.setdefault("edge_id", f"edge_soft_{i + 1:03d}")
        e.setdefault("from", e.get("source_node"))
        e.setdefault("to", e.get("target_node"))
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


def extract_edges_strict(knowledge_units, nodes, config=None):
    if not knowledge_units or not nodes or len(nodes) < 2:
        return []
    return _dedupe_edges(_extract_strict_edges_with_llm(knowledge_units, nodes, config))


def extract_edges_soft(transcript, nodes, strict_edges, config=None):
    if not transcript or not transcript.strip() or not nodes or len(nodes) < 2:
        return []
    return _dedupe_edges(_extract_soft_edges_with_llm(transcript, nodes, strict_edges, config))


# 메인 진입점: Strict + Soft 다 돌리고 합쳐서 반환
def extract_edges(transcript, nodes, knowledge_units=None, include_soft=True, config=None):
    cfg = config or {}
    if not nodes or len(nodes) < 2:
        return []

    do_strict = cfg.get("include_strict", True)
    do_soft = cfg.get("include_soft", include_soft)

    strict_edges = (extract_edges_strict(knowledge_units, nodes, config)
                    if (knowledge_units and do_strict) else [])

    soft_edges = []
    if do_soft:
        strict_keys = {
            (e.get("from") or e.get("source_node"),
             e.get("to") or e.get("target_node"),
             e.get("edge_type"))
            for e in strict_edges
        }
        soft_raw = extract_edges_soft(transcript, nodes, strict_edges, config)
        soft_edges = [
            e for e in soft_raw
            if (e.get("from") or e.get("source_node"),
                e.get("to") or e.get("target_node"),
                e.get("edge_type")) not in strict_keys
        ]

    combined = strict_edges + soft_edges
    for i, e in enumerate(combined):
        e["edge_id"] = f"edge_{i + 1:03d}"
    return combined



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_edges.py <transcript> [--strict-only] [--nodes-file <file>]",
              file=sys.stderr)
        sys.exit(1)

    transcript_file = sys.argv[1]
    strict_only = "--strict-only" in sys.argv
    nodes_json_file = None
    if "--nodes-file" in sys.argv:
        idx = sys.argv.index("--nodes-file")
        if idx + 1 < len(sys.argv):
            nodes_json_file = sys.argv[idx + 1]

    print("=" * 70, file=sys.stderr)
    print("  Edge Extraction Pipeline", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"  Transcript: {transcript_file}", file=sys.stderr)
    print(f"  Nodes: {nodes_json_file or '자동 추출됨'}", file=sys.stderr)
    print("=" * 70 + "\n", file=sys.stderr)

    print("→ 트랜스크립트 읽는 중...", file=sys.stderr, end='', flush=True)
    try:
        with open(transcript_file, 'r', encoding='utf-8') as f:
            transcript = f.read()
        print(f" 완료 ({len(transcript):,}자)", file=sys.stderr)
    except (FileNotFoundError, OSError) as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    node_extraction_time = 0
    if nodes_json_file:
        knowledge_units = []
        try:
            with open(nodes_json_file, 'r', encoding='utf-8') as f:
                nodes = json.load(f).get('nodes', [])
            print(f"노드 파일 로드: {len(nodes)}개", file=sys.stderr)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        if not EXTRACT_NODES_AVAILABLE:
            print("Error: extract_nodes 모듈 import 실패.", file=sys.stderr)
            sys.exit(1)
        print("\n--- STEP 1: 노드 추출 ---\n", file=sys.stderr)
        t0 = time.time()
        meta = extract_nodes_with_ku_metadata(transcript, verbose=True)
        nodes = meta["nodes"]
        knowledge_units = meta.get("knowledge_units") or []
        node_extraction_time = time.time() - t0
        if not nodes:
            print("Error: 노드를 추출하지 못했습니다.", file=sys.stderr)
            sys.exit(1)
        with open("nodes.json", 'w', encoding='utf-8') as f:
            json.dump({"nodes": nodes}, f, indent=2, ensure_ascii=False)
        print(f"\n✓ 노드 {len(nodes)}, KU {len(knowledge_units)} ({node_extraction_time:.2f}s)",
              file=sys.stderr)

    print("\n--- STEP 2: 엣지 추출 ---\n", file=sys.stderr)
    t1 = time.time()
    if strict_only:
        edges = extract_edges_strict(knowledge_units, nodes)
        print("STRICT only", file=sys.stderr)
    else:
        edges = extract_edges(transcript, nodes,
                              knowledge_units=knowledge_units or [],
                              include_soft=True)
        sc = sum(1 for e in edges if "justification_sentence" in e)
        so = sum(1 for e in edges
                 if "justification_span" in e or e.get("confidence_score") is not None)
        print(f"strict {sc} + soft {so}", file=sys.stderr)
    edge_extraction_time = time.time() - t1

    output = {"edges": edges}
    with open("edges.json", 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n=== 완료 ===", file=sys.stderr)
    print(f"  노드: {len(nodes)}", file=sys.stderr)
    print(f"  엣지: {len(edges)}", file=sys.stderr)
    if not nodes_json_file:
        print(f"  노드 추출: {node_extraction_time:.2f}s", file=sys.stderr)
    print(f"  엣지 추출: {edge_extraction_time:.2f}s", file=sys.stderr)

    print(json.dumps(output, indent=2))
