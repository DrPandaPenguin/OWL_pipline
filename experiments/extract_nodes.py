# 강의 transcript -> Knowledge Unit -> Node 추출
# 흐름: Phase1(LLM, KU 추출) -> Glue1(파이썬, ID/timestamp) -> Phase3(LLM, KU 묶어 Node) -> Glue2(파이썬, Node ID + min timestamp)

import json
import re
import sys
import os
import time
from difflib import SequenceMatcher

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


# 타임스탬프 한 줄: m:ss 또는 h:mm:ss
_TIMESTAMP_ONLY_RE = re.compile(r"^\d{1,2}:\d{1,2}(:\d{1,2})?$")


# transcript을 줄 단위로 분리. 타임스탬프 줄은 다음 타임스탬프 나올 때까지 이어지는 모든 줄에 적용
def _split_into_sentences(text):
    lines = text.splitlines()
    out = []
    current_ts = None
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if _TIMESTAMP_ONLY_RE.match(line):
            current_ts = line
            continue
        out.append((len(out), line, current_ts))
    return out


# Phase 1: Knowledge Unit Extraction (LLM)
PHASE1_SYSTEM = _load_prompt("ku_extraction_system", """\
You are an information extraction system.

Your task is to extract Knowledge Units (KUs) from a lecture transcript.

A Knowledge Unit is a minimal, explicit piece of meaning that contributes to:
- a definition
- a claim or assertion
- an explanation
- an example used to support understanding
- a factual statement relevant to the lecture content

Exclude:
- pure filler, jokes, or conversational noise
- acknowledgements, greetings, or crowd interaction unless they convey meaning
- ambiguous references with no clear semantic content

Do NOT abstract, merge, or interpret.
Do NOT judge importance.
Prefer recall, but remove clearly meaningless utterances.

Return strict JSON only.\
""")

PHASE1_USER_TEMPLATE = _load_prompt("ku_extraction_user", """\
Extract Knowledge Units from the following lecture transcript.

Each Knowledge Unit should correspond to a single meaningful idea
explicitly stated in the transcript.
If two sentences express the same idea, extract them as separate KUs.

Extract EVERYTHING that could be a knowledge unit. Over-extract. Redundancy is acceptable.

Lecture transcript:
---
{transcript}
---

Output strict JSON with this exact structure:
{{
  "knowledge_units": [
    {{ "text": "exact or close quote from transcript" }},
    {{ "text": "..." }}
  ]
}}

Do NOT add IDs. Do NOT filter by importance. Extract ALL knowledge units.\
""")


# Phase 1: transcript에서 raw KU 뽑기 (LLM, recall 우선)
def extract_knowledge_units(transcript, config=None):
    if not transcript or not transcript.strip():
        return []
    client = _get_openai_client()
    if not client:
        return []
    cfg = config or {}
    model = cfg.get("ku_model", "gpt-5.2")
    temperature = cfg.get("ku_temperature", 0.1)
    system_prompt = cfg.get("ku_system_prompt") or PHASE1_SYSTEM
    user_template = cfg.get("ku_user_template") or PHASE1_USER_TEMPLATE
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_template.format(transcript=transcript)},
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
        )
        data = json.loads(response.choices[0].message.content)
        raw = data.get("knowledge_units", [])
        return [{"text": (u.get("text") or "").strip()} for u in raw if (u.get("text") or "").strip()]
    except Exception as e:
        print(f"Phase 1 error: {e}", file=sys.stderr)
        return []


# Glue 1: ID 부여 + timestamp 매핑

# KU text가 transcript의 어느 줄에서 왔는지 매칭 (정확히 포함 -> 안되면 fuzzy)
def _match_sentence_index(text, sentences, config=None):
    cfg = config or {}
    threshold = cfg.get("ku_match_threshold", 0.5)
    norm = text.lower().strip()
    if not norm:
        return 0
    for idx, sent, _ in sentences:
        if norm in sent.lower() or sent.lower() in norm:
            return idx
    best_idx, best_ratio = 0, 0.0
    for idx, sent, _ in sentences:
        r = SequenceMatcher(None, norm, sent.lower().strip()).ratio()
        if r > best_ratio:
            best_ratio, best_idx = r, idx
    return best_idx if best_ratio > threshold else 0


def add_ids_and_timestamps(knowledge_units, transcript, config=None):
    sentences = _split_into_sentences(transcript)
    out = []
    for i, ku in enumerate(knowledge_units):
        text = (ku.get("text") or "").strip()
        if not text:
            continue
        si = _match_sentence_index(text, sentences, config)
        ts = {"method": "B", "sentence_index": si}
        if si < len(sentences) and sentences[si][2] is not None:
            ts["time"] = sentences[si][2]
        out.append({
            "id": f"ku_{i + 1:03d}",
            "text": text,
            "sentence_index": si,
            "timestamp": ts,
        })
    return out


# Phase 3: Node construction (LLM)
PHASE3_SYSTEM = _load_prompt("node_construction_system", """\
You are a lecture structure analyst.

Your task is to identify Nodes: stable, memorable concepts
that a student should retain after the lecture.

A Node should represent:
- a core concept
- a key motivation
- a foundational definition
- a central example used to explain a concept

Do NOT create nodes for:
- administrative logistics
- jokes or anecdotes unless they serve a conceptual role
- timing, scheduling, or course management details

Each Node must be supported by one or more Knowledge Units.
Return strict JSON only.\
""")

PHASE3_USER_TEMPLATE = _load_prompt("node_construction_user", """\
Given the following Knowledge Units (each has an ID and text), group them into Nodes.
Each Node should have a short label and list of supporting_ku_ids (the KU IDs that support this node).

Knowledge Units (format: [id] text):
---
{ku_list}
---

Instructions:
- Only promote KUs to Nodes if they contribute to conceptual understanding.
- Prefer fewer, higher-quality nodes over many shallow ones.
- Labels should be short noun phrases (not summaries or sentences).
- If a KU does not fit any conceptual node, leave it unassigned.
Output strict JSON only:
{{
  "nodes": [
    {{ "label": "Short label for the node", "supporting_ku_ids": ["ku_001", "ku_005"] }},
    {{ "label": "...", "supporting_ku_ids": ["ku_002"] }}
  ]
}}\
""")


# Phase 3: KU들을 Node로 묶기 (LLM, precision 우선)
def construct_nodes(kus_with_ids, config=None):
    if not kus_with_ids:
        return []
    cfg = config or {}
    model = cfg.get("node_model", "gpt-5.2")
    temperature = cfg.get("node_temperature", 0.2)
    system_prompt = cfg.get("node_system_prompt") or PHASE3_SYSTEM
    user_template = cfg.get("node_user_template") or PHASE3_USER_TEMPLATE
    ku_list_str = "\n".join(f"  [{u['id']}] {u['text']}" for u in kus_with_ids)
    client = _get_openai_client()
    if not client:
        return []
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_template.format(ku_list=ku_list_str)},
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
        )
        data = json.loads(response.choices[0].message.content)
        return data.get("nodes", [])
    except Exception as e:
        print(f"Phase 3 error: {e}", file=sys.stderr)
        return []


# Glue 2: Node ID 부여 + timestamp = min(supporting KUs)
def process_nodes(nodes, kus_with_ids):
    ku_by_id = {u["id"]: u for u in kus_with_ids}
    valid = set(ku_by_id.keys())
    out = []
    cnt = 0
    for n in nodes:
        label = (n.get("label") or "").strip()
        supporting = [k for k in (n.get("supporting_ku_ids") or []) if k in valid]
        if not supporting:
            continue
        min_si = min(ku_by_id[k]["sentence_index"] for k in supporting)
        cnt += 1
        out.append({
            "id": f"node_{cnt:03d}",
            "label": label or f"Node {cnt}",
            "supporting_ku_ids": supporting,
            "timestamp": {"method": "B", "sentence_index": min_si},
        })
    return out


def _run_pipeline(transcript, verbose, config):
    """Phase 1 → Glue 1 → Phase 3 → Glue 2. Returns (kus_with_ids, nodes)."""
    if verbose:
        print("\n  [Phase 1] KU extraction (LLM)...", flush=True)
    t = time.time()
    raw_kus = extract_knowledge_units(transcript, config)
    if verbose:
        print(f"  → {len(raw_kus)} KUs in {time.time() - t:.2f}s", flush=True)

    if not raw_kus:
        return [], []

    if verbose:
        print("  [Glue 1] KU IDs + timestamps...", flush=True)
    t = time.time()
    kus_with_ids = add_ids_and_timestamps(raw_kus, transcript, config)
    if verbose:
        print(f"  → {len(kus_with_ids)} KUs in {time.time() - t:.2f}s", flush=True)

    if verbose:
        print("  [Phase 3] Node construction (LLM)...", flush=True)
    t = time.time()
    raw_nodes = construct_nodes(kus_with_ids, config)
    if verbose:
        print(f"  → {len(raw_nodes)} nodes in {time.time() - t:.2f}s", flush=True)

    if verbose:
        print("  [Glue 2] Node IDs + min timestamp...", flush=True)
    t = time.time()
    nodes = process_nodes(raw_nodes, kus_with_ids)
    if verbose:
        print(f"  → {len(nodes)} nodes in {time.time() - t:.2f}s", flush=True)

    for n in nodes:
        n["node_id"] = n["id"]
    return kus_with_ids, nodes


# 외부에서 쓰는 메인 진입점: transcript -> nodes
def extract_nodes(transcript, verbose=False, config=None):
    if not transcript or not transcript.strip():
        return []
    _, nodes = _run_pipeline(transcript, verbose, config)
    return nodes



def compute_orphan_kus(kus_with_ids, nodes):
    used = set()
    for n in nodes:
        for kid in n.get("supporting_ku_ids") or []:
            used.add(kid)
    return [u for u in kus_with_ids if u["id"] not in used]


# nodes 뿐 아니라 KU/orphan 정보까지 같이 반환 (Edge 추출 시 필요)
def extract_nodes_with_ku_metadata(transcript, verbose=False, config=None):
    empty = {
        "nodes": [], "knowledge_units": [], "orphan_kus": [],
        "total_ku": 0, "orphan_count": 0, "orphan_ratio": 0.0,
    }
    if not transcript or not transcript.strip():
        return empty

    kus_with_ids, nodes = _run_pipeline(transcript, verbose, config)
    if not kus_with_ids:
        return empty

    orphans = compute_orphan_kus(kus_with_ids, nodes)
    total = len(kus_with_ids)
    return {
        "nodes": nodes,
        "knowledge_units": kus_with_ids,
        "orphan_kus": orphans,
        "total_ku": total,
        "orphan_count": len(orphans),
        "orphan_ratio": (len(orphans) / total) if total > 0 else 0.0,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_nodes.py <transcript> [--save-ku]", file=sys.stderr)
        sys.exit(1)
    path = sys.argv[1]
    save_ku = "--save-ku" in sys.argv
    with open(path, "r", encoding="utf-8") as f:
        transcript = f.read()

    if save_ku:
        raw_kus = extract_knowledge_units(transcript)
        kus_with_ids = add_ids_and_timestamps(raw_kus, transcript)
        with open("knowledge_units.json", "w", encoding="utf-8") as f:
            json.dump({"knowledge_units": kus_with_ids}, f, indent=2, ensure_ascii=False)
        print("Saved knowledge_units.json with", len(kus_with_ids), "KUs")
        raw_nodes = construct_nodes(kus_with_ids)
        nodes = process_nodes(raw_nodes, kus_with_ids)
        for n in nodes:
            n["node_id"] = n["id"]
        with open("nodes.json", "w", encoding="utf-8") as f:
            json.dump({"nodes": nodes}, f, indent=2, ensure_ascii=False)
        print("Saved nodes.json with", len(nodes), "nodes")
    else:
        nodes = extract_nodes(transcript, verbose=True)
        with open("nodes.json", "w", encoding="utf-8") as f:
            json.dump({"nodes": nodes}, f, indent=2, ensure_ascii=False)
        print("Saved nodes.json with", len(nodes), "nodes")
