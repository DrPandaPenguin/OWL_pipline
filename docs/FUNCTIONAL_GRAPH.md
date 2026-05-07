# Functional Graph — Pipeline Flow

OWL Pipeline의 핵심 production 파이프라인 (`slide_anchored`)의 함수 호출 흐름.

---

## 1. Top-level entry

```
run_graph.py (CLI)
    │
    └─→ run(lec_dir, pipeline_name="slide_anchored", enrich=True)
            │
            ├─→ get_pipeline("slide_anchored")  # src/pipelines.py 등록기에서 lookup
            │
            └─→ pipeline_slide_anchored(transcript, config)
                    │
                    └─→ KG dict 반환
```

```
dash_app.py (Web UI)
    │
    └─→ build_pipeline_sync(...)
            │
            └─→ get_pipeline("slide_anchored")(transcript, config)
                    │
                    └─→ pipeline_slide_anchored(transcript, config)
```

---

## 2. pipeline_slide_anchored 내부 흐름

```
ENTRY: pipeline_slide_anchored(transcript, config)
│
├─ STEP 1 — slide parsing (Python only, no LLM)
│   ├─ _extract_part_ids(slide_text)
│   │   └─ regex: PART [I|II|...] — Title
│   ├─ _extract_part_sections(slide_text)
│   │   ├─ Core Focus 추출
│   │   ├─ Key Ideas (bullets)
│   │   └─ Why This Part Exists
│   └─ _format_slide_sections_text(sections)
│       └─ LLM 입력용 textual format
│
├─ STEP 2 — node extraction (LLM, 1 call)
│   └─ openai.chat.completions.create(
│         system = _SLIDE_ANCHORED_NODE_SYSTEM,
│         user   = _SLIDE_ANCHORED_NODE_USER.format(
│                    slide_sections_text, transcript))
│      → JSON: nodes [{label, slide_anchor_id, source_sentence,
│                      is_backbone, parent_id}]
│
├─ STEP 3 — node validation + ID 부여 (Python)
│   ├─ _split_transcript_sentences(transcript)
│   │   └─ re.split(r'(?<=[.!?])\s+')
│   │
│   ├─ for each raw node:
│   │   ├─ slide_anchor_id ∈ valid_slide_ids 검증 (없으면 drop)
│   │   ├─ id 부여: node_001, node_002, ...
│   │   └─ store dict
│   │
│   ├─ _find_sentence_index(source_sentence, transcript_sentences)
│   │   └─ difflib.SequenceMatcher → best fuzzy index
│   │
│   ├─ _resolve_parent_id(parent_label, all_nodes, threshold=0.85)
│   │   └─ difflib → 가장 유사한 node_id (or None)
│   │
│   └─ _compute_ordering(nodes)
│       ├─ section_order: PART 안에서 sentence_index 정렬 → 1, 2, 3, ...
│       └─ lecture_order: backbone만 (PART 번호, sentence_index) 정렬 → 1, 2, ...
│
├─ STEP 4a — grounded edge (LLM Pass 1)
│   ├─ _DEFAULT_EDGE_TYPES (from src/extract_edges.py)
│   ├─ _build_edge_types_section(edge_types)  # 프롬프트용
│   └─ openai.chat.completions.create(
│         system = _EDGE_PASS1_SYSTEM (verbatim 강제),
│         user   = _EDGE_PASS1_USER.format(
│                    nodes_text, section_ids, transcript, types))
│      → JSON: edges [{from, to, edge_type, justification, reason,
│                      evidence_section}]
│
├─ STEP 4b — soft edge (LLM Pass 2)
│   └─ openai.chat.completions.create(
│         system = _EDGE_PASS2_SYSTEM (interpretive paraphrase 허용),
│         user   = _EDGE_PASS2_USER.format(
│                    nodes_text, section_ids,
│                    pass1_edges_text,           # ← 중복 방지
│                    transcript, types))
│      → JSON: edges [{..., confidence_score 0.0~1.0}]
│
├─ STEP 5 — edge validation (Python)
│   │
│   ├─ for each raw edge in (pass1, pass2):
│   │   _build_edge_sa(raw_e, edge_source)
│   │     ├─ _normalize_edge_from_to(e)  # source_node→from 등
│   │     ├─ from/to ∈ node_ids?
│   │     ├─ edge_type ∈ valid types?
│   │     ├─ self-loop check (from ≠ to)
│   │     ├─ evidence_section ∈ valid_slide_ids?
│   │     ├─ duplicate check (from, to, edge_type)
│   │     └─ build dict {from, to, edge_type, justification, reason,
│   │                    evidence_section, edge_source, confidence_score}
│   │
│   ├─ STEP 5b — drives 방향 보정
│   │   for each edge:
│   │     if edge_type == "drives" and src_PART > tgt_PART:
│   │       edge.from, edge.to = edge.to, edge.from   # auto swap
│   │
│   └─ edge_id 부여: edge_001, edge_002, ...
│
└─ STEP 6 — enrichment (LLM, optional)
    if config["enrich_graph"]:
      enrich_graph(nodes, edges, transcript, config)
        ├─ batch (size 20)
        ├─ for each batch:
        │   _enrich_batch(batch_nodes, batch_edges)
        │     └─ openai LLM call
        │        system = _GRAPH_ENRICH_SYSTEM,
        │        user   = _GRAPH_ENRICH_USER.format(graph_json)
        │        → 각 node에 description, why_it_matters
        │        → 각 edge에 reason
        └─ merge enrichments back into nodes/edges

▼
RETURN: {
  "nodes": [...],
  "edges": [...],
  "kus":   [],     # slide_anchored은 KU 안 씀
  "timing": {step별 시간}
}
```

---

## 3. 모듈 의존 관계

```
src/pipelines.py
    │
    ├─→ src.prompt_loader    (load_prompt — file → string)
    │
    ├─→ src.extract_edges
    │       _DEFAULT_EDGE_TYPES
    │       _build_edge_types_section
    │       _normalize_edge_from_to
    │
    └─→ openai.OpenAI      (LLM client)


src/build_graph.py
    └─→ build_graph(nodes, edges) → dict   # post-pipeline assembly용 (slide_anchored은 직접 호출 안 함)


src/hallucination_checker.py     # post-pipeline, 사용자 트리거
    └─→ google.genai           (Gemini + Search grounding)


src/visualize_graph.py             # CLI용 정적 PNG
    ├─→ networkx
    └─→ matplotlib


dash_app.py                         # Web UI
    ├─→ dash, dash-cytoscape       (UI framework)
    ├─→ src.pipelines              (get_pipeline, enrich_graph, DEFAULT_CONFIG)
    ├─→ extract_edges              (extract_edges, extract_edges_soft)
    ├─→ build_graph                (assembly)
    └─→ hallucination_checker      (Gemini check 버튼)


run_graph.py                       # CLI
    ├─→ src.pipelines              (get_pipeline, list_pipelines, DEFAULT_CONFIG)
    └─→ src.usage_tracker          (install, reset, snapshot, report)


viewer.py (read-only, AI-built)
    ├─→ dash, dash-cytoscape
    └─→ (no pipeline imports — 단순히 graphs/ JSON 로드해서 표시)
```

---

## 4. 데이터 흐름 (input → output)

```
INPUT
  transcript.txt (Panopto)         ────┐
  slides.txt (pptx2md)             ────┤
                                       ▼
                             [pipeline_slide_anchored]
                                       │
                                       ▼
INTERMEDIATE (in-memory)
  parsed_sections                       ── PART id, title, key_ideas
  raw_nodes (LLM out)                   ── label, slide_anchor_id, source_sentence
  validated_nodes                       ── id, sentence_index, parent_id, *_order
  raw_edges_pass1 (LLM out)             ── grounded edges
  raw_edges_pass2 (LLM out)             ── soft edges + confidence
  validated_edges                       ── edge_id, edge_source, ...

                                       ▼
OUTPUT (KG.json)
  {
    "nodes": [
      {
        "id": "node_001",
        "label": "Type Safety",
        "node_type": "concept",
        "is_backbone": true,
        "parent_id": null,
        "slide_anchor_id": "part_002",
        "slide_anchor_title": "...",
        "source_sentence": "...",
        "sentence_index": 23,
        "section_order": 1,
        "lecture_order": 5,
        "description": "..." (enrichment),
        "why_it_matters": "..." (enrichment)
      }, ...
    ],
    "edges": [
      {
        "edge_id": "edge_001",
        "from": "node_001",
        "to": "node_005",
        "edge_type": "requires",
        "justification": "...",
        "reason": "..." (enrichment),
        "evidence_section": "part_002",
        "edge_source": "explicit",   # or "inferred"
        "confidence_score": null     # null for explicit, 0~1 for inferred
      }, ...
    ]
  }

                                       ▼
DOWNSTREAM
  graphs/<lecture>.json          → Cytoscape viewer (dash_app, viewer.py)
  outputs/<lecture>/             → 전체 자료 (kg + transcript + usage)
  hallucination check (선택)     → Gemini + Google Search → issues.json
```

---

## 5. 함수 / 변수 schema 요약

### Node schema
```python
{
  "id": str,                    # node_001 (Python 부여)
  "node_id": str,               # 동일 (호환성)
  "label": str,                 # LLM 추출
  "node_type": "concept" | "example",
  "is_backbone": bool,          # 핵심 vs 보조
  "parent_id": str | None,      # sub-concept면 부모 node_id
  "slide_anchor_id": str,       # part_001 등 (LLM이 정함)
  "slide_anchor_title": str,    # PART 제목
  "source_sentence": str,       # transcript 인용
  "sentence_index": int,        # transcript 안 위치
  "section_order": int,         # PART 안 순서 (1부터)
  "lecture_order": int | None,  # backbone만, 강의 전체 흐름 순서
  "description": str,           # enrichment
  "why_it_matters": str,        # enrichment
  "supporting_ku_ids": [],      # slide_anchored은 항상 빈 리스트
  "timestamp": {                # transcript 타임스탬프
    "method": str,
    "sentence_index": int
  }
}
```

### Edge schema
```python
{
  "edge_id": str,               # edge_001
  "from": str,                  # source node_id
  "to": str,                    # target node_id
  "edge_type": str,             # defines/requires/explains/details/example_of/contrasts/drives
  "justification": str,         # explicit: verbatim from transcript
                                # inferred: paraphrase
  "reason": str,                # enrichment
  "evidence_section": str,      # PART id (어느 PART에서 온 evidence인지)
  "edge_source": "explicit" | "inferred",
  "confidence_score": float | None  # explicit: None, inferred: 0.0~1.0
}
```

---

## 6. 핵심 디자인 결정

| 결정 | 이유 |
|---|---|
| ID는 Python이 부여 | LLM이 부여하면 일관성 안 보장 + JSON 깨질 위험 |
| Node anchor를 slide ID로 | Lecturer 직접 작성 = KU(LLM 추출)보다 신뢰 가능 |
| Edge 2-pass (grounded vs soft) | precision vs recall trade-off 분리 |
| Edge 검증 fuzzy match (>=0.8) | LLM이 "near-verbatim" 못 지킬 때 보호 |
| 추출과 enrichment 분리 | 추출은 grounded, 설명은 별도 LLM call로 audit 가능 |
| Hallucination check 별도 (Gemini) | 같은 모델이 self-check하면 같은 hallucination 못 잡음 |
