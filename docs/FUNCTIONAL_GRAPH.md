# Functional Graph — slide_anchored (최신)

OWL Pipeline production 파이프라인 (`slide_anchored`)의 함수 흐름.

---

## Top-level entry

```
run_graph.py (CLI)                  dash_app.py (Web UI)
   │                                   │
   └→ get_pipeline("slide_anchored")   └→ build_pipeline_sync(...)
            └→ pipeline_slide_anchored(transcript, config)
```

---

## pipeline_slide_anchored 흐름

```
ENTRY: pipeline_slide_anchored(transcript, config)
│
config 검증:
  ├ slide_text 없으면 → return error
  └ OPENAI_API_KEY 없으면 → return error
│
▼
STEP 1 — 슬라이드 파싱 (Python only, no LLM)
  ─ _extract_part_ids(slide_text)
       └ regex r"^PART\s+([IVXLCDM]+|\d+)\s*[—–-]\s*(.+)$"
          → [{slide_id, title}, ...]
  ─ _extract_part_sections(slide_text)
       ├ Core Focus 추출 (regex)
       ├ Key Ideas (bullet)
       └ Why This Part Exists
          → [{section_id, title, core_focus, key_ideas[],
              why_this_part_exists}, ...]
  ─ _format_slide_sections_text(sections)
       └ Python dict → LLM이 읽을 textual format
         ("[part_001] Title\n  Core Focus: ...\n  Key Ideas: ...")
│
▼
STEP 2 — 노드 추출 (LLM × 1 call)
  입력:
    [system] _SLIDE_ANCHORED_NODE_SYSTEM
             ("강의 노드 추출자. 각 노드를 슬라이드 섹션에 anchor")
    [user]   slide_sections_text + transcript 전체
  LLM 작업:
    transcript 통째로 읽음 → 슬라이드 섹션을 ground truth로 → 섹션별 노드 후보 추출
  LLM 출력 (JSON):
    [
      {label, slide_anchor_id, source_sentence, is_backbone, node_type},
      ...
    ]
    ※ parent_id 없음 (제거됨)
│
▼
STEP 3 — 노드 검증 + ID 부여 (Python)
  _split_transcript_sentences(transcript)
    └ re.split(r'(?<=[.!?])\s+')
       → [sent1, sent2, ...]
  for each raw_node:
    ├ label 비었으면 → drop
    ├ slide_anchor_id ∉ valid_slide_ids → drop
    ├ id 부여: node_001, node_002, ...
    └ store
  for each node:
    _find_sentence_index(source_sentence, transcript_sentences)
      └ SequenceMatcher fuzzy → best_idx
    node["sentence_index"] = best_idx
  _compute_ordering(nodes)
    ├ slide_anchor_id로 그룹핑 → sentence_index 정렬 → section_order
    └ is_backbone=True만 추출 → (PART#, sentence_index) 정렬 → lecture_order
│
▼
STEP 4a — Grounded Edge (LLM Pass 1)
  입력:
    [system] _EDGE_PASS1_SYSTEM ("verbatim from transcript")
    [user]   nodes_text + section_ids + transcript + edge_types (7개)
  LLM 작업:
    각 노드 쌍 검토 → "강사가 직접 말했나?" 판단
    말했으면 그 문장 그대로 justification에 복사
  출력:
    [{from, to, edge_type, justification, reason, evidence_section}, ...]
    ※ confidence_score 없음 (확실한 것만)
│
▼
STEP 4b — Soft Edge (LLM Pass 2)
  입력: Pass 1 입력 + pass1_edges_text (중복 방지)
  LLM 작업:
    "함의된 관계가 있나?" → paraphrase로 justification + confidence 0~1
  출력:
    [{..., confidence_score: 0.75}, ...]
│
▼
STEP 5 — Edge 검증 + 통합 (Python)
  for each raw_edge in (pass1, pass2):
    _build_edge_sa(raw_e, edge_source) → edge | None
      ├ from = e["from"] or e["source_node"]    ← inline (normalize 함수 없음)
      ├ to   = e["to"]   or e["target_node"]
      ├ from/to 없거나 unknown → None
      ├ self-loop (from==to) → None
      ├ edge_type 화이트리스트 외 → None
      ├ evidence_section 유효 X → None으로 (edge는 살림)
      └ edge dict 반환
    edge None이면 silently drop
    edge 있으면:
      ├ duplicate (from, to, edge_type) → silently drop
      └ 유효하면 edges.append
  edge_id 부여: edge_001, edge_002, ...
  ※ Step 5b 없음 (drives 방향 swap/log 다 제거)
│
▼
STEP 6 — Enrichment (LLM, optional)
  if config["enrich_graph"]:
    enrich_graph(nodes, edges, transcript, config)
      ├ 20개씩 batch
      └ LLM call per batch:
          → 각 node에 description, why_it_matters
          → 각 edge에 reason
│
▼
RETURN: {
  "nodes":  [...],
  "edges":  [...],
  "kus":    [],
  "timing": {step별 시간}
}
※ issues 필드 없음 (제거됨)
```

---

## 호출 그래프 (depth-first)

```
pipeline_slide_anchored
├── _extract_part_ids                    [stdlib re]
├── _extract_part_sections               [stdlib re]
├── _format_slide_sections_text          [no deps]
├── openai.chat.completions.create       [STEP 2: 노드 추출]
├── _split_transcript_sentences          [stdlib re]
├── _find_sentence_index                 [difflib]
├── _compute_ordering                    [collections]
├── _DEFAULT_EDGE_TYPES                  [from src/extract_edges]
├── _build_edge_types_section            [from src/extract_edges]
├── openai.chat.completions.create       [STEP 4a: grounded]
├── openai.chat.completions.create       [STEP 4b: soft]
├── _build_edge_sa (local)               [no deps]
└── enrich_graph (optional)              [STEP 6]
    └── openai.chat.completions.create
```

---

## Output Schema

### Node
```python
{
  "id":                   "node_001",
  "node_id":              "node_001",
  "label":                "Type Safety",
  "node_type":            "concept" | "example",
  "is_backbone":          True | False,
  "slide_anchor_id":      "part_002",
  "slide_anchor_title":   "Designing the Business Model",
  "source_sentence":      "transcript에서 인용",
  "sentence_index":       23,
  "section_order":        1,
  "lecture_order":        5 | None,     # backbone만
  "description":          "..." (enrichment),
  "why_it_matters":       "..." (enrichment),
}
# 제거됨: parent_id, supporting_ku_ids, timestamp
```

### Edge
```python
{
  "edge_id":           "edge_001",
  "from":              "node_001",
  "to":                "node_005",
  "edge_type":         "requires",  # 7가지 중
  "justification":     "...",       # 증거 (transcript)
  "reason":            "...",       # 왜 (LLM 해석, enrichment)
  "evidence_section":  "part_002" | None,
  "edge_source":       "explicit" | "inferred",
  "confidence_score":  None | 0.0~1.0
}
# 제거됨: _direction_swapped
```

---

## 모듈 의존

```
src/pipelines.py  (production 메인)
   │
   ├ src.prompt_loader      → load_prompt
   ├ src.extract_edges      → _DEFAULT_EDGE_TYPES, _build_edge_types_section,
   │                          extract_edges (direct에서만)
   └ openai                 → LLM client

src/build_graph.py
   └ build_graph(nodes, edges) → dict

src/hallucination_checker.py
   └ google.genai (Gemini + Search grounding)

dash_app.py / viewer.py
   └ dash, dash-cytoscape

run_graph.py
   ├ src.pipelines (get_pipeline, DEFAULT_CONFIG)
   └ src.usage_tracker
```

---

## 핵심 디자인 결정

| 결정 | 이유 |
|---|---|
| ID는 Python이 부여 | LLM이 ID 부여하면 일관성 안 보장 |
| Slide를 ground truth로 | Lecturer 직접 작성, 신뢰 가능 |
| Edge 2-pass (grounded vs soft) | precision vs recall 분리 |
| Edge fuzzy match | LLM "near-verbatim" 못 지킬 때 보호 |
| 추출 vs enrichment 분리 | 추출은 grounded, 설명은 별도 LLM call |
| Hallucination check 별도 (Gemini) | 같은 모델 self-check 안 함 |
| Drop된 거 silently | 학생 코드답게 — 검증 layer 단순화 |
