# TODO — 검증/개선 항목

production 코드에 들어가 있지만 **실험적 근거가 부족한 휴리스틱**들. 졸업 후 또는 시간 되면 검증.

---

## 1. Self-loop 자동 drop 검증

### 현재 동작
`pipeline_slide_anchored` step 5에서 `from == to`인 edge를 무조건 drop.

```python
if fr == to:
    return None, "self_loop"   # → issues.json에 기록만
```

### 검증해야 할 것
- 실제로 LLM이 self-loop을 얼마나 자주 출력하나? (per-lecture 평균)
- 그 self-loop들이 진짜 LLM 에러인가? 아니면 의미 있는 self-reference가 있나?
- 특히 recursive concept (예: "Recursion이 자기 자신을 정의함") 같은 경우 self-loop이 정당한지

### 실험 셋업 (제안)
1. self-loop drop 로직 OFF 버전 만들기 (또는 issues.json에서 self-loop만 추출)
2. 8개 강의 다 돌려서 self-loop 후보 수집
3. Manual review:
   - `false_positive`: drop이 옳음 (LLM 에러)
   - `legitimate`: drop이 틀림 (의미 있는 self-reference)
4. 결론: "X% are LLM errors, Y% are legitimate self-references"

### 예상 결과
대부분 LLM 에러일 것. 그래도 paper에 evidence로 인용 가능.

---

## 2. drives edge 시간 역방향 — flag 의 false positive rate

### 현재 동작
production은 swap 안 함. 단지 `issues.json`에 의심으로 기록만:
```json
{
  "type": "drives_direction_suspicious",
  "from": "node_005",
  "to":   "node_002",
  "src_part": 3,
  "tgt_part": 1,
  "note": "backward drives across PART boundary"
}
```

### 검증해야 할 것
- "later → earlier drives"가 정말 항상 잘못된 방향인가?
- Lecturer가 retrospective하게 motivation을 설명하는 경우 (예: "이건 사실 PART 1에서 봤던 X를 motivate한 거죠") 정당한 backward driving 아닌가?
- false positive rate

### 실험 셋업
1. 모든 backward drives flag된 edge 모음
2. 해당 transcript 부분 manual review
3. 카운트:
   - `swap_correct`: 방향 바꾸는 게 맞음
   - `swap_wrong`: 그대로가 맞음 (legitimate retrospective)
4. flag 정확도 측정

### 결정점
- false positive rate < 10% → swap 자동화 OK (현재 OFF인데 켤지 결정)
- false positive rate > 30% → flag 자체도 의미 없으니 제거

---

## 3. evidence_section invalid 처리

### 현재 동작
LLM이 `evidence_section`을 잘못 출력하면 (`part_999` 같이 존재하지 않는 ID) `None`으로 변경:
```python
if ev_section not in valid_slide_ids:
    ev_section = None
```

### 검증해야 할 것
- Invalid evidence_section 빈도
- LLM이 헷갈리는 패턴이 있나? (예: 항상 +1/-1 off-by-one?)

### 실험
issues.json에서 `edge_dropped/invalid_evidence_section` 케이스 추출 → 패턴 분석.

---

## 4. parent_id 기능 자체 — 제거 결정 근거 (이미 제거됨)

production에서 빠짐. 이유: dash_app inspector display 외 거의 안 씀.

### 향후 고려
- 만약 paper에서 hierarchy 평가가 필요해지면 다시 추가
- 지금은 `slide_anchor_id`로 충분

---

## 5. fuzzy match threshold 튜닝

### 현재 동작
- `_find_sentence_index`: SequenceMatcher로 가장 비슷한 문장 무조건 반환
- `_validate_strict_edge` (extract_edges.py): justification fuzzy match >= 0.8

### 검증해야 할 것
- 0.8이 최적인가? 0.7 / 0.9 / 0.85 비교
- false negative (정당한 grounded edge가 0.8 미만이라 drop되는 경우) vs false positive (LLM이 paraphrase한 걸 verbatim이라고 인정하는 경우)

### 실험
threshold를 [0.7, 0.75, 0.8, 0.85, 0.9]로 sweep → strict edge 수 + manual quality 평가.

---

## 6. Hallucination check coverage

### 현재 동작
Gemini + Google Search grounding. 사용자 트리거 (UI 버튼).

### 검증해야 할 것
- Gemini가 잡는 hallucination 종류 분포 (factual / wrong relationship / misuse_of_term)
- false positive rate (Gemini가 정상 노드/엣지를 잘못 issue로 표시하는 비율)
- 어떤 도메인에서 더 잘 작동? (CS vs e-commerce vs other)

### 실험
8개 강의 hallucination check 결과 → 카테고리별 distribution + manual verification.

---

## 우선순위

| 항목 | 중요도 | 시간 |
|---|---|---|
| 1. Self-loop 검증 | 중간 (paper에 인용 가능) | 30분 |
| 2. drives flag FP rate | 높음 (production 영향) | 1시간 |
| 3. evidence_section | 낮음 | 20분 |
| 5. fuzzy threshold | 중간 (parameter sensitivity) | 2시간 (sweep) |
| 6. hallucination eval | 높음 (논문 evaluation) | 2시간 |

5/16 마감 전: 2번만 가볍게 수동 검토 권장 (issues.json 한 번 훑으면 됨).
