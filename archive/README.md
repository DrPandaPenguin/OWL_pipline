# Archive — 비교 실험에서 시도했던 파이프라인 변종들

production 코드는 [src/pipelines.py](../src/pipelines.py) 참고.
이 폴더는 **참조용**. import 안 됨.

## pipelines_experimental.py

논문 평가 단계에서 비교했던 파이프라인 변종들이 모두 들어있는 파일.
production에 채택된 `slide_anchored` + fallback `multi_stage`만 `src/pipelines.py`로 옮겼고,
나머지는 여기 보관:

| 파이프라인 | 설명 | 채택 안 한 이유 |
|---|---|---|
| `single_pass` | 1번의 LLM 호출로 노드+엣지 모두 추출 | recall 낮음 (V1 baseline) |
| `direct` | KU 단계 생략, 노드 바로 추출 후 soft edge만 | 검증 layer 부족 |
| `slide_structure` | 슬라이드를 노드 backbone으로, KU 사용 | slides + transcript 통합이 더 좋음 |
| `slide_no_ku` | 슬라이드 segment를 KU 대체로 | grounding 약함 |
| `multi_stage_refined` | multi_stage + LLM refinement pass | refinement가 slide_anchored에 통합됨 |
| `slide_anchored_full` | 슬라이드+transcript 둘 다에서 노드/엣지 추출 (one-shot) | multi-pass(slide_anchored)가 정확도 더 높음 |
| `segmented_*` (12개) | TF-IDF / LLM / fuzzy / embedding 기반 transcript 분할 + 3가지 merge 전략 조합 | segment-then-merge가 holistic보다 일관성 떨어짐 |
