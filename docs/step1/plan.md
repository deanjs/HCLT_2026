# step 1 — RQ2 층 스윕 + K/V 분해 (최우선)

**한 줄:** step C는 **L25 한 층에서만** "위반 표기 형태 신호가 표기 결정의 원인"임을 확인했다. step 1은 그 개입을 **전 층에 반복**하고 **Key/Value 경로를 분해**해, (1) 회복이 어느 층에서 일어나는지(피크 층)와 (2) 신호가 어텐션(Key) 경로인지 Value 경로인지를 가른다. 관측 측에서는 **층별 v 코사인 궤적**을 함께 기록한다.

대응: **RQ2** — 매개 신호가 형태인가, 어느 층인가, 어느 경로인가. → step 5(Value 경로 스티어링)의 층·경로 선택 근거.

## step C에서 이어지는 근거 (리뷰어가 물을 두 가지)

1. **왜 하필 25층인가?** 단일 층 결과는 발견이 아니다. 전 층 회복률 곡선이 없으면 방법론의 층 선택이 임의로 보인다. → 전 층 스윕.
2. **‖v‖(크기)는 안 변했는데(배율 1.002) 방향은?** 크기 지표만으로는 Value 경로(축 B)를 판정할 수 없다. 잔차에 더해지는 것은 크기가 아니라 방향이 결정한다. → 층별 v 코사인(방향) 궤적.

## 무엇을 바꾸고 무엇을 재나

### A. 개입 측 — 전 층 × K/V 분해 (인과)

- **바꾸는 것:** 위반 선행(snake 12, POOL n=0)의 KV 캐시를 층을 0→L−1 하나씩 돌며 준수판 값으로 치환. 각 층에서 세 경로: **Key만 / Value만 / Key+Value**. (치환은 KV group 단위 — §3 GQA 주의.)
- **재는 것:** 세 상태 준수 선호 점수 `S`(clean/baseline/intervened)와 **회복률** = `(S_int − S_base)/(S_clean − S_base)`. step C와 동일 지표를 **모든 (층 × kind)에서** 산출.
- **통제:** donor 2종 — `compliant`(같은 이름 준수판, 주효과) / `unrelated_camel`(무관 준수형, 형태 통제). 두 곡선이 전 층에서 겹치면 "내용이 아니라 형태" 결론이 L25만이 아니라 전 층에서 유지됨을 보인다. (음성통제 `unrelated_snake`는 step C에서 L25 회복률 0.000으로 이미 확인 → 스윕에서는 생략.)

### B. 관측 측 — 전 층 v 코사인 궤적 (방향)

- **재는 것:** 같은 이름의 **camel판 v와 snake판 v 사이 코사인 유사도**를 층마다(KV head·이름 토큰쌍 평균). 같은 선행을 전부 camel / 전부 snake로만 달리 렌더해 두 번 forward하고, 역할별로 정렬한 이름 토큰 위치에서 비교.
- **왜:** `‖v‖`는 방향 정보를 지운다. 코사인만이 "크기는 같지만 다른 정보를 싣는가"를 답한다.

## 결과의 의미 (어느 쪽이 나오든)

| 결과 | 해석 | 다음 |
|---|---|---|
| 회복률이 특정 층에서 **피크** | 표기 형태 결정이 그 처리 단계에서 굳는다(§2.5). L25=25/35(후반 0.71)가 피크면 "출력 직전 결정" | step 6 로짓 렌즈로 뒷받침 |
| **Key만**으로 회복 | 형태 신호가 어텐션(Key) 경로 | step 5 개입을 Key로 |
| **Value만**으로 회복 | 형태 신호가 Value 경로 | step 5 개입을 Value로 |
| **둘 다 부분** 회복 | 두 경로 협력 | step 5를 양쪽 개입으로 설계 |
| v 코사인 **낮음** | 크기 같아도 다른 정보 → Value 치환 효과와 맞물림(축 B 성립) | Value 경로 확정 |
| v 코사인 **높은데** Value 치환이 효과 | 신호가 Value 아닌 Key 경로 → 재해석 | Value 결과와 교차검증 |

- **곡선 교차검증:** Value 치환 회복률 피크 층과 v 코사인이 뚝 떨어지는 층이 **일치**하면 축 B(Value 경로·방향)가 확정된다.

## 조건 축 (스키마 값 조합, 새 스크립트 없음 — §3)

| 축 | 값 |
|---|---|
| model | Qwen/Qwen2.5-Coder-3B-Instruct (fp16, 36층, GQA group 8) |
| preceding | POOL n_compliant=0, n_functions=12 (전부 위반 snake) |
| instruction | positive · target=camel · weak |
| intervention | kind ∈ {key, value, key_value}, **layers="sweep"**, donor ∈ {compliant, unrelated_camel} |
| seed | 0..9 (10 seed) |

- 개입 스윕: 36층 × 3 kind × 2 donor × 10 seed. 관측 코사인: 2 donor 불필요(같은 이름 camel/snake만) → 10 seed.
- **착수 전:** `config.json`에서 `num_hidden_layers`를 확인해 L25의 상대 위치를 확정한다(추정 금지). 노트북 셀 2에서 `handle.num_layers`·`relative_layer(25)`로 출력.

## 구현 (하네스 확장)

- `attention_probe.cosine(u, v)` — 순수 코사인(방향). 크기 0이면 None. 단위 테스트.
- `intervention.peak_layer(recovery_by_layer)` — 피크 층 요약. None 층 제외.
- `model.intervene_preference_sweep(...)` — 선행 프롬프트 1회 forward + work 캐시 재사용(층·kind마다 편집→측정→복구)으로 전 층 × K/V. `intervene_preference`(단일 층)와 `_preference_context`·`_score_layer_kind`를 공유(중복 금지).
- `model.name_v_cosine_sweep(...)` — camel/snake 두 forward에서 층별 이름 토큰 v 코사인.
- `runner`: `intervention.is_sweep` → 개입 스윕 경로, `mode="vcosine"` → 코사인 경로. `_preference_setup`으로 단일/스윕 입력 공유.

## 실행 전 예측

- 회복률 피크는 **후반부(≈L25 부근, 상대 0.6~0.75)**에 국소화. 초기·중기 층 치환은 회복 미미.
- `compliant`와 `unrelated_camel` 곡선은 전 층에서 거의 겹친다(형태 결론 유지).
- K/V 분해: Key+Value ≥ 각 단독. Value 단독이 유의하게 회복하면 축 B 성립. Key 단독만 회복하면 신호는 어텐션 경로.
- v 코사인은 초기 층에서 높다가 **피크 층 부근에서 하락** — Value 회복률 피크와 같은 층에서 갈라지면 축 B 확정.

산출물은 `results/step1/`에 불변 저장(§6). 예측은 결과와 함께 보존한다.
