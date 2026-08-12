# step 4 — RQ3 인과 (지침 Key/Value 분해, Spotlight 최종 판정)

**한 줄:** stepC/step1이 **코드** 신호의 레버가 Key(어텐션)가 아니라 **Value**임을 인과로 보였다. step4는 **같은 Key/Value 치환을 지침 토큰에** 적용해, 지침의 레버가 Key인지 Value인지(또는 아예 불활성인지)를 가른다. = 어텐션을 키우는 **Spotlight의 정면 인과 시험.**

대응: **RQ3 인과.** (directive-token = RQ3 관측 → step4 = RQ3 인과.)

---

## directive-token(관측)에서 이어지는 근거

- 지침 지시어("camelCase")는 **읽힌다**(관측). 단 (a) 층 순서는 Qwen 특유, (b) 정규화하면 3/4 모델에서만 "코드만큼 읽힘"(granite 예외).
- 남은 질문: **읽히는 그 지시어가 표기 결정에 실제로 기여하나? 하면 Key 경로냐 Value 경로냐?** → 관측으론 못 정한다. 개입이 답한다.

---

## 무엇을 바꾸나 (개입) — stepC를 지침에 적용

**대상:** 지침의 **표기 지시어 토큰**(`camelCase`/`snake_case`, directive-token의 `instr_target_word`와 동일 span).
**donor:** **반대 지침.** 같은 선행 코드로 지침만 반대 표기(snake)로 렌더한 별도 forward에서, 지시어 토큰의 층 L KV를 뽑아 공여값으로 쓴다. → 텍스트(`camelCase`)는 그대로, 모델 내부가 읽는 지침만 반대로.
**단위(GQA):** KV group 단위(§3). **경로 3종:** Key만 / Value만 / Key+Value. **층:** 전 층 스윕.

**정렬 주의:** `camelCase`와 `snake_case`는 토큰 수가 다를 수 있다 → 코드 이름 정렬과 동일하게 **겹치는 서브토큰만 치환, 불일치는 스킵**(스킵 수 기록).

---

## 무엇을 재나

세 상태 준수 선호 점수 `S = logP(준수 후보) − logP(위반 후보)` (stepC와 동일 지표):

| 상태 | 지침(내부) | 기대 |
|---|---|---|
| **base** | camel 그대로 | S_camel |
| **opp(천장/바닥 기준)** | snake 지침 그대로 | S_snake |
| **int** | camel 텍스트인데 지시어 KV만 snake로 치환 | ? |

**전이율 = (S_int − S_camel) / (S_snake − S_camel).** 지침 KV를 반대로 바꿨을 때 행동이 반대 지침 쪽으로 얼마나 넘어가나. **(층 × kind)** 마다 산출.

---

## 결과 분기 (Spotlight 최종 판정)

| 관측 | 지침 레버 | Spotlight |
|---|---|---|
| **Value** 치환에 전이 O | 내용(Value) | ❌ Key 증폭이라 구조적 실패 → 방법론은 Value(step5) |
| **Key** 치환에 전이 O | 어텐션(Key) | ✅ 유효 여지 → 방법론을 Key로 재조정 |
| **둘 다 무반응** | **불활성** | ❌ 없는 걸 키우는 꼴 (stepA-2 "지침 바꿔도 출력 동일"과 합치) |

directive-token(읽힘) + step4 결과를 합치면:
- **읽히는데 Value로 작동** → Spotlight 헛다리, 내용 개입이 정답
- **읽히는데 불활성** → **가장 강한 반박**(attended but inert)
- **Key로 작동** → 그때만 Spotlight 재고

---

## 전제 조건 / 주의 (§4 미리)

1. **지침 레버리지 틈이 있어야 잰다.** S_camel ≠ S_snake 인 조건이라야 전이율이 의미. 합성에서 Python prior가 세면 틈이 작을 수 있음 → **틈이 큰 조건을 먼저 스크리닝**. 틈이 0이면 그 자체가 "불활성" 답(개입 없이도).
2. **방향 선택.** Python prior상 snake가 강하니, `지침=snake→camel 치환`보다 `지침=camel→snake 치환`이 잘 움직일 수 있음(둘 다 재고 대칭 확인).
3. **모델 다양성.** directive-token이 4모델에서 갈렸으니, step4도 **Qwen 우선 + 가능하면 deepseek/granite/stability**로 일반화 확인. 특히 granite(지시어 미주목)에서 인과도 약한지 교차.
4. **정렬 스킵.** camelCase/snake_case 토큰 수 불일치 → 부분 치환. 스킵 수 기록, 전이율 해석 시 감안.

---

## 하네스 변경 (개입 경로 재사용)

| 대상 | 할 일 |
|---|---|
| `conditions.py` | 개입 타깃 축 추가 — 코드(기존) vs **지침(신규)**. 예: `Intervention.target='instruction'` 또는 새 kind 라우팅 |
| `runner.py` `_preference_context` | 지침 타깃일 때 **viol/donor를 지침 지시어 토큰 + 반대지침 forward**로 구성(코드 이름 대신) |
| `model.py` 치환 훅 | 기존 KV group 치환 재사용 — 대상 토큰 위치만 지침 지시어로 |
| 슬러그 | 지침 타깃·방향 드러나게 |

핵심: **stepC의 반사실 KV 치환 로직을 그대로 쓰되, 치환 대상 토큰을 "코드 이름"에서 "지침 지시어"로, donor를 "같은 이름 준수판"에서 "반대 지침"으로** 바꾼다.

---

## 산출

- 노트북: `notebooks/step4_instruction-kv.ipynb` (§5 7셀, 재개). 요약 = (층×kind) 전이율 곡선 + 피크 + S 세 상태.
- 결과: `results/step4/` (불변, §6).
- 판정: `docs/step4/results.md` — Value/Key/불활성 → Spotlight 최종 결론, step5 설계 근거.

---

## 연결
directive-token(RQ3 관측) → **step4(RQ3 인과)** → 방법론 step5(Value 경로 스티어링, Spotlight 반대). qna Q2·Q6와 정합.
