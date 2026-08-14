# step A-1 — 선행 distinct (원래 step 3)

**한 줄:** step A의 선행 12개를 "같은 일 12개 클론"에서 "**서로 다른 과제 12개**"로 바꿔, 관측된 바닥·primacy·잠금 패턴이 그대로 재현되는지 본다.

대응: **RQ1** — 대조군. step A(clone)와 나란히 비교한다.

## 무엇을 방어하나

step A는 선행 12개가 **같은 함수의 이름만 다른 복제**였다. 그래서 이런 반론이 가능하다:

> "이건 지침 준수 실패가 아니라, 반복된 데모를 그대로 따라 한 **in-context 데모 오버라이드**일 뿐이다."

선행을 **서로 다른 과제**로 바꿔도 같은 패턴이 나오면, 효과는 특정 과제의 반복 모방이 아니라 **표기 형태 신호**에 있다는 뜻이 된다 → RQ1 견고.

## step A와 딱 하나만 다르다

| 축 | step A | **step A-1** |
|---|---|---|
| 선행 `composition` | clone (같은 과제 12복제) | **distinct (서로 다른 과제 12개)** |
| 선행 `n_compliant` | 4/3/2/1/0 | 4/3/2/1/0 (동일) |
| 지침 | 긍정·camel·약함 | 동일 |
| 개입 | none | none |
| 반복 | seed 20 | seed 20 |
| 생성 3 | clamp/count/merge | 동일 (선행 12과 겹치지 않음) |

→ 5 × 20 = **100회**. 비교가 깨끗하도록 목표 표기도 step A와 같은 **camelCase**로 둔다(설정 하나만 바꿔 clone 효과와 대비).

## 프롬프트 구조

step A와 동일한 "선행 12 + 생성 3", 순차 3턴. 차이는 선행 12개가 **12가지 다른 과제**(각각 camel 또는 snake로 렌더링, n_compliant개가 camel, 위치 셔플)라는 점뿐. 생성 경로를 그대로 재사용하므로 **원문(`turn_texts`)이 자동 저장**된다.

## 재는 것

step A와 동일: 준수율(첫 함수)·자기증폭·위반 정도(위반 턴 분포). **clone과 겹쳐 그려** 비교한다. primacy 가설도 distinct에서 같은지 확인(첫 자리 표기 vs 출력).

## 예상과 해석 (§4: 조건 안 맞춤, 그대로 기록)

| 결과 | 해석 |
|---|---|
| distinct에서도 같은 바닥·primacy·잠금 | 효과가 데모 모방이 아니라 **형태 신호** → 데모 오버라이드 반론 무력화, RQ1 견고 |
| distinct에서 패턴 사라짐(준수율 안정/무관) | clone 효과는 데모 모방이었을 수 있음 → 반론 유효, 프레이밍 축소 |
| clone보다 약하지만 존재 | 데모 요소 + 형태 요소 공존 → 둘 다 서술 |

## 구현 범위

step A 코드를 재사용하고, `composition=distinct` 경로만 채운다:

| 대상 | 할 일 |
|---|---|
| `tasks.py` | **DISTINCT_TASKS** 풀 추가 — 서로 다른 과제 12개(생성 3과 겹치지 않게). 모두 두 단어 이상 |
| `prompt.py` | `build_preceding_code`의 distinct 분기 구현(현재 `NotImplementedError`). 각 과제를 n_compliant 표기로 렌더링·셔플 |
| 나머지 | 생성·측정·저장 경로는 그대로 (원문 자동 저장) |

개입·어텐션은 건드리지 않는다.

## 설정

- 모델 `Qwen/Qwen2.5-Coder-3B-Instruct` (fp16), max_new_tokens 256 — step A와 동일(비교 조건 통제)
- 결과: `results/stepA-1/<조건-슬러그>.json` (불변, §6). 슬러그의 `composition`이 `distinct`로 드러남
- 노트북: `notebooks/stepA-1_prefix-distinct.ipynb` (§5 7셀, 재개 가능)

## 다음

clone(step A) vs distinct(step A-1) 비교 결과를 `docs/stepA-1/results.md`에 기록 → RQ1 방어 판정.
