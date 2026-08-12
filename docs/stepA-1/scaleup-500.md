# step A-1 — scaleup-500 (RQ1 대조군, distinct 선행, 500 확대 재실행)

> 파일럿(`docs/stepA-1/plan.md`, 준수 4~0 × seed 20 = 100회)의 **확대 재실행**이다.
> 결론을 바꾸려는 게 아니라 준수 격자(8~0)와 반복(seed 56)을 늘려 **n≈500**으로 대조 주장을 단단히 한다.
> 파일럿 결과는 `results/`에 불변으로 남는다(CLAUDE.md §6).

---

## 1. RQ1 — 무엇이 궁금한가

**대응: RQ1 (대조군).** stepA(clone)와 나란히 비교한다. (연구계획서 §3)

**무엇이 궁금한가:** 선행 12개를 서로 다른 과제(distinct)로 바꿔도, stepA에서 본 **절벽·자기증폭**이 그대로 재현되는가. 재현되면 효과는 특정 과제의 반복 모방이 아니라 **표기 형태 신호** → 데모 오버라이드 반론 무력화, RQ1 견고. 사라지면 clone 효과는 데모 모방이었을 수 있음 → 주장 범위 축소.

---

## 2. 어떤 실험을 하는가

### stepA와 딱 하나만 다르다

| 축 | stepA (scaleup-500) | **stepA-1 (scaleup-500)** |
|---|---|---|
| 선행 `composition` | `CLONE` (같은 과제 12복제) | **`DISTINCT` (서로 다른 과제 12개)** |
| 선행 `n_compliant` | 8~0 (9단계) | 동일 |
| 지침 | 긍정 · camel · 약함 | 동일 |
| 개입 | none | none |
| 반복 `seed` | 56 | 동일 |
| 생성 3 | remove/count/merge | 동일 (선행 12과 겹치지 않음) |

→ **9단계 × 56 seed = 504회 ≈ 500.** 비교가 깨끗하도록 목표 표기도 stepA와 같은 **camelCase**로 둔다.
**바꾸는 건 `composition` 하나뿐**이라, 차이가 나면 그건 clone/distinct 차이로 귀속된다.

### 프롬프트 구조

stepA와 동일한 "선행 12 + 생성 3", 순차 3턴. 유일한 차이는 선행 12개가 **12가지 다른 과제**라는 점.
각 과제를 camel 또는 snake로 렌더링하고, `n_compliant`개가 camel, 위치는 seed로 셔플한다.
생성 경로를 그대로 재사용하므로 원문(`turn_texts`)도 자동 저장된다.

### 재는 것

stepA와 동일: **준수율(첫 함수)**, **자기증폭**(첫 위반 시 이후 연쇄). 요약에서 **clone(stepA)과 겹쳐 그려** 비교한다.

### 결과 해석 (어느 쪽이든 그대로 기록, §4)

| 나온 결과 | 해석 |
|---|---|
| distinct에서도 **같은 절벽·자기증폭** | 데모 모방이 아니라 **형태 신호** → 반론 무력화, RQ1 견고 |
| distinct에서 **패턴 사라짐**(준수율 안정/무관) | clone 효과는 데모 모방이었을 수 있음 → 반론 유효, 프레이밍 축소 |
| clone보다 **약하지만 존재** | 데모 요소 + 형태 요소 공존 → 둘 다 서술 |

---

## 3. 쓰는 데이터셋

| 항목 | 값 |
|---|---|
| 데이터 종류 | **합성(SYNTHETIC)** |
| 선행 구성 | **`DISTINCT`** — 서로 다른 과제 12개 |
| 선행 과제 | `DISTINCT_TASKS` 12종 (`src/harness/tasks.py`) |
| 생성 과제 | `GENERATION_TASKS` 3종 (`remove duplicates` → `count vowels` → `merge dicts`) |
| POOL(50×50) | **미사용** — stepA-1도 이름 풀이 필요 없다 |
| 총 관측 | 9단계 × 56 seed = **≈500** |

### 선행 과제 `DISTINCT_TASKS` (12종, 모두 두 단어)

| # | 이름(단어) | 설명 |
|---|---|---|
| 1 | sum / list | returns the sum of a list |
| 2 | max / value | returns the largest value in a list |
| 3 | reverse / string | reverses a string |
| 4 | is / even | checks whether a number is even |
| 5 | to / upper | converts a string to upper case |
| 6 | first / item | returns the first item of a list |
| 7 | last / item | returns the last item of a list |
| 8 | square / number | returns the square of a number |
| 9 | join / words | joins a list of words with spaces |
| 10 | strip / spaces | strips leading and trailing spaces |
| 11 | double / value | returns the value doubled |
| 12 | abs / diff | returns the absolute difference of two numbers |

예시 이름: `sumList`, `maxValue`, `reverseString`, `isEven` (camel) / `sum_list`, `max_value`, `reverse_string`, `is_even` (snake).

### 생성 과제 `GENERATION_TASKS` (측정 대상, stepA와 동일)

| # | 이름(단어) | 설명 | 기대 이름(camel/snake) |
|---|---|---|---|
| 1 | remove / duplicates | removes duplicate items from a list, preserving order | `removeDuplicates` / `remove_duplicates` |
| 2 | count / vowels | counts the vowels in a string | `countVowels` / `count_vowels` |
| 3 | merge / dicts | merges two dictionaries | `mergeDicts` / `merge_dicts` |

> 선행 12과 생성 3은 서로 겹치지 않는다 — 이름 복사(모방)를 막아 모델의 "직접 선택"만 측정한다.

---

## 4. 데이터를 어떻게 구성했는가

데이터는 저장된 파일이 아니라 **매 실행 시 조건(seed 포함)으로 조립**한다.
stepA와 통제 원칙은 같고, **선행 12개가 클론이 아니라 서로 다른 과제**라는 점만 다르다.

### 통제 원칙

- 선행 함수의 **본문·매개변수는 각 과제 고유값으로 고정**하고, 조작하는 것은 이름 표기(camel/snake)뿐.
  → 준수율 차이가 "내용"이 아니라 "형태(표기)"에서 왔음을 가른다. (stepA는 본문까지 동일, stepA-1은 과제별 본문이 다름 — 그럼에도 표기만 조작 변수)
- 이름은 **두 단어 이상**이어야 한다. 한 단어는 camel == snake라 표기 구분 불가.
  → `DISTINCT_TASKS` 12종 전부 두 단어.
- 생성 3개는 선행 12과도, 서로와도 **겹치지 않게** 한다.

### 조립 절차 (한 조건 = 준수 개수 `n` + seed 1개)

1. **표기 배열** — 준수(camel) `n`개 + 위반(snake) `12-n`개로 12칸 표기 리스트를 만들고, `seed`로 순서를 셔플한다.
2. **함수 렌더** — `DISTINCT_TASKS` 12과제를 앞에서부터, 각 자리를 표기 리스트대로 camel/snake로 렌더링한다. (예: `sumList`, `max_value`, `reverseString`, …)
3. **프롬프트 조립** — `system`(지침) + `user`(선행 12개 + `Add a function that removes duplicate items …` 생성 요청). 이후 턴은 생성 과제 2·3만 추가.

> **clone(stepA)과 차이:** stepA는 seed가 표기 위치만 바꿨다(함수가 다 같으니까).
> stepA-1도 과제 순서는 고정이고 **seed는 표기 배치만** 바꾼다 — 즉 두 실험 모두 seed는 이름을 새로 만들지 않고 "어느 자리가 준수/위반인가"만 변주한다. 비교가 깨끗한 이유다.

### 실제 프롬프트 예시 (`n_compliant=2`, 지침=camel)

```
[system] You are helping extend an existing Python module.
         In this project we generally write function names in camelCase.
         Every function name uses one of two styles only: camelCase or snake_case.

[user]   Here is the current module:
         ```python
         def sumList(items):
             return sum(items)
         def max_value(items):              ← 서로 다른 12과제, camel/snake 위치는 seed가 셔플
             return max(items)
         ... (총 12개 서로 다른 과제, 이 중 camel 2개 / snake 10개) ...
         ```
         Add a function that removes duplicate items from a list, preserving order.
```

---

## 5. 코드 동일 보장 (재현성)

노트북은 실험 로직을 쓰지 않고 파일럿과 **똑같은 `harness.run`을 호출**만 한다.
조건 축만 넓혔으므로, 겹치는 부분(**준수 4~0 × seed 0~19, distinct**)은 파일럿(`results/stepA-1/`) 결과와 **동일**하게 나온다.
scaleup 결과는 `results/stepA-1_scaleup500/`에 저장해 파일럿과 분리·불변 보존한다(§6).
