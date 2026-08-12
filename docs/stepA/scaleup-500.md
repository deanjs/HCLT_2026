# step A — scaleup-500 (RQ1 절벽, 500 확대 재실행)

> 파일럿(`docs/stepA/plan.md`, 준수 4~0 × seed 20 = 100회)의 **확대 재실행**이다.
> 결론을 바꾸려는 게 아니라, 준수 격자를 넓히고(8~0) 반복을 늘려(seed 56) **n≈500**으로
> RQ1 절벽 주장을 단단히 한다. 파일럿 결과는 `results/`에 불변으로 남는다(CLAUDE.md §6).

---

## 1. RQ1 — 무엇이 궁금한가

**대응: RQ1.** 선행 코드의 표기 위반이 이후 생성의 준수율을 낮추는가. (연구계획서 §3)

**무엇이 궁금한가:** 선행에 준수 예시가 몇 개 남았느냐에 따라 준수율이 어떻게 변하는가. 특히 (1) 준수 예시가 줄다 어느 지점에서 **급락하는 절벽**이 있는가, (2) 첫 함수를 위반하면 이후로 **연쇄(자기증폭)** 되는가.

---

## 2. 어떤 실험을 하는가

### 프롬프트 구조: 선행 12개 + 생성 3개

| 구획 | 개수 | 내용 | 표기 | 역할 |
|---|---|---|---|---|
| **선행(preceding)** | 12 | **모두 동일한 함수**(clone) | 준수 `n`개 camel + 나머지 snake, 위치는 seed로 셔플 | 모델이 *읽기만* 하는 재료 |
| **생성(generation)** | 3 | **서로 다른 함수** | 모델이 *직접 선택* | 측정 대상 |

- 선행 12개는 프롬프트에 미리 써넣는다. 모델은 읽기만 한다. 같은 일을 하고 **이름 표기만** 다르다.
- 생성 3개는 모델이 새로 작성한다. **첫 번째** 함수의 표기가 준수율, **2·3번째**는 자기증폭 관측용.
- 지침은 "이 프로젝트는 camelCase를 쓴다"(긍정·약한 어조). 즉 **camel = 준수, snake = 위반**.

### 바꾸는 조건 (조건 축)

| 축 | 값 | 파일럿 대비 |
|---|---|---|
| 선행 준수 개수 `n_compliant` | **8 / 7 / 6 / 5 / 4 / 3 / 2 / 1 / 0** (9단계) | 4~0(5단계)에서 **확장** |
| 선행 구성 `composition` | `CLONE` | 동일 |
| 선행 출처 `source` | `SYNTHETIC` | 동일 |
| 지침 | 긍정 · 목표=camel · 약한 어조 | 동일 |
| 개입 `intervention` | `none` | 동일 |
| 반복 `seed` | 조건당 **56** | 20에서 **확장** |

→ **9단계 × 56 seed = 504회 생성 ≈ 500.**

> 준수 개수만 바꾸고 나머지(본문·매개변수·지침·모델)는 전부 고정한다. 그래서 준수율의 변화를
> "내용"이나 "지침"이 아니라 **선행에 남은 준수 예시 수** 하나로 귀속시킬 수 있다.

### 재는 것 (측정 지표)

- **준수율:** 생성 첫 함수가 camel인가 → `n_compliant` 단계별 비율
- **자기증폭:** 첫 함수 위반 시, 2·3번째 함수로 위반이 연쇄되는 비율

### 결과 해석 (어느 쪽이 나오든 그대로 기록, §4)

| 나온 결과 | 해석 | 다음으로 |
|---|---|---|
| 준수 ≥1개까진 높게 유지되다가 **0개에서 급락** | **절벽 재현** = RQ1 확인 | RQ2("무엇이 매개하나")로 |
| 8→0 **매끄러운 하락** | 절벽 아님 (단순 비례) | ICL 오버라이드와 구별 필요 |
| 첫 함수 위반 → 뒤 함수 연쇄 위반 | **자기증폭 확인** | 자기 출력이 다시 입력이 되는 경로 |

---

## 3. 쓰는 데이터셋

| 항목 | 값 |
|---|---|
| 데이터 종류 | **합성(SYNTHETIC)** |
| 선행 구성 | `CLONE` — 같은 함수 1종을 인덱스 1..12로 12복제 |
| 선행 과제 | `CLONE_TASK` (`scale value`, `src/harness/tasks.py`) |
| 생성 과제 | `GENERATION_TASKS` 3종 (`remove duplicates` → `count vowels` → `merge dicts`) |
| POOL(50×50) | **미사용** — stepA는 CLONE이라 이름 풀이 필요 없다 |
| 총 관측 | 9단계 × 56 seed = **≈500** |

### 선행 과제 `CLONE_TASK`

| 항목 | 값 |
|---|---|
| 단어 | `scale`, `value` → `scaleValue{i}` / `scale_value_{i}` |
| 설명 | returns the value multiplied by a factor |
| 매개변수 | `value`, `factor` |
| 본문 | `return value * factor` |
| 렌더 수 | 12 (인덱스 1..12) |

### 생성 과제 `GENERATION_TASKS` (측정 대상)

| # | 이름(단어) | 설명 | 기대 이름(camel/snake) |
|---|---|---|---|
| 1 | remove / duplicates | removes duplicate items from a list, preserving order | `removeDuplicates` / `remove_duplicates` |
| 2 | count / vowels | counts the vowels in a string | `countVowels` / `count_vowels` |
| 3 | merge / dicts | merges two dictionaries | `mergeDicts` / `merge_dicts` |

> 원래 1번 과제는 `clamp`였으나 모델이 한 단어 `clamp`로 축약해 표기 판정 불가(`other`)가 되어,
> **두 단어가 강제되는** `remove duplicates`로 교체했다(stepA-1·A-2 결과 권고).

---

## 4. 데이터를 어떻게 구성했는가

데이터는 저장된 파일이 아니라 **매 실행 시 조건(seed 포함)으로 조립**한다.
핵심은 **변하는 것을 이름 표기 하나로 좁히는 통제**다.

### 통제 원칙

- 선행 12개는 **본문·매개변수를 고정**하고 이름 표기(camelCase / snake_case)만 바꾼다.
  → 준수율 차이가 "내용"이 아니라 "형태(표기)"에서 왔음을 깨끗하게 가른다.
- 이름은 **두 단어 이상**이어야 한다. 한 단어는 camel == snake라 표기 구분이 안 된다.
  → CLONE의 `scale value`, 생성의 `remove duplicates` 등 전부 두 단어.
- 생성 3개는 선행과도, 서로와도 **겹치지 않게** 한다.
  → 이름 복사(모방)를 막아 모델의 "직접 선택"만 측정한다.

### 조립 절차 (한 조건 = 준수 개수 `n` + seed 1개)

1. **표기 배열** — 준수(camel) `n`개 + 위반(snake) `12-n`개로 12칸짜리 표기 리스트를 만들고, `seed`로 순서를 셔플한다.
2. **함수 렌더** — `CLONE_TASK`를 인덱스 1..12로 12복제하고, 각 자리를 표기 리스트대로 camel/snake로 찍는다. (예: `scaleValue1`, `scale_value_2`, `scaleValue3`, …)
3. **프롬프트 조립** — `system`(지침 문장) + `user`(선행 12개 + `Add a function that removes duplicate items …` 생성 요청). 이후 턴은 생성 과제 2·3만 추가.

> 본문·매개변수가 고정이라 **seed를 바꾸면 오직 준수/위반의 배치 순열만 달라진다.**
> 그래서 같은 `n_compliant`라도 seed 56개로 반복하면, 관측된 효과가 특정 배치의 우연이
> 아님을 보장한다. (stepA는 CLONE이라 seed가 이름을 바꾸지는 않는다 — 바뀌는 건 위치뿐.)

### 실제 프롬프트 예시 (`n_compliant=2`, 지침=camel)

```
[system] You are helping extend an existing Python module.
         In this project we generally write function names in camelCase.
         Every function name uses one of two styles only: camelCase or snake_case.

[user]   Here is the current module:
         ```python
         def scaleValue1(value, factor):
             return value * factor
         def scale_value_2(value, factor):     ← camel/snake 위치는 seed가 셔플
             return value * factor
         ... (총 12개, 이 중 camel 2개 / snake 10개) ...
         ```
         Add a function that removes duplicate items from a list, preserving order.
```
