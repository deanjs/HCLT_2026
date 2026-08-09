# step A — 과제 스펙과 프롬프트

step A 생성 경로(`runner.py`의 4단계)가 실제로 쓰는 과제 정의와 프롬프트. 코드 위치는
`src/harness/{tasks,naming,prompt}.py`, 조립은 `runner.run()`.

## ① 과제 스펙 (`tasks.py`)

각 과제 = `(단어들, 설명, 매개변수, 본문)`. 이름은 단어들을 camel/snake로 **렌더링**해 만든다.
표기가 구분되려면 **두 단어 이상**이어야 한다(한 단어면 camel==snake).

```python
render_camel(["scale","value"], idx=2)  -> "scaleValue2"
render_snake(["scale","value"], idx=3)  -> "scale_value_3"
```

**선행용 clone 과제 1개** (`CLONE_TASK`) — 같은 일을 하는 12개를 인덱스로 찍어낸다:

| 필드 | 값 |
|---|---|
| 단어 | `("scale","value")` |
| 설명 | returns the value multiplied by a factor |
| 본문 | `return value * factor` |
| 12개 | `scaleValue1..12` / `scale_value_1..12` 중 n_compliant개만 camel |

**생성용 과제 3개** (`GENERATION_TASKS`, 서로 다르고 선행과도 다름):

| # | 단어 | 설명 |
|---|---|---|
| 1 | `("clamp","number")` | clamps a number between a low and high bound |
| 2 | `("count","vowels")` | counts the vowels in a string |
| 3 | `("merge","dicts")` | merges two dictionaries |

(step A-1용 distinct 12개 풀은 그 step에서 이 파일에 추가.)

## ② 프롬프트 (`prompt.py`)

**형태:** system(지침) + user(기존 코드 + 새 함수 요청)를 **3번 순차**. 모델의 이전 답이
대화에 남아 다음 입력이 되므로 자기증폭이 자연히 누적된다.

**system — 지침** (`build_instruction_text`, 긍정·camel·약한 어조·닫힌 후보):
```
You are helping extend an existing Python module.
In this project we generally write function names in camelCase.
Every function name uses one of two styles only: camelCase or snake_case.
```
- 부정형이면 둘째 줄이 `please avoid writing function names in snake_case.`로 바뀐다.
- 강한 어조면 `generally`→`always`(긍정) / `avoid`→`never`(부정).

**user 턴 1** (`first_user_message`, 선행 12 + 첫 요청) — 예: `n_compliant=2`:
```
Here is the current module:

​```python
def scale_value_1(value, factor):
    return value * factor

def scaleValue2(value, factor):
    return value * factor
... (총 12개, 2개만 camel, seed로 위치 셔플) ...
​```

Add a function that clamps a number between a low and high bound.
```

**user 턴 2·3** (`next_user_message`):
```
Add a function that counts the vowels in a string.
Add a function that merges two dictionaries.
```

## ③ 측정 (`runner._run_generation`)

각 턴 응답에서 첫 `def <name>(`를 파싱(`first_def_name`) → `classify_name`으로 camel/snake/other.

| 지표 | 정의 |
|---|---|
| `compliance_rate` | 턴1 표기 == 목표(camel)면 1.0, 아니면 0.0 (조건별 평균은 seed 집계에서) |
| `extra.turn_notations` | 세 턴의 표기 `["camel","snake",...]` |
| `extra.first_violated` | 턴1이 위반인가 |
| `extra.subsequent_violation_rate` | 턴1 이후 위반 비율 = **자기증폭** |

## ④ 테스트 (`tests/test_stepA.py`)

모델 없이 `generate_fn` seam으로 파이프라인을 끝까지 돌린다: 렌더링·판정, 선행 12개의
camel/snake 개수(=n_compliant), seed 셔플 결정성, 지침 문구, 생성 경로 준수율·자기증폭 집계.
실제 100회 생성은 Colab 노트북에서. (`python3 -m pytest -q` → 24 passed)
