# step4 결과 — 모델은 지침을 본다 (RQ3 관측)

> **스텝:** step4 · **RQ:** RQ3 (지침을 얼마나 참조하는가 — 관측)
> **결과:** `results/step4_instr-observe/` 336개 · **재현:** `python scripts/observe_per_token.py results/step4_instr-observe`

---

## 1. 무엇을 어떻게 쟀나

지침 문장은 표기어를 두 곳에서 언급한다.

```
규칙문   : In this project we generally write function names in camelCase.
후보열거 : Every function name uses one of two styles only: camelCase or snake_case.
```

실제 명령은 **규칙문의 지시어 하나**다. 이것을 후보열거와 분리해 잡고,
**새 이름을 쓰기 직전 시점**에 지시어와 코드 이름을 각각 얼마나 참조하는지 층별로 관측한다.

**관측 장치는 step2와 완전히 같다** — 프롬프트 끝에 `"def "`를 붙여 이름이 올 자리를 만들고,
그 **마지막 위치의 query 한 줄**만 층별로 꺼낸다. 자세한 절차는
[`../step2/results.md`](../step2/results.md) §1-2~1-4. 다른 것은 **어느 열을 합치느냐**뿐이다.

**이 스텝의 구간 5종** (토큰 수는 Qwen 기준)

| 구간 | 프롬프트의 어디 | 정확히 무엇 | 토큰 |
|---|---|---|---|
| `instr_rule_word` | system 둘째 줄 | **규칙문의 지시어** — 진짜 명령 | 2 |
| `instr_cand_camel` | system 셋째 줄 | 후보열거의 `camelCase` | 2 |
| `instr_cand_snake` | system 셋째 줄 | 후보열거의 `snake_case` | 2 |
| `code_camel` | user 코드 블록 | camel 이름 6개의 **이름 부분만** | 12 |
| `code_snake` | user 코드 블록 | snake 이름 6개의 **이름 부분만** | 12 |

> 초판 키 `instr_target_word`·`instr_viol_word`도 함께 저장돼 있으나, 규칙문과 후보열거를
> **합쳐** 세므로 요구 표기어가 2회로 부풀려진다. **해석에는 `instr_rule_word`를 쓴다.**

- 모델 4개 × 묶음 42개 × 지침 방향 2종 = 336개. 시드 42.
- **토큰 수로 나눈 값을 쓴다.** 규칙문 지시어는 2~4토큰, 코드 이름 6개는 12~23토큰으로
  길이가 몇 배 다르다. 합끼리 비교하면 긴 쪽이 무조건 이긴다.
  단, **합과 토큰당 평균은 서로 다른 것을 재는 지표**다(→ §3 해석 주의).

---

## 2. 결과

### 역할별 구간을 전부 그리면

이 스텝의 기여는 **표기어를 역할별로 나눈 것**인데, 그동안 그림은 `instr_rule_word` vs
`code_camel` 두 선만 그렸다. 저장된 8개 구간을 역할별로 다 그리면 이렇게 된다.

| Qwen2.5-Coder-3B | DeepSeek-Coder-6.7B |
|---|---|
| ![qwen](figures/spans_qwen.png) | ![deepseek](figures/spans_deepseek.png) |
| **Llama-3.2-3B (범용)** | **StableCode-3B** |
| ![llama](figures/spans_llama.png) | ![stability](figures/spans_stability.png) |

### 「초판 키」가 무엇이고 왜 버렸나

**처음 step4를 돌릴 때는** 지침의 표기어를 이렇게 쟀다 — *프롬프트 전체에서 그 단어가
나오는 곳을 전부 찾아 합친다.* 그렇게 만든 두 구간을 **초판 키**라 부른다.

```
[system]
You are helping extend an existing Python module.
In this project we generally write function names in camelCase.            ← ①
Every function name uses one of two styles only: camelCase or snake_case.  ← ② ③
                                                 └───①과 같은 단어───┘
```

| 초판 키 | 무엇을 합쳤나 | 등장 횟수 |
|---|---|---|
| `instr_target_word` (요구 표기어) | ① 규칙문의 `camelCase` **+** ② 후보열거의 `camelCase` | **2회** |
| `instr_viol_word` (반대 표기어) | ③ 후보열거의 `snake_case` 만 | **1회** |

**여기서 문제가 생긴다.** 이 두 값을 비교해 "모델이 요구 표기어를 더 본다"는 결론을 냈는데,
요구 표기어는 **애초에 두 번 나온다.** 많이 봐서인지 많이 나와서인지 가릴 수 없다.

그래서 앵커(`one of two styles only:`)를 기준으로 문장을 잘라 **역할별로 나눴다**:

| 분리 키 | 무엇 | 등장 | 역할 |
|---|---|---|---|
| `instr_rule_word` | ①만 | 1회 | **진짜 명령** ← 해석에 쓰는 값 |
| `instr_cand_camel` | ②만 | 1회 | 후보 나열 ┐ 역할이 같은 |
| `instr_cand_snake` | ③만 | 1회 | 후보 나열 ┘ **대칭 통제** |

토큰 수로 검산하면 정확히 맞는다(8개 조건 전부):

```
instr_target_word = instr_rule_word + 후보열거 쪽    Qwen 4 = 2 + 2 · DeepSeek 6 = 3 + 3
instr_viol_word   = 반대쪽 후보열거만                Qwen 2 = 2   · DeepSeek 4 = 4
```

**그림에서 `(2 occ.)` · `(1 occ.)`의 `occ.`는 occurrence, 즉 등장 횟수다.**

| Qwen | DeepSeek |
|---|---|
| ![qwen](figures/initial_vs_split_qwen.png) | ![deepseek](figures/initial_vs_split_deepseek.png) |

**그림 읽는 법** — 세 선을 겹쳐 놓았다.

| 선 | 무엇 |
|---|---|
| 회색 파선 `initial: required word (2 occ.)` | 초판 키. 규칙문+후보열거 **합쳐서** 잰 것 |
| 회색 점선 `initial: opposite word (1 occ.)` | 초판 키. 반대 표기어(후보열거만) |
| **초록 실선** `split: rule word (1 occ.)` | **분리 키. 규칙문 지시어만** ← 우리가 쓰는 값 |

초록 실선이 회색 파선보다 **높은 층**에서는, 초판 키가 후보열거(주목도가 낮은 쪽)를
섞는 바람에 규칙문의 신호를 **희석**했다는 뜻이다. 반대로 회색이 높은 층에서는
등장이 잦아 **부풀려졌다**는 뜻이다. 어느 쪽이든 **초판 키로는 "규칙문을 얼마나 보나"를
잴 수 없다.**

> 초판 키는 결과 파일에 그대로 남아 있다(§6 불변 규약). 지웠으면 이 비교 자체가 불가능했다.

---

### ★ 정규화가 이 스텝의 결론을 뒤집는다

`ratio_*`는 **"지시어를 코드 이름보다 얼마나 더 보나"** 를 층마다 하나의 수로 만든 것이다.

```
                규칙문 지시어의 참조량
      비율 = ─────────────────────────
                코드 camel 이름의 참조량
```

**세로축이 로그 눈금**이라 위아래가 대칭으로 보인다. 가로선 **1.0**이 기준이다.

| 값 | 뜻 |
|---|---|
| **1.0 위** | 지시어를 코드보다 **더** 본다 |
| 1.0 | 똑같이 본다 |
| **1.0 아래** | 코드를 더 본다 |

같은 층에서 **두 기준으로 각각** 계산해 겹쳐 그렸다.

| 선 | 어떻게 계산 |
|---|---|
| **초록 실선** `per token (used)` | (지시어 합 ÷ 지시어 토큰 수) ÷ (코드 합 ÷ 코드 토큰 수) |
| 회색 파선 `span sum` | 지시어 합 ÷ 코드 합 — 토큰 수를 무시 |

**두 선이 1.0을 사이에 두고 갈리면, 정규화 여부가 결론을 바꾼다는 뜻이다.**

| Qwen2.5-Coder-3B | DeepSeek-Coder-6.7B |
|---|---|
| ![qwen](figures/ratio_qwen.png) | ![deepseek](figures/ratio_deepseek.png) |
| **Llama-3.2-3B (범용)** | **StableCode-3B** |
| ![llama](figures/ratio_llama.png) | ![stability](figures/ratio_stability.png) |

| 모델 | 1.0 넘는 층 (토큰당) | 1.0 넘는 층 (합) | 전 층 평균(토큰당) | 전 층 평균(합) |
|---|---|---|---|---|
| Qwen | **19 / 36** | 5 / 36 | 2.39 | 0.38 |
| DeepSeek | **24 / 32** | 6 / 32 | 3.25 | 0.68 |
| StableCode | **21 / 32** | 1 / 32 | 2.34 | 0.40 |
| Llama (범용) | **12 / 28** | 1 / 28 | 1.59 | 0.25 |

**두 기준이 1.0을 사이에 두고 갈린다.** 토큰당으로 보면 지시어를 더 보고,
합으로 보면 코드를 더 본다. 네 모델 모두 그렇다.

> ⚠️ **어느 쪽이 "옳은" 것이 아니다.** 둘은 서로 다른 것을 재는 지표다.
>
> | 지표 | 답하는 질문 |
> |---|---|
> | 구간 **합** | 지침 구간 **전체**가 받은 어텐션 질량은 얼마인가 |
> | **토큰당** 평균 | 지침 토큰 **하나**가 코드 토큰 하나보다 강하게 참조되는가 |
>
> 우리 주장("모델이 지침을 거의 안 봐서 어긴다는 설명은 성립하지 않는다")에는
> **토큰당**이 맞는 자다 — 지시어는 2~4토큰, 코드 이름은 12~22토큰이라
> 합끼리 비교하면 길이가 결과를 지배한다.
>
> **다만 이것만으로 "문제는 어텐션이 아니다"까지 갈 수 없다.** 그 판단은 관측이 아니라
> **step5의 인과 개입**이 한다. 이 절은 "덜 본다는 설명이 성립하지 않는다"까지만 주장한다.

**토큰당 참조량 — 붉은 점선이 step5의 인과 봉우리 층**

![qwen](figures/layer_alignment_qwen.png)
![deepseek](figures/layer_alignment_deepseek.png)
![stability](figures/layer_alignment_stability.png)
![llama](figures/layer_alignment_llama.png)

**전 층 평균, 규칙문 지시어 ÷ 코드 이름**

| 모델 | 배수 | 봉우리 층 | 그 층에서 |
|---|---|---|---|
| Qwen2.5-Coder-3B | **1.55배** | **L27** | 코드의 약 10배 |
| DeepSeek-Coder-6.7B | **2.46배** | **L17** | 코드의 약 9배 |
| StableCode-3B | **1.84배** | **L19** | 코드의 약 20배 |
| Llama-3.2-3B | 0.95배 | L12~13 | 코드의 2.5배 |

**코드 특화 3모델은 지침 지시어를 코드 이름보다 토큰당 더 본다.** 범용 모델 Llama만 대등한데,
중반 층(L12~13)에서는 지시어가 앞서고 후반에서 코드가 올라와 전 층 평균이 상쇄된 결과다.
**어느 모델도 "지침을 덜 본다"고 할 수준이 아니다.**

이 결과는 **지침 방향 양쪽(camel·snake)에서 모두** 성립한다.

### 관측 층과 인과 층이 일치한다

| 모델 | 관측 봉우리(step4) | 인과 봉우리(step5) |
|---|---|---|
| Qwen | L27 | **L27** |
| DeepSeek | L17 | **L17** |
| StableCode | L19 | **L19** |
| Llama | L13 | L24 (판정 불가) |

판정 가능한 세 모델에서 **지침을 가장 강하게 참조하는 층과 지침 개입이 실제로 작동하는 층이
정확히 같다.**

---

## 3. 해석

**(1) 준수 실패는 "지침을 안 봐서"가 아니다.**
모델은 지침 지시어를 코드 이름보다 토큰당 더 참조하면서도 지침을 어긴다.
**어텐션을 키우는 처방(Spotlight 계열)은 이미 충족된 조건을 더 채우는 셈이다.**

**(2) 지침 참조는 특정 후반 층에 몰린다.**
Qwen L27, StableCode L19처럼 한 층에서 코드의 10~20배로 솟는다.
"이름을 쓰기 직전 특정 처리 단계에서 지침을 강하게 참조"하는 구조다.

**(3) 범용 모델은 보는 층이 더 이르다.**
Llama는 중반에 지시어를 보고 후반에는 코드로 돌아간다.

---

## 4. 이 스텝이 다루지 않는 것

- **인과가 아니다.** 참조량 관측이다. 그 참조가 행동을 바꾸는지는 step5의 몫이다.
- 요구 표기어는 규칙문과 후보열거에서 2회, 반대 표기어는 1회 나온다.
  **두 등장을 합친 값은 구조적으로 2:1이라 쓰지 않는다.** 규칙문 지시어만 쓴다.
- 코드 구간끼리의 비교는 step2 결과를 인용한다.
