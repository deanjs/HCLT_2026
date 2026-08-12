# step A-2 — scaleup-500 (RQ1 외적 타당성, 실코드, 500 확대 재실행)

> 파일럿(`docs/stepA-2/plan.md`, 12조건 = 2언어 × 3파일 × 2지침)의 **확대 재실행**이다.
> 결론을 바꾸려는 게 아니라 **실파일 수를 대량으로 늘려** n≈500으로 외적 타당성을 단단히 한다.
> 파일럿 결과는 `results/`에 불변으로 남는다(CLAUDE.md §6).

---

## 1. RQ1 — 무엇이 궁금한가

**대응: RQ1 (외적 타당성).** 합성이 아닌 **실제 코드**에서도 형태 신호가 지침을 이기는가. (연구계획서 §3)

**무엇이 궁금한가:** 선행 코드를 진짜 소스 파일로 바꾸고, 언어마다 고정된 표기 관습(Python=snake, JS=camel)을 **자연 실험**으로 이용한다. 모델이 지침보다 **언어 관습(형태)** 을 따르면, stepA·A-1의 현상이 합성 아티팩트가 아니라 실제 코드에서도 성립한다는 뜻이다.

---

## 2. 어떤 실험을 하는가

### 언어 관습 2×2 (실코드는 무조작, 지침만 배정)

| 선행 파일 | 지침=camelCase | 지침=snake_case |
|---|---|---|
| **Python** (관습 snake) | **충돌** | 일치 |
| **JavaScript** (관습 camel) | 일치 | **충돌** |

- 실제 코드는 **손대지 않고**, **지침만 camel/snake로 배정**해 2×2를 만든다.
- **충돌 칸**에서 준수율이 떨어지면 → 모델이 지침보다 **언어 관습(형태)** 을 따른다는 증거.

### 바꾸는 조건 (조건 축)

| 축 | 값 | 파일럿 대비 |
|---|---|---|
| 선행 `source` | `REPO` (실제 파일) | 동일 |
| 선행 `repo_lang` | python / javascript | 동일 |
| 선행 `repo_file` | **77개 실파일** (py 37 + js 40) | 6개 → **대량 확장** |
| 지침 | camel / snake | 동일 |
| 개입 | none | 동일 |

→ **77파일 × 2지침 = 154조건**, 조건당 생성 3함수 = **462 관측 ≈ 500.**

> **왜 seed로 안 늘리나:** 파일럿에서 **지침을 camel↔snake로 바꿔도 출력이 바이트까지 동일**했다(언어 관습이 지침을 완전히 압도). 즉 지침·seed는 변량을 만들지 못한다. 변량은 오직 **파일**에서 나오므로, 500은 **실파일 수를 늘려** 채운다.

### 재는 것

- **준수율(생성):** 새 함수 이름의 표기 → 2×2 각 칸 비율. (내부 개입 없음 — 실코드는 대응쌍 불가)
- **판별 핵심:** 2×2가 **열(지침)로 갈리는가, 행(언어)으로 갈리는가.** 행으로 갈리면 형태(언어)가 지침을 이긴 것.

### 결과 해석 (어느 쪽이든 그대로 기록, §4)

| 나온 결과 | 해석 |
|---|---|
| 일치 칸 높음 / 충돌 칸 낮음 (출력이 **행=언어**로 갈림) | 언어 관습(형태)이 지침을 압도 → 실코드에서도 형태 지배, RQ1 외적 타당성 확보 |
| 충돌 칸도 준수율 유지 (출력이 **열=지침**으로 갈림) | 합성에서만 생긴 인공물일 수 있음 → 주장 범위를 합성으로 축소 |
| 언어별로 방향 다름 | 언어·모델별 prior 강도 차이 → 별도 서술 |

---

## 3. 쓰는 데이터셋

| 항목 | 값 |
|---|---|
| 데이터 종류 | **실코드(REPO)** |
| 선행 | `data/repo_files/{python,javascript}/` 의 **실제 소스 파일** (원문 그대로, 무조작) |
| 파일 수 | python **37** + javascript **40** = **77파일** (파일럿 3+3에서 확장) |
| 생성 과제 | `GENERATION_TASKS` 3종 (`remove duplicates` → `count vowels` → `merge dicts`) |
| 지침 | camel / snake (파일당 둘 다) |
| 총 관측 | 77 × 2 × 3 = **462 ≈ 500** |
| 출처·라이선스 | `data/repo_files/SOURCE.md` 에 파일마다 기록 (허용 라이선스만) |

### 파일럿(현재 번들, 6파일)

| 파일 | 언어 | 출처 |
|---|---|---|
| `python/fnmatch.py`, `textwrap.py`, `string.py` | Python(snake) | CPython v3.11.0 (PSF) |
| `javascript/utils.js`, `buildURL.js`, `formDataToJSON.js` | JavaScript(camel) | axios v1.6.0 (MIT) |

### 확장 번들 (완료, 허용 라이선스만)

| 출처 | 라이선스 | 태그 | 파일 수 |
|---|---|---|---|
| **CPython** stdlib | PSF-2.0 | `v3.11.0` | Python 37 |
| **axios** | MIT | `v1.6.0` | JS 14 |
| **lodash** | MIT | `4.17.21-npm` | JS 26 |

- 선정 기준: **두 단어 이상 이름이 드러나는 파일**(한 단어 이름은 판정 불가 `other`). Python은 def의 snake 두 단어 ≥2, JS는 파일 내 camelCase 두 단어 식별자 ≥3.
- 크기 필터: Python 2.5~25KB / JS 0.4~25KB (모델 컨텍스트 초과 방지, 최대 선행 ≈24.5KB).
- 파일마다 `data/repo_files/SOURCE.md`에 URL·태그·라이선스 기록. 파일럿 6개는 재현성 위해 무조건 포함.

> **파일럿의 0.67 artifact는 해소됨:** 파일럿에서 생성 1번 과제 `clamp`가 한 단어로 축약돼 판정 불가(1/3 손실)였다. 현재 하네스는 `clamp`→`remove duplicates`(두 단어 강제)로 교체돼, 이 잡음이 사라진다.

---

## 4. 데이터를 어떻게 구성했는가

### 통제 원칙

- 실파일은 **원문 그대로** 선행에 싣는다(수정 금지). 조작하는 것은 **지침의 목표 표기(camel/snake)** 뿐.
  → 언어 관습은 손대지 않은 자연 변수, 지침만 배정 변수 → 2×2가 깨끗하게 성립.
- 판정은 언어 무관(camel/snake 판별) — 단, **두 단어 이상 이름**에서만 표기가 드러난다.
- 생성 3과제는 선행 파일의 함수명과 겹칠 일이 없다(다른 과제).

### 조립 절차 (한 조건 = 실파일 1개 + 지침 표기 1개)

1. **선행 로드** — `data/repo_files/<lang>/<file>` 원문을 그대로 선행 코드로 싣는다.
2. **지침 배정** — 같은 파일에 대해 지침 목표를 camel / snake 두 가지로 각각 만든다.
3. **프롬프트 조립** — `system`(지침) + `user`(실파일 원문 + 같은 언어로 `Add a function that …` 생성 요청). 이후 턴은 생성 과제 2·3만 추가.

### 실제 프롬프트 예시 (Python 파일, 지침=camel → **충돌 칸**)

```
[system] You are helping extend an existing Python module.
         In this project we generally write function names in camelCase.

[user]   Here is the current module:
         ```python
         # (실제 CPython 파일 원문 그대로, 예: textwrap.py)
         def _munge_whitespace(text): ...
         def dedent(text): ...
         ```
         Add a function that removes duplicate items from a list, preserving order.
```

→ 관습(snake)과 지침(camel)이 충돌. 모델이 `remove_duplicates`(snake)로 쓰면 **형태(언어)가 지침을 이긴 것.**

---

## 5. 코드 동일 보장 (재현성)

노트북은 실험 로직을 쓰지 않고 파일럿과 **똑같은 `harness.run`을 호출**만 한다.
파일럿의 6파일은 그대로 두고 파일만 추가하므로, 겹치는 조건(**기존 6파일 × 2지침**)은 파일럿(`results/stepA-2/`)과 **동일**하게 나온다.
scaleup 결과는 `results/stepA-2_scaleup500/`에 저장해 파일럿과 분리·불변 보존한다(§6).

> **실파일 번들 완료:** 77파일(CPython v3.11.0 · axios v1.6.0 · lodash 4.17.21-npm)을 `data/repo_files/`에 번들하고 `SOURCE.md`에 출처·라이선스를 기록했다. 노트북은 `data/repo_files/`를 **자동 glob**하므로 파일 수만큼 조건이 잡힌다(현재 77파일 → 154조건).
