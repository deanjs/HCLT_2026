# step 4 모델 다양성 — RQ3 인과, 504 통일 (설계)

**한 줄:** step4(Qwen)에서 본 "지침 레버 = Value, L27"을 **deepseek/Llama/stable**로 확장해,
한 모델 특성인지 일반 원리인지 가른다. step2가 step1(코드)을 여러 모델로 확장한 것의 지침 판.

**대응:** RQ3 **인과**, 모델 일반화. **원자료(불변, §6):** `results/step4_modeldiv/`.

---

## 왜 다시 하나 (이전 4모델 시도의 문제)

첫 4모델 실행(80풀, `results/step4_results/`)은 **비Qwen 3모델에서 `nsub=0`** — 지시어
`camelCase`/`snake_case`가 모델마다 토큰 수가 달라 `token_unit='all'`이 **전부 스킵**됐다.
그 "0.00"은 불활성이 아니라 **측정 실패**였다. 또 이름 풀이 80(구버전)이라 step1/B/C(504)와 규모도
어긋났다. → 두 가지를 바로잡아 재실행한다.

## 바로잡는 것 두 가지

1. **데이터셋 504 통일.** step1/B/C와 동일한 504 풀 + 블록 커버(42블록). (step4 하네스를 pool-500
   베이스에 포팅해 `pool_block` 사용.)
2. **`token_unit='mean'` (mean-pool).** `'all'`은 지시어 토큰 수가 모델마다 달라 비Qwen에서 전부
   스킵(nsub=0), `'last'`는 마지막 1개만이라 신호가 약해 gap 있는 stable도 못 봤다(1차 last 재실행에서
   확인, `results/step4_modeldiv_results/`). **mean-pool = 공여(반대지침) 토큰들의 KV를 평균 내
   위반 지시어의 모든 토큰 자리에 브로드캐스트**한다. 개수 불일치 허용(스킵 없음) + 단어 전체를 덮어
   (last보다 강함) → 모든 모델에서 whole-word급으로 측정. Qwen(2:2)은 사실상 all과 동급 신호.

> **1차 last 재실행 결과(참고, `results/step4_modeldiv_results/`):** Qwen Value>Key@L27(작음),
> stable gap 있으나 last론 0.0x(측정한계), deepseek/granite gap≈0. → last의 한계가 mean-pool 도입의
> 근거. last 원자료는 §6대로 보존.

## gap 스크리닝 (필수)

전이율 = (S_int−S_base)/(S_clean−S_base). 분모 **gap=S_clean−S_base ≈ 0**이면(지침을 텍스트로
바꿔도 행동이 안 변하면) 전이율은 **0으로 나누기 = 무의미**(첫 시도의 granite ±1.3 노이즈가 이것).
→ `|gap| < 1.0` 조건은 **"지침 레버 없음"**으로 분류하고 전이율 숫자는 버린다. gap≈0 자체가
"그 모델에선 지침이 애초에 레버가 아님"이라는 답(granite directive-token '덜 읽힘'과 합치).

## 조건

| 축 | 값 | 개수 |
|---|---|---|
| 모델 | Qwen / deepseek / **Llama-3.2-3B** / stable | 4 |
| 방향 | camel→snake, snake→camel | 2 |
| 선행 | 균형 6/6, 전부위반 0/12 | 2 |
| 블록 | 0–41 (504 이름 전부) | 42 |
| 정렬 | token_unit='mean' (mean-pool) | (고정) |
| 개입 | KV group 치환, 전 층 스윕, 지침 지시어 타깃 | (고정) |

> **모델 선택:** granite(1차 last에서 gap≈0 = 지침 레버 없음, `results/step4_modeldiv_results/`에 기록)
> 대신 **Llama-3.2-3B-Instruct**를 넣는다. Llama는 **게이트 모델**(HF 로그인+라이선스 동의 필요)이고
> **범용 모델**(나머지 3개는 코드 전용)이라, "코드 모델이라 그런가 vs 범용도 그런가" 대조 축이 된다.
> mean-pool이 Llama 토크나이저 분할과 무관하게 동작한다.

**조건 수 계산:**

```
모델당 = 방향 2 × 선행 2 × 블록 42 = 168 스윕
전체   = 168 × 모델 4            = 672 스윕
```

- **168** = 한 모델이 도는 스윕 수. 스윕 1개 = 그 설정에서 **전 층 × Key/Value/Key+Value 3경로**를
  모두 재므로, 168개만으로 모델별 (층×kind) 전이율 곡선이 다 나온다.
- **672** = 4모델 합계.
- 재개 지원(슬러그에 모델·블록·방향·선행·tok 포함 → 파일 분리, 완료분 건너뜀).
- **규모 축소:** 무거우면 `BLOCKS=range(12)` → 방향2×선행2×블록12 = **48/모델, 192 전체**.
  블록만 줄고 504 커버가 144로 줄 뿐, 축 구조는 동일(재개로 나중에 42까지 확장 가능).

## 결과 분기 (모델별)

| 관측 | 판정 |
|---|---|
| gap 충분 + **Value** 전이 | 그 모델도 지침 레버=Value → Spotlight 반박 **일반화** |
| gap 충분 + **Key** 전이 | 그 모델은 어텐션 경로 → **예외**(그대로 기록) |
| **gap≈0** | 지침 레버 없음 → 전이율 무의미, 별도 분류(1차 last에서 deepseek·granite가 이 경우) |

## 연결

Qwen 인과(지침=Value, whole-word 0.94)는 `docs/step4/results.md`(step4/instruction-kv 브랜치).
이 실험은 그 결론의 **모델 일반화**. → 일반화되면 방법론 step5(Value 경로 스티어링)가 모델 보편.
