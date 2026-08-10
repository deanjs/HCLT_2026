# step 2 — RQ2 모델 다양성 (최우선)

**한 줄:** step1은 **Qwen2.5-Coder-3B 하나**에서 "표기 형태 신호 = L25 단일 층 · Value 경로 · 형태 · 방향"을 보였다. step2는 **패밀리가 다른 코드 instruct 모델 2~3개**에서 step1을 그대로 반복해, 이 결론이 한 모델의 특성이 아니라 **일반 원리**인지 확인한다.

대응: **RQ2** — 매개 신호의 성격(형태·경로·층)이 모델 간에 보존되는가. → 보존되면 step7 방법론(형태 인식 선택적 스티어링)이 모델 일반적으로 성립.

## step1에서 이어지는 근거

- step1 결론(단일 모델): 회복률 **L25 단일 스파이크**(상대 0.71) / **Value 우세, Key 무반응** / graft 무관(형태) / L25 코사인 국소 딥(방향).
- 남은 질문: 같은 계열의 크기 변주(3B/7B)는 학습 데이터·토크나이저가 같아 "다른 모델"이 아니다. **패밀리가 다른** 모델에서도 같은 그림이 나오는가? → step2.

## 확인할 세 가지 (계획서 §5 step2)

1. **피크 층 위치가 모델마다 다른가** — 절대 층이 아니라 **상대 위치(peak/num_layers)**로 비교.
2. **국소화 자체는 공통인가** — 위치는 달라도 항상 단일/소수 층에 몰리는가. (위험표: "위치는 다르되 항상 국소화된다" 자체가 발견.)
3. **K·V 기여 비율이 보존되는가** — Value 우세가 모델 공통인가, 아니면 모델마다 경로가 다른가.

## 모델 3종

| 모델 | 패밀리 | 크기 | 라이선스(논문) | 비고 |
|---|---|---|---|---|
| `ibm-granite/granite-3b-code-instruct-2k` | IBM Granite | 3B | **Apache-2.0** | 재현성 앵커. 3B로 크기 통일 |
| `deepseek-ai/deepseek-coder-6.7b-instruct` | DeepSeek | 6.7B | DeepSeek License(연구 OK) | 대표성 앵커. 크기 강건성 덤 |
| `stabilityai/stable-code-instruct-3b` | Stability | 3B | Stability Community(연구 무료) | 3B 세 번째 패밀리 |

- 세 패밀리 모두 Qwen과 다름. 대부분 **MHA**로 알려져 있어, Qwen(GQA)과 합치면 **MHA×GQA 아키텍처 일반화**까지 커버(GQA 전용 아티팩트가 아님을 보임).
- **전부 fp16로 T4 구동 목표** → 양자화 불필요(양자화 시 표현 치환 정밀도 검증이 선행돼야 하는데, fp16이면 그 리스크가 없다, 계획서 §5).
- 정확한 HF ID·층수·head 구성은 **착수 시 `config.json`에서 확정**(추정 금지).

## 절차 (세 모델 동일)

**0. sanity 먼저 (실행 전 필수).**
- (a) `config.json`에서 `num_hidden_layers`·`num_attention_heads`·`num_key_value_heads`·`head_dim` 확인 → **GQA group** 확정. 치환은 KV group 단위이므로(하네스 §3) 모델마다 재확인. 하네스는 캐시 `[B, n_kv, seq, d]`를 직접 편집하므로 group 수와 무관하게 동작하지만, 값은 매번 확인해 기록.
- (b) **토크나이저 정렬률** — 이 모델에서 camel/snake 이름이 몇 토큰인지, `align_name_tokens`가 몇 쌍을 스킵하는지. Qwen은 2:2로 스킵 0이었다. 스킵률이 높으면 표본이 약해지므로 결과에 caveat.
- (c) chat template 존재 확인(`apply_chat_template`). 없으면 그 모델은 제외/보류.

**1. step1 그대로 반복 (코드 변경 없음, ModelSpec만 교체).**
- 개입 스윕: 전 층 × {key / value / key_value} × graft 2종 × seed. (`intervene_preference_sweep`)
- v 코사인 스윕: 같은 이름 camel/snake v의 층별 코사인. (`name_v_cosine_sweep`)

**2. 결과 저장.** `results/step2/<model_slug>/`에 불변 저장(§6). 모델별 `docs/step2/<model>/results.md`.

## 조건 축 (step1과 동일)

| 축 | 값 |
|---|---|
| preceding | POOL n_compliant=0, n_functions=12 (전부 위반 snake) |
| instruction | positive · camel · weak |
| intervention | kind ∈ {key, value, key_value}, layers="sweep", graft ∈ {compliant, unrelated_camel} |
| seed | 0..9 (10 seed) |

- graft(이식값) = L25류 치환에서 위반 이름의 Value를 무엇으로 덮어쓰느냐. `compliant`=같은이름-camel, `unrelated_camel`=다른이름-camel(형태 통제).
- **비교는 상대 위치로.** granite-3b·stable-3b·deepseek-6.7b·Qwen-3b는 층수가 다르다(예 Qwen 36). 피크는 `peak_layer/num_layers`로 정규화해 겹쳐 본다.
- deepseek-6.7b는 6.7B라 스윕이 느림(캐시 deepcopy 규모↑). 너무 느리면 그 모델만 seed 축소 가능 — 실행 중 판단·기록.

## 결과 해석 (어느 쪽이든)

| 결과 | 해석 |
|---|---|
| 피크 위치 다르되 **단일/소수 층 국소화 공통** | 국소화가 일반 원리. 방법론은 모델별 1회 스윕으로 층 선택 |
| **Value 우세 보존** | "Value 경로"가 일반. step7 개입을 Value로 |
| 모델마다 K/V 비율 다름 | 두 경로 협력으로 재서술, 방법론을 양쪽 개입으로 |
| graft ①≈② 재현 | 형태 결론이 모델 공통 |
| 피크 층에서 코사인 국소 딥 재현 | 축 B(방향)가 모델 공통 |
| 국소화가 깨지고 여러 층에 분산 | 그 자체가 결과 — Qwen 특이성으로 스코프 축소, 그대로 기록 |

## 실행 전 예측

- 각 모델에서 회복률이 **단일 또는 소수 층에 국소화**된다(위치는 모델마다 다를 수 있음).
- **Value 우세**가 보존된다(Key 단독은 약함).
- graft 통제(`compliant`≈`unrelated_camel`)가 재현된다 → 형태 결론 공통.
- 피크 **상대 위치**는 모델마다 흩어질 수 있으나 후반부 경향(0.5~0.8)일 것으로 예상.

## 브랜치 구조

통합 브랜치 `step2/model-diversity`에 이 plan을 두고, 모델별로 뻗는다:

```
step2/model-diversity   (plan.md, 최종 모델 간 비교 synthesis)
 ├─ step2/granite-3b-code
 ├─ step2/deepseek-coder-6.7b
 └─ step2/stable-code-3b
```

각 모델 브랜치 = 그 모델의 notebook + `results/step2/<model>/` + 모델별 results.md. 셋이 통합 브랜치로 머지된 뒤, **모델 간 비교(피크 상대 위치·K/V 비율·형태 결론 보존)**를 통합 브랜치에서 정리해 main으로 올린다.

산출물은 `results/step2/`에 불변 저장(§6). 예측은 결과와 함께 보존한다.
