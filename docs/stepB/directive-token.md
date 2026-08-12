# step B — directive-token (지침 속 표기 지시어 토큰 관측)

**대응 RQ:** RQ2 관측 보강. stepB pool-500의 "지침 어텐션 평탄"이 **문장 통째** 기준이라, 그 안의 **핵심 지시어 토큰**('camelCase')을 실제로 보는지는 아직 모른다. 이걸 직접 잰다.

---

## 1. 무엇이 궁금한가

stepB는 지침을 **문장 전체(≈37토큰) 한 span**으로만 쟀다. 그 합이 평탄하다고 해서, 지침 안의 **"function names in camelCase"의 `camelCase` 토큰**을 모델이 봤다는 뜻은 아니다.

> **모델이 지침 문장은 보되, 정작 "camelCase 써라"는 그 단어 토큰은 스킵하는가?**

이게 갈리면 Spotlight(지침 어텐션 증폭) 해법의 전제가 성립하는지도 갈린다.

---

## 2. 정확히 무엇을 측정하나

### 새로 잡는 span (표기 단어)
지침 문장에서 표기 단어를 **통째 지침과 별도**로 잡는다. 이 단어는 지침에만 등장하므로 코드·과제와 혼입되지 않는다(고유 문자열).

| span 이름 | 무엇 | 지침=camel일 때 등장 위치 |
|---|---|---|
| `instr_target_word` | 지침이 **요구하는** 표기 단어 = 지시어 | `camelCase` — rule줄 + closed줄 (2곳) |
| `instr_viol_word` | 반대 표기 단어 | `snake_case` — closed줄 (1곳) |
| `instruction` (기존) | 지침 문장 전체 | 37토큰 |
| `code_camel`·`code_snake` (기존) | 선행 코드 이름 토큰 | 조건별 |

> 지침 예시(camel): 2번째 줄 `... function names in **camelCase**.`(지시어) + 3번째 줄 `... only: **camelCase** or **snake_case**.`(열거)

### 각 span에서 재는 값 (기존 지표 그대로)
이름 생성 시점(`def ` teacher-forcing) query 한 행에서, span 토큰에 대해:

| 지표 | 정의 |
|---|---|
| `attention_weight` | span 토큰 어텐션 **합** (query head 평균). softmax라 한 행 합=1 → span이 가져간 **어텐션 몫** |
| `av_norm` (‖av‖) | `a·‖v‖`의 span 합 (잔차 기여) |
| `v_norm` (‖v‖) | span 토큰 ‖v‖ 평균 |
| `n_tokens` | span 토큰 수 |

### 핵심 비교 지표 — **토큰당 어텐션**
```
per-token attention(span) = attention_weight(span) / n_tokens(span)
```
"합"은 토큰 수에 비례하므로(예: 지침 37토큰 vs 지시어 2토큰), **공정 비교는 토큰당**이다.
→ **`instr_target_word`(지시어)의 토큰당 어텐션**을 `code_camel`·`code_snake`(코드 토큰)·`instruction`(전체)와 **같은 층(L25)에서** 비교한다.

---

## 3. 결과를 어떻게 읽나 (어느 쪽이든 §4 기록)

| L25 per-token 관측 | 뜻 | Spotlight |
|---|---|---|
| 지시어 토큰 ≈ 또는 > 코드 토큰 | 지침을 **토큰 단위로도 잘 봄** — 부족 아님 | **헛다리 확정** (우리 주장 강화) |
| 지시어 토큰 ≪ 코드 토큰 | 문장은 읽되 **핵심 단어를 스킵** | **여지 인정** → step4로 지침 Key/Value 인과 확인 |

보조 관측: `instr_target_word` vs `instr_viol_word`(모델이 요구 표기 단어와 반대 표기 단어 중 어느 쪽을 더 보나), 층별 궤적(지시어가 코드처럼 L25에서 급등하나).

---

## 4. 하네스 변경 (측정 로직은 동일)

| 파일 | 변경 |
|---|---|
| `model.py::observe_generation_query` | `notation_spans` 파라미터 추가 — {span이름: 찾을 문자열}을 char_span으로 잡아 관측에 포함 (선택 파라미터, 후방호환) |
| `runner.py::_run_observation` | `_STYLE`로 target/viol 표기 단어를 구해 `notation_spans`로 넘김 |

- **측정 방식은 그대로** (같은 forward, 같은 span_metrics). span 목록에 표기 단어만 추가.
- 표기 단어는 지침에만 등장 → 코드/생성 과제와 **혼입 없음**(검증: camel 지침에서 `camelCase` 2곳·`snake_case` 1곳만 매치).
- 후방호환: `notation_spans` 없으면 기존과 동일. 테스트 80개 통과.

---

## 5. 실행·산출

- 노트북: `notebooks/stepB_directive-token.ipynb`. 조건은 stepB와 동일(View1 flip + View2 스윕), **블록 축소**(기본 10) — 지시어 어텐션은 조건마다 나오므로 층 프로파일엔 소수 블록이면 충분(원하면 최대 42).
- 결과: `results/stepB_directive/` (기존 `results/stepB_scaleup500/` 294와 **분리·불변**, §6).
- 요약: L25 토큰당 어텐션 표(지시어/반대단어/지침전체/코드) + 층별 per-token 궤적 그래프.

---

## 6. 이 측정이 답하는 것 (한 줄)

> **"지침 어텐션 평탄"이 문장 전체가 아니라 `camelCase` 지시어 토큰 수준에서도 성립하는가** — 성립하면 "지침은 부족하지 않다, 어텐션 증폭(Spotlight)은 헛다리"가 토큰 수준에서 확증되고, 아니면 Spotlight의 여지가 열려 step4(지침 Key/Value 인과)로 넘어간다.
