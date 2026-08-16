# step4 코드 — 지침 관측 (RQ3 관측)

> **읽기 전:** [`../코드_하네스공통.md`](../코드_하네스공통.md) · [`../step2/code.md`](../step2/code.md)(같은 관측 경로)
> **방법·해석:** [`방법론.md`](방법론.md) · [`results.md`](results.md)
> **결과:** `results/step4_instr-observe/` 336개 · **노트북:** `notebooks/step4_instr-observe.ipynb`

---

## 1. 한 문장

**step2와 같은 관측 장치를 쓰되, 보는 대상을 코드 이름에서 "지침 문장의 지시어"로 옮긴다.**
"모델이 지침을 덜 봐서 어기는가"에 답한다.

```python
run(condition, handle, mode="observe")     # step2와 같은 mode. 조건만 다르다
```

**코드 경로가 step2와 완전히 동일하다.** 다른 것은 `notation_spans`에 무엇을 넣느냐뿐 —
그것이 "RQ별 스크립트를 만들지 않는다" 원칙의 가장 좋은 예다.

---

## 2. 이 스텝만의 코드 — 지시어를 역할별로 나눈다

### 2-1. 문제: 지침은 표기어를 두 곳에서 말한다

```
You are helping extend an existing Python module.
In this project we generally write function names in camelCase.      ← 규칙문 (진짜 지시)
Every function name uses one of two styles only: camelCase or snake_case.  ← 후보열거 (대칭 나열)
```

| 표기어 | 등장 횟수 | 어디 |
|---|---|---|
| 요구 표기 `camelCase` | **2회** | 규칙문 + 후보열거 |
| 반대 표기 `snake_case` | **1회** | 후보열거만 |

**그냥 합쳐서 재면 요구어가 "많이 나와서" 부풀려진다.** step4 초판이 그렇게 쟀고,
"모델이 요구 표기어를 더 본다"는 결론이 나왔다 — 토큰이 2배라서일 수 있는데도.

### 2-2. 해결: 앵커로 잘라 역할별 구간을 만든다

```python
# prompt.py:74
_CANDIDATE_ANCHOR = "one of two styles only:"      # 이 앞 = 규칙문, 뒤 = 후보열거

# prompt.py:77  instruction_notation_spans
ci = text.find(_CANDIDATE_ANCHOR)
for a, b in occ(camel):
    (cand_camel if ci >= 0 and a >= ci else rule_word).append((a, b))
for a, b in occ(snake):
    (cand_snake if ci >= 0 and a >= ci else rule_word).append((a, b))
return {"instr_rule_word": rule_word,      # 규칙문의 지시어 — "지침을 보는가"의 핵심 신호
        "instr_cand_camel": cand_camel,    # 후보열거의 camelCase
        "instr_cand_snake": cand_snake}    # 후보열거의 snake_case
```

이제 **공정한 비교 두 개**가 가능해진다:

| 비교 | 왜 공정한가 |
|---|---|
| `instr_rule_word` vs `code_camel` | 지시어 1회 등장 vs 코드 이름들 |
| `instr_cand_camel` vs `instr_cand_snake` | 둘 다 후보열거에 1회씩 — **완전 대칭** |

초판 키(`instr_target_word`·`instr_viol_word`)도 **지우지 않고 함께 저장**한다.
결과는 불변이고(§6), 초판과 개선판을 나란히 볼 수 있어야 한다.

```python
# runner.py:769
notation_spans = {
    "instr_target_word": _STYLE[ins.target_notation],       # 초판: 프롬프트 전체에서 그 단어 전부
    "instr_viol_word":   _STYLE[ins.violation_notation],
}
notation_spans.update(instruction_notation_spans(condition))  # 개선판: 역할별 3구간
```

### 2-3. 두 종류의 구간 지정을 한 함수가 받는다

`notation_spans`의 값이 **문자열이면 전역 검색, 튜플 목록이면 지침 내부 offset**이다.

```python
# model.py:165
inst_start = inst_char[0][0] if inst_char else None      # 지침이 프롬프트에 박힌 시작 위치
for span_name, spec in notation_spans.items():
    if isinstance(spec, str):
        char_spans[span_name] = find_char_spans(prompt_text, [spec])          # 초판
    elif inst_start is None:
        char_spans[span_name] = []
    else:
        char_spans[span_name] = [(inst_start + a, inst_start + b) for a, b in spec]  # 재기준
```

**재기준(re-basing)이 필요한 이유** — `instruction_notation_spans`는 지침 문장만 보고
offset을 계산한다(모델·chat 템플릿을 모른다, 순수 함수라서). 그 값은 지침 문장 내부 기준이므로,
지침이 전체 프롬프트의 어디에 박혔는지를 더해 줘야 실제 위치가 된다.

```
프롬프트:  <|im_start|>system\nYou are helping … in camelCase. …
                              └────────┬───────┘
                              inst_start = 20

지침 내부 offset (60, 69)  →  실제 (80, 89)
```

---

## 3. 파이프라인

```
Condition(composition=POOL, n_compliant=6, instruction=긍정형, target=camel|snake)
   │
   ▼  runner.py:738  _run_observation                    ← step2와 같은 함수
   │
   ├─ groups = {"code_camel": […], "code_snake": […]}    ← step2와 동일
   ├─ notation_spans = {초판 2키} ∪ {역할별 3키}          ← ★ step4만
   │
   ▼  model.py:98  observe_generation_query
   │   ① "def " 교사강제  ② offset_mapping  ③ 문자구간(+재기준)  ④ 토큰 인덱스
   │   ⑤ forward(output_attentions=True)  ⑥ 마지막 query 행 + ‖v‖  ⑦ span_metrics
   ▼
Metrics(per_layer={L: {"instr_rule_word__attention_weight": …, …}},
        extra={span_token_counts, …})
   │
   ▼  scripts/observe_per_token.py         ← 토큰 수로 나눠 재집계 (★ 결론이 뒤집힌 곳)
```

---

## 4. 조건 구성 — 실제로 돈 것

```python
# 4모델 × 목표표기 2 × 블록 42 = 336
for target in (Notation.CAMEL, Notation.SNAKE):
    for b in range(42):
        Condition(model=…, seed=42,
                  preceding=PrecedingCode(n_compliant=6, n_functions=12,
                                          composition=Composition.POOL, pool_block=b),
                  instruction=Instruction(form=InstructionForm.POSITIVE,
                                          target_notation=target))
```

`n_compliant=6` — **camel 6개 / snake 6개 균형 배치**다. 코드 쪽 두 그룹의
어텐션을 비교하려면 개수가 같아야 한다.

⚠️ **부정형(`NEGATIVE`)은 돌리지 않았다.** 스키마는 지원하지만 336개 결과가 전부
`form="positive"`다. "부정형 지침에서 반대 표기어를 더 보는가"는 이 데이터로 답할 수 없다.

---

## 5. ★ 결론이 뒤집힌 지점 — 토큰당 평균

`attention_weight`는 구간 **합**이다(→ [`../step2/code.md`](../step2/code.md) §4-5).
그런데 비교하는 두 구간의 토큰 수가 6배 차이 난다:

| 구간 | 토큰 수 (Qwen) |
|---|---|
| `instr_rule_word` (규칙문 지시어) | **2** |
| `code_camel` (코드 이름 6개) | **12** |

합끼리 비교하면 코드 쪽이 유리한 게 당연하다.

```python
# scripts/observe_per_token.py:64
raw[L].append(xa - xb)                    # 합 기준
if na and nb:
    per[L].append(xa / na - xb / nb)      # 토큰당 평균 기준
```

**4모델 전부 부호가 바뀌었다.** 합으로 보면 코드 이름이 더 많이 받지만,
토큰당으로 보면 **지침 지시어가 더 많이 받는다.**

> 이게 이 연구의 반전이다. "모델이 지침을 덜 봐서 어긴다"는 통념과 반대로,
> 모델은 **토큰 하나하나 기준으로 지침을 더 보고 있으면서도** 어긴다.
> → 문제는 어텐션이 아니라 내용(Value)에 있다는 step5·step6의 주장으로 이어진다.

재실험은 필요 없었다 — `span_token_counts`가 결과에 함께 저장돼 있어
**불변 규약을 지키면서 다시 계산**할 수 있었다(§6). 그 필드를 안 남겼으면
336개를 다시 돌려야 했다.

---

## 6. 결과 JSON 읽는 법

```jsonc
"metrics": {
  "per_layer": {
    "27": {
      "instr_rule_word__attention_weight":  0.0121,   // 규칙문 지시어 ← 핵심
      "instr_rule_word__av_norm":           0.28,
      "instr_cand_camel__attention_weight": 0.0043,   // 후보열거 camel  ┐ 완전 대칭
      "instr_cand_snake__attention_weight": 0.0041,   // 후보열거 snake  ┘
      "instr_target_word__attention_weight":0.0164,   // 초판(합침) — 참고용
      "instr_viol_word__attention_weight":  0.0041,
      "instruction__attention_weight":      0.0312,   // 지침 문장 전체
      "code_camel__attention_weight":       0.0184,
      "code_snake__attention_weight":       0.0231
    }, …
  },
  "extra": {
    "instruction_form": "positive",
    "target": "camel",
    "span_token_counts": {"code_camel": 12, "code_snake": 12, "instruction": 37,
                          "instr_target_word": 4, "instr_viol_word": 2,
                          "instr_rule_word": 2, "instr_cand_camel": 2,
                          "instr_cand_snake": 2},        // ★ 없으면 비교 불가
    "gqa": {...}, "seq_len": 512, "token_detail": {...}
  }
}
```

읽는 순서:

1. `span_token_counts`를 **먼저** 본다. 토큰 수가 다르면 합끼리 비교하지 않는다.
2. `instr_rule_word`를 쓴다. `instr_target_word`는 초판 키(2회 등장으로 부풀려짐).
3. 후보열거 두 키(`instr_cand_*`)는 **대칭 통제**다. 여기서 차이가 크면
   측정 장치 자체를 의심해야 한다.

```bash
python scripts/observe_per_token.py results/step4_instr-observe
```

---

## 7. 처음과 달라진 것 (코드·개념)

| 무엇 | 처음 | 지금 | 왜 |
|---|---|---|---|
| 지시어 구간 | 프롬프트 전체에서 단어 검색(합침) | **규칙문 / 후보열거 분리** | 요구어가 2회 나와 불공정 |
| 비교 기준 | 구간 합 | **토큰당 평균** | 4모델 전부 부호가 뒤집혔다 |
| 초판 키 | — | 함께 보존 | 결과 불변 + 초판/개선판 대조 |
| offset 기준 | 지침 내부 그대로 | **프롬프트 기준으로 재기준** | 지침이 박힌 위치만큼 어긋났다 |
| 못 찾은 구간 | `None` 저장 | **예외** | "안 봤다"로 오독됐다 |

---

## 8. 직접 확인 (GPU 불필요)

```bash
# 지침이 실제로 어떻게 잘리는지
python - <<'PY'
from harness.conditions import *
from harness.prompt import build_instruction_text, instruction_notation_spans
c = Condition(model=ModelSpec(name="x/y", family="x"),
              preceding=PrecedingCode(n_compliant=6, composition=Composition.POOL),
              instruction=Instruction(form=InstructionForm.POSITIVE,
                                      target_notation=Notation.CAMEL))
t = build_instruction_text(c); print(t); print()
for k, spans in instruction_notation_spans(c).items():
    print(f"{k:20} {spans}  →  {[t[a:b] for a,b in spans]}")
PY

pytest tests/test_step4.py -q
```

기대 출력 — `instr_rule_word`는 `['camelCase']` 하나,
`instr_cand_camel`·`instr_cand_snake`는 각각 하나씩.
