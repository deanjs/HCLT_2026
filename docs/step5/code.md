# step5 코드 — 지침 인과 (RQ3 인과)

> **읽기 전:** [`../코드_하네스공통.md`](../코드_하네스공통.md) · [`../step3/code.md`](../step3/code.md)(같은 치환 장치)
> **방법·해석:** [`방법론.md`](방법론.md) · [`results.md`](results.md)
> **결과:** `results/step5_instr-cause/` 336개 + `step5_instr-cause-control/` 2016개
> **노트북:** `notebooks/step5_instr-cause.ipynb` · `step5_instr-cause-control.ipynb`

---

## 1. 한 문장

**step3과 똑같은 KV 치환 기계를 쓰되, 바꾸는 대상을 코드 이름에서 "지침 문장의 지시어"로 옮긴다.**
"지침의 영향력도 코드와 같은 통로(Value)로 흐르는가"에 답한다.

```python
run(condition, handle, mode="generate")    # intervention.target = "instruction"
```

---

## 2. step3에서 바뀐 것은 스위치 두 개뿐

```python
Intervention(kind=…, layers="sweep", donor="opposite",
             target="instruction")        # ← 스위치 ①
```

| 스위치 | step3 | step5 | 어디서 갈리나 |
|---|---|---|---|
| ① 치환 대상 | 선행 코드 이름 | **지침 지시어 단어** | `runner.py:147` → `_preference_setup_instruction` |
| ② 찾는 방식 | `def <nm>(`의 이름 부분 | 문자열 `"camelCase"`의 첫 등장 | `span_kind = "literal"` |

```python
# runner.py:146
instr_target = iv.target == "instruction"
setup = _preference_setup_instruction(condition) if instr_target else _preference_setup(condition)
span_kind = "literal" if instr_target else "def_name"
```

**나머지 기계(캐시 편집·평균 덮어쓰기·복구·채점)는 글자 하나 다르지 않다.**
그래서 두 스텝의 값을 같은 자로 잰 것으로 비교할 수 있다.

첫 등장을 쓰는 것에 근거가 있다 — 지침 문장에서 규칙문이 후보열거보다 **앞**에 오므로,
`"camelCase"`의 첫 등장이 곧 실제 지시어 위치다(→ [`../step4/code.md`](../step4/code.md) §2).

```python
# model.py:816
if span_kind == "literal":
    spans = find_char_spans(text, [nm])
    s, e = spans[0]                      # ← 첫 등장 = 규칙문의 지시어
```

---

## 3. 세 프롬프트가 step3과 다르게 만들어진다

step3은 **선행 코드**를 바꿔 세 상태를 만들었다. step5는 **지침**을 바꾼다.

```python
# runner.py:471  _preference_setup_instruction
opp_instruction = replace(condition.instruction, target_notation=violation)   # 지침만 뒤집기
opp_condition   = replace(condition, instruction=opp_instruction)

preceding_text = build_preceding_code(condition)      # ← 선행은 두 프롬프트가 공유한다
base_system = build_instruction_text(condition)       # "camelCase로 써라"
opp_system  = build_instruction_text(opp_condition)   # "snake_case로 써라"
```

| 상태 | 프롬프트 | 기호 | 뜻 |
|---|---|---|---|
| 기준 | 조건 지침 그대로 | `S_base` | 지금 조건 |
| 천장 | **반대** 지침 그대로 | `S_clean` | 지침이 완전히 바뀌면 얼마나 움직이나 |
| 개입 | 기준 텍스트인데 지시어 KV만 반대 지침값 | `S_int` | — |

```
전이율 = (S_int − S_base) / (S_clean − S_base)
```

⚠️ **`comp_messages`가 여기서는 "깨끗"이 아니라 "반대 지침"이다.** 변수 이름은
step3과 공유하느라 그대로지만 의미가 다르다. `S_clean`도 "위반 없는 상태"가 아니라
"지침을 뒤집었을 때의 상태"다. 결과를 읽을 때 반드시 기억할 것.

---

## 4. ★ 통제 세 종류 — 이 스텝의 설계 핵심

```python
# runner.py:511
donor_kind = condition.intervention.donor or "opposite"
if donor_kind == "opposite":
    donor_messages = None                          # donor_cache = comp_cache(반대 지침)
    donor_names = [notation_word(opp_instruction.token_notation)]     # "snake_case"
elif donor_kind == "self":
    donor_messages = msgs(base_system)             # 같은 지침을 한 번 더 forward
    donor_names = [notation_word(condition.instruction.token_notation)]  # "camelCase"
elif donor_kind == "unrelated_word":
    donor_messages = msgs(base_system)
    donor_names = [NEUTRAL_INSTRUCTION_WORD]       # "project"
else:
    raise ValueError(f"지침 개입의 공여 종류가 올바르지 않다: {donor_kind!r} …")
```

| donor | 무엇을 덮나 | 정상이라면 | 무엇을 죽이나 |
|---|---|---|---|
| `opposite` | 반대 지침의 지시어 | 전이 **큼** | (처치) |
| `self` | **같은 지침의 같은 지시어** | 전이 **≈ 0** | 새 정보가 0인데도 움직이면 = 덮어쓰기 자체의 교란 |
| `unrelated_word` | 같은 지침의 `"project"` | 전이 **≈ 0** | 표기와 무관한 내용을 덮었을 때의 기준선 |

**`self` 통제가 이 연구에서 가장 예리한 도구다.** `camelCase`를 `camelCase`로 덮으면
정보는 하나도 안 바뀌고 "덮어쓰는 행위"만 남는다. 그런데도 점수가 움직인다면
그건 전이가 아니다.

실제 결과: **Key 처치(0.106)와 Key 자기통제(0.107)가 소수점 셋째 자리까지 같았다.**
Key 쪽 전이는 전부 덮어쓰기 교란이었다는 뜻 — 어텐션 경로는 지침 정보를 안 실어 나른다.

`"project"`가 중립어로 쓰이는 근거:

```python
# prompt.py:22
NEUTRAL_INSTRUCTION_WORD = "project"
# rule 문장("In this project ...")에 형식·어조와 무관하게 **한 번만** 나온다
```

형식(긍정/부정)이나 어조(강/약)를 바꿔도 등장 횟수가 안 변한다 — 통제로서의 조건이 갖춰졌다.

---

## 5. 파이프라인

```
Condition(intervention=Intervention(kind=…, layers="sweep"|[L], donor="opposite"|"self"|"unrelated_word",
          kinds=("key","value","key_value"), target="instruction"), token_unit="mean")
   │
   ▼  runner.py:123  _run_intervention → _run_intervention_sweep
   │
   ├─ runner.py:471  _preference_setup_instruction        ← ★ step5 전용
   │     base/opp 지침 × 공유 선행 → 세 메시지
   │     viol_names = ["camelCase"] · donor_names = 공여에 따라
   │
   ▼  model.py:455  intervene_preference_sweep(span_kind="literal")
   │     ↳ _preference_context → 층×kind마다 _score_layer_kind    ← step3과 **동일 코드**
   ▼
Metrics(per_layer={L: {"value__recovery": …}}, extra={S_clean, S_base, donor, …})
```

---

## 6. 두 실행분이 나뉜 이유

| 폴더 | 개수 | 무엇 | 층 |
|---|---|---|---|
| `step5_instr-cause` | 336 | **처치만** (`opposite`) | 전 층 스윕 |
| `step5_instr-cause-control` | 2016 | **통제** (`self`·`unrelated_word`) | 봉우리 층 `[L]` + 스윕 |

처치를 먼저 돌려 봉우리 층을 찾고, 그 층에서 통제를 돌렸다.
통제 쪽 조건이 6배 많은 것은 공여 2종 × 층 지정 방식 2종(단일층·스윕) × kind 3종을
전부 덮었기 때문이다.

⚠️ **두 폴더의 `extra` 스키마가 다르다.**

| 필드 | `step5_instr-cause` | `step5_instr-cause-control` |
|---|---|---|
| `gap`·`undecidable` | **없음** (그 뒤에 추가된 필드) | 있음 |
| `recovery`·`layer` | 없음 (`per_layer`에 있다) | 있음 (단일 층 경로) |

처치분에서는 `S_clean`·`S_base`로 직접 계산한다:

```python
gap = abs(extra["S_clean"] - extra["S_base"]);  undecidable = gap < 1.0
```

---

## 7. 결과 JSON 읽는 법

**스윕(처치, `step5_instr-cause`)**

```jsonc
"metrics": {
  "per_layer": {"17": {"value__S_int": …, "value__recovery": 0.31,
                       "key__S_int": …,   "key__recovery": 0.11, …}, …},
  "extra": {
    "mode": "intervene_sweep",
    "intervention_target": "instruction",       // ← step3과 구분되는 지점
    "donor": "opposite_instruction",
    "target": "camel",
    "S_clean": …, "S_base": …,
    "viol_names": ["camelCase"],                // 덮인 단어
    "donor_names": ["snake_case"],              // 덮어넣은 단어
    "n_substituted_tokens": 2,                  // 지시어는 보통 2조각
    "kinds": ["key", "value", "key_value"]
  }
}
```

**단일 층(통제, `step5_instr-cause-control`)**

```jsonc
"extra": {
  "mode": "intervene",
  "donor": "control_self",                      // 또는 control_unrelated_word
  "layer": 17, "kind": "value",
  "S_clean": …, "S_base": …, "S_int": …,
  "recovery": 0.107,                            // ← 여기 있다 (스윕에는 없다)
  "gap": 3.2, "undecidable": false, "undecidable_gap_min": 1.0,
  "viol_names": ["camelCase"], "donor_names": ["camelCase"]   // self = 같은 단어
}
```

`donor_names`를 보면 어떤 통제인지 즉시 알 수 있다:
`["snake_case"]`=처치 · `["camelCase"]`=자기통제 · `["project"]`=음성통제.

집계:

```bash
python scripts/step5_reaggregate.py     # 방향 분리 · 중앙값 · 임계 민감도 · 순효과
```

**방향을 분리해서 보는 이유** — camel→snake와 snake→camel은 난이도가 다르다.
합쳐서 평균 내면 한쪽이 천장에 닿아 있는 것이 가려진다.

---

## 8. 알아야 할 필수 요소

### ① `S`의 부호 규약

```python
# intervention.py:126
def preference_score(logp_compliant, logp_violation):
    return logp_compliant - logp_violation      # 양수 = 조건 지침이 요구한 표기를 선호
```

step5에서 `S_clean`은 **반대 지침** 상태의 점수다. 지침이 실제로 작동한다면
`S_clean < S_base`가 되어 **분모가 음수**다. 회복률 공식은 부호에 무관하게
성립하지만(비율이므로), `S` 값 자체를 그래프로 그릴 때는 방향을 명시해야 한다.

### ② 공여 종류를 틀리면 예외

```python
# runner.py:521
else:
    raise ValueError(f"지침 개입의 공여 종류가 올바르지 않다: {donor_kind!r} "
                     "(opposite | self | unrelated_word)")
```

예전에는 알 수 없는 문자열이 오면 조용히 기본 경로로 빠졌다.
**파일명에는 그 통제 이름이 남고 실제로는 처치가 돌아간** 결과가 생길 수 있었다.

### ③ 지시어는 토큰이 2개뿐이다

`n_substituted_tokens`가 보통 2다(step3은 20~30). 덮는 자리가 훨씬 적으므로
같은 회복률이라도 **자리당 효과는 step5가 훨씬 크다.** 두 스텝의 절대값을
"지침이 코드보다 약하다"로 읽으면 안 된다.

### ④ 생성 경로로는 못 돌린다

```python
# runner.py:668
if iv.target != "code":
    raise NotImplementedError("생성 기반 개입은 아직 target='code'만 지원한다 …")
```

`mode="intervene_generate"`는 `iv.target`을 읽지 않고 늘 코드 이름을 치환한다.
지침 타깃으로 부르면 **파일명만 '지침'이고 내용은 코드를 바꾼 결과**가 남는다. 그래서 막았다.

---

## 9. 처음과 달라진 것 (코드·개념)

| 무엇 | 처음 | 지금 | 왜 |
|---|---|---|---|
| 통제 | 없음(처치만) | **`self`·`unrelated_word` 추가** | 전이율이 전부 덮어쓰기 교란일 수 있었다 |
| 결론 | "지침도 Value로 흐른다" | **Key는 자기통제와 동일 → 어텐션은 아무것도 안 나른다** | 통제를 뺐더니 Key 순효과가 0 |
| 집계 | 방향 합산 평균 | **방향 분리 + 중앙값 + 임계 민감도** | 한쪽 방향이 천장에 닿아 가려졌다 |
| 판정 불가 | 없음 | `gap`·`undecidable` (통제분부터) | 분모가 작은 조건이 섞였다 |
| 공여 검증 | 조용히 기본값 | **예외** | 파일명과 실제가 어긋날 수 있었다 |
| 중립어 | 미정 | `"project"`(rule 문장에 1회 고정) | 형식·어조와 무관해야 통제가 된다 |

---

## 10. 직접 확인 (GPU 불필요)

```bash
# 세 공여가 실제로 어떤 단어를 덮는지
PYTHONPATH=src python - <<'PY'
from harness.conditions import *
from harness import runner
for donor in ("opposite", "self", "unrelated_word"):
    c = Condition(model=ModelSpec(name="x/y", family="x"), seed=42,
        preceding=PrecedingCode(n_compliant=0, composition=Composition.POOL, pool_block=0),
        instruction=Instruction(form=InstructionForm.POSITIVE, target_notation=Notation.CAMEL),
        intervention=Intervention(kind=InterventionKind.VALUE, layers=[17],
                                  donor=donor, target="instruction"),
        token_unit="mean")
    s = runner._preference_setup_instruction(c)
    print(f"{donor:16} 덮이는 단어={s['viol_names']}  덮어넣는 단어={s['donor_names']}  "
          f"별도 공여 forward={'예' if s['donor_messages'] else '아니오'}")
    print(f"{'':16} 슬러그={c.slug()}")
PY
```

기대 출력 — `opposite`는 `['snake_case']`, `self`는 `['camelCase']`,
`unrelated_word`는 `['project']`를 덮어넣는다.
