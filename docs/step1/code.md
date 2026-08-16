# step1 코드 — 준수율 절벽 (RQ1)

> **읽기 전:** [`../코드_하네스공통.md`](../코드_하네스공통.md) (모듈 지도·HuggingFace 기초)
> **방법·해석:** [`방법론.md`](방법론.md) · [`results.md`](results.md)
> **결과:** `results/step1_cliff/` 2184개 · **노트북:** `notebooks/step1_cliff.ipynb`

---

## 1. 한 문장

**모델을 그냥 돌려서 나온 이름을 정규식으로 판정한다.** 모델 내부를 전혀 건드리지 않는
유일한 스텝이다 — 그래서 코드가 가장 단순하고, 나머지 스텝의 "행동" 기준점이 된다.

```python
run(condition, handle, mode="generate", max_turns=1)   # intervention.kind = NONE
```

---

## 2. 쓰인 개념 → 코드 대응

| 개념 | 트랜스포머의 어디 | 코드 | 이 스텝에서 하는 일 |
|---|---|---|---|
| chat 템플릿 | 입력 문자열 | `tokenizer.apply_chat_template` | system(지침)+user(코드) → 모델이 아는 형식 |
| 자기회귀 생성 | 전체 forward 반복 | `model.generate` | 다음 토큰을 하나씩 뽑는다 |
| 그리디 디코딩 | 로짓 argmax | `do_sample=False` | 온도 0 → **같은 입력이면 늘 같은 출력** |
| 대화 히스토리 | 프롬프트 누적 | `messages.append({"role": "assistant", …})` | 자기증폭 측정을 위한 장치(미사용, §6) |
| 표기 판정 | — (순수 문자열) | `naming.classify_name` | camel / snake / other |

**내부는 안 본다.** 어텐션도 KV도 잔차도 만지지 않는다. 이 스텝의 값은
"모델이 실제로 어떻게 행동하는가"이고, step2~6이 그 행동의 기제를 판다.

---

## 3. 파이프라인

```
Condition(preceding.n_compliant=k, instruction=camel, lang=python|javascript, seed=42)
   │
   ▼  runner.py:75  run(mode="generate", intervention=NONE)
runner._run_generation                                            (runner.py:809)
   │
   ├─ prompt.build_instruction_text(condition)          → system 메시지
   │     "You are helping extend an existing Python module.
   │      In this project we generally write function names in camelCase.
   │      Every function name uses one of two styles only: camelCase or snake_case."
   │
   ├─ prompt.first_user_message(condition)              → user 메시지
   │     └─ build_preceding_code   : 이름 풀에서 12개 뽑아 표기 배치 후 렌더
   │
   ├─ handle.chat_generate(messages)                    → 생성 텍스트   (model.py:61)
   │     apply_chat_template → model.generate(do_sample=False) → decode
   │
   ├─ naming.first_function_name(text, lang)            → "removeDuplicates"
   └─ naming.classify_name(name)                        → "camel"
   │
   ▼
Metrics(compliance_rate=1.0 if 표기==목표 else 0.0, extra={turn_*, …})
   │
   ▼  results/step1_cliff/<슬러그>.json
```

---

## 4. 핵심 코드 읽기

### 4-1. 선행 코드 만들기 — 위반을 몇 개 섞을지

절벽 실험의 조작 변수는 **12개 함수 중 규약을 지킨 개수**다.

```python
# prompt.py:143  build_preceding_code
notations = [target] * p.n_compliant + [violation] * (p.n_functions - p.n_compliant)
random.Random(condition.seed).shuffle(notations)      # 위치를 섞는다
...
idxs  = _pool_indices(condition)                      # 블록 b의 이름 12개
funcs = [NAME_PAIR_POOL[j].render(nt, lang=lang) for j, nt in zip(idxs, notations)]
return "\n\n".join(funcs)
```

읽을 점 세 가지:

1. **개수는 조건이, 위치는 시드가 정한다.** `n_compliant`가 수량, `shuffle`이 배치.
   시드를 바꿔도 "준수 3개"는 그대로고 어디에 놓이는지만 바뀐다.
2. **`random.Random(seed)`를 매번 새로 만든다.** 전역 `random`을 쓰면 호출 순서에
   따라 결과가 달라진다. 여기서는 같은 조건이면 언제 불러도 같은 코드가 나온다.
3. **`_pool_indices`를 세 함수가 공유한다**(`build_preceding_code`·`preceding_specs`·
   `unrelated_specs`). 각자 계산하면 개입 실험에서 "치환하려는 이름"과
   "실제 프롬프트의 이름"이 어긋난다.

### 4-2. 이름 렌더링 — camel과 snake는 같은 단어에서 나온다

```python
# naming.py:17
def render_camel(words, idx=None):
    first, *rest = words
    return first.lower() + "".join(w.capitalize() for w in rest)   # ("parse","header") → parseHeader

def render_snake(words, idx=None):
    return "_".join(w.lower() for w in words)                      # ("parse","header") → parse_header
```

**같은 단어쌍에서 두 표기를 만든다.** 그래서 표기 외의 모든 것(의미·길이·희귀도)이
통제된다. 이게 이 연구의 자극 설계 전체를 관통하는 원칙이다.

⚠️ 단어가 하나면 `camel == snake`라 판정이 불가능하다. 그래서 이름 풀은 전부
**동사+명사 두 단어**다(`tasks.py:127`).

### 4-3. 표기 판정 — 정규식 두 개

```python
# naming.py:34
_CAMEL_RE = re.compile(r"^[a-z][a-z0-9]*(?:[A-Z][a-z0-9]*)+$")   # 소문자 시작 + 대문자 경계 ≥1
_SNAKE_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")       # 밑줄 경계 ≥1

def classify_name(name):
    if _SNAKE_RE.match(name): return "snake"
    if _CAMEL_RE.match(name): return "camel"
    return "other"
```

| 입력 | 판정 | 왜 |
|---|---|---|
| `removeDuplicates` | camel | 소문자 시작 + 대문자 경계 |
| `remove_duplicates` | snake | 밑줄 경계 |
| `RemoveDuplicates` | **other** | 대문자 시작 = PascalCase |
| `remove` | **other** | 경계가 없다 |
| `xt_remove_duplicates` | snake | 밑줄이 있으므로 |

`other`가 있는 게 중요하다. 두 값으로만 나누면 판정 불가를 억지로 한쪽에 넣게 된다.
**step6에서 이 설계가 결정적으로 작동했다** — 조향을 세게 걸었더니
`RemoveDuplicates`(PascalCase)가 나왔고, `other`로 잡혀 "준수 아님"이 됐다.
두 값만 있었으면 camel로 세어 성공으로 오독했을 것이다.

### 4-4. 이름 뽑기 — 언어별 정규식

```python
# naming.py:49
_DEF_RE = re.compile(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")            # Python

_JS_RES = (                                                                # JavaScript
    re.compile(r"\bfunction\s*\*?\s+([A-Za-z_$][\w$]*)\s*\("),             # function foo(
    re.compile(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*"
               r"(?:async\s+)?(?:function\b|\(|[A-Za-z_$][\w$]*\s*=>)"),   # const foo = () =>
)

def first_js_name(text):
    best = None
    for rgx in _JS_RES:
        m = rgx.search(text)
        if m and (best is None or m.start() < best[0]):   # ← 더 앞에 나온 것을 고른다
            best = (m.start(), m.group(1))
    return best[1] if best else None
```

JS는 정의 형태가 여러 가지라 정규식이 둘이다. 각각 `search`한 뒤
**문서에서 더 앞에 있는 것**을 고른다 — 단순히 첫 정규식을 우선하면
`const foo = () => …`가 뒤의 `function bar()`에 밀린다.

`lang`은 조건에서 온다:

```python
# prompt.py:38
def _lang(condition):
    p = condition.preceding
    if p.source is Source.REPO:  return p.repo_lang
    return p.lang or "python"
```

### 4-5. 생성 루프

```python
# runner.py:832
messages = [{"role": "system", "content": build_instruction_text(condition)}]
for turn in range(n_turns):
    user = first_user_message(condition) if turn == 0 else next_user_message(turn)
    messages.append({"role": "user", "content": user})
    text = generate_fn(messages)
    messages.append({"role": "assistant", "content": text})   # ← 히스토리에 쌓인다
    names.append(first_function_name(text, lang))
    notations.append(classify_name(name) if name else "other")
```

`generate_fn`이 주입 가능한 것에 주의:

```python
# runner.py:821
if generate_fn is None:
    generate_fn = lambda msgs: handle.chat_generate(msgs, max_new_tokens=…, seed=condition.seed)
```

**모델 없이 파이프라인을 테스트하기 위한 이음매(seam)**다. 테스트는 가짜
`generate_fn`을 넣어 프롬프트 조립·판정 로직만 GPU 없이 검증한다.

### 4-6. 실제 생성 호출

```python
# model.py:61  chat_generate
enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                              return_tensors="pt", return_dict=True)   # ← return_dict 필수
enc = {k: v.to(self.model.device) for k, v in enc.items()}
input_len = enc["input_ids"].shape[1]
torch.manual_seed(seed)
with torch.no_grad():
    out = self.model.generate(**enc, max_new_tokens=max_new_tokens,
                              do_sample=False, pad_token_id=tok.eos_token_id)
return tok.decode(out[0, input_len:], skip_special_tokens=True)
```

| 줄 | 왜 이렇게 |
|---|---|
| `return_dict=True` | 안 주면 transformers 버전에 따라 텐서/딕셔너리가 갈려 `.shape`에서 깨진다 |
| `**enc` | `attention_mask`를 함께 넘긴다. `input_ids`만 주면 pad 경고 + 잘못된 마스크 |
| `out[0, input_len:]` | **새로 생성된 부분만** 자른다. 안 자르면 프롬프트가 통째로 딸려 온다 |
| `do_sample=False` | 그리디 = 결정적. 시드는 형식상 고정(실제로 영향 없음) |

---

## 5. 알아야 할 필수 요소

### ① 이 스텝만 `max_turns`가 있다

```python
n_turns = len(GENERATION_TASKS) if max_turns is None else min(max_turns, len(GENERATION_TASKS))
```

절벽(첫 함수 준수율)만 보면 1턴이면 충분하다. **실제로 2184개 결과가 전부 1턴이다.**
그래서 `subsequent_violation_rate`는 전부 `null`이고, **자기증폭은 측정되지 않았다**
(→ `results.md` 한계). 코드는 3턴을 지원하지만 돌리지 않았다는 뜻이다.

### ② 선행 코드의 본문은 전부 같다

```python
# tasks.py:143  이름 풀 항목
TaskSpec(words=(vb, nn), description=f"{vb}s the {nn} and returns it",
         params=("value",), body=("return value",))     # ← 전 항목 동일
```

의도된 통제다. 함수들 사이에서 변하는 것을 **오직 이름 표기**로 한정한다.
"모델이 위반 이름을 따라 하는가"를 재려면 내용 차이가 끼면 안 된다.

### ③ 언어는 껍데기만 바꾼다

```python
# tasks.py:31  render
if lang in ("js", "javascript"):
    return f"function {nm}({sig}) {{\n{body}\n}}"
head = f"def {nm}({sig}):"
```

이름·본문(`return value`)은 두 언어에서 동일하다. 파이썬/JS 비교가
**언어 자체의 차이가 아니라 표기 규약 관습의 차이**를 보게 만드는 설계다.
(JS는 camelCase가 관습, Python은 snake_case가 관습 — 그 사전 지식이 개입한다.)

### ④ 슬러그에 언어가 붙는 조건

```python
# conditions.py:329
if p.lang and p.lang != "python":
    pre += f"-{_slugify(p.lang)}"     # python은 생략 → 기존 슬러그와 호환
```

파이썬은 이름에 안 붙는다. 이전 결과 파일명을 깨뜨리지 않기 위한 결정이고,
집계 스크립트가 "이름에 `-javascript`가 없으면 파이썬"으로 읽는다.

---

## 6. 결과 JSON 읽는 법

```jsonc
"metrics": {
  "compliance_rate": 0.0,              // 첫 함수가 목표 표기면 1.0
  "extra": {
    "target": "camel",                 // 지침이 요구한 표기
    "turn_notations": ["snake"],       // 턴별 판정. 길이 1 = 1턴만 돌았다
    "turn_names": ["remove_duplicates"],// 모델이 실제로 쓴 이름 ← 판정의 근거
    "turn_texts": ["def remove_dup…"], // 생성 원문. 눈으로 검증할 수 있다
    "first_compliant": false,
    "first_violated": true,
    "subsequent_violation_rate": null  // 1턴이라 항상 null (§5-①)
  }
}
```

**`turn_names`를 꼭 함께 보라.** `compliance_rate`가 0인 이유가
"snake를 썼다"인지 "이름을 못 뽑았다(other)"인지는 여기서만 갈린다.

집계는 `scripts/step1_reaggregate.py`:

```python
# 조건(모델 × 위반개수 × 언어)별로 묶어 평균 → 절벽 곡선
```

---

## 7. 처음과 달라진 것 (코드)

| 무엇 | 처음 | 지금 | 왜 |
|---|---|---|---|
| 생성 과제 | `clamp` (한 단어) | `removeDuplicates` (두 단어) | 한 단어는 camel==snake라 전부 `other`로 판정됐다 |
| 언어 | 파이썬만 | 파이썬+JS | 표기 관습이 언어마다 다르다 → RQ1 일반화 |
| 이름 추출 | `first_def_name` 고정 | `first_function_name(text, lang)` | JS `function`·화살표 함수를 못 잡았다 |
| 이름 풀 | 80개 | 504개 (앞 80은 순서 보존) | 42블록 × 12로 중복 없이 덮기 |
| 위반율 집계 | 전체 평균 | **첫 함수 위반 조건부** | 조건부가 아닌 값은 자기증폭의 근거가 못 된다 |

---

## 8. 직접 확인 (GPU 불필요)

```bash
# 프롬프트가 실제로 어떻게 생기는지 눈으로 보기
python - <<'PY'
from harness.conditions import *
from harness.prompt import build_instruction_text, first_user_message
c = Condition(
    model=ModelSpec(name="Qwen/Qwen2.5-Coder-3B-Instruct", family="qwen"),
    preceding=PrecedingCode(n_compliant=0, composition=Composition.POOL, pool_block=0),
    instruction=Instruction(form=InstructionForm.POSITIVE, target_notation=Notation.CAMEL),
    seed=42)
print(build_instruction_text(c)); print("---"); print(first_user_message(c)[:600])
print("슬러그:", c.slug())
PY

# 표기 판정이 맞는지
python -c "
from harness.naming import classify_name
for n in ['removeDuplicates','remove_duplicates','RemoveDuplicates','remove','xt_remove_duplicates']:
    print(f'{n:24} {classify_name(n)}')"

pytest tests/test_step1.py -q
```
