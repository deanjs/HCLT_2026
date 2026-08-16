# step2 코드 — 코드 신호 관측 (RQ2 관측)

> **읽기 전:** [`../코드_하네스공통.md`](../코드_하네스공통.md) (모듈 지도·HuggingFace 기초)
> **방법·해석:** [`방법론.md`](방법론.md) · [`results.md`](results.md)
> **결과:** `results/step2_code-observe/` 168개 · **노트북:** `notebooks/step2_code-observe.ipynb`

---

## 1. 한 문장

**이름을 쓰기 직전 그 한 순간을 정지시켜, 모델이 문맥의 어디를 얼마나 보는지 층별로 뜯는다.**
개입은 없다(관측). 인과는 step3.

```python
run(condition, handle, mode="observe")
```

---

## 2. 쓰인 개념 → 코드 대응

| 개념 | 트랜스포머의 어디 | 코드 | 이 스텝에서 하는 일 |
|---|---|---|---|
| 강제 접두 조건화 | 입력 끝 | `prompt_text + "def "` | **함수 이름이 올 자리**를 만든다 |
| 어텐션 가중치 | softmax(QKᵀ/√d) | `output_attentions=True` | 마지막 query 행 `[Hq, seq]` |
| Value 크기 | KV 캐시의 V | `values[layer].norm(dim=-1)` | 토큰별 ‖v‖ `[Hkv, seq]` |
| 기여량 **상한** | Σ a·‖v‖ | `attention_probe._token_stats` | 축 A × 축 B (실제 기여량 아님) |
| GQA | Q헤드 ↔ KV헤드 매핑 | `h // group_size` | 헤드 수가 다른 둘을 곱하기 |
| 구간(span) | 토큰 인덱스 집합 | `locate_token_spans` | "코드의 camel 이름들" 같은 묶음 |

**이 스텝이 재는 세 지표는 전부 "마지막 query 한 행"에서 나온다.**
전체 어텐션 행렬 `[seq, seq]` 중 우리에게 필요한 건 마지막 줄 하나뿐이다.

---

## 3. 파이프라인

```
Condition(composition=POOL, pool_block=b, n_compliant=6)   ← camel 6 / snake 6 균형
   │
   ▼  runner.py:738  _run_observation
   │
   ├─ build_preceding_code       → 선행 코드 문자열
   ├─ re.findall(r"def (\w+)\(") → 이름 12개 뽑아서
   │     classify_name으로 groups = {"code_camel": [...], "code_snake": [...]}
   ├─ build_instruction_text     → 지침 문장
   │
   ▼  model.py:98  observe_generation_query
   │
   ① apply_chat_template(tokenize=False) + "def "        ← 교사강제
   ② tok(text, return_offsets_mapping=True)              ← 문자↔토큰 다리
   ③ find_char_spans("def <name>(") → +4 / −1 로 이름만  ← 구간 문자 위치
   ④ locate_token_spans(offsets, char_spans)             ← 구간 토큰 인덱스
   ⑤ model(**enc, output_attentions=True, use_cache=True)  ← forward 1회
   ⑥ 층마다: attentions[L][0,:,-1,:]  → [Hq, seq]  (마지막 query 행)
   │          values[L][0].norm(-1)   → [Hkv, seq] (토큰별 ‖v‖)
   ⑦ span_metrics(...)                                   ← 순수 함수 집계
   │
   ▼
Metrics(per_layer={L: {"code_camel__attention_weight": …, …}}, extra={span_token_counts, token_detail})
```

---

## 4. 핵심 코드 읽기

### 4-1. 교사강제 — "그 순간"을 인위적으로 만든다

```python
# model.py:134
prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
prompt_text = prompt_text + forced_prefix        # forced_prefix = "def "
```

왜 필요한가 — 그냥 생성하면 모델이 설명문("Sure, here's…")부터 쓸 수 있고,
**이름을 결정하는 순간이 어디인지 알 수 없다.** `"def "`를 붙여 두면 다음 자리가
**함수 이름이 올 자리**가 된다. 그 시점의 query 한 행이 우리 관측 대상.

> 엄밀히는 정답 토큰을 먹이는 teacher forcing이라기보다 **강제 접두 조건화**(forced-prefix
> conditioning)다. 모델이 반드시 이름을 낸다는 보장도 없다 — 실제로 StableCode는
> `xt_remove_duplicates` 같은 접두가 붙은 출력을 낸다(→ `../step6/results.md`).

`tokenize=False`인 이유는 **직접 토큰화하면서 offset을 받아야** 하기 때문(4-2).

### 4-2. 문자 위치 → 토큰 인덱스

```python
# model.py:140
enc = tok(prompt_text, return_tensors="pt",
          return_offsets_mapping=True, add_special_tokens=False)
offsets = [tuple(o) for o in enc.pop("offset_mapping")[0].tolist()]
```

`add_special_tokens=False`가 필수다 — chat 템플릿이 이미 특수 토큰을 넣었는데
여기서 또 BOS를 붙이면 **BOS가 두 번** 들어간다. (offset과 input_ids에 함께 추가되므로
인덱스가 어긋나지는 않는다. 문제는 **모델이 보는 입력 분포가 달라진다**는 것이다.)

이름 구간은 `def <name>(`로 찾은 뒤 앞뒤를 깎는다:

```python
# model.py:155
for s, e in find_char_spans(prompt_text, [f"def {nm}("]):
    ranges.append((s + 4, e - 1))     # "def " 4자 뒤 ~ "(" 앞 = 이름만
```

이름만 단독으로 찾으면 본문·호출에서 우연히 겹친다. `def …(` 패턴으로 **정의부만** 잡는다.

```python
# attention_probe.py:93  겹치면 그 토큰을 구간에 넣는다
if any(ts < ce and te > cs for (cs, ce) in ranges):
    hit.append(ti)
```

이름이 몇 조각으로 쪼개지든 전부 잡힌다 — 그게 이 방식을 쓰는 이유다.

### 4-3. forward 한 번에서 두 가지를 꺼낸다

```python
# model.py:190
with torch.no_grad():
    out = self.model(**enc, output_attentions=True, use_cache=True)
attentions = out.attentions                     # tuple[L], 각 [1, Hq, seq, seq]
values = _value_cache(out.past_key_values)      # list[L],  각 [1, Hkv, seq, d]
```

| 무엇 | 어디서 | 모양 |
|---|---|---|
| 어텐션 (축 A) | `output_attentions=True` | `[1, Hq, seq, seq]` — **eager에서만** |
| Value (축 B) | `use_cache=True`의 KV 캐시 | `[1, Hkv, seq, head_dim]` |

**헤드 수가 다르다.** 어텐션은 Q헤드 개, Value는 KV헤드 개. GQA 때문이다.

메모리 주의 — `output_attentions`는 `[Hq, seq, seq]`를 층마다 뜬다.
실제 우리 프롬프트는 seq 190~250이라 **층당 1~4 MB, 전 층 합쳐 40~130 MB** 수준이다.
seq에 제곱으로 커지므로 긴 컨텍스트에는 축약 경로가 필요하다.

### 4-4. 마지막 query 행만 파이썬으로 옮긴다

```python
# model.py:201
attn_last = attentions[layer][0, :, -1, :].float().cpu().tolist()   # [Hq, seq]
vnorm_kv  = values[layer][0].float().norm(dim=-1).cpu().tolist()    # [Hkv, seq]
m = span_metrics(attn_last, vnorm_kv, group_size=gqa.group_size, spans=spans, detail=…)
```

`[0, :, -1, :]` — 배치 0, 전 헤드, **마지막 query**, 전 key 위치.
큰 텐서를 파이썬으로 옮기지 않고 필요한 한 행만 리스트로 만든다.
그래서 집계(`attention_probe.py`)가 torch 없이 돌고 단위 테스트가 가능하다.

### 4-5. 집계 — 합인가 평균인가 (★ 여기서 한 번 틀렸다)

```python
# attention_probe.py:23
a_j  = sum(attn_last[h][j] for h in range(Hq)) / Hq                          # 헤드 평균
av_j = sum(attn_last[h][j] * vnorm_kv[h // group_size][j] for h in range(Hq)) / Hq
v_j  = sum(vnorm_kv[k][j] for k in range(Hkv)) / Hkv

# attention_probe.py:74
attention_weight = sum(a  for a, _, _ in per_tok)          # 구간 합   ← ★
av_norm          = sum(av for _, av, _ in per_tok)         # 구간 합   ← ★
v_norm           = sum(v  for _, _, v in per_tok) / len(per_tok)   # 구간 평균
```

> ⚠️ **`av_norm`은 "실제 기여량"이 아니다.** 코드가 계산하는 것은 `Σⱼ aⱼ‖vⱼ‖`이고
> 실제 어텐션 출력의 크기는 `‖Σⱼ aⱼvⱼ‖`다. 전자는 **벡터 상쇄를 무시**하고,
> **출력 사영 `W_O`도 거치지 않았다.** 정확히는 **pre-W_O 토큰별 노름 질량 대리치**이고
> 성격상 **상한**이다. 같은 층 안의 구간 비교에만 쓰고, 층·모델 간 절대 비교에는 쓰지 않는다.

| 지표 | 집계 | 왜 |
|---|---|---|
| `attention_weight` | **합** | 각 헤드의 어텐션 행은 전체에 대해 1로 합해진다 → 구간 합 = 그 구간에 준 질량 |
| `av_norm` | **합** | Σⱼ aⱼ‖vⱼ‖ — 기여량의 **상한 대리치**(아래 경고) |
| `v_norm` | 평균 | 크기 지표라 토큰당 평균이 맞다 |

⚠️ **합이라서 생기는 편향이 이 스텝의 가장 중요한 함정이다.**
비교하는 두 구간의 토큰 수가 다르면 합은 불공정하다:

```
formatMatrix   → ["format", "Matrix"]         2조각
format_matrix  → ["format", "_", "matrix"]    3조각   ← DeepSeek·StableCode
```

실제 저장된 토큰 수(같은 12개 이름, 블록 0):

| 모델 | code_camel | code_snake | 비율 |
|---|---|---|---|
| Qwen2.5-Coder-3B | 12 | 15 | 1.25× |
| Llama-3.2-3B | 12 | 13 | 1.08× |
| StableCode-3B | 12 | **20** | 1.67× |
| DeepSeek-Coder-6.7B | 23 | **27** | 1.17× |

snake 구간이 토큰이 많아서 합이 큰 것을 "snake를 더 본다"로 읽으면 틀린다.
그래서 **토큰 수를 반드시 함께 저장**한다:

```python
# model.py:224
"spans": {k: len(v) for k, v in spans.items()},     # → extra["span_token_counts"]
```

재집계는 이 값으로 나눈다:

```python
# scripts/observe_per_token.py:64
per[L].append(xa / na - xb / nb)      # 토큰당 평균의 차이
```

**결과가 실제로 뒤집혔다** — step4에서 4모델 전부 부호가 바뀌었다(→ `../step4/results.md`).
결과 파일은 불변이므로(§6) 재실험 없이 저장된 토큰 수만으로 다시 계산했다.

### 4-6. `token_detail` — 왜 통째로 저장하나

```python
# model.py:197
detail_spans = ("code_camel", "code_snake")
token_detail[sp]["per_layer"][str(layer)] = {"a": [...], "av": [...], "v": [...]}
```

구간 합만 저장하면 나중에 "밑줄 토큰 하나가 값을 다 먹은 건 아닌가" 같은
질문에 답할 수 없다. 결과는 불변이라 **다시 못 뽑는다.** 그래서 토큰 단위 값을
그대로 남긴다 — 파일이 커지는 대가로 사후 분석의 자유를 산다.

### 4-7. 보조 경로 — v 코사인 (`mode="vcosine"`)

크기(‖v‖)가 안 변해도 **방향**이 갈리는지 본다.

```python
# model.py:613  name_v_cosine_sweep
camel_vals = _value_cache(self.model(camel_ids, use_cache=True).past_key_values)
snake_vals = _value_cache(self.model(snake_ids, use_cache=True).past_key_values)
pairs, skipped = align_name_tokens(…, mode=token_unit)     # 같은 이름의 camel↔snake 자리
for cp, sp in pairs:
    for k in range(n_kv):
        cosines.append(cosine(cvals[k, cp, :].tolist(), svals[k, sp, :].tolist()))
```

`token_unit="mean"`을 **거절한다**:

```python
# model.py:662
if token_unit == "mean":
    raise ValueError("v 코사인 관측은 token_unit='mean'을 지원하지 않는다 (코사인은 1:1 짝이 필요)")
```

코사인은 짝지어진 두 벡터가 있어야 정의된다. 조용히 `last`로 바꿔치기하면
**다른 정렬 단위로 잰 값이 통일 규격인 척** 저장된다.

---

## 5. 알아야 할 필수 요소

### ① `attn_implementation="eager"` 없이는 어텐션이 안 나온다

```python
load_model(spec, attn_implementation="eager")
```

sdpa/flash 어텐션은 최적화된 커널이라 **가중치 행렬을 만들지 않는다.**
`output_attentions=True`를 줘도 `None`이 온다 — 예외 없이 조용히.
관측 스텝(step2·4)은 반드시 eager.

### ② 구간을 못 찾으면 예외

```python
# model.py:181
empty = [name for name, idxs in spans.items() if not idxs]
if empty:
    raise ValueError(f"프롬프트에서 찾지 못한 관측 구간: {empty}. …")
```

chat 템플릿이 문장을 변형하는 모델에서 **실제로 생긴다.** 빈 구간을 그대로 저장하면
지표가 `None`이 되고, 집계에서 `.get(키, 0)`을 쓰면 **"0을 봤다"로 오독**된다.

### ③ GQA 매핑을 틀리면 에러 없이 잘못된 값

```python
# attention_probe.py:59
if group_size <= 0 or Hq != Hkv * group_size:
    raise ValueError(f"GQA 매핑 불일치: Hq={Hq}, Hkv={Hkv}, group_size={group_size}")
```

`a[h,j] · ‖v[h//gs, j]‖`에서 `//gs`를 빼먹으면 헤드 0~1만 반복해서 곱한다.
숫자는 나오는데 의미가 없다. 그래서 모양을 먼저 검사한다.

모델별 실제 값:

| 모델 | Hq | Hkv | group_size |
|---|---|---|---|
| Qwen2.5-Coder-3B | 16 | 2 | 8 |
| DeepSeek-Coder-6.7B | 32 | 32 | 1 (GQA 아님) |
| Llama-3.2-3B | 24 | 8 | 3 |
| StableCode-3B | 32 | 32 | 1 |

### ④ `‖av‖`는 W_O 이전 값이다

`av_norm`은 어텐션 출력 사영(`W_O`)을 거치기 **전** 값이다. 층 사이·모델 사이
절대 비교에 쓰면 안 되고, **같은 층 안에서 구간끼리 비교**하는 데만 쓴다.

---

## 6. 결과 JSON 읽는 법

> **필드 정의(무엇을 뜻하는 값인가)는 [`results.md`](results.md) §1에 있다.** 여기서는 모양만 본다.

```jsonc
"metrics": {
  "per_layer": {
    "27": {
      "code_camel__attention_weight": 0.0184,   // 구간 합 (헤드 평균)
      "code_camel__av_norm":          0.412,
      "code_camel__v_norm":           22.4,     // 토큰당 평균
      "code_snake__attention_weight": 0.0231,
      "instruction__attention_weight":0.0092,
      …
    }, …
  },
  "extra": {
    "span_token_counts": {"code_camel": 12, "code_snake": 15, "instruction": 37,
                          "instr_target_word": 4, "instr_viol_word": 2},   // ★ 정규화용
    "group_names": {"code_camel": ["parseHeader", …], "code_snake": ["build_token", …]},
    "gqa": {"num_attention_heads": 16, "num_key_value_heads": 2, "group_size": 8},
    "seq_len": 512,
    "token_detail": {"code_camel": {"tokens": [...], "per_layer": {...}}, …}
  }
}
```

**`span_token_counts` 없이 `attention_weight`를 비교하지 말 것**(§4-5).

```bash
python scripts/observe_per_token.py results/step2_code-observe   # 합 vs 토큰당 평균
```

---

## 7. 처음과 달라진 것 (코드)

| 무엇 | 처음 | 지금 | 왜 |
|---|---|---|---|
| 구간 비교 | 합만 봤다 | 토큰 수 함께 저장 → 토큰당 평균 재집계 | 밑줄 토큰 때문에 snake 구간이 더 길다 |
| 빈 구간 | `None` 저장 후 진행 | **예외** | "안 봤다"로 오독됐다 |
| GQA 매핑 | 검사 없음 | 모양 불일치 시 예외 | 조용히 틀린 값이 나온다 |
| v 코사인 + mean | 조용히 허용 | **예외** | 짝이 없는데 코사인을 냈다 |
| `token_detail` | 없음 | 토큰 단위 값 보존 | 결과가 불변이라 사후 재분석이 불가능했다 |

---

## 8. 직접 확인 (GPU 불필요)

```bash
# 구간 집계 수식이 맞는지 (손계산과 대조)
python - <<'PY'
from harness.attention_probe import span_metrics
attn = [[0.1,0.2,0.7],[0.3,0.3,0.4]]      # Hq=2, seq=3
vn   = [[1.0,2.0,3.0]]                    # Hkv=1  → group_size=2
print(span_metrics(attn, vn, group_size=2, spans={"a":[0,1],"b":[2]}))
PY

# 문자 구간 → 토큰 인덱스
python -c "
from harness.attention_probe import locate_token_spans
offs=[(0,3),(3,4),(4,10),(10,11)]
print(locate_token_spans(offs, {'name':[(4,10)]}))"

pytest tests/test_step2.py -q
```
