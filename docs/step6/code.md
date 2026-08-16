# step6 코드 — 처방: 잔차 조향 vs Spotlight

> ⚠️ **이름 주의.** 우리는 이 처방을 오래 "값 조향"이라 불렀지만, 구현은 **디코더 블록 출력
> (잔차 스트림) 전체에 벡터를 더한다.** step3의 V-캐시 치환과 다른 개입이며, 이후 층의
> Q·K·V·FFN을 **모두** 바꾼다. 논문에서는 **잔차 스트림 조향(activation steering)** 으로 쓴다.

> **읽기 전:** [`../코드_하네스공통.md`](../코드_하네스공통.md)
> **방법·해석:** [`방법론.md`](방법론.md) · [`results.md`](results.md)
> **결과:** `results/step6_steer/` 2184 · `step6_steer-generate/` 420 · `step6_steer-crosslayer/` 336
> **노트북:** `notebooks/step6_steer.ipynb` · `step6_verify.ipynb`

---

## 1. 한 문장

**step3·5가 "어디가 문제인가"를 밝혔다면, step6은 "그럼 고쳐지는가"를 묻는다.**
우리 처방(값 조향)과 기존 접근(Spotlight, 어텐션 키우기)을 같은 자로 재서 비교한다.

```python
run(condition, handle, mode="steer")            # 점수 회복
run(condition, handle, mode="steer_generate")   # 실제 이름 생성 (검증)
```

---

## 2. step3·5와 근본적으로 다른 점 — 캐시가 아니라 훅

| | step3·5 | step6 |
|---|---|---|
| 개입 방식 | KV 캐시를 **편집** | forward **훅**·어텐션 함수 교체 |
| 개입 시점 | forward 끝난 뒤 | forward **도중** |
| 대상 | 특정 토큰 자리의 K/V | ①층 전체 잔차 ②전 층·전 헤드 어텐션 |
| 파일 | `model.py` | **`steer.py`** |

캐시 편집으로는 안 되는 이유 — 값 조향은 **잔차 스트림**에 더하는 것이고,
Spotlight는 **어텐션 소프트맥스 이후**를 재가중한다. 둘 다 이미 계산이 끝난
KV 캐시로는 표현할 수 없다.

---

## 3. 방법 A — 잔차 스트림 조향 (우리 것)

### 3-1. 개념: 잔차에 방향을 더한다

```
층 L 출력:   h_L  ←  h_L + 세기 × (camel 방향 − snake 방향)
```

CAA(ACL 2024)·ITI(NeurIPS 2023) 계열. "표기 개념이 잔차 공간의 **한 선형 방향**으로
표현되고 전역 덧셈으로 조향된다"는 가정 위에 서 있다.

> ⚠️ **step3이 이 가정을 증명한 것이 아니다.** 두 가설은 다르다.
>
> | | 무엇을 보였나 |
> |---|---|
> | **step3** | 특정 토큰 자리의 **V-캐시**를 바꾸면 행동이 바뀐다 |
> | **step6** | 잔차 공간에 **선형 camel−snake 방향**이 있고 전역 덧셈으로 조향된다 |
>
> step6은 step3에서 **동기를 얻은 별도의 실험**이지, step3의 따름정리가 아니다.

### 3-2. 방향 만들기 — 모델마다 자체 유도

```python
# steer.py:208  build_steer_vector
for smp in samples:                                   # 8묶음 × (camel판, snake판)
    for msg_key, name_key, acc in (("camel_messages","camel_names",cam),
                                   ("snake_messages","snake_names",sna)):
        text = tok.apply_chat_template(smp[msg_key], tokenize=False,
                                       add_generation_prompt=True) + "def "
        enc  = tok(text, return_tensors="pt", return_offsets_mapping=True,
                   add_special_tokens=False)
        pos  = [p for lst in _locate_target_tokens(text, offsets, smp[name_key], "def_name")
                for p in lst]                          # 이름 토큰 자리
        hs = model(enc["input_ids"], output_hidden_states=True).hidden_states[layer + 1][0]
        acc.append(hs[pos].float().mean(0))            # 이름 자리 잔차의 평균
vec = torch.stack(cam).mean(0) - torch.stack(sna).mean(0)     # ← camel − snake
```

읽을 점:

1. **`hidden_states[layer + 1]`** — `[0]`은 임베딩 출력이다. `+1`을 빼먹으면
   한 층 앞의 잔차로 방향을 만든다.
2. **이름 토큰 자리만** 평균 낸다. 문장 전체를 쓰면 표기와 무관한 성분이 섞인다.
3. **같은 선행을 전부 camel / 전부 snake로만 달리 렌더**한다(`runner.py:237 _steer_samples`).
   내용은 같고 표기만 다르다. 다만 **"표기 성분만 남는다"고는 말할 수 없다** — 두 프롬프트는
   토큰 수·뒤쪽 위치·조각 구성이 달라지므로 차이 벡터에 그 성분도 섞인다.
4. 방향은 모델·층·출처마다 한 번만 만들고 캐시한다:
   ```python
   # runner.py:321
   key = (condition.model.name, src_layer, iv.steer_source)
   if key not in _STEER_VEC:
       _STEER_VEC[key] = steer_mod.build_steer_vector(handle, _steer_samples(condition), src_layer)
   ```
5. 표본 절반 이상을 못 찾으면 **예외**(§6-③).

### 3-3. 주입 — forward 훅

```python
# steer.py:78  value_add_hook
def __enter__(self):
    add = self.strength * self.vec
    def fn(_mod, _inp, out):
        self.n_calls += 1                                   # ← 실제로 불렸는지
        if isinstance(out, tuple):                          # 디코더 블록은 튜플 반환
            hs = out[0]
            return (hs + add.to(hs.dtype).to(hs.device),) + tuple(out[1:])
        return out + add.to(out.dtype).to(out.device)
    self._h = self.layer_module.register_forward_hook(fn)
    return self

def __exit__(self, *exc):
    self._h.remove()                                        # ← 반드시 뗀다
```

| 왜 | 설명 |
|---|---|
| `isinstance(out, tuple)` | 디코더 블록은 `(hidden, attn, cache)` 튜플을 준다. `out[0]`만 바꾸고 나머지는 그대로 |
| `.to(hs.dtype).to(hs.device)` | 방향은 fp32로 만들었고 모델은 fp16이다 |
| `with` 컨텍스트 | 훅을 안 떼면 **다음 측정까지 오염**된다 |
| `n_calls` | 0이면 예외 — "개입했는데 효과 없음"과 "개입이 안 걸림"을 구분 |

`decoder_layers`가 모델 계열별 경로를 흡수한다:

```python
# steer.py:65
for attr in ("model", "transformer", "gpt_neox"):    # 껍데기
    if hasattr(base, attr): base = getattr(base, attr); break
for attr in ("layers", "h"):                         # 블록 리스트
    if hasattr(base, attr): return getattr(base, attr)
raise RuntimeError("디코더 블록 리스트를 찾지 못했다 (model.model.layers 예상)")
```

---

## 4. 방법 B — Spotlight (기존 접근, 재구현)

### 4-1. 수식을 먼저 순수 함수로 분리했다

모델 없이 검증할 수 있어야 "구현을 못해서 진 것"이라는 반박을 막는다.

```python
# steer.py:34  spotlight_reweight
before = sum(x for x, m in zip(p, mask) if m)
if before >= psi_target:              # ① 게이팅 — 이미 충분히 보면 손대지 않는다(원문 식 3)
    return p, before, before
factor = psi_target / max(before, 1e-12)
new = [x * factor if m else x for x, m in zip(p, mask)]    # ② 스팬만 곱하고
new = [x / sum(new) for x in new]                          # ③ 재정규화
after = sum(x for x, m in zip(new, mask) if m)
```

**목표에 정확히 도달하지 않는 것이 정상이다.** 재정규화 때문에 언더슛한다:

```
ψ_after = ψ_target / (1 + ψ_target − ψ_before)          (원문 식 5)
```

이 폐형식이 코드와 일치하는지 테스트한다(`tests/test_step6.py`).
언더슛을 모르고 "목표치에 못 미쳤으니 구현 실패"로 읽으면 안 된다.

### 4-2. 어텐션 함수를 통째로 갈아끼운다

```python
# steer.py:181
ALL_ATTENTION_FUNCTIONS["spotlight_step6"] = spotlight_fn
self._prev_impl = self.model.config._attn_implementation
self.model.config._attn_implementation = "spotlight_step6"
# __exit__에서 되돌린다
```

훅으로는 안 된다 — 소프트맥스 **이후** 확률을 재가중하고 그걸로 다시 V를 곱해야 하는데,
그 중간값은 훅이 볼 수 없다.

### 4-3. ★ 인과 마스크 — 실제로 있던 버그

```python
# steer.py:152
attn = torch.matmul(query, k.transpose(2, 3)) * scaling
if attention_mask is not None:
    attn = attn + attention_mask[..., : k.shape[-2]]
else:
    # 최신 transformers는 mask=None으로 넘기고 인과성을 내부에서 처리한다.
    # 여기서 직접 씌우지 않으면 **미래 토큰까지 보게 되어** 모델이 망가진다.
    q_len, k_len = attn.shape[-2], attn.shape[-1]
    causal = torch.ones(q_len, k_len, dtype=torch.bool,
                        device=attn.device).tril(diagonal=k_len - q_len)
    attn = attn.masked_fill(~causal, torch.finfo(attn.dtype).min)
```

`else` 가지가 없던 초판은 **미래 토큰까지 보는 모델**을 측정하고 있었다.
그러면 Spotlight는 무조건 진다 — 그리고 그 패배가 **우리 결론과 같은 방향**이다.
돌리기 전에 잡았고, 고친 뒤 무개입 기준선과 5e-08까지 일치함을 확인했다.

`tril(diagonal=k_len - q_len)`의 오프셋에 주의 — 캐시를 쓰면 `q_len < k_len`이라
대각선이 오른쪽으로 밀린다.

기타 세부:

```python
k = repeat_kv(key, groups)                  # GQA: KV헤드를 Q헤드 수만큼 복제
probs = torch.softmax(attn, dim=-1, dtype=torch.float32)   # fp16 NaN 방지
probs = probs.index_copy(-1, idx, probs.index_select(-1, idx) * factor)  # 스팬 열만
probs = probs / probs.sum(-1, keepdim=True)                # 재정규화
out = torch.matmul(probs, v).transpose(1, 2).contiguous()  # transformers 규약 모양
```

### 4-4. 어디를 밀 것인가 — 스팬 두 종류

```python
# runner.py:258  _spotlight_span_positions
if span == "rule_word":
    word = notation_word(condition.instruction.token_notation)     # "camelCase"
    pos = [p for lst in _locate_target_tokens(text, offsets, [word], "literal") for p in lst]
else:                                                              # "instruction"
    char_spans = {"instruction": find_char_spans(text, [build_instruction_text(condition)])}
    pos = list(locate_token_spans(offsets, char_spans)["instruction"])
if not pos:
    raise ValueError(f"Spotlight 스팬을 프롬프트에서 찾지 못했다 ({what}). …")
```

| span | 무엇 | 왜 |
|---|---|---|
| `rule_word` | 규칙문 지시어만 | **step5에서 값을 바꾼 자리와 정확히 같다** → 공정한 대결 |
| `instruction` | 지침 문장 전체 | 넓은 스팬 (원 논문은 스팬을 사용자가 지정하게 두므로, 이건 **우리가 고른 한 가지**다) |

두 가지를 다 돌린 것이 중요하다 — 한 가지만 돌리고 졌다면
"스팬을 잘못 잡아서"라는 반박이 가능했다.

### 4-5. 개입 전·후 비중을 기록한다

```python
# steer.py:175
outer.psi_before.append(float(before[..., -1, :].mean()))    # 마지막 query 행 기준
outer.psi_after.append(float(after[..., -1, :].mean()))
```

이 기록이 없으면 "Spotlight가 실패했다"와 "Spotlight를 못 걸었다"를 구분할 수 없다.
실제 결과는 0.002 → 0.231(**100배 이상**)로 확실히 걸렸고, 그런데도 회복이 없었다.

---

## 5. 검증 경로 — 점수만 믿지 않는다

### 5-1. 실제로 이름을 생성해 본다

```python
# steer.py:359  steer_generate
with make_ctx(), torch.no_grad():        # 조향을 건 채로
    for _ in range(max_new_tokens):
        out = handle.model(cur, past_key_values=past, use_cache=True)
        nxt = int(out.logits[0, -1].argmax())     # 그리디
        ...
```

### 5-2. 이름이 멀쩡한지 검사한다

```python
# steer.py:331  name_health
checks = {
    "식별자 형식":      bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)),
    "길이 정상":        2 <= len(name) <= 40,
    "같은 조각 반복 없음": not _has_repeat(name),      # removeDuplicatesDuplicates
}
```

### 5-3. ★ 이 검증이 실제로 잡아낸 것

**선호 점수와 실제 준수율이 갈라진다.**

| Qwen 세기 | 회복률(점수) | 실제 준수율 | 나온 이름 |
|---|---|---|---|
| 1 | 1.243 | **1.000** | `removeDuplicates` |
| 2 | 1.643 | **1.000** | `removeDuplicates` |
| **4** | **1.733** (최고) | **0.000** | `RemoveDuplicates` ← PascalCase |

점수만 보면 세기 4가 가장 성공적이다. 실제로는 완전 실패다.
camel 방향을 과하게 밀어 **첫 글자까지 대문자로 넘어갔고**,
`classify_name`이 이를 `other`로 센다(→ [`../step1/code.md`](../step1/code.md) §4-3).

> **선호 점수는 위가 막혀 있지 않다.** 세게 밀수록 계속 오르지만,
> 그 지점에서 모델은 이미 다른 표기로 넘어가 있다. 회복률 > 1 구간은
> **되살아난 것이 아니라 지나친 것**이다.

이 검증을 안 했으면 "회복률 1.7 달성"으로 논문을 썼을 것이다.

---

## 6. 조건 구성과 조용한 실패 방지

### 6-1. 실제로 돈 조건 (모델당 546개)

```
무개입                                  42묶음 ×  1 =  42
값 조향  맞는층 × 세기 4종               42 × 4 = 168
값 조향  엉뚱층(초반) × 세기 4종          42 × 4 = 168
Spotlight  스팬 2종 × ψ 2종              42 × 4 = 168
                                          합계    546  × 4모델 = 2184
```

모델별 층:

| 모델 | 총 층 | 맞는 층 | 엉뚱한 층 |
|---|---|---|---|
| Qwen2.5-Coder-3B | 36 | 25 | 5 |
| DeepSeek-Coder-6.7B | 32 | 20 | 5 |
| Llama-3.2-3B | 28 | 15 | 3 |
| StableCode-3B | 32 | 18 | 4 |

교차 주입(`step6_steer-crosslayer`)은 **맞는 층에서 뽑은 방향을 엉뚱한 층에 넣는다**:

```python
# runner.py:320
src_layer = iv.steer_layer if iv.steer_layer is not None else layer
#   같으면 → "이 층에 밀면 되나"
#   다르면 → "방향 자체가 층 특이적인가"
```

### 6-2. 무개입도 같은 자로 잰다

```python
# runner.py:104
if mode == "steer" or condition.intervention.kind in (VALUE_ADD, ATTENTION_AMPLIFY):
    return _run_steer(condition, handle)
```

무개입 하한선은 `kind=NONE`이라 라우팅상 생성 경로로 샐 수 있다. 그래서
`mode="steer"`를 명시해 **같은 교사강제 점수**로 재게 한다. 다른 자로 잰 값과
비교하면 회복률의 분모가 어긋난다.

### 6-3. 막아 둔 조용한 실패

| 어디 | 무엇 | 없으면 |
|---|---|---|
| `steer.py:192` | 어텐션 함수 호출 0회 | 회복 0 = "Spotlight 무익" ← **우리 결론과 같은 방향** |
| `steer.py:310` | 값 조향 훅 호출 0회 | 회복 0 = "값 조향 실패" |
| `steer.py:240` | 방향 표본 절반 이상 스킵 | 근거가 얇은 방향으로 조향 |
| `steer.py:130` | Spotlight 스팬 비어 있음 | 어디를 밀지 정해지지 않은 채 실행 |
| `runner.py:287` | 스팬을 프롬프트에서 못 찾음 | 위와 같음 |
| `runner.py:311` | 값 조향에 층 목록 | 조용히 첫 층만 (step3에서 겪은 사고) |

**`steer.py:192`가 이 스텝에서 가장 중요한 방어다.** Spotlight가 안 걸렸는데
"어텐션 접근은 무익하다"는 결론을 냈다면 연구 전체가 무너진다.

---

## 7. 결과 JSON 읽는 법

> **필드 정의(회복률·조향 세기·Spotlight 기록이 무슨 값인가)는 [`results.md`](results.md) §1에 있다.**
> 여기서는 모양만 본다.

**점수 회복 (`step6_steer`)**

```jsonc
"extra": {
  "mode": "steer", "method": "attn_amplify",     // none | value_add | attn_amplify
  "span": "rule_word", "psi_target": 0.1,        // Spotlight 전용
  "strength": null, "steer_source": null,        // 값 조향 전용
  "layer": null, "kind": "spotlight",
  "S_clean": 3.929, "S_base": -1.556, "S_int": -3.327,
  "recovery": -0.323,                            // ← 헤드라인
  "gap": 5.485, "undecidable": false,
  "attn_span_before": 0.00080,                   // ★ 개입이 실제로 걸렸는지
  "attn_span_after":  0.09053,
  "attn_calls": 64,                              // 0이면 예외였다
  "n_span_tokens": 3,
  "n_compliant": 0, "target": "camel", "tag": "cliff"
}
```

값 조향이면 `"method": "value_add"`, `"strength": 2`, `"layer": 25`,
`"hook_calls": …`, `"steer_vector": {"n_samples": 8, "n_skipped": 0, "vec_norm": …}`.

**생성 검증 (`step6_steer-generate`)**

```jsonc
"metrics": {
  "compliance_rate": 0.0,
  "extra": {
    "mode": "steer_generate", "method": "value_add", "strength": 4, "layer": 25,
    "generated_text": "def RemoveDuplicates(items):\n    …",
    "name": "RemoveDuplicates",
    "notation": "other",          // ← camel이 아니다!
    "compliant": false,
    "name_ok": true, "name_reason": "", "name_length": 16
  }
}
```

**`recovery`와 `compliant`를 반드시 함께 볼 것**(§5-3).

집계:

```bash
python scripts/step6_summary.py --figs docs/step6/figures
```

---

## 8. 처음과 달라진 것 (코드·개념)

| 무엇 | 처음 | 지금 | 왜 |
|---|---|---|---|
| 인과 마스크 | `mask=None`이면 안 씌움 | **직접 씌운다** | 미래 토큰을 보는 모델을 재고 있었다 |
| 개입 확인 | 없음 | 호출 횟수 · ψ 전후 기록 | "실패"와 "안 걸림"이 같은 숫자였다 |
| 검증 | 점수만 | **실제 생성 + 이름 건전성** | 점수 최고 지점에서 준수율이 0이었다 |
| 층 특이성 | "맞는 층이어야 한다" | **철회** | 교차 주입이 더 높은 경우가 많다 |
| Spotlight 스팬 | 한 가지 | **두 가지**(지시어·문장 전체) | 스팬 탓이라는 반박을 막는다 |
| 무개입 | 생성 경로 | `mode="steer"` 명시 | 다른 자로 재면 분모가 어긋난다 |
| 결론 범위 | "어텐션 접근은 무익" | **"코드 특화 모델에서 무익"** | Llama에서는 Spotlight가 0.904로 작동 |

---

## 9. 직접 확인 (GPU 불필요)

```bash
pytest tests/test_step6.py -q     # 11개: 언더슛 폐형식·게이팅·인과 마스크·훅 정리

# Spotlight 수식이 원문 식(5)와 맞는지 손으로
PYTHONPATH=src python - <<'PY'
from harness.steer import spotlight_reweight
p    = [0.9, 0.05, 0.05]
mask = [False, True, True]        # 스팬 비중 ψ_before = 0.10
new, b, a = spotlight_reweight(p, mask, 0.3)
print(f"전 {b:.4f} → 후 {a:.4f}")
print(f"폐형식 예측 {0.3/(1+0.3-b):.4f}")        # 두 값이 같아야 한다
print("게이팅:", spotlight_reweight(p, mask, 0.05)[1:])   # 이미 충분 → 손대지 않음
PY

# 이름 건전성 검사
python -c "
from harness.steer import name_health
for n in ['removeDuplicates','RemoveDuplicates','removeDuplicatesDuplicates','ize',None]:
    print(f'{str(n):30} {name_health(n)}')"
```
