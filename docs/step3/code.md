# step3 코드 — 코드 신호 인과 (RQ2 인과)

> **읽기 전:** [`../코드_하네스공통.md`](../코드_하네스공통.md) (모듈 지도·HuggingFace 기초)
> **방법·해석:** [`방법론.md`](방법론.md) · [`results.md`](results.md) · **장치 검증:** [`../diag/results.md`](../diag/results.md)
> **결과:** `results/step3_code-cause/` 504개 · **노트북:** `notebooks/step3_code-cause.ipynb`

---

## 1. 한 문장

**모델이 읽은 문맥의 내부 표현을 몰래 바꿔치기하고, 행동이 따라 바뀌는지 본다.**
텍스트는 그대로 두고 **KV 캐시만** 편집한다 — 이 스텝이 이 연구의 인과 주장 전체를 떠받친다.

```python
run(condition, handle, mode="generate")   # intervention.kind ≠ NONE, layers="sweep"
```

---

## 2. 쓰인 개념 → 코드 대응 (여기가 이 문서의 핵심)

### 2-1. 트랜스포머의 어디를 만지나

한 층의 어텐션은 세 벡터를 만든다:

```
q_j = W_Q x_j     "나는 무엇을 찾나"      (질문)
k_j = W_K x_j     "나는 어떤 이름표를 다나" (색인)   ← 축 A: 얼마나 보나
v_j = W_V x_j     "나를 읽으면 뭐가 오나"  (내용물) ← 축 B: 무엇이 오나

출력_i = Σ_j softmax(q_i·k_j/√d)_j · v_j
                  └── K가 정한다 ──┘   └ V가 정한다 ┘
```

**우리가 만지는 것은 이미 계산돼 캐시에 저장된 `k_j`와 `v_j`다.**
`W_K`·`W_V`(가중치)는 안 건드린다. 특정 **토큰 자리**의 K/V 값만 바꾼다.

```
KV 캐시 (층 L)                  [배치, KV헤드수, 토큰수, head_dim]
                                              └── 이 축의 특정 자리만 덮는다
  … "def" "format" "_" "matrix" "(" …
              └──────┬──────┘
              위반 이름의 토큰 자리 → 준수판(formatMatrix)의 값으로 덮기
```

| 무엇을 덮나 | 어떤 가설을 검증하나 | 코드 |
|---|---|---|
| Key만 | 문제가 **어디를 보는가**(어텐션)에 있다 | `kind="key"` |
| Value만 | 문제가 **무엇이 실려 오는가**(내용)에 있다 | `kind="value"` |
| 둘 다 | **결합 개입**(상한이 아니다 — 실제로 Value 단독보다 작다, §7) | `kind="key_value"` |

> ⚠️ 결과를 말할 때 범위를 지킬 것. 이 개입은 **평균 덮어쓰기로 post-RoPE KV 캐시를 바꾸는
> 특정 규격**이다. "Key 순효과가 작다"는 이 규격에서의 결과이고, **native Key 경로가 표기
> 정보를 갖지 않는다**는 뜻이 아니다(→ [`../diag/code.md`](../diag/code.md)).

**FFN·임베딩·가중치는 전혀 안 건드린다.** 그것이 곧 이 연구의 범위 주장이다.

### 2-2. 왜 "생성"이 아니라 "점수"인가

생성은 이산적이라 미세한 변화를 못 잡는다. 대신 **후보 두 개를 고정**하고
어느 쪽을 더 좋아하는지 연속 점수로 잰다.

```
S = logP("removeDuplicates") − logP("remove_duplicates")
```

세 상태를 잰다:

| 상태 | 프롬프트 | 기호 |
|---|---|---|
| 깨끗 | 선행이 전부 준수 | `S_clean` |
| 기준 | 선행에 위반이 섞임 (실제 조건) | `S_base` |
| 개입 | 기준 텍스트인데 KV만 준수값으로 | `S_int` |

```
되돌림 정도 = (S_int − S_base) / (S_clean − S_base)
```

1이면 완전 회복, 0이면 아무 효과 없음.

---

## 3. 파이프라인

```
Condition(composition=POOL, n_compliant=6, intervention=Intervention(
    kind=VALUE, layers="sweep", donor="compliant"|"unrelated_camel"|"unrelated_snake",
    kinds=("key","value","key_value")), token_unit="mean")
   │
   ▼  runner.py:123  _run_intervention → (sweep이므로) _run_intervention_sweep
   │
   ├─ runner.py:420  _preference_setup                      ← 세 프롬프트 + 이름 + 후보
   │     viol_messages : 위반 섞인 선행
   │     comp_messages : 전부 준수인 선행     (= 천장, donor 기본값)
   │     donor_messages: 무관 코드 (통제일 때만)
   │     viol_names / donor_names / 후보 두 개
   │
   ▼  model.py:455  intervene_preference_sweep
   │
   ├─ model.py:233  _preference_context                     ← 준비 (여기가 비용의 대부분)
   │     ① 세 프롬프트를 각 1회 forward → 캐시 3개
   │     ② 이름들의 토큰 자리 찾기 (_locate_target_tokens)
   │     ③ 위반 자리 ↔ 공여 자리 묶기 (align_name_groups)
   │     ④ 채점 함수 S(cache, last) 만들기
   │     ⑤ S_clean, S_base 계산 (층과 무관 → 한 번만)
   │
   └─ 층마다 × kind마다:  model.py:350  _score_layer_kind
         백업 → 덮어쓰기 → S 측정 → **원상 복구**
   │
   ▼
Metrics(per_layer={L: {"value__recovery": …, "key__recovery": …, …}},
        extra={S_clean, S_base, gap, undecidable, n_substituted_tokens, …})
```

**비용 설계:** 프롬프트 forward는 3번뿐이고, 층×kind마다 도는 것은
"캐시 일부 덮고 짧은 채점 forward"다. 그래서 36층 × 3방식 = 108회를 T4에서 감당한다.

---

## 4. 핵심 코드 읽기

### 4-1. 준비 — 캐시를 만들되 마지막 토큰은 남긴다

```python
# model.py:286
with torch.no_grad():
    viol_cache = self.model(viol_ids[:, :-1], use_cache=True).past_key_values
    comp_cache = self.model(comp_ids[:, :-1], use_cache=True).past_key_values
viol_last, comp_last = viol_ids[:, -1:], comp_ids[:, -1:]
```

왜 `[:, :-1]`인가 — 마지막 토큰은 **채점 forward에서 넣는다.**
캐시에 다 넣어 버리면 채점할 때 넣을 입력이 없다(모델은 최소 토큰 1개가 필요).

### 4-2. 바꿀 자리 찾기

```python
# model.py:816  _locate_target_tokens
if span_kind == "literal":                     # step5: 지침 지시어 "camelCase"
    s, e = find_char_spans(text, [nm])[0]
else:                                          # step3: 코드 이름
    s, e = find_char_spans(text, [f"def {nm}("])[0]
    s, e = s + 4, e - 1                         # "def " 뒤 ~ "(" 앞
out.append([ti for ti, (ts, te) in enumerate(offsets) if te > ts and ts < e and te > s])
```

`span_kind` 하나로 step3(코드 이름)와 step5(지침 지시어)가 **같은 함수를 공유**한다.
이게 "RQ별 스크립트를 만들지 않는다" 원칙의 실제 모습이다.

### 4-3. ★ 짝짓기 문제 — 왜 평균 덮어쓰기인가

같은 이름의 두 표기가 **다른 개수로 쪼개진다**:

```
formatMatrix   → ["format", "Matrix"]         2조각
format_matrix  → ["format", "_", "matrix"]    3조각
```

1:1로 짝지으려면 개수가 같아야 하는데 안 맞는다. 세 가지 방식이 있었다:

| 방식 | 동작 | 문제 |
|---|---|---|
| `all` | 첫↔첫, 둘째↔둘째 | 개수 다르면 **그 이름 통째로 제외** → 모델에 따라 전부 스킵 |
| `last` | 마지막 조각만 1:1 | 스킵은 없지만 단어의 일부만 덮어 신호가 약함 |
| **`mean`** | 공여 조각들을 **평균** 내 위반 이름의 **모든 자리**에 브로드캐스트 | 스킵 없음 + 단어 전체를 덮음 ← **채택** |

```python
# intervention.py:58  align_name_groups — 개수 불일치를 허용한다
for i, (vt, dt) in enumerate(zip(viol_names_tokens, donor_names_tokens)):
    if not vt or not dt:
        skipped.append(i); continue
    groups.append(([int(x) for x in vt], [int(x) for x in dt]))
```

**토크나이저에 무관한 규격**이라 4모델에 같은 코드를 쓸 수 있다(CLAUDE.md §3).
대가는 §5-②에 적었다.

### 4-4. ★ 치환 — 편집 → 측정 → 복구

이 함수가 step3의 심장이다.

```python
# model.py:350  _score_layer_kind
ek, ev = _cache_kv(work_cache, layer)          # 편집 대상 (원본 참조!)
dk, dv = _cache_kv(ctx["donor_cache"], layer)  # 공여
edit_k = kind in ("key", "key_value")
edit_v = kind in ("value", "key_value")

bk, bv = {}, {}
for vps, dps in ctx["groups"]:                 # 이름마다
    if edit_k:
        km = dk[:, :, dps, :].mean(dim=2)      # ① 공여 조각들 평균 → [B, n_kv, d]
        for vp in vps:
            bk[vp] = ek[:, :, vp, :].clone()   # ② 백업
            ek[:, :, vp, :] = km               # ③ 덮어쓰기 (브로드캐스트)
    if edit_v:
        vm = dv[:, :, dps, :].mean(dim=2)
        for vp in vps:
            bv[vp] = ev[:, :, vp, :].clone()
            ev[:, :, vp, :] = vm

s_int = ctx["S"](work_cache, ctx["viol_last"])  # ④ 측정
for vp, b in bk.items(): ek[:, :, vp, :] = b    # ⑤ 원상 복구
for vp, b in bv.items(): ev[:, :, vp, :] = b
return s_int
```

읽을 점 네 가지:

1. **`_cache_kv`가 원본 참조를 준다.** 그래서 `ek[…] = km`이 캐시를 실제로 바꾼다.
2. **`dim=2`가 토큰 축이다.** `[B, n_kv, seq, d]`에서 2번이 토큰. 여기를 평균 내면
   `[B, n_kv, d]`가 되고, 위반 자리마다 그대로 대입되며 브로드캐스트된다.
3. **KV group 단위로 자동 처리된다.** 슬라이스 `[:, :, vp, :]`가 KV헤드 축 전체를
   포함하므로 GQA에서도 그룹 단위로 덮인다(CLAUDE.md §3·§7).
4. **복구가 필수다.** 층 스윕은 캐시 **하나**를 재사용한다. 복구를 빼면
   L0의 편집이 L1 측정에 남아 **효과가 누적**된다.
   → 이걸 지키는 테스트가 `tests/test_step3.py`의 복구 불변성 검사다.

### 4-5. 채점 — 캐시를 복제해서 forward

```python
# model.py:322
def logp_candidate(cache, last_tok, cand):
    c = _clone_cache(cache)                                    # ← 복제 필수
    inp = torch.cat([last_tok, cand[:-1].unsqueeze(0)], dim=1)
    with torch.no_grad():
        logits = self.model(inp, past_key_values=c, use_cache=True).logits[0]
    lp = torch.log_softmax(logits.float(), dim=-1)
    return float(sum(lp[k, cand[k]].item() for k in range(cand.shape[0])))
```

| 줄 | 왜 |
|---|---|
| `_clone_cache` | forward가 캐시를 **늘린다.** 복제 안 하면 후보를 두 번 채점할 때 오염된다 |
| `cand[:-1]` | 마지막 후보 토큰은 **입력이 아니라 예측 대상**이라 넣지 않는다 |
| `.float()` | fp16 소프트맥스는 NaN이 날 수 있다 |
| `lp[k, cand[k]]` | 위치 k의 로짓에서 실제 후보 토큰 k의 로그확률 |

토큰 수가 다른 두 후보를 비교하는 것에 주의 —
`removeDuplicates`(2조각) vs `remove_duplicates`(3조각). **길이 정규화를 하지 않는다.**
`S`의 절대값은 모델 간 비교에 쓰지 않고, 우리는 **같은 조건 안의 차이(회복률)** 만 본다.

### 4-6. 공여 세 종류 — 통제가 설계의 핵심

```python
# runner.py:450
donor_kind = condition.intervention.donor or "compliant"
donor_messages = None                                   # 기본: comp_cache를 그대로 씀
if donor_kind.startswith("unrelated"):
    un = Notation.CAMEL if donor_kind.endswith("camel") else Notation.SNAKE
    u_specs = unrelated_specs(condition, len(viol_specs))    # 선행에 안 쓰인 이름들
    donor_names = [s.name(un) for s in u_specs]
    donor_messages = msgs(render_preceding([(s, un) for s in u_specs]))
```

| donor | 무엇을 덮나 | 뭘 죽이나 |
|---|---|---|
| `compliant` | 같은 이름의 준수판 | — (처치) |
| `unrelated_camel` | **다른 이름**의 camel판 | "camel이면 아무거나 되는가" |
| `unrelated_snake` | 다른 이름의 snake판 | **"덮어쓰기 자체의 교란"** ← 이게 핵심 |

**`unrelated_snake`가 왜 중요한가** — snake를 snake로 덮으면 표기 정보는 하나도
새로 안 들어간다. 그런데도 점수가 움직이면, 그건 "정보 전달"이 아니라
**원래 표현을 지우는 효과**다. 실제로 이 값이 회복률의 절반 이상이었다.

그래서 보고는 **처치 − 통제(순효과)** 로 한다(CLAUDE.md §3):

```python
# scripts/step3_net_effect.py — 같은 묶음(블록)끼리 먼저 뺀다
짝맞춤(권장) = recovery[unrelated_camel][b] - recovery[unrelated_snake][b]
보조        = recovery[compliant][b]       - recovery[unrelated_snake][b]
```

**권장 추정량은 `다른camel − 다른snake`다.** 두 공여는 문맥 구성이 완전히 같고(둘 다 6함수·
단일 표기 별도 모듈) 차이가 오직 표기에서만 온다. `같은camel − 다른snake`는 처치 쪽이
본래 조건(12함수 준수판)이라 문맥 구성이 달라 **보조 지표**다. 두 값을 모두 보고한다.

묶음끼리 먼저 빼는 이유 — 블록마다 이름이 달라 난이도가 다르다. 평균끼리 빼면
블록 구성 차이가 섞인다.

### 4-7. 판정 불가

```python
# intervention.py:91
UNDECIDABLE_GAP = 1.0
def effect_size(s_clean, s_base):  return abs(s_clean - s_base)
def is_undecidable(s_clean, s_base, gap_min=UNDECIDABLE_GAP):
    return effect_size(s_clean, s_base) < gap_min
```

회복률의 **분모**가 `S_clean − S_base`다. 이게 0에 가까우면
"문맥이 애초에 행동을 못 흔든 조건"이고, 회복률은 0/0에 가까운 잡음이 된다.

`gap`과 `undecidable`을 **결과에 저장**하므로 집계에서 임계를 바꿔 다시 자를 수 있다
(논문에는 임계 민감도를 함께 싣는다).

---

## 5. 알아야 할 필수 요소

### ① 층 목록을 주면 예외 (실제로 겪은 사고)

```python
# runner.py:137
layers = list(iv.layers)
if len(layers) != 1:
    raise ValueError(f"단일 층 개입에는 층을 하나만 준다 (받음: {layers}). …")
```

예전 코드는 `list(iv.layers)[0]`이라 **목록을 줘도 첫 층만 돌고 나머지는 조용히 버려졌다.**
파일명에는 그 첫 층만 남아 "여러 층을 쟀다"고 오해하기 쉽다. GPU를 몇 판 날린 뒤 막았다.

### ② 평균 덮어쓰기가 값을 쪼그라뜨릴 수 있다 — 그래서 진단했다

서로 다른 위치의 조각을 평균 내면, 방향이 엇갈린 벡터끼리 지워져
**원래보다 짧은 벡터**가 된다. 특히 Key에는 위치 회전(RoPE)이 이미 발라져 있다.

> 만약 Key만 더 망가진다면, "Key를 바꿔도 안 돌아온다"는 우리 결론이
> **어텐션 경로의 성질이 아니라 우리가 값을 망가뜨린 탓**일 수 있다.
> 편향 방향이 우리 주장과 같으므로 반드시 확인해야 했다.

→ `mode="kv_diagnose"`로 따로 쟀고(**진단 A**), **우려는 기각**됐다.
Key 줄어듦 0.944~0.963 > Value 0.785~0.892 — Key가 **덜** 줄었다.
자세한 코드는 [`../diag/code.md`](../diag/code.md).

### ③ 층 효과는 누적되지 않는다 (설계이자 한계)

각 층을 **독립적으로** 재고 복구한다. "L20과 L27을 함께 바꾸면?"은 이 설계로 답할 수 없다.
층별 곡선은 "각 층 단독의 기여"지 분해가 아니다.

### ④ 이 개입이 정확히 무엇을 반사실로 만드는가

바꾸는 것은 **"선행 코드 이름 토큰들이 층 L에서 갖는 K/V 값"** 뿐이다.
바뀌지 않는 것: 토큰 ID, 위치, 다른 토큰의 K/V, 다른 층, FFN, 가중치.
논문에서 인과 주장의 범위를 이 문장으로 한정해야 한다.

---

## 6. 결과 JSON 읽는 법

```jsonc
"metrics": {
  "per_layer": {
    "27": {
      "value__S_int":      -1.24,
      "value__recovery":     0.42,   // ← 헤드라인
      "key__S_int":        -3.01,
      "key__recovery":       0.02,
      "key_value__S_int":  -1.10,
      "key_value__recovery": 0.45
    }, …
  },
  "extra": {
    "mode": "intervene_sweep",
    "donor": "compliant",            // 또는 unrelated_camel / unrelated_snake
    "intervention_target": "code",
    "kinds": ["key", "value", "key_value"],
    "S_clean": 1.83, "S_base": -3.21,
    "n_substituted_tokens": 31,      // 실제로 덮은 자리 수 (0이면 예외였다)
    "skipped_names": [],             // 정렬 실패한 이름 인덱스
    "viol_names": [...], "donor_names": [...]
  }
}
```

⚠️ **저장된 504개 파일에는 `gap`·`undecidable`이 없다.** 그 필드는 step3을 돌린 뒤에
추가됐다(step5 통제분·step6부터 들어 있다). step3에서는 집계 때 `S_clean`·`S_base`로
직접 계산한다 — 값은 같다:

```python
gap = abs(extra["S_clean"] - extra["S_base"])       # = effect_size(...)
undecidable = gap < 1.0                             # = is_undecidable(...)
```

⚠️ **`extra["recovery"]`·`extra["layer"]`는 스윕 결과에 없다.** 층별 값은
`per_layer`의 `<kind>__recovery`에 있다. (단일 층 경로에만 `extra["recovery"]`가 있다.)
이 차이를 놓쳐 빈 표를 출력한 적이 있다.

집계:

```bash
python scripts/step3_net_effect.py          # 처치 − 통제 (순효과) ← 논문에 쓰는 값
```

---

## 7. 처음과 달라진 것 (코드·개념)

| 무엇 | 처음 | 지금 | 왜 |
|---|---|---|---|
| 정렬 단위 | `all` (개수 같아야) | **`mean`** 통일 | 모델마다 스킵률이 달라 비교가 안 됐다 |
| 보고 값 | 회복률 그대로 | **처치 − 통제(순효과)** | 덮어쓰기 자체의 교란이 절반 이상 |
| 통제 차감 | 평균끼리 뺐다 | **묶음끼리 먼저** 뺀다 | 블록 난이도 차이가 섞였다 |
| 층 목록 | 조용히 첫 층만 | **예외** | 결과와 파일명이 어긋났다 |
| 방식 선택 | 스윕이 3종 하드코딩 | `Intervention.kinds` | 조건 객체가 실행을 규정해야 한다 |
| 판정 불가 | 없음 | `gap`·`undecidable` 저장 | 분모가 작은 조건을 섞어 평균 냈다 |
| RoPE 우려 | 미확인 | **진단 A로 기각** | 편향 방향이 결론과 같았다 |
| 「어텐션은 잉여」 | 주장했다 | **철회** | K+V 순효과 < Value 순효과 — 잉여라면 같아야 한다 |
| 「Key 순효과는 0」 | 주장했다 | **철회** | 신뢰구간이 0을 **배제**한다(스크립트가 '0을 무나: 아니오'로 찍는다). "실용적으로 무시할 수준(Value의 2~11%)"이 정확한 표현이다 |

---

## 8. 직접 확인 (GPU 불필요)

```bash
pytest tests/test_step3.py -q     # 12개: 복구 불변성·kind 격리·평균 정확성·층 격리

# 평균 덮어쓰기 짝짓기가 어떻게 되는지
python -c "
from harness.intervention import align_name_groups, align_name_tokens
viol=[[10,11,12]]; donor=[[20,21]]        # snake 3조각 ↔ camel 2조각
print('mean :', align_name_groups(viol,donor))
print('all  :', align_name_tokens(viol,donor,mode='all'))   # ← 통째로 스킵된다
print('last :', align_name_tokens(viol,donor,mode='last'))"

# 되돌림 정도와 판정 불가
python -c "
from harness.intervention import recovery_rate, is_undecidable
print(recovery_rate(2.0,-3.0,-0.5))      # (−0.5+3)/(2+3) = 0.5
print(is_undecidable(0.2,-0.3))          # gap 0.5 < 1.0 → True"
```
