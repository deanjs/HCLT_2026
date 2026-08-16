# 진단 A 코드 — 평균 덮어쓰기가 Key를 망가뜨리는가

> **읽기 전:** [`../코드_하네스공통.md`](../코드_하네스공통.md) · [`../step3/code.md`](../step3/code.md)(진단 대상 장치)
> **방법·결과:** [`방법론.md`](방법론.md) · [`results.md`](results.md)
> **결과:** `results/diag_kv-phase/` 72개 · **노트북:** `notebooks/diag_kv-phase.ipynb`

---

## 1. 한 문장

**실험이 아니라 측정 장치 검사다.** step3·step5가 공유하는 평균 덮어쓰기가
Key 쪽에만 불리하게 작동하는지 확인한다.

```python
run(condition, handle, mode="kv_diagnose")     # 점수를 안 매긴다 → 가볍다
```

---

## 2. 왜 이 검사가 필요했나

### 2-1. 의심의 구조

우리는 서로 다른 **위치**에 있던 조각들의 K/V를 **평균** 내서 덮는다.

```
공여 이름 formatMatrix 의 조각들
   위치 148: k₁      위치 149: k₂
   평균 = (k₁ + k₂)/2   ← 이걸 위반 이름의 모든 자리에 넣는다
```

문제는 **Key에는 그 토큰이 있던 위치의 회전(RoPE)이 이미 발라져 있다**는 것이다.
Value에는 위치 정보가 없다.

```
RoPE:  k_j  =  회전(위치 j) · W_K x_j        ← 위치가 벡터에 섞여 있다
       v_j  =  W_V x_j                       ← 위치와 무관
```

회전 각도가 다른 벡터를 평균하면 서로 지워져 **원래보다 짧은 벡터**가 된다.

### 2-2. 왜 반드시 확인해야 했나

우리 결론은 **"Key를 바꿔도 안 돌아온다 → 어텐션은 문제가 아니다"** 이다.
그런데 만약 Key만 더 망가진다면, 그 결론은 어텐션 경로의 성질이 아니라
**우리가 Key를 더 심하게 부순 탓**일 수 있다.

> **편향의 방향이 우리가 원하는 결론과 같다.** 이런 자리는 반드시 검사해야 한다.

---

## 3. 무엇을 재나 — 두 값을 나란히

핵심 아이디어: **Key만 재면 판단할 수 없다.** 서로 다른 토큰을 평균 내면
회전이 없어도 조금은 줄어들기 때문이다. 그래서 **Value의 같은 값을 기준선으로 함께 잰다.**

```
줄어듦 = ‖조각들의 평균‖ / (조각 노름들의 평균)
```

| 값 | 뜻 |
|---|---|
| 1에 가까움 | 조각들이 같은 방향 → 평균 내도 안 줄었다 = 문제 없음 |
| 많이 작음 | 서로 지워졌다 = 쪼그라든 값을 넣고 있었다 |

| 비교 | 판정 |
|---|---|
| Key 줄어듦 ≈ Value 줄어듦 | 회전 탓의 **추가 손해 없음** → 우려 기각 |
| Key 줄어듦 ≪ Value 줄어듦 | **RoPE 탓** → step3 Key 결과는 해석 불가 |

```python
# model.py:567
def shrink_and_spread(t):                       # t: [B, n_kv, n조각, d]
    mean_norm = t.mean(dim=2).norm(dim=-1)      # ‖평균‖          [B, n_kv]
    norm_mean = t.norm(dim=-1).mean(dim=2)      # 노름들의 평균     [B, n_kv]
    ratio = float((mean_norm / norm_mean.clamp_min(1e-9)).mean())

    u = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-9)      # 단위벡터로
    gram = torch.einsum("bknd,bkmd->bknm", u, u)              # 조각쌍 코사인 행렬
    off = (gram.sum(dim=(-1, -2)) - n) / (n * (n - 1))        # 대각선 제외 평균
    return ratio, float(off.mean())
```

`einsum("bknd,bkmd->bknm", u, u)` — 조각 n개끼리의 모든 코사인을 한 번에 만든다.
대각선(자기 자신, 값 1)을 빼고 평균 내면 **방향이 얼마나 흩어졌나**가 나온다.

함께 남기는 것: **덮을 자리와 공여 자리의 위치 차이**. 위치가 어긋날수록 회전 위상도 어긋난다.

```python
# model.py:599
offsets = [(sum(vps)/len(vps)) - (sum(dps)/len(dps)) for vps, dps in groups]
```

---

## 4. 왜 가벼운가 — 점수를 안 매긴다

```python
# model.py:511  kv_substitution_diagnostics
ctx = self._preference_context(...)          # ← step3과 **완전히 같은 입력**
groups = ctx["groups"]

with torch.no_grad():
    for layer in range(self.num_layers):
        dk, dv = _cache_kv(ctx["donor_cache"], layer)
        for _vps, dps in groups:
            shrink_and_spread(dk[:, :, dps, :])    # Key
            shrink_and_spread(dv[:, :, dps, :])    # Value
```

step3 스윕은 층×kind마다 **채점 forward**를 돈다(36층 × 3방식 = 108회).
이 진단은 프롬프트 forward 몇 번으로 캐시를 만든 뒤 **텐서 연산만** 한다.
그래서 72개 조건이 금방 끝났다.

**`_preference_context`를 그대로 재사용하는 것이 핵심이다.** 진단용으로 입력을
따로 만들면 "진짜 그 실험의 값을 검사한 것인가"가 흔들린다.

```python
# model.py:552  규격 강제
if token_unit != "mean":
    raise ValueError("이 진단은 평균 덮어쓰기(mean) 규격을 대상으로 한다 …")
```

---

## 5. 조건 구성

```
4모델 × 공여 3종 × 6묶음 = 72
```

| 공여 | 대상 | 어느 스텝의 장치를 검사하나 |
|---|---|---|
| `compliant` (target=code) | 코드 이름 | step3 |
| `opposite` (target=instruction, camel 지침) | 지침 지시어 | step5 |
| `opposite` (target=instruction, snake 지침) | 지침 지시어 | step5 (반대 방향) |

```python
# runner.py:196
instr_target = condition.intervention.target == "instruction"
setup = _preference_setup_instruction(condition) if instr_target else _preference_setup(condition)
span_kind = "literal" if instr_target else "def_name"
```

step3용·step5용 장치를 **둘 다** 검사한다. 지시어는 조각이 2개뿐이라
평균의 손해가 다르게 나올 수 있어서다.

---

## 6. 결과 JSON 읽는 법

```jsonc
"metrics": {
  "per_layer": {
    "17": {
      "key_shrink":         0.9512,   // Key 줄어듦   (1이면 손해 없음)
      "value_shrink":       0.8417,   // Value 줄어듦 (회전 없는 기준선)
      "key_minus_value":    0.1095,   // ← ★ 음수가 크면 Key만 손해
      "key_piece_cosine":   0.83,     // 조각들 방향이 얼마나 모여 있나
      "value_piece_cosine": 0.61
    }, …
  },
  "extra": {
    "mode": "kv_diagnose",
    "intervention_target": "code",              // 또는 instruction
    "donor": "compliant",
    "position_offsets": [-3.5, 12.0, …],        // 덮을 자리 − 공여 자리
    "position_offset_abs_mean": 8.4,            // 평균 위치 어긋남
    "donor_piece_counts": [2, 2, 3, …],         // 공여 조각 수
    "target_piece_counts": [3, 3, 2, …],        // 덮을 자리 수
    "n_substituted_tokens": 31,
    "S_clean": …, "S_base": …, "gap": …, "undecidable": false
  }
}
```

**`key_minus_value` 하나만 보면 된다.** 양수면 Key가 **덜** 줄었다는 뜻이고,
그러면 RoPE 우려는 성립하지 않는다.

```bash
python scripts/diag_kv_phase_summary.py results/diag_kv-phase
```

---

## 7. 판정 — 우려는 기각됐다

| 모델 | 대상 | Key 줄어듦 | Value 줄어듦 | 차이 (K−V) |
|---|---|---|---|---|
| Qwen2.5-Coder-3B | 코드 (step3) | 0.948 | 0.814 | **+0.134** |
| | 지침 (step5) | 0.963 | 0.892 | **+0.071** |
| DeepSeek-Coder-6.7B | 코드 | 0.957 | 0.785 | **+0.172** |
| | 지침 | 0.944 | 0.788 | **+0.156** |
| Llama-3.2-3B | 코드 | 0.948 | 0.810 | **+0.138** |
| | 지침 | 0.957 | 0.874 | **+0.083** |
| StableCode-3B | 코드 | 0.955 | 0.785 | **+0.170** |
| | 지침 | 0.948 | 0.829 | **+0.119** |

**8개 조합 전부 Key가 Value보다 덜 줄었다.** 예상과 반대 방향이다.

조각들의 방향 흩어짐을 보면 이유가 보인다:

| 대상 | Key 조각 코사인 | Value 조각 코사인 |
|---|---|---|
| 코드 | 0.80 ~ 0.85 | **0.26 ~ 0.33** |
| 지침 | 0.83 ~ 0.86 | 0.47 ~ 0.59 |

Key 벡터들은 **원래 서로 방향이 비슷하다**(코사인 0.8대). 평균을 내도 덜 지운다.
Value는 조각마다 내용이 달라 방향이 흩어져 있고(0.3대), 오히려 평균에서 더 손해를 본다.

그래서 "Key만 불리하게 망가뜨렸다"는 가설은 성립하지 않고,
**step3·step5의 Key 결과를 그대로 해석해도 된다.**

> 이 진단이 없었다면 논문 리뷰에서 반드시 나올 질문이었고,
> 답할 수 없으면 인과 주장 전체가 흔들렸다.

---

## 8. 이 진단이 다루지 않는 것

- **줄어듦만 본다.** 방향이 얼마나 틀어졌는지(회전 위상 자체)는 재지 않았다.
- 공여 값의 성질만 본다. **덮은 뒤 모델이 그 값을 어떻게 쓰는지**는 step3의 몫이다.
- 위치 어긋남과 줄어듦의 **상관**은 저장만 하고 분석하지 않았다.
- 6묶음뿐이다(step3은 42묶음). 값이 층마다 안정적이라 적게 돌렸다.

---

## 9. 직접 확인 (GPU 불필요)

```bash
# 줄어듦 지표가 직관과 맞는지 — 같은 방향이면 1, 반대면 0에 가깝다
python - <<'PY'
import torch
def shrink(t):
    return float((t.mean(dim=2).norm(dim=-1) / t.norm(dim=-1).mean(dim=2)).mean())
same = torch.tensor([[[[1.,0.],[1.,0.]]]])      # 같은 방향 두 조각
opp  = torch.tensor([[[[1.,0.],[-1.,0.]]]])     # 반대 방향 두 조각
orth = torch.tensor([[[[1.,0.],[0.,1.]]]])      # 직교
print("같은 방향:", shrink(same))   # 1.0
print("반대 방향:", shrink(opp))    # 0.0  ← 완전히 지워진다
print("직교    :", shrink(orth))    # 0.707
PY

python scripts/diag_kv_phase_summary.py results/diag_kv-phase
```
