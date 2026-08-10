"""어텐션·‖v‖·‖av‖ 집계 — GQA 매핑을 포함한 순수 함수(모델 의존성 없음).

이름을 생성하는 디코딩 시점의 query 한 행(마지막 위치)이 컨텍스트의 각 구간을
얼마나 보는지(축 A), 그 구간 토큰이 실어 나르는 표현의 크기(‖v‖, 축 B),
잔차 스트림에 실제 기여하는 양(‖av‖, 축 A×B)을 구간별로 집계한다.

**GQA 방법 A(repeat 후 정렬).** 어텐션 `a`는 query head별(Hq종), value는 KV head별
(Hkv종)이다. `av[h,j] = a[h,j]·‖v[h÷group_size, j]‖`로, query head h를 그 그룹의
KV head(h÷group_size)에 매핑해 곱한다. a가 스칼라(≥0)이므로 ‖av[h,j]‖ = a[h,j]·‖v‖는
근사가 아니라 정확하다(계획서 §2.4의 항등식).

이 파일은 torch/numpy 없이 리스트만 다룬다. 모델 쪽(model.py)이 작은 배열
(마지막 query 행 [Hq, seq], 토큰별 ‖v‖ [Hkv, seq])만 리스트로 변환해 넘긴다.
큰 텐서를 파이썬으로 옮기지 않으므로 가볍고, 여기서 단위 테스트가 가능하다.
"""

from __future__ import annotations

from typing import Optional, Sequence


def span_metrics(
    attn_last: Sequence[Sequence[float]],   # [Hq][seq]  마지막 query의 어텐션(query head별)
    vnorm_kv: Sequence[Sequence[float]],    # [Hkv][seq] 토큰별 value norm(KV head별)
    group_size: int,                        # Hq // Hkv (한 KV를 공유하는 query head 수)
    spans: dict[str, Sequence[int]],        # 구간 이름 → 토큰 인덱스 목록
) -> dict[str, dict]:
    """구간별 (어텐션 합·‖av‖ 합·‖v‖ 평균)을 계산한다.

    - attention_weight : 구간 토큰에 대한 어텐션 **합**(query head 평균). 각 head의 행은
                         전체 위치에 대해 1로 합해지므로, 구간 합은 그 head가 구간에 준
                         어텐션 질량(0~1)이고, head 평균은 그 평균 질량이다(§2.7 "구간별 합").
    - av_norm          : ‖av‖의 구간 **합**(query head 평균). 구간이 잔차에 기여한 총량.
    - v_norm           : 구간 토큰 ‖v‖의 **평균**(고유 KV head로 접어서). 크기 지표라 합이
                         아니라 토큰당 평균이 맞다.
    - n_tokens         : 구간 토큰 수(합↔평균 환산·해석용).

    빈 구간은 값이 None이다.
    """
    Hq = len(attn_last)
    Hkv = len(vnorm_kv)
    if group_size <= 0 or Hq != Hkv * group_size:
        raise ValueError(
            f"GQA 매핑 불일치: Hq={Hq}, Hkv={Hkv}, group_size={group_size} "
            f"(Hq == Hkv*group_size 이어야 한다)"
        )

    out: dict[str, dict] = {}
    for name, idxs in spans.items():
        idxs = list(idxs)
        if not idxs:
            out[name] = {"attention_weight": None, "av_norm": None,
                         "v_norm": None, "n_tokens": 0}
            continue

        # 축 A — 구간 어텐션 합, query head 평균
        a_per_head = [sum(attn_last[h][j] for j in idxs) for h in range(Hq)]
        attention_weight = sum(a_per_head) / Hq

        # 축 A×B — ‖av‖ 구간 합, query head 평균. v는 h÷group_size로 매핑(방법 A).
        av_per_head = [
            sum(attn_last[h][j] * vnorm_kv[h // group_size][j] for j in idxs)
            for h in range(Hq)
        ]
        av_norm = sum(av_per_head) / Hq

        # 축 B(크기) — ‖v‖ 토큰당 평균(고유 KV head로 접어서). head 0..g-1은 동일 원본이라
        # 16개 평균이 아니라 KV head 수만큼만 평균낸다(독립 측정 과장 방지).
        v_vals = [vnorm_kv[k][j] for k in range(Hkv) for j in idxs]
        v_norm = sum(v_vals) / len(v_vals)

        out[name] = {
            "attention_weight": attention_weight,
            "av_norm": av_norm,
            "v_norm": v_norm,
            "n_tokens": len(idxs),
        }
    return out


def locate_token_spans(
    offsets: Sequence[tuple[int, int]],
    char_spans: dict[str, Sequence[tuple[int, int]]],
) -> dict[str, list[int]]:
    """토큰 offset_mapping과 문자 구간을 받아 각 구간에 겹치는 토큰 인덱스를 찾는다.

    offsets    : 토큰별 (문자시작, 문자끝). 특수 토큰은 보통 (0,0) 등 빈 구간이라 건너뛴다.
    char_spans : 구간 이름 → [(문자시작, 문자끝), ...] (한 구간에 여러 등장 가능).
    반환       : 구간 이름 → 겹치는 토큰 인덱스 목록(정렬·중복 제거).

    토큰 [ts,te)와 문자구간 [cs,ce)가 겹치면(ts<ce and te>cs) 그 토큰을 구간에 넣는다.
    이름이 하위토큰 여러 개로 쪼개져도 전부 포착된다.
    """
    out: dict[str, list[int]] = {}
    for name, ranges in char_spans.items():
        hit: list[int] = []
        for ti, (ts, te) in enumerate(offsets):
            if te <= ts:                     # 특수 토큰/빈 offset
                continue
            if any(ts < ce and te > cs for (cs, ce) in ranges):
                hit.append(ti)
        out[name] = hit
    return out


def find_char_spans(text: str, needles: Sequence[str]) -> list[tuple[int, int]]:
    """text 안에서 needles의 모든 (겹치지 않는) 등장 위치를 문자 구간으로 반환한다.

    같은 이름이 선행 코드에 여러 번 나올 수 있으므로 전부 찾는다. 정의부만 잡도록
    호출부에서 `def <name>(` 형태의 needle을 넘기는 것을 권장한다(이름 단독은 본문·
    호출에서 우연히 겹칠 수 있음). 여기서는 순수 문자열 검색만 한다.
    """
    spans: list[tuple[int, int]] = []
    for needle in needles:
        if not needle:
            continue
        start = 0
        while True:
            i = text.find(needle, start)
            if i < 0:
                break
            spans.append((i, i + len(needle)))
            start = i + len(needle)
    return spans
