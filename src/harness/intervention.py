"""개입(step C) 순수 헬퍼 — 정렬·회복률·후보 채점(모델 의존성 없음).

모델 내부(KV 캐시 편집)는 model.py가 맡고, 여기서는 torch 없이 리스트/스칼라만 다뤄
단위 테스트가 가능한 부분을 담는다: 위반↔공여 이름 토큰 위치 정렬, 준수 선호 점수의
회복률, 후보 로그확률 합.
"""

from __future__ import annotations

from typing import Optional, Sequence


def align_name_tokens(
    viol_names_tokens: Sequence[Sequence[int]],
    donor_names_tokens: Sequence[Sequence[int]],
) -> tuple[list[tuple[int, int]], list[int]]:
    """위반 이름들과 공여 이름들의 토큰 위치를 **역할별(첫↔첫, 둘째↔둘째)로 정렬**한다.

    viol_names_tokens : 위반 프롬프트에서 각 이름의 토큰 인덱스 목록(이름 순).
    donor_names_tokens: 공여(준수) 프롬프트에서 각 이름의 토큰 인덱스 목록(같은 순서·개수).

    반환:
      pairs   : [(위반_위치, 공여_위치), ...] — 이 위치쌍대로 KV를 치환한다.
      skipped : 토큰 수가 달라 정렬 불가로 **제외**된 이름 인덱스 목록(기록용).

    우리 풀은 camel·snake 모두 이름당 2토큰이라 대부분 2:2로 정렬된다.
    """
    if len(viol_names_tokens) != len(donor_names_tokens):
        raise ValueError(
            f"이름 개수 불일치: 위반 {len(viol_names_tokens)} vs 공여 {len(donor_names_tokens)}"
        )
    pairs: list[tuple[int, int]] = []
    skipped: list[int] = []
    for i, (vt, dt) in enumerate(zip(viol_names_tokens, donor_names_tokens)):
        if len(vt) != len(dt):
            skipped.append(i)
            continue
        for vp, dp in zip(vt, dt):
            pairs.append((int(vp), int(dp)))
    return pairs, skipped


def recovery_rate(
    s_clean: float, s_base: float, s_int: float, eps: float = 1e-9
) -> Optional[float]:
    """준수 선호 점수 세 상태로 회복률을 계산한다.

        회복률 = (S_개입 − S_위반) / (S_깨끗 − S_위반)

    - 1.0 = 깨끗(위반 없음) 수준까지 완전 회복.
    - 0.0 = 개입해도 위반 baseline과 동일(회복 없음).
    - 분모(깨끗−위반)가 0에 가까우면(구별 불가) None.
    """
    denom = s_clean - s_base
    if abs(denom) < eps:
        return None
    return (s_int - s_base) / denom


def sum_logprob(token_logprobs: Sequence[float]) -> float:
    """후보 이름의 토큰별 로그확률 합 = 그 후보 문자열의 로그확률."""
    return float(sum(token_logprobs))


def preference_score(logp_compliant: float, logp_violation: float) -> float:
    """준수 선호 점수 S = logP(준수 후보) − logP(위반 후보). 양수=준수 선호."""
    return logp_compliant - logp_violation
