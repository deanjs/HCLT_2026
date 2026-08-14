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
    mode: str = "all",
) -> tuple[list[tuple[int, int]], list[int]]:
    """위반 이름들과 공여 이름들의 토큰 위치를 **역할별로 정렬**한다.

    viol_names_tokens : 위반 프롬프트에서 각 이름의 토큰 인덱스 목록(이름 순).
    donor_names_tokens: 공여(준수) 프롬프트에서 각 이름의 토큰 인덱스 목록(같은 순서·개수).

    mode:
      - "all"  : 이름당 **전체 토큰**을 첫↔첫·둘째↔둘째로 정렬(step1 방식). 토큰 수가
                 다르면(예 snake 3 vs camel 2) 그 이름은 정렬 불가로 **제외**된다.
      - "last" : 이름당 **마지막 토큰 하나만** 정렬(step2 방식, 옵션 B). 토큰 수가 달라도
                 항상 1:1이라 제외가 없다. camel/snake 토크나이저가 밑줄을 다르게 쪼개
                 all 모드가 전부 스킵될 때 쓴다. 표기 차이(`matrix`↔`Matrix`)가 주로
                 마지막 토큰에 담긴다.

    반환:
      pairs   : [(위반_위치, 공여_위치), ...] — 이 위치쌍대로 KV를 치환한다.
      skipped : 정렬 불가(빈 토큰 또는 all 모드 토큰 수 불일치)로 **제외**된 이름 인덱스.
    """
    if mode not in ("all", "last"):
        raise ValueError('mode는 "all" 또는 "last"만 허용한다')
    if len(viol_names_tokens) != len(donor_names_tokens):
        raise ValueError(
            f"이름 개수 불일치: 위반 {len(viol_names_tokens)} vs 공여 {len(donor_names_tokens)}"
        )
    pairs: list[tuple[int, int]] = []
    skipped: list[int] = []
    for i, (vt, dt) in enumerate(zip(viol_names_tokens, donor_names_tokens)):
        if not vt or not dt:                 # 한쪽이 비면(이름 못 찾음) 정렬 불가
            skipped.append(i)
            continue
        if mode == "last":
            pairs.append((int(vt[-1]), int(dt[-1])))   # 마지막 토큰만 1:1
            continue
        if len(vt) != len(dt):               # all 모드: 토큰 수 다르면 제외
            skipped.append(i)
            continue
        for vp, dp in zip(vt, dt):
            pairs.append((int(vp), int(dp)))
    return pairs, skipped


def align_name_groups(
    viol_names_tokens: Sequence[Sequence[int]],
    donor_names_tokens: Sequence[Sequence[int]],
) -> tuple[list[tuple[list[int], list[int]]], list[int]]:
    """mean-pool 치환용 — 이름별 (위반 위치들, 공여 위치들) 묶음을 돌려준다.

    align_name_tokens(1:1 짝짓기)와 달리 **개수 불일치를 허용**한다. 치환 시 공여 토큰들의
    KV를 평균 내(한 벡터) 위반 이름의 **모든 토큰 자리**에 브로드캐스트해 넣는다
    (model._score_layer_kind의 mean 경로). last(1개)보다 단어 전체를 덮어 신호가 강하고,
    all과 달리 토큰 수가 달라도 스킵되지 않는다(step4 모델 다양성).

    반환:
      groups  : [(위반_위치들, 공여_위치들), ...] — 이름 순, 한쪽이라도 비면 제외.
      skipped : 한쪽 위치가 비어(이름 못 찾음) 제외된 이름 인덱스.
    """
    if len(viol_names_tokens) != len(donor_names_tokens):
        raise ValueError(
            f"이름 개수 불일치: 위반 {len(viol_names_tokens)} vs 공여 {len(donor_names_tokens)}"
        )
    groups: list[tuple[list[int], list[int]]] = []
    skipped: list[int] = []
    for i, (vt, dt) in enumerate(zip(viol_names_tokens, donor_names_tokens)):
        if not vt or not dt:
            skipped.append(i)
            continue
        groups.append(([int(x) for x in vt], [int(x) for x in dt]))
    return groups, skipped


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


def peak_layer(recovery_by_layer: dict[int, Optional[float]]) -> Optional[tuple[int, float]]:
    """층별 회복률에서 **피크 층**과 그 값을 찾는다(step 1 층 스윕 요약).

    회복률 피크가 어느 층인가 = "표기 형태 결정이 어느 처리 단계에서 굳는가"의 답
    (계획서 §2.5). None(구별 불가) 층은 제외한다. 유효 값이 없으면 None.
    """
    valid = {L: r for L, r in recovery_by_layer.items() if r is not None}
    if not valid:
        return None
    L = max(valid, key=lambda k: valid[k])
    return L, valid[L]
