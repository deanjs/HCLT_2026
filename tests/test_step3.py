"""step3·step5의 측정 장치 자체를 검사한다 — KV 치환의 편집→측정→복구 루프.

이 루프는 연구의 **측정 도구 그 자체**다. 여기서 조용한 오류가 나면(복구 누락으로 층 효과가
누적되는 등) 모든 결과가 틀리는데, 지금까지 이를 검사하는 테스트가 하나도 없었다
(docs/step3/구조적결함.md B5 · docs/step5/구조적_결함.md C4).

torch가 없는 환경에서는 건너뛴다. 모델은 필요 없다 — 작은 가짜 캐시와 가짜 채점 함수로
`ModelHandle._score_layer_kind`의 계약만 검사한다.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="KV 치환 검사는 torch가 필요하다")

from harness.model import ModelHandle, _cache_kv          # noqa: E402


N_LAYERS, B, N_KV, SEQ, D = 3, 1, 2, 10, 4


def make_cache(seed: int):
    """legacy tuple 레이아웃의 가짜 KV 캐시 — [(k, v), ...] 층별."""
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(B, N_KV, SEQ, D, generator=g),
             torch.randn(B, N_KV, SEQ, D, generator=g)) for _ in range(N_LAYERS)]


def make_ctx(donor_cache, *, groups=None, pairs=None, score=None):
    """_score_layer_kind가 요구하는 최소 ctx."""
    return {
        "donor_cache": donor_cache,
        "groups": groups,
        "pairs": pairs,
        "viol_last": None,
        "S": score or (lambda cache, last: 0.0),
    }


def clone(cache):
    return [(k.clone(), v.clone()) for k, v in cache]


def same(a, b) -> bool:
    return all(torch.equal(ka, kb) and torch.equal(va, vb) for (ka, va), (kb, vb) in zip(a, b))


@pytest.mark.parametrize("kind", ["key", "value", "key_value"])
@pytest.mark.parametrize("use_groups", [True, False])
def test_캐시가_측정_후_원상복구된다(kind, use_groups):
    """측정이 끝나면 캐시는 **비트 단위로** 원래와 같아야 한다.

    이게 깨지면 층 스윕에서 앞 층의 편집이 뒤 층에 누적되어 층별 독립성이 무너진다.
    """
    work, donor = make_cache(0), make_cache(1)
    before = clone(work)
    groups = [([2, 3], [5, 6, 7])] if use_groups else None
    pairs = None if use_groups else [(2, 5), (3, 6)]
    ctx = make_ctx(donor, groups=groups, pairs=pairs)

    ModelHandle._score_layer_kind(None, ctx, work, layer=1, kind=kind)

    assert same(work, before), "측정 후 캐시가 원래대로 돌아오지 않았다"


@pytest.mark.parametrize("kind,touch_k,touch_v", [("key", True, False),
                                                  ("value", False, True),
                                                  ("key_value", True, True)])
def test_kind가_지정한_텐서만_바뀐다(kind, touch_k, touch_v):
    """측정 시점의 캐시를 들여다봐, key/value 중 지정한 쪽만 바뀌었는지 본다."""
    work, donor = make_cache(0), make_cache(1)
    before = clone(work)
    seen: dict[str, bool] = {}

    def spy(cache, _last):
        k, v = _cache_kv(cache, 1)
        seen["k"] = not torch.equal(k, before[1][0])
        seen["v"] = not torch.equal(v, before[1][1])
        return 0.0

    ctx = make_ctx(donor, groups=[([2, 3], [5, 6])], score=spy)
    ModelHandle._score_layer_kind(None, ctx, work, layer=1, kind=kind)

    assert seen["k"] is touch_k, f"kind={kind}인데 key 변경 여부가 {seen['k']}"
    assert seen["v"] is touch_v, f"kind={kind}인데 value 변경 여부가 {seen['v']}"


def test_평균_덮어쓰기_값이_공여_평균과_같다():
    """mean-pool은 공여 위치들의 평균 한 벡터를 대상 자리 **전부**에 넣는다."""
    work, donor = make_cache(0), make_cache(1)
    vps, dps = [2, 3, 4], [5, 6]
    captured: dict[str, torch.Tensor] = {}

    def spy(cache, _last):
        _, v = _cache_kv(cache, 1)
        captured["v"] = v.clone()
        return 0.0

    ctx = make_ctx(donor, groups=[(vps, dps)], score=spy)
    ModelHandle._score_layer_kind(None, ctx, work, layer=1, kind="value")

    expected = donor[1][1][:, :, dps, :].mean(dim=2)
    for vp in vps:
        assert torch.allclose(captured["v"][:, :, vp, :], expected), \
            f"{vp}번 자리에 공여 평균이 들어가지 않았다"


def test_다른_층은_건드리지_않는다():
    """층 L만 편집해야 한다 — 다른 층이 바뀌면 '층 국소화' 결론이 무의미해진다."""
    work, donor = make_cache(0), make_cache(1)
    before = clone(work)
    untouched: dict[int, bool] = {}

    def spy(cache, _last):
        for L in range(N_LAYERS):
            k, v = _cache_kv(cache, L)
            untouched[L] = torch.equal(k, before[L][0]) and torch.equal(v, before[L][1])
        return 0.0

    ctx = make_ctx(donor, groups=[([2], [5])], score=spy)
    ModelHandle._score_layer_kind(None, ctx, work, layer=1, kind="key_value")

    assert untouched[0] and untouched[2], "편집 대상이 아닌 층이 바뀌었다"
    assert not untouched[1], "편집 대상 층이 안 바뀌었다"


def test_미지원_개입종류는_조용히_0을_돌려주지_않는다():
    """attn_amplify는 구현이 없다. 조용히 회복률 0을 돌려주면 우리가 원하는 결론이
    미구현 코드에서 자동으로 나온다 — 반드시 예외여야 한다."""
    work, donor = make_cache(0), make_cache(1)
    ctx = make_ctx(donor, groups=[([2], [5])])
    with pytest.raises(NotImplementedError):
        ModelHandle._score_layer_kind(None, ctx, work, layer=1, kind="attn_amplify")
