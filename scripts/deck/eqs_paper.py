# -*- coding: utf-8 -*-
"""논문 정리 덱에만 쓰는 수식. `eqs.py`(20종) 위에 덧붙인다.

**영어 줄임말을 수식에 넣지 않는다.** matplotlib mathtext는 한글 글리프가 없어
`\\mathrm{처치}`가 두부로 렌더된다(확인함). 그래서 수식에는 **기호만** 두고,
슬라이드에서 `D.symbols()`로 **한글 뜻풀이를 바로 아래**에 붙인다.

    ✕  R_b^{treat} - R_b^{ctrl}          treat·ctrl이 무엇인지 그림만 봐선 모른다
    ○  R_b - R'_b   +  아래에 "R 처치 · R′ 통제"

초판 `eqs.py`의 `score`는 프롬프트 조건부가 빠져 있었다(S가 상수가 되어 R 식이
성립하지 않는다). 여기서 `score2`로 바로잡는다.
"""
from kit import eq

E = {
 # ── 선호 점수 — 프롬프트 x 조건부를 명시한다. 후보는 +(준수) / −(위반)
 "logp2":   r"\log P(y \mid x) \;=\; \sum_{k=1}^{|y|} \log P(y_k \mid x,\; y_{<k})",
 "score2":  r"S(x) \;=\; \log P(y^{+} \mid x) \;-\; \log P(y^{-} \mid x)",

 # ── 되돌림률 — 세 상태를 번호로 두고 아래에서 한글로 푼다
 "recov2":  r"R \;=\; \frac{S_{2} - S_{0}}{S_{1} - S_{0}}",
 "undec2":  r"\left| S_{1} - S_{0} \right| \;<\; 1.0",

 # ── 자리 통제 — f는 자리, g는 표기
 "poscore": r"a \;=\; f(p) \;+\; g(n)",
 "poscancel": r"\frac{1}{2}\left[\Delta_{42} + \Delta_{67}\right] \;=\; 0 \;+\; 6\,g",

 # ── 관측 3지표
 "spanmetrics": r"a_{S}=\sum_{j \in S} a_j \qquad "
                r"(av)_{S}=\sum_{j \in S} a_j\|v_j\| \qquad "
                r"v_{S}=\frac{1}{|S|}\sum_{j \in S}\|v_j\|",

 # ── 순효과 — 처치 R, 통제 R′. 묶음끼리 먼저 뺀다
 "net2":    r"\bar\Delta \;=\; \frac{1}{42}\sum_{b=1}^{42}\left(R_b - R'_b\right)",

 # ── 평균 덮어쓰기 — D 공여 자리, P 덮을 자리
 "meanpool2": r"\tilde k \;=\; \frac{1}{|D|}\sum_{d \in D} k_d "
              r"\quad \longrightarrow \quad k_p \;\;\; \forall\, p \in P",

 # ── 처방
 "steer2":  r"d^{(\ell)} = \bar h^{(\ell)}_{+} - \bar h^{(\ell)}_{-}"
            r"\;,\qquad h^{(\ell)} \leftarrow h^{(\ell)} + \alpha\, d^{(\ell)}",
 "spot2":   r"r=\frac{\psi^{*}}{\psi}\;,\qquad "
            r"p_j' = \frac{p_j\, r^{\,[\,j \in S\,]}}{\sum_i p_i\, r^{\,[\,i \in S\,]}}",
 "spotafter": r"\psi' \;=\; \frac{\psi^{*}}{1+\psi^{*}-\psi}",

 # ── 어텐션 출력 — 초판 output의 \mathrm{with}(영어)를 뺀다
 "output2": r"o_i \;=\; \sum_{j} a_{ij}\, v_j \;,\qquad \sum_{j} a_{ij} = 1",

 # ── 진짜 준수 — 초판 realcompl의 "notation ok"·"on-task"(영어)를 기호로
 "compl2":  r"c \;=\; n \cdot t \;,\qquad n,\, t \in \{0,\, 1\}",
}
for k, v in E.items():
    eq(k, v)
print("paper equations:", len(E))
