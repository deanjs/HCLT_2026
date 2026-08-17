# -*- coding: utf-8 -*-
"""논문 정리 덱에만 쓰는 수식. `eqs.py`(20종) 위에 덧붙인다.

초판 덱의 `score`는 프롬프트 조건부가 빠져 있었다(S가 상수가 되어 R 식이 성립하지 않는다).
여기서 `score2`로 바로잡는다.
"""
from kit import eq

E = {
 # 선호 점수 — **프롬프트 x 조건부**를 명시한다
 "logp2":   r"\log P(y \mid x) \;=\; \sum_{k=1}^{|y|} \log P(y_k \mid x,\; y_{<k})",
 "score2":  r"S(x) \;=\; \log P(y^{\mathrm{comp}} \mid x) \;-\; \log P(y^{\mathrm{viol}} \mid x)",
 "recov2":  r"R \;=\; \frac{S(x^{\mathrm{int}}) - S(x^{\mathrm{viol}})}"
            r"{S(x^{\mathrm{clean}}) - S(x^{\mathrm{viol}})}",
 # 자리 통제 — 여집합 배치를 더하면 자리 항이 식에서 빠진다
 "poscore": r"a_{\mathcal{S}} \;=\; f(\text{position}) \;+\; g(\text{notation})",
 "poscancel": r"\frac{1}{2}[\Delta^{(s_{42})} + \Delta^{(s_{67})}] \;=\; "
              r"0 \;+\; 6\,g",
 # 관측 3지표
 "spanmetrics": r"a_{\mathcal{S}}=\sum_{j \in \mathcal{S}}\!\! a_j \qquad "
                r"av_{\mathcal{S}}=\sum_{j \in \mathcal{S}}\!\! a_j\|v_j\| \qquad "
                r"v_{\mathcal{S}}=\frac{1}{|\mathcal{S}|}\sum_{j \in \mathcal{S}}\!\!\|v_j\|",
 "pertoken2": r"\bar a_{\mathcal{S}} \;=\; \frac{1}{|\mathcal{S}|}\sum_{j \in \mathcal{S}} a_j",
 # 순효과 — 묶음끼리 먼저 뺀다
 "net2":    r"\bar\Delta \;=\; \frac{1}{42}\sum_{b=1}^{42}(R_b^{\mathrm{treat}} - R_b^{\mathrm{ctrl}})",
 # 처방
 "steer2":  r"d^{(\ell)} = \bar h^{(\ell)}_{\mathrm{camel}} - \bar h^{(\ell)}_{\mathrm{snake}}"
            r"\;,\qquad h^{(\ell)} \leftarrow h^{(\ell)} + \alpha\, d^{(\ell)}",
 "spot2":   r"r=\frac{\psi_{\mathrm{target}}}{\psi_{\mathrm{before}}}\;,\qquad "
            r"p_j^{\mathrm{new}} = \frac{p_j\, r^{\,\mathbf{1}[j \in \mathcal{S}]}}"
            r"{\sum_i p_i\, r^{\,\mathbf{1}[i \in \mathcal{S}]}}",
 # 진짜 준수율
 "realcompl": r"\mathrm{compliance} \;=\; \mathbf{1}[\text{notation ok}]\;\cdot\;"
              r"\mathbf{1}[\text{on-task}]",
 # 평균 덮어쓰기
 "meanpool2": r"\tilde k \;=\; \frac{1}{|D|}\sum_{d \in D} k_d^{\mathrm{donor}} "
              r"\quad \longrightarrow \quad k_p \;\;\; \forall\, p \in P",
}
for k, v in E.items():
    eq(k, v)
print("paper equations:", len(E))
