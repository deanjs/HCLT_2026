from kit import eq
E = {
 "attn":      r"\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V",
 "qkv":       r"q_i = x_i W_Q,\qquad k_j = x_j W_K,\qquad v_j = x_j W_V",
 "output":    r"o_i=\sum_{j} a_{ij}\,v_j \qquad \mathrm{with} \qquad \sum_{j} a_{ij}=1",
 "softmax":   r"a_{ij}=\frac{\exp(q_i \cdot k_j/\sqrt{d_k})}{\sum_{j'}\exp(q_i \cdot k_{j'}/\sqrt{d_k})}",
 "pertoken":  r"\bar a_{S}=\frac{1}{|S|}\sum_{j \in S} a_{ij}",
 "avnorm":    r"\sum_{j} a_{ij}\left\| v_j \right\| \;\; \geq \;\; \left\| \sum_{j} a_{ij} v_j \right\|",
 "logp":      r"\log P(w_{1:n}) = \sum_{t=1}^{n} \log P\left(w_t \mid w_{<t}\right)",
 "score":     r"S \;=\; \log P(\mathtt{camelCase}) \;-\; \log P(\mathtt{snake\_case})",
 "recovery":  r"R \;=\; \frac{S_{\mathrm{int}} - S_{\mathrm{base}}}{S_{\mathrm{clean}} - S_{\mathrm{base}}}",
 "undec":     r"\left| S_{\mathrm{clean}} - S_{\mathrm{base}} \right| \;<\; 1.0 \;\; \Rightarrow \;\; \mathrm{undecidable}",
 "meanpool":  r"\tilde v \;=\; \frac{1}{|D|}\sum_{t \in D} v_t \;\; \longrightarrow \;\; v_j \;\; \mathrm{for\ all}\; j \in T",
 "net":       r"\Delta_b = R_b^{\,\mathrm{treat}} - R_b^{\,\mathrm{ctrl}}, \qquad \bar\Delta \pm 1.96\,\frac{s_\Delta}{\sqrt{n}}",
 "rope":      r"k_i = R_{\theta,\,i}\left(x_i W_K\right), \qquad v_i = x_i W_V",
 "residual":  r"h^{(\ell+1)} = h^{(\ell)} + \mathrm{Attn}\left(h^{(\ell)}\right) + \mathrm{FFN}(\cdot)",
 "steer":     r"h^{(\ell)} \leftarrow h^{(\ell)} + \alpha\,d, \qquad d = \bar h_{\mathrm{camel}} - \bar h_{\mathrm{snake}}",
 "spotlight": r"\psi_{\mathrm{after}} = \frac{\psi_{\mathrm{target}}}{1+\psi_{\mathrm{target}}-\psi_{\mathrm{before}}}",
 "shrink":    r"\rho_K = \frac{\left\| \tilde k \right\|}{\left\| k \right\|}, \qquad \rho_V = \frac{\left\| \tilde v \right\|}{\left\| v \right\|}",
 "compl":     r"\mathrm{compliance} = \frac{\#\{\mathrm{names\ in\ target\ notation}\}}{\#\{\mathrm{conditions}\}}",
 "ratio":     r"\frac{\bar a_{\mathrm{instr\, rule}}}{\bar a_{\mathrm{code\, camel}}} \;\; \gtrless \;\; 1",
 "peak":      r"\ell^{*} = \arg\max_{\ell} \; \bar\Delta^{(\ell)}",
}
for k, v in E.items():
    eq(k, v)
print("equations:", len(E))
