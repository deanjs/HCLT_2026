"""개념부 도해."""
from kit import *

# ── 1. 트랜스포머 블록 ──────────────────────────────────────────────────────
d = Dia(1000, 470)
d.box(40, 30, 260, 44, "Token embeddings", fs=9.5)
d.arrow((170, 74), (170, 108))
for i, (y, lab, sub) in enumerate([(108, "Layer 1", None), (176, "Layer 2", None)]):
    d.box(40, y, 260, 44, lab, fs=9.5)
    d.arrow((170, y + 44), (170, y + 68))
d.text(170, 262, "⋮", ha="center", fs=13, color=MUTE)
d.box(40, 288, 260, 44, "Layer L", fs=9.5, ec=BRAND)
d.arrow((170, 332), (170, 366))
d.box(40, 366, 260, 44, "Next-token logits", fs=9.5)
d.line(340, 30, 340, 440, color=HAIR)
d.eyebrow(376, 46, "inside one layer")
d.box(376, 70, 560, 96, None)
d.text(400, 96, "Self-attention", fs=10, color=WHITE)
d.box(400, 110, 150, 40, "Q · K", "who to look at", fs=10, subfs=8)
d.box(566, 110, 150, 40, "a · V", "what to take", fs=10, subfs=8, ec=BRAND)
d.text(738, 130, "→ output", fs=9, color=MUTE)
d.arrow((656, 250), (656, 210), style="-|>")
d.box(376, 250, 560, 74, None)
d.text(400, 274, "Feed-forward network", fs=10, color=WHITE)
d.text(400, 300, "position-wise, no token mixing", fs=9, color=MUTE)
d.text(376, 356, "Residual stream: every layer adds to the same vector.", fs=10, color=ASH)
d.text(376, 384, "Two roads: how much (Key), and what (Value).", fs=10, color=ASH)
d.text(376, 418, "This study cuts along that seam.", fs=10, color=BRAND)
d.save("c_block")

# ── 2. 어텐션 두 경로 ────────────────────────────────────────────────────────
d = Dia(1000, 380)
d.eyebrow(20, 24, "path 1 — how much to look  (Key side)")
d.box(20, 44, 170, 52, "query  q_i", fs=9.5, mono=True)
d.arrow((190, 70), (250, 70))
d.box(250, 44, 170, 52, "keys  k_j", fs=9.5, mono=True)
d.arrow((420, 70), (480, 70))
d.box(480, 44, 200, 52, "scores → softmax", fs=9)
d.arrow((680, 70), (740, 70))
d.box(740, 44, 190, 52, "weights  a_ij", fs=9.5, mono=True, ec=HAIR)
d.eyebrow(20, 174, "path 2 — what to take  (Value side)")
d.box(20, 194, 170, 52, "values  v_j", fs=9.5, mono=True, ec=BRAND)
d.arrow((190, 220), (740, 220), rad=-0.06)
d.box(740, 194, 190, 52, "output  o_i", fs=9.5, mono=True)
d.arrow((835, 96), (835, 194), ls=(0, (3, 3)))
d.text(852, 148, "weighted", fs=8, color=MUTE)
d.line(20, 288, 930, 288, color=HAIR)
d.text(20, 316, "Key decides the share of attention.  Value decides the content that arrives.", fs=9.5, color=ASH)
d.text(20, 344, "Overwrite one and leave the other intact — that is the whole experimental design.", fs=9.5, color=ASH)
d.save("c_paths")

# ── 3. 헤드 분할 ────────────────────────────────────────────────────────────
d = Dia(980, 360)
d.eyebrow(20, 22, "one hidden vector is split across heads")
d.box(20, 44, 620, 40, None, ec=HAIR)
for i in range(8):
    d.cell(24 + i * 77, 48, 73, 32, SOFT if i != 3 else "#3a2a24", HAIR)
    d.text(24 + i * 77 + 36, 64, f"h{i+1}", ha="center", fs=8, color=WHITE if i != 3 else BRAND, mono=True)
d.text(660, 64, "d_model = 8 × d_head", fs=9, color=MUTE, mono=True)
d.arrow((176, 84), (176, 130))
d.box(20, 130, 300, 150, None, ec=BRAND)
d.text(40, 156, "Head 4 — its own Q, K, V", fs=9.5, color=WHITE)
d.text(40, 182, "sees the same tokens", fs=8.5, color=MUTE)
d.text(40, 204, "keeps its own attention row", fs=8.5, color=MUTE)
d.text(40, 226, "learns its own specialty", fs=8.5, color=MUTE)
d.text(40, 254, "We average over all heads.", fs=8.5, color=BRAND)
d.box(360, 130, 600, 150, None)
d.text(384, 158, "Why average and not pick one head", fs=10, color=WHITE)
d.text(384, 190, "— No head was pre-identified as the notation head; picking one after", fs=9, color=ASH)
d.text(408, 210, "seeing the result would be selection on the outcome.", fs=9, color=ASH)
d.text(384, 236, "— The claim is about the layer, not about a single head. Averaging keeps", fs=9, color=ASH)
d.text(408, 256, "the estimand at the layer level and the comparison honest.", fs=9, color=ASH)
d.text(20, 322, "Head count differs by model: Qwen 16, Llama 24, DeepSeek 32, StableCode 32.", fs=9, color=MUTE)
d.save("c_heads")

# ── 4. 쿼리 행 ─────────────────────────────────────────────────────────────
d = Dia(980, 400)
n = 14
cw, ch, x0, y0 = 40, 22, 40, 60
d.eyebrow(x0, 30, "attention matrix  a_ij   (rows = query position, cols = key position)")
import random
random.seed(42)
for i in range(n):
    for j in range(n):
        if j > i:
            d.cell(x0 + j * cw, y0 + i * ch, cw - 2, ch - 2, "#141414", "#141414")
        else:
            g = 0.10 + 0.5 * random.random() * (1.0 if i != n - 1 else 0)
            col = BRAND if i == n - 1 else (str(g))
            a = 1.0 if i == n - 1 else 1.0
            d.cell(x0 + j * cw, y0 + i * ch, cw - 2, ch - 2, col if i != n - 1 else BRAND, None)
d.text(x0 - 10, y0 + (n - 1) * ch + 10, "last", ha="right", fs=8, color=BRAND, mono=True)
d.text(x0 - 10, y0 + 10, "first", ha="right", fs=8, color=MUTE, mono=True)
d.text(x0 + n * cw + 16, y0 + (n - 1) * ch + 10, "← the row we read", fs=9, color=BRAND)
d.text(x0, y0 + n * ch + 30, "Upper triangle is masked: a token cannot attend to the future.", fs=9, color=MUTE)
d.box(40, 296, 900, 84, None)
d.text(64, 322, "We read one row only — the position that is about to emit the function name.", fs=9.5, color=WHITE)
d.text(64, 350, "That row sums to 1, so each span's value is literally its share of the model's attention.", fs=9, color=ASH)
d.save("c_queryrow")

# ── 5. KV 캐시 ─────────────────────────────────────────────────────────────
d = Dia(980, 400)
d.eyebrow(20, 24, "what the cache holds, per layer, per position")
d.box(20, 46, 440, 130, None)
d.text(44, 74, "past_key_values", fs=10, color=WHITE, mono=True)
d.text(44, 104, "tuple of L layers", fs=8.5, color=MUTE)
d.text(44, 128, "each layer: (K, V)", fs=8.5, color=MUTE, mono=True)
d.text(44, 152, "shape  [batch, kv_heads, seq, d_head]", fs=8.5, color=MUTE, mono=True)
d.box(500, 46, 460, 130, None, ec=BRAND)
d.text(524, 74, "Why this is the intervention surface", fs=10, color=WHITE)
d.text(524, 104, "It is the only place where “what the model already", fs=9, color=ASH)
d.text(524, 124, "read” sits as an editable tensor, split into the two", fs=9, color=ASH)
d.text(524, 144, "roads — K and V — before they are ever combined.", fs=9, color=ASH)
seq = ["def", " split", "_config", "(", "value", ")", ":", "…"]
x = 40
d.eyebrow(20, 224, "sequence positions of one violating name")
for i, t in enumerate(seq):
    w = 96
    hit = t in (" split", "_config")
    d.box(x, 246, w - 8, 40, t, fs=8.5, mono=True,
          ec=BRAND if hit else HAIR, tc=BRAND if hit else ASH)
    x += w
d.text(40, 312, "Only the name pieces are overwritten — the keyword, parentheses and body stay untouched.", fs=9, color=ASH)
d.text(40, 340, "Same positions, same length, same prompt. Only the cached content changes.", fs=9, color=ASH)
d.save("c_kvcache")

# ── 6. RoPE ────────────────────────────────────────────────────────────────
d = Dia(980, 360)
d.eyebrow(20, 24, "position enters through the key, not the value")
d.box(20, 46, 200, 56, "x_i", "hidden state", fs=10, mono=True, subfs=7.5)
d.arrow((220, 62), (300, 62)); d.arrow((220, 90), (300, 90))
d.box(300, 34, 180, 48, "W_K", fs=10, mono=True)
d.box(300, 96, 180, 48, "W_V", fs=10, mono=True, ec=BRAND)
d.arrow((480, 58), (560, 58))
d.box(560, 34, 200, 48, "rotate by i", fs=9.5)
d.arrow((760, 58), (830, 58))
d.box(830, 34, 120, 48, "k_i", fs=10, mono=True)
d.arrow((480, 120), (830, 120), rad=0.0)
d.box(830, 96, 120, 48, "v_i", fs=10, mono=True, ec=BRAND)
d.text(560, 130, "no rotation applied", fs=9, color=BRAND)
d.line(20, 190, 950, 190, color=HAIR)
d.text(20, 220, "Consequence for this study", fs=10, color=WHITE)
d.text(20, 250, "— Overwriting a Key transplants a vector whose rotation belongs to a different position.", fs=9, color=ASH)
d.text(44, 272, "That is a real confound and it is why the diagnostic step exists.", fs=9, color=ASH)
d.text(20, 300, "— Values carry no direct rotation, but deep-layer values are still built from context,", fs=9, color=ASH)
d.text(44, 322, "so “Value is position-free” is a claim we do not make.", fs=9, color=ASH)
d.save("c_rope")

# ── 7. GQA ─────────────────────────────────────────────────────────────────
d = Dia(980, 380)
d.eyebrow(20, 24, "query heads share one key/value pair")
rows = [("Qwen2.5-Coder-3B", 16, 2, 8), ("Llama-3.2-3B", 24, 8, 3),
        ("DeepSeek-Coder-6.7B", 32, 32, 1), ("StableCode-3B", 32, 32, 1)]
y = 52
d.text(20, y, "model", fs=8.5, color=MUTE); d.text(300, y, "Q heads", fs=8.5, color=MUTE)
d.text(430, y, "KV heads", fs=8.5, color=MUTE); d.text(570, y, "group size", fs=8.5, color=MUTE)
d.text(730, y, "what one overwrite touches", fs=8.5, color=MUTE)
d.line(20, y + 14, 960, y + 14, color=HAIR)
for i, (name, q, kv, g) in enumerate(rows):
    yy = y + 46 + i * 40
    d.text(20, yy, name, fs=9.5, color=WHITE)
    d.text(300, yy, str(q), fs=9.5, color=ASH, mono=True)
    d.text(430, yy, str(kv), fs=9.5, color=ASH, mono=True)
    d.text(570, yy, str(g), fs=9.5, color=BRAND if g > 1 else ASH, mono=True)
    d.text(730, yy, f"{g} query head{'s' if g>1 else ''}", fs=9, color=ASH)
d.line(20, y + 46 + 4 * 40 - 18, 960, y + 46 + 4 * 40 - 18, color=HAIR)
d.text(20, 288, "Substitution is performed per KV group, never per query head.", fs=9.5, color=WHITE)
d.text(20, 316, "Getting this wrong on Qwen would silently edit 8 heads while reporting 1 —", fs=9, color=ASH)
d.text(20, 338, "so the group layout is re-read from the config for every model, every run.", fs=9, color=ASH)
d.save("c_gqa")

# ── 8. 잔차 스트림 ──────────────────────────────────────────────────────────
d = Dia(980, 340)
d.eyebrow(20, 24, "the running vector every layer writes into")
d.line(60, 90, 900, 90, color=HAIR, lw=2)
for i, lab in enumerate(["L1", "L2", "L3", "…", "L20", "…", "L36"]):
    x = 60 + i * 140
    d.box(x - 44, 60, 88, 60, lab, fs=9, mono=True,
          ec=BRAND if lab == "L20" else HAIR)
    if i:
        d.arrow((x - 140 + 44, 90), (x - 44, 90))
d.text(60, 156, "Each block adds its output back into the same stream — it never replaces it.", fs=9.5, color=ASH)
d.text(60, 184, "So a vector added at layer 20 is still present at layer 36, reshaped by everything between.", fs=9.5, color=ASH)
d.box(60, 220, 840, 96, None, ec=BRAND)
d.text(84, 248, "Why the prescription is called residual-stream steering, not “Value steering”", fs=10, color=WHITE)
d.text(84, 278, "The hook adds a direction to the decoder block's output. Downstream layers then recompute", fs=9, color=ASH)
d.text(84, 298, "their own Q, K, V and FFN from it — so the edit is not confined to the Value road.", fs=9, color=ASH)
d.save("c_residual")
print("concept ok")
