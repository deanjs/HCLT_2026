"""연구계획·종합부 도해."""
from kit import *

# 덱 지도
d = Dia(1680, 300)
secs = [("01", "Plan", "what we set out to test"), ("02", "Concepts", "transformer & intervention"),
        ("03", "Audit", "why v1 was discarded"), ("04", "Harness", "one engine, many conditions"),
        ("05", "Experiments", "step 1 · 2 · 3 · 4 · diag · 6"), ("06", "Synthesis", "answers & limits")]
w = 265
for i, (n, t, s) in enumerate(secs):
    x = i * (w + 15)
    d.box(x, 40, w, 140, None, ec=BRAND if i == 4 else HAIR)
    d.text(x + 22, 74, n, fs=11, color=BRAND if i == 4 else MUTE, mono=True)
    d.text(x + 22, 108, t, fs=12, color=WHITE)
    d.text(x + 22, 140, s, fs=8.5, color=MUTE)
    if i: d.arrow((x - 15, 110), (x - 2, 110))
d.text(0, 232, "Step 5 (instruction causality) is held back until its full-layer control sweep is run —", fs=9.5, color=ASH)
d.text(0, 258, "its slides join this deck once the missing control curve exists.", fs=9.5, color=ASH)
d.save("p_map")

# 주장: 어텐션 대 내용
d = Dia(1000, 400)
d.eyebrow(20, 24, "the rival account")
d.box(20, 46, 460, 130, None)
d.text(44, 78, "“The model does not look at the instruction enough.”", fs=10.5, color=WHITE)
d.text(44, 110, "Prescription: raise the attention mass on the", fs=9, color=ASH)
d.text(44, 130, "instruction span. This is what Spotlight does.", fs=9, color=ASH)
d.text(44, 158, "Locus: the Key road.", fs=9, color=MUTE)
d.eyebrow(520, 24, "what we argue")
d.box(520, 46, 460, 130, None, ec=BRAND)
d.text(544, 78, "“It looks. What arrives is wrong.”", fs=10.5, color=WHITE)
d.text(544, 110, "Prescription: push the content carried by the", fs=9, color=ASH)
d.text(544, 130, "Value road at a specific late layer.", fs=9, color=ASH)
d.text(544, 158, "Locus: the Value road, layer-local.", fs=9, color=BRAND)
d.line(20, 216, 980, 216, color=HAIR)
d.text(20, 248, "The two accounts make opposite predictions under the same measurement,", fs=9.5, color=ASH)
d.text(20, 274, "so a single experiment can separate them — that is what step 6 does.", fs=9.5, color=ASH)
d.box(20, 306, 960, 78, None)
d.text(44, 334, "Both are tested on one instrument, one seed, one dataset, four models.", fs=9.5, color=WHITE)
d.text(44, 362, "Neither side gets a tuned advantage: the sweep grid is fixed before results are seen.", fs=9, color=MUTE)
d.save("p_claim")

# RQ 지도
d = Dia(1680, 330)
rq = [("RQ1", "Does violating context\nlower compliance?", ["step 1"], BRAND),
      ("RQ2", "Is the code signal content\nor form — and which layer?", ["step 2 · observe", "step 3 · cause"], None),
      ("RQ3", "Does the instruction travel\nthe same road?", ["step 4 · observe", "step 5 · cause"], None),
      ("Rx", "Push the value, not\nthe attention.", ["step 6 · prescription", "diag · instrument check"], None)]
x = 0
for i, (n, q, steps, col) in enumerate(rq):
    w = 400
    d.box(x, 34, w, 200, None, ec=BRAND if i == 3 else HAIR)
    d.text(x + 24, 66, n, fs=11.5, color=BRAND if i == 3 else WHITE, mono=True)
    for k, line in enumerate(q.split("\n")):
        d.text(x + 24, 102 + k * 22, line, fs=9.5, color=ASH)
    for k, s in enumerate(steps):
        d.text(x + 24, 164 + k * 24, "— " + s, fs=9, color=MUTE)
    if i: d.arrow((x - 20, 134), (x - 4, 134))
    x += w + 20
d.text(0, 274, "Step 6 is not a fourth research question. It is the prescription that RQ1–RQ3 imply,", fs=9.5, color=ASH)
d.text(0, 300, "measured on the same instrument — a test of whether the diagnosis was right.", fs=9.5, color=ASH)
d.save("p_rq")

# 데이터셋
d = Dia(1000, 400)
d.eyebrow(20, 24, "one name pool, reused by every step")
d.box(20, 46, 210, 78, "50 verbs", "split · fetch · render …", fs=10, subfs=7.5)
d.text(246, 88, "×", fs=13, color=MUTE, ha="center")
d.box(266, 46, 210, 78, "50 nouns", "config · token · buffer …", fs=10, subfs=7.5)
d.arrow((490, 86), (540, 86))
d.box(540, 46, 200, 78, "504 names", "sampled, seed 42", fs=10, subfs=7.5, ec=BRAND)
d.arrow((740, 86), (790, 86))
d.box(790, 46, 190, 78, "42 blocks", "12 names each", fs=10, subfs=7.5)
d.text(20, 168, "Every name appears in exactly one block, so blocks partition the pool with no overlap.", fs=9.5, color=ASH)
d.text(20, 196, "Each block is one experimental unit — paired differences are taken within a block.", fs=9.5, color=ASH)
d.box(20, 232, 960, 148, None)
d.text(44, 262, "One block, as the model sees it", fs=10, color=WHITE)
d.text(44, 294, "6 names in the required notation", fs=9, color=ASH, mono=False)
d.text(400, 294, "decodeNode  fetchToken  renderBuffer …", fs=8.5, color=MUTE, mono=True)
d.text(44, 322, "6 names in the opposite notation", fs=9, color=ASH)
d.text(400, 322, "split_config  filter_session  format_request …", fs=8.5, color=BRAND, mono=True)
d.text(44, 356, "The 6/6 split is the balanced case; step 1 sweeps this ratio from 0 to 12.", fs=9, color=MUTE)
d.save("p_data")

# 지표
d = Dia(1000, 380)
d.eyebrow(20, 24, "one score, one rate, one difference")
d.box(20, 46, 310, 150, None)
d.text(44, 76, "Preference score  S", fs=10.5, color=WHITE)
d.text(44, 106, "Log-probability of the required", fs=9, color=ASH)
d.text(44, 126, "notation minus the opposite one.", fs=9, color=ASH)
d.text(44, 156, "Unit: nat. Sign tells direction,", fs=9, color=MUTE)
d.text(44, 176, "magnitude tells confidence.", fs=9, color=MUTE)
d.box(345, 46, 310, 150, None, ec=BRAND)
d.text(369, 76, "Transfer rate  R", fs=10.5, color=WHITE)
d.text(369, 106, "How far the intervention moved", fs=9, color=ASH)
d.text(369, 126, "S from baseline toward clean.", fs=9, color=ASH)
d.text(369, 156, "0 = nothing moved, 1 = fully", fs=9, color=MUTE)
d.text(369, 176, "restored. Not capped above 1.", fs=9, color=MUTE)
d.box(670, 46, 310, 150, None)
d.text(694, 76, "Net effect  Δ", fs=10.5, color=WHITE)
d.text(694, 106, "Treatment minus control, paired", fs=9, color=ASH)
d.text(694, 126, "block by block, then averaged.", fs=9, color=ASH)
d.text(694, 156, "This is the number we report —", fs=9, color=MUTE)
d.text(694, 176, "never the raw transfer rate.", fs=9, color=MUTE)
d.line(20, 232, 980, 232, color=HAIR)
d.text(20, 264, "Three reference points define R", fs=10, color=WHITE)
d.text(20, 296, "S_clean", fs=9.5, color=ASH, mono=True); d.text(140, 296, "context fully compliant — the ceiling", fs=9, color=MUTE)
d.text(20, 322, "S_base", fs=9.5, color=ASH, mono=True); d.text(140, 322, "context violating, nothing touched — the floor", fs=9, color=MUTE)
d.text(20, 348, "S_int", fs=9.5, color=BRAND, mono=True); d.text(140, 348, "context violating, cache overwritten — what we measure", fs=9, color=MUTE)
d.save("p_metrics")

# 통제·판정불가
d = Dia(1000, 400)
d.eyebrow(20, 24, "raw = signal + disturbance")
bars = [("raw (treatment)", 1.00, "0.54", "#8a8a8a"), ("bubble (control)", 0.49, "0.26", MUTE),
        ("net (reported)", 0.51, "0.28", BRAND)]
for i, (lab, frac, val, col) in enumerate(bars):
    y = 56 + i * 62
    d.text(20, y + 18, lab, fs=9.5, color=WHITE if i == 2 else ASH)
    d.cell(260, y, 560 * frac, 34, col)
    d.text(830, y + 18, val, fs=9.5, color=col if i == 2 else ASH, mono=True)
d.text(260, 266, "Qwen, layer 25 — overwriting anything at all already moves the score.", fs=9, color=MUTE)
d.text(20, 266, "example", fs=8.5, color=MUTE)
d.line(20, 296, 980, 296, color=HAIR)
d.text(20, 326, "Undecidable conditions are set aside, not silently averaged in", fs=10, color=WHITE)
d.text(20, 358, "If the clean and baseline scores are within 1.0 nat of each other, the denominator of R is", fs=9, color=ASH)
d.text(20, 380, "too small to divide by. Those conditions are flagged in the result file and excluded.", fs=9, color=ASH)
d.save("p_control")

# 실행 순서 vs 서술 순서
d = Dia(1000, 340)
d.eyebrow(20, 24, "order we ran  (cheapest GPU first)")
order_run = ["step 2", "step 4", "step 1", "step 3", "step 5", "diag", "step 6"]
x = 20
for i, s in enumerate(order_run):
    d.box(x, 46, 116, 46, s, fs=9.5)
    if i: d.arrow((x - 14, 69), (x - 2, 69))
    x += 130
d.eyebrow(20, 150, "order we present  (the argument)")
order_tell = ["step 1", "step 2", "step 3", "step 4", "diag", "step 6"]
x = 20
for i, s in enumerate(order_tell):
    d.box(x, 172, 116, 46, s, fs=9.5, ec=BRAND if i == 5 else HAIR)
    if i: d.arrow((x - 14, 195), (x - 2, 195))
    x += 130
d.text(802, 195, "step 5 pending", fs=9, color=MUTE, style="italic")
d.line(20, 254, 980, 254, color=HAIR)
d.text(20, 286, "The two differ on purpose. Running order was set by memory cost; telling order follows the", fs=9.5, color=ASH)
d.text(20, 312, "argument — observe, then intervene, then prescribe. No result was reordered after the fact.", fs=9.5, color=ASH)
d.save("p_order")
print("plan ok")
