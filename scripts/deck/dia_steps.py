"""스텝별 도해."""
from kit import *
import json
S = json.load(open("stats.json"))

# step1 격자
d = Dia(1000, 380)
d.eyebrow(20, 24, "step 1 condition grid — 2,184 generations")
cols = list(range(13))
cw, x0, y0 = 62, 130, 60
d.text(20, y0 + 16, "camel", fs=9, color=ASH); d.text(20, y0 + 62, "snake", fs=9, color=ASH)
for j, c in enumerate(cols):
    d.text(x0 + j * cw + cw / 2 - 4, 44, str(c), fs=8, color=MUTE, ha="center", mono=True)
    for i in range(2):
        v = S["step1"].get(f"qwen|{'camel' if i==0 else 'snake'}|{c}")
        g = 0.12 + 0.6 * (1 - (v if v is not None else 0))
        d.cell(x0 + j * cw, y0 + i * 46, cw - 4, 40, str(round(g, 2)), HAIR)
d.text(x0, 44 - 22, "number of violating names in the preceding context  (0 → 12)", fs=8.5, color=MUTE)
d.text(x0, y0 + 118, "darker = lower compliance.  Qwen shown; StableCode has its own grid.", fs=8.5, color=MUTE)
d.line(20, 176, 980, 176, color=HAIR)
rows = [("preceding context", "0–12 violating names out of 12", "13"),
        ("instruction direction", "camelCase required / snake_case required", "2"),
        ("name blocks", "42 disjoint blocks of 12 names", "42"),
        ("models", "Qwen2.5-Coder-3B, StableCode-3B", "2")]
for i, (a, b, n) in enumerate(rows):
    y = 214 + i * 34
    d.text(20, y, a, fs=9.5, color=WHITE)
    d.text(300, y, b, fs=9, color=ASH)
    d.text(950, y, n, fs=9.5, color=BRAND if i == 3 else ASH, mono=True, ha="right")
d.line(20, 352, 980, 352, color=HAIR)
d.text(20, 372, "13 × 2 × 42 × 2 = 2,184 conditions, one generation each, seed 42.", fs=9, color=MUTE)
d.save("s1_grid")

# step1 파이프라인
d = Dia(1000, 280)
steps = [("build prompt", "instruction + 12 names"), ("generate", "greedy, one function"),
         ("extract name", "parse def <name>("), ("classify", "camel / snake / neither"),
         ("score", "compliance = 1 or 0")]
x = 20
for i, (t, s) in enumerate(steps):
    w = 178
    d.box(x, 46, w, 78, t, s, fs=9.5, subfs=7.5, ec=BRAND if i == 3 else HAIR)
    if i: d.arrow((x - 14, 85), (x - 2, 85))
    x += w + 14
d.line(20, 164, 980, 164, color=HAIR)
d.text(20, 196, "No teacher forcing here — this step measures behaviour, not preference.", fs=9.5, color=ASH)
d.text(20, 224, "A name that is neither camel nor snake counts as non-compliant and is kept in the record,", fs=9.5, color=ASH)
d.text(20, 250, "so the denominator never silently shrinks.", fs=9.5, color=ASH)
d.save("s1_pipe")

# step2 프롬프트 해부
d = Dia(1000, 420)
d.eyebrow(20, 24, "the prompt, and the five spans read from it")
d.box(20, 44, 960, 150, None)
d.text(44, 74, "system", fs=8, color=MUTE, mono=True)
d.text(110, 74, "You are helping extend an existing Python module.", fs=9, color=ASH, mono=True)
d.text(110, 96, "In this project we generally write function names in ", fs=9, color=ASH, mono=True)
d.text(600, 96, "camelCase", fs=9, color=BRAND, mono=True)
d.text(694, 96, ".", fs=9, color=ASH, mono=True)
d.text(110, 118, "Every function name uses one of two styles only: camelCase or snake_case.", fs=9, color=ASH, mono=True)
d.text(44, 150, "user", fs=8, color=MUTE, mono=True)
d.text(110, 150, "def decodeNode(value): …   def split_config(value): …   × 12", fs=9, color=ASH, mono=True)
d.text(110, 174, "Add one more function.", fs=9, color=ASH, mono=True)
spans = [("code_camel", "the 6 compliant name tokens", BRAND),
         ("code_snake", "the 6 violating name tokens", ASH),
         ("instruction", "the whole system message", ASH),
         ("instr_target_word", "the required notation word", ASH),
         ("instr_viol_word", "the opposite notation word", ASH)]
d.line(20, 226, 980, 226, color=HAIR)
for i, (n, s, c) in enumerate(spans):
    y = 258 + i * 30
    d.text(20, y, n, fs=9, color=c, mono=True)
    d.text(280, y, s, fs=9, color=MUTE)
d.text(560, 258, "Only name characters are taken —", fs=9, color=MUTE)
d.text(560, 288, "not def, not parentheses, not the body.", fs=9, color=MUTE)
d.text(560, 330, "Token counts differ per span and per", fs=9, color=MUTE)
d.text(560, 360, "model, so every value is per token.", fs=9, color=MUTE)
d.text(20, 404, "Direction is fixed to camelCase in this step — that limit is what step 4 removes.", fs=9, color=BRAND)
d.save("s2_prompt")

# step2 추정량 두 가지
d = Dia(1000, 360)
d.eyebrow(20, 24, "two honest ways to summarise a span — they answer different questions")
d.box(20, 46, 460, 150, None)
d.text(44, 78, "Span sum", fs=10.5, color=WHITE)
d.text(44, 110, "How much of the attention row this", fs=9, color=ASH)
d.text(44, 130, "span received in total.", fs=9, color=ASH)
d.text(44, 162, "Longer spans win by being longer.", fs=9, color=MUTE)
d.box(520, 46, 460, 150, None, ec=BRAND)
d.text(544, 78, "Per token  (used here)", fs=10.5, color=WHITE)
d.text(544, 110, "Sum divided by the span's own token", fs=9, color=ASH)
d.text(544, 130, "count — attention per unit of text.", fs=9, color=ASH)
d.text(544, 162, "Comparable across spans and models.", fs=9, color=MUTE)
d.line(20, 232, 980, 232, color=HAIR)
d.text(20, 262, "Neither is “the fair one”. We report per token and state where the two disagree:", fs=9.5, color=ASH)
d.text(20, 292, "in step 2 the peak layer is unchanged under both, and sign flips occur in 0 of 36 layers", fs=9.5, color=ASH)
d.text(20, 314, "for Qwen and 0 of 28 for Llama, but 7 of 32 and 9 of 32 for the other two.", fs=9.5, color=ASH)
d.text(20, 348, "In step 4 the two estimands straddle 1.0 — that one is reported explicitly.", fs=9.5, color=BRAND)
d.save("s2_estimand")

# step3 공여 3종
d = Dia(1000, 400)
d.eyebrow(20, 24, "three donors, one difference")
rows = [("compliant", "the same name, written correctly", "formatMatrix", "treatment (raw)", MUTE),
        ("unrelated_camel", "a different module, all camel", "buildPayload", "treatment (paired)", BRAND),
        ("unrelated_snake", "the same module, all snake", "resolve_packet", "control", MUTE)]
for i, (n, s, ex, role, c) in enumerate(rows):
    y = 52 + i * 78
    d.box(20, y, 960, 62, None, ec=BRAND if i == 1 else HAIR)
    d.text(44, y + 31, n, fs=10, color=WHITE, mono=True)
    d.text(280, y + 31, s, fs=9, color=ASH)
    d.text(620, y + 31, ex, fs=9, color=c, mono=True)
    d.text(800, y + 31, role, fs=9, color=BRAND if i == 1 else MUTE)
d.line(20, 306, 980, 306, color=HAIR)
d.text(20, 336, "The reported estimand is unrelated_camel minus unrelated_snake.", fs=10, color=WHITE)
d.text(20, 366, "Both donors are six functions from a separate module in a single notation, so the two", fs=9, color=ASH)
d.text(20, 388, "conditions differ in notation and nothing else. Context length cancels exactly.", fs=9, color=ASH)
d.save("s3_donors")

# step3 무엇을 덮나
d = Dia(1000, 360)
d.eyebrow(20, 24, "what is overwritten, and what is left alone")
d.box(20, 46, 470, 130, None, ec=BRAND)
d.text(44, 78, "Touched", fs=10.5, color=WHITE)
d.text(44, 110, "The cached K and/or V at the token", fs=9, color=ASH)
d.text(44, 130, "positions of the six violating names,", fs=9, color=ASH)
d.text(44, 150, "at one layer, in one KV group.", fs=9, color=ASH)
d.box(510, 46, 470, 130, None)
d.text(534, 78, "Untouched", fs=10.5, color=WHITE)
d.text(534, 110, "The prompt text, the instruction, the", fs=9, color=ASH)
d.text(534, 130, "compliant names, every other layer,", fs=9, color=ASH)
d.text(534, 150, "and the sequence length.", fs=9, color=ASH)
d.line(20, 212, 980, 212, color=HAIR)
d.text(20, 242, "Three intervention kinds are measured on the same forward pass", fs=10, color=WHITE)
for i, (k, s) in enumerate([("key", "attention road only — how much the model looks"),
                            ("value", "content road only — what arrives when it looks"),
                            ("key_value", "both roads together")]):
    y = 276 + i * 28
    d.text(20, y, k, fs=9.5, color=BRAND if k == "value" else ASH, mono=True)
    d.text(200, y, s, fs=9, color=MUTE)
d.save("s3_what")

# step4 역할 분리
d = Dia(1000, 420)
d.eyebrow(20, 24, "the required word appears twice, in two different roles")
d.box(20, 46, 960, 108, None)
d.text(44, 78, "rule sentence", fs=8, color=MUTE)
d.text(180, 78, "In this project we generally write function names in ", fs=9, color=ASH, mono=True)
d.text(651, 78, "camelCase", fs=9, color=BRAND, mono=True)
d.text(745, 78, ".", fs=9, color=ASH, mono=True)
d.text(44, 122, "candidate list", fs=8, color=MUTE)
d.text(180, 122, "Every function name uses one of two styles only: ", fs=9, color=ASH, mono=True)
d.text(618, 122, "camelCase", fs=9, color=ASH, mono=True)
d.text(712, 122, " or ", fs=9, color=ASH, mono=True)
d.text(748, 122, "snake_case", fs=9, color=ASH, mono=True)
d.text(842, 122, ".", fs=9, color=ASH, mono=True)
d.line(20, 186, 980, 186, color=HAIR)
d.text(20, 216, "Summing both occurrences builds a 2 : 1 advantage into the required word by construction.", fs=9.5, color=ASH)
d.text(20, 242, "That inflated key is kept in the results for transparency but is never interpreted.", fs=9.5, color=ASH)
keys = [("instr_rule_word", "1 occurrence", "the actual directive — interpreted", BRAND),
        ("instr_cand_camel", "1 occurrence", "symmetric control, same role", ASH),
        ("instr_cand_snake", "1 occurrence", "symmetric control, same role", ASH),
        ("instr_target_word", "2 occurrences", "initial key — structurally inflated", MUTE),
        ("instr_viol_word", "1 occurrence", "initial key — not comparable to the above", MUTE)]
for i, (n, occ, s, c) in enumerate(keys):
    y = 286 + i * 28
    d.text(20, y, n, fs=9, color=c, mono=True)
    d.text(300, y, occ, fs=8.5, color=MUTE, mono=True)
    d.text(440, y, s, fs=9, color=ASH if c is BRAND else MUTE)
d.save("s4_split")

# diag 설계
d = Dia(1000, 400)
d.eyebrow(20, 24, "the premise both causal steps rest on")
d.box(20, 46, 960, 92, None, ec=BRAND)
d.text(44, 78, "If mean-pool damages Keys more than Values, then “Key does nothing” could be an", fs=9.5, color=WHITE)
d.text(44, 104, "artefact of the instrument rather than a fact about the model.", fs=9.5, color=WHITE)
d.line(20, 172, 980, 172, color=HAIR)
d.text(20, 202, "What the diagnostic measures", fs=10, color=WHITE)
d.text(20, 234, "— Norm shrink: how much shorter the substituted vector is than the original, K versus V.", fs=9, color=ASH)
d.text(20, 262, "— Piece cosine: how well the averaged pieces still point the original way.", fs=9, color=ASH)
d.text(20, 290, "— Position offset: how far the donor's positions sit from the target's.", fs=9, color=ASH)
d.text(20, 330, "What it does not measure", fs=10, color=WHITE)
d.text(20, 362, "Rotational phase. The norm-collapse hypothesis is rejected; a phase-mismatch account is not.", fs=9, color=BRAND)
d.text(20, 388, "That gap is stated wherever the Key result is used, not only here.", fs=9, color=ASH)
d.save("d_design")

# step6 두 처방
d = Dia(1000, 420)
d.eyebrow(20, 24, "same instrument, two prescriptions, opposite predictions")
d.box(20, 46, 470, 168, None, ec=BRAND)
d.text(44, 78, "Residual-stream steering", fs=10.5, color=WHITE)
d.text(44, 110, "Add a contrast direction to the block", fs=9, color=ASH)
d.text(44, 130, "output at one late layer.", fs=9, color=ASH)
d.text(44, 162, "Direction is built from the difference", fs=9, color=MUTE)
d.text(44, 182, "between camel and snake contexts.", fs=9, color=MUTE)
d.text(44, 206, "Predicted by RQ1–RQ3 to work.", fs=9, color=BRAND)
d.box(510, 46, 470, 168, None)
d.text(534, 78, "Spotlight  (attention reweighting)", fs=10.5, color=WHITE)
d.text(534, 110, "Raise the attention mass on the", fs=9, color=ASH)
d.text(534, 130, "instruction span toward a target.", fs=9, color=ASH)
d.text(534, 162, "Undershoot form, applied at every", fs=9, color=MUTE)
d.text(534, 182, "attention call.", fs=9, color=MUTE)
d.text(534, 206, "Predicted by RQ1–RQ3 to fail.", fs=9, color=MUTE)
d.line(20, 250, 980, 250, color=HAIR)
d.text(20, 280, "Two checks, not one", fs=10, color=WHITE)
d.text(20, 312, "— Does the preference score recover? (steer)", fs=9, color=ASH)
d.text(20, 340, "— Does the model still emit a usable function name, or did we break it? (steer_generate)", fs=9, color=ASH)
d.text(20, 380, "A score that rises while the names fall apart is a failed prescription, so both are reported", fs=9, color=MUTE)
d.text(20, 402, "together and neither is shown alone.", fs=9, color=MUTE)
d.save("s6_two")

# 종합: 관통선
d = Dia(1680, 320)
items = [("RQ1", "violating context\ncollapses compliance", "step 1"),
         ("RQ2a", "the model looks more\nat violations, not less", "step 2"),
         ("RQ2b", "the causal road is\nValue, not Key", "step 3"),
         ("RQ3a", "the instruction is read\nper token, competitively", "step 4"),
         ("Rx", "pushing value restores;\nraising attention does not", "step 6")]
w = 310
for i, (n, s, src) in enumerate(items):
    x = i * (w + 32)
    d.box(x, 40, w, 170, None, ec=BRAND if i == 4 else HAIR)
    d.text(x + 22, 72, n, fs=11, color=BRAND if i == 4 else WHITE, mono=True)
    for k, line in enumerate(s.split("\n")):
        d.text(x + 22, 110 + k * 22, line, fs=9.5, color=ASH)
    d.text(x + 22, 182, src, fs=8.5, color=MUTE, mono=True)
    if i: d.arrow((x - 32, 125), (x - 4, 125))
d.text(0, 258, "Every arrow is a prediction that the next step tested — none of them was fitted after the fact.", fs=9.5, color=ASH)
d.text(0, 288, "The one link still missing is instruction causality with a full-layer control. That is step 5.", fs=9.5, color=BRAND)
d.save("z_line")

# 다음 할 일
d = Dia(1000, 380)
d.eyebrow(20, 24, "what is pending, and why it is pending rather than wrong")
d.box(20, 46, 470, 140, None)
d.text(44, 78, "What exists", fs=10.5, color=WHITE)
d.text(44, 110, "Treatment swept across all layers.", fs=9, color=ASH)
d.text(44, 132, "Controls measured at one layer —", fs=9, color=ASH)
d.text(44, 154, "the treatment's raw peak.", fs=9, color=ASH)
d.box(510, 46, 470, 140, None, ec=BRAND)
d.text(534, 78, "What is missing", fs=10.5, color=WHITE)
d.text(534, 110, "A layer-wise net-effect curve, because", fs=9, color=ASH)
d.text(534, 132, "subtraction is only possible where a", fs=9, color=ASH)
d.text(534, 154, "control exists.", fs=9, color=ASH)
d.line(20, 222, 980, 222, color=HAIR)
d.text(20, 252, "Missing is not wrong. No reported step-5 number depends on the absent layers, and the", fs=9.5, color=ASH)
d.text(20, 278, "single control layer is the treatment's own peak in all four models.", fs=9.5, color=ASH)
d.text(20, 314, "In step 3 — the only step with controls at every layer — the raw peak and the net peak", fs=9.5, color=ASH)
d.text(20, 340, "coincide in all four models. Step 5 assumes the same; the sweep will check it.", fs=9.5, color=ASH)
d.text(20, 374, "Cost: 672 sweeps, roughly two GPU hours.", fs=9.5, color=BRAND)
d.save("z_next")
print("steps ok")
