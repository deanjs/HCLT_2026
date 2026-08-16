# -*- coding: utf-8 -*-
"""HCLT 2026 실험 정리 덱 — 91장. step5는 전층 스윕이 끝난 뒤 붙인다."""
import json
from deck import *
from lay import *

S = json.load(open("stats.json"))
D = Deck("Naming-convention probing")

C0, C1 = "00 · Overview", "01 · Plan"
C2, C3 = "02 · Concepts", "03 · Audit"
C4, C5 = "04 · Harness", "05 · Experiments"
C6 = "06 · Synthesis"

# ═══════════════════════════════════════════════════ 00 · Overview ═══════
s = D.slide(C0, "Why code models break naming conventions",
            "A probing study on four code-generation models. Unified re-run, seed 42, "
            "504 function names, 7,212 analysed conditions.")
x, y, w, h = D.panel(s, ML, BODY_TOP, 1120, 330, "the claim, in one sentence")
D._tb(s, x, y + 10, w, 200,
      ["Models do look at the instruction.", "What arrives when they look is wrong."],
      28, WHITE, spacing=-1.2, line=1.30)
D._tb(s, x, y + 190, w, 60,
      "So the handle is not attention. It is content — the Value road — at one late layer.",
      11.5, ASH)
x, y, w, h = D.panel(s, ML + 1144, BODY_TOP, MR - ML - 1144, 330, "scale of the study")
for i, (v, lab) in enumerate([("4", "models"), ("504", "function names"),
                              ("7,212", "analysed conditions")]):
    D.stat(s, x, y + i * 88, w, v, lab, size=34)
y2 = BODY_TOP + 330 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "three contributions",
                     "Each is a measurement, not an argument — every number in this deck comes from a stored result file.")
cw = (w - 2 * GAP) / 3
for i, (t, b) in enumerate([
        ("Separating the two roads", "Attention share and attended content are intervened on independently, "
         "so “does not look” and “reads wrong” become distinguishable claims."),
        ("Pricing the disturbance", "Overwriting anything at all moves the score. We measure that bubble and "
         "report treatment minus control, never the raw number."),
        ("Testing the prescription", "The remedy implied by the diagnosis is measured on the same instrument "
         "as the rival remedy, on the same conditions.")]):
    xx = x + i * (cw + GAP)
    D.cardtitle(s, xx, y, t, w=cw)
    D.body(s, xx, y + 32, cw, b, fit_h=h - 40, tag=f"contrib{i}")

s = D.slide(C0, "How this deck is arranged",
            "Six sections. The experiments are told in the order of the argument, not the order they were run.")
x, y, w, h = D.panel(s, ML, BODY_TOP, MR - ML, 400, "section map",
                     "Step 5 (instruction causality) is held back — its full-layer control sweep has not been run yet.")
D.fit(s, A + "p_map.png", x, y, w, h, align="left")
y2 = BODY_TOP + 400 + GAP
x, y, w, h = D.panel(s, ML, y2, 840, BODY_BOT - y2, "how to read a step")
kv_rows(D, s, x, y, w, [
    ("Plan", "the question, the conditions, what each outcome would mean"),
    ("Code", "the pipeline, the lines that matter, the stored fields"),
    ("Results", "the figures, the numbers, and what the step does not settle")], pitch=36)
x, y, w, h = D.panel(s, ML + 864, y2, MR - ML - 864, BODY_BOT - y2, "conventions used throughout")
kv_rows(D, s, x, y, w, [
    ("Clay", "marks the one series being argued about"),
    ("Per token", "attention is always divided by the span's token count"),
    ("Net", "reported effects are treatment minus control, paired by block")], pitch=36)

s = D.slide(C0, "The through-line",
            "Five findings, each one a prediction that the next experiment tested.")
x, y, w, h = D.panel(s, ML, BODY_TOP, MR - ML, 380, "from symptom to handle",
                     "No arrow was drawn after the fact — each step was specified before the previous one's results were read.")
D.fit(s, A + "z_line.png", x, y, w, h, align="left")
y2 = BODY_TOP + 380 + GAP
cw = (MR - ML - 2 * GAP) / 3
for i, (eb, t, b) in enumerate([
        ("what fails", "Compliance collapses with context",
         "A dozen violating names in the preceding file is enough to drive compliance to the floor, "
         "in both notation directions."),
        ("where it lives", "The road is Value, not Key",
         "Overwriting attended content moves behaviour; overwriting attention share does not, "
         "once the disturbance is subtracted."),
        ("what to do", "Push content, not attention",
         "Adding a contrast direction at one late layer restores preference; raising attention "
         "mass on the instruction does not.")]):
    xx = ML + i * (cw + GAP)
    x, y, w, h = D.panel(s, xx, y2, cw, BODY_BOT - y2, eb)
    D.cardtitle(s, x, y, t, w=w)
    D.body(s, x, y + 34, w, b, fit_h=h - 42, tag=f"tl{i}")

# ═══════════════════════════════════════════════════ 01 · Plan ═══════════
s = D.slide(C1, "The symptom: the file overrides the instruction",
            "The instruction is explicit and unchanged. Only the surrounding file changes.")
lx, lw, rx, rw = split(1180)
x, y, w, h = D.panel(s, lx, BODY_TOP, lw, 470, "step 1 · compliance against context violations",
                     "Both instruction directions are plotted. The collapse is not an artefact of one notation.")
D.plot(s, f"{FIG}/step1/figures/cliff.png", x, y, w, h)
x, y, w, h = D.panel(s, rx, BODY_TOP, rw, 470, "what the plot says")
D.deflist(s, x, y, w, [
    ("Axis", "how many of the twelve preceding function names violate the instruction"),
    ("Value", "share of generations that follow the instruction"),
    ("Shape", "not a gentle slope — compliance falls away over a narrow band"),
    ("Both lines", "camelCase-required and snake_case-required behave the same way"),
], fit_h=h, tag="cliff-read")
y2 = BODY_TOP + 470 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "why this is the right starting point",
                     "If context did not matter, there would be nothing to localise and no prescription to test.")
D.body(s, x, y, w,
       "The instruction text is identical in every condition on this plot. The only thing that moves is how "
       "many names in the file already break it. That isolates context as the cause and makes the rest of the "
       "study a question about mechanism rather than about prompt wording.",
       fit_h=h, tag="cliff-why")

s = D.slide(C1, "Two accounts of the same failure",
            "They disagree about the locus, so they disagree about the remedy — and a single instrument can separate them.")
x, y, w, h = D.panel(s, ML, BODY_TOP, MR - ML, 420, "the disagreement",
                     "Prior work that raises attention mass on the instruction is committing to the left-hand account.")
D.fit(s, A + "p_claim.png", x, y, w, h, align="left")
y2 = BODY_TOP + 420 + GAP
x, y, w, h = D.panel(s, ML, y2, 900, BODY_BOT - y2, "what would settle it")
D.deflist(s, x, y, w, [
    ("If attention", "raising the instruction's attention share should restore compliance"),
    ("If content", "it should not — and pushing the attended content should"),
], fit_h=h, tag="settle")
x, y, w, h = D.panel(s, ML + 924, y2, MR - ML - 924, BODY_BOT - y2, "how we avoid stacking the deck")
D.body(s, x, y, w,
       "Both remedies are swept over a fixed grid chosen before any result was read, on the same conditions, "
       "the same seed and the same four models. Neither is tuned after seeing the other's outcome.",
       fit_h=h, tag="fair")

s = D.slide(C1, "Research questions",
            "Three questions and one prescription. Step 6 is not a fourth question — it is the test of whether the first three were read correctly.")
x, y, w, h = D.panel(s, ML, BODY_TOP, MR - ML, 340, "structure",
                     "Observation and causation are kept apart: every “where” claim has a matching intervention.")
D.fit(s, A + "p_rq.png", x, y, w, h, align="left")
y2 = BODY_TOP + 340 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "what each answer would license",
                     "Stated before the runs, so that an unexpected result cannot be reinterpreted into a success.")
table(D, s, x, y, w,
      ["question", "if yes", "if no"],
      [["RQ1  context lowers compliance",
        "there is a mechanism to localise",
        "the study has no subject; stop here"],
       ["RQ2  the code signal is content",
        "the handle is the Value road at a specific layer",
        "attention-based remedies remain live"],
       ["RQ3  the instruction travels the same road",
        "one prescription covers both signals",
        "code and instruction need separate treatment"]],
      [660, 520, 540], size=10.5, fit_h=h, tag="rq")

s = D.slide(C1, "Fixed setup",
            "One pool of names, four models, one seed. Every step draws its conditions from the same material.")
x, y, w, h = D.panel(s, ML, BODY_TOP, 1080, 380, "the name pool",
                     "Blocks are disjoint, so a paired difference within a block is a difference on the same names.")
D.fit(s, A + "p_data.png", x, y, w, h, align="left")
x, y, w, h = D.panel(s, ML + 1104, BODY_TOP, MR - ML - 1104, 380, "the four models")
kv_rows(D, s, x, y, w, [
    ("Qwen2.5-Coder-3B", "code-specialised, 36 layers"),
    ("DeepSeek-Coder-6.7B", "code-specialised, 32 layers"),
    ("Llama-3.2-3B", "general purpose, 28 layers"),
    ("StableCode-3B", "code-specialised, 32 layers")], pitch=46)
D._tb(s, x, y + 200, w, 60,
      "The general-purpose model is kept in deliberately: it is the one that most often lands in the "
      "undecidable band, and that is itself a result.", 10, MUTE)
y2 = BODY_TOP + 380 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "held constant across every step",
                     "Python only. The harness supports JavaScript but no JavaScript condition was ever run, so no claim covers it.")
kv_rows(D, s, x, y, w // 2 - 20, [
    ("Seed", "42, everywhere"),
    ("Instruction form", "positive phrasing, weak strength"),
    ("Substitution", "mean-pool, per KV group")], pitch=34)
kv_rows(D, s, x + w // 2, y, w // 2, [
    ("Language", "Python only — 0 JavaScript results"),
    ("Context", "12 preceding functions"),
    ("Undecidable", "gap below 1.0 nat is excluded")], pitch=34)

s = D.slide(C1, "What we measure",
            "One score for preference, one rate for how far an intervention moved it, one difference for what survives the control.")
x, y, w, h = D.panel(s, ML, BODY_TOP, MR - ML, 360, "the three quantities",
                     "The transfer rate is not capped above 1 — values above 1 mean the intervention overshot the clean condition.")
D.fit(s, A + "p_metrics.png", x, y, w, h, align="left")
y2 = BODY_TOP + 360 + GAP
hh = BODY_BOT - y2
x, y, w, h = D.panel(s, ML, y2, 840, hh, "eq. 2 — preference score",
                     "Teacher-forced with the prefix “def ”, no length normalisation.")
D.equation(s, A + "eq_score.png", x, y + 4, h=52)
D.symbols(s, x, y + 76, w, [("P", "next-token probability under the model")])
x, y, w, h = D.panel(s, ML + 864, y2, MR - ML - 864, hh, "eq. 3 — transfer rate",
                     "Clean is the ceiling, base the floor, int the intervened run.")
D.equation(s, A + "eq_recovery.png", x, y + 4, h=76)

s = D.slide(C1, "Controls are not optional here",
            "Overwriting a cached vector removes the original as well as inserting the new one. The raw number contains both.")
x, y, w, h = D.panel(s, ML, BODY_TOP, 1120, 400, "the decomposition",
                     "Bubble measured at the peak layer of each model: 34% (StableCode) to 49% (Qwen) of the raw value.")
D.fit(s, A + "p_control.png", x, y, w, h, align="left")
x, y, w, h = D.panel(s, ML + 1144, BODY_TOP, MR - ML - 1144, 400, "bubble share by model",
                     "Raw transfer rates are never reported alone.")
table(D, s, x, y, w, ["model", "bubble ÷ raw"],
      [[n, f"{round(S['step3'][f]['bubble_pct']*100)}%"] for f, n in MODELS],
      [340, 180], size=10, fit_h=h, tag="bubble")
y2 = BODY_TOP + 400 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "eq. 4 — when a condition cannot be judged",
                     "The flag and the gap are both stored, so the threshold's sensitivity can be re-checked without re-running anything.")
D.equation(s, A + "eq_undec.png", x, y + 2, h=48)
D.symbols(s, x, y + 66, w, [("S_clean", "compliant context"), ("S_base", "violating context"),
                            ("1.0", "threshold in nats, stored with every result")])
D.body(s, x, y + 96, w, "Below the threshold the model had no strong preference to restore, so the transfer "
       "rate becomes unstable. Those conditions are counted, reported and set aside.",
       fit_h=h - 100, tag="undec-body")

s = D.slide(C1, "The order of work",
            "Running order was chosen by memory cost. Telling order follows the argument. Neither was changed after results appeared.")
x, y, w, h = D.panel(s, ML, BODY_TOP, MR - ML, 340, "two orders",
                     "Step 5 sits in the running order but not yet in this deck — its control sweep is still outstanding.")
D.fit(s, A + "p_order.png", x, y, w, h, align="left")
y2 = BODY_TOP + 340 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "condition census — what exists on disk",
                     "7,212 conditions enter the analysis; 1,344 step-5 control runs are excluded and footnoted.")
cens = [("step 1 · cliff", 2184), ("step 2 · code observe", 168), ("step 3 · code cause", 504),
        ("step 4 · instruction observe", 336), ("diag · instrument", 72),
        ("step 6 · steer", 2184), ("step 6 · generate", 420), ("step 6 · cross-layer", 336)]
cw = (w - 3 * GAP) / 4
rowh = (h - GAP) / 2
for i, (n, v) in enumerate(cens):
    xx = x + (i % 4) * (cw + GAP); yy = y + (i // 4) * (rowh + GAP)
    D.stat(s, xx, yy, cw, f"{v:,}", n, size=24, fit_h=rowh, tag=f"cens{i}")

# ═══════════════════════════════════════════════════ 02 · Concepts ═══════
s = D.slide(C2, "Inside one layer: two separable roads",
            "Attention splits into how much to look (Key) and what arrives when it looks (Value). Every intervention in this study cuts along that seam.")
H1 = 440; Y2 = BODY_TOP + H1 + GAP; H2 = BODY_BOT - Y2
lx, lw, rx, rw = split(1060)
x, y, w, h = D.panel(s, lx, BODY_TOP, lw, H1, "transformer block · one layer",
                     "The Value box is where this study pushes; the Key box is where prior work pushes.")
D.fit(s, A + "c_block.png", x, y, w, h, align="left")
x, y, w, h = D.panel(s, rx, BODY_TOP, rw, H1, "what each road carries",
                     "We read one query row; it sums to 1, so each span's value is its share.")
D.deflist(s, x, y, w, [
    ("Query", "the row of the position about to emit the function name"),
    ("Key", "decides the share of attention each earlier token receives"),
    ("Value", "the content actually mixed into the output"),
    ("Residual", "the running vector each layer adds to, never replaces")],
    pitch=48, fit_h=h, tag="roads")
x, y, w, h = D.panel(s, lx, Y2, lw, H2, "eq. 5 — the attention output at one position",
                     "a is the Key road, v is the Value road. Overwriting v alone leaves a untouched.")
D.equation(s, A + "eq_output.png", x, y + 6, h=64)
D.symbols(s, x, y + 86, w, [("i", "query position"), ("j", "earlier token"),
                            ("a", "attention weight"), ("v", "value vector")])
x, y, w, h = D.panel(s, rx, Y2, rw, H2, "why the seam is testable",
                     "This is what separates “looks enough” from “reads wrong”.")
D.body(s, x, y, w, "The cached K and V are separate tensors, so one can be replaced while the "
       "other is left exactly as the model computed it — same prompt, same length.",
       fit_h=h, tag="seam")

s = D.slide(C2, "Where Q, K and V come from",
            "Three linear maps of the same hidden state. Nothing about them is intrinsically about position or content — the roles come from how they are used.")
x, y, w, h = D.panel(s, ML, BODY_TOP, MR - ML, 300, "eq. 6 — the three projections",
                     "Same input, three learned matrices. The split into “how much” and “what” happens downstream.")
D.equation(s, A + "eq_qkv.png", x, y + 20, h=58)
D.symbols(s, x, y + 106, w, [("x", "hidden state at a position"),
                             ("W", "learned projection, one per role"),
                             ("d_k", "head dimension")])
y2 = BODY_TOP + 300 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "the two roads, drawn out",
                     "The dashed line is the only place the two roads meet — weights computed on one side scale the other.")
D.fit(s, A + "c_paths.png", x, y, w, h, align="left")

s = D.slide(C2, "Softmax, and why a row sums to one",
            "The normalisation is what turns raw scores into shares — and it is what makes span-level attention interpretable at all.")
lx, lw, rx, rw = split(820)
x, y, w, h = D.panel(s, lx, BODY_TOP, lw, 400, "eq. 7 — attention weights",
                     "Exponentiate the scores, then divide by their sum.")
D.equation(s, A + "eq_softmax.png", x, y + 30, h=86)
D.symbols(s, x, y + 150, w, [("q·k", "score, before scaling"), ("√d", "temperature")])
x, y, w, h = D.panel(s, rx, BODY_TOP, rw, 400, "consequences we rely on")
D.deflist(s, x, y, w, [
    ("Shares", "a span's total is the fraction of attention it received"),
    ("Competition", "one span can only rise if another falls"),
    ("Comparability", "the same quantity is meaningful across models and layers"),
    ("Length", "a longer span accumulates more, so we divide by token count")],
    fit_h=h, tag="softmax")
y2 = BODY_TOP + 400 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "eq. 8 — attention per token",
                     "Reported everywhere in this deck. The span-sum estimand is reported alongside wherever the two disagree.")
D.equation(s, A + "eq_pertoken.png", x, y + 4, h=64)

s = D.slide(C2, "What a head is, and why we average over them",
            "Each head runs the whole attention computation on its own slice of the hidden state.")
x, y, w, h = D.panel(s, ML, BODY_TOP, MR - ML, 420, "heads within one layer",
                     "Averaging is a choice with a cost: a single specialised head would be invisible in our numbers.")
D.fit(s, A + "c_heads.png", x, y, w, h, align="left")
y2 = BODY_TOP + 420 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "the honest version of the claim",
                     "Head-level structure is left for future work; nothing here depends on its absence.")
D.body(s, x, y, w, "Our statements are about layers, not heads. When we say a layer carries the notation "
       "signal, we mean the head-averaged quantity at that layer moved — not that every head did, and not "
       "that one particular head is responsible.", fit_h=h, tag="heads-claim")

s = D.slide(C2, "The query row: the one row we read",
            "Attention is a matrix. We read a single row of it — the position that is about to write the function name.")
lx, lw, rx, rw = split(1180)
x, y, w, h = D.panel(s, lx, BODY_TOP, lw, 470, "the matrix, and the row",
                     "Masked upper triangle: a token cannot attend to tokens that come after it.")
D.fit(s, A + "c_queryrow.png", x, y, w, h, align="left")
x, y, w, h = D.panel(s, rx, BODY_TOP, rw, 470, "why this row and not an average")
D.deflist(s, x, y, w, [
    ("Decision point", "this is the position whose output becomes the first name token"),
    ("Sums to one", "so span values are shares, not arbitrary magnitudes"),
    ("Stable", "the same row is read in every condition, so conditions are comparable"),
    ("Limitation", "later name tokens are not read — the study is about the first choice")],
    fit_h=h, tag="qrow")
y2 = BODY_TOP + 470 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "what a span value means",
                     "A span's value is “how much of this decision's attention went to this stretch of text”, per token of that stretch.")
D.body(s, x, y, w, "Because the row is a probability distribution, comparing two spans is comparing two "
       "shares of one fixed budget. That is what makes the instruction and the code names directly comparable "
       "without any further normalisation across models.", fit_h=h, tag="qrow2")

s = D.slide(C2, "The KV cache: the surface we intervene on",
            "Everything the model has already read is stored as two tensors per layer. That storage is where this study operates.")
x, y, w, h = D.panel(s, ML, BODY_TOP, MR - ML, 420, "cache structure and the positions we touch",
                     "Sequence length, prompt text and token identities are unchanged by every intervention in this study.")
D.fit(s, A + "c_kvcache.png", x, y, w, h, align="left")
y2 = BODY_TOP + 420 + GAP
lx, lw, rx, rw = split(900)
x, y, w, h = D.panel(s, lx, y2, lw, BODY_BOT - y2, "what an intervention is, exactly")
D.body(s, x, y, w, "Run the prompt once, take the cache, replace selected slices of K and/or V, then continue "
       "the forward pass from that edited cache. Nothing about the text changes.", fit_h=h, tag="int-def")
x, y, w, h = D.panel(s, rx, y2, rw, BODY_BOT - y2, "what that buys")
D.body(s, x, y, w, "Any behaviour change is attributable to the replaced content alone — not to a different "
       "prompt, a different length, or a different tokenisation.", fit_h=h, tag="int-buy")

s = D.slide(C2, "Position lives in the Key, not the Value",
            "Rotary embeddings are applied to keys before caching. This is a real asymmetry between the two roads, and we do not hide it.")
x, y, w, h = D.panel(s, ML, BODY_TOP, MR - ML, 360, "where the rotation is applied",
                     "A transplanted key carries a rotation that belongs to a different position; a transplanted value does not.")
D.fit(s, A + "c_rope.png", x, y, w, h, align="left")
y2 = BODY_TOP + 360 + GAP
lx, lw, rx, rw = split(880)
x, y, w, h = D.panel(s, lx, y2, lw, BODY_BOT - y2, "eq. 9 — the asymmetry",
                     "R is the position-dependent rotation.")
D.equation(s, A + "eq_rope.png", x, y + 6, h=52)
x, y, w, h = D.panel(s, rx, y2, rw, BODY_BOT - y2, "how we handle it",
                     "The diagnostic step exists precisely because of this asymmetry.")
D.body(s, x, y, w, "We never claim Values are position-free — deep-layer values are built from context and carry "
       "positional information indirectly. We claim only that the substitution does not damage Keys more than "
       "Values by norm, and we measure that.", fit_h=h, tag="rope-h")

s = D.slide(C2, "Grouped-query attention: what one overwrite touches",
            "Query heads outnumber key/value heads on two of the four models, so a substitution is inherently a group-level edit.")
x, y, w, h = D.panel(s, ML, BODY_TOP, MR - ML, 400, "group layout per model",
                     "Read from each model's configuration at load time, never assumed.")
D.fit(s, A + "c_gqa.png", x, y, w, h, align="left")
y2 = BODY_TOP + 400 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "why this is a correctness issue, not a detail",
                     "Getting it wrong would edit eight heads on Qwen while the record claimed one.")
D.body(s, x, y, w, "Because the cache stores one K and one V per group, replacing a slice necessarily affects "
       "every query head in that group. The intervention is therefore defined at group granularity, and the "
       "group size is part of what is stored with each result.", fit_h=h, tag="gqa-w")

s = D.slide(C2, "The residual stream, and why it is not “Value steering”",
            "Each block adds to a running vector. Anything added at one layer is still present, transformed, at every later layer.")
x, y, w, h = D.panel(s, ML, BODY_TOP, MR - ML, 340, "the running vector",
                     "This is why an edit at layer 20 is not confined to layer 20 — and not confined to one road.")
D.fit(s, A + "c_residual.png", x, y, w, h, align="left")
y2 = BODY_TOP + 340 + GAP
lx, lw, rx, rw = split(860)
x, y, w, h = D.panel(s, lx, y2, lw, BODY_BOT - y2, "eq. 10 — a block's contribution")
D.equation(s, A + "eq_residual.png", x, y + 10, h=50)
D.symbols(s, x, y + 80, w, [("h", "the residual stream at layer ℓ")])
x, y, w, h = D.panel(s, rx, y2, rw, BODY_BOT - y2, "the naming discipline",
                     "The cache intervention is Value-specific. The steering prescription is not.")
D.body(s, x, y, w, "Step 6 adds a direction to the block output, so downstream layers recompute their own Q, K, "
       "V and FFN from it. Calling that “Value steering” would overstate what is being manipulated.",
       fit_h=h, tag="resid-name")

s = D.slide(C2, "Log-probability and the preference score",
            "The score is a difference of two log-probabilities under teacher forcing. Its unit is the nat, and only differences are interpreted.")
lx, lw, rx, rw = split(1000)
x, y, w, h = D.panel(s, lx, BODY_TOP, lw, 340, "eq. 11 — sequence log-probability",
                     "A sum over positions, each conditioned on everything before it.")
D.equation(s, A + "eq_logp.png", x, y + 24, h=66)
x, y, w, h = D.panel(s, rx, BODY_TOP, rw, 340, "reading a log-probability")
D.deflist(s, x, y, w, [
    ("Always negative", "probabilities are below 1, so their logs are below 0"),
    ("Closer to zero", "means the model finds the string more likely"),
    ("Additive", "so a sequence's score is the sum of its tokens' scores")],
    fit_h=h, tag="logp")
y2 = BODY_TOP + 340 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "eq. 12 — the preference score used throughout",
                     "Positive means the model prefers camelCase; negative means it prefers snake_case. Zero is genuine indifference.")
D.equation(s, A + "eq_score.png", x, y + 6, h=52)
D.body(s, x, y + 84, w, "No length normalisation is applied. The two candidate strings are the notation forms of the "
       "same name, so their token counts are close but not identical — dividing by length would introduce a "
       "different distortion rather than remove one.", fit_h=h - 90, tag="score-note")

s = D.slide(C2, "Teacher forcing and the forced prefix",
            "The score is read at one position, with the same prefix in every condition, so conditions differ only where we intend them to.")
lx, lw, rx, rw = split(940)
x, y, w, h = D.panel(s, lx, BODY_TOP, lw, 420, "the procedure")
code(D, s, x, y, w, h, [
    "# the model is not asked to generate here",
    "prompt = system + preceding_code + request",
    "»prefix = \"def \"                    # forced, identical everywhere",
    "",
    "# one forward pass; read the distribution at the next position",
    "logits = model(prompt + prefix).logits[:, -1]",
    "",
    "S = logprob(camel_form) - logprob(snake_form)",
    "#   > 0  prefers camelCase      < 0  prefers snake_case"])
x, y, w, h = D.panel(s, rx, BODY_TOP, rw, 420, "why force a prefix at all",
                     "Without it, the model could answer in prose and the comparison would never happen.")
D.deflist(s, x, y, w, [
    ("Same position", "the score is always read at the same point in the sequence"),
    ("Same context", "differences come from the manipulation, not from what the model chose to write"),
    ("No sampling", "no seed-dependent generation noise enters the measurement")],
    fit_h=h, tag="tf")
y2 = BODY_TOP + 420 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "the trade this makes",
                     "Step 1 and step 6's generation check exist to cover exactly this gap.")
D.body(s, x, y, w, "Teacher forcing measures preference, not behaviour. A model can prefer the right notation and "
       "still emit the wrong one. That is why the preference-based steps are always paired with a step that "
       "actually generates a name and inspects it.", fit_h=h, tag="tf-trade")

s = D.slide(C2, "Mean-pool substitution: what actually gets written",
            "Donor pieces are averaged into one vector, which is then written into every target slot.")
x, y, w, h = D.panel(s, ML, BODY_TOP, MR - ML, 400, "the operation",
                     "Tokenizer-independent by construction: no name is ever skipped for having the wrong number of pieces.")
D.fit(s, A + "h_meanpool.png", x, y, w, h, align="left")
y2 = BODY_TOP + 400 + GAP
lx, lw, rx, rw = split(900)
x, y, w, h = D.panel(s, lx, y2, lw, BODY_BOT - y2, "eq. 13 — the substitution")
D.equation(s, A + "eq_meanpool.png", x, y + 6, h=54)
x, y, w, h = D.panel(s, rx, y2, rw, BODY_BOT - y2, "the cost, stated up front",
                     "This is the single reason controls are mandatory in this study.")
D.body(s, x, y, w, "Mean-pooling inserts new content and destroys the original in the same stroke. The raw effect "
       "therefore contains both. Only the difference against a control isolates the part attributable to notation.",
       fit_h=h, tag="mp-cost")

s = D.slide(C2, "Donors: the vocabulary of every intervention",
            "A donor is the source of the values written in. Choosing donors well is what turns an intervention into a controlled comparison.")
x, y, w, h = D.panel(s, ML, BODY_TOP, MR - ML, 330, "code donors",
                     "The paired estimand compares two donors that differ in notation and in nothing else.")
D.fit(s, A + "s3_donors.png", x, y, w, h, align="left")
y2 = BODY_TOP + 330 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "instruction donors",
                     "Two controls, not one: an identity-ish donor and a semantically unrelated one bracket the disturbance from both sides.")
table(D, s, x, y, w, ["donor", "what it writes", "role"],
      [["opposite", "the directive word from the opposite instruction", "treatment"],
       ["self", "the same directive word from the same instruction", "control — zero new information"],
       ["unrelated_word", "an unrelated word from the same instruction", "control — unrelated content"]],
      [400, 760, 560], size=10, fit_h=h, tag="donor-instr")

s = D.slide(C2, "The bubble, and the number we actually report",
            "The control's value is not noise. It is the measurable disturbance of the act of overwriting.")
lx, lw, rx, rw = split(1120)
x, y, w, h = D.panel(s, lx, BODY_TOP, lw, 430, "raw, bubble, net",
                     "Reading the raw curve alone would overstate every effect in this study by roughly a factor of two.")
D.plot(s, f"{FIG}/step3/figures/explain_bubble.png", x, y, w, h)
x, y, w, h = D.panel(s, rx, BODY_TOP, rw, 430, "eq. 14 — paired net effect",
                     "Paired within a block, then averaged over blocks.")
D.equation(s, A + "eq_net.png", x, y + 6, h=54)
D.body(s, x, y + 92, w, "Pairing matters: blocks differ in difficulty, and taking the difference first removes that "
       "variation before the confidence interval is computed.", fit_h=h - 100, tag="net-note")
y2 = BODY_TOP + 430 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "how large the bubble is, in practice",
                     "Measured at each model's peak layer in step 3, on the Value road.")
cw = (w - 3 * GAP) / 4
for i, (fam, name) in enumerate(MODELS):
    xx = x + i * (cw + GAP)
    D.stat(s, xx, y, cw, f"{round(S['step3'][fam]['bubble_pct']*100)}%", name, size=30, fit_h=h, tag="bub"+fam)

# ═══════════════════════════════════════════════════ 03 · Audit ══════════
s = D.slide(C3, "Why the earlier experiments were discarded",
            "Not because the numbers were bad. Because the harness had forked, and no branch could reproduce the whole study.")
x, y, w, h = D.panel(s, ML, BODY_TOP, MR - ML, 340, "how the project got here",
                     "The archive is kept read-only as evidence of what was discarded, and is never cited as a result.")
D.fit(s, A + "h_timeline.png", x, y, w, h, align="left")
y2 = BODY_TOP + 340 + GAP
lx, lw, rx, rw = split(900)
x, y, w, h = D.panel(s, lx, y2, lw, BODY_BOT - y2, "the failure mode")
D.body(s, x, y, w, "Each step had grown its own copy of the engine. Two steps could disagree about how a span was "
       "located or how a substitution was performed, and nothing in the results would show it.",
       fit_h=h, tag="fork")
x, y, w, h = D.panel(s, rx, y2, rw, BODY_BOT - y2, "the rule that replaced it")
D.body(s, x, y, w, "One engine, one entry point, one results convention. A new experiment adds a value to the "
       "condition schema — never a new script.", fit_h=h, tag="fork-fix")

s = D.slide(C3, "Silent failure is the defect class we build against",
            "If “no effect” and “the experiment never ran” are stored as the same number, no amount of analysis can tell them apart.")
x, y, w, h = D.panel(s, ML, BODY_TOP, MR - ML, 420, "guards in place",
                     "Each guard raises rather than returning a plausible value; a crashed run is recoverable, a silent one is not.")
D.fit(s, A + "h_silent.png", x, y, w, h, align="left")
y2 = BODY_TOP + 420 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "the one place we deliberately do not raise",
                     "Undecidable conditions are legitimate measurements whose ratio is unstable — they are recorded, flagged and excluded.")
D.body(s, x, y, w, "An undecidable condition is not a bug: the model simply had no strong preference to restore, so "
       "the denominator of the transfer rate is near zero. Both the flag and the raw gap are stored, so the "
       "threshold can be varied afterwards without re-running anything.", fit_h=h, tag="undec-note")

s = D.slide(C3, "A concrete silent failure, found in our own results",
            "Seven layers were requested. One was measured. Nothing in the file name or the stored condition said so.")
x, y, w, h = D.panel(s, ML, BODY_TOP, MR - ML, 400, "requested versus measured",
                     "Found by comparing each record's stored layer list against the layer its metrics were actually keyed on.")
D.fit(s, A + "h_layerbug.png", x, y, w, h, align="left")
y2 = BODY_TOP + 400 + GAP
lx, lw, rx, rw = split(1100)
x, y, w, h = D.panel(s, lx, y2, lw, BODY_BOT - y2, "why no reported number is affected",
                     "The bug is a coverage loss, not a contamination.")
D.body(s, x, y, w, "Every affected record stores an empty per-layer block, so no analysis script could include it "
       "even by accident. All 672 belong to step 5's controls, which are excluded and footnoted.",
       fit_h=h, tag="bug-safe")
x, y, w, h = D.panel(s, rx, y2, rw, BODY_BOT - y2, "swept everywhere else")
D.body(s, x, y, w, "The same pattern was searched for across every results folder. Occurrences outside step 5's "
       "controls: zero.", fit_h=h, tag="bug-sweep")

s = D.slide(C3, "What the audit changed, and what it did not",
            "Most findings were corrections to how results were described, not to the results themselves.")
x, y, w, h = D.panel(s, ML, BODY_TOP, MR - ML, BODY_BOT - BODY_TOP, "verdict per finding",
                     "Re-running was reserved for cases where the stored numbers could not answer the question; everything else was re-analysed from the existing files.")
table(D, s, x, y, w, ["finding", "verdict", "what was done"],
      [["Span attention reported as a sum while the text said per token",
        "re-analysed", "recomputed from stored values; peak layers unchanged"],
       ["“av_norm is the actual contribution”",
        "wording", "corrected to an upper bound — it ignores cancellation and precedes the output projection"],
       ["“The Key net effect is zero”",
        "wording", "corrected to negligible — 2–11% of the Value effect, intervals do not contain zero"],
       ["“RoPE concern rejected”",
        "wording", "narrowed to the norm-collapse hypothesis only; phase was never measured"],
       ["Step 1 figure plotted only one instruction direction",
        "re-plotted", "both directions now shown; the collapse holds in both"],
       ["672 step-5 control records measured one layer, not seven",
        "excluded", "flagged, footnoted, and guarded against in the runner"]],
      [780, 220, 720], size=10.5, fit_h=h, tag="audit")

s = D.slide(C3, "A determinism check that came free",
            "Two steps share 168 conditions. If the harness is deterministic, they must agree exactly — and they do.")
lx, lw, rx, rw = split(1000)
x, y, w, h = D.panel(s, lx, BODY_TOP, lw, 420, "the check")
D.body(s, x, y, w, "Step 2's conditions are a subset of step 4's: the same models, the same blocks, the same "
       "instruction direction. Their shared spans were compared value by value, across every layer.",
       fit_h=h, tag="det")
D.hline(s, x, y + 130, w, HAIR)
kv_rows(D, s, x, y + 156, w, [
    ("Overlapping conditions", "168"),
    ("Values compared", "80,640"),
    ("Mismatches", "0"),
    ("Maximum absolute difference", "0.000e+00")], pitch=40)
x, y, w, h = D.panel(s, rx, BODY_TOP, rw, 420, "what it does and does not license",
                     "It is a check on the instrument, not on the science.")
D.deflist(s, x, y, w, [
    ("Licenses", "that the measurement pipeline is deterministic under a fixed seed"),
    ("Licenses", "that step 2 and step 4 can be read as one measurement of one quantity"),
    ("Does not", "say the quantity is the right one to measure"),
    ("Does not", "substitute for a control — determinism is not validity")],
    fit_h=h, tag="det-lic")
y2 = BODY_TOP + 420 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "why it is worth a slide",
                     "Reproducibility claims are cheap to make and rarely checked; this one is checkable from the stored files alone.")
D.body(s, x, y, w, "The two steps were run months apart, on different machines, in different sessions. Exact "
       "agreement across 80,640 values is evidence that the seed, the prompt construction and the attention "
       "read-out are all pinned down.", fit_h=h, tag="det-why")

# ═══════════════════════════════════════════════════ 04 · Harness ════════
s = D.slide(C4, "Three design principles",
            "The engine is boring on purpose. Everything interesting lives in the condition, not in the code path.")
cw = (MR - ML - 2 * GAP) / 3
for i, (eb, t, b) in enumerate([
        ("principle 1", "One entry point",
         "Every experiment calls the same run() with a condition and a mode. There is no per-question script, "
         "so two steps cannot silently disagree about how a measurement is taken."),
        ("principle 2", "Fail loudly",
         "Unsupported kinds, empty substitution sets and unlocatable spans raise. A result file never contains "
         "a number that was produced by an experiment that did not happen."),
        ("principle 3", "Results are immutable",
         "A finished file is never rewritten. Resume works by checking whether the path exists, which also "
         "makes every run interruptible without loss.")]):
    xx = ML + i * (cw + GAP)
    x, y, w, h = D.panel(s, xx, BODY_TOP, cw, 340, eb)
    D.cardtitle(s, x, y, t, w=w)
    D.body(s, x, y + 34, w, b, fit_h=h - 42, tag=f"prin{i}")
y2 = BODY_TOP + 340 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "what this costs",
                     "The condition schema is wide, and reading a slug takes practice. That is the price of one reproducible engine.")
D.body(s, x, y, w, "Every knob any experiment ever needed lives in one dataclass, so the schema carries fields "
       "most conditions leave unset. We consider that better than the alternative the audit found: several "
       "engines that each looked simple and none of which agreed.", fit_h=h, tag="prin-cost")

s = D.slide(C4, "Module map",
            "One router, five workers. Nothing calls anything sideways.")
x, y, w, h = D.panel(s, ML, BODY_TOP, MR - ML, 430, "call structure",
                     "Workers are called by the router only — so a change in one cannot quietly alter another's behaviour.")
D.fit(s, A + "h_modules.png", x, y, w, h, align="left")
y2 = BODY_TOP + 430 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "what each module owns")
cw = (w - 2 * GAP) / 3
kv_rows(D, s, x, y, cw, [("prompt", "text and span offsets"), ("attention_probe", "one query row")], pitch=36)
kv_rows(D, s, x + cw + GAP, y, cw, [("intervention", "cache edits"), ("metrics", "scores and rates")], pitch=36)
kv_rows(D, s, x + 2 * (cw + GAP), y, cw, [("steer", "residual add, reweight"), ("model", "loading and cache")], pitch=36)

s = D.slide(C4, "Run modes",
            "One function, eight behaviours. The mode decides which workers the router calls.")
x, y, w, h = D.panel(s, ML, BODY_TOP, MR - ML, BODY_BOT - BODY_TOP, "run(condition, mode=…)",
                     "Modes marked with a dash have no per-layer output by design — they produce a behaviour, not a layer profile.")
table(D, s, x, y, w, ["mode", "what it does", "per-layer output", "used by"],
      [["generate", "generate a function, classify its notation", "—", "step 1"],
       ["observe", "read one attention row and its value norms", "yes", "step 2 · step 4"],
       ["intervene", "overwrite the cache at one layer, then score", "one layer", "step 3 · step 5"],
       ["intervene_sweep", "the same, repeated across every layer", "yes", "step 3 · step 5"],
       ["vcosine", "compare value directions without scoring", "yes", "diagnostics"],
       ["kv_diagnose", "measure what the substitution does to K and V", "yes", "diag"],
       ["steer", "add a direction, or reweight attention, then score", "—", "step 6"],
       ["steer_generate", "the same, but generate and inspect the name", "—", "step 6"]],
      [300, 760, 300, 360], size=10.5, fit_h=h, tag="modes")

s = D.slide(C4, "The condition schema",
            "A condition is data. Reading one tells you the whole experiment without opening any code.")
lx, lw, rx, rw = split(1160)
x, y, w, h = D.panel(s, lx, BODY_TOP, lw, 470, "one condition, in full")
code(D, s, x, y, w, h, [
    "Condition(",
    "    model      = ModelSpec('Qwen/Qwen2.5-Coder-3B-Instruct', 'qwen'),",
    "    preceding  = PrecedingCode(n_compliant=6, n_functions=12,",
    "                               composition=POOL, pool_block=0),",
    "    instruction= Instruction(form=POSITIVE, target_notation=CAMEL),",
    "»    intervention= Intervention(kind=KEY_VALUE, layers='sweep',",
    "»                               donor='unrelated_camel', target='code',",
    "»                               kinds=('key','value','key_value')),",
    "    seed=42, token_unit='mean')"], size=9)
x, y, w, h = D.panel(s, rx, BODY_TOP, rw, 470, "the fields that carry the design")
D.deflist(s, x, y, w, [
    ("preceding", "how many of the twelve names violate, and which block"),
    ("instruction", "which notation is required, and in what phrasing"),
    ("intervention", "which road, which layer, and from which donor"),
    ("token_unit", "how donor pieces are combined — mean-pool throughout")],
    fit_h=h, tag="cond-fields")
y2 = BODY_TOP + 470 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "one field that is not a condition axis",
                     "It survives in the schema because removing it would invalidate the archive; it is never varied in the unified re-run.")
D.body(s, x, y, w, "The instruction form has a negative variant. It belongs to a research question that was cut, "
       "and every condition in this study fixes it to positive. Keeping the field but never varying it is safer "
       "than deleting it and losing the ability to read archived results.", fit_h=h, tag="negform")

s = D.slide(C4, "Building the prompt",
            "Three parts, assembled the same way in every step. Nothing about the prompt varies except what a condition says varies.")
lx, lw, rx, rw = split(1120)
x, y, w, h = D.panel(s, lx, BODY_TOP, lw, 460, "assembly")
code(D, s, x, y, w, h, [
    "# 1 · system — the instruction",
    "You are helping extend an existing Python module.",
    "»In this project we generally write function names in camelCase.",
    "Every function name uses one of two styles only:",
    "camelCase or snake_case.",
    "",
    "# 2 · user — twelve existing functions from one block",
    "def decodeNode(value): ...      # compliant   x n",
    "»def split_config(value): ...    # violating   x (12 - n)",
    "",
    "# 3 · request",
    "Add one more function."], size=9)
x, y, w, h = D.panel(s, rx, BODY_TOP, rw, 460, "what each part controls")
D.deflist(s, x, y, w, [
    ("System", "the instruction direction — which notation is required"),
    ("Body", "the violation ratio, swept from 0 to 12 in step 1 and fixed at 6 elsewhere"),
    ("Order", "names are interleaved, not grouped, so recency cannot explain the effect"),
    ("Request", "identical in every condition, so the decision point is always the same")],
    fit_h=h, tag="prompt-parts")
y2 = BODY_TOP + 460 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "why the context is synthetic",
                     "Real repository files would confound notation with topic, length, and difficulty all at once.")
D.body(s, x, y, w, "Every name is drawn from the same verb-noun pool and every function body has the same shape, so "
       "the only thing that differs between a compliant and a violating context is the notation itself. The cost "
       "is external validity, which is stated as a limitation rather than argued away.", fit_h=h, tag="synth")

s = D.slide(C4, "Locating the spans inside the prompt",
            "Spans are located by character offset in the assembled text, then mapped onto tokens.")
x, y, w, h = D.panel(s, ML, BODY_TOP, MR - ML, 400, "text to character span to token span",
                     "A span that cannot be located raises — it is never reported as zero attention.")
D.fit(s, A + "h_span.png", x, y, w, h, align="left")
y2 = BODY_TOP + 400 + GAP
lx, lw, rx, rw = split(1000)
x, y, w, h = D.panel(s, lx, y2, lw, BODY_BOT - y2, "the mapping, in code")
code(D, s, x, y, w, h, [
    "start = text.index(target_word)",
    "»spans['instr_rule_word'] = char_to_token(enc, start, len(target_word))",
    "if not spans['instr_rule_word']:",
    "    raise SpanNotFound('instr_rule_word')   # never return zero"], size=9)
x, y, w, h = D.panel(s, rx, y2, rw, BODY_BOT - y2, "why offsets and not string search on tokens",
                     "Token strings differ per model; character offsets do not.")
D.body(s, x, y, w, "The same span therefore covers the same characters on all four models, even though it covers "
       "a different number of tokens on each. That difference is exactly what the per-token normalisation removes.",
       fit_h=h, tag="span-why")

s = D.slide(C4, "Performing the substitution",
            "Read the cache, average the donor, write it into the target slots of one KV group at one layer.")
lx, lw, rx, rw = split(1160)
x, y, w, h = D.panel(s, lx, BODY_TOP, lw, 450, "the core of the intervention")
code(D, s, x, y, w, h, [
    "k, v = past_key_values[layer]        # [b, kv_heads, seq, d_head]",
    "",
    "donor_v = v[:, :, donor_positions, :].mean(dim=2, keepdim=True)",
    "»v[:, :, target_positions, :] = donor_v      # broadcast, no skips",
    "",
    "if kind == 'key':      write only k",
    "if kind == 'value':    write only v",
    "if kind == 'key_value': write both",
    "",
    "if len(target_positions) == 0:",
    "    raise EmptySubstitution(layer, donor)    # never silently no-op"], size=9)
x, y, w, h = D.panel(s, rx, BODY_TOP, rw, 450, "what is guaranteed here")
D.deflist(s, x, y, w, [
    ("Coverage", "every target position is written, on every model"),
    ("Granularity", "the write is per KV group, as the architecture requires"),
    ("Locality", "exactly one layer is touched unless the mode is a sweep"),
    ("Loudness", "an empty target set raises instead of returning the baseline")],
    fit_h=h, tag="int-guar")
y2 = BODY_TOP + 450 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "the three kinds are measured on one forward pass",
                     "So the Key, Value and combined numbers on any slide come from the same prompt, the same cache and the same conditions.")
D.body(s, x, y, w, "Separating them across runs would introduce a difference in nothing but bookkeeping, and would "
       "make the paired comparison between roads weaker than it needs to be.", fit_h=h, tag="int-three")

s = D.slide(C4, "Where results go, and why the path matters",
            "One directory per step, one file per condition, and a slug that encodes the whole condition.")
x, y, w, h = D.panel(s, ML, BODY_TOP, MR - ML, 380, "path and slug",
                     "Split this path by model and both immutability and resume break at once.")
D.fit(s, A + "h_slug.png", x, y, w, h, align="left")
y2 = BODY_TOP + 380 + GAP
lx, lw, rx, rw = split(1000)
x, y, w, h = D.panel(s, lx, y2, lw, BODY_BOT - y2, "what a result file contains")
kv_rows(D, s, x, y, w, [
    ("condition", "the full condition, as stored data"),
    ("metrics.per_layer", "layer-keyed measurements, when the mode produces them"),
    ("metrics.extra", "mode, donor, the three reference scores, gap, undecidable"),
    ("meta", "git sha, library versions, GPU — filled in automatically")], pitch=38)
x, y, w, h = D.panel(s, rx, y2, rw, BODY_BOT - y2, "why extra carries the reference scores",
                     "Any later re-analysis can recompute the rate from the stored scores.")
D.body(s, x, y, w, "Storing the clean, base and intervened scores rather than only the derived rate means the "
       "undecidability threshold can be moved afterwards, and the sensitivity reported, without a single GPU hour.",
       fit_h=h, tag="extra-why")

s = D.slide(C4, "Reproducibility, concretely",
            "What is pinned, what is checked, and what can be verified without a GPU.")
cw = (MR - ML - 2 * GAP) / 3
for i, (eb, items) in enumerate([
        ("pinned", [("Seed", "42 in every condition, stored with the result"),
                    ("Versions", "library versions recorded per run"),
                    ("Commit", "git sha stamped into every file")]),
        ("checked", [("Determinism", "80,640 shared values, 0 mismatches"),
                     ("Guards", "unit tests assert that each guard raises"),
                     ("Census", "condition counts recomputed from disk, not remembered")]),
        ("verifiable offline", [("Prompts", "built and printed without loading a model"),
                                ("Spans", "located and inspected on CPU"),
                                ("Aggregation", "every figure regenerated from stored JSON")])]):
    xx = ML + i * (cw + GAP)
    x, y, w, h = D.panel(s, xx, BODY_TOP, cw, 400, eb)
    D.deflist(s, x, y, w, items, fit_h=h, tag=f"repro{i}")
y2 = BODY_TOP + 400 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "the rule that keeps figures honest",
                     "A number that appears on a slide but in no script is a number nobody can check.")
D.body(s, x, y, w, "Every aggregate and every figure in this deck is produced by a script that reads the stored "
       "result files. No value was transcribed by hand, and no figure was drawn by hand.", fit_h=h, tag="fig-rule")

# ═══════════════════════════════════ 05 · Experiments — step 1 ═══════════
s = D.slide(C5, "Step 1 · The question",
            "Does a file that already breaks the convention make the model break it too?")
lx, lw, rx, rw = split(880)
x, y, w, h = D.panel(s, lx, BODY_TOP, lw, 430, "what is manipulated")
D.deflist(s, x, y, w, [
    ("Held fixed", "the instruction text, the request, the twelve function bodies"),
    ("Varied", "how many of the twelve names violate the instruction"),
    ("Measured", "whether the generated function name follows the instruction"),
    ("Repeated", "over 42 disjoint name blocks and both notation directions")],
    fit_h=h, tag="s1-q")
x, y, w, h = D.panel(s, rx, BODY_TOP, rw, 430, "what each outcome would have meant",
                     "Stated before the run — the observed outcome is the first row.")
table(D, s, x, y, w, ["if we had seen", "the reading"],
      [["a sharp fall with more violations", "context drives the failure; there is a mechanism to find"],
       ["a flat line", "the failure is about the instruction, not the context; the study stops"],
       ["a fall in one direction only", "the effect is about snake_case, not about violation as such"]],
      [42, 58], size=10, fit_h=h, tag="s1-out")
y2 = BODY_TOP + 430 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "why this step uses generation, not preference",
                     "Every later step measures preference; this one has to establish that the behaviour itself moves.")
D.body(s, x, y, w, "A preference score would be cheaper and less noisy, but it would not show that the model actually "
       "writes the wrong name. Step 1 pays the cost of generation so that the rest of the study can use the "
       "cheaper measure with a known anchor.", fit_h=h, tag="s1-why")

s = D.slide(C5, "Step 1 · The condition grid",
            "2,184 generations: thirteen violation levels, two directions, forty-two blocks, two models.")
x, y, w, h = D.panel(s, ML, BODY_TOP, MR - ML, 420, "grid and its axes",
                     "Only two of the four models were run here — the two code-specialised ones. That is a stated limitation, not an oversight.")
D.fit(s, A + "s1_grid.png", x, y, w, h, align="left")
y2 = BODY_TOP + 420 + GAP
lx, lw, rx, rw = split(1000)
x, y, w, h = D.panel(s, lx, y2, lw, BODY_BOT - y2, "why both directions matter",
                     "Without the second direction, a notation preference would be indistinguishable from a violation effect.")
D.body(s, x, y, w, "If only the camelCase-required direction were run, a model that simply likes snake_case would "
       "produce the same curve as a model that is influenced by violations. Running both directions separates them.",
       fit_h=h, tag="s1-dir")
x, y, w, h = D.panel(s, rx, y2, rw, BODY_BOT - y2, "one generation per condition",
                     "Greedy decoding, so the only randomness is in which names a block contains.")
D.body(s, x, y, w, "The 42 blocks provide the replication instead of repeated sampling. That keeps the design paired: "
       "the same names appear at every violation level.", fit_h=h, tag="s1-rep")

s = D.slide(C5, "Step 1 · Pipeline",
            "Build, generate, extract, classify, score. Names that are neither notation are kept, not dropped.")
x, y, w, h = D.panel(s, ML, BODY_TOP, MR - ML, 300, "five stages",
                     "Dropping unclassifiable names would silently shrink the denominator and inflate compliance.")
D.fit(s, A + "s1_pipe.png", x, y, w, h, align="left")
y2 = BODY_TOP + 300 + GAP
lx, lw, rx, rw = split(1120)
x, y, w, h = D.panel(s, lx, y2, lw, BODY_BOT - y2, "classification, in code")
code(D, s, x, y, w, h, [
    "name = re.search(r'def\\s+([A-Za-z_][A-Za-z0-9_]*)', text).group(1)",
    "",
    "if '_' in name:                    notation = 'snake'",
    "elif re.search(r'[a-z][A-Z]', name): notation = 'camel'",
    "»else:                              notation = 'neither'   # still counted",
    "",
    "compliant = (notation == condition.instruction.target_notation)"], size=9)
x, y, w, h = D.panel(s, rx, y2, rw, BODY_BOT - y2, "what is stored per generation",
                     "The full text is kept so any classification question can be re-examined.")
kv_rows(D, s, x, y, w, [("turn_names", "the extracted name"), ("turn_notations", "camel, snake or neither"),
                        ("turn_texts", "the generated text, verbatim"),
                        ("compliance_rate", "1 or 0 for this condition")], pitch=38)

s = D.slide(C5, "Step 1 · Result — compliance collapses with context",
            "The instruction never changes. The surrounding file does, and that is enough.")
lx, lw, rx, rw = split(1180)
x, y, w, h = D.panel(s, lx, BODY_TOP, lw, 400, "compliance against violation count, both directions",
                     "Two models, two instruction directions. The collapse is present in all four curves.")
D.plot(s, f"{FIG}/step1/figures/cliff.png", x, y, w, h)
x, y, w, h = D.panel(s, rx, BODY_TOP, rw, 400, "reading the curves")
D.deflist(s, x, y, w, [
    ("Left edge", "a fully compliant file — the model follows the instruction"),
    ("Right edge", "a fully violating file — the instruction is largely ignored"),
    ("Middle", "the transition is narrow, not gradual"),
    ("Both directions", "the same shape appears whichever notation is required")],
    fit_h=h, tag="s1-read")
y2 = BODY_TOP + 400 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "the numbers behind the curves",
                     "Averaged over both instruction directions and 42 name blocks.")
def _c(fam, viol):
    vv = [S["step1"][f"{fam}|{d}|{viol}"] for d in ("camel", "snake")]
    return f"{sum(vv)/2:.0%}"
table(D, s, x, y, w, ["model", "0 violating", "3", "6", "9", "12 violating"],
      [["Qwen2.5-Coder-3B"] + [_c("qwen", v) for v in (0, 3, 6, 9, 12)],
       ["StableCode-3B"] + [_c("stability", v) for v in (0, 3, 6, 9, 12)]],
      [34, 14, 12, 12, 12, 16], size=10.5, fit_h=h, tag="s1-nums")

s = D.slide(C5, "Step 1 · Result — the same picture, read a second way",
            "A conditional view of the same generations, to check that the collapse is not an artefact of averaging.")
lx, lw, rx, rw = split(1180)
x, y, w, h = D.panel(s, lx, BODY_TOP, lw, 470, "explanatory view",
                     "Same 2,184 generations, re-aggregated; nothing is re-run.")
D.plot(s, f"{FIG}/step1/figures/explain_cliff.png", x, y, w, h)
x, y, w, h = D.panel(s, rx, BODY_TOP, rw, 470, "why a second view was needed")
D.body(s, x, y, w, "A mean over 42 blocks can hide a bimodal population: most blocks unaffected and a few collapsing "
       "completely would produce a similar average curve. Re-aggregating conditionally shows the shift is in the "
       "bulk, not in a tail.", fit_h=h, tag="s1-cond")
y2 = BODY_TOP + 470 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "both views come from one set of stored files",
                     "No condition was added to produce the second figure.")
D.body(s, x, y, w, "This is the practical payoff of storing the generated text rather than only the compliance flag: "
       "a new question about the same runs costs an aggregation script, not GPU time.", fit_h=h, tag="s1-store")

s = D.slide(C5, "Step 1 · What this step does not settle",
            "It establishes the symptom. It says nothing about mechanism, and its coverage is narrower than the rest of the study.")
cw = (MR - ML - 3 * GAP) / 4
for i, (t, b) in enumerate([
        ("Two models only", "Both are code-specialised. The general-purpose model was not run here, so the "
         "collapse is not established for it."),
        ("Python only", "The harness supports JavaScript; zero JavaScript conditions were run. No claim covers "
         "other languages."),
        ("First function only", "One generation per condition, so whether violations amplify across successive "
         "functions cannot be judged."),
        ("Synthetic context", "Names come from one pool and bodies share a shape, which buys control at the cost "
         "of external validity.")]):
    xx = ML + i * (cw + GAP)
    x, y, w, h = D.panel(s, xx, BODY_TOP, cw, 380, f"limit {i+1}")
    D.cardtitle(s, x, y, t, w=w)
    D.body(s, x, y + 34, w, b, fit_h=h - 42, tag=f"s1-lim{i}")
y2 = BODY_TOP + 380 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "how the rest of the study compensates",
                     "None of these limits is repaired by argument; each is either covered by a later step or carried forward as a stated limitation.")
D.body(s, x, y, w, "Steps 2 to 6 run all four models, so the mechanism claims are not restricted to two. Language "
       "coverage and multi-function amplification are not repaired anywhere and remain open.", fit_h=h, tag="s1-comp")

s = D.slide(C5, "Step 1 · Reading the result file",
            "One JSON per condition. The compliance flag alone cannot tell you why a condition failed.")
lx, lw, rx, rw = split(1160)
x, y, w, h = D.panel(s, lx, BODY_TOP, lw, 470, "results/step1_cliff/<slug>.json")
code(D, s, x, y, w, h, [
    '"metrics": {',
    '  "compliance_rate": 0.0,            # 1.0 if the first name matches',
    '  "extra": {',
    '    "target": "camel",               # notation the instruction required',
    '»    "turn_names": ["remove_duplicates"],   # what the model actually wrote',
    '    "turn_notations": ["snake"],     # camel | snake | other',
    '    "turn_texts": ["def remove_dup…"],      # verbatim generation',
    '    "first_compliant": false,',
    '    "first_violated": true,',
    '    "subsequent_violation_rate": null       # always null — one turn only',
    '  }',
    '}'], size=9)
x, y, w, h = D.panel(s, rx, BODY_TOP, rw, 470, "how to read it")
D.deflist(s, x, y, w, [
    ("Start here", "turn_names — a zero compliance rate means either a wrong notation or no extractable name, and only this field separates them"),
    ("Verify by eye", "turn_texts keeps the raw generation, so any classification decision can be re-checked"),
    ("Ignore", "subsequent_violation_rate is null everywhere; one generation per condition")],
    fit_h=h, tag="s1-json")
y2 = BODY_TOP + 470 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "aggregation",
                     "Groups by model, violation count and direction, then averages over the 42 blocks.")
code(D, s, x, y, w, h, ["python scripts/step1_reaggregate.py"], size=10)

# ═══════════════════════════════════ 05 · step 2 ═════════════════════════
s = D.slide(C5, "Step 2 · The question",
            "When the file breaks the convention, does the model stop looking at it — or look at it more?")
lx, lw, rx, rw = split(900)
x, y, w, h = D.panel(s, lx, BODY_TOP, lw, 420, "what is read")
D.deflist(s, x, y, w, [
    ("One row", "the attention of the position about to write the name"),
    ("Five spans", "compliant names, violating names, the instruction, and two words in it"),
    ("Every layer", "28 to 36 depending on the model"),
    ("Averaged", "over heads, and over 42 blocks")], fit_h=h, tag="s2-what")
x, y, w, h = D.panel(s, rx, BODY_TOP, rw, 420, "what each outcome would have meant",
                     "The observed outcome is the second row — the opposite of the attention-deficit account.")
table(D, s, x, y, w, ["if we had seen", "the reading"],
      [["less attention on violations", "consistent with an attention deficit; Spotlight-style remedies plausible"],
       ["more attention on violations", "the model is not ignoring them; the failure is about content, not looking"],
       ["no difference", "attention is the wrong place to look at all"]],
      [40, 60], size=10, fit_h=h, tag="s2-out", hi=1)
y2 = BODY_TOP + 420 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "this step is correlational, and says so",
                     "The matching causal step is step 3; nothing here is presented as a mechanism.")
D.body(s, x, y, w, "Attention is observed, not manipulated. A difference between spans shows where the model's "
       "attention sits, not whether that attention is what produces the behaviour. Treating an observation as "
       "a mechanism is the specific error this study is organised to avoid.", fit_h=h, tag="s2-corr")

s = D.slide(C5, "Step 2 · The prompt, and the spans read from it",
            "Five spans, located by character offset, then measured per token.")
x, y, w, h = D.panel(s, ML, BODY_TOP, MR - ML, 460, "prompt anatomy",
                     "Only the name characters are taken — not the def keyword, the parentheses, or the body.")
D.fit(s, A + "s2_prompt.png", x, y, w, h, align="left")
y2 = BODY_TOP + 460 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "span token counts differ by model, which is why we divide",
                     "One name can be two pieces on one tokenizer and four on another; the span is the same characters either way.")
D.body(s, x, y, w, "Because the number of tokens per span is model-dependent, comparing raw span sums across models "
       "would compare tokenizations as much as behaviour. Dividing by the span's own token count removes that.",
       fit_h=h, tag="s2-tok")

s = D.slide(C5, "Step 2 · How the attention is read",
            "One forward pass with attentions returned, one row selected, spans summed and divided.")
lx, lw, rx, rw = split(1120)
x, y, w, h = D.panel(s, lx, BODY_TOP, lw, 440, "the read-out")
code(D, s, x, y, w, h, [
    "out = model(input_ids, output_attentions=True)",
    "",
    "for layer, att in enumerate(out.attentions):    # [b, heads, seq, seq]",
    "»    row = att[0, :, -1, :].mean(dim=0)          # heads averaged, last row",
    "    for name, idx in spans.items():",
    "        total = row[idx].sum().item()",
    "»        per_token = total / len(idx)            # what we report",
    "",
    "# the row sums to 1, so total is a share of this decision's attention"], size=9)
x, y, w, h = D.panel(s, rx, BODY_TOP, rw, 440, "three quantities per span")
D.deflist(s, x, y, w, [
    ("attention_weight", "the share of attention the span received"),
    ("v_norm", "the average length of the value vectors at those positions"),
    ("av_norm", "the sum of weight times value length — an upper bound, not a contribution")],
    fit_h=h, tag="s2-three")
y2 = BODY_TOP + 440 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "why av_norm is called an upper bound",
                     "It ignores cancellation between vectors and it precedes the output projection.")
D.equation(s, A + "eq_avnorm.png", x, y + 2, h=52)

s = D.slide(C5, "Step 2 · Two estimands, and where they disagree",
            "Span sum and per token answer different questions. We report per token and state the disagreement rather than hiding it.")
x, y, w, h = D.panel(s, ML, BODY_TOP, MR - ML, 340, "the two summaries",
                     "Neither is “the fair one” — they answer different questions about the same measurement.")
D.fit(s, A + "s2_estimand.png", x, y, w, h, align="left")
y2 = BODY_TOP + 340 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "sensitivity, measured rather than asserted",
                     "Peak layers are unchanged under both estimands in every model.")
table(D, s, x, y, w, ["model", "layers", "sign flips between estimands"],
      [[n, S["step2"][f]["n_layers"], v] for (f, n), v in zip(
          MODELS, ["0 of 36", "7 of 32", "0 of 28", "9 of 32"])],
      [40, 20, 40], size=10.5, fit_h=h, tag="s2-sens")

s = D.slide(C5, "Step 2 · Result — the model looks more, not less",
            "Attention per token on violating names sits above compliant names in every model.")
four_plots(D, s, "step2", "attention_{}.png", BODY_TOP, 430,
           "Attention per token, by layer. Violating names above compliant names.")
y2 = BODY_TOP + 430 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "what this rules out",
                     "The attention-deficit account predicted the opposite sign. It does not survive this measurement.")
D.body(s, x, y, w, "If the model were failing because it does not look at the offending material, violating names "
       "would receive less attention than compliant ones. They receive more. Whatever is going wrong, it is not "
       "that the model is failing to attend.", fit_h=h, tag="s2-rule")

s = D.slide(C5, "Step 2 · Result — the same spans, seen together",
            "All five spans on one axis, so the instruction and the code can be compared directly.")
four_plots(D, s, "step2", "spans_{}.png", BODY_TOP, 430,
           "All five spans, attention per token, by layer.")
y2 = BODY_TOP + 430 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "a caution about the instruction span here",
                     "The role-separated version of this comparison is step 4; this step's instruction keys are the initial ones.")
D.body(s, x, y, w, "In this step the instruction's notation word is counted wherever it appears, which includes the "
       "candidate list as well as the rule sentence. That inflates it by construction, and is exactly the defect "
       "step 4 was built to fix.", fit_h=h, tag="s2-caut")

s = D.slide(C5, "Step 2 · Result — value norms and the upper bound",
            "Attention share is not the only thing that differs between spans; the vectors themselves differ in length.")
four_plots(D, s, "step2", "av_{}.png", BODY_TOP, 400,
           "Attention times value norm, per token — an upper bound on contribution.")
y2 = BODY_TOP + 400 + GAP
lx, lw, rx, rw = split(1080)
x, y, w, h = D.panel(s, lx, y2, lw, BODY_BOT - y2, "why this is reported at all",
                     "It is the first hint that the two roads carry different amounts of signal.")
D.body(s, x, y, w, "If violating names attracted more attention but carried shorter value vectors, the extra "
       "attention might deliver no extra content. The upper bound shows that is not what happens.",
       fit_h=h, tag="s2-av")
x, y, w, h = D.panel(s, rx, y2, rw, BODY_BOT - y2, "and why it is not a contribution")
D.body(s, x, y, w, "The true contribution is the norm of the weighted sum, after the output projection. Vectors can "
       "cancel; this quantity cannot see that.", fit_h=h, tag="s2-av2")

s = D.slide(C5, "Step 2 · Result — the gap, and where it peaks",
            "The difference between violating and compliant attention, layer by layer.")
four_plots(D, s, "step2", "gap_{}.png", BODY_TOP, 430,
           "Violating minus compliant, attention per token, by layer.")
y2 = BODY_TOP + 430 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "the peak layers this step nominates",
                     "These are observation peaks. Whether they are the layers that matter causally is step 3's question, not this one's.")
cw = (w - 3 * GAP) / 4
for i, (fam, name) in enumerate(MODELS):
    D.stat(s, x + i * (cw + GAP), y, cw, f"L{S['step2'][fam]['peak']}",
           f"{name} · of {S['step2'][fam]['n_layers']} layers", size=30, fit_h=h, tag="s2pk" + fam)

s = D.slide(C5, "Step 2 · What this step does not settle",
            "It is one direction, one language, one estimand choice, and no causation.")
cw = (MR - ML - 3 * GAP) / 4
for i, (t, b) in enumerate([
        ("Not causal", "Attention is observed, never manipulated. Step 3 supplies the intervention that this "
         "step cannot."),
        ("One direction", "Only camelCase-required conditions were run, so “violating” is always snake_case here. "
         "Step 4 runs both."),
        ("Head-averaged", "A single specialised head would be invisible. Claims are made at the layer level only."),
        ("Upper bound", "The contribution proxy ignores cancellation and precedes the output projection.")]):
    xx = ML + i * (cw + GAP)
    x, y, w, h = D.panel(s, xx, BODY_TOP, cw, 380, f"limit {i+1}")
    D.cardtitle(s, x, y, t, w=w)
    D.body(s, x, y + 34, w, b, fit_h=h - 42, tag=f"s2-lim{i}")
y2 = BODY_TOP + 380 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "the one that matters most",
                     "This is why the study does not stop after step 2, and why the headline claim is stated causally only after step 3.")
D.body(s, x, y, w, "Because only one instruction direction was run, this step cannot separate “the model attends more "
       "to violations” from “the model attends more to snake_case”. Step 4 runs both directions and finds the same "
       "ordering, which is what licenses the general reading.", fit_h=h, tag="s2-key")

s = D.slide(C5, "Step 2 · Reading the result file",
            "Layer-keyed attention values, and the one field without which none of them are comparable.")
lx, lw, rx, rw = split(1180)
x, y, w, h = D.panel(s, lx, BODY_TOP, lw, 470, "results/step2_code-observe/<slug>.json")
code(D, s, x, y, w, h, [
    '"metrics": {',
    '  "per_layer": {',
    '    "27": {',
    '      "code_camel__attention_weight": 0.0184,   # span SUM, heads averaged',
    '      "code_camel__av_norm":          0.412,    # upper bound, not contribution',
    '      "code_camel__v_norm":           22.4,',
    '      "code_snake__attention_weight": 0.0231, … } },',
    '  "extra": {',
    '»    "span_token_counts": {"code_camel": 12, "code_snake": 15, …},',
    '    "gqa": {"num_attention_heads": 16, "num_key_value_heads": 2,',
    '            "group_size": 8},',
    '    "seq_len": 512, "token_detail": {…} } }'], size=9)
x, y, w, h = D.panel(s, rx, BODY_TOP, rw, 470, "how to read it")
D.deflist(s, x, y, w, [
    ("Read first", "span_token_counts — attention_weight is a span sum, so comparing two spans without dividing by their counts compares lengths"),
    ("Then", "per_layer, keyed by layer index as a string"),
    ("Check", "gqa records the group layout that was actually used for this model")],
    fit_h=h, tag="s2-json")
y2 = BODY_TOP + 470 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "aggregation",
                     "Prints both estimands side by side so the choice is visible rather than assumed.")
code(D, s, x, y, w, h, ["python scripts/observe_per_token.py results/step2_code-observe"], size=10)

# ═══════════════════════════════════ 05 · step 3 ═════════════════════════
s = D.slide(C5, "Step 3 · The question",
            "Attention sits on the violations. Does the content it carries actually drive the behaviour — and which road carries it?")
lx, lw, rx, rw = split(900)
x, y, w, h = D.panel(s, lx, BODY_TOP, lw, 400, "from correlation to cause")
D.deflist(s, x, y, w, [
    ("Manipulate", "the cached vectors at the violating names' positions"),
    ("Separately", "Key only, Value only, and both together"),
    ("At each layer", "one layer at a time, swept across the whole stack"),
    ("Against controls", "so the disturbance of overwriting is subtracted")],
    fit_h=h, tag="s3-q")
x, y, w, h = D.panel(s, rx, BODY_TOP, rw, 400, "what each outcome would have meant",
                     "The observed outcome is the first row.")
table(D, s, x, y, w, ["if we had seen", "the reading"],
      [["Value moves, Key does not", "the notation signal travels as content, not as attention share"],
       ["Key moves, Value does not", "attention share carries the signal; attention remedies are the right family"],
       ["both move equally", "the two roads are not separable by this instrument"]],
      [40, 60], size=10, fit_h=h, tag="s3-out", hi=0)
y2 = BODY_TOP + 400 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "why this is the pivotal step",
                     "Everything after it — RQ3's framing and step 6's prescription — depends on this answer.")
D.body(s, x, y, w, "This is the first point in the study where a claim about mechanism becomes possible. If the "
       "Key road had moved behaviour, the attention-raising family of remedies would still be live and the "
       "prescription in step 6 would be the wrong one to test.", fit_h=h, tag="s3-pivot")

s = D.slide(C5, "Step 3 · What is touched, and what is not",
            "Six name positions, one layer, one KV group. The prompt and its length are identical to the baseline.")
x, y, w, h = D.panel(s, ML, BODY_TOP, MR - ML, 360, "scope of the edit",
                     "Three intervention kinds are measured on one forward pass, so they share every other condition exactly.")
D.fit(s, A + "s3_what.png", x, y, w, h, align="left")
y2 = BODY_TOP + 360 + GAP
lx, lw, rx, rw = split(1040)
x, y, w, h = D.panel(s, lx, y2, lw, BODY_BOT - y2, "the three reference runs",
                     "All three are stored, so the transfer rate can be recomputed later under a different threshold.")
kv_rows(D, s, x, y, w, [
    ("S_clean", "the same block with all twelve names compliant — the ceiling"),
    ("S_base", "six names violating, nothing touched — the floor"),
    ("S_int", "six names violating, cache overwritten — the measurement")], pitch=40)
x, y, w, h = D.panel(s, rx, y2, rw, BODY_BOT - y2, "why a ratio and not a difference",
                     "Models differ in how strongly they prefer either notation.")
D.body(s, x, y, w, "Expressing the effect as a fraction of the clean-to-base distance makes the four models "
       "comparable. The cost is that the ratio is unstable when that distance is small — which is what the "
       "undecidable rule handles.", fit_h=h, tag="s3-ratio")

s = D.slide(C5, "Step 3 · Three donors, one clean comparison",
            "The reported estimand compares two donors whose contexts are identical in everything but notation.")
x, y, w, h = D.panel(s, ML, BODY_TOP, MR - ML, 340, "donor definitions",
                     "The compliant donor is kept as a secondary reading: it is the natural treatment but its context differs from the control's.")
D.fit(s, A + "s3_donors.png", x, y, w, h, align="left")
y2 = BODY_TOP + 340 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "why unrelated-camel minus unrelated-snake",
                     "Both donors are six functions from a separate module written in a single notation, so length and structure cancel exactly.")
D.body(s, x, y, w, "Using the compliant donor against the unrelated-snake control would leave a second difference "
       "in play: the treatment's donor comes from a twelve-function context and the control's from a "
       "six-function one. The paired estimand removes that; the compliant reading is reported beside it and "
       "the two agree within their intervals.", fit_h=h, tag="s3-est")

s = D.slide(C5, "Step 3 · The intervention, in code",
            "Read the cache, mean-pool the donor, write it into the target positions, continue the forward pass.")
lx, lw, rx, rw = split(1160)
x, y, w, h = D.panel(s, lx, BODY_TOP, lw, 460, "one condition, one layer")
code(D, s, x, y, w, h, [
    "past = model(prompt_ids, use_cache=True).past_key_values",
    "",
    "k, v = past[layer]                      # [b, kv_heads, seq, d_head]",
    "donor = v[:, :, donor_pos, :].mean(dim=2, keepdim=True)",
    "»v[:, :, target_pos, :] = donor          # Value road only",
    "",
    "S_int = score(model, past, forced_prefix='def ')",
    "»R     = (S_int - S_base) / (S_clean - S_base)",
    "",
    "record['per_layer'][layer]['value__recovery'] = R"], size=9)
x, y, w, h = D.panel(s, rx, BODY_TOP, rw, 460, "what the sweep produces")
D.deflist(s, x, y, w, [
    ("Per layer", "one transfer rate for each of the three intervention kinds"),
    ("Per block", "one record per name block, so differences can be paired"),
    ("Per donor", "three donors, each a separate condition and a separate file")],
    fit_h=h, tag="s3-sweep")
y2 = BODY_TOP + 460 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "504 files, and what they cover",
                     "42 blocks × 3 donors × 4 models, each sweeping every layer of its model.")
D.body(s, x, y, w, "Every donor is run on every block on every model, so the paired difference is always between "
       "two runs that share the same names, the same seed and the same prompt.", fit_h=h, tag="s3-cov")

s = D.slide(C5, "Step 3 · Result — the raw curves, by donor",
            "Before any subtraction: all three donors move the score, including the one that carries no notation signal.")
four_plots(D, s, "step3", "donors_{}.png", BODY_TOP, 420,
           "Transfer rate by layer, Value road, three donors. The control curve is not flat — that is the bubble.")
y2 = BODY_TOP + 420 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "the finding that forces the control",
                     "A study that reported only the treatment curve would be reporting roughly twice the true effect.")
D.body(s, x, y, w, "The unrelated-snake donor contains no camelCase information at all, yet overwriting with it "
       "moves the preference substantially. That movement is the cost of destroying the original vectors, and "
       "it has to be subtracted before anything is claimed.", fit_h=h, tag="s3-bub")

s = D.slide(C5, "Step 3 · Result — raw, bubble and net side by side",
            "The same curves, decomposed. The blue line is the only one that appears in any claim.")
four_plots(D, s, "step3", "bubble_{}.png", BODY_TOP, 420,
           "Raw treatment, control-only bubble, and the paired net effect, by layer.")
y2 = BODY_TOP + 420 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "bubble share at each model's peak layer",
                     "Between a third and a half of the raw effect is attributable to the act of overwriting rather than to notation.")
cw = (w - 3 * GAP) / 4
for i, (fam, name) in enumerate(MODELS):
    D.stat(s, x + i * (cw + GAP), y, cw, f"{round(S['step3'][fam]['bubble_pct']*100)}%",
           f"{name} · L{S['step3'][fam]['peak']}", size=30, fit_h=h, tag="s3b" + fam)

s = D.slide(C5, "Step 3 · Result — which road carries the effect",
            "Value moves behaviour. Key barely moves it. The two are measured on the same forward pass.")
x, y, w, h = D.panel(s, ML, BODY_TOP, MR - ML, 380, "net effect at each model's peak layer",
                     "Paired within block, then averaged over 42 blocks; intervals are 95% normal approximations.")
table(D, s, x, y, w,
      ["model", "peak layer", "Value net", "Key net", "K+V net", "Key ÷ Value"],
      [[n, f"L{S['step3'][f]['peak']}",
        f"{S['step3'][f]['value'][0]:.3f} ± {S['step3'][f]['value'][1]:.3f}",
        f"{S['step3'][f]['key'][0]:.3f} ± {S['step3'][f]['key'][1]:.3f}",
        f"{S['step3'][f]['key_value'][0]:.3f} ± {S['step3'][f]['key_value'][1]:.3f}",
        f"{abs(S['step3'][f]['key'][0]/S['step3'][f]['value'][0]):.0%}"] for f, n in MODELS],
      [26, 14, 20, 18, 18, 14], size=10.5, fit_h=h, tag="s3-net")
y2 = BODY_TOP + 380 + GAP
lx, lw, rx, rw = split(1080)
x, y, w, h = D.panel(s, lx, y2, lw, BODY_BOT - y2, "the careful version of the Key claim",
                     "“Zero” would be wrong; “negligible” is what the numbers support.")
D.body(s, x, y, w, "The Key intervals do not contain zero in three of the four models. The honest statement is that "
       "the Key net effect is 2–11% of the Value net effect — small enough to be practically irrelevant, not "
       "small enough to call absent.", fit_h=h, tag="s3-key")
x, y, w, h = D.panel(s, rx, y2, rw, BODY_BOT - y2, "and the combined road",
                     "Editing both is worse than editing content alone.")
D.body(s, x, y, w, "K+V lands below Value alone in every model, which is consistent with the transplanted key "
       "carrying a rotation that belongs to another position.", fit_h=h, tag="s3-kv")

s = D.slide(C5, "Step 3 · Result — the layer profile, net of control",
            "The effect is not spread evenly across the stack; it concentrates in a band of late-middle layers.")
four_plots(D, s, "step3", "code_causality_{}.png", BODY_TOP, 350,
           "Net transfer rate by layer, Value and Key. This is the figure the paper's causality claim rests on.")
y2 = BODY_TOP + 350 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "split-half validation",
                     "Choosing the peak layer on half the blocks and measuring it on the other half — so the peak is not selected on the data it is reported from.")
table(D, s, x, y, w, ["model", "layer chosen on blocks 1–21", "net on held-out 22–42", "net at that layer, all 42"],
      [["Qwen2.5-Coder-3B", "L25", "0.261", "0.253"],
       ["DeepSeek-Coder-6.7B", "L20", "0.139", "0.125"],
       ["Llama-3.2-3B", "L15", "0.365", "0.358"],
       ["StableCode-3B", "L18", "0.193", "0.189"]],
      [30, 26, 24, 24], size=10.5, fit_h=h, tag="s3-split")

s = D.slide(C5, "Step 3 · What this step does not settle",
            "One direction, one signal, one instrument — and an instrument whose asymmetry is measured but not fully characterised.")
cw = (MR - ML - 3 * GAP) / 4
for i, (t, b) in enumerate([
        ("One direction", "camelCase-required only. Whether the same layers carry the signal in the other "
         "direction is not tested here."),
        ("Code, not instruction", "This step localises the code signal. Whether the instruction travels the "
         "same road is RQ3's question."),
        ("Instrument asymmetry", "Mean-pool substitution damages Keys and Values differently; the diagnostic "
         "step measures norm, not phase."),
        ("Layer, not head", "The peak is a layer. Which heads within it carry the signal is not resolved.")]):
    xx = ML + i * (cw + GAP)
    x, y, w, h = D.panel(s, xx, BODY_TOP, cw, 380, f"limit {i+1}")
    D.cardtitle(s, x, y, t, w=w)
    D.body(s, x, y + 34, w, b, fit_h=h - 42, tag=f"s3-lim{i}")
y2 = BODY_TOP + 380 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "the strongest thing this step licenses",
                     "Stated in the form the evidence supports, with the instrument caveat attached rather than dropped.")
D.body(s, x, y, w, "Under mean-pool substitution with paired controls, the notation signal from preceding code "
       "acts through the attended content at a narrow band of late-middle layers, and not through the attention "
       "share. The qualifier about the instrument is part of the claim, not a footnote to it.", fit_h=h, tag="s3-lic")

s = D.slide(C5, "Step 3 · Reading the result file",
            "Transfer rates live under per_layer, not under extra — a distinction that once produced an empty table.")
lx, lw, rx, rw = split(1120)
x, y, w, h = D.panel(s, lx, BODY_TOP, lw, 460, "results/step3_code-cause/<slug>.json")
code(D, s, x, y, w, h, [
    '"metrics": {',
    '  "per_layer": {',
    '    "27": {',
    '»      "value__recovery":     0.42,   # the headline number',
    '      "key__recovery":       0.02,',
    '      "key_value__recovery": 0.45,',
    '      "value__S_int": -1.24, … } },',
    '  "extra": {',
    '    "mode": "intervene_sweep",',
    '    "donor": "compliant",            # or unrelated_camel | unrelated_snake',
    '    "S_clean": 1.83, "S_base": -3.21,',
    '»    "n_substituted_tokens": 31,      # 0 would have raised',
    '    "skipped_names": [] } }'], size=9)
x, y, w, h = D.panel(s, rx, BODY_TOP, rw, 460, "two traps in this step's files")
D.deflist(s, x, y, w, [
    ("No gap field", "these 504 files predate it; compute it from S_clean and S_base — the value is identical"),
    ("No extra.recovery", "sweep runs store rates per layer only; extra.recovery exists on single-layer runs"),
    ("Always paired", "the reported number is a difference between two donors on the same block, never a raw rate")],
    fit_h=h, tag="s3-json")
y2 = BODY_TOP + 460 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "aggregation",
                     "Computes the paired net effect and its interval — this is the script the paper's numbers come from.")
code(D, s, x, y, w, h, ["python scripts/step3_net_effect.py results/step3_code-cause"], size=10)

# ═══════════════════════════════════ 05 · step 4 ═════════════════════════
s = D.slide(C5, "Step 4 · The question",
            "The code signal is content at a late layer. Does the instruction get read the same way — and is it even competitive with the code?")
lx, lw, rx, rw = split(920)
x, y, w, h = D.panel(s, lx, BODY_TOP, lw, 400, "what changes from step 2")
D.deflist(s, x, y, w, [
    ("Both directions", "camelCase-required and snake_case-required, 42 blocks each"),
    ("Eight spans", "the instruction's notation words separated by role"),
    ("All four models", "the same four as steps 2, 3 and 6"),
    ("Same read-out", "one query row, head-averaged, per token")], fit_h=h, tag="s4-q")
x, y, w, h = D.panel(s, rx, BODY_TOP, rw, 400, "what each outcome would have meant")
table(D, s, x, y, w, ["if we had seen", "the reading"],
      [["instruction attended far less than code", "the instruction is drowned out; an attention remedy is plausible"],
       ["instruction competitive per token", "the instruction is being read; the failure is downstream of looking"],
       ["result depends on the estimand", "the honest answer is that the comparison is estimand-dependent"]],
      [40, 60], size=10, fit_h=h, tag="s4-out")
y2 = BODY_TOP + 400 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "the third outcome is the one we got, and it is reported as such",
                     "Per token the instruction leads in most layers; by span sum it rarely does. Both are shown.")
D.body(s, x, y, w, "This is the clearest case in the study where the answer depends on how a span is summarised. "
       "Reporting only the estimand that favours our reading would be the easiest way to overstate the result, "
       "so both are plotted on the same axis with the crossover line drawn in.", fit_h=h, tag="s4-est")

s = D.slide(C5, "Step 4 · The role split, and why it was necessary",
            "The required notation word appears twice, in two different roles. Summing them builds a 2 : 1 advantage into the measurement.")
x, y, w, h = D.panel(s, ML, BODY_TOP, MR - ML, 440, "instruction anatomy and the eight span keys",
                     "The inflated keys are kept in the stored results for transparency, and interpreted nowhere.")
D.fit(s, A + "s4_split.png", x, y, w, h, align="left")
y2 = BODY_TOP + 440 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "the candidate list is a symmetric control",
                     "Both notation words appear there in identical roles, so any difference between them is not about which notation is required.")
D.body(s, x, y, w, "That symmetry is what makes the rule-sentence word interpretable: it is the only place where the "
       "two directions differ in what the text actually demands.", fit_h=h, tag="s4-sym")

s = D.slide(C5, "Step 4 · Result — the instruction, by role",
            "Five spans on one axis: the rule word, both candidate words, and both code groups.")
four_plots(D, s, "step4", "spans_{}.png", BODY_TOP, 400,
           "Attention per token by layer, averaged over both instruction directions.")
y2 = BODY_TOP + 400 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "peak layer of the rule word, per model",
                     "These land close to the causal peaks from step 3, which is the alignment the next slide examines.")
cw = (w - 3 * GAP) / 4
for i, (fam, name) in enumerate(MODELS):
    D.stat(s, x + i * (cw + GAP), y, cw, f"L{S['step4'][fam]['peak']}",
           f"{name} · of {S['step4'][fam]['n_layers']}", size=30, fit_h=h, tag="s4pk" + fam)

s = D.slide(C5, "Step 4 · Result — why the initial keys were replaced",
            "The same measurement, with the inflated key drawn beside the role-separated one.")
four_plots(D, s, "step4", "initial_vs_split_{}.png", BODY_TOP, 400,
           "Initial required-word key (2 occurrences) against the rule-sentence word alone (1 occurrence).")
y2 = BODY_TOP + 400 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "what the gap between the two curves is",
                     "It is not a finding about the model. It is the size of a measurement artefact we removed.")
D.body(s, x, y, w, "The distance between the two curves is entirely due to counting the candidate-list occurrence "
       "alongside the rule sentence. Any claim built on the higher curve would be measuring the prompt's "
       "construction rather than the model's behaviour.", fit_h=h, tag="s4-art")

s = D.slide(C5, "Step 4 · Result — instruction against code, both estimands",
            "The ratio of instruction to code attention, plotted on a log axis with the crossover marked.")
four_plots(D, s, "step4", "ratio_{}.png", BODY_TOP, 350,
           "Rule word divided by compliant code names. Above 1.0 the instruction leads.")
y2 = BODY_TOP + 350 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "how many layers put the instruction ahead",
                     "The two estimands straddle 1.0 — which is why neither is presented as the answer on its own.")
table(D, s, x, y, w, ["model", "layers", "instruction ahead · per token", "instruction ahead · span sum"],
      [[n, S["step4"][f]["n_layers"], f"{S['step4'][f]['above_per_token']} layers",
        f"{S['step4'][f]['above_sum']} layers"] for f, n in MODELS],
      [30, 14, 28, 28], size=10.5, fit_h=h, tag="s4-ratio")

s = D.slide(C5, "Step 4 · Result — observation peaks against causal peaks",
            "Where the instruction is most attended, set beside where the code signal was shown to act.")
four_plots(D, s, "step4", "layer_alignment_{}.png", BODY_TOP, 400,
           "Observation peak (this step) and causal peak (step 3) on one axis.")
y2 = BODY_TOP + 400 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "what alignment would and would not show",
                     "Alignment is suggestive, not causal. The instruction's own causal test is step 5, which is not in this deck yet.")
D.body(s, x, y, w, "If the two peaks coincided, that would be consistent with a single late-layer mechanism handling "
       "both signals. It would not establish it: only an intervention on the instruction can do that, and that "
       "is precisely the step still waiting on its control sweep.", fit_h=h, tag="s4-align")

s = D.slide(C5, "Step 4 · Determinism, and what is not settled",
            "The overlap with step 2 gives a free reproducibility check; the rest of the limits carry forward.")
lx, lw, rx, rw = split(900)
x, y, w, h = D.panel(s, lx, BODY_TOP, lw, 400, "the overlap check",
                     "Step 2's 168 conditions are a subset of step 4's.")
kv_rows(D, s, x, y, w, [
    ("Overlapping conditions", "168"),
    ("Values compared across all layers", "80,640"),
    ("Mismatches", "0"),
    ("Maximum absolute difference", "0.000e+00")], pitch=42)
x, y, w, h = D.panel(s, rx, BODY_TOP, rw, 400, "limits")
D.deflist(s, x, y, w, [
    ("Not causal", "attention is observed; the instruction's causal test is step 5"),
    ("Estimand-dependent", "the instruction-versus-code comparison changes with the summary used"),
    ("Head-averaged", "as in step 2, claims are at the layer level"),
    ("One phrasing", "positive form, weak strength — other phrasings are untested")],
    fit_h=h, tag="s4-lim")
y2 = BODY_TOP + 400 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "what step 4 does license",
                     "Enough to rule out the strongest form of the attention-deficit account, and not more.")
D.body(s, x, y, w, "The instruction is not being ignored: per token it is among the most attended material in the "
       "prompt across most layers, in both directions. Whether that attention is what drives behaviour is a "
       "separate question this step deliberately does not answer.", fit_h=h, tag="s4-lic")

s = D.slide(C5, "Step 4 · Reading the result file",
            "Eight span keys, three of which are role-separated and two of which are kept only for continuity.")
lx, lw, rx, rw = split(1180)
x, y, w, h = D.panel(s, lx, BODY_TOP, lw, 470, "results/step4_instr-observe/<slug>.json")
code(D, s, x, y, w, h, [
    '"per_layer": { "27": {',
    '»  "instr_rule_word__attention_weight":  0.0121,   # interpret this one',
    '  "instr_cand_camel__attention_weight": 0.0043,   # symmetric control',
    '  "instr_cand_snake__attention_weight": 0.0041,   # symmetric control',
    '  "instr_target_word__attention_weight":0.0164,   # initial key, inflated',
    '  "instr_viol_word__attention_weight":  0.0041,   # initial key',
    '  "instruction__attention_weight":      0.0312,',
    '  "code_camel__attention_weight":       0.0184,',
    '  "code_snake__attention_weight":       0.0231 } },',
    '"extra": { "span_token_counts": {…, "instr_rule_word": 2,',
    '           "instr_cand_camel": 2, "instr_cand_snake": 2}, … }'], size=9)
x, y, w, h = D.panel(s, rx, BODY_TOP, rw, 470, "reading order")
D.deflist(s, x, y, w, [
    ("First", "span_token_counts — if two spans differ in token count, do not compare their sums"),
    ("Use", "instr_rule_word; instr_target_word counts two occurrences and is inflated by construction"),
    ("Careful", "the candidate-list keys are role-symmetric, but token counts match only on Qwen and Llama, so read them per token")],
    fit_h=h, tag="s4-json")
y2 = BODY_TOP + 470 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "aggregation",
                     "The same script as step 2, pointed at this step's folder — one estimand comparison, one code path.")
code(D, s, x, y, w, h, ["python scripts/observe_per_token.py results/step4_instr-observe"], size=10)

# ═══════════════════════════════════ 05 · diag ═══════════════════════════
s = D.slide(C5, "Diagnostic · Is the instrument fair to both roads?",
            "If mean-pool substitution damages Keys more than Values, the central result could be an artefact of the method.")
x, y, w, h = D.panel(s, ML, BODY_TOP, MR - ML, 400, "the premise being checked",
                     "This is a check on the instrument, run on the same conditions as the experiment it checks.")
D.fit(s, A + "d_design.png", x, y, w, h, align="left")
y2 = BODY_TOP + 400 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "why this is not optional",
                     "A null result on the Key road is only informative if the Key road was given a fair chance.")
D.body(s, x, y, w, "The whole argument turns on Value moving behaviour while Key does not. If the substitution "
       "systematically destroyed Keys, that asymmetry would be manufactured by us rather than found in the model.",
       fit_h=h, tag="diag-why")

s = D.slide(C5, "Diagnostic · What is measured",
            "Three quantities, no scoring. The run is cheap because nothing is generated and nothing is compared to a baseline.")
lx, lw, rx, rw = split(1100)
x, y, w, h = D.panel(s, lx, BODY_TOP, lw, 420, "kv_diagnose mode")
code(D, s, x, y, w, h, [
    "# no score is computed — this mode only inspects the substitution",
    "k_new = k[:, :, donor_pos, :].mean(dim=2)",
    "v_new = v[:, :, donor_pos, :].mean(dim=2)",
    "",
    "»key_shrink   = k_new.norm() / k[:, :, target_pos, :].norm()",
    "»value_shrink = v_new.norm() / v[:, :, target_pos, :].norm()",
    "key_minus_value = key_shrink - value_shrink",
    "",
    "position_offsets = [abs(d - t) for d, t in zip(donor_pos, target_pos)]"], size=9)
x, y, w, h = D.panel(s, rx, BODY_TOP, rw, 420, "eq. 15 — the shrink ratios",
                     "A ratio near 1 means the averaged vector kept its length.")
D.equation(s, A + "eq_shrink.png", x, y + 10, h=76)
D.body(s, x, y + 110, w, "If the Key ratio were far below the Value ratio, the Key intervention would be a weaker "
       "manipulation and its null result uninformative.", fit_h=h - 120, tag="diag-eq")
y2 = BODY_TOP + 420 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "72 conditions, six blocks, four models",
                     "Deliberately small: this is a property of the substitution, not of the dataset.")
D.body(s, x, y, w, "Because no score is computed, the run is a fraction of the cost of a causal sweep, which is why "
       "it covers six blocks rather than forty-two.", fit_h=h, tag="diag-n")

s = D.slide(C5, "Diagnostic · Result — Keys shrink less, not more",
            "The direction of the asymmetry is the opposite of the one that would undermine the central claim.")
lx, lw, rx, rw = split(1080)
x, y, w, h = D.panel(s, lx, BODY_TOP, lw, 430, "norm retained after substitution, by layer",
                     "Key above Value in every model at every layer.")
D.plot(s, f"{FIG}/diag/figures/kv_phase_shrink.png", x, y, w, h)
x, y, w, h = D.panel(s, rx, BODY_TOP, rw, 430, "averaged over layers, blocks and models")
table(D, s, x, y, w, ["model", "Key retained", "Value retained", "Key − Value"],
      [[n, f"{S['diag'][f]['key_shrink']:.3f}", f"{S['diag'][f]['value_shrink']:.3f}",
        f"+{S['diag'][f]['key_minus_value']:.3f}"] for f, n in MODELS],
      [34, 22, 22, 22], size=10.5, fit_h=h, tag="diag-tab")
y2 = BODY_TOP + 430 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "what this rejects",
                     "One hypothesis, cleanly. Not the whole family of instrument objections.")
D.body(s, x, y, w, "The norm-collapse account of the Key null is rejected: Keys retain more of their length than "
       "Values do, so the Key manipulation is not the weaker one by this measure.", fit_h=h, tag="diag-rej")

s = D.slide(C5, "Diagnostic · What remains unproven",
            "Norm is not phase. The measurement rules out one mechanism and is silent about another.")
lx, lw, rx, rw = split(960)
x, y, w, h = D.panel(s, lx, BODY_TOP, lw, 420, "the gap we do not close")
D.body(s, x, y, w, "Rotary embeddings encode position as a rotation. A substituted key can retain its full length "
       "and still point in a direction that belongs to a different position. This diagnostic measures length, "
       "not angle, so a phase-mismatch account of the Key null survives it.", fit_h=h, tag="diag-gap")
x, y, w, h = D.panel(s, rx, BODY_TOP, rw, 420, "the one piece of positional evidence we do have",
                     "Where the donor and target positions coincide, a phase mismatch cannot arise.")
table(D, s, x, y, w, ["model", "mean position offset"],
      [[n, f"{S['diag'][f]['pos_offset']:.2f}"] for f, n in MODELS],
      [56, 44], size=10.5, fit_h=h, tag="diag-pos")
y2 = BODY_TOP + 420 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "how this is carried into the claims",
                     "The qualifier travels with the Key result wherever it appears, including in the paper.")
D.body(s, x, y, w, "On two models the donor and target sit at the same positions on average, so no rotation "
       "mismatch is introduced there and the Key null still holds. On the other two an offset exists and the "
       "phase account cannot be excluded. We state the Key result with that qualifier rather than without it.",
       fit_h=h, tag="diag-carry")

s = D.slide(C5, "Diagnostic · Reading the result file",
            "No scores, no rates — two shrink ratios, two cosines, and the positional evidence.")
lx, lw, rx, rw = split(1140)
x, y, w, h = D.panel(s, lx, BODY_TOP, lw, 460, "results/diag_kv-phase/<slug>.json")
code(D, s, x, y, w, h, [
    '"per_layer": { "17": {',
    '    "key_shrink":         0.9512,   # 1.0 = no length lost',
    '    "value_shrink":       0.8417,   # the no-rotation baseline',
    '»    "key_minus_value":    0.1095,   # positive = Key lost LESS',
    '    "key_piece_cosine":   0.83,     # how aligned the donor pieces were',
    '    "value_piece_cosine": 0.61 } },',
    '"extra": {',
    '    "mode": "kv_diagnose", "donor": "compliant",',
    '»    "position_offset_abs_mean": 8.4,   # the only phase proxy we have',
    '    "donor_piece_counts": [2, 2, 3, …],',
    '    "target_piece_counts": [3, 3, 2, …],',
    '    "n_substituted_tokens": 31 }'], size=9)
x, y, w, h = D.panel(s, rx, BODY_TOP, rw, 460, "how to read it")
D.deflist(s, x, y, w, [
    ("Headline", "key_minus_value — positive means the Key kept more of its length, which rejects the norm-collapse account"),
    ("Never alone", "position_offset_abs_mean must be read beside it; a key can keep its length and still carry the wrong rotation"),
    ("No score", "this mode computes no transfer rate, which is why the run is cheap")],
    fit_h=h, tag="diag-json")
y2 = BODY_TOP + 460 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "aggregation",
                     "Averages the shrink ratios over layers and blocks, and reports the offset alongside.")
code(D, s, x, y, w, h, ["python scripts/diag_kv_phase_summary.py results/diag_kv-phase"], size=10)

# ═══════════════════════════════════ 05 · step 6 ═════════════════════════
s = D.slide(C5, "Step 6 · The prescription, tested rather than proposed",
            "RQ1–RQ3 say the handle is content at a late layer. If that reading is right, pushing content should work and raising attention should not.")
x, y, w, h = D.panel(s, ML, BODY_TOP, MR - ML, 440, "two prescriptions, opposite predictions",
                     "Both are measured on the same conditions, the same seed and the same four models.")
D.fit(s, A + "s6_two.png", x, y, w, h, align="left")
y2 = BODY_TOP + 440 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "what would have falsified the diagnosis",
                     "The prediction was recorded before the sweep and is stored in the result files.")
D.body(s, x, y, w, "If attention reweighting had restored compliance, the conclusion of steps 2 and 3 would be wrong "
       "regardless of how clean those measurements were. Step 6 is the place where the diagnosis is exposed to "
       "that risk rather than protected from it.", fit_h=h, tag="s6-risk")

s = D.slide(C5, "Step 6 · The two interventions, in code",
            "One adds a direction to the residual stream; the other rescales attention onto a span.")
lx, lw, rx, rw = split(1140)
x, y, w, h = D.panel(s, lx, BODY_TOP, lw, 460, "residual-stream steering")
code(D, s, x, y, w, h, [
    '# direction built from contrasting contexts, not hand-specified',
    'd = mean(h_camel_contexts) - mean(h_snake_contexts)      # at steer_layer',
    '',
    'def hook(module, args, output):',
    '»    return output[0] + alpha * d, *output[1:]           # block output',
    '',
    'model.model.layers[layer].register_forward_hook(hook)',
    '',
    '# alpha is swept over a fixed grid: 0.5, 1, 2, 4'], size=9)
x, y, w, h = D.panel(s, rx, BODY_TOP, rw, 460, "Spotlight-style attention reweighting")
code(D, s, x, y, w, h, [
    '# raise the span mass toward a target, with the undershoot form',
    'psi_after = psi_target / (1 + psi_target - psi_before)',
    '',
    'w[:, :, :, span] *= psi_after / psi_before',
    'w = w / w.sum(dim=-1, keepdim=True)          # renormalise the row',
    '',
    '»record["attn_span_before"], record["attn_span_after"] = …',
    '»record["attn_calls"] = n                    # 0 would have raised'], size=9)
y2 = BODY_TOP + 460 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "eq. 16 — the undershoot form",
                     "Stored before-and-after span masses confirm the reweighting actually took effect in every run.")
D.equation(s, A + "eq_spotlight.png", x, y + 2, h=60)

s = D.slide(C5, "Step 6 · Result — strength sweep",
            "Recovery against steering strength, with the attention-reweighting result on the same axis.")
four_plots(D, s, "step6", "strength_{}.png", BODY_TOP, 350,
           "Transfer rate against intervention strength. Undecidable conditions excluded.")
y2 = BODY_TOP + 350 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "best result each method reached, per model",
                     "Value steering exceeds 1.0 on three models, meaning it pushed preference past the fully compliant context.")
table(D, s, x, y, w, ["model", "steering · best α", "steering · recovery", "Spotlight · best ψ", "Spotlight · recovery"],
      [[n, f"α = {S['step6'][f]['value_add_best'][0]:g}", f"{S['step6'][f]['value_add_best'][1]:.2f}",
        f"ψ = {S['step6'][f]['spotlight_best'][0]:g}", f"{S['step6'][f]['spotlight_best'][1]:.2f}"]
       for f, n in MODELS],
      [28, 18, 18, 18, 18], size=10.5, fit_h=h, tag="s6-sweep")

s = D.slide(C5, "Step 6 · Result — the headline comparison",
            "The prescription implied by the diagnosis works. The prescription implied by the rival account does not.")
lx, lw, rx, rw = split(1140)
x, y, w, h = D.panel(s, lx, BODY_TOP, lw, 440, "content steering against attention reweighting",
                     "Same conditions, same models, same measurement.")
D.plot(s, f"{FIG}/step6/figures/explain_headline.png", x, y, w, h)
x, y, w, h = D.panel(s, rx, BODY_TOP, rw, 440, "not a null of convenience")
D.deflist(s, x, y, w, [
    ("It fired", "the stored before-and-after span masses show the reweighting took effect every time"),
    ("It was swept", "two target levels, on the same grid decided in advance"),
    ("It is not zero", "Llama shows a positive but small recovery; DeepSeek is negative")],
    fit_h=h, tag="s6-null")
y2 = BODY_TOP + 440 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "the claim this licenses, stated at its true strength",
                     "A prescription that beats the rival on this instrument, not a general method claim.")
D.body(s, x, y, w, "On these four models, on this dataset, and under this measurement, adding a contrast direction at a "
       "late layer restores notation preference while raising the instruction's attention mass does not. That is "
       "a strong result about the diagnosis and a weak one about deployment.", fit_h=h, tag="s6-lic")

s = D.slide(C5, "Step 6 · Result — does the score survive generation?",
            "A preference score that rises while the emitted names fall apart is not a remedy.")
lx, lw, rx, rw = split(1140)
x, y, w, h = D.panel(s, lx, BODY_TOP, lw, 430, "score against actual generated names",
                     "The two measurements come apart, and the gap is the point of the slide.")
D.plot(s, f"{FIG}/step6/figures/explain_score_vs_real.png", x, y, w, h)
x, y, w, h = D.panel(s, rx, BODY_TOP, rw, 430, "generated names under steering")
table(D, s, x, y, w, ["model", "well-formed", "compliant"],
      [[n, f"{S['step6_gen'][f]['value_add'][0]:.0%}", f"{S['step6_gen'][f]['value_add'][1]:.0%}"]
       for f, n in MODELS],
      [46, 27, 27], size=10.5, fit_h=h, tag="s6-gen")
y2 = BODY_TOP + 430 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "the honest reading",
                     "Reported together, always — the score alone would overstate what steering achieves.")
D.body(s, x, y, w, "Steering moves the preference score decisively on all four models, but the generated names follow "
       "only partly, and on StableCode barely at all while name quality degrades. The mechanism claim survives; "
       "a claim that this is a deployable fix does not.", fit_h=h, tag="s6-honest")

s = D.slide(C5, "Step 6 · Result — name health under steering",
            "Whether the model still writes a usable identifier, plotted against the strength that moved the score.")
lx, lw, rx, rw = split(1180)
x, y, w, h = D.panel(s, lx, BODY_TOP, lw, 430, "well-formedness and compliance against strength",
                     "Dashed lines mark the unsteered baseline for each model.")
D.plot(s, f"{FIG}/step6/figures/name_health.png", x, y, w, h)
x, y, w, h = D.panel(s, rx, BODY_TOP, rw, 430, "what degrades, and where")
D.deflist(s, x, y, w, [
    ("Baseline", "every unsteered generation produces a well-formed name"),
    ("Under steering", "well-formedness falls to 58% on StableCode and 79% on DeepSeek"),
    ("Trade-off", "the strengths that maximise the score are not the ones that keep names intact")],
    fit_h=h, tag="s6-health")
y2 = BODY_TOP + 430 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "why this slide is in the deck at all",
                     "It is the result that most weakens our own prescription, and it comes from the same runs as the headline.")
D.body(s, x, y, w, "Leaving it out would have been easy and would have made the story cleaner. It is here because "
       "the score-only version of this result is the kind of claim this study was reorganised to stop making.",
       fit_h=h, tag="s6-why")

s = D.slide(C5, "Step 6 · Result — is the effect layer-local?",
            "The direction is built at the causal peak, then added at other layers to see whether the location matters.")
lx, lw, rx, rw = split(1000)
x, y, w, h = D.panel(s, lx, BODY_TOP, lw, 430, "what the cross-layer runs vary")
kv_rows(D, s, x, y, w, [
    ("Direction", "always built at the step-3 causal peak"),
    ("Injection layer", "swept across the stack, including early layers"),
    ("Strength", "held at a single value so only location varies"),
    ("Conditions", "336 runs, the same blocks as the main sweep")], pitch=44)
D._tb(s, x, y + 200, w, 80, "An effect that survived injection anywhere would mean the layer claim from step 3 "
      "was not carrying real information.", 10.5, ASH)
x, y, w, h = D.panel(s, rx, BODY_TOP, rw, 430, "what the records store",
                     "steer_from_layer records where the direction came from; layer records where it was applied.")
code(D, s, x, y, w, h, [
    '"extra": {',
    '  "method": "value_add", "strength": 1.0,',
    '»  "steer_from_layer": 20,        # where the direction was built',
    '»  "layer": 5,                    # where it was injected',
    '  "steer_vector": {"n_samples": 8, "n_skipped": 0,',
    '                   "vec_norm": 30.48},',
    '  "recovery": -0.153,            # injecting early hurts',
    '  "hook_calls": 2 }'], size=9)
y2 = BODY_TOP + 430 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "the reading",
                     "Consistent with step 3's layer profile, measured on a different intervention.")
D.body(s, x, y, w, "Injecting the same direction at an early layer does not restore preference and can move it the "
       "wrong way. The effect is tied to where it is applied, which is what a layer-local mechanism predicts.",
       fit_h=h, tag="s6-cross")

s = D.slide(C5, "Step 6 · Reading the result files",
            "Two folders, two record shapes. The score record and the generation record must be read together.")
lx, lw, rx, rw = split(1000)
x, y, w, h = D.panel(s, lx, BODY_TOP, lw, 470, "the score record")
code(D, s, x, y, w, h, [
    '# results/step6_steer/<slug>.json',
    '"extra": {',
    '  "mode": "steer",',
    '  "method": "attn_amplify",   # none | value_add | attn_amplify',
    '  "kind": "spotlight", "span": "rule_word",',
    '  "psi_target": 0.1,',
    '  "S_clean": 3.929, "S_base": -1.556, "S_int": -3.327,',
    '»  "recovery": -0.323,         # the headline number',
    '  "gap": 5.485, "undecidable": false,',
    '»  "attn_span_before": 0.00080,   # proof it fired',
    '»  "attn_span_after":  0.09053,',
    '  "attn_calls": 64, "n_span_tokens": 3 }'], size=9.5)
x, y, w, h = D.panel(s, rx, BODY_TOP, rw, 470, "the generation record")
code(D, s, x, y, w, h, [
    '# results/step6_steer-generate/<slug>.json',
    '"metrics": {',
    '  "compliance_rate": 0.0,',
    '  "extra": {',
    '    "mode": "steer_generate",',
    '    "method": "value_add",',
    '    "strength": 4, "layer": 25,',
    '    "generated_text": "def RemoveDuplicates(x): …",',
    '    "name": "RemoveDuplicates",',
    '»    "notation": "other",     # not camel',
    '»    "compliant": false,',
    '    "name_ok": true, "name_length": 16 } }'], size=9.5)
y2 = BODY_TOP + 470 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "the rule for this step",
                     "Reading recovery without compliant is how a score-only success story gets written.")
D.body(s, x, y, w, "Always read recovery and compliant together. The example above is a run where the preference "
       "score improved and the emitted name was neither notation.", fit_h=h, tag="s6-json")

# ═══════════════════════════════════════════════════ 06 · Synthesis ══════
s = D.slide(C6, "The answers, in the order they were asked",
            "Three questions, three answers, and the strength each one is entitled to.")
x, y, w, h = D.panel(s, ML, BODY_TOP, MR - ML, BODY_BOT - BODY_TOP, "what the study establishes",
                     "Each row's qualifier is part of the claim, not a hedge appended to it.")
table(D, s, x, y, w, ["question", "answer", "on what evidence", "qualifier"],
      [["RQ1  does violating context lower compliance",
        "yes, sharply",
        "2,184 generations, both directions, 42 blocks",
        "two code-specialised models, Python, first function only"],
       ["RQ2  is the code signal content or attention",
        "content — the Value road",
        "504 sweeps with paired controls; net effect 0.13–0.36 on Value, 2–11% of that on Key",
        "under mean-pool substitution; phase not measured"],
       ["RQ2  where in the network",
        "a narrow band of late-middle layers",
        "layer sweeps plus split-half validation on held-out blocks",
        "layer level, head-averaged"],
       ["RQ3  is the instruction read the same way",
        "it is read, and competitively per token",
        "336 observations, both directions, role-separated spans",
        "observational only — the causal half is step 5, still pending"]],
      [24, 18, 33, 25], size=10.5, fit_h=h, tag="answers")

s = D.slide(C6, "Why raising attention does not fix it",
            "Three independent measurements point the same way, and the prescription test closes the argument.")
x, y, w, h = D.panel(s, ML, BODY_TOP, MR - ML, 360, "the chain")
cw = (w - 3 * GAP) / 4
for i, (t, b) in enumerate([
        ("The model already looks", "Violating names receive more attention per token than compliant ones, "
         "in every model and at every layer band."),
        ("Attention share carries little", "Overwriting the Key road moves behaviour by 2–11% of what the "
         "Value road moves, net of control."),
        ("Content carries it", "Overwriting attended content at one late layer restores preference by "
         "0.13 to 0.36, net of control."),
        ("And the remedy follows", "Adding a contrast direction at that layer restores preference; "
         "reweighting attention onto the instruction does not.")]):
    xx = x + i * (cw + GAP)
    D.cardtitle(s, xx, y, t, w=cw)
    D.body(s, xx, y + 34, cw, b, fit_h=h - 42, tag=f"chain{i}")
y2 = BODY_TOP + 360 + GAP
lx, lw, rx, rw = split(1080)
x, y, w, h = D.panel(s, lx, y2, lw, BODY_BOT - y2, "what this does not say about prior work",
                     "The disagreement is about the locus of the failure, not about the quality of the earlier method.")
D.body(s, x, y, w, "Attention reweighting may well help on tasks where the model genuinely under-attends. Our claim "
       "is narrower: on this failure, on these models, under this measurement, it does not, because "
       "under-attention is not what is wrong.", fit_h=h, tag="prior")
x, y, w, h = D.panel(s, rx, y2, rw, BODY_BOT - y2, "the one link still open",
                     "The instruction's causal test is specified and half-run.")
D.body(s, x, y, w, "Everything about the instruction in this deck is observational. The intervention exists but its "
       "layer-wise control does not yet, so no causal claim about the instruction is made here.",
       fit_h=h, tag="open")

s = D.slide(C6, "How claims are worded, and why",
            "The study has a fixed vocabulary for claim strength. Wording drift is the failure mode the audit found most often.")
x, y, w, h = D.panel(s, ML, BODY_TOP, MR - ML, BODY_BOT - BODY_TOP, "forbidden phrasing and its replacement",
                     "Each row on the left is a sentence that appeared in an earlier draft of this project's documentation.")
table(D, s, x, y, w, ["what we do not write", "what we write instead", "why"],
      [["“the Key path carries no information”",
        "“the Key net effect is 2–11% of the Value net effect”",
        "the intervals do not contain zero in three of four models"],
       ["“av_norm is the contribution”",
        "“av_norm is an upper bound on the contribution”",
        "it ignores cancellation and precedes the output projection"],
       ["“RoPE concerns are rejected”",
        "“the norm-collapse hypothesis is rejected”",
        "phase was never measured; only length was"],
       ["“steering fixes the problem”",
        "“steering restores preference; generated names follow only partly”",
        "well-formedness falls to 58% on one model"],
       ["“the model ignores the instruction”",
        "“per token the instruction is among the most attended spans”",
        "the opposite of the measurement, under either estimand"]],
      [30, 38, 32], size=10.5, fit_h=h, tag="wording")

s = D.slide(C6, "Limitations, collected",
            "Gathered in one place rather than distributed where they are least visible.")
cw = (MR - ML - 2 * GAP) / 3
groups = [("coverage", [
    ("Language", "Python only; zero JavaScript conditions were ever run"),
    ("Models", "four, three of them code-specialised and all under 7B"),
    ("Context", "synthetic, one name pool, one function shape"),
    ("Prompt", "one phrasing — positive form, weak strength")]),
    ("measurement", [
    ("Preference", "teacher-forced score, not behaviour, except in steps 1 and 6"),
    ("Heads", "averaged; a single specialised head would be invisible"),
    ("Estimand", "per token throughout; the span-sum reading disagrees in step 4"),
    ("Position", "first function only; amplification across functions untested")]),
    ("instrument", [
    ("Substitution", "mean-pool destroys the original as it inserts the new"),
    ("Phase", "measured only through a positional-offset proxy"),
    ("Steering", "residual-stream, so not confined to the Value road"),
    ("Step 5", "instruction causality has no layer-wise control yet")])]
for i, (eb, items) in enumerate(groups):
    xx = ML + i * (cw + GAP)
    x, y, w, h = D.panel(s, xx, BODY_TOP, cw, 430, eb)
    D.deflist(s, x, y, w, items, fit_h=h, tag=f"lim{i}")
y2 = BODY_TOP + 430 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "none of these is repaired by argument",
                     "Where a limit is not covered by a later step, it is carried into the paper as a stated limitation.")
D.body(s, x, y, w, "The list is longer than it would be if limits were only mentioned where they were discovered. "
       "That is deliberate: a reader deciding how far to trust the headline needs them in one place.",
       fit_h=h, tag="lim-note")

s = D.slide(C6, "What is next — step 5's control sweep",
            "The one experiment this deck is missing, why its absence is a coverage gap rather than an error, and what it costs.")
x, y, w, h = D.panel(s, ML, BODY_TOP, MR - ML, 400, "the state of step 5",
                     "The treatment swept every layer; the controls ran at one layer only, so no layer-wise net curve can be drawn.")
D.fit(s, A + "z_next.png", x, y, w, h, align="left")
y2 = BODY_TOP + 400 + GAP
lx, lw, rx, rw = split(1080)
x, y, w, h = D.panel(s, lx, y2, lw, BODY_BOT - y2, "what the sweep would add",
                     "A step-5 curve directly comparable to step 3's, and a check on the peak-layer assumption.")
D.body(s, x, y, w, "672 runs: four models, 42 blocks, two directions, two control donors, each sweeping every "
       "layer instead of one. Roughly two GPU hours.", fit_h=h, tag="next-cost")
x, y, w, h = D.panel(s, rx, y2, rw, BODY_BOT - y2, "what it cannot change",
                     "The reported step-5 numbers do not depend on the missing layers.")
D.body(s, x, y, w, "If the net peak turns out to sit elsewhere, the effect size grows rather than shrinks — the "
       "direction and sign of the result are not at stake.", fit_h=h, tag="next-safe")

s = D.slide(C6, "The through-line, once more",
            "From a symptom anyone can reproduce to a handle that can be tested — and to the one link still open.")
x, y, w, h = D.panel(s, ML, BODY_TOP, MR - ML, 380, "five findings, each predicting the next experiment",
                     "The dashed link at the right is step 5: specified, half-run, and not claimed.")
D.fit(s, A + "z_line.png", x, y, w, h, align="left")
y2 = BODY_TOP + 380 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "the study in four numbers",
                     "Every one recomputed from the stored result files for this deck, not transcribed from notes.")
cw = (w - 3 * GAP) / 4
for i, (v, lab) in enumerate([
        ("7,212", "conditions analysed"),
        ("0.13–0.36", "Value net effect at peak"),
        ("2–11%", "Key effect, as a share of Value"),
        ("34–49%", "of the raw effect was bubble")]):
    D.stat(s, x + i * (cw + GAP), y, cw, v, lab, size=30, fit_h=h, tag=f"final{i}")

s = D.slide(C6, "Closing",
            "What to take away, and what to be sceptical about.")
lx, lw, rx, rw = split(1080)
x, y, w, h = D.panel(s, lx, BODY_TOP, lw, 430, "take away")
D.deflist(s, x, y, w, [
    ("The diagnosis", "the failure is not inattention; the model looks at the violating names more, not less"),
    ("The locus", "the notation signal acts through attended content at a narrow band of late-middle layers"),
    ("The handle", "pushing content at that band restores preference where raising attention does not"),
    ("The discipline", "every effect is reported net of a control, because overwriting alone moves the score by a third to a half")],
    fit_h=h, tag="take")
x, y, w, h = D.panel(s, rx, BODY_TOP, rw, 430, "be sceptical about")
D.deflist(s, x, y, w, [
    ("Generality", "four models under 7B, Python, one synthetic context, one prompt phrasing"),
    ("Deployability", "steering moves the score decisively and the generated names only partly"),
    ("The instrument", "mean-pool substitution is not neutral; its asymmetry is measured by norm, not by phase"),
    ("The instruction", "everything about it here is observational; its causal test is not yet complete")],
    fit_h=h, tag="scept")
y2 = BODY_TOP + 430 + GAP
x, y, w, h = D.panel(s, ML, y2, MR - ML, BODY_BOT - y2, "where the material lives",
                     "Figures and aggregates in this deck are produced by scripts that read the stored result files; none was drawn or transcribed by hand.")
kv_rows(D, s, x, y, w // 2 - 20, [
    ("Plan", "docs/연구계획_통일재실험.md"),
    ("Per step", "docs/<step>/{방법론, code, results}.md")], pitch=36)
kv_rows(D, s, x + w // 2, y, w // 2, [
    ("Results", "results/<step>/<slug>.json — immutable"),
    ("Scripts", "scripts/ — every figure and every aggregate")], pitch=36)
