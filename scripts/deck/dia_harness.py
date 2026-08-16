"""결함검증 · 하네스 도해."""
from kit import *

# v1 폐기 타임라인
d = Dia(1680, 300)
stages = [("Pilot", "one model, ad-hoc scripts", "archive_v1"),
          ("First expansion", "per-step forks of the harness", "archive_v1"),
          ("Audit", "every claim re-checked against results", None),
          ("Unified re-run", "single engine, one results convention", "current")]
w = 390
for i, (t, s, tag) in enumerate(stages):
    x = i * (w + 40)
    d.box(x, 40, w, 130, None, ec=BRAND if i == 3 else HAIR)
    d.text(x + 24, 74, t, fs=11, color=WHITE)
    d.text(x + 24, 106, s, fs=9, color=ASH)
    if tag: d.text(x + 24, 142, tag, fs=8.5, color=BRAND if i == 3 else MUTE, mono=True)
    if i: d.arrow((x - 40, 105), (x - 4, 105))
d.text(0, 222, "The forks were the failure: each step drifted its own harness, so no branch could reproduce the whole study.", fs=9.5, color=ASH)
d.text(0, 250, "Archived runs are kept read-only as evidence of what was discarded — they are never cited as results.", fs=9.5, color=ASH)
d.text(0, 284, "Nothing from archive_v1 appears in any number in this deck.", fs=9.5, color=BRAND)
d.save("h_timeline")

# 조용한 실패 → 예외
d = Dia(1000, 400)
d.eyebrow(20, 24, "the failure mode this codebase is built against")
d.box(20, 46, 460, 120, None)
d.text(44, 78, "Silent failure", fs=10.5, color=WHITE)
d.text(44, 110, "Zero substitution sites, an unsupported", fs=9, color=ASH)
d.text(44, 130, "intervention kind, a span that was never", fs=9, color=ASH)
d.text(44, 150, "found — all return a number anyway.", fs=9, color=ASH)
d.arrow((490, 106), (540, 106))
d.box(552, 46, 428, 120, None, ec=BRAND)
d.text(576, 78, "Raise instead", fs=10.5, color=WHITE)
d.text(576, 110, "“No effect” and “the experiment never", fs=9, color=ASH)
d.text(576, 130, "ran” must not be the same stored value,", fs=9, color=ASH)
d.text(576, 150, "or the result cannot expose its own bug.", fs=9, color=ASH)
d.line(20, 204, 980, 204, color=HAIR)
guards = [("zero substitution sites", "raise"), ("unknown intervention kind", "raise"),
          ("span not located in prompt", "raise"), ("layer list where one layer is expected", "raise"),
          ("donor and target lengths disagree", "mean-pool, no skip"), ("undecidable denominator", "flag & exclude")]
for i, (g, act) in enumerate(guards):
    y = 240 + (i % 3) * 36; x = 20 + (i // 3) * 490
    d.text(x, y, "—", fs=9, color=MUTE)
    d.text(x + 22, y, g, fs=9, color=ASH)
    d.text(x + 330, y, act, fs=8.5, color=BRAND if act == "raise" else MUTE, mono=True)
d.save("h_silent")

# 층 목록 버그
d = Dia(1000, 380)
d.eyebrow(20, 24, "a concrete silent failure, found in our own results")
d.box(20, 46, 460, 150, None)
d.text(44, 78, "What the condition said", fs=10, color=WHITE)
d.text(44, 114, "layers = [14, 15, 16, 17, 18, 19, 20]", fs=9.5, color=ASH, mono=True)
d.text(44, 150, "The file name and the stored condition", fs=9, color=MUTE)
d.text(44, 170, "both read as a seven-layer window.", fs=9, color=MUTE)
d.arrow((490, 120), (540, 120))
d.box(552, 46, 428, 150, None, ec=BRAND)
d.text(576, 78, "What was actually measured", fs=10, color=WHITE)
d.text(576, 114, "layer = 14", fs=9.5, color=BRAND, mono=True)
d.text(576, 150, "The runner took the first element and", fs=9, color=MUTE)
d.text(576, 170, "discarded the rest without a word.", fs=9, color=MUTE)
d.line(20, 232, 980, 232, color=HAIR)
d.text(20, 264, "672 records affected — all of them step-5 controls, none of them used in any reported number.", fs=9.5, color=ASH)
d.text(20, 292, "Their per-layer block is empty, so no analysis script could have included them even by mistake.", fs=9.5, color=ASH)
d.text(20, 328, "The guard now raises when a layer list reaches a single-layer path.", fs=9.5, color=WHITE)
d.text(20, 358, "Every other step was swept for the same pattern: zero occurrences.", fs=9.5, color=BRAND)
d.save("h_layerbug")

# 모듈 지도
d = Dia(1000, 420)
d.box(20, 40, 220, 56, "conditions.py", "what to run", fs=9.5, subfs=7.5, mono=True)
d.arrow((240, 68), (300, 68))
d.box(300, 40, 220, 56, "runner.py", "route by mode", fs=9.5, subfs=7.5, mono=True, ec=BRAND)
d.arrow((520, 68), (580, 68))
d.box(580, 40, 200, 56, "model.py", "load & cache", fs=9.5, subfs=7.5, mono=True)
d.arrow((780, 68), (840, 68))
d.box(840, 40, 140, 56, "results/", "json", fs=9.5, subfs=7.5, mono=True)
for i, (n, s) in enumerate([("prompt.py", "build text, locate spans"),
                            ("attention_probe.py", "read one query row"),
                            ("intervention.py", "overwrite K / V"),
                            ("metrics.py", "score, recovery, gap"),
                            ("steer.py", "residual add / reweight")]):
    y = 140 + i * 54
    d.box(300, y, 480, 44, None)
    d.text(322, y + 22, n, fs=9.5, color=WHITE, mono=True)
    d.text(520, y + 22, s, fs=9, color=MUTE)
    d.arrow((410, 96), (410, y), color=HAIR, style="-")
d.text(20, 160, "called by", fs=8.5, color=MUTE)
d.text(20, 182, "the router,", fs=8.5, color=MUTE)
d.text(20, 204, "never by", fs=8.5, color=MUTE)
d.text(20, 226, "each other", fs=8.5, color=MUTE)
d.text(20, 394, "There is no run_rq1.py. Adding an experiment means adding a value to the condition schema,", fs=9, color=ASH)
d.text(20, 416, "never a new script — that is what keeps every step reproducible from one branch.", fs=9, color=ASH)
d.save("h_modules")

# 구간 찾기
d = Dia(1000, 400)
d.eyebrow(20, 24, "spans are located by character offset, then mapped to tokens")
d.box(20, 46, 960, 96, None)
d.text(44, 76, "prompt text", fs=8.5, color=MUTE)
d.text(44, 104, "… we generally write function names in ", fs=9, color=ASH, mono=True)
d.text(478, 104, "camelCase", fs=9, color=BRAND, mono=True)
d.text(572, 104, ". Every function name uses …", fs=9, color=ASH, mono=True)
d.arrow((520, 142), (520, 176))
d.box(20, 176, 960, 84, None)
d.text(44, 206, "char span → token span", fs=8.5, color=MUTE)
toks = ["…", " in", " camel", "Case", ".", " Every", "…"]
x = 300
for t in toks:
    hit = t in (" camel", "Case")
    d.box(x, 218, 92, 30, t, fs=8.5, mono=True, ec=BRAND if hit else HAIR, tc=BRAND if hit else ASH)
    x += 100
d.line(20, 292, 980, 292, color=HAIR)
d.text(20, 322, "A span that cannot be located raises — it is never quietly reported as zero attention.", fs=9.5, color=ASH)
d.text(20, 350, "Token counts differ per model because tokenizers differ, which is exactly why every", fs=9.5, color=ASH)
d.text(20, 372, "attention number in this deck is divided by the span's own token count.", fs=9.5, color=ASH)
d.save("h_span")

# 평균 덮어쓰기
d = Dia(1000, 400)
d.eyebrow(20, 24, "mean-pool substitution, one KV group at a time")
d.text(20, 62, "donor pieces", fs=9, color=MUTE)
for i, t in enumerate(["split", "Config"]):
    d.box(150 + i * 130, 44, 118, 38, t, fs=9, mono=True, ec=HAIR)
d.arrow((420, 63), (470, 63))
d.box(482, 44, 150, 38, "mean", fs=9.5, ec=BRAND)
d.arrow((640, 63), (690, 63))
d.text(20, 152, "target positions", fs=9, color=MUTE)
for i, t in enumerate(["split", "_", "config"]):
    d.box(150 + i * 130, 134, 118, 38, t, fs=9, mono=True, ec=BRAND, tc=BRAND)
    d.arrow((556, 82), (209 + i * 130, 134), color=HAIR, ls=(0, (3, 3)), style="-")
d.text(700, 63, "one vector, broadcast", fs=9, color=ASH)
d.text(700, 92, "to every target slot", fs=9, color=ASH)
d.line(20, 216, 980, 216, color=HAIR)
d.text(20, 248, "Why mean-pool and not piece-by-piece alignment", fs=10, color=WHITE)
d.text(20, 280, "— Tokenizers split the same name into different numbers of pieces, so a positional pairing", fs=9, color=ASH)
d.text(44, 302, "would skip names on some models and not others. Coverage would depend on the tokenizer.", fs=9, color=ASH)
d.text(20, 330, "— Mean-pool applies to every name on every model, so the four models are compared on the", fs=9, color=ASH)
d.text(44, 352, "same treatment. The cost is that it also erases the original vector — hence the controls.", fs=9, color=ASH)
d.save("h_meanpool")

# 결과 규약
d = Dia(1000, 380)
d.eyebrow(20, 24, "one path, no per-model folders")
d.box(20, 46, 960, 74, None)
d.text(44, 88, "results/<step>/<slug>.json", fs=11, color=WHITE, mono=True)
d.text(500, 88, "the slug already carries the model", fs=9, color=MUTE)
d.eyebrow(20, 160, "slug anatomy")
parts = [("qwen2-5-coder-3b-instruct", "model"), ("pre-c6of12-pool-syn-b0", "context"),
         ("ins-pos-camel-w", "instruction"), ("int-kv-L25-compliant", "intervention")]
x = 20
for s, lab in parts:
    w = 12 + len(s) * 6.4
    d.box(x, 184, w, 44, s, fs=8.5, mono=True)
    d.text(x + w / 2, 248, lab, fs=8, color=MUTE, ha="center")
    x += w + 14
d.line(20, 288, 980, 288, color=HAIR)
d.text(20, 318, "Results are immutable: a finished file is never overwritten, and resume works by", fs=9.5, color=ASH)
d.text(20, 344, "checking whether the path already exists. Split the path and both properties break.", fs=9.5, color=ASH)
d.text(20, 374, "Run environment — git sha, library versions, GPU — is stamped automatically.", fs=9.5, color=BRAND)
d.save("h_slug")
print("harness ok")
