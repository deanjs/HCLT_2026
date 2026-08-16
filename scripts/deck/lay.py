"""반복되는 배치 몇 가지. 슬라이드마다 격자를 바꿔 쓰기 위한 재료."""
from deck import *
from PIL import Image

FIG = "/home/user/HCLT_2026/docs"
A = "assets/"
MODELS = [("qwen", "Qwen2.5-Coder-3B"), ("deepseek", "DeepSeek-Coder-6.7B"),
          ("llama", "Llama-3.2-3B"), ("stability", "StableCode-3B")]


def split(wl):
    """좌/우 2단 기하. wl = 왼쪽 폭."""
    return ML, wl, ML + wl + GAP, MR - (ML + wl + GAP)


def rows(*hs):
    """세로 밴드 y좌표. hs 합 + 간격이 본문 높이가 되도록 호출부가 맞춘다."""
    out, y = [], BODY_TOP
    for h in hs:
        out.append((y, h)); y += h + GAP
    return out


def four_plots(D, s, step, pattern, y, h, caption, labels=None):
    """모델 4개 소형 다중 그래프 한 줄. 캡션은 줄 전체에 하나만 단다(카드 크기 통일)."""
    cap_h = 34 if caption else 0
    ph = h - cap_h
    w = (MR - ML - 3 * GAP) / 4
    for i, (fam, name) in enumerate(MODELS):
        x = ML + i * (w + GAP)
        cx, cy, cw, ch = D.panel(s, x, y, w, ph, name)
        D.plot(s, f"{FIG}/{step}/figures/{pattern.format(fam)}", cx, cy, cw, ch)
    if caption:
        D._tb(s, ML, y + ph + 10, MR - ML, 24, caption, 9.5, MUTE)
    return y + h


def two_plots(D, s, paths, y, h, eyebrows, captions):
    w = (MR - ML - GAP) / 2
    for i in range(2):
        x = ML + i * (w + GAP)
        cx, cy, cw, ch = D.panel(s, x, y, w, h, eyebrows[i], captions[i])
        D.plot(s, paths[i], cx, cy, cw, ch)


def kv_rows(D, s, x, y, w, items, size=10.5, pitch=34, key_w=None, color=ASH):
    """왼쪽 키 / 오른쪽 값 두 칸 목록."""
    kw = key_w or max(text_w(k, size, bold=False) for k, _ in items) + 26
    for i, (k, v) in enumerate(items):
        yy = y + i * pitch
        D._tb(s, x, yy, kw, pitch, k, size, WHITE)
        D._tb(s, x + kw, yy, w - kw, pitch, v, size, color)
    return len(items) * pitch


def table(D, s, x, y, w, headers, rows_, col_w, size=10, pitch=32, hi=None,
          fit_h=None, tag=""):
    """가는 규칙선 표. 각 행의 높이를 실제 줄바꿈으로 재고, 남는 높이는 고르게 나눈다."""
    # col_w는 비율로 해석한다 — 합이 언제나 카드 안쪽 폭 w에 정확히 들어맞는다
    tot = float(sum(col_w))
    col_w = [w * c / tot for c in col_w]
    xs, acc = [], x
    for cwv in col_w:
        xs.append(acc); acc += cwv
    for j, hd in enumerate(headers):
        D._tb(s, xs[j], y + 10, col_w[j], 18, hd, 8.5, MUTE, bold=True, caps=True, spacing=1.1)
    D.hline(s, x, y + 34, w, HAIR)

    def _need(sz, pad):
        return [max(wrap_lines(str(c), col_w[k] - 26, sz) for k, c in enumerate(r))
                * line_px(sz, 1.3) + pad for r in rows_]

    pad = 14
    need = _need(size, pad)
    base = y + 46
    slack = 0.0
    if fit_h is not None:
        room = (y + fit_h) - base
        for sz, pd in ((size, 14), (size, 11), (10.5, 11), (10, 10), (9.5, 10), (9.5, 6)):
            cand = _need(sz, pd)
            if sum(cand) <= room:
                size, pad, need = sz, pd, cand
                break
        else:
            raise ValueError(f"table overflows [{tag}]: need {sum(_need(9.5, 6)):.0f}px, "
                             f"have {room:.0f}px ({len(rows_)} rows)")
        slack = (room - sum(need)) / len(rows_)
    yy = base
    for i, r in enumerate(rows_):
        rh = need[i] + slack
        for k, cell in enumerate(r):
            col = BRAND if (hi is not None and i == hi and k == 0) else (
                WHITE if k == 0 else ASH)
            D._tb(s, xs[k], yy + 4, col_w[k] - 26, rh, str(cell), size, col, line=1.3)
        if i < len(rows_) - 1:
            D.hline(s, x, yy + rh - 6, w, RGBColor(0x26, 0x26, 0x26))
        yy += rh
    return yy - y


def code(D, s, x, y, w, h, lines, size=9.5, pitch=None):
    """코드 패널 — 모노, 6px 라운드, 어두운 바닥."""
    D.rect(s, x, y, w, h, fill=RGBColor(0x14, 0x14, 0x14), line=HAIR, radius=6)
    avail = w - 36
    widest = max((text_w(ln.lstrip("»"), size, font=MONO) for ln in lines), default=1)
    # 폭과 높이 양쪽에 맞춰 자동 축소한다 — 코드 줄이 줄바꿈되어 겹치는 일을 막는다
    for sz in [size] + [round(size - k * 0.5, 1) for k in range(1, 9)]:
        wpx = widest * sz / size
        ppx = pitch or sz * 2 * 1.45
        if wpx <= avail and 14 + len(lines) * ppx + 14 <= h:
            size, pitch = sz, ppx
            break
    else:
        raise ValueError(f"code block does not fit: widest {widest:.0f}px / avail {avail:.0f}px, "
                         f"{len(lines)} lines in {h:.0f}px")
    need = 14 + len(lines) * pitch + 14
    for i, ln in enumerate(lines):
        col = MUTE if ln.strip().startswith("#") else ASH
        if ln.startswith("»"):
            col, ln = BRAND, ln[1:]
        D._tb(s, x + 18, y + 14 + i * pitch, avail, pitch, ln, size, col,
              font=MONO)
    return need


def figure_row(D, s, x, y, w, h, path, eyebrow, caption, items=None, body=None,
               tag="", frac=0.56):
    """한 카드 안에 그림(왼쪽, 높이에 맞춤)과 설명(오른쪽)을 나란히 둔다.

    그림은 자기 비율대로 왼쪽에 붙고, 남는 폭을 설명이 가져간다 —
    넓은 카드 안에서 그림만 작게 떠 보이는 문제를 없앤다.
    """
    cx, cy, cw, ch = D.panel(D and s, x, y, w, h, eyebrow, caption)
    src = D.trim(path)
    iw, ih = Image.open(src).size
    pad = 12
    gw = min(cw * frac, (ch - 2 * pad) * iw / ih + 2 * pad)
    D.rect(s, cx, cy + (ch - ((gw - 2 * pad) * ih / iw + 2 * pad)) / 2, gw,
           (gw - 2 * pad) * ih / iw + 2 * pad, fill=LIGHT, line=HAIR, radius=12)
    dh = (gw - 2 * pad) * ih / iw
    s.shapes.add_picture(str(src), E(cx + pad),
                         E(cy + (ch - dh - 2 * pad) / 2 + pad), E(gw - 2 * pad), E(dh))
    tx = cx + gw + 40
    tw = cx + cw - tx
    if items:
        D.deflist(s, tx, cy, tw, items, fit_h=ch, tag=tag)
    elif body:
        D.body(s, tx, cy, tw, body, fit_h=ch, tag=tag)
    return (tx, cy, tw, ch)
