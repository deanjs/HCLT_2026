set -e
cp ../../docs/발표/HCLT2026_논문정리.pptx ./paper.pptx
rm -f paper.pdf
soffice --headless --norestore -env:UserInstallation=file:///tmp/lo_p4 --convert-to pdf --outdir . paper.pptx >/dev/null 2>&1
python3 - <<'PY'
import pymupdf, os
from PIL import Image, ImageDraw
d = pymupdf.open("paper.pdf"); os.makedirs("qa_paper", exist_ok=True)
cols, rows = 4, 3; per = cols*rows; tw, th = 620, 349
for g in range((len(d)+per-1)//per):
    sheet = Image.new("RGB", (cols*tw, rows*(th+20)), (235,235,235)); dr = ImageDraw.Draw(sheet)
    for k in range(per):
        n = g*per+k
        if n >= len(d): break
        pix = d[n].get_pixmap(dpi=52)
        im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).resize((tw, th))
        x, y = (k%cols)*tw, (k//cols)*(th+20)
        sheet.paste(im, (x, y+20)); dr.text((x+6, y+5), f"{n+1}", fill=(180,40,0))
    sheet.save(f"qa_paper/sheet{g+1}.png")
print("pages", len(d))
PY
