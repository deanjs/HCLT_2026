set -e
python3 -c "
import os; os.chdir('.')
exec(open('build.py').read()); D.save('HCLT2026_experiments.pptx'); print('slides', D.n)"
rm -f HCLT2026_experiments.pdf
soffice --headless --norestore -env:UserInstallation=file:///tmp/lo_p3 --convert-to pdf --outdir . HCLT2026_experiments.pptx >/dev/null 2>&1
python3 - <<'PY'
import pymupdf, os
from PIL import Image, ImageDraw
d = pymupdf.open("HCLT2026_experiments.pdf")
os.makedirs("qa", exist_ok=True)
cols, rows = 4, 3; per = cols*rows; tw, th = 620, 349
for g in range((len(d)+per-1)//per):
    sheet = Image.new("RGB", (cols*tw, rows*(th+18)), (30,30,30)); dr = ImageDraw.Draw(sheet)
    for k in range(per):
        n = g*per+k
        if n >= len(d): break
        pix = d[n].get_pixmap(dpi=50)
        im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).resize((tw, th))
        x, y = (k%cols)*tw, (k//cols)*(th+18)
        sheet.paste(im, (x, y+18)); dr.text((x+6, y+4), f"{n+1}", fill=(255,200,160))
    sheet.save(f"qa/sheet{g+1}.png")
print("pages", len(d))
PY
