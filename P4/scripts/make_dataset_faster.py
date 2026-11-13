from pathlib import Path
from PIL import Image
import shutil, yaml

Image.MAX_IMAGE_PIXELS = None
SRC = Path(r"C:\Users\aserg\OneDrive\Documentos\GitHub\VC\P4\dataset")
DST = Path(r"C:\datasets\plates")
MAX_SIDE = 1280
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

for p in sorted(SRC.rglob("*")):
    if not p.is_file() or p.name in ("data.yaml", "_coco.json"):
        continue
    out = DST / p.relative_to(SRC)
    out.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in EXT:
        with Image.open(p) as im:
            w, h = im.size
            if max(w, h) > MAX_SIDE:
                s = MAX_SIDE / max(w, h)
                im.convert("RGB").resize((round(w * s), round(h * s)), Image.LANCZOS).save(out, quality=92)
                continue
    shutil.copy2(p, out)

with open(SRC / "data.yaml", encoding="utf-8") as f:
    data = yaml.safe_load(f)
data["path"] = str(DST)
with open(DST / "data.yaml", "w", encoding="utf-8") as f:
    yaml.dump(data, f, sort_keys=False, allow_unicode=True)
print("listo")