#!/usr/bin/env python3
"""
Convierte:
    dataset/
        annotations/   (un .json por imagen: [{"rectangle": [x, y, w, h], "plate": "..."}, ...])
        images/

en formato YOLO:
    TGC_RBNW/
        train/{images,labels}
        val/{images,labels}
        test/{images,labels}
        data.yaml
        classes.txt

"rectangle" = [x, y, w, h] normalizados (0-1).
Por defecto (x, y) es la esquina superior izquierda; usa --origin center
si en tu caso es el centro de la caja.

Uso:
    python convert_dataset.py --src dataset --dst TGC_RBNW
    python convert_dataset.py --src dataset --dst TGC_RBNW --split 0.8 0.1 0.1 --seed 42
    python convert_dataset.py --src dataset --dst TGC_RBNW --origin center --move
"""
import argparse
import json
import random
import shutil
import sys
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_NAME = "plate"  # una sola clase: matrícula


def clamp(v):
    return max(0.0, min(1.0, v))


def convert_file(json_path: Path, origin: str):
    """Devuelve (lista de líneas YOLO, nº de cajas descartadas)."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    lines, skipped = [], 0
    for obj in data:
        try:
            x, y, w, h = (float(v) for v in obj["rectangle"])
        except (KeyError, ValueError, TypeError):
            skipped += 1
            continue
        if w <= 0 or h <= 0:
            skipped += 1
            continue
        if origin == "topleft":
            xc, yc = x + w / 2, y + h / 2
        else:
            xc, yc = x, y
        lines.append(f"0 {clamp(xc):.6f} {clamp(yc):.6f} {clamp(w):.6f} {clamp(h):.6f}")
    return lines, skipped


def find_annotation(stem: str, ann_by_stem: dict):
    """Busca el json de una imagen: 'img.json' o 'img.jpg.json'."""
    if stem in ann_by_stem:
        return ann_by_stem[stem]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dataset", help="carpeta origen")
    ap.add_argument("--dst", default="TGC_RBNW", help="carpeta destino")
    ap.add_argument("--split", nargs=3, type=float, default=[0.7, 0.2, 0.1],
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--origin", choices=["topleft", "center"], default="topleft",
                    help="significado de (x, y) en 'rectangle'")
    ap.add_argument("--move", action="store_true", help="mover en lugar de copiar")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    img_dir, ann_dir = src / "images", src / "annotations"
    if not img_dir.is_dir() or not ann_dir.is_dir():
        sys.exit("Se esperan las carpetas <src>/images y <src>/annotations")

    images = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXT)
    if not images:
        sys.exit("No se encontraron imágenes.")

    # índice de anotaciones por nombre (acepta 'foto.json' y 'foto.jpg.json')
    ann_by_stem = {}
    for f in ann_dir.rglob("*.json"):
        stem = f.stem
        if Path(stem).suffix.lower() in IMG_EXT:
            stem = Path(stem).stem
        ann_by_stem[stem] = f

    by_stem = {}
    for p in images:
        if p.stem in by_stem:
            print(f"[aviso] nombre de imagen duplicado: {p.stem}")
        by_stem[p.stem] = p

    # convertir anotaciones
    labels, no_ann, skipped_total = {}, 0, 0
    for stem in by_stem:
        f = find_annotation(stem, ann_by_stem)
        if f is None:
            no_ann += 1
            labels[stem] = []
            continue
        lines, skipped = convert_file(f, args.origin)
        labels[stem] = lines
        skipped_total += skipped
    if no_ann:
        print(f"[aviso] {no_ann} imágenes sin json -> label vacío")
    if skipped_total:
        print(f"[aviso] {skipped_total} cajas inválidas descartadas")

    # reparto train/val/test
    total = sum(args.split)
    r_train, r_val = args.split[0] / total, args.split[1] / total
    stems = sorted(by_stem)
    random.Random(args.seed).shuffle(stems)
    n = len(stems)
    n_train, n_val = int(n * r_train), int(n * r_val)
    splits = {
        "train": stems[:n_train],
        "val": stems[n_train:n_train + n_val],
        "test": stems[n_train + n_val:],
    }

    op = shutil.move if args.move else shutil.copy2
    for split, items in splits.items():
        (dst / split / "images").mkdir(parents=True, exist_ok=True)
        (dst / split / "labels").mkdir(parents=True, exist_ok=True)
        for stem in items:
            img = by_stem[stem]
            op(str(img), str(dst / split / "images" / img.name))
            text = "\n".join(labels[stem]) + ("\n" if labels[stem] else "")
            (dst / split / "labels" / f"{stem}.txt").write_text(text, encoding="utf-8")
        print(f"{split:5s}: {len(items)} imágenes")

    (dst / "classes.txt").write_text(CLASS_NAME + "\n", encoding="utf-8")
    yaml = [
        f"path: {dst.resolve().as_posix()}",
        "train: train/images",
        "val: val/images",
        "test: test/images",
        "nc: 1",
        "names:",
        f"  0: {CLASS_NAME}",
    ]
    (dst / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    print(f"Listo -> {dst}")


if __name__ == "__main__":
    main()
