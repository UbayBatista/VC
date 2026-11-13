#!/usr/bin/env python3
"""Compara predicciones de RT-DETR con las etiquetas (GT) sobre un split del dataset.

Dibuja en cada imagen:
    verde  = etiqueta real (GT)
    rojo   = predicción con su score (solo las que superan --thr)

Además imprime métricas a un umbral fijo (precisión, recall, IoU medio de las
detecciones acertadas) y las desglosa por tamaño de placa (tercil del área de
la GT). Sirve para ver si el problema son placas pequeñas, cajas imprecisas,
falsos positivos o placas no detectadas.

Los nombres de fichero guardados llevan el prefijo "<n>fn_<n>fp_" (falsos
negativos / falsos positivos de esa imagen), así las peores quedan agrupadas
al ordenar por nombre.

Uso (desde la carpeta P4, con el entorno rtdetr activado):
    python scripts/visualize_val.py --dataset C:\\datasets\\plates ^
        --checkpoint RT-DETR\\rtdetr_pytorch\\output\\rtdetr_plate_ft_unfrozen\\checkpoint.pth ^
        --thr 0.5 --only-errors
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import yaml
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent / "RT-DETR" / "rtdetr_pytorch"
sys.path.insert(0, str(REPO))
from src.core import YAMLConfig  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
GREEN = (0, 220, 0)
RED = (255, 40, 40)


# ----------------------------------------------------------------- dataset
def load_split_dirs(dataset, split):
    dataset = Path(dataset)
    with open(dataset / "data.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    root = Path(data.get("path", dataset))
    if not root.is_absolute():
        root = (dataset / root).resolve()
    img_dir = root / data[split]
    return img_dir, img_dir.parent / "labels"


def read_gt(label_path, w, h):
    boxes = []
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            xc, yc, bw, bh = (float(v) for v in p[1:5])
            boxes.append([(xc - bw / 2) * w, (yc - bh / 2) * h,
                          (xc + bw / 2) * w, (yc + bh / 2) * h])
    return np.array(boxes, dtype=np.float32).reshape(-1, 4)


# ------------------------------------------------------------------ métricas
def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match(gt, pred, scores, iou_thr=0.5):
    """Emparejamiento voraz por score. Devuelve (iou por GT, nº de FP)."""
    gt_iou = np.zeros(len(gt), dtype=np.float32)  # 0 = no detectada
    used = set()
    fp = 0
    order = np.argsort(-scores)
    ious = iou_matrix(pred[order], gt)
    for i in range(len(order)):
        best_j, best_iou = -1, iou_thr
        for j in range(len(gt)):
            if j not in used and ious[i, j] >= best_iou:
                best_j, best_iou = j, ious[i, j]
        if best_j >= 0:
            used.add(best_j)
            gt_iou[best_j] = best_iou
        else:
            fp += 1
    return gt_iou, fp


# -------------------------------------------------------------------- modelo
def load_model(config, checkpoint, device):
    cfg = YAMLConfig(config)
    # Los pesos vienen del checkpoint: no hace falta bajar ImageNet.
    for key in ("PResNet", "HGNetv2"):
        if key in cfg.yaml_cfg:
            cfg.yaml_cfg[key]["pretrained"] = False

    try:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:  # torch antiguo sin weights_only
        ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
    cfg.model.load_state_dict(state)

    class Deploy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_sizes):
            return self.postprocessor(self.model(images), orig_sizes)

    return Deploy().to(device).eval()


# ---------------------------------------------------------------------- dibujo
def draw(img, gt, pred, scores):
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    lw = max(2, round(max(img.size) / 400))
    for b in gt:
        d.rectangle([float(v) for v in b], outline=GREEN, width=lw)
    for b, s in zip(pred, scores):
        d.rectangle([float(v) for v in b], outline=RED, width=lw)
        d.text((float(b[0]) + 2, max(0.0, float(b[1]) - 11)), f"{s:.2f}", fill=RED, font=font)
    return img


def main():
    ap = argparse.ArgumentParser(description="Predicciones vs etiquetas sobre un split.")
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--config", default="scripts/rtdetr_plate.yml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="viz_val", help="carpeta de salida de las imágenes")
    ap.add_argument("--thr", type=float, default=0.5, help="umbral de score para dibujar/contar")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU mínimo para dar una predicción por acertada")
    ap.add_argument("--size", type=int, default=800, help="resolución de entrada del modelo")
    ap.add_argument("--only-errors", action="store_true",
                    help="guardar solo imágenes con algún falso positivo o negativo")
    ap.add_argument("--max-images", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.config, args.checkpoint, device)
    tf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor()])

    img_dir, label_dir = load_split_dirs(args.dataset, args.split)
    paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    if args.max_images:
        paths = paths[:args.max_images]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_area_frac, gt_iou_all = [], []
    total_fp = total_preds = 0
    saved = 0

    with torch.no_grad():
        for p in paths:
            img = Image.open(p).convert("RGB")
            w, h = img.size
            x = tf(img).unsqueeze(0).to(device)
            sizes = torch.tensor([[w, h]], dtype=torch.float32, device=device)
            _, boxes, scores = model(x, sizes)
            boxes, scores = boxes[0].cpu().numpy(), scores[0].cpu().numpy()
            keep = scores >= args.thr
            boxes, scores = boxes[keep], scores[keep]

            gt = read_gt(label_dir / (p.stem + ".txt"), w, h)
            gt_iou, fp = match(gt, boxes, scores, args.iou)
            fn = int((gt_iou == 0).sum())

            gt_area_frac += [float((b[2] - b[0]) * (b[3] - b[1]) / (w * h)) for b in gt]
            gt_iou_all += gt_iou.tolist()
            total_fp += fp
            total_preds += len(boxes)

            if args.only_errors and fn == 0 and fp == 0:
                continue
            vis = draw(img.copy(), gt, boxes, scores)
            vis.save(out_dir / f"{fn}fn_{fp}fp_{p.stem}.jpg", quality=92)
            saved += 1

    # ------------------------------------------------------------- resumen
    gt_iou_all = np.array(gt_iou_all)
    area = np.array(gt_area_frac)
    hit = gt_iou_all > 0
    tp = int(hit.sum())
    n_gt = len(gt_iou_all)
    print(f"\nImágenes: {len(paths)} | GT: {n_gt} | predicciones (score>={args.thr}): {total_preds}")
    print(f"Recall    @IoU{args.iou}: {tp / max(n_gt, 1):.3f}  ({n_gt - tp} placas sin detectar)")
    print(f"Precisión @IoU{args.iou}: {tp / max(total_preds, 1):.3f}  ({total_fp} falsos positivos)")
    if tp:
        print(f"IoU medio de las detectadas: {gt_iou_all[hit].mean():.3f}")

    if n_gt >= 3:
        lo, hi = np.percentile(area, [33.3, 66.7])
        groups = {
            f"pequeñas (<{lo * 100:.2f}% de la imagen)": area < lo,
            "medianas": (area >= lo) & (area < hi),
            f"grandes (>={hi * 100:.2f}%)": area >= hi,
        }
        print("\nPor tamaño de la GT (terciles del área):")
        for name, m in groups.items():
            if m.sum() == 0:
                continue
            h_m = hit[m]
            iou_txt = f"{gt_iou_all[m][h_m].mean():.3f}" if h_m.any() else "-"
            print(f"  {name:32s} n={int(m.sum()):4d}  recall={h_m.mean():.3f}  IoU medio={iou_txt}")

    print(f"\n{saved} imágenes guardadas en {out_dir.resolve()}")
    print("Verde = GT, rojo = predicción (con score).")


if __name__ == "__main__":
    main()