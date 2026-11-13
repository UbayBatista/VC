#!/usr/bin/env python3
"""Entrena / hace fine-tuning de RT-DETR (lyuwenyu/rtdetr_pytorch) sobre un dataset YOLO.

1. Lee <dataset>/data.yaml (rutas, nº de clases y nombres).
2. Convierte las etiquetas YOLO (.txt: clase xc yc w h normalizadas) a COCO JSON
   para los splits train y val.
3. Escribe un fichero de configuración YAML de RT-DETR ajustado al nº de clases,
   a las rutas del dataset, al tamaño de lote y a la estrategia de fine-tuning.
4. Lanza RT-DETR/rtdetr_pytorch/tools/train.py con esa configuración.

Fine-tuning:
    --finetune                 descarga (si hace falta) los pesos COCO del --size y los usa
    --weights RUTA.pth         usa esos pesos (implica fine-tuning)
    --freeze-backbone          congela el backbone entero (útil con datasets pequeños)
    --lr / --backbone-lr-mult  lr del resto del modelo y multiplicador para el backbone

Nota: al cambiar el nº de clases, las capas de clasificación (enc_score_head,
dec_score_head, denoising_class_embed) no coinciden con las de COCO. El cargador
--tuning de RT-DETR las omite y se inicializan de cero; el resto se reutiliza.

Uso:
    python scripts/train_rtdetr.py --dataset dataset --size r50 --finetune --epochs 30 --amp
    python scripts/train_rtdetr.py --dataset dataset --size r50 --finetune --freeze-backbone --lr 1e-4
    python scripts/train_rtdetr.py --dataset dataset --size r50 --weights pesos/rtdetr_r50vd_6x_coco_from_paddle.pth
    python scripts/train_rtdetr.py --dataset dataset --size r50 --finetune --dry-run
"""

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import yaml
from PIL import Image

REPO = Path(__file__).resolve().parent.parent / "RT-DETR" / "rtdetr_pytorch"
TRAIN_PY = REPO / "tools" / "train.py"

# Config base por tamaño: define arquitectura (backbone, encoder, decoder) que
# debe coincidir con la de los pesos preentrenados.
BASE_CONFIGS = {
    "r18":  "rtdetr_r18vd_6x_coco.yml",
    "r34":  "rtdetr_r34vd_6x_coco.yml",
    "r50m": "rtdetr_r50vd_m_6x_coco.yml",
    "r50":  "rtdetr_r50vd_6x_coco.yml",
    "r101": "rtdetr_r101vd_6x_coco.yml",
}

WEIGHTS = {
    "r18":  "rtdetr_r18vd_dec3_6x_coco_from_paddle.pth",
    "r34":  "rtdetr_r34vd_dec4_6x_coco_from_paddle.pth",
    "r50m": "rtdetr_r50vd_m_6x_coco_from_paddle.pth",
    "r50":  "rtdetr_r50vd_6x_coco_from_paddle.pth",
    "r101": "rtdetr_r101vd_6x_coco_from_paddle.pth",
}
WEIGHTS_TAG = "v0.1"
WEIGHTS_URL = f"https://github.com/lyuwenyu/storage/releases/download/{WEIGHTS_TAG}/{{}}"

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Valores por defecto de fine-tuning (más suaves que el entrenamiento desde cero)
FT_DEFAULT_LR = 5e-5
FT_DEFAULT_BACKBONE_MULT = 0.1


def load_data_yaml(dataset):
    data_yaml = Path(dataset) / "data.yaml"
    with open(data_yaml, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    root = Path(data.get("path", dataset))
    if not root.is_absolute():
        root = (Path(dataset) / root).resolve()
    nc = int(data["nc"])
    names = [data["names"][i] for i in range(nc)]
    return {
        "root": root,
        "train": root / data["train"],
        "val": root / data["val"],
        "nc": nc,
        "names": names,
    }


def image_size(path):
    with Image.open(path) as im:
        return im.size  # (width, height)


def yolo_to_coco(images_dir, labels_dir, out_json, categories):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    img_paths = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXT
    )
    coco = {"images": [], "annotations": [], "categories": categories}
    ann_id = 1
    for img_id, img_path in enumerate(img_paths, start=1):
        w, h = image_size(img_path)
        coco["images"].append({
            "id": img_id,
            "file_name": img_path.name,
            "width": w,
            "height": h,
        })
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        for line in label_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls = int(parts[0])
            xc, yc, bw, bh = (float(v) for v in parts[1:5])
            x = (xc - bw / 2.0) * w
            y = (yc - bh / 2.0) * h
            bw_abs = bw * w
            bh_abs = bh * h
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls + 1,
                "bbox": [round(float(x), 2), round(float(y), 2),
                         round(float(bw_abs), 2), round(float(bh_abs), 2)],
                "area": round(float(bw_abs * bh_abs), 2),
                "iscrowd": 0,
            })
            ann_id += 1
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(coco), encoding="utf-8")
    return len(img_paths), ann_id - 1


def build_optimizer_cfg(lr, backbone_lr_mult, freeze_backbone):
    """Optimizador AdamW con los mismos grupos que optimizer.yml de RT-DETR,
    pero con lr configurable. Si el backbone está congelado se omite su grupo."""
    params = []
    if not freeze_backbone:
        params.append({"params": "backbone", "lr": lr * backbone_lr_mult})
    params += [
        {"params": "^(?=.*encoder(?=.*bias|.*norm.*weight)).*$", "weight_decay": 0.0},
        {"params": "^(?=.*decoder(?=.*bias|.*norm.*weight)).*$", "weight_decay": 0.0},
    ]
    return {
        "type": "AdamW",
        "params": params,
        "lr": lr,
        "betas": [0.9, 0.999],
        "weight_decay": 0.0001,
    }


def write_config(out_path, size, train_json, train_img, val_json, val_img,
                 num_classes, batch_size, epochs, output_dir,
                 finetune=False, freeze_backbone=False,
                 lr=None, backbone_lr_mult=FT_DEFAULT_BACKBONE_MULT,
                 workers=4, lr_drop=None):
    base = (REPO / "configs" / "rtdetr" / BASE_CONFIGS[size]).resolve()
    cfg = {
        "__include__": [str(base)],
        "task": "detection",
        "num_classes": num_classes,
        "remap_mscoco_category": False,
        "train_dataloader": {
            "dataset": {
                "img_folder": str(train_img),
                "ann_file": str(train_json),
            },
            "batch_size": batch_size,
            "num_workers": workers,
            "persistent_workers": workers > 0,
        },
        "val_dataloader": {
            "dataset": {
                "img_folder": str(val_img),
                "ann_file": str(val_json),
            },
            "batch_size": max(1, batch_size),
            "num_workers": workers,
            "persistent_workers": workers > 0,
        },
        "output_dir": str(output_dir),
    }

    presnet = {}
    if finetune:
        # Los pesos COCO ya incluyen el backbone: no hace falta bajar ImageNet.
        presnet["pretrained"] = False
    if freeze_backbone:
        presnet["freeze_at"] = 4  # congela stem + los 4 stages
    if presnet:
        cfg["PResNet"] = presnet

    if lr is not None:
        cfg["optimizer"] = build_optimizer_cfg(lr, backbone_lr_mult, freeze_backbone)

    if lr_drop is not None:
        # El lr se multiplica por 0.1 al llegar a esa época (se cuenta desde 0)
        cfg["lr_scheduler"] = {
            "type": "MultiStepLR",
            "milestones": [int(lr_drop)],
            "gamma": 0.1,
        }

    if epochs is not None:
        cfg["epoches"] = int(epochs)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, sort_keys=False)
    return out_path


def download_weights(size, dst):
    wfile = WEIGHTS[size]
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print(f"Pesos ya presentes: {dst}")
        return dst
    url = WEIGHTS_URL.format(wfile)
    print(f"Descargando pesos {size} desde {url} ...")
    urllib.request.urlretrieve(url, dst)
    print(f"Guardado en {dst}")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description="Entrena / fine-tuning de RT-DETR sobre un dataset YOLO.")
    ap.add_argument("--dataset", default="dataset", help="carpeta del dataset (con data.yaml)")
    ap.add_argument("--size", choices=list(WEIGHTS), default="r50",
                    help="tamaño del modelo (arquitectura + pesos preentrenados)")
    ap.add_argument("--epochs", type=int, default=None, help="nº de épocas (por defecto 72 del config)")
    ap.add_argument("--batch-size", type=int, default=4, help="tamaño de lote")
    ap.add_argument("--patience", type=int, default=None,
                    help="épocas de paciencia (early stopping). RT-DETR base no lo implementa nativamente; se monitoriza el log y se detiene si no mejora la métrica de val")

    # --- Fine-tuning ---
    ap.add_argument("--finetune", action="store_true",
                    help="hacer fine-tuning desde pesos COCO (descarga los del --size si no das --weights)")
    ap.add_argument("--weights", type=str, default=None,
                    help="ruta a pesos .pth para fine-tuning (implica --finetune)")
    ap.add_argument("--download-weights", action="store_true",
                    help="descargar los pesos COCO oficiales del tamaño elegido (implica --finetune)")
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="congelar el backbone durante el entrenamiento")
    ap.add_argument("--lr", type=float, default=None,
                    help=f"learning rate base (con fine-tuning, por defecto {FT_DEFAULT_LR}; "
                         "sin fine-tuning se respeta el del config)")
    ap.add_argument("--backbone-lr-mult", type=float, default=FT_DEFAULT_BACKBONE_MULT,
                    help="multiplicador del lr para el backbone (ignorado si está congelado)")

    ap.add_argument("--lr-drop", type=int, default=None,
                    help="época en la que el lr se multiplica por 0.1 (ej. 25 con --epochs 30)")
    ap.add_argument("--workers", type=int, default=4,
                    help="num_workers de los dataloaders (con persistent_workers). "
                         "En Windows, valores altos hacen lento el arranque de cada época")

    ap.add_argument("--config-out", default="scripts/rtdetr_plate.yml",
                    help="donde escribir el fichero de configuración generado")
    ap.add_argument("--output-dir", default="./output/rtdetr_plate",
                    help="directorio de salidas de entrenamiento")
    ap.add_argument("--amp", action="store_true", help="mezcla de precisión mixta")
    ap.add_argument("--seed", type=int, default=None, help="semilla")
    ap.add_argument("--resume", type=str, default=None, help="reanudar desde un checkpoint")
    ap.add_argument("--no-convert", action="store_true",
                    help="saltar la conversión YOLO -> COCO (asume que ya existe)")
    ap.add_argument("--dry-run", action="store_true",
                    help="solo genera COCO y config, no lanza el entrenamiento")
    args = ap.parse_args()

    dataset = Path(args.dataset).resolve()
    if not (dataset / "data.yaml").exists():
        sys.exit(f"No se encontró {dataset / 'data.yaml'}")

    finetune = args.finetune or args.download_weights or args.weights is not None
    if args.freeze_backbone and not finetune:
        sys.exit("--freeze-backbone solo tiene sentido con fine-tuning "
                 "(usa --finetune o --weights).")
    if args.weights and not Path(args.weights).exists():
        sys.exit(f"No existe el fichero de pesos: {args.weights}")

    lr = args.lr
    if finetune and lr is None:
        lr = FT_DEFAULT_LR

    info = load_data_yaml(dataset)
    print(f"Dataset: {info['root']} | clases: {info['nc']} -> {info['names']}")

    categories = [{"id": i + 1, "name": name} for i, name in enumerate(info["names"])]

    if not args.no_convert:
        for split in ("train", "val"):
            img_dir = info[split]
            label_dir = info[split].parent / "labels"
            out_json = info[split].parent / "_coco.json"
            n_img, n_ann = yolo_to_coco(img_dir, label_dir, out_json, categories)
            print(f"{split}: {n_img} imágenes, {n_ann} anotaciones -> {out_json}")

    train_json = info["train"].parent / "_coco.json"
    val_json = info["val"].parent / "_coco.json"
    for f in (train_json, val_json):
        if not f.exists():
            sys.exit(f"Falta el fichero COCO: {f}. Ejecuta sin --no-convert.")

    # category_id empieza en 1, así que num_classes = nc + 1 (el índice 0 queda sin usar)
    config_path = write_config(
        args.config_out, args.size, train_json, info["train"],
        val_json, info["val"], info["nc"] + 1,
        args.batch_size, args.epochs, args.output_dir,
        finetune=finetune, freeze_backbone=args.freeze_backbone,
        lr=lr, backbone_lr_mult=args.backbone_lr_mult,
        workers=args.workers, lr_drop=args.lr_drop,
    )
    print(f"Configuración escrita en {config_path}")
    if finetune:
        print(f"Fine-tuning: lr={lr}, backbone_lr_mult={args.backbone_lr_mult}, "
              f"backbone {'congelado' if args.freeze_backbone else 'entrenable'}")

    weights = None
    if args.weights:
        weights = Path(args.weights)
    elif finetune and not args.dry_run:
        weights = download_weights(args.size, Path("weights") / WEIGHTS[args.size])

    if args.dry_run:
        print("Dry-run completo. Revisa el config y lanza el entrenamiento manualmente.")
        return

    if args.patience is not None:
        ep = args.epochs if args.epochs is not None else "todas las"
        print(f"[aviso] RT-DETR base no implementa early stopping nativo: entrenará {ep} épocas completas.")
        print(f"        Usa --patience solo como referencia; los checkpoints se guardan cada época en --output-dir.")

    cmd = [sys.executable, str(TRAIN_PY), "-c", str(Path(config_path).resolve())]
    if weights:
        cmd += ["--tuning", str(Path(weights).resolve())]
    if args.amp:
        cmd.append("--amp")
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    if args.resume:
        cmd += ["--resume", args.resume]

    print("Lanzando entrenamiento:")
    print(" ", " ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO), check=True)


if __name__ == "__main__":
    main()