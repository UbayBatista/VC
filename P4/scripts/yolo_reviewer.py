#!/usr/bin/env python3
"""
Visor / depurador de datasets con anotaciones en formato YOLO, guardadas en .txt y/o .json.

Estructura esperada:
    dataset/
        images/        -> imágenes (jpg, jpeg, png, bmp, webp)
        annotations/   -> nombre.txt y/o nombre.json (mismo nombre que la imagen)

Al eliminar una imagen se borran la imagen y TODAS sus anotaciones
asociadas (.txt y/o .json).

Uso:
    python yolo_reviewer.py /ruta/al/dataset
    python yolo_reviewer.py /ruta/al/dataset --classes classes.txt

Controles:
    Flecha derecha / rueda ratón abajo    -> siguiente imagen
    Flecha izquierda / rueda ratón arriba -> imagen anterior
    Supr / botón "Eliminar"               -> eliminar imagen + anotaciones (con confirmación)
    Inicio / Fin                          -> primera / última imagen
    Esc                                   -> salir

Dependencias:
    pip install pillow
"""

import argparse
import colorsys
import json
import sys
import tkinter as tk
import zlib
from pathlib import Path
from tkinter import messagebox

from PIL import Image, ImageDraw, ImageFont, ImageTk

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ANN_EXTS = (".txt", ".json")


def class_color(color_id: int) -> tuple:
    """Color distinto y estable para cada clase."""
    hue = (color_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def label_color_id(label: str) -> int:
    return zlib.crc32(label.encode("utf-8")) % 1000


def load_font(size: int):
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


# Cada objeto se representa como: (etiqueta, color_id, tipo, puntos_normalizados)
#   tipo "box":  puntos = [(x1, y1), (x2, y2)]
#   tipo "poly": puntos = [(x, y), (x, y), ...]

TEXT_KEYS = ("plate", "plate_text", "plate_number", "license_plate", "text")  # la etiqueta es el propio texto
CLASS_KEYS = ("class_id", "class", "cls", "category_id", "label", "category", "class_name", "name")


def resolve_label(cls_value, class_names: list, as_text: bool = False):
    """Devuelve (etiqueta, color_id) a partir de un id numérico o de un nombre."""
    if as_text:  # texto literal (p. ej. una matrícula): no interpretarlo como número
        label = str(cls_value)
        return label, label_color_id(label)
    try:
        cls = int(float(cls_value))
    except (ValueError, TypeError):
        label = str(cls_value)
        return label, label_color_id(label)
    label = class_names[cls] if 0 <= cls < len(class_names) else str(cls)
    return label, cls


def yolo_item(cls_value, vals, class_names: list, as_text: bool = False):
    """vals normalizados: [xc, yc, w, h] (caja YOLO) o [x1, y1, x2, y2, ...] (polígono)."""
    try:
        v = [float(x) for x in vals]
    except (ValueError, TypeError):
        return None
    label, color_id = resolve_label(cls_value, class_names, as_text)
    if len(v) == 4:
        xc, yc, w, h = v
        return (label, color_id, "box", [(xc - w / 2, yc - h / 2), (xc + w / 2, yc + h / 2)])
    if len(v) >= 6 and len(v) % 2 == 0:
        return (label, color_id, "poly", [(v[i], v[i + 1]) for i in range(0, len(v), 2)])
    return None


def parse_yolo_line(line: str, class_names: list):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    return yolo_item(parts[0], parts[1:], class_names)


def parse_txt(path: Path, class_names: list) -> list:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = parse_yolo_line(line, class_names)
            if item:
                items.append(item)
    return items


def first_key(d: dict, keys: tuple):
    for k in keys:
        if k in d:
            return d[k]
    return None


def parse_json_object(o, class_names: list, img_size: tuple):
    """Interpreta un objeto de un JSON en formato YOLO (varias variantes habituales)."""
    # "0 0.5 0.5 0.2 0.2"
    if isinstance(o, str):
        return parse_yolo_line(o, class_names)

    # [0, 0.5, 0.5, 0.2, 0.2]
    if isinstance(o, (list, tuple)):
        return yolo_item(o[0], o[1:], class_names) if len(o) >= 5 else None

    if not isinstance(o, dict):
        return None

    as_text = False
    cls = first_key(o, TEXT_KEYS)
    if cls is not None:
        as_text = True
    else:
        cls = first_key(o, CLASS_KEYS)
        if cls is None:
            cls = "?"

    # {"class": 0, "x_center": .., "y_center": .., "width": .., "height": ..}
    xc = first_key(o, ("x_center", "xc", "cx", "x_centre"))
    yc = first_key(o, ("y_center", "yc", "cy", "y_centre"))
    w = first_key(o, ("width", "w"))
    h = first_key(o, ("height", "h"))
    if None not in (xc, yc, w, h):
        return yolo_item(cls, [xc, yc, w, h], class_names, as_text)

    # {"class": 0, "bbox": [xc, yc, w, h]}  (normalizado = YOLO); también "rectangle"
    box = first_key(o, ("rectangle", "bbox", "box", "bounding_box"))
    if isinstance(box, (list, tuple)) and len(box) == 4:
        vals = [float(v) for v in box]
        if max(vals) <= 1.0:
            return yolo_item(cls, vals, class_names, as_text)
        # Si vienen en píxeles se asume [x, y, ancho, alto] (esquina superior izquierda)
        iw, ih = img_size
        x, y, bw, bh = vals
        return yolo_item(cls, [(x + bw / 2) / iw, (y + bh / 2) / ih, bw / iw, bh / ih], class_names, as_text)

    # {"class": 0, "points": [x1, y1, x2, y2, ...]}  (segmentación)
    poly = first_key(o, ("points", "polygon", "segmentation"))
    if isinstance(poly, (list, tuple)) and poly:
        flat = [c for p in poly for c in p] if isinstance(poly[0], (list, tuple)) else list(poly)
        flat = [float(c) for c in flat]
        if max(flat) > 1.0:  # píxeles -> normalizar
            iw, ih = img_size
            flat = [c / (iw if i % 2 == 0 else ih) for i, c in enumerate(flat)]
        return yolo_item(cls, flat, class_names, as_text)

    return None


def parse_json(path: Path, img_size: tuple, class_names: list) -> list:
    """
    JSON con contenido en formato YOLO. Variantes soportadas:
      - Lista de listas:     [[cls, xc, yc, w, h], ...]
      - Lista de strings:    ["cls xc yc w h", ...]
      - Lista de dicts:      [{"class": 0, "x_center": .., "y_center": .., "width": .., "height": ..}, ...]
                             o con "bbox": [xc, yc, w, h] normalizado
      - Detector de matrículas: [{"rectangle": [xc, yc, w, h], "plate": "8712CHC"}]
        (la matrícula se muestra como etiqueta de la caja)
      - Dict contenedor con la lista en "annotations", "objects", "labels", "boxes",
        "detections", "lines" o "data".
      - Además, formato LabelMe ("shapes") por compatibilidad.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    w, h = img_size
    items = []

    if isinstance(data, dict) and isinstance(data.get("shapes"), list):  # LabelMe
        w = data.get("imageWidth") or w
        h = data.get("imageHeight") or h
        for s in data["shapes"]:
            label = str(s.get("label", "?"))
            pts = [(float(p[0]) / w, float(p[1]) / h) for p in s.get("points", [])]
            if s.get("shape_type") == "rectangle" and len(pts) >= 2:
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                items.append((label, label_color_id(label), "box",
                              [(min(xs), min(ys)), (max(xs), max(ys))]))
            elif len(pts) >= 3:
                items.append((label, label_color_id(label), "poly", pts))
        return items

    if isinstance(data, list):
        # Un único objeto como lista plana de números: [0, 0.5, 0.5, 0.2, 0.2]
        if data and all(isinstance(x, (int, float)) for x in data):
            objs = [data]
        else:
            objs = data
    elif isinstance(data, dict):
        objs = None
        for key in ("annotations", "objects", "labels", "boxes", "detections", "lines", "data"):
            if isinstance(data.get(key), list):
                objs = data[key]
                break
        if objs is None:
            objs = [data]  # el propio dict es un único objeto
    else:
        objs = []

    for o in objs:
        try:
            item = parse_json_object(o, class_names, img_size)
        except Exception:
            item = None
        if item:
            items.append(item)
    return items


class YoloReviewer:
    def __init__(self, root: tk.Tk, images_dir: Path, ann_dir: Path, class_names: list):
        self.root = root
        self.images_dir = images_dir
        self.ann_dir = ann_dir
        self.class_names = class_names

        self.images = sorted(
            p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        if not self.images:
            messagebox.showerror("Error", f"No se encontraron imágenes en:\n{images_dir}")
            root.destroy()
            sys.exit(1)

        self.index = 0
        self.current_pil = None
        self.current_path = None
        self.tk_image = None
        self.font = load_font(14)
        self._redraw_pending = False

        self._build_ui()
        self._bind_events()
        self.load_current()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        self.root.title("Dataset Reviewer (YOLO / JSON)")
        self.root.geometry("1100x750")
        self.root.minsize(600, 450)

        top = tk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)
        self.info_label = tk.Label(top, text="", anchor="w", font=("TkDefaultFont", 10, "bold"))
        self.info_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.ann_label = tk.Label(top, text="", anchor="e")
        self.ann_label.pack(side=tk.RIGHT)

        self.canvas = tk.Canvas(self.root, bg="#202020", highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        bottom = tk.Frame(self.root)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=6)

        tk.Button(bottom, text="◀ Anterior", width=12, command=self.prev_image).pack(side=tk.LEFT)
        tk.Button(bottom, text="Siguiente ▶", width=12, command=self.next_image).pack(side=tk.LEFT, padx=6)
        tk.Button(
            bottom, text="🗑 Eliminar", width=14, bg="#c0392b", fg="white",
            activebackground="#e74c3c", activeforeground="white",
            command=self.delete_current,
        ).pack(side=tk.RIGHT)

    def _bind_events(self):
        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Home>", lambda e: self.go_to(0))
        self.root.bind("<End>", lambda e: self.go_to(len(self.images) - 1))
        self.root.bind("<Delete>", lambda e: self.delete_current())
        self.root.bind("<Escape>", lambda e: self.root.destroy())

        self.root.bind("<MouseWheel>", self._on_wheel)
        self.root.bind("<Button-4>", lambda e: self.prev_image())
        self.root.bind("<Button-5>", lambda e: self.next_image())

        self.canvas.bind("<Configure>", lambda e: self._schedule_redraw())

    def _on_wheel(self, event):
        if event.delta > 0:
            self.prev_image()
        else:
            self.next_image()

    # ----------------------------------------------------------- Navegación
    def go_to(self, idx: int):
        if 0 <= idx < len(self.images):
            self.index = idx
            self.load_current()

    def next_image(self):
        if self.index < len(self.images) - 1:
            self.index += 1
            self.load_current()

    def prev_image(self):
        if self.index > 0:
            self.index -= 1
            self.load_current()

    # ----------------------------------------------------------- Anotaciones
    def annotation_files(self, img_path: Path) -> list:
        """Todas las anotaciones existentes (.txt y/o .json) de una imagen."""
        candidates = [self.ann_dir / (img_path.stem + ext) for ext in ANN_EXTS]
        return [p for p in candidates if p.exists()]

    def parse_annotations(self, files: list, img_size: tuple):
        if not files:
            return [], "Sin archivo de anotación"
        items, errors = [], []
        for f in files:
            try:
                if f.suffix.lower() == ".txt":
                    items += parse_txt(f, self.class_names)
                else:
                    items += parse_json(f, img_size, self.class_names)
            except Exception as exc:
                errors.append(f"{f.name}: {exc}")

        kinds = "+".join(f.suffix for f in files)
        status = f"{len(items)} objeto(s) [{kinds}]" if items else f"Anotación vacía [{kinds}]"
        if errors:
            status += f"  ⚠ error: {'; '.join(errors)}"
        return items, status

    # ----------------------------------------------------------- Carga/dibujo
    def load_current(self):
        self.current_path = self.images[self.index]
        try:
            self.current_pil = Image.open(self.current_path).convert("RGB")
        except Exception as exc:
            self.current_pil = None
            messagebox.showwarning("Aviso", f"No se pudo abrir la imagen:\n{self.current_path}\n\n{exc}")
        self._schedule_redraw()

    def _schedule_redraw(self):
        if not self._redraw_pending:
            self._redraw_pending = True
            self.root.after_idle(self.redraw)

    def redraw(self):
        self._redraw_pending = False
        self.canvas.delete("all")

        self.info_label.config(
            text=f"[{self.index + 1}/{len(self.images)}]  {self.current_path.name}"
        )

        files = self.annotation_files(self.current_path)

        if self.current_pil is None:
            self.ann_label.config(text="")
            return

        annotations, status = self.parse_annotations(files, self.current_pil.size)
        self.ann_label.config(text=status)

        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        iw, ih = self.current_pil.size
        scale = min(cw / iw, ch / ih)
        new_w, new_h = max(1, int(iw * scale)), max(1, int(ih * scale))
        img = self.current_pil.resize((new_w, new_h), Image.LANCZOS)
        draw = ImageDraw.Draw(img)

        line_w = max(2, int(min(new_w, new_h) / 250))

        for label, color_id, kind, pts in annotations:
            color = class_color(color_id)
            px = [(x * new_w, y * new_h) for x, y in pts]

            if kind == "box":
                (x1, y1), (x2, y2) = px[0], px[1]
                draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)
                anchor = (x1, y1)
            else:
                draw.line(px + [px[0]], fill=color, width=line_w)
                anchor = (min(p[0] for p in px), min(p[1] for p in px))

            tb = draw.textbbox((0, 0), label, font=self.font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            lx = max(0, anchor[0])
            ly = anchor[1] - th - 6
            if ly < 0:
                ly = anchor[1]
            draw.rectangle([lx, ly, lx + tw + 6, ly + th + 6], fill=color)
            draw.text((lx + 3, ly + 2), label, fill=(0, 0, 0), font=self.font)

        self.tk_image = ImageTk.PhotoImage(img)
        self.canvas.create_image(cw // 2, ch // 2, image=self.tk_image, anchor="center")

    # ---------------------------------------------------------------- Borrado
    def delete_current(self):
        img_path = self.current_path
        ann_files = self.annotation_files(img_path)

        if ann_files:
            ann_txt = "\n".join(f"  Anotación:  {p.name}" for p in ann_files)
        else:
            ann_txt = "  Anotación:  (no existe ninguna)"

        confirm = messagebox.askyesno(
            "Confirmar eliminación",
            "¿Seguro que quieres ELIMINAR definitivamente?\n\n"
            f"  Imagen:     {img_path.name}\n"
            f"{ann_txt}\n\n"
            "Esta acción no se puede deshacer.",
            icon="warning",
            default=messagebox.NO,
        )
        if not confirm:
            return

        # Liberar la imagen (importante en Windows para poder borrar el archivo)
        if self.current_pil is not None:
            self.current_pil.close()
            self.current_pil = None

        errors = []
        for p in [img_path] + ann_files:
            try:
                if p.exists():
                    p.unlink()
            except Exception as exc:
                errors.append(f"{p.name}: {exc}")

        if errors:
            messagebox.showerror("Error al eliminar", "\n".join(errors))
            if img_path.exists():
                self.load_current()
                return

        del self.images[self.index]

        if not self.images:
            messagebox.showinfo("Fin", "No quedan más imágenes en el dataset.")
            self.root.destroy()
            return

        if self.index >= len(self.images):
            self.index = len(self.images) - 1
        self.load_current()


def main():
    parser = argparse.ArgumentParser(description="Visor y depurador de datasets YOLO / JSON")
    parser.add_argument("dataset", type=Path, help="Ruta a la carpeta raíz del dataset")
    parser.add_argument("--images", default="images", help="Nombre de la carpeta de imágenes (def: images)")
    parser.add_argument("--annotations", default="annotations", help="Nombre de la carpeta de anotaciones (def: annotations)")
    parser.add_argument("--classes", type=Path, default=None,
                        help="Archivo de texto con un nombre de clase por línea (opcional, para .txt YOLO)")
    args = parser.parse_args()

    images_dir = args.dataset / args.images
    ann_dir = args.dataset / args.annotations

    if not images_dir.is_dir():
        sys.exit(f"No existe la carpeta de imágenes: {images_dir}")
    if not ann_dir.is_dir():
        sys.exit(f"No existe la carpeta de anotaciones: {ann_dir}")

    class_names = []
    if args.classes:
        if not args.classes.is_file():
            sys.exit(f"No existe el archivo de clases: {args.classes}")
        class_names = [l.strip() for l in args.classes.read_text(encoding="utf-8").splitlines() if l.strip()]

    root = tk.Tk()
    YoloReviewer(root, images_dir, ann_dir, class_names)
    root.mainloop()


if __name__ == "__main__":
    main()