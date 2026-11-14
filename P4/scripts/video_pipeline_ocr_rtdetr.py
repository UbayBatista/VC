
"""
Detección + tracking + matrículas + OCR sobre el vídeo de test.

Por cada modelo genera:
    salida_rtdetr_<modelo>.mp4
    tracking_rtdetr_<modelo>.csv

Al finalizar:
    video_summary.csv

YOLO11n: detección y tracking de vehículos.
RT-DETR personalizado: detección de matrículas.
OCR: lectura de matrículas.
"""

import csv
import gc
import os
import time

import cv2
import numpy as np
import pandas as pd
import torch

from PIL import Image
from ultralytics import YOLO

from load_rtdetr import load_rtdetr
from ocr_models import build_model, clean_text, MODEL_NAMES


# ---------------------------------------------------------------- configuración

PROJECT_DIR = os.path.abspath(os.environ.get("PROJECT_DIR", "."))

VIDEO_PATH = os.path.join(PROJECT_DIR, "C0142.MP4")

PLATE_WEIGHTS = os.path.join(
    PROJECT_DIR,
    "checkpoint.pth"
)

PLATE_CONFIG = os.path.join(
    PROJECT_DIR,
    "rtdetr_plate.yml"
)

OUTPUT_DIR = os.path.join(PROJECT_DIR, "video_results")

OCR_LIST = MODEL_NAMES

OUT_WIDTH = 1280

IMGSZ = 640              # Tamaño de entrada de YOLO11n
PLATE_IMGSZ = 800        # Tamaño de entrada de RT-DETR (entrenamiento)

SHOW = False
SAVE = True

CLASSES = [0, 2, 3, 5]   # COCO: person, car, motorcycle, bus

VEHICLES = {"car", "bus", "motorcycle"}

PLATE_CONF = 0.25        # Confianza mínima de RT-DETR
OCR_EVERY = 10           # Fotogramas entre lecturas OCR
MIN_PLATE_CHARS = 4
PADDING = 0.05

MAX_FRAMES = None        # Por ejemplo, 300 para pruebas rápidas

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# -----------------------------------------------------------------------------


# ---------------------------------------------------------------- RT-DETR

def load_plate_detector():
    """
    Carga el RT-DETR personalizado y su postprocesador.

    Se utiliza el checkpoint entrenado y exactamente la configuración
    empleada durante el entrenamiento.
    """

    if not os.path.isfile(PLATE_WEIGHTS):
        raise FileNotFoundError(
            f"No se encuentra el checkpoint RT-DETR: {PLATE_WEIGHTS}"
        )

    if not os.path.isfile(PLATE_CONFIG):
        raise FileNotFoundError(
            f"No se encuentra la configuración RT-DETR: {PLATE_CONFIG}"
        )

    model, postprocessor = load_rtdetr(
        model_path=PLATE_WEIGHTS,
        config_path=PLATE_CONFIG,
        device=DEVICE
    )

    model.eval()

    return model, postprocessor


@torch.inference_mode()
def detect(bgr, model, postprocessor, conf=PLATE_CONF):
    """
    Detecta matrículas en una imagen BGR.

    Entrada:
        bgr: imagen OpenCV.

    Salida:
        boxes: coordenadas x1, y1, x2, y2.
        scores: confianzas.
        labels: clases detectadas.
    """

    if bgr is None or bgr.size == 0:
        return (
            np.empty((0, 4)),
            np.empty((0,)),
            np.empty((0,), dtype=int)
        )

    h, w = bgr.shape[:2]

    # Convertimos BGR -> RGB y redimensionamos a 800x800.
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    im = Image.fromarray(rgb).resize(
        (PLATE_IMGSZ, PLATE_IMGSZ),
        Image.Resampling.BILINEAR
    )

    # Convertimos a tensor [1, 3, 800, 800].
    x = torch.from_numpy(
        np.array(im)
    ).permute(2, 0, 1).float().div(255.0)

    x = x.unsqueeze(0).to(DEVICE)

    # El postprocesador necesita las dimensiones originales
    # en formato [alto, ancho], no [ancho, alto].
    orig_sizes = torch.tensor(
        [[h, w]],
        dtype=torch.float32,
        device=DEVICE
    )

    outputs = model(x)

    labels, boxes, scores = postprocessor(
        outputs,
        orig_sizes
    )

    labels = labels[0].cpu().numpy()
    boxes = boxes[0].cpu().numpy()
    scores = scores[0].cpu().numpy()

    keep = scores >= conf

    return boxes[keep], scores[keep], labels[keep]


def best_plate(vehicle_crop, model, postprocessor):
    """
    Busca matrículas dentro del recorte de un vehículo
    y devuelve la de mayor confianza.
    """

    boxes, scores, _ = detect(
        vehicle_crop,
        model,
        postprocessor,
        conf=PLATE_CONF
    )

    if len(boxes) == 0:
        return None

    index = int(np.argmax(scores))

    px1, py1, px2, py2 = boxes[index]

    return (
        px1,
        py1,
        px2,
        py2,
        float(scores[index])
    )


# -----------------------------------------------------------------------------


# ---------------------------------------------------------------- procesamiento

def process_video(ocr_name):
    """
    Procesa el vídeo completo con un OCR concreto.
    Devuelve un diccionario con el resumen.
    """

    out_video = os.path.join(
        OUTPUT_DIR,
        f"salida_rtdetr_{ocr_name}.mp4"
    )

    out_csv = os.path.join(
        OUTPUT_DIR,
        f"tracking_rtdetr_{ocr_name}.csv"
    )

    print(f"Cargando modelos para {ocr_name}...")

    # Modelo de vehículos: YOLO11n.
    vehicle_model = YOLO("yolo11n.pt")

    # Detector de matrículas: RT-DETR personalizado.
    # Se carga una instancia nueva en cada pasada.
    plate_model, postprocessor = load_plate_detector()

    # Modelo OCR.
    ocr = build_model(ocr_name)

    # -------------------------------------------------------------------------
    # Vídeo de entrada

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(
            f"No se pudo abrir el vídeo: {VIDEO_PATH}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    if src_w <= 0 or src_h <= 0:
        raise RuntimeError("Dimensiones del vídeo no válidas.")

    out_h = int(src_h * OUT_WIDTH / src_w)

    scale = OUT_WIDTH / src_w

    # -------------------------------------------------------------------------
    # Vídeo de salida

    writer = None

    if SAVE:
        writer = cv2.VideoWriter(
            out_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (OUT_WIDTH, out_h)
        )

        if not writer.isOpened():
            raise RuntimeError(
                f"No se pudo crear el vídeo: {out_video}"
            )

    # -------------------------------------------------------------------------
    # Tracking y OCR

    person_ids = set()
    vehicle_ids = set()

    # Mejor lectura de cada vehículo:
    # track_id -> {"text": ..., "det_conf": ...}
    best_reading = {}

    last_ocr_frame = {}

    ocr_calls = 0
    ocr_time = 0.0

    frame_idx = 0

    t_total = time.perf_counter()

    # CSV
    with open(
        out_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        csv_writer = csv.writer(csv_file)

        csv_writer.writerow([
            "fotograma",
            "tipo_objeto",
            "confianza_objeto",
            "identificador_tracking",
            "x1", "y1", "x2", "y2",
            "matricula_en_su_caso",
            "confianza_matricula",
            "mx1", "my1", "mx2", "my2",
            "texto_matricula"
        ])

        # YOLO realiza la detección y el tracking.
        results = vehicle_model.track(
            source=VIDEO_PATH,
            stream=True,
            persist=True,
            classes=CLASSES,
            imgsz=IMGSZ,
            verbose=False
        )

        for r in results:

            if MAX_FRAMES is not None and frame_idx >= MAX_FRAMES:
                break

            # Imagen original, sin anotaciones.
            orig = r.orig_img

            oh, ow = orig.shape[:2]

            # Imagen anotada por YOLO, redimensionada para la salida.
            frame = cv2.resize(
                r.plot(),
                (OUT_WIDTH, out_h)
            )

            # -----------------------------------------------------------------
            # Procesamiento de detecciones

            for box in r.boxes:

                if box.id is None:
                    continue

                name = vehicle_model.names[int(box.cls[0])]

                track_id = int(box.id[0])

                obj_conf = float(box.conf[0])

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                # Limitar coordenadas al tamaño original.
                x1 = max(x1, 0)
                y1 = max(y1, 0)
                x2 = min(x2, ow)
                y2 = min(y2, oh)

                plate_label = ""
                plate_conf = ""

                mx1 = my1 = mx2 = my2 = ""

                plate_text = ""

                # =============================================================
                # VEHÍCULOS
                # =============================================================

                if name in VEHICLES:

                    vehicle_ids.add(track_id)

                    vehicle_crop = orig[y1:y2, x1:x2]

                    if vehicle_crop.size > 0:

                        # Detectar matrícula mediante RT-DETR.
                        found = best_plate(
                            vehicle_crop,
                            plate_model,
                            postprocessor
                        )

                        if found is not None:

                            px1, py1, px2, py2, pconf = found

                            # Coordenadas respecto al fotograma completo.
                            mx1 = x1 + px1
                            my1 = y1 + py1
                            mx2 = x1 + px2
                            my2 = y1 + py2

                            plate_label = "plate"
                            plate_conf = pconf

                            # -------------------------------------------------
                            # OCR periódico

                            due = (
                                frame_idx -
                                last_ocr_frame.get(
                                    track_id,
                                    -OCR_EVERY
                                )
                            ) >= OCR_EVERY

                            if due:

                                bw = mx2 - mx1
                                bh = my2 - my1

                                cx1 = max(
                                    int(mx1 - bw * PADDING),
                                    0
                                )

                                cy1 = max(
                                    int(my1 - bh * PADDING),
                                    0
                                )

                                cx2 = min(
                                    int(mx2 + bw * PADDING),
                                    ow
                                )

                                cy2 = min(
                                    int(my2 + bh * PADDING),
                                    oh
                                )

                                plate_crop = orig[cy1:cy2, cx1:cx2]

                                if plate_crop.size > 0:

                                    t0 = time.perf_counter()

                                    try:
                                        text, _ = ocr.read(plate_crop)

                                    except Exception as e:
                                        text = ""

                                        print(
                                            f"Fallo de OCR en "
                                            f"fotograma {frame_idx}: {e}"
                                        )

                                    ocr_time += (
                                        time.perf_counter() - t0
                                    )

                                    ocr_calls += 1

                                    text = clean_text(text)

                                    last_ocr_frame[track_id] = frame_idx

                                    # -----------------------------------------
                                    # Guardar mejor lectura

                                    prev = best_reading.get(track_id)

                                    if (
                                        len(text) >= MIN_PLATE_CHARS
                                        and (
                                            prev is None
                                            or pconf > prev["det_conf"]
                                        )
                                    ):

                                        best_reading[track_id] = {
                                            "text": text,
                                            "det_conf": pconf
                                        }

                            # -------------------------------------------------
                            # Dibujar matrícula

                            cv2.rectangle(
                                frame,
                                (
                                    int(mx1 * scale),
                                    int(my1 * scale)
                                ),
                                (
                                    int(mx2 * scale),
                                    int(my2 * scale)
                                ),
                                (0, 0, 255),
                                2
                            )

                    # ---------------------------------------------------------
                    # Mostrar lectura sobre el vehículo

                    if track_id in best_reading:

                        plate_text = best_reading[track_id]["text"]

                        tx = int(x1 * scale)
                        ty = int(y2 * scale) + 22

                        (tw, th), _ = cv2.getTextSize(
                            plate_text,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            2
                        )

                        cv2.rectangle(
                            frame,
                            (tx, ty - th - 6),
                            (tx + tw + 6, ty + 6),
                            (0, 0, 0),
                            -1
                        )

                        cv2.putText(
                            frame,
                            plate_text,
                            (tx + 3, ty),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255),
                            2
                        )

                # =============================================================
                # PERSONAS
                # =============================================================

                elif name == "person":

                    person_ids.add(track_id)

                # =============================================================
                # CSV
                # =============================================================

                csv_writer.writerow([
                    frame_idx,
                    name,
                    obj_conf,
                    track_id,
                    x1, y1, x2, y2,
                    plate_label,
                    plate_conf,
                    mx1, my1, mx2, my2,
                    plate_text
                ])

            # -----------------------------------------------------------------
            # Información del vídeo

            cv2.rectangle(
                frame,
                (10, 10),
                (280, 90),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                frame,
                f"Vehiculos: {len(vehicle_ids)}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Personas: {len(person_ids)}",
                (20, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"OCR: {ocr_name}",
                (OUT_WIDTH - 300, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            # -----------------------------------------------------------------
            # Guardar / mostrar

            if writer is not None:
                writer.write(frame)

            if SHOW:

                cv2.imshow("Tracking", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1

    # -------------------------------------------------------------------------
    # Finalización

    total_time = time.perf_counter() - t_total

    if writer is not None:
        writer.release()

    if SHOW:
        cv2.destroyAllWindows()

    # -------------------------------------------------------------------------
    # Liberar memoria

    del ocr
    del vehicle_model
    del plate_model
    del postprocessor

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Resumen

    summary = {
        "ocr": ocr_name,
        "frames": frame_idx,
        "vehicles": len(vehicle_ids),
        "persons": len(person_ids),
        "plates_read": len(best_reading),
        "ocr_calls": ocr_calls,
        "ocr_time_s": round(ocr_time, 2),
        "ocr_time_per_call_s": (
            round(ocr_time / ocr_calls, 4)
            if ocr_calls else None
        ),
        "total_time_s": round(total_time, 2)
    }

    print(summary)

    return summary


# -----------------------------------------------------------------------------


# ---------------------------------------------------------------- ejecución

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary_path = os.path.join(
        OUTPUT_DIR,
        "video_summary.csv"
    )

    # Recuperar modelos ya procesados.
    done = {}

    if os.path.exists(summary_path):

        done = {
            row["ocr"]: row
            for row in pd.read_csv(summary_path).to_dict("records")
        }

    for name in OCR_LIST:

        expected_video = os.path.join(
            OUTPUT_DIR,
            f"salida_rtdetr_{name}.mp4"
        )

        if (
            name in done
            and (not SAVE or os.path.exists(expected_video))
        ):

            print(f"{name}: ya procesado, se salta")

            continue

        print(f"\n=== Vídeo con {name} ===")

        try:

            done[name] = process_video(name)

        except Exception as e:
            print(f"{name} falló:")
            import traceback
            traceback.print_exc()

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            continue

        # Guardar el resumen después de cada modelo.
        pd.DataFrame(
            list(done.values())
        ).to_csv(
            summary_path,
            index=False
        )

    print("\nResumen final:")

    print(
        pd.DataFrame(
            list(done.values())
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()
