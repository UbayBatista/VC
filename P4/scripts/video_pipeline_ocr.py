"""
Detección + tracking + matrículas + OCR sobre el vídeo de test, repetido para cada OCR.
Por cada modelo genera:
    salida_<modelo>.mp4      vídeo anotado con la matrícula leída
    tracking_<modelo>.csv    csv de volcado con la columna texto_matricula
y al final un resumen con el tiempo empleado por cada uno (video_summary.csv).

Se ejecuta desde Colab con la carpeta del proyecto en Drive (ver celdas del cuaderno
benchmark_ocr) o con `python video_pipeline_ocr.py`.
"""

import csv
import gc
import os
import time

import cv2
import pandas as pd
import torch
from ultralytics import YOLO

from ocr_models import build_model, clean_text, MODEL_NAMES

# ---------------------------------------------------------------- configuración
PROJECT_DIR = os.environ.get("PROJECT_DIR", ".")
VIDEO_PATH = os.path.join(PROJECT_DIR, "C0142.MP4")
PLATE_WEIGHTS = os.path.join(PROJECT_DIR, "best.pt")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "video_results")

OCR_LIST = MODEL_NAMES    # se pueden quitar modelos de la lista para probar con uno solo

OUT_WIDTH = 1280
IMGSZ = 640
SHOW = False              # True si hay entorno gráfico (en Colab siempre False)
SAVE = True

CLASSES = [0, 2, 3, 5]    # COCO: person, car, motorcycle, bus
VEHICLES = {"car", "bus", "motorcycle"}

PLATE_CONF = 0.25         # confianza mínima del detector de matrículas
OCR_EVERY = 10            # cada cuántos fotogramas se vuelve a leer la matrícula de un vehículo
MIN_PLATE_CHARS = 4       # lecturas más cortas se descartan por ser ruido
PADDING = 0.05            # margen al recortar la matrícula
MAX_FRAMES = None         # límite de fotogramas a analizar. None procesa todo el vídeo.
# -----------------------------------------------------------------------------


def best_plate(plate_model, vehicle_crop):
    """Busca matrículas dentro del recorte del vehículo y devuelve la de mayor confianza."""
    res = plate_model.predict(vehicle_crop, imgsz=IMGSZ, conf=PLATE_CONF, verbose=False)[0]
    if len(res.boxes) == 0:
        return None
    i = int(res.boxes.conf.argmax())
    px1, py1, px2, py2 = map(int, res.boxes.xyxy[i])
    return px1, py1, px2, py2, float(res.boxes.conf[i])


def process_video(ocr_name):
    """Procesa el vídeo completo con un OCR concreto. Devuelve un diccionario con el resumen."""
    out_video = os.path.join(OUTPUT_DIR, f"salida_{ocr_name}.mp4")
    out_csv = os.path.join(OUTPUT_DIR, f"tracking_{ocr_name}.csv")

    # Se recrean los modelos en cada experimento para reiniciar el estado del tracker
    # y garantizar que todos los OCR se evalúan bajo las mismas condiciones.
    vehicle_model = YOLO("yolo11n.pt")
    plate_model = YOLO(PLATE_WEIGHTS)
    ocr = build_model(ocr_name)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    out_h = int(src_h * OUT_WIDTH / src_w)
    scale = OUT_WIDTH / src_w          # para pasar las coordenadas originales al vídeo de salida

    writer = None
    if SAVE:
        writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (OUT_WIDTH, out_h))

    person_ids, vehicle_ids = set(), set()

    # mejor lectura de cada vehículo hasta el momento: track_id -> dict
    best_reading = {}
    last_ocr_frame = {}

    ocr_calls = 0
    ocr_time = 0.0

    csv_file = open(out_csv, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["fotograma", "tipo_objeto", "confianza", "identificador_tracking",
                         "x1", "y1", "x2", "y2", "matrícula_en_su_caso", "confianza",
                         "mx1", "my1", "mx2", "my2", "texto_matricula"])

    results = vehicle_model.track(source=VIDEO_PATH, stream=True, persist=True,
                                  classes=CLASSES, imgsz=IMGSZ, verbose=False)

    t_total = time.perf_counter()
    frame_idx = 0
    for r in results:
        if MAX_FRAMES is not None and frame_idx >= MAX_FRAMES:
            break

        orig = r.orig_img                      # fotograma original, sin cajas dibujadas
        oh, ow = orig.shape[:2]

        frame = cv2.resize(r.plot(), (OUT_WIDTH, out_h))

        for box in r.boxes:
            if box.id is None:
                continue

            name = vehicle_model.names[int(box.cls[0])]
            track_id = int(box.id[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, ow), min(y2, oh)

            plate_label, plate_conf = "", ""
            mx1 = my1 = mx2 = my2 = ""
            plate_text = ""

            if name in VEHICLES:
                vehicle_ids.add(track_id)
                vehicle_crop = orig[y1:y2, x1:x2]

                if vehicle_crop.size > 0:
                    found = best_plate(plate_model, vehicle_crop)

                    if found is not None:
                        px1, py1, px2, py2, pconf = found

                        # coordenadas de la matrícula respecto al fotograma completo
                        mx1, my1, mx2, my2 = x1 + px1, y1 + py1, x1 + px2, y1 + py2
                        plate_label, plate_conf = "plate", pconf

                        # limita la frecuencia de OCR por vehículo, ya que procesar la matrícula en
                        # cada fotograma aumentaría el coste computacional sin aportar información relevante.
                        due = frame_idx - last_ocr_frame.get(track_id, -OCR_EVERY) >= OCR_EVERY
                        if due:
                            bw, bh = mx2 - mx1, my2 - my1
                            cx1 = max(int(mx1 - bw * PADDING), 0)
                            cy1 = max(int(my1 - bh * PADDING), 0)
                            cx2 = min(int(mx2 + bw * PADDING), ow)
                            cy2 = min(int(my2 + bh * PADDING), oh)
                            plate_crop = orig[cy1:cy2, cx1:cx2]

                            if plate_crop.size > 0:
                                t0 = time.perf_counter()
                                try:
                                    text, _ = ocr.read(plate_crop)
                                except Exception as e:
                                    text = ""
                                    print(f"  fallo de OCR en fotograma {frame_idx}: {e}")
                                ocr_time += time.perf_counter() - t0
                                ocr_calls += 1

                                text = clean_text(text)
                                last_ocr_frame[track_id] = frame_idx

                                # actualizamos la mejor lectura de la matrícula si el texto es válido y la detección es más fiable
                                prev = best_reading.get(track_id)
                                if len(text) >= MIN_PLATE_CHARS and (prev is None or pconf > prev["det_conf"]):
                                    best_reading[track_id] = {"text": text, "det_conf": pconf}

                        # dibujamos la caja de la matrícula en el vídeo de salida
                        cv2.rectangle(frame, (int(mx1 * scale), int(my1 * scale)),
                                      (int(mx2 * scale), int(my2 * scale)), (0, 0, 255), 2)

                # mostrar la mejor matrícula reconocida para este vehículo
                if track_id in best_reading:
                    plate_text = best_reading[track_id]["text"]
                    tx, ty = int(x1 * scale), int(y2 * scale) + 22
                    (tw, th), _ = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(frame, (tx, ty - th - 6), (tx + tw + 6, ty + 6), (0, 0, 0), -1)
                    cv2.putText(frame, plate_text, (tx + 3, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            elif name == "person":
                person_ids.add(track_id)

            csv_writer.writerow([frame_idx, name, conf, track_id, x1, y1, x2, y2,
                                 plate_label, plate_conf, mx1, my1, mx2, my2, plate_text])

        # contadores y nombre del OCR usado
        cv2.rectangle(frame, (10, 10), (280, 90), (0, 0, 0), -1)
        cv2.putText(frame, f"Vehiculos: {len(vehicle_ids)}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Personas: {len(person_ids)}", (20, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"OCR: {ocr_name}", (OUT_WIDTH - 300, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if writer:
            writer.write(frame)

        if SHOW:
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    total_time = time.perf_counter() - t_total
    csv_file.close()
    if writer:
        writer.release()
    if SHOW:
        cv2.destroyAllWindows()

    # liberar memoria antes de cargar el siguiente modelo
    del ocr, vehicle_model, plate_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    summary = {
        "ocr": ocr_name,
        "frames": frame_idx,
        "vehicles": len(vehicle_ids),
        "persons": len(person_ids),
        "plates_read": len(best_reading),
        "ocr_calls": ocr_calls,
        "ocr_time_s": round(ocr_time, 2),
        "ocr_time_per_call_s": round(ocr_time / ocr_calls, 4) if ocr_calls else None,
        "total_time_s": round(total_time, 2),
    }
    print(summary)
    return summary


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_path = os.path.join(OUTPUT_DIR, "video_summary.csv")

    # si existe un resumen previo, cargamos los resultados ya obtenidos para
    # reanudar el benchmark sin repetir los modelos completados
    done = {}
    if os.path.exists(summary_path):
        done = {row["ocr"]: row for row in pd.read_csv(summary_path).to_dict("records")}

    for name in OCR_LIST:
        if name in done and os.path.exists(os.path.join(OUTPUT_DIR, f"salida_{name}.mp4")):
            print(f"{name}: ya procesado, se salta")
            continue
        print(f"\n=== Vídeo con {name} ===")
        try:
            done[name] = process_video(name)
        except Exception as e:
            print(f"{name} falló: {e}")
            continue
        # guardamos tras cada modelo por si se cae la sesión
        pd.DataFrame(list(done.values())).to_csv(summary_path, index=False)

    print("\nResumen final:")
    print(pd.DataFrame(list(done.values())).to_string(index=False))


if __name__ == "__main__":
    main()
