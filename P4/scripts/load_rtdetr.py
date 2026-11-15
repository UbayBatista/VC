import os, sys, subprocess, urllib.request
import cv2
import numpy as np
import torch
from PIL import Image

# Carpeta del repo de RT-DETR que contiene el paquete `src`.
# Se calcula respecto a este archivo, así no depende de desde dónde lances el notebook.
_P4_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RTDETR_SRC_DIR = os.path.join(_P4_DIR, "RT-DETR", "rtdetr_pytorch")

if RTDETR_SRC_DIR not in sys.path:
    sys.path.insert(0, RTDETR_SRC_DIR)

from src.core import YAMLConfig
from src.core.yaml_utils import load_config

REPO = RTDETR_SRC_DIR
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Para usar el modelo, necesitamos descargar los pesos oficiales desde el Repositorio.
# Actualmente, existen 8 modelos entrenados, divididos en dos familias y cuatro tamaños:
# La familia RT-DETRv1 y la V2. 
# Cada familia tiene cuatro tamaños, de menor a mayor, S,M,L y X.

# (config, tag de la release, fichero de pesos COCO)
MODELS = {
    "v1": {
        "r18":  ("rtdetr/rtdetr_r18vd_6x_coco.yml",    "v0.1", "rtdetr_r18vd_dec3_6x_coco_from_paddle.pth"),
        "r34":  ("rtdetr/rtdetr_r34vd_6x_coco.yml",    "v0.1", "rtdetr_r34vd_dec4_6x_coco_from_paddle.pth"),
        "r50m": ("rtdetr/rtdetr_r50vd_m_6x_coco.yml",  "v0.1", "rtdetr_r50vd_m_6x_coco_from_paddle.pth"),
        "r50":  ("rtdetr/rtdetr_r50vd_6x_coco.yml",    "v0.1", "rtdetr_r50vd_6x_coco_from_paddle.pth"),
        "r101": ("rtdetr/rtdetr_r101vd_6x_coco.yml",   "v0.1", "rtdetr_r101vd_6x_coco_from_paddle.pth"),
    },
    "v2": {
        "r18":  ("rtdetrv2/rtdetrv2_r18vd_120e_coco.yml",  "v0.2", "rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"),
        "r34":  ("rtdetrv2/rtdetrv2_r34vd_120e_coco.yml",  "v0.1", "rtdetrv2_r34vd_120e_coco_ema.pth"),
        "r50m": ("rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml",  "v0.1", "rtdetrv2_r50vd_m_7x_coco_ema.pth"),
        "r50":  ("rtdetrv2/rtdetrv2_r50vd_6x_coco.yml",    "v0.1", "rtdetrv2_r50vd_6x_coco_ema.pth"),
        "r101": ("rtdetrv2/rtdetrv2_r101vd_6x_coco.yml",   "v0.1", "rtdetrv2_r101vd_6x_coco_from_paddle.pth"),
    },
}

SIZE_ALIASES = {"s": "r18", "m": "r50m", "l": "r50", "x": "r101"}   # nombres cortos

def load_rtdetr( family="v2", size="s", device=DEVICE, model_path=None, config_path=None ):
    if model_path is not None:
        if config_path is None:
            raise ValueError( "Para usar un modelo local debes indicar también config_path." )

        if not os.path.exists(model_path):
            raise FileNotFoundError( f"No se encuentra el modelo local: {model_path}" )

        if not os.path.exists(config_path):
            raise FileNotFoundError( f"No se encuentra la configuración: {config_path}" )

        load_config.__defaults__[0].clear()
        cfg = YAMLConfig(config_path)

        ckpt = torch.load( model_path, map_location="cpu", weights_only=True )

        if "ema" in ckpt:
            state_dict = ckpt["ema"]["module"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt

        cfg.model.load_state_dict(state_dict)

        model = cfg.model.deploy().to(device)
        postprocessor = cfg.postprocessor.deploy().to(device)

        model.eval()

        return model, postprocessor

    family = family.lower()
    size = size.lower()

    size = SIZE_ALIASES[size] if size in SIZE_ALIASES else size

    if family not in MODELS or size not in MODELS[family]:
        raise ValueError(
            f"Modelo desconocido: {family}/{size}. "
            "Inserte familia v2 o v1 y tamaño "
            "s,m,l,x o r18,r34,r50m,r50,r101"
        )

    cfg_name, tag, wfile = MODELS[family][size]

    WEIGHTS = f"weights/{wfile}"

    if not os.path.exists(WEIGHTS):
        print(
            f'No se detectaron los pesos del modelo '
            f'{family}/{size}. Se estan descargando y guardando en {WEIGHTS}'
        )

        os.makedirs("weights", exist_ok=True)

        urllib.request.urlretrieve( f"https://github.com/lyuwenyu/storage/releases/download/{tag}/{wfile}", WEIGHTS )

    load_config.__defaults__[0].clear()

    cfg = YAMLConfig(f"{REPO}/configs/{cfg_name}")

    ckpt = torch.load( WEIGHTS, map_location="cpu", weights_only=True )

    cfg.model.load_state_dict( ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"] )

    model = cfg.model.deploy().to(device)
    postprocessor = cfg.postprocessor.deploy().to(device)

    model.eval()

    return model, postprocessor