"""
Envoltorios (wrappers) para los cinco OCR de la práctica.

Todos exponen la misma interfaz:
    model = build_model("easyocr")
    text, conf = model.read(image)   # image: ruta (str) o array BGR de OpenCV

Así el cuaderno de la comparativa y el script del vídeo usan exactamente
el mismo código para cada modelo.
"""

import os
import re
import sys
import glob
import shutil
import tempfile

import cv2
import numpy as np
import torch

MODEL_NAMES = ["easyocr", "tesseract", "paddleocr", "unlimited_ocr", "smolvlm"]

HAS_CUDA = torch.cuda.is_available()


def clean_text(text):
    """Deja solo letras y números en mayúsculas, para comparar con el texto real."""
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def _as_bgr(image):
    """Devuelve la imagen como array BGR, venga de ruta o de array."""
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(image)
        return img
    return image


def _as_path(image, tmp_dir):
    """Devuelve una ruta a disco; si nos pasan un array lo guardamos en tmp_dir."""
    if isinstance(image, str):
        return image
    path = os.path.join(tmp_dir, "tmp_input.jpg")
    cv2.imwrite(path, image)
    return path


def _find_tesseract(explicit=None):
    """
    Localiza el ejecutable de Tesseract sin depender de una ruta fija.
    Orden: argumento -> variable TESSERACT_CMD -> PATH -> conda -> Program Files.
    Devuelve la ruta o None si no se encuentra.
    """
    candidates = [explicit, os.environ.get("TESSERACT_CMD"), shutil.which("tesseract")]

    exe = "tesseract.exe" if os.name == "nt" else "tesseract"
    prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
    candidates += [
        os.path.join(prefix, "Library", "bin", exe),   # conda-forge en Windows
        os.path.join(prefix, "bin", exe),              # conda en Linux/macOS
    ]
    if os.name == "nt":
        candidates += [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]

    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return None


def _find_tessdata(prefix):
    """Busca una carpeta tessdata que contenga datos de idioma dentro del entorno."""
    for sub in (("Library", "share", "tessdata"), ("share", "tessdata"),
                ("Library", "bin", "tessdata")):
        d = os.path.join(prefix, *sub)
        if os.path.isdir(d):
            return d
    return None


class BaseOCR:
    name = "base"
    device = "cpu"

    def load(self):
        raise NotImplementedError

    def read(self, image):
        """Devuelve (texto, confianza). La confianza es None si el modelo no la da."""
        raise NotImplementedError


# ----------------------------------------------------------------------------
# 1. EasyOCR
# ----------------------------------------------------------------------------
class EasyOCRModel(BaseOCR):
    name = "easyocr"

    def __init__(self, use_gpu=HAS_CUDA):
        self.use_gpu = use_gpu
        self.device = "cuda" if use_gpu else "cpu"

    def load(self):
        import easyocr
        self.reader = easyocr.Reader(["es"], gpu=self.use_gpu, verbose=False)

    def read(self, image):
        # EasyOCR acepta tanto la ruta como el array directamente
        res = self.reader.readtext(image)
        if not res:
            return "", None
        text = " ".join(r[1] for r in res)
        conf = float(np.mean([r[2] for r in res]))
        return text, conf


# ----------------------------------------------------------------------------
# 2. Tesseract (solo CPU)
# ----------------------------------------------------------------------------
class TesseractModel(BaseOCR):
    name = "tesseract"
    device = "cpu"

    def __init__(self, tesseract_cmd=None, lang="spa"):
        self.tesseract_cmd = tesseract_cmd
        self.lang = lang

    def load(self):
        import pytesseract
        self.pytesseract = pytesseract

        cmd = _find_tesseract(self.tesseract_cmd)
        if cmd is None:
            raise RuntimeError(
                "No se encuentra Tesseract. Instálalo con "
                "'conda install -c conda-forge tesseract' o define la variable "
                "de entorno TESSERACT_CMD con la ruta de tesseract.exe."
            )
        pytesseract.pytesseract.tesseract_cmd = cmd

        # Si no hay TESSDATA_PREFIX, intentamos usar la carpeta del entorno conda
        if not os.environ.get("TESSDATA_PREFIX"):
            prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
            tessdata = _find_tessdata(prefix)
            if tessdata:
                os.environ["TESSDATA_PREFIX"] = tessdata

        # Comprobamos ahora que funciona, para que el error salga al cargar
        # y no una vez por fotograma
        try:
            pytesseract.get_tesseract_version()
            available = pytesseract.get_languages(config="")
        except Exception as e:
            raise RuntimeError(f"Tesseract no responde ({cmd}): {e}") from e

        if self.lang not in available:
            print(f"[tesseract] Falta el idioma '{self.lang}' (disponibles: {available}). "
                  f"Se usa 'eng'. Copia {self.lang}.traineddata en tessdata para arreglarlo.")
            self.lang = "eng" if "eng" in available else available[0]

        print(f"[tesseract] usando {cmd} (idioma: {self.lang})")

    def read(self, image):
        rgb = cv2.cvtColor(_as_bgr(image), cv2.COLOR_BGR2RGB)
        # psm 7: tratar la imagen como una sola línea de texto (una matrícula lo es)
        text = self.pytesseract.image_to_string(rgb, lang=self.lang, config="--psm 7")
        return text.strip(), None


# ----------------------------------------------------------------------------
# 3. PaddleOCR
# ----------------------------------------------------------------------------
class PaddleOCRModel(BaseOCR):
    name = "paddleocr"

    def __init__(self, use_gpu=False):
        # Con paddlepaddle (versión CPU) use_gpu debe quedarse en False
        self.use_gpu = use_gpu
        self.device = "cuda" if use_gpu else "cpu"

    def load(self):
        import paddleocr
        from paddleocr import PaddleOCR

        # La API cambió en la versión 3: lo detectamos
        self.major = int(paddleocr.__version__.split(".")[0])

        if self.major >= 3:
            # Solo reconocimiento de texto: sin corrección de documento ni orientación de línea
            kwargs = dict(lang="es",
                          use_doc_orientation_classify=False,
                          use_doc_unwarping=False,
                          use_textline_orientation=False)
            try:
                # mkldnn da errores en algunas versiones de paddle 3 con CPU
                self.ocr = PaddleOCR(enable_mkldnn=False, **kwargs)
            except TypeError:
                self.ocr = PaddleOCR(**kwargs)
        else:
            self.ocr = PaddleOCR(use_angle_cls=True, lang="es",
                                 use_gpu=self.use_gpu, show_log=False)

    def read(self, image):
        if self.major >= 3:
            result = self.ocr.predict(image)
            if not result:
                return "", None
            texts = result[0]["rec_texts"]
            scores = result[0]["rec_scores"]
            if len(texts) == 0:
                return "", None
            return " ".join(texts), float(np.mean(scores))

        result = self.ocr.ocr(image, cls=True)
        # Si no detecta nada, result[0] viene como None
        if not result or result[0] is None:
            return "", None
        lines = result[0]
        text = " ".join(line[1][0] for line in lines)
        conf = float(np.mean([line[1][1] for line in lines]))
        return text, conf


# ----------------------------------------------------------------------------
# 4. Unlimited-OCR (Baidu)
# ----------------------------------------------------------------------------
class UnlimitedOCRModel(BaseOCR):
    name = "unlimited_ocr"

    def __init__(self, device="cuda" if HAS_CUDA else "cpu"):
        self.device = device
        self.out_dir = tempfile.mkdtemp(prefix="unlimited_out_")
        self.tmp_dir = tempfile.mkdtemp(prefix="unlimited_in_")

    def load(self):
        from transformers import AutoModel, AutoTokenizer

        use_cuda = self.device == "cuda" and torch.cuda.is_available()

        # El código del modelo llama a .cuda() por dentro; sin GPU lo anulamos
        if not use_cuda:
            torch.Tensor.cuda = lambda self, *a, **k: self
            torch.nn.Module.cuda = lambda self, *a, **k: self

        model_id = "baidu/Unlimited-OCR"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_safetensors=True,
            # bfloat16 en CPU es muy lento; float32 es lo razonable sin GPU
            torch_dtype=torch.bfloat16 if use_cuda else torch.float32,
        )
        self.model = self.model.eval().to("cuda" if use_cuda else "cpu")

    @staticmethod
    def _strip_markup(text):
        # El modelo puede devolver etiquetas de grounding tipo <|ref|>..<|/ref|><|det|>..<|/det|>
        text = re.sub(r"<\|ref\|>.*?<\|/ref\|>", "", text, flags=re.S)
        text = re.sub(r"<\|det\|>.*?<\|/det\|>", "", text, flags=re.S)
        text = re.sub(r"<\|.*?\|>", "", text, flags=re.S)
        return text.strip()

    def _read_saved_output(self):
        # infer() con save_results=True deja el resultado en un fichero de texto
        files = []
        for ext in ("*.mmd", "*.md", "*.txt"):
            files += glob.glob(os.path.join(self.out_dir, "**", ext), recursive=True)
        if not files:
            return ""
        newest = max(files, key=os.path.getmtime)
        with open(newest, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def read(self, image):
        path = _as_path(image, self.tmp_dir)

        # Limpiamos la carpeta de salida para no leer el resultado de la imagen anterior
        shutil.rmtree(self.out_dir, ignore_errors=True)
        os.makedirs(self.out_dir, exist_ok=True)

        ret = self.model.infer(
            self.tokenizer,
            prompt="<image>document parsing.",
            image_file=path,
            output_path=self.out_dir,
            base_size=1024, image_size=640, crop_mode=True,
            max_length=32768,
            no_repeat_ngram_size=35, ngram_window=128,
            save_results=True,
        )

        # Según la versión, infer puede devolver el texto o solo guardarlo en disco
        text = ret if isinstance(ret, str) and ret.strip() else self._read_saved_output()
        return self._strip_markup(text), None


# ----------------------------------------------------------------------------
# 5. SmolVLM
# ----------------------------------------------------------------------------
class SmolVLMModel(BaseOCR):
    name = "smolvlm"

    PROMPT = ("Read the license plate in the image. "
              "Answer only with the characters of the plate, nothing else.")

    def __init__(self, device="cuda" if HAS_CUDA else "cpu"):
        self.device = device

    def load(self):
        from transformers import AutoProcessor, AutoModelForImageTextToText

        model_id = "HuggingFaceTB/SmolVLM-Instruct"
        # Sin dividir la imagen en trozos: la matrícula es pequeña y así va mucho más rápido
        self.processor = AutoProcessor.from_pretrained(model_id, do_image_splitting=False)

        use_cuda = self.device == "cuda"
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=torch.float16 if use_cuda else torch.float32,
            _attn_implementation="sdpa" if use_cuda else "eager",
        ).to(self.device)
        self.model.eval()

    def read(self, image):
        from PIL import Image

        rgb = cv2.cvtColor(_as_bgr(image), cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": self.PROMPT}],
        }]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[pil_img], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=15, do_sample=False)

        # Nos quedamos solo con los tokens nuevos; si no, el texto incluye el prompt
        new_tokens = generated[:, inputs["input_ids"].shape[1]:]
        text = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return text.strip(), None


# ----------------------------------------------------------------------------
def build_model(name, **kwargs):
    """Crea y carga el modelo pedido por nombre."""
    classes = {
        "easyocr": EasyOCRModel,
        "tesseract": TesseractModel,
        "paddleocr": PaddleOCRModel,
        "unlimited_ocr": UnlimitedOCRModel,
        "smolvlm": SmolVLMModel,
    }
    if name not in classes:
        raise ValueError(f"Modelo desconocido: {name}. Opciones: {list(classes)}")
    model = classes[name](**kwargs)
    model.load()
    return model