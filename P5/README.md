# Autores

- [Ubay Antonio Batista Santana](https://github.com/UbayBatista)  
- [Ángel Sergio Reyes Guedes](https://github.com/asergiorg)

# Trabajo realizado

El contenido de esta práctica está compuesto por 3 filtros que utilizan información biométrica, principalmente de la cara. El primero, es un filtro que muestra el famoso con el que compartes mayor parecido; el segundo, te permite dibujar usando la nariz y manejar esos dibujos a tu antojo; y, el tecero, te permite retarte con un compañero a un duelo al más puro estilo del viejo oeste.

## desCONOCIDOS

Para este filtro, se entrenó un extractor de características faciales a partir de un ResNet18 sobre el dataset CelebA, mediante triplets loss con un espacio vectorial sobre el que proyectar los rostros. Se explica más a detalle este proceso en la comparativa final, ya que se probaron distintos procedimientos para ver cuál daba mejor resultado.

El punto de partida del código es el proporcionado en el repositorio de la asignatura, en el fichero `detectores`. Se optó por el uso de DNN para detectar la cara ya que ofrecía una respuesta muy fluida con menos falsos positivos que los otros modelos presentados. Para el desarrollo de este, se ha realizado un pipeline similar al de la práctica anterior. En primer lugar, el modelo DNN detecta la cara, y se le pasan tanto el frame como el recuadro la función `get_celeba_crop`, que transforma el recorte original en un recorte más similar al que se puede encontrar en el dataset. Esto, con el fin de mejorar la precisión del modelo. Para ello, se calculan el ancho, altura y el centro del recuadro que el primer modelo ha detectado como cara. Luego, se calculan los nuevos alto y ancho, asignando espacio de sobra y atendiendo a las proporciones de las imágenes del dataset, para evitar que se deforme en caso de redimensionarse. Por último, se transforman las coordenadas para garantizar que se encuentran dentro del frame y se retorna el nuevo crop.

~~~python
def get_celeba_crop(frame, box):
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box

    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0

    new_h = h * 1.65
    new_w = new_h * (178.0 / 218.0)

    cy_shifted = cy - (h * 0.08)

    nx1 = int(max(0, cx - new_w / 2.0))
    ny1 = int(max(0, cy_shifted - new_h / 2.0))
    nx2 = int(min(W, cx + new_w / 2.0))
    ny2 = int(min(H, cy_shifted + new_h / 2.0))

    return frame[ny1:ny2, nx1:nx2], (nx1, ny1, nx2, ny2)
~~~

Una vez generado el nuevo crop, se le pasa a la función `get_most_similar_famous`. Esta, genera una imagen a partir del rector, que luego transofrma en un tensor que se utilizará para generar el embedding con las características esenciales de la cara en la imagen. Luego, se calcula el producto vectorial de un embedding preentrenado, para obtener resultados más rápidos, obteniendo el famoso más parecido. Por último, se configura la ruta y se añade a una cola circular en la que se almacena el resultado de los últimos 20 frames, de forma en que la imagen que se muestra corresponde a la ruta con mayor cantidad de votaciones: 

~~~python
def get_most_similar_famous(face):
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_image = Image.fromarray(face_rgb)

    face_tensor = transform(face_image).unsqueeze(0).to(device)

    with torch.no_grad():
        query_embedding = model(face_tensor)
        query_embedding = F.normalize(query_embedding, p=2, dim=1)

    similarities = (query_embedding @ gallery_embeddings.T).squeeze(0)

    best_index = torch.argmax(similarities).item()

    route = "." + filenames[best_index]
    votation.append(route)

    most_common_route, _ = Counter(votation).most_common(1)[0]
    return most_common_route
~~~

Una vez obtenida la ruta, se lee la imagen y se sobrepone en el centro de la parte superior del frame:

~~~python
overlay = cv2.imread(found)
    if overlay is not None:
        h, w = overlay.shape[:2]
        x = max(0, min((W - w) // 2, W - w))
        y = max(0, min(20, H - h))
        frame[y : y + h, x : x + w] = overlay
~~~

### Comparativa entre modelos

Ambos enfoques entrenan una arquitectura **ResNet-18** para proyectar rostros del dataset CelebA en un espacio vectorial (*embedding*) de 128 dimensiones, pero difieren radicalmente en su formulación matemática y estrategia de aprendizaje:

* **Modelo 1 (Triplet Loss aleatorio):** Utiliza aprendizaje métrico puramente contrastivo sobre ternas (*Anchor*, *Positive*, *Negative*) elegidas al azar, optimizando directamente la distancia euclidiana ($m=0.3$) sin emplear etiquetas de clase globales.

<p align="center">
    <img src="images/MODELO1.png" width="680">
</p>

* **Modelo 2 (ArcFace + Batch-Hard Triplet Loss):** Combina un clasificador angular de margen aditivo ($s=30.0$, $m=0.3$) sobre todas las identidades de entrenamiento con Hard Triplet Loss sobre lotes estructurados, aplicando normalización $L_2$ estricta y recortando la imagen sin distorsión geométrica.

<p align="center">
    <img src="images/MODELO2.png"  width="680">
</p>

---

#### 1. Análisis de las curvas de entrenamiento

* **Modelo 1:** La pérdida de entrenamiento (`Train Loss`) cae rápidamente de ~0.070 a menos de 0.005 hacia el final de las 20 épocas. Sin embargo, la pérdida de validación (`Validation Loss`) se estanca en torno a 0.031–0.045 con frecuentes oscilaciones. Esta brecha confirma que la red memoriza los casos fáciles de entrenamiento mientras los tripletes aleatorios dejan de aportar gradientes informativos para generalizar a identidades nuevas.
* **Modelo 2:** La curva muestra una dinámica controlada en dos fases:
  1. **Épocas 1 a 4 (*Warm-up* del margen):** La pérdida total (`Train Loss`) desciende inicialmente de ~9.5 a ~7.7 mientras el margen angular sube progresivamente desde $0.0$, experimentando un repunte esperado hasta ~8.5 en la época 4 al alcanzarse el margen completo.
  2. **Épocas 4 a 20 (Convergencia sostenida):** A partir de ese punto, tanto la componente angular (`Train ArcFace`, que cae de ~8.3 a ~3.05) como la componente local (`Train Triplet`, que baja suavemente de ~0.68 a ~0.19) descienden de forma monótona y sin sobreajuste abrupto, permitiendo al modelo seguir refinando la separación hasta la época 18.

<p align="center">
    <img src="images/curva_loss.png" width="480">
    <img src="images/train_losses.png"  width="480">
</p>

---

#### 2. Comparativa cuantitativa de resultados

Al cruzar los registros finales de `evaluation_metrics.csv` y `evaluation_metrics_arcface.csv`, el modelo híbrido triplica la capacidad de identificación exacta (**Recall@1**):

| Métrica / Parámetro | Modelo 1 (Triplet aleatorio) | Modelo 2 (ArcFace + Batch-Hard) | Diferencia / Impacto |
| :--- | :---: | :---: | :---: |
| **Mejor época** | 15 | 18 | Mayor aprovechamiento del entrenamiento |
| **Recall@1** | **16.10 %** (`0.1610`) | **49.24 %** (`0.4924`) | **+33.14 puntos** (3.06× superior) |
| **Recall@5** | **33.50 %** (`0.3350`) | **66.24 %** (`0.6624`) | **+32.74 puntos** (1.98× superior) |
| **Recall@10** | **41.40 %** (`0.4140`) | **73.16 %** (`0.7316`) | **+31.76 puntos** (1.77× superior) |
| **Triplet Accuracy** | 96.06 % (`0.9606`) | 97.46 % (`0.9746`) | +1.40 puntos |
| **Verificación (Acc / AUC)** | *No medida* | **92.56 %** / **0.9736** | Excelente separabilidad 1:1 |
| **Separación del espacio** | Dist. pos: `0.556` / neg: `1.276` | Sim. pos: `0.449` / neg: `0.005` | Pares negativos casi ortogonales ($\approx 0$) |
| **Protocolo de evaluación** | 1 000 queries / galería 5 000 | 5 000 queries / galería 2 031 | Evaluación más amplia y representativa |

*(Nota sobre el protocolo: en el Modelo 1 se evaluó sobre 1 000 consultas frente a 5 000 imágenes de galería, mientras que en el Modelo 2 se emplearon 5 000 consultas frente a 2 031 identidades únicas en galería; aun considerando el distinto tamaño de galería, la ganancia de +33 puntos en Recall@1 refleja una mejora estructural sustancial en la discriminación fina).*

---

#### 3. Conclusiones clave

1. **Engaño de la *Triplet Accuracy*:** Aunque el Modelo 1 alcanza un **96.06 %** de *Triplet Accuracy* (acierta al comparar un par positivo contra un único negativo aleatorio), su **Recall@1 cae al 16.10 %** al buscar a una persona entre miles de candidatos. Comparar contra un negativo al azar es demasiado fácil y no garantiza fronteras de decisión compactas, ya que el modelo acaba aprendiendo características superficiales no distintivas.
2. **Eficacia de la geometría angular:** El Modelo 2 logra separar identidades distintas de forma notoria (similitud del coseno media negativa de **`0.0046`** frente a **`0.4494`** en positivos), logrando elevar el **Recall@1 al 49.24 %** y el **Recall@10 al 73.16 %** con una precisión de verificación 1:1 del **92.56 %** y un `AUC` de 0.9736.
3. **Organización:** la forma en que se configuran los tripletes para el entrenamiento afecta notoriamente al resultado que produce cualquier extractor de características.

#### 4. Aspectos a mejorar para obtener un mejor modelo

Pese a que los resultados obtenidos con el segundo modelo fueron satisfactorios para el objetivo que se pretendía alcanzar, el resultado habría experimentado una mejora considerable si se hubieran aplicado los siguientes cambios:

1. **Curado del dataset:** el dataset utilizado posee una gran cantidad de imágenes, entre las que es frecuente encontrar imágenes mal etiquetadas, con posturas extremas u oclusión, por lo que se dificulta la identificación de algunas identidades.
2. **Enfoque del entrenamiento**: para evitar que el entorno sea determinante en el resultado de la imagen, sería ideal que, previo al entrenamiento, se extrajeran únicamente las caras de los famosos para buscar parecidos en base a estas

## Nose dibujarlo

La premisa del filtro es la siguiente: decórate a ti y a tu entorno mediante dibujos que tú mismo puedes hacer, no obstante, para hacerlo más divertido, tú único pincel será tu nariz. 

Para el desarrollo de este filtro, se ha partido del código base proporcionado en el repositorio de la asignatura. Se ha usado únicamente mediapipe, por lo que fue necesario descargar las opciones básicas de los modelos de mano y cara. A partir de este código, y con las opciones y landmarkers creados, se separó la lógica en dos bloques diferentes:

### 1. Dibujando:

Para acceder a este modo se debe mantener los ojos cerrados durante 18 frames consecutivos, saliendo de este de la misma forma. Por lo que fue necesario solicitar a la Inteligencia Artificial una función que detectara el parpadeo a partir de la relación entre la distancia vertical entre los párpados y la anchura del ojo.

~~~python
def calculate_ear(landmarks, indexes, w, h):
    # Convertir coordenadas normalizadas a píxeles
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in indexes])

    # Distancias verticales
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])

    # Distancia horizontal
    horiz = np.linalg.norm(pts[0] - pts[3])

    return (v1 + v2) / (2.0 * horiz)
~~~

Una vez dentro del modo, se sigue el landmark de la nariz para dibujar trazos que se almacenan vectorialmente, como una lista de puntos, `current_line`, que se grafica en cada frame:

~~~python
cv2.polylines(frame, [np.array(current_line, dtype=np.int32)],
                          isClosed=False, color=(0, 255, 0), thickness=5)
~~~

Cuando el usuario termina de dibujar, saliendo del modo dibujo, se almacen este dibujo en `lines`, que se muestra también con una función similar a la anterior.

### 2. Gestión de las manos

Cuando no se está dibujando, el usuario puede mover los dibujos, si existen, pudiendo arrastrarlos por la pantalla y colocárselos como accesorios en la cara. Para ello, fue necesario, a partir del landmark de la mano, separar en los landmarks de los dedos índice y pulgar, para el movimiento de pinzar '🤏', la muñeca y los nudillos, para conocer el tamaño de la palma de la mano. Con estos valores, se calcula la distancia euclídea entre los dedos y entre muñeca y nudillos, para obtener la distancia normalizada entre los dedos con respecto al tamaño de la mano, retornando las coordenadas del punto intermedio en el que se 'pellizca' para arrastrar el dibujo. Resultando en la siguiente función:

~~~python
def is_user_grabbing(landmarks, w, h, umbral=GRAB_THRESHOLD):
    thumb = landmarks[4]
    forefinger = landmarks[8]
    wrist = landmarks[0]
    media_base = landmarks[9]

    tip_distance = calculate_distance(thumb, forefinger)
    palm_size = calculate_distance(wrist, media_base)

    if palm_size == 0:
        return None, None, False

    pixel_x = int((thumb.x + forefinger.x) / 2 * w)
    pixel_y = int((thumb.y + forefinger.y) / 2 * h)
    distance = tip_distance / palm_size

    return pixel_x, pixel_y, distance < umbral
~~~

Cuando el usuario intenta agarrar un trazo, se realiza una búsqueda de la línea más próxima en un radio delimitado, de forma en que se retorna el índice de aquella línea cuya distancia euclídea es la menor al punto en que se agarró, si no se encontrara ninguna recta en el radio especificado devolvería `None`:

~~~python
def nearest_line(x, y, radius=SELECT_RADIUS):
    for i, line in enumerate(lines):
        # Distancia euclidiana de (x, y) a todos los puntos del trazo
        distancias = np.linalg.norm(line - np.array([x, y]), axis=1)
        if np.min(distancias) < radius:
            return i
    return None
~~~

Una vez capturada la línea, se considera como línea posible a usar como accesorio, por lo que se almacena en el diccionario `anchors`, que contiene los trazos que siguen a la cara del usuario. Luego, se actualiza la posición y se desplaza el dibujo junto con la mano. Por último, cuando el usuario suelta el dibujo que estaba arrastrando, se llama a la función `release_line`, generada por la IA, que, si está cerca de la cara, en este caso el parámetro frame, lo ancla a la cara aplicando las transformaciones necesarias, como rotaciones, que aplica a partir de la función `rotate`, que genera matrices 2x2 de rotación, o transformando a local las coordenadas mediante la función `to_local`:

~~~python 
def release_line(i, frame):
    if i is None or i >= len(lines) or frame is None:
        return
    origin, scale, _ = frame
    d = np.min(np.linalg.norm(lines[i] - origin, axis=1))
    if d < ATTACH_DISTANCE * scale:
        anchors[i] = to_local(lines[i].astype(np.float64), frame)

def rotation(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])

def to_local(points, frame):
    origin, scale, angle = frame
    return ((points - origin) @ rotation(angle)) / scale
~~~

Por último, con la asociación de los objetos a la cara es necesario que se actualiza la posición en cada frame para ubicarla sobre la cara, por lo que se crea la variable `frame_ref` a partir de la función `face_frame`, que genera los resultados que utilizan las funciones anteriores, y la próxima. En esta función, se calcula, a partir de los landmark de las comisuras externas de los ojos, la distancia entre los ojos, que es la escala, el origen, que es el punto medio entre el vector que une los ojos, aproximadamente el puente de la nariz, y el ángulo de este vector, para aplicar las transformaciones correspondientes en `to_local` y en `to_world`.

~~~python
def face_frame(face, w, h):
    a, b = face[33], face[263]
    pa = np.array([a.x * w, a.y * h])
    pb = np.array([b.x * w, b.y * h])
    v = pb - pa
    scale = np.linalg.norm(v)
    if scale < 1e-6:
        return None
    return (pa + pb) / 2.0, scale, math.atan2(v[1], v[0])

def to_world(local, frame):
    origin, scale, angle = frame
    return origin + scale * (local @ rotation(angle).T)
~~~

Con estas funciones, en cada frame se actualiza la posición de los dibujos anclados a la cara de la siguiente forma:

~~~python
lines[i] = to_world(local, frame_ref).astype(np.int32)
~~~

## Trabajo individual de cada miembro

# Ejecución del cuaderno

## Comandos para crear el environment con lo necesario:

```
conda env create -f environments.yml
conda activate VC_P5
```

Para poder ejecutar el cuaderno, el directorio debe tener la siguiente estructura:

```
P5
 ├── best_face_embedding_resnet18.pth
 ├── content
 │   └── datos
 │       └── dataset
 │           ├── identity_CelebA.txt
 │           └── img_align_celeba
 ├── deploy.prototxt.txt
 ├── environments.yml
 ├── gallery_embeddings.pt
 ├── README.md
 ├── res10_300x300_ssd_iter_140000.caffemodel
 └── VC_P5.ipynb
```

# Fuentes

- Enlace al dataset: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

## Conversaciones con la IA

- Para sobreponer la imagen del famoso sobre el frame: https://chatgpt.com/share/6b05e067-ece0-83eb-8341-fe86febccd56

- Para extraer la información de los landmarks de la cara en el primer filtro: https://share.gemini.google/hreKMcZP6LcJ

- Para controlar el parpadeo: https://share.gemini.google/LUzMzPup7coK

- Para anclar los dibujos a las caras: https://claude.ai/share/4d7dd9ed-3fc0-4d70-8aa1-152e76c1f8d9

- Para generar el entrenamiento del primer modelo: https://chatgpt.com/share/6b076451-f7a8-83eb-bead-1603b8875e5f

- Para generar el modelo del segundo entrenamiento: https://claude.ai/share/d8502458-ad75-48f8-adc7-0c6b8e80565d