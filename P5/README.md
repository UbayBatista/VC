# Autores

# Trabajo realizado

## desCONOCIDOS

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
conda env create -f environment.yml
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

## Conversaciones con la IA

- Para sobreponer la imagen del famoso sobre el frame: https://chatgpt.com/share/6b05e067-ece0-83eb-8341-fe86febccd56

- Para extraer la información de los landmarks de la cara en el primer filtro: https://share.gemini.google/hreKMcZP6LcJ

- Para controlar el parpadeo: https://share.gemini.google/LUzMzPup7coK

- Para anclar los dibujos a las caras: https://claude.ai/share/4d7dd9ed-3fc0-4d70-8aa1-152e76c1f8d9