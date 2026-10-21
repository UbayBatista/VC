# Autores
[Ubay Antonio Batista Santana](https://github.com/UbayBatista)<br>
[Ángel Sergio Reyes Guedes](https://github.com/asergiorg)

# Trabajo desempeñado
## Tarea 1

Para el desarrollo de esta tarea, se ha dividido el proceso de captura de monedas en dos subproblemas: el primero, es obtener una aproximación del tamaño al que equivale cada píxel; el segundo, es obtener el tamaño de la moneda y seleccionar la que corresponda con esta aproximación.

Para solucionar el primer problema se han utilizado dos funciones comunes que se usan para ambos problemas. La primera permite extraer los contornos, segmentando a partir de la distancia al fondo en el espacio L\*a\*b*. Solución por la que se optó para evitar problemas al transformar la imagen a blanco y negro, por los cambios de intensidad provocados por el viñeteo de la cámara o de la propia iluminación:

~~~python
def segment(frame):
    # 1. Desenfoque para reducir el ruido de croma
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    # 2. Lab en uint8 -> float32 para poder restar con signo
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.float32)
    l = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]

    # 3. Color del fondo: mediana de a y b en la franja de borde
    border = np.zeros(a.shape, dtype=bool)
    border[:BORDER_PX, :] = True
    border[-BORDER_PX:, :] = True
    border[:, :BORDER_PX] = True
    border[:, -BORDER_PX:] = True
    l_bg = np.median(l[border])
    a_bg = np.median(a[border])
    b_bg = np.median(b[border])

    # 4. Distancia de crominancia al fondo
    dist_chroma = cv2.magnitude(a - a_bg, b - b_bg)
    dist_l = np.maximum(np.abs(l - l_bg) - L_DEADZONE, 0) * L_WEIGHT
    dist = np.maximum(dist_chroma, dist_l)

    # 5. Escalado con rango fijo a uint8
    dist_u8 = np.clip(dist * (255.0 / DIST_MAX), 0, 255).astype(np.uint8)

    # 6. Si no hay señal suficiente, devolvemos máscara vacía (evita segmentar ruido)
    if np.percentile(dist, 99.5) < MIN_SIGNAL:
        return np.zeros(a.shape, dtype=np.uint8), dist_u8

    # 7. Otsu con objetos en blanco (sin INV)
    _, binary = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 8. Cierre para rellenar huecos (logo, parte plateada) y apertura para quitar motas
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, CLOSE_KERNEL)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, OPEN_KERNEL)

    return binary, dist_u8
~~~
Una vez obtenida la imagen binarizada, se calculan los contornos, a partir de la función `cv2.findContours(binarized_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)`, y se calcula la transformada de la distancia, para hallar la distancia al borde de los objetos desde su centro. Esto se hace con las siguientes funciones:

~~~python
def create_countour_elements(binary_image, index):
    # Creamos una máscara para el contorno actual y calculamos el valor máximo de la transformada de distancia dentro de esa máscara
    mask = np.zeros(binary_image.shape, dtype=np.uint8)
    cv2.drawContours(mask, contours, index, 255, thickness=-1)
    distance_image = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    return mask, distance_image

def find_distance_from_center_px(contour, binary_image, index):
    # Eliminamos ruido de contornos pequeños
    if cv2.contourArea(contour) < 100:
        return None

    mask, distance_image = create_countour_elements(binary_image, index)

    _, distance_from_center, _, _ = cv2.minMaxLoc(distance_image, mask=mask)

    return distance_from_center
~~~
 
Estas funciones se llaman desde la de calibración, en la que se almacenan en el buffer las estimaciones del tamaño del píxel:

~~~python
def calibrate(contours, binary_image):
    for index, contour in enumerate(contours):
        distance = find_distance_from_center_px(contour, binary_image, index)

        if distance:
            # Calculamos el tamaño de pixel a partir del radio de referencia y lo añadimos al buffer
            pixel_size_buffer.append(REFERENCE_OBJECT_MM / distance)
~~~

Una vez se tienen los valores de la aproximación en el buffer, se utiliza la mediana para obtener un tamaño de píxel con la aproximación más fiel posible:

~~~python
        # Calibramos el fondo durante los primeros segundos
        if time.time() - start_time < CALIBRATION_SS:
            # Leemos el radio del contorno de referencia y establecemos un tamaño de pixel
            calibrate(contours, binarized_image)

            # Calculamos el tamaño de pixel a partir de la mediana de las lecturas previas
            pixel_size = np.median(pixel_size_buffer)

            # Mostramos los fotogramas para aportar feedback visual al usuario
            cv2.imshow('Fotograma', bgr_frame)

            # Detenemos pulsando ESC
            if cv2.waitKey(20) == 27:
                break

            continue
~~~

Para solucionar el problema de clasificación de las monedas, si la calibración ha sido exitosa, se usan funciones que extraen el color de las monedas para clasificarlas en los grupos `bronce`, `dorado` y `bicolor`. Una vez separadas, se usa el radio para buscar la moneda que mejor encaja en el grupo correspondiente. Si se aleja demasiado de los radios, se descarta:

~~~python
def meassure_color(hsv, mask, dist_img, max_val):
    # Píxeles lejos del borde
    center = ((dist_img > 0.75 * max_val) & (mask > 0)).astype(np.uint8) * 255
    # Píxeles cerca del borde, pero sin tocarlo
    ring = ((dist_img > 0.08 * max_val) & (dist_img < 0.40 * max_val)
              & (mask > 0)).astype(np.uint8) * 255

    h_c, s_c, _, _ = cv2.mean(hsv, mask=center)
    h_r, s_r, _, _ = cv2.mean(hsv, mask=ring)
    return (h_c, s_c), (h_r, s_r)

def classify_group(centro, anillo, umbral_bimetal=12):
    (h_c, s_c), (h_r, s_r) = centro, anillo

    # Monedas con dos tonos (1€ y 2€) tienen un umbral de diferencia de tono más alto
    if abs(h_c - h_r) > umbral_bimetal or abs(s_c - s_r) > 40:
        return 'bimetal'

    # Monedas de un solo tono se clasifican en cobre o dorado según el valor medio del tono
    h = (h_c + h_r) / 2
    return 'cobre' if h < H_LIMITE else 'dorado'

def classify(radius, group):
    # Buscamos la moneda más cercana al radio medido para el grupo determinado
    key = min(MONEY[group], key=lambda k: abs(k - radius))
    if abs(key - radius) > TOLERANCE_MM:
        return None          # no encaja con ninguna moneda del grupo
    return MONEY[group][key]
~~~

### Añadido a la tarea

Como parte extra de esta tarea, se ha modificado el código anterior para permitir el cálculo de monedas y billetes simultáneamente. Para ello, se ha añadido al diccionario de dinero las medidas de las dimensiones de los billetes como clave y sus valores asociados en el grupo `billete`, y se crearon las siguientes funciones, con el fin de identificar el billete en base a su tamaño (se ha usado la relación de aspecto para evitar la lectura de elementos que no son billetes):

~~~python
def classify_bill(short, long, group):
    # Buscamos el billete cuyo tamaño sea el más cercano al del contorno detectado
    key = min(MONEY[group], key=lambda k: abs(k[0] - short + k[1] - long))
    if abs(key[0] - short) > TOLERANCE_BILL_MM[0] and abs(key[1] - long) > TOLERANCE_BILL_MM[1]:
        return None          # no encaja con ningún billete
    return MONEY[group][key]

def get_bill_shape(contour):
    # Obtenemos el ancho y largo del contorno del billete
    _, (w, h), _ = cv2.minAreaRect(contour)

    # Ordenamos w y h para obtener siempre el mismo orden, sin importar el ángulo
    short_px, long_px = sorted((w, h))

    # Conversión a mm con la escala de la calibración
    short_mm = short_px * pixel_size
    long_mm = long_px * pixel_size

    # Relación de aspecto (con protección frente a división por cero)
    aspect = long_px / short_px if short_px > 0 else 0

    return short_mm, long_mm, aspect
~~~

Por último, se modificó la función original de `get_amount` para clasificar como billete aquellos contornos cuyo radio supere los 0.2 mm, resultando de la siguiente forma:

~~~python
def get_amount(contours, binary_image, hsv):
    total_amount = 0
    for index, contour in enumerate(contours):
        distance = find_distance_from_center_px(contour, binary_image, index)

        if distance is None or distance <= 0:
            continue

        mask, distance_image = create_countour_elements(binary_image, index)

        # Calculamos el radio en mm a partir del valor máximo de la transformada de distancia y el tamaño de pixel
        radius = distance * pixel_size

        if radius > 20:
            short_mm, long_mm, aspect = get_bill_shape(contour)

            # Utilizamos la relación de aspectos para evitar leer elementos que no sean similares a billetes
            if aspect < 1.7 or aspect > 2.05:
                continue

            value = classify_bill(short_mm, long_mm, 'billete')

            if value is not None:
                total_amount += value
                print(f'Billete: {value}€ - Tamaño: {short_mm:.2f}x{long_mm:.2f}mm')
            else:
                print(f'Billete desconocido - Tamaño: {short_mm:.2f}x{long_mm:.2f}mm')
        elif radius > 7:
            # Establecemos el grupo al que pertenece la moneda en base a su color
            measurements = meassure_color(hsv, mask, distance_image, distance)
            group = classify_group(*measurements)

            if group is not None:
                # Clasificamos la moneda y sumamos su valor al total de la mesa
                value = classify(radius, group)
                if value is not None:
                    total_amount += value
                    print(f'Moneda: {value}€ - Radio: {radius:.2f}mm - Grupo: {group}')
                else:
                    print(f'Moneda desconocida - Radio: {radius:.2f}mm - Grupo: {group}')

    # Añadimos el total de la mesa a la lista de lecturas
    amounts_read.append(total_amount)
    # Se muestra el valor de la mesa como la mediana de las lecturas obtenidas
    print(f'Valor en la mesa: {np.median(amounts_read)}')
~~~

Vídeo demo:

[▶️ Ver vídeo del funcionamiento](https://drive.google.com/file/d/1tWXhc6PlrUwpVhcqW-xvLlX8S9mNnN9S/view?usp=sharing)

## Tarea 2

### Conversaciones con la Inteligencia Artificial
Para generar las funciones de segmentación y extracción de color en la tarea 1:


## Aportación de cada miembro del grupo en las tareas realizadas en esta práctica
- Tarea 1: Sergio
- Tarea 2: Ubay
- README: Ubay y Sergio