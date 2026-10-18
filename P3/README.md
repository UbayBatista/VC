# Autores
[Ubay Antonio Batista Santana](https://github.com/UbayBatista)<br>
[Ángel Sergio Reyes Guedes](https://github.com/asergiorg)

# Trabajo desempeñado
## Tarea 1

Para el desarrollo de esta tarea, se ha dividido el proceso de captura de monedas en dos subproblemas: el primero, es obtener una aproximación del tamaño al que equivale cada píxel; el segundo, es obtener el tamaño de la moneda y seleccionar la que corresponda con esta aproximación.

Para solucionar el primer problema se han utilizado dos funciones comúnes que se usan para ambos problemas. La primera permite extraer los contornos, segmentando a partir de la distancia al fondo en el espacio L\*a\*b*. Solución por la que se optó para evitar problemas al transformar la imagen a blanco y negro, por los cambios de intensidad provocados por el viñeteo de la cámara o de la propia iluminación:

~~~python
def segment(frame):
    # 1. Desenfoque para reducir el ruido de croma
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    # 2. Lab en uint8 -> float32 para poder restar con signo
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.float32)
    a = lab[:, :, 1]
    b = lab[:, :, 2]

    # 3. Color del fondo: mediana de a y b en la franja de borde
    border = np.zeros(a.shape, dtype=bool)
    border[:BORDER_PX, :] = True
    border[-BORDER_PX:, :] = True
    border[:, :BORDER_PX] = True
    border[:, -BORDER_PX:] = True
    a_bg = np.median(a[border])
    b_bg = np.median(b[border])

    # 4. Distancia de crominancia al fondo
    dist = cv2.magnitude(a - a_bg, b - b_bg)

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
Una vez obtenida la imagen binarizada, se calculan los contornos a partir de la función `cv2.findContours(binarized_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)`. Tras esto, se aplica la siguiente función, que obtiene el tamaño de los objetos. En esta función, se aplica una máscara sobre el contono extraído y se calcula la transformada de la distancia, para ubicar el centro del objeto y calcular el radio. No obstante, si se encuentra en la fase de calibración se utiliza un objeto cuyo tamaño se conoce previamente y se almacena la estimación en un buffer:

~~~python
def calculate_radius(contours, binary_image, hsv, calibrating):
    total_amount = 0
    for index, contour in enumerate(contours):
        # Eliminamos ruido de contornos pequeños
        if cv2.contourArea(contour) < 100:
            continue

        # Creamos una máscara para el contorno actual y calculamos el valor máximo de la transformada de distancia dentro de esa máscara
        mask = np.zeros(binary_image.shape, dtype=np.uint8)
        cv2.drawContours(mask, contours, index, 255, thickness=-1)
        distance_image = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

        _, max_val, _, _ = cv2.minMaxLoc(distance_image, mask=mask)

        if calibrating:
            # Calculamos el tamaño de pixel a partir del radio de referencia y lo añadimos al buffer
            pixel_size_buffer.append(REFERENCE_OBJECT_MM / max_val)

        else:
            # Calculamos el radio en mm a partir del valor máximo de la transformada de distancia y el tamaño de pixel
            radius = max_val * pixel_size

            # Establecemos el grupo al que pertenece la moneda en base a su color
            measurements = meassure_color(hsv, mask, distance_image, max_val)
            group = classify_group(*measurements)

            if group is not None:
                # Clasificamos la moneda y sumamos su valor al total de la mesa
                value = classify(radius, group)
                if value is not None:
                    total_amount += value
                    print(f'Moneda: {value}€ - Radio: {radius:.2f}mm - Grupo: {group}')
                else:
                    print(f'Moneda desconocida - Radio: {radius:.2f}mm - Grupo: {group}')
    if not calibrating:
        # Añadimos el total de la mesa a la lista de lecturas
        amounts_readed.append(total_amount)
        # Se muestra el valor de la mesa como la mediana de las lecturas obtenidas
        print(f'Valor en la mesa: {np.median(amounts_readed)}')
~~~

Una vez se tienen los valores de la aproximación en el buffer, se utiliza la mediana para obtener un tamaño de píxel con la aproximación más fiel posible:

~~~python
        # Calibramos el fondo durante los primeros segundos
        if time.time() - start_time < CALIBRATION_SS:
            # Leemos el radio del contorno de referencia y establecemos un tamaño de pixel
            calculate_radius(contours, binarized_image, hsv, True)

            # Calculamos el tamaño de pixel a partir de la mediana de las lecturas previas
            pixel_size = np.median(pixel_size_buffer)

            # Mostramos los fotogramas para aportar feedback visual al usuario
            cv2.imshow('Fotograma', bgr_frame)

            # Detenemos pulsando ESC
            if cv2.waitKey(20) == 27:
                break

            continue
~~~

Para solucionar el problema de clasificación de las monedas, si la calibración ha sido exitosa, se usan funciones que extraen el color de las monedas para clasificar en los grupos `bronce`, `dorado` y `bicolor`. Una vez separadas, se usa el radio para buscar la moneda que mejor encaja en el grupo correspondiente. Si se aleja demasiado de los radios, se descarta:

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

## Tarea 2

### Conversaciones con la Inteligencia Artificial
Para generar las funciones de segmentación y extracción de color en la tarea 1:


## Aportación de cada miembro del grupo en las tareas realizadas en esta práctica
- Tarea 1: Sergio
- Tarea 2: Ubay
- README: Ubay y Sergio