# Autores
Ubay Antonio Batista Santana  
Ángel Sergio Reyes Guedes

# Trabajo desempeñado
En esta sección incluiremos el desempeño individual de cada miembro del grupo en las tareas realizadas en esta práctica.

## Tarea 1
Para esta tarea, Ubay aportó el código generado manualmente y Sergio fue el encargado de solicitar a la Inteligencia Artificial el código para generar el tablero de ajedrez. Obteniendo los siguientes resultados:

<p align="center">
<img src="images/ajedrez_manual.png" width="300" style="margin-right: 30px;">
<img src="images/ajedrez_ia.png" width="300">
</p>


Como se puede observar, se obtuvieron resultados diferentes, tanto a nivel algorítmico como visual. La Inteligencia Artificial, en este caso, se excedió y añadió colores al tablero de ajedrez. No obstante, se puede ver cierto parecido en la comprobación del color que se hace tanto en el código manual como en el que genera la Inteligencia Artificial, ya que ambos comprueban si la suma de la fila y de la columna es par para decidir si pintar de claro u oscuro.

Conversación con la Inteligencia Artificial: https://claude.ai/share/1b370e5f-102c-4093-bbb3-1cbde082c3d1

## Tarea 2

El código que genera esta imagen, que sigue el estilo Mondrian, lo aportó Sergio, que optó por un conjunto de rectángulos sencillo que imita la obra original aportado en la práctica, pero con líneas menos simétricas y una paleta de colores más sencilla.

<p align="center">
<img src="images/arte_mondrian.png" width="300">
</p>

## Tarea 3

Para el desarrollo de esta tarea, Sergio consultó con Inteligencia Artificial la forma más eficiente de encontrar los máximos y mínimos dentro de una matriz. Como resultado de esta conversación se optó por un algoritmo que recorriera los píxeles de la matriz comprobando el píxel actual con el de la misma columna, pero en la siguiente fila, para decidir cual era el mayor y menor de esta pareja y compararlos con los mínimos y máximos encontrados hasta el momento, reduciendo las comparaciones de 2N, siendo N el número de elementos de la matriz, a 1.5N. No obstante, tras finalizar el desarrollo del algoritmo de búsqueda, el agente de Copilot nos sugirió sustituirlo por las funciones optimizadas de numpy. Resultando en una experiencia fluida y sin saltos. Ubay arregló un problema con el tamaño y color de la imagen tras el redimensionado que se usó para simplificar la imagen y reducir los datos a procesar.

Conversación con la Inteligencia Artificial: https://share.gemini.google/TMVdLhjfAJ9t

Sugerencia de cambio del algoritmo:

<p align="center">
<img src="images/sugerencia_ia.png" width="500">
</p>

## Tarea 4

Durante la implementación de este apartado, Sergio aportó el código que dividía la pantalla en distintos frames para el diseño del Pop Art, para lo que se utilizó la conversación que se incluye más abajo. Ubay solucionó un error con la escala de las imágenes y aplicó las transformaciones de los colores.

Conversación con la Inteligencia Artificil: https://share.gemini.google/8Nasadt5PnZ2

## Tarea 5

Esta tarea se trata de un añadido a la práctica original que tuvimos la idea de implementar. Se trata de un modulador de los canales de colores mediante el ratón, pudiendo crear tu propio filtro de color. El funcionamiento del mismo es el siguiente: se usan los botones del ratón para cambiar entre los canales de la imagen, y con la rueda se puede modificar el valor de dicho canal. Para ello, Sergio estableció la base del funcionamiento, creando la función que permitía capturar el movimiento del ratón, para lo que usó la conversación que se incluye a continuación, y la comunicación de la lectura de imagen y el ratón en tiempo real, y Ubay desarrolló la transición de canales mediante los botones del ratón y adaptó la solución inicialmente aportada, ya que no contemplaba cambiar el canal dinámicamente.

Conversación con la Inteligencia Artificial: https://share.gemini.google/odFoPvnl4YnD

## A la hora de ejecutar

Durante el desarrollo de esta práctica, se tuvo que añadir la siguiente opción para permitir la ejecución del cuaderno en Jupyter Notebook:
```
cv2.CAP_DSHOW
```
En la siguiente línea, resultando de esta manera:

```
vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
```

# Fuentes
Códigos RGB: https://htmlcolorcodes.com/es/