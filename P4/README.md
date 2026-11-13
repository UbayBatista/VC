# Autores
[Ubay Antonio Batista Santana](https://github.com/UbayBatista)<br>
[Ángel Sergio Reyes Guedes](https://github.com/asergiorg)

# Trabajo realizado

Para el desarrollo de esta práctica se optó por dividir el trabajo en dos líneas independientes de desarrollo para optimizar el tiempo de trabajo.

## Entrenamiento de los modelos

### Curado del dataset

En primer lugar, se entrenó el modelo YOLO con el dataset generado por los compañeros, no obstante, fue necesario eliminar imágenes mal etiquetadas y que se habían rotado, por lo que se le pidió a la IA un script para hacer esto. Una vez limpio el dataset, se tuvo que transformar a la estructura de entrenamiento que esperaban los modelos, por lo que, con el fin de hacerlo de forma aleatoria, se le solicitó a la IA otro script para que la estructura del dataset fuera la siguiente:
<div align="center">
<div style="text-align: left; display: inline-block;">

```text 
dataset
   ├───test
   │   ├───images
   │   └───labels
   ├───train
   │   ├───images
   │   └───labels
   └───val
       ├───images
       └───labels
```

</div>
</div>

Para el aprendizaje de YOLO, se optó por mantener el dataset sin modificar, ya que el entrenamiento es ligero. No obstante, a la hora de entrenar el modelo de RTDTR se presentó un gran retardo en el entrenamiento por el coste de lectura de las imágenes, por lo que, con otro script generado con Inteligencia Artificial se redujo el tamaño de las imágenes y se ubicó en una posición diferente, en la que el retardo era menor.

### Configuración del entrenamiento de los modelos

Una vez entrenado el modelo de YOLO, se hizo lo mismo con el modelo de RTDETR, que supuso un reto mayor porque la resolución con la que se entrenaba no permitía que el modelo lograra detectar apenas matrículas pequeñas, por lo que fue necesario modificarlo. Asimismo, por problemas con versiones de cuda y pytorch, fue necesario modificar el tipo de objetos equivalentes a nuevas versiones. 

---

Una vez estaban los modelos ya entrenados, con el fin de tener un código más sencillo y sin repeticiones, se optó por usar únicamente un modelo de YOLO para la detección de vehículos, ya que el comportamiento en las detecciones es diferente. Con esto, se obtuvo el siguiente código:

~~~python

~~~

Obteniendo los siguientes resultados: 

Aquí se añaden las demos de los modelos, aunque sean unos pocos segundos y se ponen los enlaces al drive con los resultados

## Comparativa de los modelos de OCR

Aquí explica Ubay lo de las gráficas y todo lo que le apetezca:

## Anonimización de las personas en el vídeo

## Conteo de personas en base al sentido que toman

## Aportación de cada miembro del grupo en las tareas realizadas en esta práctica

Sergio: tratado del dataset, detección de vehículos y entrenamiento de los modelos de detección de matrículas
Ubay: extracción de texto con OCR y comparativa de los distintos modelos

# Fuentes

## Conversaciones con la Inteligencia Artificial