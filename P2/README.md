# Autores
[Ubay Antonio Batista Santana](https://github.com/UbayBatista)
[Ángel Sergio Reyes Guedes](https://github.com/asergiorg)

# Trabajo desempeñado
## Tarea 1

Para la realización de esta tarea, hemos optado por un código sencillo, que parte del aportado como ejemplo para el conteo de las columnas. Para adaptar dicho código, se ha convertido el array resultante tras la suma en un vector de una sola dimensión, ya que al sumar por filas, el resultado es algo similar a lo siguiente: [[256],[243],...,[123]]. Para ello, se ha añadido la siguiente línea de código:

```
rows = rows.flatten()
```

Para encontrar el máximo, se ha utilizado la función 'max' que ofrecen los propios array, ya que rows tiene dimensión 1:

```
maxfil = rows.max()
```

Por último, como el array 'rows', ya contiene las filas ordenadas, sólo queda comparar el valor que ocupa cada posición con el 90% de maxfil y, si es mayor, graficarlo con la función 'axhline' de la librería matplotlib:

```
for i in range(len(rows)):
    if rows[i] >= maxfil*0.9:
        plt.axhline(y=i, color='r', linestyle='-')
```

Obteniendo el siguiente resultado:

<p align="center">
<img src="images/exercise1.png" width="500">
</p>

## Tarea 2



## Tarea 3



## Tarea Adicional



## Inconvenientes encontrados



## Aportación de cada miembro del grupo en las tareas realizadas en esta práctica


