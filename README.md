#  biker_pro

Predicción de la **demanda de bicicletas** en función de variables climáticas utilizando **K-Nearest Neighbors (KNN)**.  
Este proyecto carga datos históricos de renta de bicicletas y clima, realiza un split temporal, entrena modelos KNN para distintos valores de `k` y evalúa el error (RMSE). Finalmente genera reportes y gráficas.

---

## Índice

- [ Estructura del proyecto](#-estructura-del-proyecto)
- [ Requerimientos](#-requerimientos)
- [ Uso](#️-uso)
- [ Parámetros](#-parámetros)
- [ Resultados](#-resultados)
- [ Desarrollo](#️-desarrollo)
- [ Notas](#-notas)

---

##  Estructura del proyecto

```plaintext
biker_pro/
├── notebooks/            # Experimentos y análisis exploratorio
├── results/              # Resultados generados (CSV, PNGs)
├── src/
│   └── error_knn_weather.py  # Script principal
├── requirements.txt      # Dependencias del proyecto
├── .gitignore
└── README.md
```




---

##  Requerimientos

El proyecto usa **Python 3.10+** y las siguientes librerías principales:

- pandas  
- numpy  
- scikit-learn  
- matplotlib  

Puedes instalar las dependencias con:

```bash
pip install -r requirements.txt
```

## Uso

El script principal se encuentra en src/error_knn_weather.py.
Puedes ejecutarlo desde línea de comandos:

```python
python src/error_knn_weather.py \
    --data "ruta/al/SeoulBikeData.csv" \
    --date_col "Date" \
    --hour_col "Hour" \
    --target "Rented Bike Count" \
    --k_values "3,5,10,15,20,50,100,300,500,1000" \
    --test_size 0.2 \
    --out_png "results/rmse_vs_k.png" \
    --out_csv "results/results.csv" \
    --show_hour_counts
```

Si da problemas la ejecución con modulos puedes intentar ejecutarlo desde la raiz del proyecto biker_pro/ con el siguiente comando:
```bash

python -m src.error_knn_weather \
    --data "ruta/al/SeoulBikeData.csv" \
    --date_col "Date" \
    --hour_col "Hour" \
    --target "Rented Bike Count" \
    --k_values "3,5,10,15,20,50,100,300,500,1000" \
    --test_size 0.2 \
    --out_png "results/rmse_vs_k.png" \
    --out_csv "results/results.csv" \
    --show_hour_counts
```

## Parámetros

| Parámetro            | Descripción                                             | Default                              |
| -------------------- | ------------------------------------------------------- | ------------------------------------ |
| `--data`             | Ruta al archivo CSV de entrada.                         | `SeoulBikeData.csv`                  |
| `--date_col`         | Nombre de la columna de fechas.                         | `Date`                               |
| `--hour_col`         | Nombre de la columna de hora (0–23).                    | `Hour`                               |
| `--target`           | Variable objetivo (ej. número de bicicletas rentadas).  | `Rented Bike Count`                  |
| `--k_values`         | Lista de valores de `k` a evaluar, separados por coma.  | `"3,5,10,15,20,50,100,300,500,1000"` |
| `--test_size`        | Proporción del dataset para test (0 < x < 1).           | `0.2`                                |
| `--out_png`          | Ruta de salida para la gráfica RMSE vs k.               | `results/rmse_vs_k.png`              |
| `--out_csv`          | Ruta de salida para guardar resultados en CSV.          | `results/results.csv`                |
| `--show_hour_counts` | Flag opcional para imprimir conteo de horas en consola. | `False`                              |


## Resultados


Al ejecutar el script se generan:

results/results.csv → tabla con los valores de k y su RMSE asociado.

results/rmse_vs_k.png → gráfica RMSE vs k, con el mejor valor marcado.

results/train_actual_vs_pred_kX.png → gráfica de entrenamiento (real vs predicho).

results/test_actual_vs_pred_kX.png → gráfica de prueba (real vs predicho).

Ejemplo de salida en consola:
```yaml
==============================================
Train: 2017-12-01 → 2018-10-31  (n=8,431)
Test : 2018-11-01 → 2018-12-31  (n=2,107)
----------------------------------------------
Mejor k: 15  |  RMSE = 133.2512
Gráfica: results/rmse_vs_k.png
Resultados: results/results.csv
==============================================
```

notebooks/ → espacio para exploración y pruebas rápidas en Jupyter.

src/error_knn_weather.py → código limpio, modular y reutilizable.

results/ → directorio autogenerado para almacenar gráficos y métricas.


---

## Notas

El dataset usado en este proyecto es el Seoul Bike Sharing Demand Data (UCI Repository).

El split de datos es temporal, no aleatorio. Esto asegura que los datos de test sean posteriores en el tiempo.

El modelo KNN usa StandardScaler para normalizar variables antes del entrenamiento.