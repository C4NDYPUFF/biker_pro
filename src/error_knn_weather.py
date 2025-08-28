import argparse 
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


from typing import Tuple, List 
import numpy as np 


from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error


def sanity_check_hours(
        df: pd.DataFrame, 
        hour_col: str, 
        show_counts: bool = False
) -> None:
    """
    Verifica que la columna de hora sea numérica, esté en 0..23 y sin nulos.
    Opcionalmente imprime value_counts() por diagnóstico.
    """
    df[hour_col] = pd.to_numeric(df[hour_col], errors="coerce")
    if df[hour_col].isna().any():
        n_bad = int(df[hour_col].isna().sum())
        raise ValueError(
            f"{n_bad} filas con '{hour_col}' inválido tras conversión numérica."
        )

    bad_range = ~df[hour_col].between(0, 23)
    if bad_range.any():
        bad_vals = df.loc[bad_range, hour_col].unique()
        raise ValueError(
            f"Valores fuera de 0..23 en '{hour_col}': {bad_vals}"
        )

    if show_counts:
        vc = df[hour_col].value_counts().sort_index()
        print("[Diagnóstico] value_counts(Hour):")
        print(vc.to_string())


def load_data (
        csv_path: Path,
        date_col: str,
        hour_col: str, 
        target_col: str, 
        show_hour_counts: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Carga el CSV, convierte fecha y crea un timestamp 'dt' = Date + Hour,
    ordena temporalmente y devuelve X (clima), y (target) y dt (marca de tiempo).

    Se usan SOLO variables de clima como features:
      Temperature(°C), Humidity(%), Wind speed (m/s), Visibility (10m),
      Dew point temperature(°C), Solar Radiation (MJ/m2), Rainfall(mm), Snowfall (cm)
    """
    # Carga del csv 
    df = pd.read_csv(csv_path, low_memory=False, encoding="latin1")

    # Asergurar tipos de variable y ordendar 
    df[date_col] = pd.to_datetime(df[date_col], format='%d/%m/%Y')
    if df[date_col].isna().any():
        raise ValueError("Fechas inválidas encontradas en {}".format(csv_path))
    
    # Verificación y nomalización de horas 
    sanity_check_hours(df, hour_col=hour_col, show_counts=show_hour_counts)

    # Constucción del Datetime completo y orden temporal 
    df['dt'] = df[date_col] + pd.to_timedelta(df[hour_col], unit= 'h')
    df = df.sort_values('dt').reset_index(drop=True)

    # Selección de Variables del Clima 
    weather_cols_exact = [
        "Temperature(°C)",
        "Humidity(%)",
        "Wind speed (m/s)",
        "Visibility (10m)",
        "Dew point temperature(°C)",
        "Solar Radiation (MJ/m2)",
        "Rainfall(mm)",
        "Snowfall (cm)",
    ]

    missing = [c for c in weather_cols_exact if c not in df.columns]
    if missing:
        raise ValueError(f'Faltan columnas de clima en el DataFrame: {missing}'
                         f'Columnas disponibles: {df.columns.tolist()}')
    
    #7 Subconjuntos 
    X = df[weather_cols_exact].copy()
    y = df[target_col].copy()
    dt = df['dt'].copy()

    #8 Chequeo de variables nulas por robustex 
    mask_valid = X.notna().all(axis=1) & y.notna()
    if not mask_valid.all():
        n_bad = int((~mask_valid).sum())
        X = X[mask_valid].reset_index(drop=True)
        y = y[mask_valid].reset_index(drop=True)
        dt = dt[mask_valid].reset_index(drop=True)
        print(f"[Aviso] Se removieron {n_bad} filas con nulos en clima/target.")
    
    return X, y , dt 

def temporal_train_test_split(
        X: pd.DataFrame,
        y: pd.Series,
        dt: pd.Series, 
        test_size = 0.2, 
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """ 
    Split temporal: la primeras observaciones son train las ultimas con test
    Mantiene el orden temporarl (X, y, dt)"""

    n = len(X)
    n_test = int(np.floor(n * test_size))
    n_train = n - n_test
    if n_train <= 0 or n_test <= 0:
        raise ValueError("Conjunto de datos muy pequeño para el tamaño de test dado.")
    
    X_train = X.iloc[:n_train, :].copy()
    y_train = y.iloc[:n_train].copy()
    dt_train = dt.iloc[:n_train].copy()

    X_test = X.iloc[n_train:, :].copy()
    y_test = y.iloc[n_train:].copy()
    dt_test = dt.iloc[n_train:].copy()

    return X_train, X_test, y_train, y_test, dt_train, dt_test

def plot_series(y_true, y_pred, title: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(pd.Series(y_true).reset_index(drop=True), label='Actual')
    plt.plot(pd.Series(y_pred).reset_index(drop=True), label='Predicho')
    plt.title(title)
    plt.xlabel('Indice (tiempo)')
    plt.ylabel('Bicicletas Rentadas')
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def fit_knn_with_k(X_train, y_train, k: int) -> Pipeline:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor(n_neighbors=k, weights="distance")),
    ])
    pipe.fit(X_train, y_train)
    return pipe

def evaluate_knn_over_k(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    k_values: List[int],
) -> pd.DataFrame:
    """
    Entrena KNN para cada k en k_values con pipeline (StandardScaler -> KNN),
    evalúa RMSE en test y devuelve DataFrame con columnas ['k', 'rmse'].
    """
    results = []
    for k in k_values:
        pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("knn", KNeighborsRegressor(n_neighbors=k, weights="distance")),
            ]
        )
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        results.append({"k": int(k), "rmse": float(rmse)})
    return pd.DataFrame(results).sort_values("k").reset_index(drop=True)

def plot_rmse_vs_k(results_df: pd.DataFrame, out_path: Path) -> None:
    """
    Grafica la figura RMSE vs k y marcca el minimo
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(results_df['k'], results_df['rmse'], marker='o')
    plt.xlabel('k (numero de vecinos)')
    plt.ylabel('RMSE')
    plt.title('Error (RMSE) vs k - KNN con variables de clima')
    plt.grid(True, linestyle=':')

    best_idx = results_df['rmse'].idxmin()
    best_k = results_df.loc[best_idx, 'k']
    best_rmse = results_df.loc[best_idx, 'rmse']

    plt.scatter([best_k], [best_rmse], s=80, zorder=5)
    plt.annotate(
        f'min@k={best_k}\nRMSE={best_rmse:.3f}',
        xy = (best_k, best_rmse),
        xytext = (10, 10),
        textcoords = 'offset points',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evalúa KNN para predecir demanda de bicicletas usando variables de clima."
    )
    parser.add_argument("--data", type=Path, default=Path('/Users/osmar/Documents/Projects /MachineLearning/SeoulBikeData.csv'),
                        help="Ruta al archivo CSV con los datos.")
    parser.add_argument("--date_col", type=str, default='Date', 
                        help='Nombre de la columna de fecha (Default: Date)')
    parser.add_argument("--hour_col", type=str, default='Hour',
                        help='Nombre de la columna de hora (Default: Hour)')
    parser.add_argument("--target", type=str, default='Rented Bike Count',
                        help='Nombre de la columna objetivo (Default: Rented Bike Count)')
    parser.add_argument("--k_values", type=str, default= '3, 5, 10, 15, 20, 50, 100, 300, 500, 1000', 
                        help='Valores de k a evaluar (Default: 3, 5, 10, 15, 20, 50, 100, 300, 500, 1000)')
    parser.add_argument("--test_size", type=float, default=0.2, 
                        help='Proporción del conjunto de datos a usar como test (Default: 0.2)')
    parser.add_argument("--out_png", type=Path, default=Path('results/rmse_vs_k.png'),
                        help='Ruta de salida para la figura RMSE vs k (Default: results/rmse_vs_k.png)'
                        )
    parser.add_argument("--out_csv", type=Path, default=Path('results/results.csv'),
                        help='Ruta de salida para el archivo CSV con los resultados (Default: results/results.csv)')
    parser.add_argument("--show_hour_counts", action='store_true',
                        help='Mostrar conteo de horas en la salida (Default: False)')
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    k_values = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]

    # Paso 1: Cargar y preparar (orden temporal, features de clima, target)
    X, y, dt = load_data(
        csv_path=args.data,
        date_col=args.date_col,
        hour_col=args.hour_col,
        target_col=args.target,
        show_hour_counts=args.show_hour_counts,
    )

    # Paso 2: Split temporal
    X_train, X_test, y_train, y_test, dt_train, dt_test = temporal_train_test_split(
        X=X, y=y, dt=dt, test_size=args.test_size
    )

    # Paso 3: Entrenar y evaluar KNN en varios k
    results_df = evaluate_knn_over_k(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        k_values=k_values,
    )

    # Paso 4: Exportar CSV y PNG
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.out_csv, index=False)
    plot_rmse_vs_k(results_df, args.out_png)

    # Paso 5: Reporte por consola (incluye rangos temporales)
    best_idx = results_df["rmse"].idxmin()
    best_k = results_df.loc[best_idx, "k"]
    best_rmse = results_df.loc[best_idx, "rmse"]
    print("==============================================")
    print(f"Train: {dt_train.min().date()} → {dt_train.max().date()}  (n={len(X_train):,})")
    print(f"Test : {dt_test.min().date()} → {dt_test.max().date()}  (n={len(X_test):,})")
    print("----------------------------------------------")
    print(f"Mejor k: {best_k}  |  RMSE = {best_rmse:.4f}")
    print(f"Gráfica: {args.out_png}")
    print(f"Resultados: {args.out_csv}")
    print("==============================================")

    # Re-entrenar con el mejor k y graficar Actual vs Predicho
    best_idx = results_df["rmse"].idxmin()
    best_k = int(results_df.loc[best_idx, "k"])

    best_pipe = fit_knn_with_k(X_train, y_train, best_k)

    y_train_pred = best_pipe.predict(X_train)
    y_test_pred  = best_pipe.predict(X_test)

    plot_series(
        y_train, y_train_pred,
        title=f"Entrenamiento - Actual vs Predicho (k={best_k})",
        out_path=Path("results/train_actual_vs_pred_k{}.png".format(best_k))
    )

    plot_series(
        y_test, y_test_pred,
        title=f"Prueba - Actual vs Predicho (k={best_k})",
        out_path=Path("results/test_actual_vs_pred_k{}.png".format(best_k))
    )


if __name__ == "__main__":
    main()