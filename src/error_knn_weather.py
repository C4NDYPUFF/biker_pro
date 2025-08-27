import argparse 
import matplotlib.pyplot as plt
import pandas as pd
import pathlib as PATH 


from typing import Tuple, List 
import numpy as np 


from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


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
    df = pd.read_csv(csv_path, low_memory=False)

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
        n_bad = int(~mask_valid).sum()
        X = X[mask_valid].reset_index(drop=True)
        y = y[mask_valid].reset_index(drop=True)
        dt = dt[mask_valid].reset_index(drop=True)
        print(f"[Aviso] Se removieron {n_bad} filas con nulos en clima/target.")
    
    return X, y , dt 


