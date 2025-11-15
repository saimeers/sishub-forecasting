# scripts/generate_project_forecasts.py
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from predictor import prepare_series_df, predict_series
from data_utils import (
    get_and_clean_project_total,
    get_and_clean_project_line_data,
    get_and_clean_project_tech_data,
    get_and_clean_project_scope_data
)

CACHE_DIR = "cache"
TOTAL_CACHE = os.path.join(CACHE_DIR, "projects_total_forecast.json")
LINE_CACHE  = os.path.join(CACHE_DIR, "projects_line_forecast.json")
TECH_CACHE  = os.path.join(CACHE_DIR, "projects_tech_forecast.json")
SCOPE_CACHE = os.path.join(CACHE_DIR, "projects_scope_forecast.json")

WEEKS = int(os.getenv("PROJECT_FORECAST_WEEKS", 16))  # default ~ semester

os.makedirs(CACHE_DIR, exist_ok=True)

def build_history_from_df(df, date_col, value_col):
    return [
        {"date": row[date_col].strftime("%Y-%m-%d"), "value": int(round(row[value_col]))}
        for _, row in df.iterrows()
    ]

def generate_total_forecast():
    df = get_and_clean_project_total()
    results = []

    if df.empty:
        print("[generate_total_forecast] No hay datos de proyectos totales")
    else:
        # prepare for prophet
        df_prep = df.rename(columns={'date':'ds', 'total':'y'}).copy()
        df_prep['y'] = df_prep['y'].clip(lower=0)
        df_prep['y'] = np.log1p(df_prep['y'])

        df_fore = predict_series(df_prep, weeks=WEEKS)

        history = build_history_from_df(df, 'date', 'total')
        forecasting = [
            {"date": row['ds'].strftime("%Y-%m-%d"), "value": int(round(row['yhat']))}
            for _, row in df_fore.iterrows()
        ]

        results.append({
            "name": "total_projects",
            "history": history,
            "forecasting": forecasting,
            "weeks": WEEKS
        })

    with open(TOTAL_CACHE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[generate_total_forecast] Guardado en {TOTAL_CACHE}")

def generate_multicol_forecasts(df_wide: pd.DataFrame, cache_path: str):
    """
    df_wide: columnas: date + col1, col2, ...
    para cada columna (excepto date) generar forecast
    """
    if df_wide.empty:
        print(f"[generate_multicol_forecasts] DataFrame vacío → {cache_path}")
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return

    cols = [c for c in df_wide.columns if c != 'date']
    results = []

    for col in cols:
        df_col = df_wide[['date', col]].copy().rename(columns={col: 'value'})
        # fill NaNs with 0
        df_col['value'] = df_col['value'].fillna(0)
        # build history
        history = [
            {"date": row['date'].strftime("%Y-%m-%d"), "value": int(round(row['value']))}
            for _, row in df_col.iterrows()
        ]
        # prepare for prophet
        df_prep = prepare_series_df(df_col, 'date', 'value')
        df_fore = predict_series(df_prep, weeks=WEEKS)
        forecasting = [
            {"date": row['ds'].strftime("%Y-%m-%d"), "value": int(round(row['yhat']))}
            for _, row in df_fore.iterrows()
        ]
        results.append({
            "name": col,
            "history": history,
            "forecasting": forecasting,
            "weeks": WEEKS
        })

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[generate_multicol_forecasts] Guardado en {cache_path}")

def generate_line_forecasts():
    df = get_and_clean_project_line_data()
    generate_multicol_forecasts(df, LINE_CACHE)

def generate_tech_forecasts():
    df = get_and_clean_project_tech_data()
    generate_multicol_forecasts(df, TECH_CACHE)

def generate_scope_forecasts():
    df = get_and_clean_project_scope_data()
    generate_multicol_forecasts(df, SCOPE_CACHE)

def generate_all():
    generate_total_forecast()
    generate_line_forecasts()
    generate_tech_forecasts()
    generate_scope_forecasts()
    print("✅ Project forecasts generados y guardados")

if __name__ == "__main__":
    generate_all()