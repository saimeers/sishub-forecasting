import os
import json
from predictor import prepare_series_df, predict_series
from data_utils import (
    get_and_clean_project_total,
    get_and_clean_project_line_data,
    get_and_clean_project_tech_data,
    get_and_clean_project_scope_data
)
import numpy as np

CACHE_DIR = os.getenv("PROJECT_CACHE_DIR", "cache")
TOTAL_CACHE = os.path.join(CACHE_DIR, "projects_total_forecast.json")
LINE_CACHE = os.path.join(CACHE_DIR, "projects_line_forecast.json")
TECH_CACHE = os.path.join(CACHE_DIR, "projects_tech_forecast.json")
SCOPE_CACHE = os.path.join(CACHE_DIR, "projects_scope_forecast.json")

SEMESTERS = 1  # predecimos solo 1 semestre
os.makedirs(CACHE_DIR, exist_ok=True)


def build_history_from_df(df, date_col, value_col):
    return [
        {"date": row[date_col].strftime("%Y-%m-%d"), "value": int(round(row[value_col]))}
        for _, row in df.iterrows()
    ]


def generate_total_forecast():
    df = get_and_clean_project_total()
    if df.empty:
        print("[generate_total_forecast] No hay datos")
        return
    df_prep = prepare_series_df(df, 'date', 'total')
    df_fore = predict_series(df_prep, semesters=SEMESTERS)

    history = build_history_from_df(df, 'date', 'total')

    forecasting = [
        {
            "date": row['ds'].strftime("%Y-%m-%d"),
            "value": int(row['yhat']),
            "lower": int(row['lower']),
            "upper": int(row['upper']),
            "confidence_pct": float(row['confidence_pct'])
        }
        for _, row in df_fore.iterrows()
    ]

    results = [{"name": "total_projects", "history": history,
                "forecasting": forecasting, "semesters": SEMESTERS}]
    with open(TOTAL_CACHE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[generate_total_forecast] Guardado en {TOTAL_CACHE}")


def generate_multicol_forecasts(df_wide, cache_path):
    if df_wide.empty:
        print(f"[generate_multicol_forecasts] DataFrame vacío → {cache_path}")
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return

    results = []
    for col in [c for c in df_wide.columns if c != 'date']:
        df_col = df_wide[['date', col]].copy().rename(columns={col: 'value'})
        df_col['value'] = df_col['value'].fillna(0)
        history = build_history_from_df(df_col, 'date', 'value')
        df_prep = prepare_series_df(df_col, 'date', 'value')
        df_fore = predict_series(df_prep, semesters=SEMESTERS)
        forecasting = [
            {
                "date": row['ds'].strftime("%Y-%m-%d"),
                "value": int(row['yhat']),
                "lower": int(row['lower']),
                "upper": int(row['upper']),
                "confidence_pct": float(row['confidence_pct'])
            }
            for _, row in df_fore.iterrows()
        ]
        results.append({"name": col, "history": history,
                        "forecasting": forecasting, "semesters": SEMESTERS})

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
    print("✅ Forecasts de proyectos generados y guardados")


if __name__ == "__main__":
    generate_all()
