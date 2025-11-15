# data_utils.py
import os
import requests
import pandas as pd
from dotenv import load_dotenv

if os.getenv("ENV") != "production":
    load_dotenv()

API_URL = os.getenv("API_URL", "").rstrip('/')
TIMEOUT = 10

def _get(endpoint: str):
    url = f"{API_URL}{endpoint}"
    resp = requests.get(url, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()

def get_and_clean_project_total():
    """
    Espera endpoint /project/weekly-projects que retorna:
      [ { "week": "YYYY-MM-DD", "total": N }, ... ]
    Devuelve DataFrame con columnas ['date','total'] y ordenado.
    """
    try:
        raw = _get("/weekly-projects")
        df = pd.DataFrame(raw)
        if df.empty:
            return pd.DataFrame(columns=['date', 'total'])
        df = df.rename(columns={'week': 'date', 'total': 'total'})
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
        return df[['date', 'total']]
    except Exception as e:
        print(f"[get_and_clean_project_total] Error: {e}")
        return pd.DataFrame(columns=['date', 'total'])

def get_and_clean_project_line_data():
    """
    Espera endpoint /project/weekly-by-line que retorna filas:
      [ { "week": "YYYY-MM-DD", "Línea A": 2, "Línea B": 1, ... }, ... ]
    Devuelve DataFrame con date + columnas por línea.
    """
    try:
        raw = _get("/weekly-by-line")
        df = pd.DataFrame(raw)
        if df.empty:
            return pd.DataFrame()
        df = df.rename(columns={'week': 'date'})
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
        return df
    except Exception as e:
        print(f"[get_and_clean_project_line_data] Error: {e}")
        return pd.DataFrame()

def get_and_clean_project_scope_data():
    """
    Espera endpoint /project/weekly-by-scope similar a lines.
    """
    try:
        raw = _get("/weekly-by-scope")
        df = pd.DataFrame(raw)
        if df.empty:
            return pd.DataFrame()
        df = df.rename(columns={'week': 'date'})
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
        return df
    except Exception as e:
        print(f"[get_and_clean_project_scope_data] Error: {e}")
        return pd.DataFrame()

def get_and_clean_project_tech_data():
    """
    Espera endpoint /project/weekly-by-tech que retorna filas:
      [ { "week": "YYYY-MM-DD", "Python": 2, "React": 1, ... }, ... ]
    """
    try:
        raw = _get("/weekly-by-tech")
        df = pd.DataFrame(raw)
        if df.empty:
            return pd.DataFrame()
        df = df.rename(columns={'week': 'date'})
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
        return df
    except Exception as e:
        print(f"[get_and_clean_project_tech_data] Error: {e}")
        return pd.DataFrame()
