import os
import requests
import pandas as pd
from dotenv import load_dotenv

if os.getenv("ENV") != "production":
    load_dotenv()

API_URL = os.getenv("API_URL", "").rstrip('/')
TIMEOUT = int(os.getenv("DATA_UTILS_TIMEOUT", 10))

def _get(endpoint: str):
    url = f"{API_URL}{endpoint}"
    resp = requests.get(url, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()

def semester_to_date(semester_str: str) -> pd.Timestamp:
    try:
        year, sem = semester_str.split("-")
        year = int(year)
        sem = int(sem)

        if sem == 1:
            return pd.Timestamp(year=year, month=2, day=1)
        elif sem == 2:
            return pd.Timestamp(year=year, month=8, day=1)
        else:
            raise ValueError(f"Semestre inválido: {semester_str}")
    except Exception:
        raise ValueError(f"Formato inválido de semestre: {semester_str}")

def get_and_clean_project_total():
    try:
        raw = _get("/semester-projects")
        df = pd.DataFrame(raw)
        if df.empty:
            return pd.DataFrame(columns=['date', 'total'])
        # convertir 'semester' -> date
        df['date'] = df['semester'].apply(semester_to_date)
        df = df[['date', 'total']].sort_values('date').reset_index(drop=True)
        return df
    except Exception as e:
        print(f"[get_and_clean_project_total] Error: {e}")
        return pd.DataFrame(columns=['date', 'total'])

def get_and_clean_project_line_data():
    try:
        raw = _get("/semester-by-line")
        df = pd.DataFrame(raw)
        if df.empty:
            return pd.DataFrame()
        df['date'] = df['semester'].apply(semester_to_date)
        df = df.drop(columns=['semester'])
        df = df.sort_values('date').reset_index(drop=True)
        return df
    except Exception as e:
        print(f"[get_and_clean_project_line_data] Error: {e}")
        return pd.DataFrame()

def get_and_clean_project_scope_data():
    try:
        raw = _get("/semester-by-scope")
        df = pd.DataFrame(raw)
        if df.empty:
            return pd.DataFrame()
        df['date'] = df['semester'].apply(semester_to_date)
        df = df.drop(columns=['semester'])
        df = df.sort_values('date').reset_index(drop=True)
        return df
    except Exception as e:
        print(f"[get_and_clean_project_scope_data] Error: {e}")
        return pd.DataFrame()

def get_and_clean_project_tech_data():
    try:
        raw = _get("/semester-by-tech")
        df = pd.DataFrame(raw)
        if df.empty:
            return pd.DataFrame()
        df['date'] = df['semester'].apply(semester_to_date)
        df = df.drop(columns=['semester'])
        df = df.sort_values('date').reset_index(drop=True)
        return df
    except Exception as e:
        print(f"[get_and_clean_project_tech_data] Error: {e}")
        return pd.DataFrame()
