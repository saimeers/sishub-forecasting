# predictor.py
from prophet import Prophet
import pandas as pd
import numpy as np

def prepare_series_df(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    """
    Prepara df para Prophet:
      - date_col -> ds, value_col -> y
      - clip lower 0, log1p
    """
    df_s = df[[date_col, value_col]].copy()
    df_s = df_s.rename(columns={date_col: "ds", value_col: "y"})
    df_s['ds'] = pd.to_datetime(df_s['ds'])
    df_s = df_s.sort_values('ds').reset_index(drop=True)
    df_s['y'] = df_s['y'].clip(lower=0)
    df_s['y'] = np.log1p(df_s['y'])
    return df_s

def es_forecast_inestable(history: list[int], forecast: list[int], umbral: float = 2.5) -> bool:
    if len(history) < 3 or len(forecast) == 0:
        return False
    avg_hist = np.mean(history[-3:])
    avg_fore = np.mean(forecast)
    return (avg_hist > 0) and (avg_fore > avg_hist * umbral)

def add_noise_int(array: np.ndarray, pct: float = 0.1) -> np.ndarray:
    factors = np.random.uniform(1 - pct, 1 + pct, size=array.shape)
    noisy = (array * factors).round().astype(int)
    return np.clip(noisy, 0, None)

def predict_series(df_prepared: pd.DataFrame, weeks: int = 16, allow_fallback: bool = True, umbral_inestabilidad: float = 2.5) -> pd.DataFrame:
    """
    Recibe df preparado para Prophet (ds, y log1p) y devuelve df con ['ds','yhat'] en escala original (int).
    Si hay pocos valores no nulos, hace fallback a promedio simple con ruido.
    """
    df = df_prepared.copy()
    non_zero_count = (df['y'] > 0).sum()

    if non_zero_count < 6 and allow_fallback:
        # fallback: promedio en escala original
        avg_val = np.expm1(df['y']).mean() if len(df) > 0 else 0
        base_val = int(round(avg_val))
        fechas = pd.date_range(start=df['ds'].max() + pd.Timedelta(weeks=1) if len(df)>0 else pd.Timestamp.today(), periods=weeks, freq='W-MON')
        yhat = add_noise_int(np.array([base_val] * weeks))
        return pd.DataFrame({'ds': fechas, 'yhat': yhat})

    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.5,
        seasonality_mode='additive'
    )
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

    model.fit(df)

    future = model.make_future_dataframe(periods=weeks, freq='W-MON')
    forecast = model.predict(future)

    forecast['yhat'] = np.expm1(forecast['yhat'].clip(lower=0))
    forecast['yhat'] = forecast['yhat'].round().astype(int)
    df_fore = forecast[['ds', 'yhat']].tail(weeks).reset_index(drop=True)

    # stability check
    hist_vals = np.expm1(df['y']).round().astype(int).tolist() if len(df)>0 else []
    fore_vals = df_fore['yhat'].tolist()

    if allow_fallback and es_forecast_inestable(hist_vals, fore_vals, umbral=umbral_inestabilidad):
        avg_last3 = int(round(np.mean(hist_vals[-3:]))) if len(hist_vals) >= 1 else 0
        fechas = pd.date_range(start=df['ds'].max() + pd.Timedelta(weeks=1) if len(df)>0 else pd.Timestamp.today(), periods=weeks, freq='W-MON')
        return pd.DataFrame({'ds': fechas, 'yhat': add_noise_int(np.array([avg_last3] * weeks))})

    # apply soft noise
    yhat_noisy = add_noise_int(df_fore['yhat'].values)
    return pd.DataFrame({'ds': df_fore['ds'], 'yhat': yhat_noisy})