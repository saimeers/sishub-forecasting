from prophet import Prophet
import pandas as pd
import numpy as np

def prepare_series_df(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    df_s = df[[date_col, value_col]].copy()
    df_s = df_s.rename(columns={date_col: "ds", value_col: "y"})
    df_s['ds'] = pd.to_datetime(df_s['ds'])
    df_s = df_s.sort_values('ds').reset_index(drop=True)
    df_s['y'] = df_s['y'].clip(lower=0)
    df_s['y'] = np.log1p(df_s['y'])
    return df_s

def add_noise_int(array: np.ndarray, pct: float = 0.05) -> np.ndarray:
    if array.size == 0:
        return array.astype(int)
    factors = np.random.uniform(1 - pct, 1 + pct, size=array.shape)
    noisy = (array * factors).round().astype(int)
    return np.clip(noisy, 0, None)

def _next_semester_date_from(dt: pd.Timestamp) -> pd.Timestamp:
    if dt.month <= 2:
        return pd.Timestamp(year=dt.year, month=8, day=1)
    return pd.Timestamp(year=dt.year + 1, month=2, day=1)

def _make_future_semester_range(start: pd.Timestamp, periods: int) -> pd.DatetimeIndex:
    dates = []
    cur = _next_semester_date_from(start)
    for _ in range(periods):
        dates.append(cur)
        cur = _next_semester_date_from(cur)
    return pd.DatetimeIndex(dates)

def predict_series(df_prepared: pd.DataFrame, semesters: int = 1, min_factor: float = 0.85) -> pd.DataFrame:
    df = df_prepared.copy()

    # Caso pocas observaciones
    if len(df) < 2:
        avg_val = int(round(np.expm1(df['y']).mean())) if len(df) > 0 else 0
        start = df['ds'].max() if len(df) > 0 else pd.Timestamp.today()
        fechas = _make_future_semester_range(start, semesters)
        return pd.DataFrame({
            'ds': fechas,
            'yhat': add_noise_int(np.array([avg_val]*semesters)),
            'lower': [max(1, int(avg_val*0.9))]*semesters,
            'upper': [int(avg_val*1.1)]*semesters,
            'confidence_pct': [95.0]*semesters
        })

    # Modelo prophet
    model = Prophet(
        weekly_seasonality=False,
        yearly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=5.0,
        seasonality_mode='additive'
    )
    model.fit(df)

    last_ds = df['ds'].max()
    future_dates = _make_future_semester_range(last_ds, semesters)
    future = pd.DataFrame({'ds': pd.to_datetime(list(df['ds']) + list(future_dates))})
    forecast = model.predict(future)
    forecast_future = forecast[forecast['ds'].isin(future_dates)].copy()

    # inverse log
    forecast_future['yhat'] = np.expm1(forecast_future['yhat'].clip(lower=0))
    forecast_future['lower_raw'] = np.expm1(forecast_future['yhat_lower'].clip(lower=0))
    forecast_future['upper_raw'] = np.expm1(forecast_future['yhat_upper'].clip(lower=0))

    last_val = int(np.expm1(df['y']).iloc[-1].round())
    min_val = max(int(last_val * min_factor), 1)

    # límites revisados: no exagerar min_val
    forecast_future['lower'] = forecast_future['lower_raw'].apply(
        lambda x: max(int(round(x)), int(min_val * 0.85))
    )
    forecast_future['upper'] = forecast_future['upper_raw'].apply(
        lambda x: max(int(round(x)), forecast_future['lower'].min() + 5)
    )

    # Ruido
    forecast_future['yhat'] = add_noise_int(forecast_future['yhat'].values)

    # Corrección: mantener yhat dentro del intervalo pero evitando igualdad exacta con lower
    def adjust_yhat(row):
        y = row['yhat']
        low = row['lower']
        up = row['upper']
        if y <= low:
            return low + 1
        if y >= up:
            return up - 1
        return y

    forecast_future['yhat'] = forecast_future.apply(adjust_yhat, axis=1)

    # Nuevo cálculo de confianza
    def compute_confidence(row):
        low = row['lower']
        up = row['upper']
        y = row['yhat']

        # si el intervalo está degenerado
        if up <= low:
            return 100.0

        # dentro del intervalo = alta confianza
        if low < y < up:
            width = up - low
            center = (up + low) / 2
            dist = abs(y - center)
            return round(100 - (dist / (width/2)) * 20, 2)  

        # fuera del intervalo
        if y <= low:
            return max(0, 100 - ((low - y) * 2))

        if y >= up:
            return max(0, 100 - ((y - up) * 2))

    forecast_future['confidence_pct'] = forecast_future.apply(compute_confidence, axis=1)

    return forecast_future[['ds','yhat','lower','upper','confidence_pct']]