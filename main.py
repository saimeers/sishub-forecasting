import os
import json
from fastapi import FastAPI, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from pydantic import BaseModel
from typing import List

from scripts.generate_project_forecasts import (
    generate_all,
    TOTAL_CACHE,
    LINE_CACHE,
    TECH_CACHE,
    SCOPE_CACHE
)

API_KEY = os.getenv("API_KEY", "")
if not API_KEY:
    raise RuntimeError("La variable de entorno API_KEY no está definida")

app = FastAPI(title="Project Forecasts API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(status_code=401, detail="Invalid or missing API Key")

scheduler = AsyncIOScheduler(timezone="America/Bogota")

@app.on_event("startup")
async def startup_event():
    try:
        if not scheduler.running:
            # Ejecutar el 20 de junio y 20 de diciembre a las 00:00 (America/Bogota)
            trigger = CronTrigger(month='6,12', day='20', hour=0, minute=0, timezone="America/Bogota")
            scheduler.add_job(generate_all, trigger, id="projects_job", replace_existing=True)
            scheduler.start()
            # Generar cache inicial si no existe
            try:
                generate_all()
            except Exception as e:
                print(f"[startup_event] Error generando cache inicial: {e}")
    except Exception as e:
        print(f"[startup_event] Error al iniciar scheduler: {e}")

def ensure_cache(path: str, generate_fn):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if (not os.path.isfile(path)) or (os.path.getsize(path) == 0):
            print(f"[ensure_cache] '{path}' no existe o está vacío → generando ahora.")
            generate_fn()
            return

        try:
            with open(path, 'r', encoding='utf-8') as f:
                _ = json.load(f)
        except Exception as e:
            print(f"[ensure_cache] '{path}' contiene JSON inválido ({e}) → regenerando.")
            generate_fn()
            return
    except Exception as e:
        print(f"[ensure_cache] Error general: {e}")

def load_cache(path: str, generate_fn) -> list:
    try:
        ensure_cache(path, generate_fn)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"[load_cache] ERROR leyendo '{path}' incluso después de generar: {e}")
            return []
        if isinstance(data, dict) and 'forecasts' in data:
            return data['forecasts']
        if isinstance(data, list):
            return data
        print(f"[load_cache] '{path}' no contiene ni lista ni 'forecasts'.")
        return []
    except Exception as e:
        print(f"[load_cache] Error inesperado: {e}")
        return []

class Point(BaseModel):
    date: str
    value: int

class ForecastItem(BaseModel):
    name: str
    history: List[Point]
    forecasting: List[Point]
    semesters: int

@app.get("/", tags=["Root"])
def read_root():
    return {
        "message": "🟢 Project Forecasts API (semesters)",
        "endpoints": [
            "/cached/forecast/project-total",
            "/cached/forecast/project-line",
            "/cached/forecast/project-tech",
            "/cached/forecast/project-scope",
            "/forecast/project-total",
            "/forecast/project-line",
            "/forecast/project-tech",
            "/forecast/project-scope"
        ]
    }

@app.get("/cached/forecast/project-total", dependencies=[Depends(get_api_key)])
def get_cached_total():
    try:
        cache = load_cache(TOTAL_CACHE, generate_all)
        return {"forecasts": cache}
    except Exception as e:
        print(f"[get_cached_total] Error: {e}")
        raise HTTPException(status_code=500, detail="Error interno al obtener cache total")

@app.get("/cached/forecast/project-line", dependencies=[Depends(get_api_key)])
def get_cached_line():
    try:
        cache = load_cache(LINE_CACHE, generate_all)
        return {"forecasts": cache}
    except Exception as e:
        print(f"[get_cached_line] Error: {e}")
        raise HTTPException(status_code=500, detail="Error interno al obtener cache line")

@app.get("/cached/forecast/project-tech", dependencies=[Depends(get_api_key)])
def get_cached_tech():
    try:
        cache = load_cache(TECH_CACHE, generate_all)
        return {"forecasts": cache}
    except Exception as e:
        print(f"[get_cached_tech] Error: {e}")
        raise HTTPException(status_code=500, detail="Error interno al obtener cache tech")

@app.get("/cached/forecast/project-scope", dependencies=[Depends(get_api_key)])
def get_cached_scope():
    try:
        cache = load_cache(SCOPE_CACHE, generate_all)
        return {"forecasts": cache}
    except Exception as e:
        print(f"[get_cached_scope] Error: {e}")
        raise HTTPException(status_code=500, detail="Error interno al obtener cache scope")

# POST endpoints to request one item from cache by name + semesters
class ForecastRequest(BaseModel):
    name: str
    semesters: int = 2

@app.post("/forecast/project-total", dependencies=[Depends(get_api_key)])
def post_forecast_total(req: ForecastRequest):
    try:
        cache = load_cache(TOTAL_CACHE, generate_all)
        for item in cache:
            if item.get('name') == req.name and item.get('semesters') == req.semesters:
                return item
        raise HTTPException(status_code=404, detail=f"Item '{req.name}' con semesters={req.semesters} no encontrado en cache")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[post_forecast_total] Error: {e}")
        raise HTTPException(status_code=500, detail="Error interno")

@app.post("/forecast/project-line", dependencies=[Depends(get_api_key)])
def post_forecast_line(req: ForecastRequest):
    try:
        cache = load_cache(LINE_CACHE, generate_all)
        for item in cache:
            if item.get('name') == req.name and item.get('semesters') == req.semesters:
                return item
        raise HTTPException(status_code=404, detail=f"Line '{req.name}' con semesters={req.semesters} no encontrada en cache")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[post_forecast_line] Error: {e}")
        raise HTTPException(status_code=500, detail="Error interno")

@app.post("/forecast/project-tech", dependencies=[Depends(get_api_key)])
def post_forecast_tech(req: ForecastRequest):
    try:
        cache = load_cache(TECH_CACHE, generate_all)
        for item in cache:
            if item.get('name') == req.name and item.get('semesters') == req.semesters:
                return item
        raise HTTPException(status_code=404, detail=f"Tech '{req.name}' con semesters={req.semesters} no encontrada en cache")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[post_forecast_tech] Error: {e}")
        raise HTTPException(status_code=500, detail="Error interno")

@app.post("/forecast/project-scope", dependencies=[Depends(get_api_key)])
def post_forecast_scope(req: ForecastRequest):
    try:
        cache = load_cache(SCOPE_CACHE, generate_all)
        for item in cache:
            if item.get('name') == req.name and item.get('semesters') == req.semesters:
                return item
        raise HTTPException(status_code=404, detail=f"Scope '{req.name}' con semesters={req.semesters} no encontrada en cache")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[post_forecast_scope] Error: {e}")
        raise HTTPException(status_code=500, detail="Error interno")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=os.getenv("ENV") != "production")