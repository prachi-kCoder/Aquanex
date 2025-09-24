
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import os
from dotenv import load_dotenv
# from providers import fetch_open_meteo, fetch_fisheries, fetch_noaa , fetch_obis , fetch_worms , fetch_bold, fetch_csv, fetch_ftp
from providers.fetch_open_meteo import fetch_open_meteo
from providers.fetch_csv import fetch_csv
from providers.fetch_fisheries import fetch_fisheries
from providers.fetch_noaa import fetch_noaa
from providers.fetch_obis import fetch_obis
from providers.fetch_worms import fetch_worms
from providers.fetch_bold import fetch_bold
from providers.fetch_ftp import fetch_ftp
# from providers.fetch_cmfri import display_report
from fastapi import APIRouter, Body 

import datetime
import requests 
load_dotenv()

from fastapi.middleware.cors import CORSMiddleware
# uvicorn main:app --reload

app = FastAPI(
    title="Scalable Data Ingestion API",
    description="Backend to fetch and standardize data from multiple providers",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to ["http://localhost:5173"] for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

database: List[Dict[str, Any]] = []

router = APIRouter()


# 🔹 NOAA fetch wrapper (singular product)
def get_noaa_record(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Payload example:
    {
        "station": "8723214",
        "product": "water_temperature",
        "begin_date": "20250101",
        "end_date": "20250105"
    }
    """
    product = payload.get("product")
    if not product or not isinstance(product, str):
        raise HTTPException(status_code=400, detail="Must provide 'product' as a string")

    records = fetch_noaa(payload)  # assumes fetch_noaa handles one product
    return records


PROVIDERS = {
    "open-meteo": fetch_open_meteo,
    "noaa": get_noaa_record,  # 🔹 now singular product
    "obis": fetch_obis,
    "worms": fetch_worms,
    "bold": fetch_bold,
    "fisheries": lambda payload: fetch_fisheries(
        payload, api_key=os.environ.get("DATA_GOV_API_KEY")
    ),
    "csv": fetch_csv,
    "ftp": fetch_ftp,
    # "cmfri": display_report,
}


# Request model
class IngestRequest(BaseModel):
    provider: str
    payload: Dict[str, Any]


@router.post("/providers/noaa")
def noaa_endpoint(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    return get_noaa_record(payload)


@app.post("/ingest/")
def ingest(req: IngestRequest):
    provider = req.provider
    payload = req.payload

    if provider not in PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")

    try:
        records = PROVIDERS[provider](payload)
        database.extend(records)
        return {"status": "success", "records": records}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


import json
from datetime import datetime
from dateutil import parser as dtparser
from models.aidata_models import TrainBatchRequest, PredictRequest, PredictGridRequest  
import numpy as np
import pandas as pd
import math
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


from aimodel import (
    features_from_df,
    build_geojson_grid,
    render_heatmap_to_png,
    train_single_species,
    load_model_meta,
    interpret_prob ,
    safe_float 
)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models_cache")
DATA_CACHE = os.path.join(BASE_DIR, "data_cache")
STATIC_DIR = os.path.join(BASE_DIR, "static")
@app.get("/data/")
def get_data() -> List[Dict[str, Any]]:
    return database

# Model end-points 
@app.get("/status")
def status():
    out = []
    for f in os.listdir(MODELS_DIR):
        if f.endswith("_meta.json"):
            with open(os.path.join(MODELS_DIR, f), "r") as fh:
                out.append(json.load(fh))
    return {"models": out}

# import logging
# logger = logging.getLogger(__name__) 
# @app.post("/train_batch")
# def train_batch(req: TrainBatchRequest):
#     results = []
#     for sp in req.species:
#         try:
#             meta = train_single_species(
#                 scientific_name=sp,
#                 max_records=req.max_records,
#                 test_size=req.test_size,
#                 random_state=req.random_state
#             )
#             results.append(meta)
#             logger = logging.getLogger(__name__)
#             logger.info(f"[API] train_single_species result for {sp}: {meta}")
#         except HTTPException as e:
#             logger = logging.getLogger(__name__)
#             logger.exception(f"[API] HTTPException during training {sp}: {e.detail}")
#             results.append({"scientific_name": sp, "status": "error", "detail": e.detail})
#         except Exception as e:
#             logger = logging.getLogger(__name__)
#             logger.exception(f"[API] Exception during training {sp}: {e}")
#             results.append({"scientific_name": sp, "status": "error", "detail": str(e)})
#     # Log summary
#     logging.getLogger(__name__).info(f"[API] train_batch completed, results: {results}")
#     return {"status": "done", "results": results}
@app.post("/train_batch")
def train_batch(req: TrainBatchRequest):
    """
    Train models for multiple species (one model per species).
    Returns an array of per-species results.
    """
    results = []
    for sp in req.species:
        try:
            meta = train_single_species(
                scientific_name=sp,
                max_records=req.max_records,
                test_size=req.test_size,
                random_state=req.random_state
            )
            results.append(meta)
        except HTTPException as e:
            results.append({"scientific_name": sp, "status": "error", "detail": e.detail})
        except Exception as e:
            results.append({"scientific_name": sp, "status": "error", "detail": str(e)})
    return {"status": "done", "results": results}

 
# ---------- Update predict endpoint to echo coords + interpretation ----------
@app.post("/predict")
def predict_point(req: PredictRequest):
    model, meta = load_model_meta(req.scientific_name)
    if req.event_date:
        try:
            dt = dtparser.parse(req.event_date)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid event_date. Use ISO format e.g. '2020-03-01'.")
    else:
        dt = datetime.utcnow()

    lat = safe_float(req.latitude)
    lon = safe_float(req.longitude)
    depth = safe_float(req.depth_m)

    row = pd.DataFrame([{
        "lat": lat,
        "lon": lon,
        "depth_m": depth,
        "eventDate_parsed": dt
    }])
    X = features_from_df(row)
    proba = safe_float(model.predict_proba(X)[:, 1][0])
    label = interpret_prob(proba)

    return {
        "scientific_name": req.scientific_name,
        "probability": proba,
        "interpretation": label,
        "query": {
            "latitude": lat,
            "longitude": lon,
            "depth_m": depth,
            "event_date": dt.isoformat()
        },
        "meta": meta
    }


# ---------- Add endpoint to remove a model (delete files) ----------
@app.post("/remove_model")
def remove_model(body: Dict[str, str] = Body(...)):
    """
    Remove a trained model and its metadata from the cache.
    Body: {"scientific_name": "Thunnus albacares"}
    """
    name = body.get("scientific_name")
    if not name:
        raise HTTPException(status_code=400, detail="scientific_name is required")
    mfile = os.path.join(MODELS_DIR, f"{name.replace(' ', '_') }_rf.pkl")
    metaf = os.path.join(MODELS_DIR, f"{name.replace(' ', '_')}_meta.json")
    removed = []
    for f in (mfile, metaf):
        try:
            if os.path.exists(f):
                os.remove(f)
                removed.append(os.path.basename(f))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to remove {f}: {e}")
    return {"removed": removed, "scientific_name": name}


@app.get("/list_models")
def list_models():
    results = []
    files = os.listdir(MODELS_DIR)
    print("Listed Models are :", files) 
    for f in os.listdir(MODELS_DIR):
        if f.endswith("_meta.json"):
            path = os.path.join(MODELS_DIR, f)
            try:
                with open(path, "r") as fp:
                    meta = json.load(fp)
                results.append(meta)
            except Exception as e:
                print(f"Failed to load {f}: {e}")
    print("DEBUG: Found meta files:", os.listdir(MODELS_DIR))  # <-- add this
    results.sort(key=lambda m: m.get("trained_at", ""), reverse=True)
    return {"models": results}

@app.post("/predict_grid")
def predict_grid(req: PredictGridRequest):
    model, meta = load_model_meta(req.scientific_name)
    minLon, minLat, maxLon, maxLat = req.bbox
    if not (minLon < maxLon and minLat < maxLat):
        raise HTTPException(status_code=400, detail="Invalid bbox. Use [minLon, minLat, maxLon, maxLat].")
    res = float(req.grid_resolution)

    cols = int(math.ceil((maxLon - minLon) / res))
    rows = int(math.ceil((maxLat - minLat) / res))
    if cols * rows > 40000:
        raise HTTPException(status_code=400, detail="Grid too large. Increase grid_resolution or shrink bbox.")

    lons = np.linspace(minLon + res/2.0, maxLon - res/2.0, num=cols)
    lats = np.linspace(maxLat - res/2.0, minLat + res/2.0, num=rows)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    dt = dtparser.parse(req.event_date) if req.event_date else datetime.utcnow()
    depth_val = safe_float(req.depth_m)

    df_pts = pd.DataFrame({
        "lat": lat_grid.ravel(),
        "lon": lon_grid.ravel(),
        "depth_m": depth_val,
        "eventDate_parsed": [dt]* (rows*cols)
    })

    X = features_from_df(df_pts)
    probs_flat = model.predict_proba(X)[:, 1]
    probs = probs_flat.reshape(rows, cols)

    features = []
    hotspot_centroids = []
    prob_values = []
    prob_threshold_hotspot = 0.7
    prob_threshold_cold = 0.3

    for r in range(rows):
        for c in range(cols):
            south = maxLat - (r+1)*res
            north = maxLat - r*res
            west = minLon + c*res
            east = minLon + (c+1)*res

            p_val = safe_float(probs[r, c])
            prob_values.append(p_val if p_val is not None else 0.0)

            poly = [
                [west, south],
                [east, south],
                [east, north],
                [west, north],
                [west, south]
            ]
            features.append({
                "type": "Feature",
                "properties": {"prob": p_val},
                "geometry": {"type": "Polygon", "coordinates": [poly]}
            })
            if p_val is not None and p_val >= prob_threshold_hotspot:
                hotspot_centroids.append({
                    "lat": safe_float((north + south)/2.0),
                    "lon": safe_float((west + east)/2.0),
                    "prob": p_val
                })

    geojson = {"type": "FeatureCollection", "features": features}

    png_name = f"{req.scientific_name.replace(' ', '_')}_{abs(hash((tuple(req.bbox), res, depth_val, dt.isoformat())))}.png"
    png_path = os.path.join(STATIC_DIR, png_name)
    render_heatmap_to_png(probs, png_path)
    png_url = f"/static/{png_name}"

    avg_prob = safe_float(np.nanmean(probs))
    hotspots = int(np.sum(probs > prob_threshold_hotspot))
    coldspots = int(np.sum(probs < prob_threshold_cold))

    hist_counts, bin_edges = np.histogram(prob_values, bins=5, range=(0.0, 1.0))
    bin_edges = [safe_float(x) for x in bin_edges]

    return {
        "scientific_name": req.scientific_name,
        "bbox": req.bbox,
        "grid_resolution": res,
        "heatmap_png_url": png_url,
        "geojson": geojson,
        "summary": {
            "average_probability": avg_prob,
            "hotspots_cells": hotspots,
            "coldspots_cells": coldspots,
            "total_cells": rows * cols
        },
        "prob_breaks": {
            "hotspot_threshold": prob_threshold_hotspot,
            "coldspot_threshold": prob_threshold_cold,
            "bin_edges": bin_edges,
            "hist_counts": [int(x) for x in hist_counts.tolist()]
        },
        "hotspot_centroids": hotspot_centroids,
        "meta": meta
    }
    
# addon on AI :
"""
FastAPI app for CMLRE: simple proof-of-concept platform that
- validates marine species names via WoRMS
- fetches georeferenced occurrences from GBIF
- enriches occurrence points with sea-surface-temperature (SST) from Open-Meteo
- computes simple, easy-to-understand insights for non-technical users

Requirements:
- Python 3.10+
- pip install fastapi uvicorn httpx pandas numpy matplotlib python-multipart

Run:
    uvicorn fastapi_cmlre_app:app --reload --port 8000

Endpoints:
- GET /validate_species?name=LATIN_NAME
- GET /occurrences?species=LATIN_NAME&limit=100
- GET /insights?species=LATIN_NAME&limit=100  -> returns JSON summary + stats
- GET /insights_plot?species=LATIN_NAME&limit=100 -> returns PNG plot (SST distribution)

Notes:
- This is a compact, self-contained demo. For production use you would add authentication, persistent caching, retries, background jobs for large downloads, and nicer UI.
"""

from fastapi import FastAPI, Query, HTTPException, Response
from typing import List, Optional, Dict, Any
import httpx
import asyncio
import pandas as pd
import numpy as np
from functools import lru_cache
from datetime import datetime
import io
import matplotlib.pyplot as plt
from urllib.parse import quote


GBIF_OCCURRENCE_URL = "https://api.gbif.org/v1/occurrence/search"
WORMS_BY_NAME = "https://www.marinespecies.org/rest/AphiaRecordsByName/{}?like=false&marine_only=true"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

client = httpx.AsyncClient(timeout=30.0)

# @lru_cache(maxsize=256)
async def validate_with_worms(name: str) -> Dict[str, Any]:
    url = WORMS_BY_NAME.format(quote(name))
    r = await client.get(url)
    if r.status_code != 200:
        return {"ok": False, "error": f"WoRMS returned {r.status_code}"}
    data = r.json()
    if not data:
        return {"ok": False, "error": "No match in WoRMS"}
    # take first match
    entry = data[0]
    return {
        "ok": True,
        "scientificname": entry.get("scientificname"),
        "AphiaID": entry.get("AphiaID"),
        "status": entry.get("status"),
        "rank": entry.get("rank"),
        "valid_name": entry.get("valid_name"),
    }


async def fetch_gbif_occurrences(species: str, limit: int = 200) -> List[Dict[str, Any]]:
    params = {
        "scientificName": species,
        "hasCoordinate": "true",
        "limit": min(limit, 300),
        "offset": 0,
    }
    r = await client.get(GBIF_OCCURRENCE_URL, params=params)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail="GBIF API error")
    res = r.json()
    results = res.get("results", [])
    # simplify
    simplified = []
    for rec in results:
        lat = rec.get("decimalLatitude")
        lon = rec.get("decimalLongitude")
        date = rec.get("eventDate") or rec.get("year")
        if lat is None or lon is None:
            continue
        simplified.append({
            "gbifID": rec.get("key"),
            "species": rec.get("scientificName"),
            "lat": float(lat),
            "lon": float(lon),
            "eventDate": date,
            "dataset": rec.get("datasetTitle"),
        })
    return simplified

async def fetch_sst_for_point(lat: float, lon: float, date_str: Optional[str] = None) -> Optional[float]:
    # Open-Meteo archive: request daily sea-surface temperature for given date
    # If date_str missing, use today's date (will likely return empty - then we allow None)
    try:
        if date_str and isinstance(date_str, str) and len(date_str) >= 4:
            # try to parse only the year-month-day part if possible
            try:
                parsed = datetime.fromisoformat(date_str)
                date = parsed.date().isoformat()
            except Exception:
                # if only year provided or bad format, fallback to None
                date = None
        else:
            date = None
        # If we have a date, query daily; otherwise query recent daily using last 3 days
        params = {
            "latitude": lat,
            "longitude": lon,
            "timezone": "UTC",
        }
        if date:
            params.update({"start_date": date, "end_date": date, "daily": "sea_surface_temperature"})
        else:
            # a short recent window
            today = datetime.utcnow().date()
            params.update({"start_date": (today - pd.Timedelta(days=3)).isoformat(),
                           "end_date": today.isoformat(),
                           "daily": "sea_surface_temperature"})
        r = await client.get(OPEN_METEO_ARCHIVE, params=params)
        if r.status_code != 200:
            return None
        data = r.json()
        # try to pull daily sea_surface_temperature
        daily = data.get("daily", {})
        sst_list = daily.get("sea_surface_temperature")
        if sst_list and isinstance(sst_list, list) and len(sst_list) > 0:
            # take median
            return float(np.nanmedian([v for v in sst_list if v is not None]))
        # fallback: some APIs may provide hourly
        hourly = data.get("hourly", {})
        hsst = hourly.get("sea_surface_temperature")
        if hsst and isinstance(hsst, list) and len(hsst) > 0:
            return float(np.nanmedian([v for v in hsst if v is not None]))
        return None
    except Exception:
        return None

@app.get("/validate_species")
async def validate_species(name: str = Query(..., description="Scientific name, e.g. Thunnus albacares")):
    res = await validate_with_worms(name)
    return res

@app.get("/occurrences")
async def occurrences(species: str = Query(...), limit: int = Query(100, ge=1, le=300)):
    occ = await fetch_gbif_occurrences(species, limit)
    # return friendly summary
    summary = {
        "requested_species": species,
        "n_points": len(occ),
        "example_points": occ[:10]
    }
    return summary

@app.get("/insights")
async def insights(species: str = Query(...), limit: int = Query(100, ge=1, le=300)):
    # 1. validate species
    worms = await validate_with_worms(species)
    # 2. fetch occurrences
    occ = await fetch_gbif_occurrences(species, limit)
    if len(occ) == 0:
        raise HTTPException(status_code=404, detail="No georeferenced occurrences found for this species")
    # 3. fetch SST for each point (async gather, but cap concurrency)
    sem = asyncio.Semaphore(8)
    async def worker(pt):
        async with sem:
            sst = await fetch_sst_for_point(pt["lat"], pt["lon"], pt.get("eventDate"))
            return {**pt, "sst": sst}
    tasks = [worker(pt) for pt in occ]
    enriched = await asyncio.gather(*tasks)
    df = pd.DataFrame(enriched)
    # compute stats ignoring nulls
    sst_vals = df["sst"].dropna().astype(float)
    if len(sst_vals) == 0:
        stats = {"message": "No SST data available for these points"}
    else:
        stats = {
            "count_sst": int(len(sst_vals)),
            "mean_sst_c": float(np.round(sst_vals.mean(), 2)),
            "median_sst_c": float(np.round(sst_vals.median(), 2)),
            "min_sst_c": float(np.round(sst_vals.min(), 2)),
            "max_sst_c": float(np.round(sst_vals.max(), 2)),
        }
    # prepare user-friendly explanation
    explanation = (
        f"We found {len(occ)} georeferenced records for '{species}'. "
        + (f"For {stats.get('count_sst',0)} of those we obtained sea-surface temperature (SST) data. "
           f"The SST values range from {stats.get('min_sst_c','N/A')}°C to {stats.get('max_sst_c','N/A')}°C, "
           f"with a mean of {stats.get('mean_sst_c','N/A')}°C. This suggests the species is typically observed in waters around the stated temperatures."
           if isinstance(stats, dict) and stats.get('count_sst',0)>0 else "We couldn't retrieve SST values for these locations.")
    )
    # pick top 5 hotspots by number of records (approx using clustering by rounding coords)
    df['lat_r'] = df['lat'].round(2)
    df['lon_r'] = df['lon'].round(2)
    hotspots = df.groupby(['lat_r','lon_r']).size().reset_index(name='n').sort_values('n',ascending=False).head(5)
    hotspots_list = hotspots.to_dict('records')
    return {
        "species_validation": worms,
        "stats": stats,
        "explanation": explanation,
        "hotspots": hotspots_list,
        "sample_enriched_points": df.head(20).to_dict('records')
    }

@app.get("/insights_plot")
async def insights_plot(species: str = Query(...), limit: int = Query(100, ge=1, le=300)):
    resp = await insights(species, limit)
    # extract sst list
    sst_vals = [pt.get('sst') for pt in resp.get('sample_enriched_points', [])]
    sst_vals = [v for v in sst_vals if v is not None]
    if len(sst_vals) == 0:
        raise HTTPException(status_code=404, detail="No SST values to plot")
    # plot histogram
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(sst_vals, bins=12)
    ax.set_title(f"SST distribution for {species} (n={len(sst_vals)})")
    ax.set_xlabel('SST (°C)')
    ax.set_ylabel('Count')
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")

# Graceful shutdown
@app.on_event("shutdown")
async def shutdown_event():
    await client.aclose()

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000, log_level='info')







