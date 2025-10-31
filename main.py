import os
import sys
import pandas as pd
import numpy as np
import json
import re
from fastapi import FastAPI, HTTPException, Response
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRouter
from typing import Optional, List
from pathlib import Path
from math import ceil, floor
from scipy.stats import entropy
import openpyxl

app = FastAPI(title="Combined Biodiversity API", version="1.0.0")

origins = [
    "https://biodiversitydashboard-ls.netlify.app",
    "https://biodiversity-actionplan.netlify.app"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.head("/")
def read_root_head():
    return Response(status_code=200)

dashboard_router = APIRouter(prefix="/dashboard")

DATA_PATH = Path(__file__).parent / "data"
_cached_df = None

def get_dataframe() -> pd.DataFrame:
    global _cached_df
    if _cached_df is None:
        print("Cache is empty. Loading data from disk...")
        parquet_files = list(DATA_PATH.glob("*.parquet"))
        if not parquet_files:
            raise HTTPException(status_code=500, detail="No parquet data files found on server.")
        try:
            df_list = [pd.read_parquet(file) for file in parquet_files]
            _df = pd.concat(df_list, ignore_index=True)
            if "Taxa" in _df.columns:
                _df = _df.rename(columns={"Taxa": "taxa"})
            if "Date" in _df.columns:
                _df["Date"] = pd.to_datetime(_df["Date"])
                _df["month"] = _df["Date"].dt.month
            if "year" in _df.columns:
                _df["year"] = _df["year"].astype("category")
            for col in ["english_name", "species", "obs", "taxa"]:
                if col in _df.columns:
                    _df[col] = _df[col].astype("category")
            if 'count' in _df.columns:
                _df['count'] = pd.to_numeric(_df['count'], errors='coerce')
            if 'id' not in _df.columns:
                _df.reset_index(inplace=True)
                _df = _df.rename(columns={'index': 'id'})
            _cached_df = _df
            print("Data loaded and cached successfully.")
        except Exception as e:
            print(f"Error loading data: {e}", file=sys.stderr)
            raise HTTPException(status_code=500, detail="Could not load or process data files.")
    return _cached_df

def apply_filters(query_df: pd.DataFrame, english_name: Optional[str] = None, species: Optional[str] = None,
                  obs: Optional[str] = None, taxa: Optional[str] = None, year: Optional[str] = None,
                  month: Optional[int] = None, bbox: Optional[str] = None) -> pd.DataFrame:
    df = query_df.copy()
    if english_name:
        df = df[df["english_name"].isin(english_name.split(","))]
    if species:
        df = df[df["species"].isin(species.split(","))]
    if obs:
        df = df[df["obs"].isin(obs.split(","))]
    if taxa:
        df = df[df["taxa"].isin(taxa.split(","))]
    if year:
        df = df[df["year"] == year]
    if month:
        df = df[df["month"] == int(month)]
    if bbox:
        try:
            xmin, ymin, xmax, ymax = map(float, bbox.split(','))
            df = df[(df['longitude'] >= xmin) & (df['longitude'] <= xmax) &
                    (df['latitude'] >= ymin) & (df['latitude'] <= ymax)]
        except (ValueError, IndexError):
            pass
    return df

def _get_options(df_source: pd.DataFrame, key_name: str):
    return sorted(df_source[key_name].dropna().unique().tolist())

@dashboard_router.get("/")
def root():
    return {"status": "ok"}

@dashboard_router.get("/health")
def health():
    return {"ok": True}

@dashboard_router.get("/api/management_years")
def get_management_years():
    years = []
    pattern = re.compile(r"management_(\d{4}-\d{2})\.geojson")
    for f in DATA_PATH.glob("management_*.geojson"):
        match = pattern.match(f.name)
        if match:
            years.append(match.group(1))
    return sorted(years, reverse=True)

@dashboard_router.get("/api/cameratrap_years")
def get_cameratrap_years():
    years = []
    pattern = re.compile(r"cameratrap_(\d{4})\.geojson")
    for f in DATA_PATH.glob("cameratrap_*.geojson"):
        match = pattern.match(f.name)
        if match:
            years.append(match.group(1))
    return sorted(set(years), reverse=True)

@dashboard_router.get("/api/filter-options")
def get_filter_options():
    df = get_dataframe()
    opts = {
        "english_name": _get_options(df, "english_name"),
        "species": _get_options(df, "species"),
        "obs": _get_options(df, "obs"),
        "taxa": _get_options(df, "taxa"),
        "year": sorted(df["year"].dropna().unique().tolist()) if "year" in df.columns else [],
        "month": list(range(1, 13))
    }
    return opts

@dashboard_router.get("/api/records")
def get_records(english_name: Optional[str] = None, species: Optional[str] = None, obs: Optional[str] = None,
                taxa: Optional[str] = None, year: Optional[str] = None, month: Optional[int] = None,
                bbox: Optional[str] = None, page: int = 1, page_size: int = 100):
    df = get_dataframe()
    filtered = apply_filters(df, english_name, species, obs, taxa, year, month, bbox)
    total = len(filtered)
    start = (page - 1) * page_size
    end = start + page_size
    page_data = filtered.iloc[start:end].dropna(subset=["latitude", "longitude"]).to_dict("records")
    return {
        "records": page_data,
        "total": total,
        "page": page,
        "page_size": page_size
    }

@dashboard_router.get("/api/management_points")
def get_management_points(year: Optional[str] = None):
    if not year:
        return []
    file_path = DATA_PATH / f"management_{year}.geojson"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Management data not found for year")
    with open(file_path) as f:
        data = json.load(f)
    return {"features": data.get("features", [])}

@dashboard_router.get("/api/cameratrap")
def get_cameratrap(year: Optional[str] = None):
    if not year:
        return []
    file_path = DATA_PATH / f"cameratrap_{year}.geojson"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Camera trap data not found for year")
    with open(file_path) as f:
        data = json.load(f)
    return {"features": data.get("features", [])}

@dashboard_router.get("/api/heatmap")
def get_heatmap(taxa: Optional[str] = None, year: Optional[str] = None):
    df = get_dataframe()
    filtered = df
    if taxa:
        filtered = filtered[filtered["taxa"] == taxa]
    if year:
        filtered = filtered[filtered["year"] == year]
    heat_data = filtered.groupby(["latitude", "longitude"]).size().reset_index(name="count").to_dict("records")
    return {"heatmap_data": heat_data}

actionplan_router = APIRouter(prefix="/actionplan")

DATA_XLSX = Path(__file__).parent / "data" / "source.xlsx"
_cached_ap_df = None
_cached_mtime = None

SPECIES_COLUMNS = [
    "all", "invertebrate", "bat", "mammal", "bird", "amphibians",
    "reptiles", "invasive", "plants", "freshwater", "coastal"
]

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(col).strip() for col in df.columns]
    df = df.rename(columns={
        "Priority 2025 (3 high, 1 low)": "priority_2025",
        "Lead Contact (provisional)": "lead_contact",
        "Implementation ranking": "implementation_ranking",
    })
    if "priority_2025" in df.columns:
        df["priority_2025"] = pd.to_numeric(df["priority_2025"], errors="coerce")
    return df

def _load_dataframe_from_disk() -> pd.DataFrame:
    if not DATA_XLSX.exists():
        raise HTTPException(status_code=500, detail=f"Data file not found: {DATA_XLSX.name}")
    df = pd.read_excel(DATA_XLSX, engine="openpyxl", skiprows=1)
    df = clean_column_names(df)
    if "Action" in df.columns:
        df = df.dropna(subset=["Action"])
    if "Status" in df.columns:
        df["Status"] = df["Status"].astype(str).str.strip().str.lower().str.capitalize()
    def get_affected_species(row):
        return [col for col in SPECIES_COLUMNS if row.get(col) == 1]
    df["affected_species"] = df.apply(get_affected_species, axis=1)
    return df

def get_ap_dataframe() -> pd.DataFrame:
    global _cached_ap_df, _cached_mtime
    try:
        mtime = DATA_XLSX.stat().st_mtime
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"Data file not found: {DATA_XLSX.name}")
    if _cached_ap_df is None or _cached_mtime != mtime:
        _cached_ap_df = _load_dataframe_from_disk()
        _cached_mtime = mtime
    return _cached_ap_df

def apply_ap_filters(df: pd.DataFrame, status: Optional[str] = None, strategy: Optional[str] = None,
                     timescale: Optional[str] = None, implementation: Optional[str] = None,
                     impact: Optional[str] = None, priority: Optional[str] = None, species: Optional[str] = None) -> pd.DataFrame:
    q = df.copy()
    if status:
        q = q[q["Status"] == status]
    if strategy:
        q = q[q["Strategy 2025"] == strategy]
    if timescale:
        q = q[q["Timescale"] == timescale]
    if implementation:
        q = q[q["implementation_ranking"] == implementation]
    if impact:
        q = q[q["Impact"] == impact]
    if priority:
        try:
            q = q[q["priority_2025"] == float(priority)]
        except (ValueError, TypeError):
            pass
    if species:
        selected = [s.strip() for s in species.split(",") if s.strip()]
        if selected:
            q = q[q["affected_species"].apply(lambda lst: any(s in lst for s in selected))]
    return q

def get_summary_stats(df: pd.DataFrame) -> dict:
    total = len(df)
    counts = df["Status"].value_counts().to_dict() if "Status" in df.columns else {}
    for k in ["Achieved and ongoing", "Underway", "Not started"]:
        counts.setdefault(k, 0)
    pct = {k: round((v / total) * 100, 1) if total else 0 for k, v in counts.items()}
    return {"total_actions": total, "status_counts": counts, "status_percentages": pct}

@actionplan_router.get("/")
def read_root():
    return {"message": "Biodiversity Action Plan API"}

@actionplan_router.get("/health")
def health():
    return {"ok": True}

@actionplan_router.get("/api/filter-options")
def get_filter_options():
    df = get_ap_dataframe()
    opt = {
        "status": sorted(df["Status"].dropna().unique().tolist()) if "Status" in df.columns else [],
        "strategy": sorted(df["Strategy 2025"].dropna().unique().tolist()) if "Strategy 2025" in df.columns else [],
        "timescale": sorted(df["Timescale"].dropna().unique().tolist()) if "Timescale" in df.columns else [],
        "implementation": sorted(df["implementation_ranking"].dropna().unique().tolist()) if "implementation_ranking" in df.columns else [],
        "impact": sorted(df["Impact"].dropna().unique().tolist()) if "Impact" in df.columns else [],
        "priority": sorted([str(int(p)) for p in df["priority_2025"].dropna().unique()]) if "priority_2025" in df.columns else [],
        "species": SPECIES_COLUMNS,
    }
    return opt

@actionplan_router.get("/api/actions")
def get_actions(status: Optional[str] = None, strategy: Optional[str] = None, timescale: Optional[str] = None,
                implementation: Optional[str] = None, impact: Optional[str] = None, priority: Optional[str] = None,
                species: Optional[str] = None, page: int = 1, page_size: int = 10):
    df = get_ap_dataframe()
    filtered = apply_ap_filters(df, status, strategy, timescale, implementation, impact, priority, species)
    stats = get_summary_stats(filtered)
    total_records = len(filtered)
    total_pages = ceil(total_records / page_size) if page_size > 0 else 1
    start = max(0, (page - 1) * page_size)
    end = start + page_size
    page_df = filtered.iloc[start:end] if page_size > 0 else filtered
    out_df = page_df.drop(columns=SPECIES_COLUMNS, errors="ignore")
    cleaned = out_df.astype(object).where(pd.notnull(out_df), None)
    actions = cleaned.to_dict(orient="records")
    return {
        "summary_stats": stats,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_records": total_records,
            "total_pages": total_pages,
        },
        "actions": actions,
    }

@actionplan_router.get("/api/summary-stats")
def get_summary_stats_endpoint():
    df = get_ap_dataframe()
    return get_summary_stats(df)

app.include_router(dashboard_router)
app.include_router(actionplan_router)

@app.get("/")
def combined_root():
    return {"status": "ok", "message": "Combined Biodiversity API"}
