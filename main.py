import os
import sys
import pandas as pd
import numpy as np
import json
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from typing import Optional, List
from pathlib import Path
from math import ceil, floor
from scipy.stats import entropy

app = FastAPI(title="Biodiversity Dashboard", version="1.0.0")

origins = [
    "https://biodiversitydashboard-ls.netlify.app",
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

DATA_PATH = Path(__file__).parent / "data"
df = None
ALL_UNIQUE_SPECIES = []

def load_data() -> pd.DataFrame:
    if not DATA_PATH.is_dir():
        print(f"ERROR: Data directory not found: {DATA_PATH}", file=sys.stderr)
        raise FileNotFoundError(str(DATA_PATH))

    parquet_files = list(DATA_PATH.glob("*.parquet"))
    if not parquet_files:
        print(f"ERROR: No .parquet files found in directory: {DATA_PATH}", file=sys.stderr)
        raise FileNotFoundError(f"No .parquet files in {DATA_PATH}")

    df_list = [pd.read_parquet(file) for file in parquet_files]
    _df = pd.concat(df_list, ignore_index=True)

    for col in ["english_name", "species", "obs", "taxa"]:
        if col in _df.columns:
            _df[col] = _df[col].astype("category")

    _df["Date"] = pd.to_datetime(_df["Date"])
    _df["year"] = _df["Date"].dt.year
    _df["month"] = _df["Date"].dt.month
    if "Taxa" in _df.columns:
        _df = _df.rename(columns={"Taxa": "taxa"})
    
    if 'latitude' in _df.columns:
        _df['latitude'] = pd.to_numeric(_df['latitude'], errors='coerce')
    if 'longitude' in _df.columns:
        _df['longitude'] = pd.to_numeric(_df['longitude'], errors='coerce')
    
    _df.reset_index(inplace=True)
    _df = _df.rename(columns={'index': 'id'})
        
    return _df

try:
    df = load_data()
    df.sort_values(by=['Date', 'species', 'id'], ascending=[True, True, True], inplace=True)
    ALL_UNIQUE_SPECIES = sorted(df['species'].dropna().unique().tolist())
except Exception as e:
    raise RuntimeError(f"Failed to load dataset: {e}") from e

def apply_filters(
    query_df: pd.DataFrame,
    english_name: Optional[str] = None,
    species: Optional[str] = None,
    obs: Optional[str] = None,
    taxa: Optional[str] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
    bbox: Optional[str] = None,
) -> pd.DataFrame:
    if english_name:
        query_df = query_df[query_df["english_name"].isin(english_name.split(","))]
    if species:
        query_df = query_df[query_df["species"].isin(species.split(","))]
    if obs:
        query_df = query_df[query_df["obs"].isin(obs.split(","))]
    if taxa:
        query_df = query_df[query_df["taxa"].isin(taxa.split(","))]
    if year:
        query_df = query_df[query_df["year"] == int(year)]
    if month:
        query_df = query_df[query_df["month"] == int(month)]
    if bbox:
        try:
            xmin, ymin, xmax, ymax = map(float, bbox.split(','))
            query_df = query_df[
                (query_df['longitude'] >= xmin) &
                (query_df['longitude'] <= xmax) &
                (query_df['latitude'] >= ymin) &
                (query_df['latitude'] <= ymax)
            ]
        except (ValueError, IndexError):
            pass
    return query_df

def _get_options(df_source: pd.DataFrame, key_name: str):
    return sorted(df_source[key_name].dropna().unique().tolist())

@app.get("/")
def root():
    return {"status": "ok", "message": "Biodiversity Dashboard API root"}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/api/habitat_polygons")
def get_habitat_polygons():
    habitat_file = DATA_PATH / "habitats.geojson"
    if not habitat_file.is_file():
        raise HTTPException(status_code=404, detail="Habitat data not found. Please run the conversion script.")
    
    with open(habitat_file, 'r') as f:
        data = json.load(f)
    return JSONResponse(content=data)

HABITAT_CLASSIFICATION_MAP = {
    "garden": ['c1f6', 'c1f7', 'Suburban mosaic vegetation', 'Gardens'],
    "freshwater": ['f2d', 'Aquatic marginal vegetation', 'Reeds', 'rivers', 'Ponds'],
    "natural grassland": ['g3c5', 'tall grassland/meadows', 's3a7'],
    "modified grassland": ['g4', 'Modified grassland'],
    "intertidal mudflat": ['t2d5'],
    "hedge": ['h2b', 'Hedges/hedgerows'],
    "buildings": ['u1e', 'Buildings'],
    "other developed land": ['u1b6', 'Other developed land'],
    "unsealed unvegetated surface": ['u1c', 'u1d', 'Artificial unvegetated, unsealed surface', 'Soft path'],
    "woodland": ['w1g7', 'broadleaved woodland', 'coniferous woodland', 'Line of trees', 'mixed woodland']
}
REVERSE_HABITAT_MAP = {value.lower(): key for key, values in HABITAT_CLASSIFICATION_MAP.items() for value in values}

def classify_habitat(properties: dict) -> str:
    broad_classification = properties.get("habitat classification (broad)")
    
    if broad_classification and isinstance(broad_classification, str) and broad_classification.lower() != 'unknown':
        source_value = broad_classification
    else:
        source_value = properties.get("UK habitat primary classification")

    if not source_value or not isinstance(source_value, str):
        return "Other"

    return REVERSE_HABITAT_MAP.get(source_value.lower(), "Other")

@app.get("/api/summary/habitat")
def get_habitat_summary(year: Optional[int] = None):
    if year and year != 2025:
        return {"summary": [], "totals": {"total_area_ha": 0, "total_no10sq": 0}}

    habitat_file = DATA_PATH / "habitats.geojson"
    metrics_file = DATA_PATH / "habitat_metrics.json"

    if not habitat_file.is_file() or not metrics_file.is_file():
        raise HTTPException(status_code=404, detail="Habitat data or metrics file not found.")

    with open(metrics_file, 'r') as f:
        habitat_metrics = json.load(f)
    
    with open(habitat_file, 'r') as f:
        geojson_data = json.load(f)

    habitat_areas = {}
    total_area_m2 = 0

    for feature in geojson_data.get("features", []):
        properties = feature.get("properties", {})
        area = properties.get("newarea") or 0
        if area > 0:
            habitat_name = classify_habitat(properties)
            habitat_areas.setdefault(habitat_name, 0)
            habitat_areas[habitat_name] += area
            total_area_m2 += area
    
    if total_area_m2 == 0:
        return {"summary": [], "totals": {"total_area_ha": 0, "total_no10sq": 0}}

    summary_list = []
    total_no10sq = sum(habitat_metrics.values())
    all_habitats = sorted(list(set(habitat_areas.keys()) | set(habitat_metrics.keys())))

    for habitat in all_habitats:
        area_m2 = habitat_areas.get(habitat, 0)
        no10sq = habitat_metrics.get(habitat, 0)
        area_ha = area_m2 / 10000
        percent_area = (area_m2 / total_area_m2) * 100 if total_area_m2 > 0 else 0
        percent_sq = (no10sq / total_no10sq) * 100 if total_no10sq > 0 else 0
        
        summary_list.append({
            "habitat": habitat.replace("_", " ").title(),
            "areaha": round(area_ha, 2),
            "percent": round(percent_area, 1),
            "no10sq": no10sq,
            "no10sq_percent": round(percent_sq, 1)
        })

    return {
        "summary": summary_list,
        "totals": {
            "total_area_ha": round(total_area_m2 / 10000, 2),
            "total_no10sq": total_no10sq
        }
    }

@app.get("/api/summary/management_biodiversity")
def get_management_biodiversity_summary():
    metrics_file = DATA_PATH / "3gpkg_metrics.json"
    if not metrics_file.is_file():
        raise HTTPException(status_code=404, detail="Special habitat metrics file not found. Please run the calculation script.")
    
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    return JSONResponse(content=data)

@app.get("/api/all_unique_species")
def get_all_unique_species(page: int = 1, page_size: int = 10):
    total_species = len(ALL_UNIQUE_SPECIES)
    start_index = (page - 1) * page_size
    end_index = start_index + page_size
    paginated_species = ALL_UNIQUE_SPECIES[start_index:end_index]

    return {
        "species_list": paginated_species,
        "total_records": total_species,
        "page": page,
        "total_pages": ceil(total_species / page_size)
    }

@app.get("/api/filter-options")
def get_filter_options(
    english_name: Optional[str] = None,
    species: Optional[str] = None,
    obs: Optional[str] = None,
    taxa: Optional[str] = None,
    year: Optional[str] = None,
    month: Optional[str] = None,
):
    base_df = df

    options = {}
    temp_df = apply_filters(base_df, species=species, obs=obs, taxa=taxa, year=year, month=month)
    options["english_name"] = _get_options(temp_df, "english_name")

    temp_df = apply_filters(base_df, english_name=english_name, obs=obs, taxa=taxa, year=year, month=month)
    options["species"] = _get_options(temp_df, "species")

    temp_df = apply_filters(base_df, english_name=english_name, species=species, taxa=taxa, year=year, month=month)
    options["obs"] = _get_options(temp_df, "obs")

    temp_df = apply_filters(base_df, english_name=english_name, species=species, obs=obs, year=year, month=month)
    options["taxa"] = _get_options(temp_df, "taxa")

    temp_df = apply_filters(base_df, english_name=english_name, species=species, obs=obs, taxa=taxa, month=month)
    options["year"] = _get_options(temp_df, "year")

    temp_df = apply_filters(base_df, english_name=english_name, species=species, obs=obs, taxa=taxa, year=year)
    options["month"] = _get_options(temp_df, "month")

    return options

@app.get("/api/records")
def get_records(
    page: int = 1,
    page_size: int = 100,
    english_name: Optional[str] = None,
    species: Optional[str] = None,
    obs: Optional[str] = None,
    taxa: Optional[str] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
    bbox: Optional[str] = None,
):
    query_df = apply_filters(df, english_name, species, obs, taxa, year, month, bbox)
    total_records = len(query_df)
    paginated_data = query_df.iloc[(page - 1) * page_size : page * page_size].copy()
    paginated_data = (
        paginated_data.replace([np.inf, -np.inf], None)
        .astype(object)
        .where(pd.notnull(paginated_data), None)
    )
    return jsonable_encoder(
        {
            "total_records": total_records,
            "page": page,
            "total_pages": int(np.ceil(total_records / page_size)),
            "records": paginated_data.to_dict(orient="records"),
        }
    )

@app.get("/api/record_page")
def get_record_page(
    record_id: int,
    page_size: int = 100,
    english_name: Optional[str] = None,
    species: Optional[str] = None,
    obs: Optional[str] = None,
    taxa: Optional[str] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
    bbox: Optional[str] = None,
):
    query_df = apply_filters(df, english_name, species, obs, taxa, year, month, bbox)
    
    try:
        indices = query_df.index.tolist()
        record_original_index = df.loc[df['id'] == record_id].index[0]
        position = indices.index(record_original_index)
        
        page = floor(position / page_size) + 1
        return {"page": page}
    except (IndexError, ValueError):
        raise HTTPException(status_code=404, detail="Record not found in the current filter context.")


@app.get("/api/map_data")
def get_map_data(
    english_name: Optional[str] = None,
    species: Optional[str] = None,
    obs: Optional[str] = None,
    taxa: Optional[str] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
    bbox: Optional[str] = None,
):
    query_df = apply_filters(df, english_name, species, obs, taxa, year, month, bbox)
    
    map_df = query_df.dropna(subset=['latitude', 'longitude']).copy()
    map_df = map_df[['id', 'english_name', 'species', 'obs', 'Date', 'taxa', 'latitude', 'longitude']]
    
    map_df = (
        map_df.replace([np.inf, -np.inf], None)
        .astype(object)
        .where(pd.notnull(map_df), None)
    )
    
    records = map_df.to_dict(orient="records")
    return JSONResponse(content=jsonable_encoder(records))


@app.get("/api/summary/diversity")
def get_diversity_summary(
    english_name: Optional[str] = None,
    species: Optional[str] = None,
    obs: Optional[str] = None,
    taxa: Optional[str] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
    bbox: Optional[str] = None,
):
    query_df = apply_filters(df, english_name, species, obs, taxa, year, month, bbox)
    species_counts = query_df.groupby("species", observed=True)["count"].sum() if not query_df.empty else pd.Series(dtype=float)
    species_richness = len(species_counts)

    if query_df.empty or "count" not in query_df.columns or query_df["count"].sum() == 0 or species_richness <= 1:
        return {"shannon": 0, "simpson": 0, "species_richness": species_richness, "total_records": len(query_df)}

    proportions = species_counts[species_counts > 0] / species_counts.sum()
    shannon_index = entropy(proportions, base=np.e)
    gini_simpson_index = 1 - (proportions**2).sum()

    return {
        "shannon": round(float(shannon_index), 3),
        "simpson": round(float(gini_simpson_index), 3),
        "species_richness": int(species_richness),
        "total_records": len(query_df)
    }

@app.get("/api/summary/annual_trends")
def get_annual_trends():
    if 'year' not in df.columns or df['year'].isnull().all():
        return {"trends": []}

    yearly_data = []
    for year, group in sorted(df.groupby('year'), key=lambda x: x[0]):
        species_counts = group.groupby("species", observed=True)["count"].sum()
        species_richness = len(species_counts)
        total_records = len(group)

        shannon_index = 0
        gini_simpson_index = 0

        if not group.empty and "count" in group.columns and group["count"].sum() > 0 and species_richness > 1:
            proportions = species_counts[species_counts > 0] / species_counts.sum()
            shannon_index = entropy(proportions, base=np.e)
            gini_simpson_index = 1 - (proportions**2).sum()

        yearly_data.append({
            "year": int(year),
            "total_records": int(total_records),
            "unique_species": int(species_richness),
            "shannon": round(float(shannon_index), 3),
            "simpson": round(float(gini_simpson_index), 3),
        })
    
    return {"trends": yearly_data}

@app.get("/api/summary/species_distribution")
def get_species_distribution(
    english_name: Optional[str] = None,
    species: Optional[str] = None,
    obs: Optional[str] = None,
    taxa: Optional[str] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
    bbox: Optional[str] = None,
):
    query_df = apply_filters(df, english_name, species, obs, taxa, year, month, bbox)
    if query_df.empty:
        return []

    species_counts = query_df['english_name'].value_counts()
    top_20_names = species_counts.nlargest(20).index.tolist()

    top_20_df = query_df[query_df['english_name'].isin(top_20_names)]
    taxa_map = top_20_df.groupby('english_name', observed=True)['taxa'].first()

    result = []
    for name in top_20_names:
        result.append({
            "name": name,
            "count": int(species_counts[name]),
            "taxa": taxa_map.get(name, "Unknown")
        })

    return result

@app.get("/api/summary/temporal_trends")
def get_temporal_trends(
    english_name: Optional[str] = None,
    species: Optional[str] = None,
    obs: Optional[str] = None,
    taxa: Optional[str] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
    bbox: Optional[str] = None,
):
    query_df = apply_filters(df, english_name, species, obs, taxa, year, month, bbox)
    if query_df.empty:
        return {}
    summary = query_df.groupby("month").size().reindex(range(1, 13), fill_value=0)
    return summary.to_dict()

@app.get("/api/summary/observer_comparison")
def get_observer_comparison(
    english_name: Optional[str] = None,
    species: Optional[str] = None,
    obs: Optional[str] = None,
    taxa: Optional[str] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
    bbox: Optional[str] = None,
):
    if not obs:
        return {}
    query_df = apply_filters(df, english_name, species, taxa=taxa, year=year, month=month, bbox=bbox)
    query_df = query_df[query_df["obs"].isin(obs.split(","))]
    if query_df.empty:
        return {}
    comparison = query_df.groupby(["obs", "taxa"], observed=True).size().unstack(fill_value=0)
    return comparison.to_dict(orient="dict")

@app.get("/api/summary/observer/{observer_name}")
def get_observer_stats(
    observer_name: str,
    english_name: Optional[str] = None,
    species: Optional[str] = None,
    taxa: Optional[str] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
    bbox: Optional[str] = None,
):
    query_df = apply_filters(df, english_name, species, None, taxa, year, month, bbox)
    observer_df = query_df[query_df["obs"] == observer_name]

    if observer_df.empty:
        return {}

    specialization = observer_df.groupby('taxa', observed=True).size().sort_values(ascending=False)
    other_breakdown = {}

    if len(specialization) > 20:
        top_20 = specialization.head(20)
        other_taxa = specialization.tail(-20)
        other_sum = other_taxa.sum()

        if other_sum > 0:
            other_breakdown = other_taxa.to_dict()
            other_series = pd.Series([other_sum], index=['Other'])
            specialization = pd.concat([top_20, other_series])

    return {
        "specialization": specialization.to_dict(),
        "other_breakdown": other_breakdown
    }