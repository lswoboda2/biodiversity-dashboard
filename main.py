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

def load_data() -> pd.DataFrame:
    """
    Loads and concatenates all parquet files from the data directory on-demand.
    This version is robust to missing columns in individual files.
    """
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
            _df["year"] = _df["Date"].dt.year
            _df["month"] = _df["Date"].dt.month
        
        for col in ["english_name", "species", "obs", "taxa"]:
            if col in _df.columns:
                _df[col] = _df[col].astype("category")
        
        if 'count' in _df.columns:
            _df['count'] = pd.to_numeric(_df['count'], errors='coerce')

        if 'id' not in _df.columns:
            _df.reset_index(inplace=True)
            _df = _df.rename(columns={'index': 'id'})

        return _df
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail="Could not load or process data files.")


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

@app.get("/api/management_years")
def get_management_years():
    years = []
    pattern = re.compile(r"management_(\d{4}-\d{2})\.geojson")
    for f in DATA_PATH.glob("management_*.geojson"):
        match = pattern.match(f.name)
        if match:
            years.append(match.group(1))
    return sorted(years, reverse=True)

@app.get("/api/management_points")
def get_management_points(year: str):
    management_file = DATA_PATH / f"management_{year}.geojson"
    if not management_file.is_file():
        raise HTTPException(
            status_code=404, 
            detail=f"Management data for year {year} not found."
        )
    with open(management_file, 'r') as f:
        data = json.load(f)
    return JSONResponse(content=data)

@app.get("/api/cameratrap_years")
def get_cameratrap_years():
    years = []
    pattern = re.compile(r"cameratraps_(\d{4}-\d{2})\.geojson")
    for f in DATA_PATH.glob("cameratraps_*.geojson"):
        match = pattern.match(f.name)
        if match:
            years.append(match.group(1))
    return sorted(years, reverse=True)

@app.get("/api/cameratrap_points")
def get_cameratrap_points(year: str):
    cameratrap_file = DATA_PATH / f"cameratraps_{year}.geojson"
    if not cameratrap_file.is_file():
        raise HTTPException(
            status_code=404, 
            detail=f"Camera trap data for year {year} not found."
        )
    with open(cameratrap_file, 'r') as f:
        data = json.load(f)
    return JSONResponse(content=data)

@app.get("/api/habitat_polygons")
def get_habitat_polygons(year: Optional[str] = "2024-25"):
    habitat_file = DATA_PATH / f"habitats_{year}.geojson"
    if not habitat_file.is_file():
        raise HTTPException(
            status_code=404, 
            detail=f"Habitat data for year {year} not found."
        )
    with open(habitat_file, 'r') as f:
        data = json.load(f)
    return JSONResponse(content=data)

@app.get("/api/summary/habitat")
def get_habitat_summary():
    summary_file = DATA_PATH / "habitat_summary.json"
    if not summary_file.is_file():
        raise HTTPException(
            status_code=404, 
            detail="Habitat summary file not found."
        )
    with open(summary_file, 'r') as f:
        data = json.load(f)
    return JSONResponse(content=data)


@app.get("/api/all_unique_species")
def get_all_unique_species(page: int = 1, page_size: int = 10):
    df = load_data()
    all_unique_species = sorted(df['species'].dropna().unique().tolist())
    
    total_species = len(all_unique_species)
    start_index = (page - 1) * page_size
    end_index = start_index + page_size
    paginated_species = all_unique_species[start_index:end_index]

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
    base_df = load_data()

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
    df = load_data()
    df.sort_values(by=['Date', 'species', 'id'], ascending=[True, True, True], inplace=True)
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
    df = load_data()
    df.sort_values(by=['Date', 'species', 'id'], ascending=[True, True, True], inplace=True)
    query_df = apply_filters(df, english_name, species, obs, taxa, year, month, bbox)
    try:
        sorted_ids = query_df['id'].tolist()
        position = sorted_ids.index(record_id)
        page = floor(position / page_size) + 1
        return {"page": page}
    except (ValueError, IndexError):
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
    df = load_data()
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
    df = load_data()
    query_df = apply_filters(df, english_name, species, obs, taxa, year, month, bbox)
    
    if query_df.empty:
        return {"shannon": 0, "simpson": 0, "species_richness": 0, "total_records": 0}

    use_count_column = "count" in query_df.columns and query_df["count"].notna().sum() > (len(query_df) / 2)

    if use_count_column:
        species_counts = query_df.groupby("species", observed=True)["count"].sum()
    else:
        species_counts = query_df.groupby("species", observed=True).size()

    species_richness = len(species_counts)
    shannon_index = 0
    gini_simpson_index = 0

    if not species_counts.empty and species_counts.sum() > 0 and species_richness > 1:
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
    df = load_data()
    if 'year' not in df.columns or df['year'].isnull().all():
        return {"trends": []}

    yearly_data = []
    for year, group in sorted(df.groupby('year', observed=True), key=lambda x: x[0]):
        total_records = len(group)
        
        use_count_column = "count" in group.columns and group["count"].notna().sum() > (len(group) / 2)

        if use_count_column:
            species_counts = group.groupby("species", observed=True)["count"].sum()
        else:
            species_counts = group.groupby("species", observed=True).size()

        species_richness = len(species_counts)
        shannon_index = 0
        gini_simpson_index = 0

        if not species_counts.empty and species_counts.sum() > 0 and species_richness > 1:
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
    df = load_data()
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
    df = load_data()
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
    df = load_data()
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
    df = load_data()
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