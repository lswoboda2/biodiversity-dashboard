import os
import sys
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from pathlib import Path

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

DATA_PATH = Path(__file__).parent / "alldata_cleaned.parquet"
df = None  


def load_data() -> pd.DataFrame:
    """Load and preprocess the dataset."""
    if not DATA_PATH.is_file():
        print(f"ERROR: Data file not found: {DATA_PATH}", file=sys.stderr)
        raise FileNotFoundError(str(DATA_PATH))

    _df = pd.read_parquet(DATA_PATH)
    _df["Date"] = pd.to_datetime(_df["Date"])
    _df["year"] = _df["Date"].dt.year
    _df["month"] = _df["Date"].dt.month
    if "Taxa" in _df.columns:
        _df = _df.rename(columns={"Taxa": "taxa"})
    return _df


try:
    df = load_data()
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
    return query_df


def _get_options(df_source: pd.DataFrame, key_name: str):
    return sorted(df_source[key_name].dropna().unique().tolist())

@app.get("/")
def root():
    return {"status": "ok", "message": "Biodiversity Dashboard API root"}


@app.get("/health")
def health():
    return {"ok": True}


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
):
    query_df = apply_filters(df.copy(), english_name, species, obs, taxa, year, month)
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


@app.get("/api/summary/diversity")
def get_diversity_summary(
    english_name: Optional[str] = None,
    species: Optional[str] = None,
    obs: Optional[str] = None,
    taxa: Optional[str] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
):
    from scipy.stats import entropy

    query_df = apply_filters(df.copy(), english_name, species, obs, taxa, year, month)
    species_counts = query_df.groupby("species")["count"].sum() if not query_df.empty else pd.Series(dtype=float)
    species_richness = len(species_counts)

    if query_df.empty or query_df["count"].sum() == 0 or species_richness <= 1:
        return {"shannon": 0, "simpson": 0, "species_richness": species_richness}

    proportions = species_counts[species_counts > 0] / species_counts.sum()
    shannon_index = entropy(proportions, base=np.e)
    gini_simpson_index = 1 - (proportions**2).sum()

    return {
        "shannon": round(float(shannon_index), 3),
        "simpson": round(float(gini_simpson_index), 3),
        "species_richness": int(species_richness),
    }


@app.get("/api/summary/species_distribution")
def get_species_distribution(
    english_name: Optional[str] = None,
    species: Optional[str] = None,
    obs: Optional[str] = None,
    taxa: Optional[str] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
):
    query_df = apply_filters(df.copy(), english_name, species, obs, taxa, year, month)
    species_counts = query_df["english_name"].value_counts()
    top_20 = species_counts.nlargest(20)
    return top_20.to_dict()


@app.get("/api/summary/temporal_trends")
def get_temporal_trends(
    english_name: Optional[str] = None,
    species: Optional[str] = None,
    obs: Optional[str] = None,
    taxa: Optional[str] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
):
    query_df = apply_filters(df.copy(), english_name, species, obs, taxa, year, month)
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
):
    if not obs:
        return {}
    query_df = apply_filters(df.copy(), english_name, species, taxa=taxa, year=year, month=month)
    query_df = query_df[query_df["obs"].isin(obs.split(","))]
    if query_df.empty:
        return {}
    comparison = query_df.groupby(["obs", "taxa"]).size().unstack(fill_value=0)
    return comparison.to_dict(orient="dict")


@app.get("/api/summary/observer/{observer_name}")
def get_observer_stats(
    observer_name: str,
    english_name: Optional[str] = None,
    species: Optional[str] = None,
    taxa: Optional[str] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
):
    query_df = apply_filters(df.copy(), english_name, species, None, taxa, year, month)
    observer_df = query_df[query_df["obs"] == observer_name]
    if observer_df.empty:
        return {}
    specialization = observer_df.groupby("taxa").size()
    return specialization.to_dict()