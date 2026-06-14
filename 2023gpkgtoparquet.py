import pandas as pd
import geopandas as gpd
import numpy as np
import sys
import os

FINAL_COLUMNS = [
    'TRUE..1', 'Date', 'species', 'Taxa', 'obs', 'height', 'radius',
    'photoid', 'count', 'year', 'month', 'day', 'comment', 'type',
    'english.name', 'longitude', 'latitude', 'english_name'
]

def convert_legacy_gpkg(input_gpkg, output_parquet, species_csv):
    """
    Converts a legacy GPKG file to a standardized Parquet file using
    paths provided as command-line arguments.
    """
    print(f"--- Starting conversion for: {input_gpkg} ---")

    if not os.path.exists(input_gpkg):
        print(f"ERROR: Input file not found at '{input_gpkg}'.")
        sys.exit(1)
    if not os.path.exists(species_csv):
        print(f"ERROR: Species lookup file '{species_csv}' not found.")
        sys.exit(1)

    print(f"Output will be saved as: {output_parquet}")

    try:
        print(f"\nLoading species lookup data from '{species_csv}'...")
        lookup_df = pd.read_csv(species_csv, encoding='latin-1')
        lookup_df = lookup_df[['english name', 'species', 'type']].copy()
        lookup_df.rename(columns={'species': 'scientific_name_from_csv'}, inplace=True)
        print(f"Loaded {len(lookup_df)} entries.")
    except Exception as e:
        print(f"ERROR: Could not read the species CSV file. Details: {e}")
        sys.exit(1)

    try:
        print(f"\nLoading GeoPackage file: '{input_gpkg}'...")
        gdf = gpd.read_file(input_gpkg)
        print(f"Read {len(gdf)} rows.")
    except Exception as e:
        print(f"ERROR: Could not read the GeoPackage file. Details: {e}")
        sys.exit(1)

    print("\nStandardizing data and transforming schema...")
    
    gdf['Date'] = pd.to_datetime(gdf['Date'], errors='coerce').astype('datetime64[ms]')
    print("Standardized 'Date' column to datetime64[ms].")

    rename_map = {
        'fid': 'TRUE..1', 'Observer': 'obs', 'taxa': 'Taxa',
        'lon': 'longitude', 'lat': 'latitude', 'species': 'english_name'
    }
    gdf.rename(columns=rename_map, inplace=True)
    print("Renamed legacy columns.")

    print("\nMapping species data...")
    gdf = pd.merge(
        gdf, lookup_df,
        left_on='english_name', right_on='english name',
        how='left'
    )
    gdf.rename(columns={'scientific_name_from_csv': 'species'}, inplace=True)
    if 'english name' in gdf.columns:
        gdf.drop(columns=['english name'], inplace=True)

    print("\nFinalizing column structure and data types...")
    
    for col in FINAL_COLUMNS:
        if col not in gdf.columns:
            gdf[col] = None
            print(f"Added missing column: '{col}'")

    if 'count' in gdf.columns:
        gdf['count'] = gdf['count'].astype(np.float64)
        print("Standardized 'count' column to float64.")
    if 'height' in gdf.columns:
        gdf['height'] = gdf['height'].astype(object)
        print("Standardized 'height' column to object.")
    if 'radius' in gdf.columns:
        gdf['radius'] = gdf['radius'].astype(object)
        print("Standardized 'radius' column to object.")

    gdf = gdf[FINAL_COLUMNS]
    print("Set final column order.")

    try:
        output_dir = os.path.dirname(output_parquet)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        print(f"\nSaving final data to '{output_parquet}'...")
        gdf.to_parquet(output_parquet, index=False)
        print(f"\n--- Conversion successful! ---")
        print(f"Saved {len(gdf)} rows to '{output_parquet}'.")
    except Exception as e:
        print(f"ERROR: Could not save the final Parquet file. Details: {e}")
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python 2023gpkgtoparquet.py <input_gpkg_path> <output_parquet_path> <species_csv_path>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    csv_path = sys.argv[3]
    
    convert_legacy_gpkg(input_path, output_path, csv_path)