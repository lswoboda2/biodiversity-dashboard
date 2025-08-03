import geopandas as gpd
import pandas as pd
import numpy as np
import json
import sys
import os
from pathlib import Path
from shapely.geometry import Polygon

TARGET_CRS = "EPSG:27700"

def perform_grid_analysis(gdf: gpd.GeoDataFrame, grid_size: int = 10) -> int:
    """Performs a grid-based analysis to count unique 10m squares."""
    print(f"\tPerforming {grid_size}m grid analysis...")
    if gdf.empty:
        print("\tInput is empty, returning 0 squares.")
        return 0
    xmin, ymin, xmax, ymax = gdf.total_bounds
    x_coords = np.arange(np.floor(xmin), np.ceil(xmax) + grid_size, grid_size)
    y_coords = np.arange(np.floor(ymin), np.ceil(ymax) + grid_size, grid_size)
    grid_cells = [Polygon([(x, y), (x + grid_size, y), (x + grid_size, y + grid_size), (x, y + grid_size)]) for x in x_coords for y in y_coords]
    grid_gdf = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=TARGET_CRS)
    joined_gdf = gpd.sjoin(grid_gdf, gdf, how="inner", predicate="intersects")
    count = joined_gdf.index.nunique()
    print(f"\tFound {count} unique {grid_size}m squares.")
    return count

def calculate_special_metrics(meadows_file, woodland_file, hedgerows_file, output_json_file):
    """Calculates statistics for the three habitat layers and saves them to a JSON file."""
    print("--- Starting Special Habitat Layer Analysis ---")
    
    for f in [meadows_file, woodland_file, hedgerows_file]:
        if not os.path.exists(f):
            print(f"ERROR: Source file not found at {f}.")
            sys.exit(1)

    metrics = {}

    print("\nProcessing Meadows...")
    meadows_gdf = gpd.read_file(meadows_file)
    meadows_gdf_proj = meadows_gdf.to_crs(TARGET_CRS)
    meadows_area_ha = meadows_gdf_proj.geometry.area.sum() / 10000
    meadows_sq_count = perform_grid_analysis(meadows_gdf_proj)
    metrics["Meadows"] = {"area_ha": round(meadows_area_ha, 2), "no_10m_sq": meadows_sq_count}
    print(f"Meadows Complete: Area={metrics['Meadows']['area_ha']}ha, Squares={metrics['Meadows']['no_10m_sq']}")

    print("\nProcessing Woodland & Copses...")
    woodland_gdf = gpd.read_file(woodland_file)
    woodland_gdf_proj = woodland_gdf.to_crs(TARGET_CRS)
    woodland_area_ha = woodland_gdf_proj.geometry.area.sum() / 10000
    woodland_sq_count = perform_grid_analysis(woodland_gdf_proj)
    metrics["Woodland & Copses"] = {"area_ha": round(woodland_area_ha, 2), "no_10m_sq": woodland_sq_count}
    print(f"Woodland & Copses Complete: Area={metrics['Woodland & Copses']['area_ha']}ha, Squares={metrics['Woodland & Copses']['no_10m_sq']}")

    print("\nProcessing Hedgerows...")
    hedgerows_gdf = gpd.read_file(hedgerows_file)
    hedgerows_gdf_proj = hedgerows_gdf.to_crs(TARGET_CRS)
    hedgerows_buffered_gdf = hedgerows_gdf_proj.copy()
    hedgerows_buffered_gdf['geometry'] = hedgerows_gdf_proj.geometry.buffer(1)
    hedgerows_area_ha = hedgerows_buffered_gdf.geometry.area.sum() / 10000
    hedgerows_sq_count = perform_grid_analysis(hedgerows_buffered_gdf)
    metrics["Hedgerows"] = {"area_ha": round(hedgerows_area_ha, 2), "no_10m_sq": hedgerows_sq_count}
    print(f"Hedgerows Complete: Area={metrics['Hedgerows']['area_ha']}ha, Squares={metrics['Hedgerows']['no_10m_sq']}")

    output_dir = os.path.dirname(output_json_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_json_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nSuccessfully saved special metrics to: {output_json_file}")
    print("--- Analysis Finished ---")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python calculate3gpkgmetrics.py <meadows_gpkg> <woodland_gpkg> <hedgerows_gpkg> <output_json>")
        sys.exit(1)
    
    calculate_special_metrics(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])