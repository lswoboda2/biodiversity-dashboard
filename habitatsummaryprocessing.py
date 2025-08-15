import pandas as pd
import geopandas as gpd
from pathlib import Path
import json
import argparse 

def process_habitat_data(polygons_gpkg_path, squares_gpkg_path, output_json_path):
    """
    Reads habitat and square count GeoPackage files from specified paths, 
    processes them to create a summary, and saves the result as a JSON file.
    """
    print("--- Starting habitat summary data processing ---")
    
    POLYGONS_GPKG = Path(polygons_gpkg_path)
    SQUARES_GPKG = Path(squares_gpkg_path)
    OUTPUT_JSON = Path(output_json_path)

    print(f"Reading polygon data from: {POLYGONS_GPKG}")
    if not POLYGONS_GPKG.is_file():
        print(f"ERROR: Polygon file not found at {POLYGONS_GPKG}")
        exit(1)

    gdf_polygons = gpd.read_file(POLYGONS_GPKG)
    print(f"Successfully read {len(gdf_polygons)} polygon features.")

    gdf_polygons['area_m2'] = gdf_polygons.geometry.area
    gdf_polygons['bscore'] = pd.to_numeric(gdf_polygons['biomscore'], errors='coerce').fillna(0)
    gdf_polygons['bscorearea'] = gdf_polygons['bscore'] * gdf_polygons['area_m2']

    total_area_per_year = gdf_polygons.groupby('year')['area_m2'].sum().reset_index()
    total_area_per_year.rename(columns={'area_m2': 'total_year_area_m2'}, inplace=True)

    poly_summary = gdf_polygons.groupby(['year', 'broad']).agg(
        total_area_m2=('area_m2', 'sum'),
        total_bscorearea=('bscorearea', 'sum')
    ).reset_index()

    poly_summary = pd.merge(poly_summary, total_area_per_year, on='year')
    poly_summary['areaha'] = poly_summary['total_area_m2'] / 10000
    poly_summary['percent_area'] = (poly_summary['total_area_m2'] / poly_summary['total_year_area_m2']) * 100
    poly_summary['biomscore'] = poly_summary['total_bscorearea'] / 1000
    
    print("Polygon summary calculated.")

    print(f"Reading 10m square data from: {SQUARES_GPKG}")
    if not SQUARES_GPKG.is_file():
        print(f"ERROR: 10m square file not found at {SQUARES_GPKG}")
        exit(1)
        
    gdf_squares = gpd.read_file(SQUARES_GPKG)
    df_squares = pd.DataFrame(gdf_squares.drop(columns='geometry'))
    print(f"Successfully read {len(df_squares)} square features.")

    square_summary = df_squares.groupby(['year', 'broad']).size().reset_index(name='no10msquares')
    total_squares_per_year = square_summary.groupby('year')['no10msquares'].sum().reset_index()
    total_squares_per_year.rename(columns={'no10msquares': 'total_year_squares'}, inplace=True)

    square_summary = pd.merge(square_summary, total_squares_per_year, on='year')
    square_summary['percent_squares'] = (square_summary['no10msquares'] / square_summary['total_year_squares']) * 100
    
    print("10m square summary calculated.")

    final_summary = pd.merge(
        poly_summary[['year', 'broad', 'areaha', 'percent_area', 'biomscore']],
        square_summary[['year', 'broad', 'no10msquares', 'percent_squares']],
        on=['year', 'broad'],
        how='outer'
    ).fillna(0)
    
    all_years = sorted(final_summary['year'].unique().tolist())
    all_habitats = sorted(final_summary['broad'].unique().tolist())
    
    output_data = {"years": all_years, "habitats": []}

    totals = {}
    for year in all_years:
        year_data = final_summary[final_summary['year'] == year]
        totals[year] = {
            "areaha": year_data['areaha'].sum(),
            "percent_area": 100.0,
            "biomscore": year_data['biomscore'].sum(),
            "no10msquares": int(year_data['no10msquares'].sum()),
            "percent_squares": 100.0
        }
    output_data["totals"] = totals

    for habitat in all_habitats:
        habitat_entry = {"name": habitat, "metrics": {}}
        for year in all_years:
            record = final_summary[(final_summary['broad'] == habitat) & (final_summary['year'] == year)]
            if not record.empty:
                habitat_entry["metrics"][year] = {
                    "areaha": record.iloc[0]['areaha'],
                    "percent_area": record.iloc[0]['percent_area'],
                    "biomscore": record.iloc[0]['biomscore'],
                    "no10msquares": int(record.iloc[0]['no10msquares']),
                    "percent_squares": record.iloc[0]['percent_squares']
                }
            else:
                habitat_entry["metrics"][year] = {
                    "areaha": 0, "percent_area": 0, "biomscore": 0, 
                    "no10msquares": 0, "percent_squares": 0
                }
        output_data["habitats"].append(habitat_entry)

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    print(f"--- SUCCESS: Processed data and saved to: {OUTPUT_JSON} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processes two habitat GeoPackage files into a single summary JSON file.")
    parser.add_argument("polygons_input", help="The full path to the input Habitat Polygons GeoPackage file.")
    parser.add_argument("squares_input", help="The full path to the input 10m Squares GeoPackage file.")
    parser.add_argument("json_output", help="The full path for the output JSON file.")
    
    args = parser.parse_args()
    
    process_habitat_data(args.polygons_input, args.squares_input, args.json_output)