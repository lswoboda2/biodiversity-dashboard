import pandas as pd
import geopandas as gpd
import numpy as np
import requests
import time
import re
import os
import sys
import fiona

FINAL_COLUMNS = [
    'TRUE..1', 'Date', 'species', 'Taxa', 'obs', 'height', 'radius',
    'photoid', 'count', 'year', 'month', 'day', 'comment', 'type',
    'english.name', 'longitude', 'latitude', 'english_name'
]

def find_best_layer(gpkg_path: str) -> str | None:
    print(f"\nInspecting layers in: {gpkg_path}")
    try:
        layer_names = fiona.listlayers(gpkg_path)
    except Exception as e:
        print(f"ERROR: Could not open or list layers in the file. Details: {e}")
        return None
    if not layer_names: return None
    layer_counts = {}
    for name in layer_names:
        try:
            with fiona.open(gpkg_path, layer=name) as layer:
                layer_counts[name] = len(layer)
        except fiona.errors.FionaError:
            print(f"Warning: Could not read layer '{name}'. It might not be a feature layer.")
            continue
    if not layer_counts:
        print("ERROR: No readable feature layers found in the file.")
        return None
    best_layer = max(layer_counts, key=layer_counts.get)
    print(f"Layer feature counts: {layer_counts}. Selected '{best_layer}'.")
    return best_layer

def get_name_from_itis(species_name: str) -> str | None:
    try:
        url = "https://www.itis.gov/ITISWebService/jsonservice/searchForAnyMatch"
        params = {'srchKey': species_name, 'searchType': 'exact'}
        time.sleep(0.5)
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json().get('anyMatchList', [{}])[0]
        if data and data.get('tsn') and data.get('commonNameList'):
            for name_info in data['commonNameList']['commonNames']:
                if name_info.get('language') == 'English':
                    return name_info.get('commonName').capitalize()
    except (requests.exceptions.RequestException, IndexError): return None
    return None

def get_name_from_ncbi(species_name: str) -> str | None:
    try:
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {'db': 'taxonomy', 'term': species_name, 'retmode': 'json'}
        time.sleep(0.5)
        search_response = requests.get(search_url, params=search_params, timeout=10)
        search_response.raise_for_status()
        id_list = search_response.json().get('esearchresult', {}).get('idlist', [])
        if not id_list: return None
        summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        summary_params = {'db': 'taxonomy', 'id': id_list[0], 'retmode': 'json'}
        summary_response = requests.get(summary_url, params=summary_params, timeout=10)
        summary_response.raise_for_status()
        result = summary_response.json().get('result', {}).get(id_list[0], {})
        if result and result.get('commonname'):
            return result['commonname'].capitalize()
    except (requests.exceptions.RequestException, IndexError): return None
    return None

def get_name_from_gbif(species_name: str) -> str | None:
    try:
        match_url = "https://api.gbif.org/v1/species/match"
        params = {'name': species_name, 'strict': 'false'}
        time.sleep(0.5)
        match_response = requests.get(match_url, params=params, timeout=10)
        match_response.raise_for_status()
        match_data = match_response.json()
        if match_data.get('matchType') != 'NONE' and 'usageKey' in match_data:
            species_key = match_data['usageKey']
            vernacular_url = f"https://api.gbif.org/v1/species/{species_key}/vernacularNames"
            vernacular_response = requests.get(vernacular_url, timeout=10)
            vernacular_response.raise_for_status()
            for name_info in vernacular_response.json().get('results', []):
                if name_info.get('language', '').lower() == 'eng':
                    return name_info.get('vernacularName').capitalize()
    except requests.exceptions.RequestException: return None
    return None

def get_best_english_name(species_name: str) -> str | None:
    if not species_name or pd.isna(species_name): return None
    cleaned_name = species_name.replace('_', ' ').strip()
    cleaned_name = re.sub(r'\s+sp\.?$', '', cleaned_name, flags=re.IGNORECASE)
    name = get_name_from_itis(cleaned_name)
    if name: return name
    name = get_name_from_ncbi(cleaned_name)
    if name: return name
    name = get_name_from_gbif(cleaned_name)
    if name: return name
    return None

def process_intern_data(input_gpkg, output_parquet, species_csv, api_cache_csv):
    """Main function to execute the full data processing pipeline."""
    
    if not os.path.exists(input_gpkg):
        print(f"ERROR: Input file not found at '{input_gpkg}'.")
        sys.exit(1)
        
    layer_to_load = find_best_layer(input_gpkg)
    if layer_to_load is None:
        print("Exiting: No suitable layer found.")
        sys.exit(1)

    print("\nLoading GeoPackage Data")
    gdf = gpd.read_file(input_gpkg, layer=layer_to_load)
    print(f"Read {len(gdf)} rows from layer '{layer_to_load}'.")

    print("\nApplying 'Intern' legacy transformations...")
    gdf.index.name = 'TRUE..1'
    gdf.reset_index(inplace=True)
    cols_to_drop = ['taxa1', 'native', 'msp']
    gdf.drop(columns=[col for col in cols_to_drop if col in gdf.columns], inplace=True, errors='ignore')
    rename_map = {'Observer': 'obs', 'Count': 'count'}
    gdf.rename(columns=rename_map, inplace=True)
    gdf['Date'] = pd.to_datetime(gdf['Date'], errors='coerce')
    gdf.dropna(subset=['Date'], inplace=True)
    if gdf['Date'].dt.tz is not None:
        gdf['Date'] = gdf['Date'].dt.tz_convert('UTC').dt.tz_localize(None)
    if 'geometry' in gdf.columns:
        gdf['longitude'] = gdf.geometry.x
        gdf['latitude'] = gdf.geometry.y
        gdf = gdf.drop(columns='geometry')

    print("\nLoading Species Name Lookups (Local & Cache)...")
    final_species_map = {}
    if os.path.exists(species_csv):
        species_df = pd.read_csv(species_csv, encoding='latin-1')
        final_species_map.update(pd.Series(species_df.english_name.values, index=species_df.species).to_dict())
    if os.path.exists(api_cache_csv):
        cache_df = pd.read_csv(api_cache_csv, encoding='latin-1')
        final_species_map.update(pd.Series(cache_df.english_name.values, index=cache_df.species).to_dict())

    all_data_species = set(gdf['species'].dropna().unique())
    species_to_lookup = sorted([s for s in all_data_species if s not in final_species_map])
    
    newly_cached_species = {}
    if species_to_lookup:
        print(f"Found {len(species_to_lookup)} new species to look up via API...")
        for i, species in enumerate(species_to_lookup):
            print(f"  ({i+1}/{len(species_to_lookup)}) Looking up: {species}...", end='')
            english_name = get_best_english_name(species)
            if english_name:
                print(f" Found: '{english_name}'")
                final_species_map[species] = english_name
                newly_cached_species[species] = english_name
            else:
                print(" Not found.")
    else:
        print("No new API lookups were required.")
    
    if newly_cached_species:
        print(f"\nUpdating API cache file at '{api_cache_csv}' with {len(newly_cached_species)} new entries.")
        new_cache_df = pd.DataFrame(list(newly_cached_species.items()), columns=['species', 'english_name'])
        new_cache_df.to_csv(api_cache_csv, mode='a', header=not os.path.exists(api_cache_csv), index=False)

    gdf['english_name'] = gdf['species'].map(final_species_map)
    print("Applied final species map to create 'english_name' column.")

    print("\nFinalizing schema and data types...")
    gdf['english.name'] = None
    for col in ['height', 'radius', 'photoid', 'comment']:
        if col not in gdf.columns: gdf[col] = None
    type_map = {'TRUE..1': 'object', 'Date': 'datetime64[ms]', 'height': 'object', 'radius': 'object', 'photoid': 'object', 'count': 'float64'}
    for col, dtype in type_map.items():
        if col in gdf.columns:
            try:
                if dtype == 'object': gdf[col] = gdf[col].astype(str).where(gdf[col].notna(), None)
                else: gdf[col] = gdf[col].astype(dtype)
            except Exception as e:
                print(f"Warning: Could not set type for '{col}'. Details: {e}")
            
    gdf = gdf[FINAL_COLUMNS]
    print("Final column order enforced.")

    print("\nSaving Final Output")
    try:
        output_dir = os.path.dirname(output_parquet)
        if output_dir: os.makedirs(output_dir, exist_ok=True)
        gdf.to_parquet(output_parquet, index=False)
        print(f"Saved data to '{output_parquet}' with {len(gdf)} rows.")
    except Exception as e:
        print(f"ERROR: Could not save the final Parquet file. Details: {e}")

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: python interngpkgtoparquet.py <input_gpkg> <output_parquet> <species_csv> <api_cache_csv>")
        sys.exit(1)
    
    process_intern_data(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])