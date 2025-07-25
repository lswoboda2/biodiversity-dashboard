import pandas as pd
import geopandas as gpd
import numpy as np
import requests
import time
import re
import os
import sys
import fiona

if len(sys.argv) > 1:
    INPUT_GPKG_PATH = sys.argv[1]
else:
    INPUT_GPKG_PATH = 'default_path.gpkg'
    print("Error: Please provide the path to the .gpkg file.")
    sys.exit(1)

SPECIES_CSV_PATH = 'species list.csv'
API_CACHE_PATH = 'species_api_cache.csv'
OUTPUT_PARQUET_PATH = 'data/data.parquet'

def find_best_layer(gpkg_path: str) -> str | None:
    """
    Inspects a GeoPackage file and returns the name of the layer with the most features.
    """
    print(f"\nInspecting layers in: {gpkg_path}")
    try:
        layer_names = fiona.listlayers(gpkg_path)
    except Exception as e:
        print(f"ERROR: Could not open or list layers in the file. It may be corrupt or invalid.")
        print(f"   Details: {e}")
        return None

    if not layer_names:
        print("WARNING: No layers found in the GeoPackage file.")
        return None

    layer_counts = {}
    for name in layer_names:
        try:
            with fiona.open(gpkg_path, layer=name) as layer:
                count = len(layer)
                layer_counts[name] = count
        except Exception as e:
            print(f"WARNING: Could not read feature count for layer '{name}'. Skipping. Details: {e}")
            layer_counts[name] = 0

    if not layer_counts or max(layer_counts.values()) == 0:
        print("WARNING: Found layers, but none contain any data.")
        return None

    best_layer = max(layer_counts, key=layer_counts.get)
    print(f"Layer feature counts: {layer_counts}. Selected '{best_layer}' as the target layer.")
    return best_layer

def get_name_from_itis(species_name: str) -> str | None:
    """Looks up a name from the Integrated Taxonomic Information System (ITIS)."""
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
    except (requests.exceptions.RequestException, IndexError):
        return None
    return None

def get_name_from_ncbi(species_name: str) -> str | None:
    """Looks up a name from the National Center for Biotechnology Information (NCBI)."""
    try:
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {'db': 'taxonomy', 'term': species_name, 'retmode': 'json'}
        time.sleep(0.5)
        search_response = requests.get(search_url, params=search_params, timeout=10)
        search_response.raise_for_status()
        id_list = search_response.json().get('esearchresult', {}).get('idlist', [])

        if not id_list:
            return None

        summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        summary_params = {'db': 'taxonomy', 'id': id_list[0], 'retmode': 'json'}
        summary_response = requests.get(summary_url, params=summary_params, timeout=10)
        summary_response.raise_for_status()
        result = summary_response.json().get('result', {}).get(id_list[0], {})

        if result and result.get('commonname'):
            return result['commonname'].capitalize()
    except (requests.exceptions.RequestException, IndexError):
        return None
    return None

def get_name_from_gbif(species_name: str) -> str | None:
    """Looks up a name from the Global Biodiversity Information Facility (GBIF)."""
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
    except requests.exceptions.RequestException:
        return None
    return None

def get_best_english_name(species_name: str) -> str | None:
    """
    Queries APIs in a waterfall sequence (ITIS -> NCBI -> GBIF) to find the best
    available English common name.
    """
    if not species_name or pd.isna(species_name):
        return None

    cleaned_name = species_name.replace('_', ' ').strip()
    cleaned_name = re.sub(r'\s+sp\.?$', '', cleaned_name, flags=re.IGNORECASE)

    name = get_name_from_itis(cleaned_name)
    if name:
        return name

    name = get_name_from_ncbi(cleaned_name)
    if name:
        return name

    name = get_name_from_gbif(cleaned_name)
    if name:
        return name

    return None

def main():
    """Main function to execute the full data processing pipeline."""
    layer_to_load = find_best_layer(INPUT_GPKG_PATH)
    if layer_to_load is None:
        print("Stopping process as no suitable data layer could be found in the GPKG file.")
        return

    print("\nLoading and Cleaning GeoPackage Data")
    try:
        gdf = gpd.read_file(INPUT_GPKG_PATH, layer=layer_to_load)
        print(f"Read {len(gdf)} rows from layer '{layer_to_load}'.")
    except Exception as e:
        print(f"ERROR: Could not read the GeoPackage file. Details: {e}")
        return

    gdf['Date'] = pd.to_datetime(gdf['Date'], errors='coerce')
    original_rows = len(gdf)
    gdf.dropna(subset=['Date'], inplace=True)
    if not gdf.empty:
        gdf = gdf[gdf['Date'].dt.year != 1970].copy()
    print(f"Removed {original_rows - len(gdf)} rows with invalid dates.")

    if 'geometry' in gdf.columns:
        gdf['longitude'] = gdf.geometry.x
        gdf['latitude'] = gdf.geometry.y
        gdf = gdf.drop(columns='geometry')
        print("Extracted longitude and latitude from geometry.")

    for col in gdf.select_dtypes(include=['object']).columns:
        if gdf[col].apply(lambda x: isinstance(x, bytes)).any():
            gdf[col] = gdf[col].apply(lambda cell: cell.decode('latin-1', 'replace') if isinstance(cell, bytes) else cell)
    print("Cleaned text columns.")

    numeric_cols = gdf.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        gdf[col] = gdf[col].replace([np.inf, -np.inf], np.nan)
        gdf[col] = gdf[col].astype(object).where(gdf[col].notna(), None)
    print("Cleaned numerical columns.")

    print("\nAdding English Species Names")

    final_species_map = {}
    try:
        if os.path.exists(SPECIES_CSV_PATH):
            species_df = pd.read_csv(SPECIES_CSV_PATH, encoding='latin-1')
            species_df.dropna(subset=['species'], inplace=True)
            final_species_map = pd.Series(species_df.english_name.values, index=species_df.species).to_dict()
            print(f"Loaded {len(final_species_map)} entries from manual list '{SPECIES_CSV_PATH}'.")
        else:
            print(f"Warning: Manual species file not found at '{SPECIES_CSV_PATH}'.")
    except Exception as e:
        print(f"ERROR: Could not read the species CSV file. Details: {e}")
        return

    try:
        if os.path.exists(API_CACHE_PATH):
            cache_df = pd.read_csv(API_CACHE_PATH, encoding='latin-1')
            cache_df.dropna(subset=['species'], inplace=True)
            cache_map = pd.Series(cache_df.english_name.values, index=cache_df.species).to_dict()
            
            original_map_size = len(final_species_map)
            for species, name in cache_map.items():
                if species not in final_species_map:
                    final_species_map[species] = name
            print(f"Loaded {len(final_species_map) - original_map_size} new entries from cache '{API_CACHE_PATH}'.")
    except Exception as e:
        print(f"Warning: Could not read API cache file. It might be created on this run. Details: {e}")

    if 'species' not in gdf.columns:
        print(f"ERROR: The main data is missing the required 'species' column.")
        return

    all_data_species = set(gdf['species'].dropna().unique())
    
    species_to_lookup = [s for s in all_data_species if s not in final_species_map or pd.isna(final_species_map.get(s))]
    
    newly_cached_entries = []
    if species_to_lookup:
        print(f"Found {len(species_to_lookup)} species requiring API lookup.")
        for i, species_name in enumerate(species_to_lookup, 1):
            print(f"Querying for '{species_name}' ({i}/{len(species_to_lookup)})...", end="")
            english_name = get_best_english_name(species_name)
            if english_name:
                print(f" Found: '{english_name}'")
                final_species_map[species_name] = english_name
                newly_cached_entries.append({'species': species_name, 'english_name': english_name})
            else:
                print(" Not found.") 
        print("API lookup complete.")
    else:
        print("No new API lookups were required. All species found in local lists.")

    gdf['english_name'] = gdf['species'].map(final_species_map)
    print("Applied final species map to create 'english_name' column.")

    unmapped_species = [s for s in all_data_species if pd.isna(final_species_map.get(s))]
    if unmapped_species:
        print(f"\nWarning: {len(unmapped_species)} species could not be mapped to an English name.")
    else:
        print("All species were successfully mapped to an English name.")

    print("\nSaving Final Output")
    try:
        output_dir = os.path.dirname(OUTPUT_PARQUET_PATH)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        gdf.to_parquet(OUTPUT_PARQUET_PATH, index=False)
        print(f"Saved data to '{OUTPUT_PARQUET_PATH}' with {len(gdf)} rows.")
    except Exception as e:
        print(f"ERROR: Could not save the final Parquet file. Details: {e}")

    if newly_cached_entries:
        print(f"Appending {len(newly_cached_entries)} new entries to API cache...")
        new_cache_df = pd.DataFrame(newly_cached_entries)
        try:
            new_cache_df.to_csv(
                API_CACHE_PATH, 
                mode='a', 
                index=False, 
                header=not os.path.exists(API_CACHE_PATH)
            )
            print(f"Successfully updated '{API_CACHE_PATH}'.")
        except Exception as e:
            print(f"ERROR: Could not write to cache file '{API_CACHE_PATH}'. Details: {e}")

if __name__ == '__main__':
    main()