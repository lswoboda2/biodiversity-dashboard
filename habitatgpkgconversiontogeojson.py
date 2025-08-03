import geopandas as gpd
import sys
import os

def convert_to_geojson(input_gpkg_path, output_geojson_path):
    """
    Reads a GeoPackage file from a specified path, re-projects it to 
    WGS 84 (EPSG:4326), and saves it as a GeoJSON file to a specified path.
    """
    print(f"--- Starting GeoJSON Conversion ---")
    print(f"Input GeoPackage: {input_gpkg_path}")
    print(f"Output GeoJSON: {output_geojson_path}")

    if not os.path.exists(input_gpkg_path):
        print(f"ERROR: Source file not found at {input_gpkg_path}")
        sys.exit(1)

    try:
        gdf = gpd.read_file(input_gpkg_path)
        print(f"Successfully read {len(gdf)} features.")
        print(f"Original CRS: {gdf.crs}")
    except Exception as e:
        print(f"ERROR: Could not read the GeoPackage file. Details: {e}")
        sys.exit(1)

    if gdf.crs != "EPSG:4326":
        print("Reprojecting to EPSG:4326...")
        gdf = gdf.to_crs("EPSG:4326")
        print("Reprojection complete.")

    try:
        output_dir = os.path.dirname(output_geojson_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        gdf.to_file(output_geojson_path, driver='GeoJSON')
        print(f"Successfully converted and saved file to: {output_geojson_path}")
        print(f"--- Conversion Successful ---")
    except Exception as e:
        print(f"ERROR: Could not save the final GeoJSON file. Details: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python habitatgpkgconversiontogeojson.py <input_gpkg_path> <output_geojson_path>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    convert_to_geojson(input_path, output_path)