import geopandas as gpd
from pathlib import Path
import argparse 

def convert_cameratraps_to_geojson(input_path, output_path):
    """
    Reads a Camera Traps GeoPackage file from a specified input path and converts it
    to a GeoJSON file at the specified output path.
    """
    print(f"--- Starting Camera Traps GeoJSON Conversion ---")
    print(f"Input GeoPackage: {input_path}")
    print(f"Output GeoJSON: {output_path}")

    input_file = Path(input_path)
    if not input_file.is_file():
        print(f"ERROR: Input file not found at '{input_path}'.")
        exit(1)

    try:
        gdf = gpd.read_file(input_file)
        print(f"Successfully read {len(gdf)} features from the GeoPackage.")
        
        print("Converting CRS to WGS84 (EPSG:4326) for web compatibility...")
        gdf = gdf.to_crs(epsg=4326)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        gdf.to_file(output_file, driver='GeoJSON')
        
        print(f"\nSUCCESS: Successfully converted the file.")
        print(f"Output saved to: {output_file}")
        print(f"--- Conversion Successful ---")

    except Exception as e:
        print(f"\nAn error occurred during the conversion process: {e}")
        exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a Camera Traps GeoPackage (.gpkg) file to a GeoJSON file.")
    
    parser.add_argument("input_gpkg", help="The full path to the input GeoPackage file.")
    parser.add_argument("output_geojson", help="The full path for the output GeoJSON file.")
    
    args = parser.parse_args()
    
    convert_cameratraps_to_geojson(args.input_gpkg, args.output_geojson)