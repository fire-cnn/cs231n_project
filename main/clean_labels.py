""" 
Clean labels from shapefile

This routine reads point data from households and does the following:
    - Build a buffer around each point in meters 
    - Calculates overlaps and remove all labels that have some overlap. The main
      goal of this is to avoid label contamination during training and keep the
      images on their corresponding target.
    - Takes the resulting non-overlapping features and saves them again.
"""

import argparse
import pandas as pd
import geopandas as gpd
from pathlib import Path

from src.utils import remove_overalpping_geometries

parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--file")
parser.add_argument("--buffer_size", default=40, type=int)
parser.add_argument("--save_file")
args = parser.parse_args()


def read_dataset(file, buffer_size, save_file):
    """ Read and process data
    """

    file_suffix = Path(file).suffix
    if ".csv" == file_suffix:
        points = pd.read_csv(file)
        gdf = gpd.GeoDataFrame(
                points,
                geometry = gpd.points_from_xy(points["lon"], points["lat"]),
                crs = 4326
                )
    elif ".shp" == file_suffix:
        gdf = gpd.read_file(file)
    else:
        raise RuntimeError(f"{file} is not a supported format.")

    # Project to meters using 3857 -- not the best, but works well for USA
    # without big distortion
    gdf_meters = gdf.to_crs("EPSG:3857")
    gdf_meters["geometry"] = gdf_meters.buffer(buffer_size, cap_style = 3)

    # Transform back to Mercator
    gdf = gdf_meters.to_crs("EPSG:4326")
    
    # Clean overlaps!
    gdf_clean = remove_overalpping_geometries(gdf, save_file)

if __name__ == "__main__":
    read_dataset(args.file, args.buffer_size, args.save_file)


