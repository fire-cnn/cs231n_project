from typing import Any, Dict

import geopandas as gpd
import pystac
from shapely import geometry
from tqdm import tqdm


def remove_overalpping_geometries(df: gpd.GeoDataFrame, save_path: str = None):
    """Remove all overlapping geometries from GeoDataFrame

    This function removes all overlapping polygons within dataframe. The use
    case of this function is to avoid label contamination. If we make sure
    that the images are indenpendent, we are making sure we will have good
    learning.

    Be careful that the process does double iteration over the dataframe. Thus,
    we have in worst-case scenario a O(n * n), so it can take a while for big
    files.

    Args:
        - df: A geopandas dataframe with unique polygons
        - save_path: A path to save the cleaned dataframe
    """

    # Check for polygons
    if "Polygon" != df.geom_type.unique()[0]:
        raise RuntimeError("Dataframe do not contain unique polygons.")

    non_overlap = []
    for idx, row in tqdm(
        df.iterrows(), total=df.shape[0], desc="Removing overlapping geoms"
    ):
        if not any(row["geometry"].overlaps(g) for g in df.geometry):
            non_overlap.append(row)

    # Concat to gpd again
    df_clean = gpd.GeoDataFrame(non_overlap)

    if save_path:
        df_clean.to_file(save_path)

    return df_clean


def intersection_percent(item: pystac.Item, aoi: Dict[str, Any]) -> float:
    """The percentage that the Item's geometry intersects the AOI. An Item that
    completely covers the AOI has a value of 100.
    """
    geom_item = geometry.shape(item.geometry)
    geom_aoi = geometry.shape(aoi)

    intersected_geom = geom_aoi.intersection(geom_item)

    intersection_percent = (intersected_geom.area * 100) / geom_aoi.area

    return intersection_percent
