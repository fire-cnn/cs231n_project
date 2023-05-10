from functools import cached_property
import gzip
import os
import shutil
from urllib.parse import urlparse
import warnings
import rioxarray
import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer
import pystac
import pystac_client
import requests
from shapely import geometry
from tqdm import tqdm


import pdb

class NAIP:
    """Search and download NAIP imagery

    This class searchs and download NAIP scenes corresponding to overlapping
    geometries and metadata.
    """

    def __init__(
        self,
        aoi_path,
        save_path,
        date_window,
        collection="naip",
        force_update=False,
    ) -> None:
        self.aoi_path: str = aoi_path
        self.save_path: str = save_path
        self.aoi_date = "fire_start"
        self.id = "homeid"
        self.collection: str = collection

        if not isinstance(date_window, tuple):
            self.date_window: tuple = tuple([date_window, date_window])
        else:
            self.date_window: tuple = date_window

        # Subset bands of interest in Landsat collection
        self.bands: list[str] = ["blue", "green", "red"]

        # Create save directory if not present
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    @cached_property
    def catalog(self) -> pystac_client.Client:
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1/",
            modifier=planetary_computer.sign_inplace,
        )

        return catalog

    @cached_property
    def aoi(self) -> gpd.GeoDataFrame:
        """Property store for area of interests"""

        if isinstance(self.aoi_path, str):
            file_suffix = Path(self.aoi_path).suffix
            if ".csv" == file_suffix:
                aoi: gdp.GeoDataFrame = gpd.GeoDataFrame(
                        points,
                        geometry=gpd.points_from_xy(points.lon, points.lat),
                        crs = "4326"
                        )
            elif ".shp" == file_suffix:
                aoi: gpd.GeoDataFrame = gpd.read_file(self.aoi_path)
            else:
                raise ValueError(f"{file_suffix} is not supported.")

        elif isinstance(self.aoi_path, gpd.GeoDataFrame):
            aoi: gpd.GeoDataFrame = self.aoi_path
        else:
            raise RuntimeError(f"{self.aoi_path} is not a valid format.")

        # Check projection and reproject to vanilla mercator
        if aoi.crs.to_epsg() != "4326":
            aoi: gpd.GeoDataFrame = aoi.to_crs("EPSG:4326")

        # Create time search variables
        aoi[self.aoi_date] = pd.to_datetime(aoi[self.aoi_date])

        left, right = self.date_window

        aoi = aoi.assign(
            pre_date=aoi[self.aoi_date] - pd.Timedelta(days=left),
            post_date=aoi[self.aoi_date] + pd.Timedelta(days=right),
        )

        return aoi

    def request_items_stac(self, start_date, end_date, geometry_obj):
        """Request landsat metadata using the PC catalog seach

        Args:
            - start_date pd.DateTime: Start time for search
            - end_date pd.DateTime: End of time for search
            - geometry_obj dict: A GeoJSON object with the bounding box of each
              AOI.

        Returns:
            pystac.Collection
        """

        # Build datetime string
        start_time_str: str = start_date.strftime("%Y-%m-%d")
        end_time_str: str = end_date.strftime("%Y-%m-%d")
        timerange: str = f"{start_time_str}/{end_time_str}"

        search = self.catalog.search(
            collections=self.collection,
            intersects=geometry_obj,
            datetime=timerange)

        # Make collection
        items_search = search.get_all_items()

        return items_search

    def execute_search_aoi(self):
        """Execute search in all the elements of the AOI"""

        dict_aoi: list = self.aoi.to_dict(orient="records")

        for aoi_element in tqdm(dict_aoi, desc="Downloading from PC..."):
            try:
                items_aoi = self.request_items_stac(
                    start_date=aoi_element["pre_date"],
                    end_date=aoi_element["post_date"],
                    geometry_obj=aoi_element["geometry"],
                )

                if len(items_aoi) > 0:

                    for item in items_aoi:
                        asset_url = item.assets["image"].href
                        ds = (
                                rioxarray.open_rasterio(asset_url)
                                .sel(band=[1, 2, 3])
                            )

                        minx, miny, maxx, maxy = aoi_element["geometry"].bounds
                        clipped = ds.rio.clip_box(minx, miny, maxx, maxy, 
                                                  crs = 4326)
                        
                        # Save file using homeid and naip id
                        name = f"{aoi_element[self.id]}_{item.id}.png"
                        save_path = os.path.join(self.save_path, name)
                        clipped.rio.to_raster(save_path, driver="PNG")

                else:
                    print(f"{aoi_element[self.id]} has no items!")

            except RuntimeError as e:
                print(f"{aoi_element[self.id]} failed with: {e}")
                pass

            except ValueError as e:
                print(f"{aoi_element[self.id]} failed with: {e}")
                pass

        return None
