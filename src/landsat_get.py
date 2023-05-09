from functools import cached_property
import gzip
import os
import shutil
from urllib.parse import urlparse
import warnings

from rasterio.enums import Resampling
import dask.dataframe as dd
import geopandas as gpd
import numpy as np
from osgeo import gdal
import pandas as pd
import planetary_computer
import polars as pl
import pystac
import pystac_client
import requests
from shapely import geometry
from tqdm import tqdm

import adlfs
import stackstac
from utils import intersection_percent


class Landsat:
    """Search and download Landsat imagery

    This class searchs and download Landsat scenes corresponding to overlapping
    geometries and metadata.
    """

    def __init__(
        self,
        aoi_path,
        save_path,
        date_window,
        buffer=10000,
        collection="naip",
        force_update=False,
    ) -> None:
        self.aoi_path: str = aoi_path
        self.save_path: str = save_path
        self.buffer: int = buffer
        self.aoi_date = "Ig_Date"
        self.collection: str = collection

        if not isinstance(date_window, tuple):
            self.date_window: tuple = tuple([date_window, date_window])
        else:
            self.date_window: tuple = date_window

        # Subset bands of interest in Landsat collection
        self.bands: list[str] = ["blue", "green", "red", "nir08", "swir16", "qa_pixel"]

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
                        points[cols],
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

        if hasattr(geometry_obj, "__geo_interface__"):
            geometry_bounds = geometry_obj.bounds
        else:
            geometry_bounds = geometry_obj

        search = self.catalog.search(
            collections=self.collection,
            bbox=geometry_bounds,
            datetime=timerange
        )

        # Make collection
        items_search = search.get_all_items()

        return items_search

    def execute_search_aoi(self):
        """Execute search in all the elements of the AOI"""

        dict_aoi: list = self.aoi.to_dict(orient="records")

        for aoi_element in tqdm(dict_aoi, desc="Downloading from PC..."):

            # Create path to later check for file existence
            path_to_save = os.path.join(
                self.save_path, f"{aoi_element['Event_ID']}.nc4"
            )

            if not os.path.exists(path_to_save):
                try:
                    items_aoi = self.request_items_stac(
                        start_date=aoi_element["pre_date"],
                        end_date=aoi_element["post_date"],
                        geometry_obj=aoi_element["geometry"],
                    )

                    if len(items_aoi) > 0:

                        data = stackstac.stack(
                            items_aoi,
                            bounds=aoi_element["geometry"].bounds,
                            assets=self.bands,
                            epsg=4326,
                            resampling=Resampling.bilinear,
                            snap_bounds=True,
                        )

                        data.attrs = {}
                        data = data.drop(
                            [
                                "instruments",
                                "raster:bands",
                                "center_wavelength",
                                "proj:epsg",
                                "gsd",
                                "landsat:collection_category",
                                "landsat:collection_number",
                                "landsat:correction",
                                "landsat:wrs_type",
                                "description",
                                "view:off_nadir",
                                "landsat:wrs_path",
                                "landsat:wrs_row",
                                "sci:doi",
                                "classification:bitfields",
                            ]
                        )

                        data.to_netcdf(path_to_save)
                    else:
                        print(f"{aoi_element['Event_ID']} has no items!")

                except RuntimeError as e:
                    print(f"{aoi_element['Event_ID']} failed with: {e}")
                    pass

                except ValueError as e:
                    print(f"{aoi_element['Event_ID']} failed with: {e}")
                    pass


        return None
