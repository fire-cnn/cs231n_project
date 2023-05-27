""" Download gridMET data from the PC
"""

import pystac_client
import planetary_computer
import xarray as xr
import rioxarray
import numpy as np
import pandas as pd
import geopandas as gpd
from affine import Affine

from dask.distributed import Client


def get_gridmet(start_date, end_date, geometry, save_path):
    """ """

    client = Client()

    # Get request from the PC
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    asset = catalog.get_collection("gridmet").assets["zarr-abfs"]

    # Load geometry
    if isinstance(geometry, gpd.GeoDataFrame):
        geom = geometry
    elif isinstance(geometry, str):
        geom = gpd.read_file(geometry)
    else:
        NotImplementedError(f"{geometry} is not a valid format")

    # Open zarr files remotely
    ds = xr.open_zarr(
        asset.href,
        storage_options=asset.extra_fields["xarray:storage_options"],
        **asset.extra_fields["xarray:open_kwargs"],
    )

    # Build affine matrix to attrs!
    ds.attrs[
        "GeoTransform"
    ] = f'{ds.attrs["geospatial_lat_max"]} \
            {ds.attrs["geospatial_lat_resolution"]} \
            0 \
            {ds.attrs["geospatial_lon_min"]} \
            0 \
            {ds.attrs["geospatial_lon_resolution"]}'

    # Build affine
    transform = Affine.from_gdal(*np.fromstring(ds.attrs["GeoTransform"], sep=" "))
    ds.rio.write_transform(transform, inplace=True)

    # Build spatial coordinate inplace
    (
        ds.rio.write_crs(4326, inplace=True)
        .rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
        .rio.write_coordinate_system(inplace=True)
    )

    # Filter data
    ds_filter = ds.sel(time=slice(start_date, end_date))
    ds_geom = ds_filter.rio.clip(
        geom.geometry.values, crs=4326, drop=True, invert=False
    )

    # Stupid hack to avoid errors during saving
    vars_list = list(ds_cali.data_vars)
    for var in vars_list:
        del ds_cali[var].attrs["grid_mapping"]

    ds_cali.to_netcdf(save_path)

    return None
