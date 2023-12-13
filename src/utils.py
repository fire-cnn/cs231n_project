import os
from typing import Any, Dict
import pandas as pd
import geopandas as gpd
import pystac
from pathlib import Path
from shapely import geometry
from tqdm import tqdm
from .config import Config
from .prompts import prompting


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


def create_dataset_clip(config_path, save_path):
    """Create a CSV file formatted for CLIP model

    This function creates a CVS file formatted for CLIP modeling. The resulting
    CSV file has three columns:
        - image_column: path to image
        - caption_column: text prompt
        - label: label of image
    These are the default names for the DualEncoder model in HF. To build this
    CSV the function will use the config file passed as argument. The config file
    should contain a prompting configuration via config.prompt_config and a valid
    path to save the files

    Args:
        - save_path: path to save the CSV files
        - config_path: path to config file  (see config.py)

    Returns:
        - None
    """

    # Open configuration
    config = Config(config_path)
    train_config = config.train_config
    prompt_config = config.prompt_config

    # Extract subset name from prompt config. If not exists, use default
    config_name = prompt_config.get("subset", "subset")

    # Pop subset name from config if exists
    if "subset" in prompt_config:
        prompt_config.pop("subset")

    # Load data from path
    text_data = pd.read_csv(train_config["tabular_data_path"])

    # Remove any row with missing values
    text_data = text_data.dropna()

    # Select the data on each folder
    train_paths = list(Path(train_config["path_to_train"]).rglob("*.png"))
    test_paths = list(Path(train_config["path_to_test"]).rglob("*.png"))

    # Get the ids
    ids_train = [int(p.stem.split("_")[0]) for p in train_paths]
    ids_test = [int(p.stem.split("_")[0]) for p in test_paths]

    # Filter data using ids
    text_data_train = text_data[text_data["homeid"].isin(ids_train)]
    text_data_test = text_data[text_data["homeid"].isin(ids_test)]

    # Create prompts
    prompts_test = prompting(df=text_data_test, special_tokens=False, **prompt_config)
    prompts_train = prompting(df=text_data_train, special_tokens=False, **prompt_config)

    # Create dataframes with prompts and paths to images looping through images
    df_train, df_test = [], []
    for path in train_paths:
        id_img = int(path.stem.split("_")[0])

        try:
            df_train.append(
                {
                    "image_column": path,
                    "caption_column": prompts_train[id_img],
                    "label": path.stem.split("_")[-1],
                }
            )
        except KeyError:
            print(f"Key {id_img} not found in prompts during train clean")

    for path in test_paths:
        id_img = int(path.stem.split("_")[0])

        try:
            df_test.append(
                {
                    "image_column": path,
                    "caption_column": prompts_test[id_img],
                    "label": path.stem.split("_")[-1],
                }
            )
        except KeyError:
            print(f"Key {id_img} not found in prompts during test clean")

    # Concat dict lists to dataframe
    df_train = pd.DataFrame(df_train)
    df_test = pd.DataFrame(df_test)

    # Save dataframes in save folder if exists (otherwise create)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    df_train.to_csv(
        os.path.join(save_path, f"clip_train_{config_name}.csv"), index=False
    )
    df_test.to_csv(os.path.join(save_path, f"clip_test_{config_name}.csv"), index=False)

    return None
