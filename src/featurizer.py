import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image


class Featurizer:
    def __init__(self, path_to_images, path_to_save, id_var):
        self.path_to_images = path_to_images
        self.path_to_save = path_to_save
        self.id_var = id_var

        self.path_list = list(Path(self.path_to_images).rglob("*.png"))

        self.bands = ["red", "green", "blue"]

    @property
    def image_paths(self):
        return list(Path(self.path_to_images).rglob("*.png"))

    def caclulate_features(self, arr_path):
        """Extract band data for each file and return band statistics"""

        img = Image.open(arr_path)
        arr = np.array(img)

        row, col, band = arr.shape
        arr_res = arr.reshape(row * col, band)

        # Convert to pandas df and start aggregation
        rgb_df = pd.DataFrame(arr_res, columns=[f"band_{i}_rgb" for i in self.bands])
        rgb_df[self.id_var] = arr_path.stem.split("_")[0]

        # Calculate agg stats for RBG
        rgb_agg = rgb_df.groupby(self.id_var, as_index=False).agg(
            {np.nanmin, np.nanmax, np.nanmean, np.nanmedian, np.std}
        )

        # Concatenate all columns and remove multiindex
        new_cols = ["_".join(t) for t in rgb_agg.columns]
        rgb_agg.columns = new_cols

        return rgb_agg.reset_index()

    def featurize_images(self):
        """Apply function to all images!"""

        list_extract = []
        for arr_path in tqdm(self.path_list, desc="Processing images..."):
            df = self.caclulate_features(arr_path=arr_path)
            list_extract.append(df)

        return pd.concat(list_extract)

    def extract_data(self):
        """Run and save data!"""

        df = self.featurize_images()
        df.to_csv(self.path_to_save, index=False)
