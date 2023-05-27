import numpy as np
import pandas as pd

import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import ToTensor

from src.prompts import prompting


class NAIPImagery(Dataset):
    def __init__(
        self,
        images_dir,
        tabular_data=None,
        transform=None,
        prompt_type=None,
        template=None,
        dict_column_names=None,
        cols_template=None,
        columns_of_interest=None,
        id_var=None,
    ) -> None:
        super().__init__()

        self.images_dir = images_dir
        self.tabular_data = tabular_data
        self.prompt_type = prompt_type
        self.id_var = id_var
        self.columns_of_interest = columns_of_interest
        self.dict_column_names = dict_column_names
        self.transform = transform
        self.paths = list(Path(self.images_dir).glob("*.png"))

        # Get label data
        class_label = []
        for path in self.paths:
            label = path.stem.split("_")[-1]
            class_label.append(int(label))

        self.targets = np.array(class_label)

        # Calculate weights
        classes, counts = np.unique(np.array(class_label), return_counts=True)
        self.weights = 1 / counts

        if isinstance(tabular_data, str):
            self.tabular_data = pd.read_csv(self.tabular_data)

        # Transform tabular data into prompts
        self.dict_prompts = prompting(
            df=self.tabular_data,
            prompt_type=self.prompt_type,
            template=template,
            id_var=self.id_var,
            column_name_map=self.dict_column_names,
            cols_template=cols_template,
            columns_of_interest=self.columns_of_interest,
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path_img = self.paths[idx]

        # Get image id and get text prompt
        id_img = int(path_img.stem.split("_")[0])
        text_img = self.dict_prompts[id_img]

        # Get image label
        label_img = int(path_img.stem.split("_")[-1])
        label_img = torch.as_tensor(np.array(label_img))

        img = Image.open(str(self.paths[idx])).resize((224, 224))

        # Tranform to tensor
        img = np.array(img)
        img = ToTensor()(img)

        if self.transform:
            img = self.transform(img)

        return img, text_img, label_img
