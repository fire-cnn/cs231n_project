from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.prompts import prompting

from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
)


class NAIPImagery(Dataset):
    def __init__(
        self,
        images_dir,
        test_size=0.2,
        tokenizer=None,
        max_prompt_len=None,
        tabular_data=None,
        transform=None,
        prompt_type=None,
        template=None,
        final_prompt=None,
        column_name_map=None,
        cols_template=None,
        columns_of_interest=None,
        id_var=None,
        add_response=None,
        label_column=None,
    ) -> None:
        """
        A dataset class to represent NAIP house data in two
        modes: aerial images and fire/house characteristics
        """
        super().__init__()

        self.images_dir = images_dir
        self.tabular_data = tabular_data
        self.prompt_type = prompt_type
        self.test_size = test_size
        self.id_var = id_var
        self.template = template
        self.cols_template = cols_template
        self.columns_of_interest = columns_of_interest
        self.column_name_map = column_name_map
        self.final_prompt = final_prompt
        self.label_column = label_column
        self.add_response = add_response
        self.transform = transform
        self.tokenizer = tokenizer

        # If no length is defined, then use the tokenizer max number. The
        # tokenizer will pad the string to that legth.
        if max_prompt_len is None:
            self.max_prompt_len = tokenizer.model_max_length
        else:
            self.max_prompt_len = max_prompt_len

        self.paths = list(Path(self.images_dir).glob("*.png"))

        if isinstance(tabular_data, str):
            self.tabular_data = pd.read_csv(self.tabular_data)

        # Transform tabular data into prompts
        if self.tokenizer is not None:
            self.dict_prompts = prompting(
                df=self.tabular_data,
                prompt_type=self.prompt_type,
                template=self.template,
                id_var=self.id_var,
                label_column=self.label_column,
                column_name_map=self.column_name_map,
                cols_template=self.cols_template,
                final_prompt=self.final_prompt,
                add_response=self.add_response,
                columns_of_interest=self.columns_of_interest,
            )

    @property
    def weights(self):
        # Calculate training weights
        labs = [p.stem.split("_")[-1] for p in self.paths]
        _, counts = np.unique(labs, return_counts=True)

        return np.sum(counts) / (counts * 2)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path_img = self.paths[idx]
        # Get image id and get text prompt
        id_img = int(path_img.stem.split("_")[0])

        # Get image label
        label_img = int(path_img.stem.split("_")[-1])
        label_img = torch.as_tensor(np.array(label_img))

        img = Image.open(str(self.paths[idx]))

        # Tranform to tensor
        if self.transform:
            img = self.transform(img)
        else:
            # Apply compose and resize the image to 224x224
            img = Compose([Resize((224, 224)), ToTensor()])(img)

        # Tokenize the text
        if self.tokenizer is not None:
            text_img = self.dict_prompts[id_img]
            embeddings_dict = self.tokenizer(
                text=text_img,
                truncation=True,
                padding="max_length",
                max_length=self.max_prompt_len,
            )

            out = {
                "pixel_values": img,
                "labels": label_img,
                "input_ids": embeddings_dict["input_ids"],
                "attention_mask": embeddings_dict["attention_mask"],
            }
        else:
            out = {"pixel_values": img[None, :], "labels": label_img}

        return out
