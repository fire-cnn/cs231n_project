""" Create constrastive loss model
"""


import argparse
from pathlib import Path
import pandas as pd

from transformers import (
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    AutoTokenizer,
    AutoImageProcessor,
)

from src.prompts import prompting
from src.config import Config


def create_model(name_vision, name_text, model_name):
    model = VisionTextDualEncoderModel.from_vision_text_pretrained(
        name_vision, name_text, n_labels=2
    )

    tokenizer = AutoTokenizer.from_pretrained(name_text)
    image_processor = AutoImageProcessor.from_pretrained(name_vision)

    tokenizer.padding_side = "left"

    # Define PAD Token = EOS Token = 50256
    tokenizer.pad_token = tokenizer.eos_token
    processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)

    # save the model and processor
    model.save_pretrained(model_name)
    processor.save_pretrained(model_name)

    return None


def create_data(config):
    config_train = config.train_config

    for type_data in ["path_to_train", "path_to_test"]:
        # Get name type
        name_type = type_data.split("_")[-1]
        print(f"Working on {name_type} data")
        # Get paths from images
        id_var = config.prompt_config["id_var"]
        paths_data = list(Path(config_train[type_data]).rglob("*.png"))
        ids = [int(p.stem.split("_")[0]) for p in paths_data]
        y_test = [int(p.stem.split("_")[-1]) for p in paths_data]

        # Get prompts from only that type from the tabular data
        tabular_data = pd.read_csv(config_train["tabular_data_path"])
        test_data_tabular = tabular_data[tabular_data[id_var].isin(ids)]
        x_test = prompting(
            df=test_data_tabular, special_tokens=("", "", ""), **config.prompt_config
        )

        # Build data frame
        paths_test_str = [str(p) for p in paths_data]
        df_paths = pd.DataFrame.from_dict(
            {"homeid": ids, "image_column": paths_test_str}
        )

        df_text = pd.DataFrame.from_dict(x_test, orient="index").reset_index()
        df_text = df_text.rename(columns={"index": "homeid", 0: "caption_column"})

        df_paths.merge(df_text, on="homeid").drop(columns="homeid").to_csv(
            f"{name_type}_data.csv", index=False
        )

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name_vision",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="HF vision model name for vision encoder",
    )
    parser.add_argument(
        "--name_text",
        type=str,
        default="roberta-base",
        help="HF text model name for text encoder",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="clip-roberta",
        help="Name of DualEncoder name to save",
    )
    parser.add_argument("--config_path", type=str, help="Path to confiig file")

    args = parser.parse_args()
    name_vision = args.name_vision
    name_text = args.name_text
    model_name = args.model_name
    path_to_config = args.config_path

    config = Config(path_to_config)

    create_model(name_vision, name_text, model_name)
    #create_data(config)
