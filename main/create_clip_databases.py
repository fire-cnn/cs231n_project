import os
from src.utils import create_dataset_clip

if __name__ == "__main__":
    list_configs = ["./config_files/config_sherlock_roberta_weather.yaml",
                    "./config_files/config_sherlock_roberta_no_weather.yaml",
                    "./config_files/config_sherlock_roberta.yaml"]

    for config in list_configs:
        create_dataset_clip(config_path=config, save_path="./clip_datasets")

