""" ViT fine-tuning
"""

import pdb
import argparse
import torch
import wandb
import evaluate
from transformers import (ViTForImageClassification,
                          ViTFeatureExtractor,
                          ViTImageProcessor,
                          TrainingArguments,
                          Trainer)

import numpy as np

from src.config import Config
from src.dataset import NAIPImagery

def load_dataset(path_to_train,
                 path_to_test,
                 tabular_data_path,
                 config_prompt,
                 tokenizer=None,
                 preprocess=None):
    """ Load data from test and train!
    """


    # Evaluation dataset
    test_dataset = NAIPImagery(images_dir=path_to_test,
                               transform=preprocess,
                               tabular_data=tabular_data_path,
                               max_prompt_len=70,
                               tokenizer=tokenizer)

    # Training dataset
    training_dataset = NAIPImagery(images_dir=path_to_train,
                                   tabular_data=tabular_data_path,
                                   tokenizer=tokenizer,
                                   max_prompt_len=70,
                                   transform=preprocess)

    return training_dataset, test_dataset


def compute_metrics(eval_pred):
    # Setup evaluation
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def collator(batch):
    """ Stucture data for tranining
    """
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return  {"pixel_values": torch.cat([x["pixel_values"]["pixel_values"] for x in batch]),
             "labels": torch.stack([x["labels"] for x in batch])
            }


def main(config, device, tags, dir_project):

    config_train = config.train_config
    model_name = config_train["model_name"]

    #feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    image_processor = ViTImageProcessor.from_pretrained(model_name)

    # Load data
    training_dataset, test_dataset = load_dataset(preprocess=image_processor,
                                                  path_to_train=config_train["path_to_train"],
                                                  path_to_test=config_train["path_to_test"],
                                                  tabular_data_path=config_train["tabular_data_path"],
                                                  config_prompt=config
                                                  )

    #pdb.set_trace()
    # Create model
    model = ViTForImageClassification.from_pretrained(model_name, num_labels=2)

    # Start trainer
    training_args = TrainingArguments(output_dir=config_train["output_dir"],
                                      num_train_epochs=config_train["epochs"],
                                      resume_from_checkpoint=config_train["resume_from_checkpoint"],
                                      load_best_model_at_end=True,
                                      save_strategy="epoch",
                                      remove_unused_columns=False,
                                      evaluation_strategy="epoch",
                                      per_device_train_batch_size=config_train["batch_size_train"],
                                      per_device_eval_batch_size=config_train["batch_size_test"],
                                      warmup_steps=config_train["warmup_steps"],
                                      learning_rate=float(config_train["learning_rate"]),
                                      weight_decay=config_train["weight_decay"],
                                      logging_dir="logs",
                                      report_to="wandb"
                                      )
    with wandb.init(
        project="cnn_wildfire_households",
        mode="online",
        tags=tags,
        dir=dir_project,
        group="vit"):

        trainer = Trainer(model=model,
                args=training_args,
                train_dataset=training_dataset,
                eval_dataset=test_dataset,
                tokenizer=image_processor,
                compute_metrics=compute_metrics,
                data_collator=collator).train()


    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, help="Path to YAML config file")
    parser.add_argument("--dir_project", type=str, default="./wandb", help="Path for W&B saved data")
    parser.add_argument("--tags", type=str, help="Tags for W&B")

    # Init args
    args = parser.parse_args()
    path_to_config = args.config_file
    path_to_dir = args.dir_project
    tags = [str(item) for item in args.tags.split(",")]


    ############################# CUDA CONFIGURATION ##############################
    device = "cpu"
    if torch.cuda.device_count() > 0 and torch.cuda.is_available():
        print("Cuda installed! Running on GPU!")
        device = "cuda"
    else:
        print("No GPU available!")
    ###############################################################################

    config =  Config(path_to_config)
    main(config, device, tags, path_to_dir)
