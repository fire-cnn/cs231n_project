""" ViT fine-tuning
"""

import pdb
import argparse
import torch
import wandb
from evaluate import load
from transformers import (ViTForImageClassification,
                          ViTFeatureExtractor,
                          ViTImageProcessor,
                          TrainingArguments,
                          Trainer)

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import random_split

from src.config import Config
from src.dataset import NAIPImagery

def load_dataset(path_to_data,
                 split_share=0.9,
                 preprocess=None):
    """ Load data from test and train!
    """
    
    gen = torch.Generator().manual_seed(42)
    
    # Evaluation dataset
    dataset = NAIPImagery(images_dir=path_to_data,
                          transform=preprocess,
                          max_prompt_len=70,
                          tokenizer=None)

    # Split dataset
    train_size = int(split_share * len(dataset))
    test_size = len(dataset) - train_size

    train, test = random_split(dataset, [train_size, test_size], generator=gen)

    return train, test


def compute_metrics(eval_pred):
    accuracy = load("accuracy")
    f1 = load("f1")
    
    # compute the accuracy and f1 scores & return them
    accuracy_score = accuracy.compute(predictions=np.argmax(eval_pred.predictions, axis=1), references=eval_pred.label_ids)
    f1_score = f1.compute(predictions=np.argmax(eval_pred.predictions, axis=1), references=eval_pred.label_ids, average="weighted")

    return {**accuracy_score, **f1_score}


def collator(batch):
    """ Stucture data for tranining
    """
    
    return  {"pixel_values": torch.cat([x["pixel_values"] for x in batch]),
             "labels": torch.stack([x["labels"] for x in batch])
            }


def main(config, device, tags, dir_project):

    config_train = config.train_config
    model_name = config_train["model_name"]

    image_processor = ViTImageProcessor.from_pretrained(model_name)

    # Load data
    training_dataset, test_dataset = load_dataset(preprocess=image_processor,
                                                  path_to_data=config_train["path_to_data"])

    # Create model
    model = ViTForImageClassification.from_pretrained(model_name, num_labels=2,
                                                      #hidden_dropout_prob=config_train["dropout_hidden"],
                                                      #attention_probs_dropout_prob=config_train["dropout_attention"],
                                                      ignore_mismatched_sizes=True)

    # Start trainer
    training_args = TrainingArguments(
        f"{model_name}-finetuned-fires",
        remove_unused_columns=False,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=20,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=20,
        num_train_epochs=3,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir="logs",
        report_to="wandb",
    )
#    training_args = TrainingArguments(output_dir=config_train["output_dir"],
#                                      num_train_epochs=config_train["epochs"],
#                                      resume_from_checkpoint=config_train["resume_from_checkpoint"],
#                                      load_best_model_at_end=True,
#                                      save_strategy="epoch",
#                                      remove_unused_columns=False,
#                                      evaluation_strategy="epoch",
#                                      per_device_train_batch_size=config_train["batch_size_train"],
#                                      per_device_eval_batch_size=config_train["batch_size_test"],
#                                      warmup_steps=config_train["warmup_steps"],
#                                      learning_rate=float(config_train["learning_rate"]),
#                                      weight_decay=config_train["weight_decay"],
#                                      logging_dir="logs",
#                                      report_to="wandb"
#                                      )
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
