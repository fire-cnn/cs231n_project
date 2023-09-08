""" ViT fine-tuning
"""


import pdb
import functools
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
from src.trainer import CustomTrainer

def load_dataset(path_to_data,
                 split_share=0.9,
                 transform=None):
    """ Load data from test and train!
    """
    
    gen = torch.Generator().manual_seed(42)
    
    # Evaluation dataset
    dataset = NAIPImagery(images_dir=path_to_data,
                          transform=transform,
                          max_prompt_len=70,
                          tokenizer=None)

    # Split dataset
    train_size = int(split_share * len(dataset))
    test_size = len(dataset) - train_size

    train, test = random_split(dataset, [train_size, test_size], generator=gen)

    return train, test

def model_init():
    return  ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224-in21k',
        num_labels=2,
    )

def compute_metrics(eval_pred):

    # Get values from Trainer loss
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    preds = np.argmax(logits, axis=1)

    # Get metrics
    metrics = dict()

    accuracy_metric = load("accuracy")
    precision_metric = load("precision")
    recall_metric = load("recall")
    f1_metric = load("f1")

    metrics.update(accuracy_metric.compute(predictions=preds, references=labels))
    metrics.update(precision_metric.compute(predictions=preds, references=labels, average="weighted"))
    metrics.update(recall_metric.compute(predictions=preds, references=labels, average="weighted"))
    metrics.update(f1_metric.compute(predictions=preds, references=labels, average="weighted"))

    return metrics


def collator(batch):
    """ Stucture data for tranining
    """
    
    return  {"pixel_values": torch.cat([x["pixel_values"] for x in batch]),
             "labels": torch.stack([x["labels"] for x in batch])
            }


def train(training_dataset, test_dataset, image_processor, wandb_dir, sweep_dir, config=None):
    """ Training loop for sweep
    """
   
    with wandb.init(config=config,
                    dir=wandb_dir):
        # Sweep config
        config = wandb.config

        # Start trainer
        training_args = TrainingArguments(
                output_dir=sweep_dir,
                learning_rate=config.learning_rate,
                per_device_train_batch_size=config.batch_size,
                num_train_epochs=10,
                weight_decay=config.weight_decay,
                per_device_eval_batch_size=16,
                warmup_steps=config.warmup_steps,
                logging_strategy="epoch",
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                logging_dir="logs",
                report_to="wandb",
                fp16=True
                )
 

        trainer = CustomTrainer(
                model=model_init(),
                args=training_args,
                train_dataset=training_dataset,
                eval_dataset=test_dataset,
                tokenizer=image_processor,
                compute_metrics=compute_metrics,
                data_collator=collator
                )

        trainer.train()

def main(config, train_fn):
    """ Sweep configuration through W&B 
    """
    
    config_train = config.train_config
    project_name = config_train["project_name"]
    model_name = config_train["model_name"]
    
    # Add image processing
    image_processor = ViTImageProcessor.from_pretrained(model_name)
 
    # Load data
    training_dataset, test_dataset = load_dataset(transform=image_processor,
                                                  path_to_data=config_train["path_to_data"])

    train_fn_partial = functools.partial(train_fn, 
                                         training_dataset, 
                                         test_dataset,
                                         image_processor,
                                         config_train["wandb_dir"],
                                         config_train["sweep_dir"]
                                         )

    # Sweep configuration for training
    sweep_configuration = {
            "method": "random",
            "name": "vit_sweep",
            "parameters": {
                "batch_size": {
                    "distribution": "q_log_uniform_values",
                    "q": 4,
                    "max": 32,
                    "min": 8
                    },
                "learning_rate": {
                    "distribution": "uniform",
                    "max": 0.1,
                    "min": 0
                    },
                "weight_decay": {
                    "distribution": "uniform",
                    "max": 0.5,
                    "min": 0.1
                    },
                "warmup_steps": {
                    "distribution": "int_uniform",
                    "max": 100,
                    "min": 0
                }}
                }
    # Set up sweep in wandb
    sweep_id = wandb.sweep(sweep_configuration, project=project_name)
    
    # Start sweep
    wandb.agent(sweep_id, train_fn_partial, count=20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, help="Path to YAML config file")

    # Init args
    args = parser.parse_args()
    path_to_config = args.config_file
    
    # Init W&B
    wandb.login()

    ############################# CUDA CONFIGURATION ##############################
    device = "cpu"
    if torch.cuda.device_count() > 0 and torch.cuda.is_available():
        print("Cuda installed! Running on GPU!")
        device = "cuda"
    else:
        print("No GPU available!")
    ###############################################################################

    config =  Config(path_to_config)
    main(config, train) 
