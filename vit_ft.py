""" ViT fine-tuning
"""

import sys
import traceback
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

from torchvision.transforms import (
    Compose,
    Normalize,
    Resize,
    RandomResizedCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomPerspective,
    RandomAdjustSharpness,
    ToTensor,
    ToPILImage
)

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import random_split

from src.config import Config
from src.dataset import NAIPImagery
from src.trainer import CustomTrainer


def load_dataset(path_to_train,
                 path_to_test,
                 transform=None):
    """ Load data from test and train!
    """


    size = transform.size["height"]

    # Set up transform
    train_aug_transforms = Compose([
        Resize((size, size)),
        RandomResizedCrop(size=size),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        ToTensor(),
        Normalize(mean=transform.image_mean, std=transform.image_std),
    ])

    valid_aug_transforms = Compose([
        Resize(size=(size, size)),
        ToTensor(),
        Normalize(mean=transform.image_mean, std=transform.image_std),
    ])

    # Evaluation dataset
    train = NAIPImagery(images_dir=path_to_train,
                          transform=train_aug_transforms,
                          max_prompt_len=70,
                          tokenizer=None)

    test = NAIPImagery(images_dir=path_to_test,
                          transform=valid_aug_transforms,
                          max_prompt_len=70,
                          tokenizer=None)

    return train, test

def model_init():
    return  ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224-in21k',
        ignore_mismatched_sizes=True,
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


def train(training_dataset, test_dataset, image_processor, wandb_dir, sweep_dir, epochs, config=None):
    """ Training loop for sweep
    """

    with wandb.init(config=config,
                    dir=wandb_dir):
        # Sweep config
        config = wandb.config

        try:
            # Start trainer
            training_args = TrainingArguments(
                    output_dir=sweep_dir,
                    learning_rate=config.learning_rate,
                    per_device_train_batch_size=config.batch_size,
                    num_train_epochs=epochs,
                    #gradient_accumulation_steps=4,
                    weight_decay=config.weight_decay,
                    per_device_eval_batch_size=32,
                    warmup_ratio=config.warmup_ratio,
                    logging_strategy="epoch",
                    evaluation_strategy="epoch",
                    save_strategy="epoch",
                    load_best_model_at_end=True,
                    metric_for_best_model="accuracy",
                    logging_dir="logs",
                    report_to="wandb",
                    fp16=False
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

        except Exception:
            print(traceback.print_exc(), file=sys.stderr)


def main(config, train_fn):
    """ Sweep configuration through W&B
    """

    config_train = config.train_config
    project_name = config_train["project_name"]
    model_name = config_train["model_name"]

    # Add image processing
    image_processor = ViTImageProcessor.from_pretrained(model_name, num_labels=2)

    # Load data
    training_dataset, test_dataset = load_dataset(transform=image_processor,
                                                  path_to_train=config_train["path_to_train"],
                                                  path_to_test=config_train["path_to_test"])

    train_fn_partial = functools.partial(train_fn,
                                         training_dataset,
                                         test_dataset,
                                         image_processor,
                                         config_train["wandb_dir"],
                                         config_train["sweep_dir"],
                                         config_train["epochs"]
                                         )

    # Sweep configuration for training
    sweep_configuration = {
            "method": "bayes",
            "name": config_train["sweep_name"],
            "metric": {
                "goal": "minimize",
                "name": "eval_loss"
            },
            "parameters": {
                "batch_size": {
                    "values": [16, 32, 64]
                    },
                "learning_rate": {
                    "values": [5e-5, 5e-4, 2e-4]
                    },
                "weight_decay": {
                    "values": [0.001, 0.002, 0.005],
                    },
                "warmup_ratio": {
                    "values": [0, 0.1],
                }}
                }
    # Set up sweep in wandb
    sweep_id = wandb.sweep(sweep_configuration, project=project_name)

    # Start sweep
    wandb.agent(sweep_id, train_fn_partial, count=20)

def main_worker(config, train_fn, sweep_id):
    """ Sweep configuration through W&B
    """

    config_train = config.train_config
    project_name = config_train["project_name"]
    model_name = config_train["model_name"]

    # Add image processing
    image_processor = ViTImageProcessor.from_pretrained(model_name)

    # Load data
    training_dataset, test_dataset = load_dataset(transform=image_processor,
                                                  path_to_train=config_train["path_to_train"],
                                                  path_to_test=config_train["path_to_test"])

    train_fn_partial = functools.partial(train_fn,
                                         training_dataset,
                                         test_dataset,
                                         image_processor,
                                         config_train["wandb_dir"],
                                         config_train["sweep_dir"]
                                         )

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

