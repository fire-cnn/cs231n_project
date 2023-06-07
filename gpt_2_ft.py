""" GPT-2 fine-tuning
"""

import pdb
import argparse
import torch
import wandb
import evaluate
from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW, 
                          AutoTokenizer,
                          DataCollatorWithPadding,
                          AutoModelForSequenceClassification
                          )
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import Config
from src.dataset import NAIPImagery
from src.trainer import CustomTrainer
from src.prompts import prompting

def load_dataset(tokenizer, 
                 path_to_train, 
                 path_to_test, 
                 tabular_data_path,
                 config_prompt,
                 preprocess=None):
    """ Load data from test and train!
    """

    # Evaluation dataset
    test_dataset = NAIPImagery(images_dir=path_to_test,
                                   transform=preprocess,
                                   tabular_data=tabular_data_path,
                                   tokenizer=tokenizer,
                                   max_prompt_len=70,
                                   **config_prompt.prompt_config)


    # Training dataset
    training_dataset = NAIPImagery(images_dir=path_to_train,
                                   transform=preprocess,
                                   tabular_data=tabular_data_path,
                                   tokenizer=tokenizer,
                                   max_prompt_len=70,
                                   **config_prompt.prompt_config)

    return training_dataset, test_dataset


def compute_metrics(eval_pred):
    # Setup evaluation
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def main(config, device, tags, dir_project):

    config_train = config.train_config
    model_name = config_train["model_name"]
    
    # Instantiate model and tokenizer
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                          bos_token="<start_of_text>",
                                          eos_token="<end_of_text>")
    tokenizer.padding_side = "left"
    
    # Define PAD Token = EOS Token = 50256
    tokenizer.pad_token = tokenizer.eos_token
    
    # Set up collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer) 

    # Set up classifier
    model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                               num_labels=2)
    # fix model padding token id
    model.config.pad_token_id = model.config.eos_token_id

    # Load data
    training_dataset, test_dataset = load_dataset(tokenizer,
                                                  path_to_train=config_train["path_to_train"],
                                                  path_to_test=config_train["path_to_test"],
                                                  tabular_data_path=config_train["tabular_data_path"],
                                                  config_prompt=config
                                                  )

    # Start trainer
    training_args = TrainingArguments(output_dir=config_train["output_dir"],
                                      num_train_epochs=config_train["epochs"],
                                      resume_from_checkpoint=config_train["resume_from_checkpoint"],
                                      load_best_model_at_end=True,
                                      save_strategy="epoch",
                                      evaluation_strategy="epoch",
                                      per_device_train_batch_size=config_train["batch_size_train"],
                                      per_device_eval_batch_size=config_train["batch_size_test"],
                                      weight_decay=config_train["weight_decay"],
                                      warmup_steps=config_train["warmup_steps"],
                                      learning_rate=float(config_train["learning_rate"]),
                                      logging_dir="logs",
                                      report_to="wandb"
                                      )
    with wandb.init(
        project="cnn_wildfire_households",
        mode="online",
        tags=tags,
        dir=dir_project,
        group="gpt"):

        Trainer(model=model, 
                args=training_args,
                train_dataset=training_dataset,
                eval_dataset=test_dataset,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
                data_collator=data_collator).train()
    
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
