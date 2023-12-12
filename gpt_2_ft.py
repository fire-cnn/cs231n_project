""" GPT-2 fine-tuning
"""

import argparse
import torch
import wandb
from evaluate import load
from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
)
import numpy as np

from src.config import Config
from src.dataset import NAIPImagery
from tqdm import tqdm


def load_dataset(
    tokenizer,
    path_to_train,
    path_to_test,
    tabular_data_path,
    config_prompt,
    preprocess=None,
):
    """Load data from test and train!"""

    # Evaluation dataset
    test_dataset = NAIPImagery(
        images_dir=path_to_test,
        transform=preprocess,
        tabular_data=tabular_data_path,
        tokenizer=tokenizer,
        max_prompt_len=100,
        **config_prompt.prompt_config
    )

    # Training dataset
    training_dataset = NAIPImagery(
        images_dir=path_to_train,
        transform=preprocess,
        tabular_data=tabular_data_path,
        tokenizer=tokenizer,
        max_prompt_len=100,
        **config_prompt.prompt_config
    )

    return training_dataset, test_dataset


def compute_metrics(eval_pred):
    accuracy = load("accuracy")
    f1 = load("f1")
    # compute the accuracy and f1 scores & return them
    accuracy_score = accuracy.compute(
        predictions=np.argmax(eval_pred.predictions, axis=1),
        references=eval_pred.label_ids,
    )
    f1_score = f1.compute(
        predictions=np.argmax(eval_pred.predictions, axis=1),
        references=eval_pred.label_ids,
        average="weighted",
    )

    return {**accuracy_score, **f1_score}


def validation(dataloader, device_):
    r"""Validation function to evaluate model performance on a
    separate set of data.

    This function will return the true and predicted labels so we can use later
    to evaluate the model's performance.

    This function is built with reusability in mind: it can be used as is as long
      as the `dataloader` outputs a batch in dictionary format that can be passed
      straight into the model - `model(**batch)`.

    Arguments:

      dataloader (:obj:`torch.utils.data.dataloader.DataLoader`):
            Parsed data into batches of tensors.

      device_ (:obj:`torch.device`):
            Device used to load tensors before feeding to model.

    Returns:

      :obj:`List[List[int], List[int], float]`: List of [True Labels, Predicted
          Labels, Train Average Loss]
    """

    # Use global variable for model.
    global model

    # Tracking variables
    predictions_labels = []
    true_labels = []
    # total loss for this epoch.
    total_loss = 0

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Evaluate data for one epoch
    for batch in tqdm(dataloader, total=len(dataloader)):
        # add original labels

        # Remove pixel values from batch
        del batch["pixel_values"]

        true_labels += batch["labels"].numpy().flatten().tolist()

        # move batch to device
        batch = {k: v.type(torch.long).to(device_) for k, v in batch.items()}

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have
            # not provided labels.
            # token_type_ids is the same as the "segment ids", which
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(**batch)

            # The call to `model` always returns a tuple, so we need to pull the
            # loss value out of the tuple along with the logits. We will use logits
            # later to to calculate training accuracy.
            loss, logits = outputs[:2]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_loss += loss.item()

            # get predicitons to list
            predict_content = logits.argmax(axis=-1).flatten().tolist()

            # update list
            predictions_labels += predict_content

    # Calculate the average loss over the training data.
    avg_epoch_loss = total_loss / len(dataloader)

    # Return all true labels and prediciton for future evaluations.
    return true_labels, predictions_labels, avg_epoch_loss


def main(config, device, tags, dir_project):
    config_train = config.train_config
    model_name = config_train["model_name"]

    # Instantiate model and tokenizer
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        # bos_token="<start_of_text>",
        # eos_token="<end_of_text>"
    )
    # tokenizer.padding_side = "left"

    # Define PAD Token = EOS Token = 50256
    tokenizer.pad_token = tokenizer.eos_token

    # Set up collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Set up classifier
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # fix model padding token id
    model.config.pad_token_id = model.config.eos_token_id

    # Load data
    training_dataset, test_dataset = load_dataset(
        tokenizer,
        path_to_train=config_train["path_to_train"],
        path_to_test=config_train["path_to_test"],
        tabular_data_path=config_train["tabular_data_path"],
        config_prompt=config,
    )

    # Start trainer
    training_args = TrainingArguments(
        output_dir=config_train["output_dir"],
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
        report_to="wandb",
    )
    with wandb.init(
        project="cnn_wildfire_households",
        mode="online",
        tags=tags,
        dir=dir_project,
        group="gpt",
    ):
        Trainer(
            model=model,
            args=training_args,
            train_dataset=training_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
        ).train()

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, help="Path to YAML config file")
    parser.add_argument(
        "--dir_project", type=str, default="./wandb", help="Path for W&B saved data"
    )
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

    config = Config(path_to_config)
    main(config, device, tags, path_to_dir)
