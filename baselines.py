""" Baseline models for comparison
"""

import argparse
import torch
import wandb
import numpy as np

from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.utils.data import random_split
from torchvision import datasets, models, transforms
from tqdm import tqdm

from src.dataset import NAIPImagery
from src.utils_training import save_batch_images
from src.sampler import BalancedBatchSampler

def make(config):
    # Split train/test
    train_len = int(len(full_dataset) * config.train_test_split)
    train_set, test_set = random_split(full_dataset, [train_len, len(full_dataset) - train_len])

    # Create DataLoaders for train and test datasets
    train_loader = DataLoader(
        train_set, 
        batch_size=config.batch_size_test,
        sampler=BalancedBatchSampler(train_set),
        num_workers=config.num_workers
    )
    
    test_loader = DataLoader(
        test_set, 
        batch_size=config.batch_size_train, 
        sampler=BalancedBatchSampler(test_set),
        num_workers=config.num_workers
    )

    # Make the model
    model = models.resnet50(weights='IMAGENET1K_V2')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    model = model.to(device)

    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    return model, train_loader, test_loader, criterion, scheduler, optimizer


def fine_tuning_parameters(model, strategy = "full"):
    """ Select fine tuning strategy for pre-trained model

    This function will take a pre-trained model (either PyTorch or HH) and will
    select the parameters for fine-tuning. We implemented several types of FT,
    being "full" the default strategy. The options are:
        - full: all parameters are fine tuned
        - 'lp': linear probbing (only linear layers are updated)
        - 'last': only last layer is updated
        - 'first': only first layer is updated
        - 'middle': middle of the parameter block is updated only


    Args:
        - model: A PyTorch model object
        - strategy str: a strategy for FT

    Returns:
        A dict of parameters to pass to optimizer
    """

    len_blocks =  len(model.transformer.h)

    if mode == 'full':
        return [x for x in model.parameters()]
    elif mode == 'last':
        return [x for x in model.transformer.h[-2:].parameters()]
    elif mode == 'first':
        return [x for x in model.transformer.h[:2].parameters()]
    elif mode == 'middle':
        mid = (len_blocks + 1) // 2
        mid_params = [model.transformer.h[i] for i in [mid, mid+1]]
        return [x for i in mid_params for x in i.parameters()]
    elif mode.startswith('lora'):
        params = [x for x in model.modules() if isinstance(x, LoRAConv1DWrapper)]

        params_lora = [x for i in params for k, x in i.named_parameters()
                       if 'lora_A' in k or 'lora_B' in k]
        return params_lora
    else:
        raise NotImplementedError()


def train_batch(images, labels, model, optimizer, criterion):

    # Optimize me!
    with torch.set_grad_enabled(True):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()

        # Forward pass ➡
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass ⬅
        loss.backward()

        # Step with optimizer
        optimizer.step()
   
    return loss, outputs


def train_model(model, train_loader, test_loader, criterion, scheduler, optimizer, config):
    
    wandb.watch(model)

    # Run training and track with wandb
    total_batches = len(train_loader) * config.epochs
    example_ct = 0  # number of examples seen
    batch_ct = 0
    for epoch in tqdm(range(config.epochs)):

        # Training step
        train_loss = []
        correct, total = 0, 0
        for _, (images, labels) in enumerate(train_loader):

            loss, out = train_batch(images, labels, model, optimizer, criterion)
            example_ct +=  len(images)
            batch_ct += 1

            # Report metrics every 25th batch
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)
            
            # Log accuracy and loss batch
            _, predicted = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels.cpu()).sum().item()

            train_loss.append(loss.item())

        # Report epoch total loss and log into WB
        print(f"Epoch [{epoch}] training loss: {sum(train_loss)/len(train_loss)}")
        wandb.log({"train-loss": sum(train_loss)/len(train_loss)})
        wandb.log({"train-acc": correct / total})

        # Validation step
        model.eval()
        with torch.no_grad():
            validation_loss = []
            correct, total = 0, 0
            for idx, (images, labels) in enumerate(test_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Validation loss
                loss = criterion(outputs, labels)
                validation_loss.append(loss)

                print(f"Validation Loss batch: {loss}")
                if loss > 2:
                    save_batch_images(images.cpu(), 
                                      f"batch_{idx}",
                                      f"Loss: {loss}"
                                      )

            print(f"Accuracy of the model on the {total} " +
                  f"test images: {correct / total:%}")
            
            wandb.log({"test-loss": sum(validation_loss)/len(validation_loss)})
            wandb.log({"test-accuracy": correct / total})
            scheduler.step()
       

def train_log(loss, example_ct, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")


def model_pipeline(hyperparameters, tags):
    # tell wandb to get started
    with wandb.init(project="cnn_wildfire_households", 
                    mode="online",
                    tags=tags,
                    config=hyperparameters):
      # access all HPs through wandb.config, so logging matches execution!
      config = wandb.config

      # make the model, data, and optimization problem
      model, train_loader, test_loader, criterion, scheduler, optimizer = make(config)

      # and use them to train the model
      train_model(model, train_loader, test_loader, criterion, scheduler, optimizer, config)

    return model


if __name__ == "__main__":

    # Arguments for argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafolder", type=str, help='Images/Labels folder')
    parser.add_argument("--subset_share", type=int, help='Divide dataset in x share')
    parser.add_argument("--tags", type=str, help='Enter tags for W&B')
    
    # Instantiate arguments
    args = parser.parse_args()
    datafolder = args.datafolder
    share = args.subset_share
    tags = [str(item) for item in args.tags.split(',')]
    
    
    ############################# CUDA CONFIGURATION ##############################
    device = 'cpu'
    if torch.cuda.device_count() > 0 and torch.cuda.is_available():
        print("Cuda installed! Running on GPU!")
        device = 'cuda'
    else:
        print("No GPU available!")
    ###############################################################################

    # Define pre-processing
    preprocess = transforms.Compose([
        transforms.Resize((224), antialias=True),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip()
        ]
    )

    # Open and subset dataset
    full_dataset = NAIPImagery(images_dir = "test_data/", transform=preprocess)
    sub_dataset = torch.utils.data.Subset(
          full_dataset, indices=range(0, len(full_dataset), share)
    )

    # Start W&B
    config = dict(
            epochs=30,
            batch_size_train=20,
            batch_size_test=20,
            learning_rate= 0.001,
            weight_decay=0,
            train_test_split=0.8,
            num_workers=5,
            dataset=sub_dataset,
            architecture="ResNET"
            )

    # Run the model!
    model = model_pipeline(config, tags=tags)
