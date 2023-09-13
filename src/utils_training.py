import os
import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from pathlib import Path
from PIL import Image


def _read_image(path):
    """ Read Image from path

    Replicate torchvision.io.read_image because Sherlock is annoying!
    (couldn't compile torchvision with libpng support)

    """
    img = Image.open(path)
    img_arr = np.array(img)

    # To tensor
    img_torch = torch.from_numpy(img_arr)

    return torch.transpose(img_torch, -1, 0)


def create_balanced_example(path_to_examples, path_to_example_dataset, size=50):
    """Create test dataset"""

    # Let's always get the same sets
    np.random.seed(42)

    # Create directory if no exist
    if not os.path.exists(path_to_example_dataset):
        os.makedirs(path_to_example_dataset, exist_ok=True)

    path_images = list(Path(path_to_examples).rglob("*.png"))

    pos_examples, neg_examples = [], []
    for p in path_images:
        if p.stem.split("_")[-1] == "1":
            pos_examples.append(p)
        else:
            neg_examples.append(p)

    # Create subset
    subset_pos = np.random.choice(pos_examples, size)
    subset_neg = np.random.choice(neg_examples, size)

    for pos, neg in zip(subset_pos, subset_neg):
        pos_target = os.path.join(path_to_example_dataset, pos.name)
        neg_target = os.path.join(path_to_example_dataset, neg.name)

        shutil.copyfile(pos, pos_target)
        shutil.copyfile(neg, neg_target)

    return None


def save_batch_images(batch_images, filename=None, title=None):
    """Display image for Tensor."""

    # Use makegrid for batch viz
    out = make_grid(batch_images, padding=5)

    # Plot stuff and save to filename
    fig, ax = plt.subplots(figsize=(20, 20))

    # Add title
    plt.title(f"{title}")

    inp = out.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)

    # Pause so plots are rendered
    plt.pause(0.001)
    ax.set_axis_off()

    if filename:
        fig.savefig(f"{filename}.png")

    return None
