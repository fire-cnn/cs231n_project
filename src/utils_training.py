import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


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
