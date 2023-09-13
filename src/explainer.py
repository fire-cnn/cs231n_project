import pdb
import warnings

warnings.filterwarnings("ignore")
from torchvision import transforms
from datasets import load_dataset
from pytorch_grad_cam import run_dff_on_image, GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import numpy as np
import cv2
import torch
from typing import List, Callable, Optional
from transformers import ViTFeatureExtractor, ViTForImageClassification

# dataset = load_dataset("huggingface/cats-image")
# image = dataset["test"]["image"][0]
# img_tensor = transforms.ToTensor()(image)

""" Model wrapper to return a tensor"""


class HuggingfaceToTensorModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(HuggingfaceToTensorModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).logits


""" Translate the category name to the category index.
    Some models aren't trained on Imagenet but on even larger datasets,
    so we can't just assume that 761 will always be remote-control.

"""


def category_name_to_index(model, category_name):
    name_to_index = dict((v, k) for k, v in model.config.id2label.items())
    return name_to_index[category_name]


""" Helper function to run GradCAM on an image and create a visualization.
    (note to myself: this is probably useful enough to move into the package)
    If several targets are passed in targets_for_gradcam,
    e.g different categories,
    a visualization for each of them will be created.
    
"""


def run_grad_cam_on_image(
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    targets_for_gradcam: List[Callable],
    reshape_transform: Optional[Callable],
    input_tensor: torch.nn.Module,
    input_image: Image,
    method: Callable = GradCAM,
):
    with method(
        model=HuggingfaceToTensorModelWrapper(model),
        target_layers=[target_layer],
        reshape_transform=reshape_transform,
    ) as cam:
        # Replicate the tensor for each of the categories we want to create Grad-CAM for:
        repeated_tensor = input_tensor[None, :].repeat(
            len(targets_for_gradcam), 1, 1, 1
        )

        batch_results = cam(input_tensor=repeated_tensor, targets=targets_for_gradcam)
        results = []
        for grayscale_cam in batch_results:
            visualization = show_cam_on_image(
                np.float32(input_image) / 255, grayscale_cam, use_rgb=True
            )
            # Make it weight less in the notebook:
            visualization = cv2.resize(
                visualization,
                (visualization.shape[1] // 2, visualization.shape[0] // 2),
            )
            results.append(visualization)
        return np.hstack(results)


def print_top_categories(model, img_tensor, top_k=1):
    logits = model(img_tensor.unsqueeze(0)).logits
    indices = logits.cpu()[0, :].detach().numpy().argsort()[-top_k:][::-1]
    for i in indices:
        print(f"Predicted class {i}: {model.config.id2label[i]}")


def reshape_transform_vit_huggingface(x):
    # pdb.set_trace()
    activations = x[:, 1:, :]
    activations = activations.view(activations.shape[0], 14, 14, activations.shape[2])
    activations = activations.transpose(2, 3).transpose(1, 2)
    return activations


if __name__ == "__main__":
    model_name = "google/vit-base-patch16-224-in21k"
    img_path = "/mnt/sherlock/oak/cnn_fire_fuel/data/test/10573_ca_m_3912112_sw_10_060_20180718_20190209_0.png"
    image_processor = ViTFeatureExtractor.from_pretrained(model_name)

    image = Image.open(img_path)
    # tensor_resized = image_processor(image, return_tensors="pt")["pixel_values"]

    targets_for_gradcam = [ClassifierOutputTarget(0)]
    target_layer_dff = model.vit.layernorm
    target_layer_gradcam = model.vit.encoder.layer[-2].output

    image_resized = image.resize((224, 224))
    tensor_resized = transforms.ToTensor()(image_resized)

    arr = run_grad_cam_on_image(
        model=model,
        target_layer=target_layer_gradcam,
        targets_for_gradcam=targets_for_gradcam,
        input_tensor=tensor_resized,
        input_image=image_resized,
        reshape_transform=reshape_transform_vit_huggingface,
    )
