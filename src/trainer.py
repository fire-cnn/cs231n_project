""" Custom HF Trainer class
"""

from transformers import Trainer
import torch


from torch import nn
from transformers import Trainer


class CustomTrainer(Trainer):

    def __init__(weights):

        super().__init__()

        self.class_weights = weights
        
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([0.9, 0.1], device=model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


