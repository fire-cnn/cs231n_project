""" Custom HF Trainer class
"""

from transformers import Trainer
import torch

import pdb

from torch import nn
from transformers import Trainer


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        pdb.set_trace()
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        weights = self.train_dataset.weights
        loss_fct = nn.CrossEntropyLoss(weight=torch.Tensor(weights).to(model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
