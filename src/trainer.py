""" Custom HF Trainer class
"""

from transformers import Trainer
import torch


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            # token_type_ids=inputs['token_type_ids']
        )
        loss = torch.nn.BCEWithLogitsLoss()(
            outputs["logits"].float(), inputs["labels"].float()
        )
        return (loss, outputs) if return_outputs else loss
