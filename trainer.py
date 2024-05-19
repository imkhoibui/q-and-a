from typing import Any, Dict, Union

import torch
from torch import nn

from transformers import Trainer as HFTrainer
from utils import label_smoothed_nll_loss

class Trainer(HFTrainer):
    """
        This is a custom Trainer class that extends the HFTrainer class
    """
    def __init__(self, label_smoothing: float = 0, **kwargs):
        super().__init__(**kwargs)
        self.label_smoothing = label_smoothing

    def _training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> float:
        model.train()
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)

        if isinstance(model, nn.DataParallel):
            inputs["return_tuple"] = True

        if self.label_smoothing == 0:
            outputs = model(**inputs)
            loss = outputs[0]
        else:
            labels = inputs.pop("labels")
            labels[labels == -100] = model.config.pad_token_id
            outputs = model(**inputs)
            lprobs = torch.nn.functional.log_softmax(outputs[0], dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(lprobs, labels, self.label_smoothing, ignore_index=model.config.pad_token_id)

        if self.args.n_gpu > 1:
            loss = loss.mean()
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
            
        loss.backward()

        return loss.item()