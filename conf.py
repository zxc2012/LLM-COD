from transformers import (AutoModel,
                          Trainer
                          )
import torch

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        ## Weighted Cross Entropy
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.63, 2.67, 4.93, 13.56, 1.17, 3.06, 4.61, 1.89, 4.6, 14.92, 5.53,
                                                                  9.07, 1.0, 7.15, 3.1, 4.44, 10.77, 13.35, 2.89, 20.04, 372.99, 181.58,
                                                                  587.47], device=logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss