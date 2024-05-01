import numpy as np

from transformers import TrainingArguments, Trainer

from src.features.metrics import compute_metrics

class HFAudioTrainer:
    def __init__(self, **kwargs) -> None:
        self.train_args = TrainingArguments(**kwargs)

        self.trainer = None

    def load_trainer(self, model, train_dset, val_dset, tokenizer):
        trainer = Trainer(
            model, 
            args=self.train_args,
            train_dataset=train_dset,
            eval_dataset=val_dset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        self.trainer = trainer

        return self
    
    def train(self):
        self.trainer.train()