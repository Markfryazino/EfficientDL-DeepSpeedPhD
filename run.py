from transformers import (
    BloomTokenizerFast,
    BloomForSequenceClassification, 
    Trainer, 
    TrainingArguments, 
    DataCollatorWithPadding,
    HfArgumentParser
)
from typing import Optional
from datasets import load_dataset
from dataclasses import dataclass, field

import wandb
import numpy as np
import os


@dataclass
class AuxArguments:
    model_name: Optional[str] = None
    wandb_run_name: Optional[str] = None
    max_examples: Optional[int] = None


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": (labels == predictions).mean()}


def main():
    parser = HfArgumentParser((AuxArguments, TrainingArguments))
    aux_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = BloomTokenizerFast.from_pretrained(aux_args.model_name)
    model = BloomForSequenceClassification.from_pretrained(aux_args.model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    def tokenize(x):
        return tokenizer(x["sentence"])

    sst = load_dataset("glue", "sst2").map(tokenize)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=sst["train"].select(range(aux_args.max_examples)),
        eval_dataset=sst["validation"],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    if os.getenv("RANK") == "0":
        wandb.init(
            entity="broccoliman",
            project="efficient_dl_deepspeed_phd",
            name=aux_args.wandb_run_name
        )

    trainer.train()


if __name__ == "__main__":
    main()
