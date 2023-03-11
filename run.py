from transformers import (
    BloomTokenizerFast,
    BloomForSequenceClassification, 
    Trainer, 
    TrainingArguments, 
    DataCollatorWithPadding,
    HfArgumentParser
)
from typing import Optional
from datasets import load_dataset, load_metric
from dataclasses import dataclass, field

import wandb
import numpy as np


@dataclass
class AuxArguments:
    model_name: Optional[str] = None
    run_name: Optional[str] = None


ACCURACY = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def main():
    parser = HfArgumentParser((AuxArguments, TrainingArguments))
    aux_args, training_args = parser.parse_args_into_dataclasses()

    wandb.init(
        entity="broccoliman",
        project="efficient_dl_week6",
        name=aux_args.run_name
    )

    tokenizer = BloomTokenizerFast.from_pretrained(aux_args.model_name)
    model = BloomForSequenceClassification.from_pretrained(aux_args.model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    def tokenize(x):
        return tokenizer(x["sentence"])

    sst = load_dataset("glue", "sst2").map(tokenize)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=sst["train"],
        eval_dataset=sst["validation"],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    wandb.finish()

if __name__ == "__main__":
    main()
