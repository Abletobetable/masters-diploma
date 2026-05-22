#!/usr/bin/env python3
import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset
from rec_rlp.config import load_yaml


def main():
    p = argparse.ArgumentParser(description="Глава 3.3: P5 на T5-base")
    p.add_argument("--config", default="train_p5.yaml")
    p.add_argument("--dataset", type=str)
    p.add_argument("--output-dir", type=str)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    dataset_path = args.dataset or cfg["dataset"]
    output_dir = args.output_dir or cfg["output_dir"]

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model_name"])

    def preprocess(batch):
        src = tokenizer(batch["prompt"], max_length=cfg["max_source_length"], truncation=True)
        tgt = tokenizer(text_target=batch["target"], max_length=cfg["max_target_length"], truncation=True)
        src["labels"] = tgt["input_ids"]
        return src

    ds = load_dataset("json", data_files=dataset_path, split="train")
    ds = ds.map(preprocess, batched=True, remove_columns=ds.column_names)

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        args=Seq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            learning_rate=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
            num_train_epochs=cfg["num_train_epochs"],
            bf16=cfg["bf16"],
            predict_with_generate=True,
            logging_steps=10,
            save_strategy="epoch",
        ),
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"checkpoint -> {output_dir}")


if __name__ == "__main__":
    main()
