#!/usr/bin/env python3
import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset
from rec_rlp.config import load_yaml


def formatting(row):
    return {
        "text": row["prompt"] + "\n" + row["target"] + "</s>",
    }


def main():
    p = argparse.ArgumentParser(description="Глава 3: QLoRA (Unsloth)")
    p.add_argument("--config", default="train_qlora.yaml")
    p.add_argument("--model", type=str)
    p.add_argument("--dataset", type=str)
    p.add_argument("--output-dir", type=str)
    p.add_argument("--epochs", type=int)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    model_name = args.model or cfg["model_name"]
    dataset_path = args.dataset or cfg["dataset"]
    output_dir = args.output_dir or cfg["output_dir"]
    epochs = args.epochs or cfg["num_train_epochs"]

    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=cfg["load_in_4bit"],
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    ds = load_dataset("json", data_files=dataset_path, split="train")
    ds = ds.map(formatting)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=cfg["max_seq_length"],
        args=TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            learning_rate=cfg["learning_rate"],
            warmup_ratio=cfg["warmup_ratio"],
            num_train_epochs=epochs,
            weight_decay=cfg["weight_decay"],
            max_grad_norm=cfg["max_grad_norm"],
            bf16=cfg["bf16"],
            optim=cfg["optim"],
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
