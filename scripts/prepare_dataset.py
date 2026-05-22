#!/usr/bin/env python3
import argparse
import random
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rec_rlp.config import load_yaml
from rec_rlp.dataset import load_users, sample_users, save_jsonl, to_sft_rows


def main():
    p = argparse.ArgumentParser(description="Глава 2: промпт-датасет")
    p.add_argument("--config", default="data.yaml")
    p.add_argument("--lang", choices=["ru", "en"])
    p.add_argument("--target-mode", choices=["wishlist", "similar_items", "similar_users", "combined"])
    p.add_argument("--n-users", type=int)
    p.add_argument("--raw", type=str)
    p.add_argument("--split", choices=["train", "val", "all"], default="all")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    lang = args.lang or cfg["lang"]
    target_mode = args.target_mode or cfg["target_mode"]
    n_users = args.n_users
    raw_path = args.raw or cfg["raw_users"]

    users = load_users(raw_path)
    if n_users:
        users = sample_users(users, n_users, seed=cfg["seed"])

    val_ratio = 1.0 - cfg["splits"]["train"]
    rng = random.Random(cfg["seed"])
    rng.shuffle(users)
    n_val = max(1, int(len(users) * val_ratio)) if val_ratio > 0 and len(users) > 1 else 0
    val_users = users[:n_val]
    train_users = users[n_val:] if n_val else users

    def rows_for(split_users):
        return to_sft_rows(
            split_users,
            lang=lang,
            target_mode=target_mode,
            max_target_items=cfg["max_target_items"],
        )

    def save_split(name: str, split_users: list[dict]) -> None:
        tag = f"{name}_{lang}_{target_mode}"
        if n_users:
            tag += f"_{n_users // 1000}k"
        out = Path(cfg["output_dir"]) / f"{tag}.jsonl"
        save_jsonl(rows_for(split_users), out)
        print(f"saved {len(split_users)} users -> {out}")

    if args.split == "all":
        save_split("train", train_users)
        if val_users:
            save_split("val", val_users)
        return

    split_users = train_users if args.split == "train" else val_users
    save_split(args.split, split_users)


if __name__ == "__main__":
    main()
