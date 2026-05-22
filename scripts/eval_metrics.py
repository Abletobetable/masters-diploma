#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rec_rlp.config import load_yaml
from rec_rlp.dataset import load_jsonl
from rec_rlp.metrics import aggregate, compute_all
from rec_rlp.parse import parse_item_ids


def load_map(path: str | None) -> dict:
    if not path or not Path(path).exists():
        return {}
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k): v for k, v in raw.items()}


def main():
    p = argparse.ArgumentParser(description="Глава 4.1: метрики")
    p.add_argument("--config", default="eval.yaml")
    p.add_argument("--predictions", type=str)
    p.add_argument("--k", type=int)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    pred_path = args.predictions or cfg["predictions"]
    k = args.k or cfg["k"]

    prices = load_map(cfg.get("prices"))
    categories = load_map(cfg.get("categories"))
    price_f = {k: float(v) for k, v in prices.items()}
    cat_f = {k: str(v) for k, v in categories.items()}

    rows = load_jsonl(pred_path)
    per_user = []
    for row in rows:
        pred = parse_item_ids(row["prediction"], k)
        rel = parse_item_ids(row["target"], k)
        per_user.append(
            compute_all(
                pred,
                rel,
                k=k,
                prices=price_f,
                item_category=cat_f,
            )
        )

    result = aggregate(per_user)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
