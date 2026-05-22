#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rec_rlp.config import load_yaml
from rec_rlp.dataset import load_jsonl
from rec_rlp.inference import generate


def main():
    p = argparse.ArgumentParser(description="Главы 4–5: инференс")
    p.add_argument("--config", default="infer.yaml")
    p.add_argument("--model-path", type=str)
    p.add_argument("--backend", choices=["vllm", "hf"])
    p.add_argument("--num-chains", type=int)
    p.add_argument("--dataset", type=str)
    p.add_argument("--output", type=str)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    if args.model_path:
        cfg["model_path"] = args.model_path
    if args.backend:
        cfg["backend"] = args.backend
    if args.num_chains:
        cfg["num_chains"] = args.num_chains
    if args.dataset:
        cfg["dataset"] = args.dataset
    if args.output:
        cfg["output"] = args.output

    rows = load_jsonl(cfg["dataset"])
    prompts = [r["prompt"] for r in rows]

    preds = generate(prompts, cfg)

    out_rows = []
    for row, pred in zip(rows, preds):
        out_rows.append({**row, "prediction": pred})

    out_path = Path(cfg["output"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"predictions -> {out_path}")


if __name__ == "__main__":
    main()
