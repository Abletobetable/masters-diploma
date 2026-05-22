#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rec_rlp.config import load_yaml

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    p = argparse.ArgumentParser(description="Матрица экспериментов из диплома")
    p.add_argument("--stage", choices=["prepare", "train", "infer", "eval", "all"], default="all")
    args = p.parse_args()

    exp = load_yaml("experiments.yaml")

    if args.stage in ("prepare", "all"):
        for item in exp["prepare"]:
            run(
                [
                    PY,
                    "scripts/prepare_dataset.py",
                    "--lang",
                    item["lang"],
                    "--target-mode",
                    item["target_mode"],
                    "--n-users",
                    str(item["n_users"]),
                ]
            )

    if args.stage in ("train", "all"):
        for item in exp["train_qlora"]:
            run(
                [
                    PY,
                    "scripts/train_lora.py",
                    "--model",
                    item["model"],
                    "--dataset",
                    item["dataset"],
                    "--output-dir",
                    f"outputs/{item['model'].split('/')[-1]}",
                ]
            )
        run([PY, "scripts/train_p5.py"])

    if args.stage in ("infer", "all"):
        for item in exp["infer_benchmark"]:
            run(
                [
                    PY,
                    "scripts/infer.py",
                    "--model-path",
                    item["model_path"],
                    "--backend",
                    item["backend"],
                    "--num-chains",
                    str(item["num_chains"]),
                ]
            )
            run(
                [
                    PY,
                    "scripts/benchmark_infer.py",
                    "--config",
                    "infer.yaml",
                ]
            )

    if args.stage in ("eval", "all"):
        run([PY, "scripts/eval_metrics.py"])


if __name__ == "__main__":
    main()
