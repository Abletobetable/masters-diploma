#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rec_rlp.config import load_yaml
from rec_rlp.dataset import load_jsonl


def main():
    p = argparse.ArgumentParser(description="Глава 5: сек/пользователь")
    p.add_argument("--config", default="infer.yaml")
    p.add_argument("--n-users", type=int)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    bench = cfg["benchmark"]
    n_users = args.n_users or bench["n_users"]

    rows = load_jsonl(cfg["dataset"])[:n_users]
    prompts = [r["prompt"] for r in rows]

    from rec_rlp.inference import generate

    run_cfg = {**cfg, "num_chains": cfg.get("num_chains", 1)}

    warmup = bench["warmup"]
    if warmup:
        generate(prompts[:warmup], run_cfg)

    t0 = time.perf_counter()
    generate(prompts, run_cfg)
    elapsed = time.perf_counter() - t0

    sec_per_user = elapsed / len(prompts)
    print(f"users={len(prompts)} elapsed={elapsed:.2f}s sec_per_user={sec_per_user:.4f}")


if __name__ == "__main__":
    main()
