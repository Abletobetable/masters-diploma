import json
import random
from pathlib import Path

from rec_rlp.prompts import build_prompt
from rec_rlp.targets import build_target


def load_users(path: str | Path) -> list[dict]:
    users = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                users.append(json.loads(line))
    return users


def sample_users(users: list[dict], n: int, seed: int = 42) -> list[dict]:
    if n >= len(users):
        return users
    rng = random.Random(seed)
    return rng.sample(users, n)


def to_sft_rows(
    users: list[dict],
    lang: str = "en",
    target_mode: str = "combined",
    max_target_items: int = 200,
) -> list[dict]:
    rows = []
    for u in users:
        rows.append(
            {
                "user_id": u["user_id"],
                "prompt": build_prompt(u, lang=lang),
                "target": build_target(u, mode=target_mode, max_items=max_target_items),
            }
        )
    return rows


def save_jsonl(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
