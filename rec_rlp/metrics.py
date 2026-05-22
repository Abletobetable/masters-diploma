from typing import Iterable


def _topk(pred: list[str], k: int) -> list[str]:
    return pred[:k]


def hit_rate(pred: list[str], rel: set[str], k: int = 200) -> float:
    top = _topk(pred, k)
    return float(any(p in rel for p in top)) if rel else 0.0


def precision(pred: list[str], rel: set[str], k: int = 200) -> float:
    top = _topk(pred, k)
    if not top:
        return 0.0
    return sum(1 for p in top if p in rel) / len(top)


def recall(pred: list[str], rel: set[str], k: int = 200) -> float:
    if not rel:
        return 0.0
    top = _topk(pred, k)
    return sum(1 for p in top if p in rel) / len(rel)


def revenue(
    pred: list[str],
    rel: set[str],
    prices: dict[str, float],
    k: int = 200,
) -> float:
    top = _topk(pred, k)
    return sum(prices.get(p, 0.0) for p in top if p in rel)


def recall_by_category(
    pred: list[str],
    rel: set[str],
    item_category: dict[str, str],
    k: int = 200,
) -> float:
    if not rel:
        return 0.0
    rel_cats = {item_category.get(i) for i in rel if item_category.get(i)}
    if not rel_cats:
        return 0.0
    top = _topk(pred, k)
    pred_cats = {item_category.get(p) for p in top if item_category.get(p)}
    return float(bool(rel_cats & pred_cats))


def compute_all(
    pred: list[str],
    rel: Iterable[str | int],
    *,
    k: int = 200,
    prices: dict[str, float] | None = None,
    item_category: dict[str, str] | None = None,
) -> dict[str, float]:
    rel_set = {str(x) for x in rel}
    prices = prices or {}
    item_category = item_category or {}
    return {
        "hit_rate": hit_rate(pred, rel_set, k),
        "precision": precision(pred, rel_set, k),
        "recall": recall(pred, rel_set, k),
        "revenue": revenue(pred, rel_set, prices, k),
        "recall_by_category": recall_by_category(pred, rel_set, item_category, k),
    }


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys}
