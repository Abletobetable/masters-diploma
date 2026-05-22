TARGET_MODES = ("wishlist", "similar_items", "similar_users", "combined")


def _uniq(items: list[int | str], limit: int) -> list[str]:
    seen, out = set(), []
    for x in items:
        s = str(x)
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= limit:
            break
    return out


def build_target(user: dict, mode: str = "combined", max_items: int = 200) -> str:
    base = list(user.get("wishlist", [])) + list(user.get("cart", []))
    similar_items = list(user.get("similar_items", []))
    similar_users = list(user.get("similar_users_items", []))

    if mode == "wishlist":
        pool = base
    elif mode == "similar_items":
        pool = base + similar_items
    elif mode == "similar_users":
        pool = base + similar_users
    elif mode == "combined":
        pool = base + similar_items + similar_users
    else:
        raise ValueError(f"unknown target mode: {mode}")

    return ", ".join(_uniq(pool, max_items))
