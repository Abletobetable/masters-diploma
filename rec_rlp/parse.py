import re


def parse_item_ids(text: str, limit: int = 200) -> list[str]:
    nums = re.findall(r"\d+", text)
    seen, out = set(), []
    for n in nums:
        if n in seen:
            continue
        seen.add(n)
        out.append(n)
        if len(out) >= limit:
            break
    return out


def merge_chains(chains: list[list[str]], limit: int = 200) -> list[str]:
    seen, out = set(), []
    for chain in chains:
        for item in chain:
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
            if len(out) >= limit:
                return out
    return out
