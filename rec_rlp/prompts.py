LANGS = ("ru", "en")

_TEMPLATES = {
    "ru": (
        "Пользователь: {user_id}, покупательская способность: {spending_power}, "
        "местоположение: {location}, пол: {gender}, возраст: {age}, "
        "история покупок: {history}.\n"
        "Сгенерируйте список товаров, которые, скорее всего, заинтересуют данного пользователя."
    ),
    "en": (
        "User: {user_id}, spending power: {spending_power}, location: {location}, "
        "gender: {gender}, age: {age}, purchase history: {history}.\n"
        "Generate a list of items this user is likely to be interested in."
    ),
}


def format_history(item_ids: list[int | str]) -> str:
    return ", ".join(str(x) for x in item_ids)


def build_prompt(user: dict, lang: str = "en") -> str:
    if lang not in _TEMPLATES:
        raise ValueError(f"lang must be one of {LANGS}")
    return _TEMPLATES[lang].format(
        user_id=user["user_id"],
        spending_power=user.get("spending_power", 0),
        location=user.get("location", 0),
        gender=user.get("gender", 0),
        age=user.get("age", 0),
        history=format_history(user.get("purchase_history", [])),
    )
