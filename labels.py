import re


CLASS_NAMES = ["Fishing", "Cargo", "Passenger"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}
CLASS_KEYWORDS = [
    ("fishing", 0),
    ("cargo", 1),
    ("passenger", 2),
]


def normalize_text(value):
    return str(value).strip().lower()


def is_blank(value):
    text = normalize_text(value)
    return text in {"", "nan", "none", "<na>"}


def match_ship_class(value):
    if is_blank(value):
        return None

    text = normalize_text(value)
    text = re.sub(r"[_/-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if text in {"non fishing", "nonfishing", "not fishing"}:
        return None

    for keyword, class_id in CLASS_KEYWORDS:
        if re.search(rf"\b{keyword}\b", text):
            return class_id
    return None


def map_ship_class(row):
    # The raw category is the primary experiment label. If it exists but is
    # outside the three target classes, the sample is excluded from YOLO.
    category_value = row.get("category", "")
    if not is_blank(category_value):
        return match_ship_class(category_value)

    for column in ["Elaborated_type", "Ship_Type", "gfw_shiptype", "gfw_geartype"]:
        class_id = match_ship_class(row.get(column, ""))
        if class_id is not None:
            return class_id

    return None
