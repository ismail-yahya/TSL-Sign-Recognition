"""
Load label mappings from SignList_ClassId_TR_EN.csv for prediction display.

Supports two CSV formats:
  - SignList_ClassId_TR_EN.csv: ClassId, TR, EN
  - labels.csv (legacy): class_id, turkish_word, english_word
"""

import csv
from pathlib import Path

DEFAULT_LABELS_FILE = "SignList_ClassId_TR_EN.csv"
FALLBACK_LABELS_FILE = "labels.csv"


def load_labels(csv_path: str | Path = None) -> tuple[dict[int, str], dict[int, str]]:
    """
    Load labels from CSV and return mappings for class_id -> turkish_word and class_id -> english_word.

    Args:
        csv_path: Path to CSV. If None, uses data/SignList_ClassId_TR_EN.csv (fallback: labels.csv).

    Returns:
        Tuple of (id_to_turkish, id_to_english) dictionaries.
    """
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"

    if csv_path is None:
        csv_path = data_dir / DEFAULT_LABELS_FILE
        if not csv_path.exists():
            csv_path = data_dir / FALLBACK_LABELS_FILE
    else:
        csv_path = Path(csv_path)

    if not csv_path.exists():
        return {}, {}

    id_to_turkish: dict[int, str] = {}
    id_to_english: dict[int, str] = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        header_lower = [h.lower() for h in header] if header else []

        if "classid" in header_lower and "tr" in header_lower and "en" in header_lower:
            class_key = "ClassId"
            tr_key = "TR"
            en_key = "EN"
        else:
            class_key = "class_id"
            tr_key = "turkish_word"
            en_key = "english_word"

        for row in reader:
            try:
                class_id = int(row[class_key])
                id_to_turkish[class_id] = row[tr_key].strip()
                id_to_english[class_id] = row[en_key].strip()
            except (KeyError, ValueError):
                continue

    return id_to_turkish, id_to_english
