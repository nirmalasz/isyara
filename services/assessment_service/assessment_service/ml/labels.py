import json

from ..config import LABEL_MAP_PATH


DEFAULT_LABELS = [
    "halo",
    "terima_kasih",
    "maaf",
    "tolong",
    "saya",
    "kamu",
    "ya",
    "tidak",
    "makan",
    "minum",
]

DISPLAY_LABELS = {
    "halo": "Halo",
    "terima_kasih": "Terima kasih",
    "maaf": "Maaf",
    "tolong": "Tolong",
    "saya": "Saya",
    "kamu": "Kamu",
    "ya": "Ya",
    "tidak": "Tidak",
    "makan": "Makan",
    "minum": "Minum",
}


def load_labels():
    if LABEL_MAP_PATH.exists():
        return json.loads(LABEL_MAP_PATH.read_text())
    return DEFAULT_LABELS


def display_text(label):
    return DISPLAY_LABELS.get(label, label.replace("_", " ").title())
