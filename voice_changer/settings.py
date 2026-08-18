"""Persistent settings stored as JSON in the user home dir."""

import json
import os

PATH = os.path.join(os.path.expanduser("~"), ".voice_changer.json")

DEFAULTS = {
    "input_device_name": "",
    "output_device_name": "",
    "pitch": 0.0,
    "formant": 1.0,
    "gate_db": -50.0,
    "gain_db": 0.0,
    "dry_wet": 1.0,
    "safety_lock": True,
    "last_preset": "",
}


def load() -> dict:
    data = dict(DEFAULTS)
    try:
        with open(PATH, "r", encoding="utf-8") as f:
            saved = json.load(f)
        if isinstance(saved, dict):
            data.update({k: v for k, v in saved.items() if k in DEFAULTS})
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return data


def save(data: dict) -> bool:
    try:
        merged = dict(DEFAULTS)
        merged.update({k: v for k, v in data.items() if k in DEFAULTS})
        with open(PATH, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)
        return True
    except OSError:
        return False
