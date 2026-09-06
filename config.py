# CiviImport — on-disk settings (API key, download directory hint)
#
# Copyright (c) 2026 Sev Kiriouchine
# SPDX-License-Identifier: MIT
# Part of CiviImport: https://github.com/Kiriouchine/CiviImport

"""CiviImport — configuration storage.

The Civitai API key is stored in config.json next to this file (gitignored).
The CIVITAI_API_KEY environment variable is used as a fallback, matching the
convention used by Civicomfy and other Civitai tooling.
"""

import json
import os
import threading

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

_lock = threading.Lock()


def load() -> dict:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"[CiviImport] Could not read config.json ({e}); using defaults.")
        return {}


def save(cfg: dict) -> None:
    with _lock:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)


def get_api_key() -> str | None:
    """Config file first, then environment variable. Returns None if unset."""
    key = (load().get("api_key") or "").strip()
    if key:
        return key
    env = (os.environ.get("CIVITAI_API_KEY") or "").strip()
    return env or None


def get_api_key_source() -> str | None:
    if (load().get("api_key") or "").strip():
        return "config"
    if (os.environ.get("CIVITAI_API_KEY") or "").strip():
        return "env"
    return None


def set_api_key(key: str) -> None:
    """Set the key; an empty string clears it."""
    cfg = load()
    key = (key or "").strip()
    if key:
        cfg["api_key"] = key
    else:
        cfg.pop("api_key", None)
    save(cfg)


def masked_key(key: str | None) -> str | None:
    """Never expose the full key back to the browser."""
    if not key:
        return None
    if len(key) <= 8:
        return "•" * len(key)
    return f"{key[:4]}…{key[-4:]}"
