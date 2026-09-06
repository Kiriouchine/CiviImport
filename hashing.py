# CiviImport — cached SHA-256 hashing of local model files
#
# Copyright (c) 2026 Sev Kiriouchine
# SPDX-License-Identifier: MIT
# Part of CiviImport: https://github.com/Kiriouchine/CiviImport

"""CiviImport — local file hashing with a persistent cache.  (STEP 5)

Civitai identifies files by SHA-256 (full) and AutoV2 (the first 10 hex chars
of the SHA-256), so one local hash covers both. Hashing multi-GB checkpoints
is slow, so results are cached in hash_cache.json next to this file and
validated by size+mtime; nothing is ever re-hashed unless the file changed.

Reads are cache-only by default — computing happens exclusively in the
explicit deep scan (see local_models.deep_hash_scan), never as a surprise.
"""

import hashlib
import json
import os
import threading

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hash_cache.json")

_lock = threading.Lock()


def _load() -> dict:
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def _save(cache: dict) -> None:
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    os.replace(tmp, CACHE_PATH)


def cache_snapshot() -> dict:
    with _lock:
        return dict(_load())


def sha256_of(path: str, chunk: int = 1 << 20, progress=None) -> str:
    h = hashlib.sha256()
    total = os.path.getsize(path)
    done = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
            done += len(b)
            if progress:
                progress(done, total)
    return h.hexdigest().upper()


def cached_sha256(path: str, compute: bool = False, progress=None) -> str | None:
    """Cached hash if the entry is still valid; compute+store when compute=True."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = os.path.abspath(path)
    with _lock:
        ent = _load().get(key)
    if ent and ent.get("size") == st.st_size and ent.get("mtime") == int(st.st_mtime):
        return ent.get("sha256")
    if not compute:
        return None
    digest = sha256_of(path, progress=progress)
    with _lock:
        cache = _load()
        cache[key] = {"sha256": digest, "size": st.st_size, "mtime": int(st.st_mtime)}
        _save(cache)
    return digest
