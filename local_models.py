# CiviImport — local model presence checks and hash matching
#
# Copyright (c) 2026 Sev Kiriouchine
# SPDX-License-Identifier: MIT
# Part of CiviImport: https://github.com/Kiriouchine/CiviImport

"""CiviImport — check whether a recipe's resources exist locally.

Inside ComfyUI this uses folder_paths, so it respects every configured model
location (including shared folders wired up via extra_model_paths.yaml).

Outside ComfyUI (standalone testing) it falls back to scanning
$CIVIIMPORT_MODELS_ROOT/<folder> if that environment variable is set, e.g.
CIVIIMPORT_MODELS_ROOT=C:\\Users\\maste\\ComfyUI-Shared\\models

Step 2 matches by filename (case-insensitive basename). Hash matching for
renamed files is planned for step 5 — the structure below leaves room for it.
"""

import os

try:
    import folder_paths  # available only inside ComfyUI
    HAVE_COMFY = True
except ImportError:
    folder_paths = None
    HAVE_COMFY = False

try:
    from . import hashing  # package context (inside ComfyUI)
except ImportError:
    import hashing  # standalone tests

_MODEL_EXTS = (".safetensors", ".ckpt", ".pt", ".pth", ".sft", ".gguf", ".bin")


def get_full_path(folder_key: str, rel: str) -> str | None:
    """Absolute path for a listed relative entry, or None."""
    if HAVE_COMFY:
        try:
            return folder_paths.get_full_path(folder_key, rel)
        except Exception:
            return None
    root = os.environ.get("CIVIIMPORT_MODELS_ROOT")
    if not root:
        return None
    p = os.path.join(root, folder_key, rel.replace("/", os.sep))
    return p if os.path.isfile(p) else None


def _standalone_listing(folder_key: str) -> list[str]:
    root = os.environ.get("CIVIIMPORT_MODELS_ROOT")
    if not root:
        return []
    base = os.path.join(root, folder_key)
    if not os.path.isdir(base):
        return []
    out = []
    for dirpath, _dirs, files in os.walk(base):
        for f in files:
            if f.lower().endswith(_MODEL_EXTS):
                rel = os.path.relpath(os.path.join(dirpath, f), base)
                out.append(rel.replace(os.sep, "/"))
    return out


def get_local_files(folder_key: str) -> list[str]:
    """Relative paths as ComfyUI's loader widgets expect them (subfolders included)."""
    if HAVE_COMFY:
        try:
            return folder_paths.get_filename_list(folder_key)
        except Exception:
            return []
    return _standalone_listing(folder_key)


def pick_upscaler_file(prefer: str = "remacri") -> str | None:
    """First local upscale model for a synthesized hires chain (step 7):
    prefer a filename containing `prefer` (Remacri, the reference workflow's
    model), else the first listed. None when none exist."""
    names = get_local_files("upscale_models")
    for n in names:
        if prefer in n.lower():
            return n
    return names[0] if names else None


def find_local_file(folder_key: str | None, filename: str | None) -> str | None:
    """Case-insensitive basename match. Returns the listed relative path
    (that exact string is what goes into a loader widget later), or None."""
    if not folder_key or not filename:
        return None
    target = os.path.basename(filename).lower()
    for entry in get_local_files(folder_key):
        if os.path.basename(entry).lower() == target:
            return entry
    return None


def _wanted_hashes(res: dict) -> tuple[str | None, str | None]:
    hashes = ((res.get("file") or {}).get("hashes")) or {}
    sha = hashes.get("SHA256") or hashes.get("sha256")
    auto = hashes.get("AutoV2") or hashes.get("autov2")
    return (sha.upper() if isinstance(sha, str) and sha else None,
            auto.upper() if isinstance(auto, str) and auto else None)


def _match_by_hash(folder_key: str, res: dict, cache_snapshot: dict) -> str | None:
    """Cache-only hash match (never computes) — recognizes renamed files
    that were hashed by a previous deep scan."""
    want_sha, want_auto = _wanted_hashes(res)
    if not (want_sha or want_auto):
        return None
    for rel in get_local_files(folder_key):
        full = get_full_path(folder_key, rel)
        if not full:
            continue
        ent = cache_snapshot.get(os.path.abspath(full))
        if not ent:
            continue
        try:
            st = os.stat(full)
        except OSError:
            continue
        if ent.get("size") != st.st_size or ent.get("mtime") != int(st.st_mtime):
            continue
        sha = (ent.get("sha256") or "").upper()
        if not sha:
            continue
        if (want_sha and sha == want_sha) or (want_auto and sha.startswith(want_auto)):
            return rel
    return None


def local_status_for_file(folder_key: str | None, file_info: dict | None) -> dict:
    """Presence check for a single {name, hashes} file dict (the model-page
    card). Same match order as annotate_local_status: filename, cached hash."""
    fname = (file_info or {}).get("name")
    if not folder_key or not fname:
        return {"present": False, "filename": None, "method": None,
                "note": "unknown type or no file info"}
    match = find_local_file(folder_key, fname)
    method = "filename"
    if not match:
        match = _match_by_hash(folder_key, {"file": file_info}, hashing.cache_snapshot())
        method = "hash"
    if match:
        return {"present": True, "filename": match, "method": method}
    return {"present": False, "filename": None, "method": None}


def annotate_local_status(recipe: dict) -> dict:
    """Mutates the recipe: adds resource['local'] and recipe['missing_count'].
    Match order: exact filename, then cached hash (renamed files)."""
    cache_snapshot = hashing.cache_snapshot()
    missing = 0
    for res in recipe.get("resources", []):
        if res.get("system"):
            res["local"] = {"present": False, "filename": None, "method": None,
                            "note": "internal Civitai resource — skipped"}
            continue  # deliberately not counted as missing
        folder = res.get("folder")
        fname = (res.get("file") or {}).get("name")
        if folder is None or fname is None:
            res["local"] = {"present": False, "filename": None, "method": None,
                            "note": "unknown type or no file info"}
            missing += 1
            continue
        match = find_local_file(folder, fname)
        method = "filename"
        if not match:
            match = _match_by_hash(folder, res, cache_snapshot)
            method = "hash"
        if match:
            res["local"] = {"present": True, "filename": match, "method": method}
        else:
            res["local"] = {"present": False, "filename": None, "method": None}
            missing += 1
    recipe["missing_count"] = missing
    recipe["local_check"] = "folder_paths" if HAVE_COMFY else (
        "standalone_root" if os.environ.get("CIVIIMPORT_MODELS_ROOT") else "unavailable"
    )
    return recipe


def deep_hash_scan(folder_keys, progress=None) -> dict:
    """Compute+cache SHA-256 for every local file that isn't cached yet.
    Runs on a background thread from the route; progress(payload) is called
    per chunk / per file. Returns {'hashed': n, 'candidates': m}."""
    todo = []
    for key in folder_keys:
        for rel in get_local_files(key):
            full = get_full_path(key, rel)
            if full and hashing.cached_sha256(full, compute=False) is None:
                todo.append((key, rel, full))
    hashed = 0
    for i, (_key, rel, full) in enumerate(todo):
        def cb(done, total, _rel=rel, _i=i):
            if progress:
                progress({"status": "hashing", "file": _rel, "index": _i + 1,
                          "total": len(todo),
                          "pct": round(done * 100 / total, 1) if total else None})
        try:
            hashing.cached_sha256(full, compute=True, progress=cb)
            hashed += 1
        except OSError as e:
            if progress:
                progress({"status": "error", "file": rel, "error": str(e)})
    if progress:
        progress({"status": "done", "hashed": hashed, "total": len(todo)})
    return {"hashed": hashed, "candidates": len(todo)}
