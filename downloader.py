# CiviImport — background model downloads with hash verification
#
# Copyright (c) 2026 Sev Kiriouchine
# SPDX-License-Identifier: MIT
# Part of CiviImport: https://github.com/Kiriouchine/CiviImport

"""CiviImport — download missing models into ComfyUI's folders.  (STEP 4)

Design
------
* The client sends only version IDs (plus an optional API host). The download
  URL, filename, and target folder are all derived SERVER-SIDE from the
  Civitai model-version data — the browser can never point us at arbitrary
  paths or URLs (avoids the path-traversal class of custom-node bugs).
* Downloads run sequentially on a daemon thread: stream to '<name>.part',
  atomic os.replace on completion, partial file removed on failure. After the
  rename, the file is verified against Civitai's published SHA-256/AutoV2
  (when available) — mismatches are deleted and reported, and the computed
  hash is cached so hash matching recognizes the file immediately.
* Progress is pushed as 'civiimport.progress' events (throttled to ~2/s);
  the panel renders bars, flips the lights, and refreshes the model lists.
* Dependency-injected (fetch_version / send / folder_roots) so the whole
  module is testable outside ComfyUI — see test_downloader.py.
"""

import os
import shutil
import threading
import time

import requests

try:
    import folder_paths  # only inside ComfyUI
    HAVE_COMFY = True
except ImportError:
    folder_paths = None
    HAVE_COMFY = False

try:
    from . import hashing  # package context (inside ComfyUI)
except ImportError:
    import hashing  # standalone tests

USER_AGENT = "CiviImport (ComfyUI extension)"
CHUNK = 1 << 20  # 1 MiB

_lock = threading.Lock()
_active: set = set()  # version_ids currently queued or downloading


class DownloadError(Exception):
    """User-facing download failure."""


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _target_dir(folder_key: str, folder_roots: dict | None = None,
                dir_hint: str | None = None) -> str:
    if folder_roots is not None:  # test override
        d = folder_roots.get(folder_key)
        if not d:
            raise DownloadError(f"No target folder configured for '{folder_key}'.")
        return d
    if not HAVE_COMFY:
        raise DownloadError("folder_paths unavailable — downloads only work inside ComfyUI.")
    paths = [p for p in (folder_paths.get_folder_paths(folder_key) or []) if os.path.isdir(p)]
    if not paths:
        raise DownloadError(f"No existing models folder for type '{folder_key}'.")
    if dir_hint:
        needle = dir_hint.casefold()
        for p in paths:
            if needle in os.path.abspath(p).casefold():
                return p
    return paths[0]


def _safe_name(name: str) -> str:
    base = os.path.basename((name or "").strip().replace("\\", "/"))
    if not base or base in (".", "..") or base.startswith("."):
        raise DownloadError(f"Refusing suspicious file name {name!r}.")
    return base


def _pick_file(version: dict) -> dict:
    files = version.get("files") or []
    for f in files:
        if f.get("primary"):
            return f
    for f in files:
        if (f.get("type") or "").lower() == "model":
            return f
    if files:
        return files[0]
    raise DownloadError("This version has no downloadable files.")


def _verify_download(path: str, hashes: dict, emit, name: str, folder_key: str) -> str | None:
    """Post-download integrity check against Civitai's published hashes. (STEP 6)

    Uses the full SHA-256 when published, else the AutoV2 prefix (the first 10
    hex chars of the SHA-256). The computed hash lands in the shared cache, so
    a fresh download is immediately recognizable by hash matching too. On a
    mismatch the file is deleted and DownloadError raised — a corrupt or
    truncated model never survives under its real name. Returns a short note
    for the 'done' event, or None when the version publishes no hash."""
    want_sha = want_auto = None
    for k, v in (hashes or {}).items():
        if not isinstance(v, str) or not v.strip():
            continue
        kl = (k or "").lower()
        if kl == "sha256":
            want_sha = v.strip().upper()
        elif kl == "autov2":
            want_auto = v.strip().upper()
    if not (want_sha or want_auto):
        return None

    last = [0.0]

    def cb(done, total):
        now = time.monotonic()
        if now - last[0] >= 0.5:
            last[0] = now
            emit("verifying", filename=name, folder=folder_key,
                 pct=round(done * 100 / total, 1) if total else None)

    emit("verifying", filename=name, folder=folder_key, pct=0)
    try:
        got = (hashing.cached_sha256(path, compute=True, progress=cb) or "").upper()
    except OSError as e:
        return f"hash check skipped ({e})"
    if not got:
        return "hash check skipped"
    if (want_sha and got == want_sha) or (not want_sha and want_auto and got.startswith(want_auto)):
        return "sha-256 verified"
    try:
        os.remove(path)
    except OSError:
        pass
    raise DownloadError(
        f"'{name}' failed hash verification after download (got {got[:10]}…, expected "
        f"{(want_sha or want_auto)[:10]}…). The file was removed — it may have been "
        "corrupted in transit or replaced upstream; try downloading again."
    )


# --------------------------------------------------------------------------
# Single download
# --------------------------------------------------------------------------

def download_one(version_id: int, *, api_key, host, fetch_version, send,
                 type_to_folder: dict, folder_roots: dict | None = None,
                 dir_hint: str | None = None):
    """Download one model version. Raises DownloadError with a friendly message."""

    def emit(status: str, **kw):
        send({"version_id": version_id, "status": status, **kw})

    version = fetch_version(version_id, api_key, host)
    model = version.get("model") or {}
    folder_key = type_to_folder.get((model.get("type") or "").strip().lower())
    if not folder_key:
        raise DownloadError(f"Unsupported model type {model.get('type')!r} for download.")

    f = _pick_file(version)
    name = _safe_name(f.get("name"))
    url = f.get("downloadUrl") or version.get("downloadUrl")
    if not url:
        raise DownloadError("No download URL on this version.")

    target_dir = _target_dir(folder_key, folder_roots, dir_hint)
    final_path = os.path.join(target_dir, name)
    if os.path.exists(final_path):
        emit("done", filename=name, folder=folder_key, note="already present")
        return name, folder_key

    total = int((f.get("sizeKB") or 0) * 1024)
    if total and shutil.disk_usage(target_dir).free < total * 1.05:
        raise DownloadError(
            f"Not enough disk space in {target_dir} ({total / 2**30:.1f} GB needed)."
        )

    headers = {"User-Agent": USER_AGENT}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    part = final_path + ".part"
    emit("starting", filename=name, folder=folder_key,
         total_mb=round(total / 2**20, 1) if total else None)
    try:
        with requests.get(url, headers=headers, stream=True,
                          timeout=(15, 120), allow_redirects=True) as r:
            if r.status_code in (401, 403):
                raise DownloadError(
                    f"Civitai refused the download (HTTP {r.status_code}) — this file "
                    "requires an API key (set one in the panel), or a logged-in account tier."
                )
            if r.status_code >= 400:
                raise DownloadError(f"Download failed with HTTP {r.status_code}.")
            if "text/html" in (r.headers.get("Content-Type") or "").lower():
                raise DownloadError(
                    "Civitai returned a web page instead of the file — usually a "
                    "login/consent wall. Set an API key in the panel."
                )
            hdr_total = int(r.headers.get("Content-Length") or 0)
            if hdr_total:
                total = hdr_total
            done = 0
            last = 0.0
            with open(part, "wb") as out:
                for chunk in r.iter_content(chunk_size=CHUNK):
                    if not chunk:
                        continue
                    out.write(chunk)
                    done += len(chunk)
                    now = time.monotonic()
                    if now - last >= 0.5:
                        last = now
                        emit("downloading", filename=name, folder=folder_key,
                             pct=round(done * 100 / total, 1) if total else None,
                             downloaded_mb=round(done / 2**20, 1),
                             total_mb=round(total / 2**20, 1) if total else None)
        os.replace(part, final_path)
    except BaseException:
        try:
            if os.path.exists(part):
                os.remove(part)
        except OSError:
            pass
        raise

    note = _verify_download(final_path, f.get("hashes") or {}, emit, name, folder_key)

    emit("done", filename=name, folder=folder_key,
         total_mb=round((total or done) / 2**20, 1),
         **({"note": note} if note else {}))
    return name, folder_key


# --------------------------------------------------------------------------
# Sequential background worker
# --------------------------------------------------------------------------

def start_downloads(version_ids, *, api_key, host, fetch_version, send,
                    type_to_folder: dict, folder_roots: dict | None = None,
                    dir_hint: str | None = None):
    """Queue versions for sequential download on a daemon thread.
    Returns (accepted_ids, skipped_ids) — skipped are already in flight."""
    accepted, skipped = [], []
    with _lock:
        for vid in version_ids:
            if vid in _active:
                skipped.append(vid)
            else:
                _active.add(vid)
                accepted.append(vid)
    if not accepted:
        return accepted, skipped

    def work():
        for vid in accepted:
            try:
                download_one(vid, api_key=api_key, host=host,
                             fetch_version=fetch_version, send=send,
                             type_to_folder=type_to_folder, folder_roots=folder_roots,
                             dir_hint=dir_hint)
            except Exception as e:
                send({"version_id": vid, "status": "error", "error": str(e)})
            finally:
                with _lock:
                    _active.discard(vid)
        send({"version_id": None, "status": "all_done"})

    threading.Thread(target=work, daemon=True, name="CiviImport-download").start()
    return accepted, skipped
