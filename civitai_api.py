# CiviImport — Civitai URL parsing, fetching and recipe normalisation
#
# Copyright (c) 2026 Sev Kiriouchine
# SPDX-License-Identifier: MIT
# Part of CiviImport: https://github.com/Kiriouchine/CiviImport

"""CiviImport — Civitai API client and recipe normalizer.

Two-hop resolution of a Civitai image URL:

  hop 1:  GET /api/v1/images?imageId={id}&withMeta=true
          -> generation meta (prompt, sampler, cfg, seed, clip skip, size)
             and meta.civitaiResources (list of {type, weight, modelVersionId})
  hop 2:  GET /api/v1/model-versions/{versionId}   (once per resource)
          -> model name/type, base model, trained words, exact file name,
             hashes and downloadUrl

The result is a normalized "recipe" dict that later steps (graph builder,
downloader) consume. Model-page URLs (/models/...) resolve too: the model
itself plus a starter recipe imported from the version's showcase gallery
(see resolve_model_page). This module is deliberately free of ComfyUI
imports so it can be tested standalone (see test_resolver.py).
"""

import json
import re
from urllib.parse import quote

import requests

API_BASE = "https://civitai.com/api/v1"  # kept for reference; hosts are chosen per URL
USER_AGENT = "CiviImport (ComfyUI extension)"
TIMEOUT = 20  # seconds per request

# Since the April 2026 split, civitai.com is the SFW-only view (the role
# civitai.green used to play) and civitai.red shows everything. Same database,
# same image IDs — but content above PG is invisible through the .com API.
DEFAULT_HOST = "civitai.com"
RED_HOST = "civitai.red"

IMAGE_URL_RE = re.compile(r"civitai\.(com|green|red)/images/(\d+)", re.IGNORECASE)
MODEL_URL_RE = re.compile(r"civitai\.(com|green|red)/models/(\d+)", re.IGNORECASE)
DOWNLOAD_URL_RE = re.compile(r"civitai\.(com|green|red)/api/download/models/(\d+)", re.IGNORECASE)
VERSION_QUERY_RE = re.compile(r"[?&]modelVersionId=(\d+)", re.IGNORECASE)


class CivitaiError(Exception):
    """User-facing error with a friendly message."""


# --------------------------------------------------------------------------
# URL parsing
# --------------------------------------------------------------------------

def parse_image_ref(url_or_id: str) -> tuple[int, str]:
    """Returns (image_id, api_host). Bare IDs default to civitai.com."""
    s = (url_or_id or "").strip()
    if not s:
        raise CivitaiError("Please paste a Civitai image URL.")
    if s.isdigit():
        return int(s), DEFAULT_HOST
    m = IMAGE_URL_RE.search(s)
    if not m:
        raise CivitaiError(
            "That doesn't look like a Civitai image URL. Expected something like "
            "https://civitai.com/images/123456 (civitai.green and civitai.red work too)."
        )
    tld = m.group(1).lower()
    host = RED_HOST if tld == "red" else DEFAULT_HOST  # green = SFW mirror of .com
    return int(m.group(2)), host


def parse_image_id(url_or_id: str) -> int:
    return parse_image_ref(url_or_id)[0]


def _host_from_tld(tld: str) -> str:
    return RED_HOST if (tld or "").lower() == "red" else DEFAULT_HOST  # green = SFW mirror


def parse_any_ref(url_or_id: str) -> dict:
    """Classify a pasted reference.  (STEP 6)

    Returns {"kind": "image", "image_id", "host"} for image URLs and bare
    numbers (backward compatible), or {"kind": "model", "model_id",
    "version_id", "host"} for model pages and API download links.
    model_id is None for download links (recovered later from the version);
    version_id is None when the page URL has no ?modelVersionId= (the page's
    default version is used).
    """
    s = (url_or_id or "").strip()
    if not s:
        raise CivitaiError("Please paste a Civitai image or model URL.")
    if s.isdigit() or IMAGE_URL_RE.search(s):
        image_id, host = parse_image_ref(s)
        return {"kind": "image", "image_id": image_id, "host": host}
    m = DOWNLOAD_URL_RE.search(s)
    if m:
        return {"kind": "model", "model_id": None,
                "version_id": int(m.group(2)), "host": _host_from_tld(m.group(1))}
    m = MODEL_URL_RE.search(s)
    if m:
        vq = VERSION_QUERY_RE.search(s)
        return {"kind": "model", "model_id": int(m.group(2)),
                "version_id": int(vq.group(1)) if vq else None,
                "host": _host_from_tld(m.group(1))}
    raise CivitaiError(
        "That doesn't look like a Civitai image or model URL. Expected something like "
        "https://civitai.com/images/123456 or https://civitai.com/models/12345 "
        "(civitai.green and civitai.red work too)."
    )


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

def _get(url: str, api_key: str | None = None, params: dict | None = None) -> dict:
    headers = {"User-Agent": USER_AGENT}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        r = requests.get(url, headers=headers, params=params, timeout=TIMEOUT)
    except requests.exceptions.Timeout:
        raise CivitaiError("Civitai did not respond in time. Try again in a moment.")
    except requests.exceptions.ConnectionError:
        raise CivitaiError("Could not reach civitai.com. Check your internet connection.")

    if r.status_code in (401, 403):
        raise CivitaiError(
            "Civitai refused the request (HTTP %d). This usually means an API key is "
            "required or the current key is invalid. Set one with the 'Set API key' button."
            % r.status_code
        )
    if r.status_code == 404:
        raise CivitaiError("Civitai returned 404 — the resource does not exist or was removed.")
    if r.status_code == 429:
        raise CivitaiError("Civitai rate limit hit (HTTP 429). Wait a bit and try again.")
    if r.status_code >= 400:
        raise CivitaiError(f"Civitai returned HTTP {r.status_code}.")

    try:
        return r.json()
    except ValueError:
        raise CivitaiError("Civitai returned a non-JSON response (site may be having issues).")


def _api(host: str) -> str:
    return f"https://{host}/api/v1"


def _other_host(host: str) -> str:
    return RED_HOST if host == DEFAULT_HOST else DEFAULT_HOST


def fetch_image_item(image_id: int, api_key: str | None = None,
                     host: str = DEFAULT_HOST) -> dict:
    """Hop 1: single-image lookup with full generation meta."""
    data = _get(
        f"{_api(host)}/images",
        api_key=api_key,
        params={"imageId": image_id, "withMeta": "true"},
    )
    items = data.get("items") or []
    if not items:
        if host == DEFAULT_HOST:
            hint = (" It may be NSFW — since the April 2026 split, such images are only "
                    "visible through civitai.red.")
        elif not api_key:
            hint = " An API key may be required (browsing-level gating on civitai.red)."
        else:
            hint = " Your account's browsing settings on civitai.red may hide this image."
        raise CivitaiError(f"Image {image_id} was not found via the {host} API.{hint}")
    return items[0]


def fetch_model_version(version_id: int, api_key: str | None = None,
                        host: str = DEFAULT_HOST) -> dict:
    """Hop 2: model-version details (files, hashes, download URL)."""
    return _get(f"{_api(host)}/model-versions/{version_id}", api_key=api_key)


def _version_from_trpc(version_id: int, node: dict, host: str) -> dict:
    """Map the site endpoint's model-version payload into REST shape.

    Differences handled: hashes may arrive as a list of {type, hash}; no
    ready downloadUrl, but the canonical pattern is
    https://{host}/api/download/models/{versionId}.
    """
    if not node or not (node.get("files") or node.get("model") or node.get("name")):
        raise CivitaiError(f"Version {version_id}: no data via the {host} site endpoint.")
    dl = f"https://{host}/api/download/models/{version_id}"
    files = []
    for f in node.get("files") or []:
        hashes = f.get("hashes")
        if isinstance(hashes, list):
            hashes = {h.get("type"): h.get("hash")
                      for h in hashes if isinstance(h, dict) and h.get("type")}
        files.append({
            "name": f.get("name"),
            "sizeKB": f.get("sizeKB"),
            "primary": f.get("primary"),
            "type": f.get("type"),
            "hashes": hashes or {},
            "downloadUrl": f.get("downloadUrl") or dl,
        })
    return {
        "id": node.get("id") or version_id,
        "modelId": node.get("modelId"),
        "name": node.get("name"),
        "baseModel": node.get("baseModel"),
        "trainedWords": node.get("trainedWords") or [],
        "model": node.get("model") or {},
        "files": files,
        "downloadUrl": dl,
    }


def fetch_model_version_trpc(version_id: int, api_key: str | None = None,
                             host: str = DEFAULT_HOST) -> dict:
    """Hop-2 fallback via the site's own endpoint (same data the model page shows)."""
    payload = quote(json.dumps({"json": {"id": int(version_id)}}))
    data = _get(f"https://{host}/api/trpc/modelVersion.getById?input={payload}",
                api_key=api_key)
    node = (((data.get("result") or {}).get("data") or {}).get("json")) or {}
    return _version_from_trpc(version_id, node, host)


def fetch_version_by_hash(file_hash: str, api_key: str | None = None,
                          host: str = DEFAULT_HOST) -> dict:
    """Resolve a model version from a file hash (AutoV2 or SHA-256)."""
    return _get(f"{_api(host)}/model-versions/by-hash/{quote(file_hash)}", api_key=api_key)


# --------------------------------------------------------------------------
# Model-page lookups  (STEP 6)
# --------------------------------------------------------------------------

def fetch_model(model_id: int, api_key: str | None = None,
                host: str = DEFAULT_HOST) -> dict:
    """Model lookup: name, creator, and the version list (first = page default)."""
    return _get(f"{_api(host)}/models/{model_id}", api_key=api_key)


def fetch_model_any(model_id: int, api_key: str | None = None,
                    host: str = DEFAULT_HOST) -> dict:
    """Model lookup with cross-host fallback (pasted host first)."""
    last = None
    for h in (host, _other_host(host)):
        try:
            return fetch_model(model_id, api_key=api_key, host=h)
        except CivitaiError as e:
            last = e
    raise CivitaiError(
        f"Model {model_id} was not found via civitai.com or civitai.red ({last}) — "
        "if the page opens in your browser, paste its URL with ?modelVersionId= "
        "(click any version chip) and try again."
    )


def fetch_version_gallery(version_id: int, api_key: str | None = None,
                          host: str = DEFAULT_HOST, limit: int = 20) -> list:
    """Gallery images posted for a version — the fallback showcase source when
    the version payload's own images carry no generation meta (or the version
    itself arrived via the site endpoint, which has no images array)."""
    data = _get(
        f"{_api(host)}/images",
        api_key=api_key,
        params={"modelVersionId": version_id, "limit": limit, "withMeta": "true"},
    )
    return data.get("items") or []


# A1111-style meta type hints -> civitaiResources types
_HASH_TYPE_MAP = {
    "model": "checkpoint", "checkpoint": "checkpoint",
    "lora": "lora", "locon": "locon", "dora": "dora", "lycoris": "lora",
    "embed": "textualinversion", "embedding": "textualinversion",
    "textualinversion": "textualinversion", "vae": "vae",
}


def _hash_entries_from_meta(meta: dict) -> list[dict]:
    """Off-site images carry hashes instead of civitaiResources:
    meta.resources = [{name, type, weight, hash}] and/or
    meta.hashes = {'model': 'abc...', 'lora:name': '...', 'embed:x': '...'}."""
    out, seen = [], set()
    for r in meta.get("resources") or []:
        h = (r.get("hash") or "").strip()
        if not h or h.lower() in seen:
            continue
        seen.add(h.lower())
        out.append({"hash": h, "type_hint": (r.get("type") or "").lower(),
                    "weight": r.get("weight"), "name": r.get("name")})
    for key, h in (meta.get("hashes") or {}).items():
        if not isinstance(h, str) or not h.strip():
            continue
        hl = h.strip().lower()
        if hl in seen:
            continue
        seen.add(hl)
        k = (key or "").lower()
        hint = "model" if k == "model" else k.split(":", 1)[0]
        out.append({"hash": h.strip(), "type_hint": hint, "weight": None,
                    "name": key.split(":", 1)[1] if ":" in key else None})
    return out


def fetch_model_version_any(version_id: int, api_key: str | None = None,
                            host: str = DEFAULT_HOST) -> dict:
    """Generalist hop 2: REST on both hosts, then the site endpoint on both."""
    for h in (host, _other_host(host)):
        try:
            return fetch_model_version(version_id, api_key=api_key, host=h)
        except CivitaiError:
            pass
    for h in (host, _other_host(host)):
        try:
            return fetch_model_version_trpc(version_id, api_key=api_key, host=h)
        except CivitaiError:
            pass
    raise CivitaiError(
        f"Version {version_id} was not found on any Civitai endpoint. This is often one of "
        "Civitai's internal system resources (e.g. the Safe Helper embeddings) that isn't "
        "publicly downloadable — it will be skipped in the workflow."
    )


def _item_from_generation_data(image_id: int, node: dict) -> dict:
    """Map the site endpoint's payload into a hop-1-style item.

    The site's generation-data lookup returns {meta, resources[...]} where each
    resource carries versionId/modelVersionId, modelType, strength, baseModel.
    We synthesize meta.civitaiResources so build_recipe treats both paths the
    same. No uploaded width/height here — but on-site meta usually contains the
    true generation width/height, which _parse_size prefers anyway.
    """
    meta = dict(node.get("meta") or {})
    resources = node.get("resources") or []
    if not meta and not resources:
        raise CivitaiError(f"Image {image_id}: the site endpoint returned no generation data.")

    if not meta.get("civitaiResources"):
        civ_res = []
        for r in resources:
            vid = r.get("modelVersionId") or r.get("versionId") or r.get("id")
            if vid is None:
                continue
            entry = {"modelVersionId": int(vid),
                     "type": ((r.get("modelType") or "").lower() or None)}
            if r.get("strength") is not None:
                entry["weight"] = r.get("strength")
            civ_res.append(entry)
        if civ_res:
            meta["civitaiResources"] = civ_res

    base_model = None
    for r in resources:
        if (r.get("modelType") or "").lower() == "checkpoint":
            base_model = r.get("baseModel")
            break
    if base_model is None and resources:
        base_model = resources[0].get("baseModel")

    return {
        "id": int(image_id),
        "meta": meta,
        "width": None,
        "height": None,
        "baseModel": base_model,
        "username": node.get("username"),
        "postId": None,
    }


def fetch_generation_data(image_id: int, api_key: str | None = None,
                          host: str = DEFAULT_HOST) -> dict:
    """Fallback hop 1 via the site's own generation-data endpoint.

    The documented REST images endpoint fails to return some images (notably
    older ones) even when they are fully public. This endpoint backs the
    image page's "Generation data" panel, so anything viewable on the site
    resolves here.
    """
    payload = quote(json.dumps({"json": {"id": int(image_id)}}))
    data = _get(f"https://{host}/api/trpc/image.getGenerationData?input={payload}",
                api_key=api_key)
    node = (((data.get("result") or {}).get("data") or {}).get("json")) or {}
    return _item_from_generation_data(image_id, node)


# --------------------------------------------------------------------------
# Sampler mapping (Civitai/A1111 names -> ComfyUI sampler + scheduler)
# --------------------------------------------------------------------------

_SAMPLER_BASE = {
    "euler a": ("euler_ancestral", "normal"),
    "euler": ("euler", "normal"),
    "lms": ("lms", "normal"),
    "heun": ("heun", "normal"),
    "dpm2": ("dpm_2", "normal"),
    "dpm2 a": ("dpm_2_ancestral", "normal"),
    "dpm++ 2s a": ("dpmpp_2s_ancestral", "normal"),
    "dpm++ 2m": ("dpmpp_2m", "normal"),
    "dpm++ sde": ("dpmpp_sde", "normal"),
    "dpm++ 2m sde": ("dpmpp_2m_sde", "normal"),
    "dpm++ 2m sde heun": ("dpmpp_2m_sde", "normal"),
    "dpm++ 3m sde": ("dpmpp_3m_sde", "normal"),
    "dpm fast": ("dpm_fast", "normal"),
    "dpm adaptive": ("dpm_adaptive", "normal"),
    "ddim": ("ddim", "normal"),
    "uni_pc": ("uni_pc", "normal"),
    "unipc": ("uni_pc", "normal"),
    "lcm": ("lcm", "sgm_uniform"),
}

_SCHEDULER_SUFFIXES = [
    (" karras", "karras"),
    (" exponential", "exponential"),
    (" sgm uniform", "sgm_uniform"),
]

DEFAULT_SAMPLER = ("euler", "normal")


def map_sampler(civitai_name: str | None) -> dict:
    """Return {'civitai', 'comfy_sampler', 'comfy_scheduler', 'mapped'}."""
    raw = (civitai_name or "").strip()
    name = raw.lower()
    scheduler_override = None
    for suffix, sched in _SCHEDULER_SUFFIXES:
        if name.endswith(suffix):
            scheduler_override = sched
            name = name[: -len(suffix)].strip()
            break

    base = _SAMPLER_BASE.get(name)
    if base is None:
        sampler, scheduler = DEFAULT_SAMPLER
        return {
            "civitai": raw or None,
            "comfy_sampler": sampler,
            "comfy_scheduler": scheduler_override or scheduler,
            "mapped": False,
        }
    sampler, scheduler = base
    return {
        "civitai": raw,
        "comfy_sampler": sampler,
        "comfy_scheduler": scheduler_override or scheduler,
        "mapped": True,
    }


# --------------------------------------------------------------------------
# Resource typing (Civitai model.type -> ComfyUI models folder key)
# --------------------------------------------------------------------------

CIVITAI_TYPE_TO_FOLDER = {
    "checkpoint": "checkpoints",
    "lora": "loras",
    "locon": "loras",
    "dora": "loras",
    "lycoris": "loras",
    "textualinversion": "embeddings",
    "vae": "vae",
    "controlnet": "controlnet",
    "upscaler": "upscale_models",
}


def folder_for_type(model_type: str | None) -> str | None:
    return CIVITAI_TYPE_TO_FOLDER.get((model_type or "").strip().lower())


def _pick_primary_file(files: list) -> dict | None:
    if not files:
        return None
    for f in files:
        if f.get("primary"):
            return f
    for f in files:
        if (f.get("type") or "").lower() == "model":
            return f
    return files[0]


# --------------------------------------------------------------------------
# Recipe building (pure — takes already-fetched data, easy to test offline)
# --------------------------------------------------------------------------

def _sane_dims(w, h) -> bool:
    try:
        return 256 <= int(w) <= 4096 and 256 <= int(h) <= 4096
    except (TypeError, ValueError):
        return False


def _parse_size(meta: dict, item: dict) -> tuple[int | None, int | None, str | None]:
    """Prefer meta['Size'] ('832x1216'), then meta width/height (on-site meta
    carries the true generation size), then the uploaded image's dimensions.
    Degenerate values (below 256 or above 4096 per side) are ignored so a bad
    metadata field can never become a broken latent."""
    size = meta.get("Size") or meta.get("size")
    if isinstance(size, str) and "x" in size:
        try:
            w, h = (int(v) for v in size.lower().split("x"))
            if _sane_dims(w, h):
                return w, h, "meta"
        except ValueError:
            pass
    mw = _num(meta.get("width"), int, None)
    mh = _num(meta.get("height"), int, None)
    if _sane_dims(mw, mh):
        return mw, mh, "meta"
    w, h = item.get("width"), item.get("height")
    if _sane_dims(w, h):
        return int(w), int(h), "image"
    return None, None, None


def _num(value, cast, default=None):
    try:
        return cast(value)
    except (TypeError, ValueError):
        return default


_FAMILY_BUCKETS = {
    # square, landscape, portrait — the standard generation resolutions
    "sdxl": [(1024, 1024), (1216, 832), (832, 1216)],
    "sd15": [(512, 512), (768, 512), (512, 768)],
}


def _model_family(base_model: str | None) -> str:
    bm = (base_model or "").lower()
    if "sd 1" in bm or "sd1" in bm:
        return "sd15"
    # SDXL, Pony, Illustrious, NoobAI, Flux… all use the 1024-class buckets here.
    return "sdxl"


def _snap_to_bucket(width, height, base_model: str | None) -> tuple[int, int]:
    """Nearest standard generation resolution, chosen by orientation."""
    square, landscape, portrait = _FAMILY_BUCKETS[_model_family(base_model)]
    if not width or not height:
        return square
    ratio = width / height
    if ratio >= 1.2:
        return landscape
    if ratio <= 1 / 1.2:
        return portrait
    return square


def _extract_upscale_hint(meta: dict) -> float | None:
    """Factor from A1111-style hires keys, when present (on-site images often omit it)."""
    for key in ("Hires upscale", "hiresUpscale", "upscale", "Upscale"):
        f = _num(meta.get(key), float, None)
        if f and f > 1.0:
            return f
    return None


def _shape_resource(vid: int, info: dict, hint_type: str | None = None,
                    weight=None) -> dict:
    """One recipe resource from a fetched model-version payload.
    Used by build_recipe for every linked resource and by the model-page flow
    to inject the pasted model when a showcase image didn't link it."""
    model = info.get("model") or {}
    model_type = model.get("type") or hint_type
    primary = _pick_primary_file(info.get("files") or [])
    kind = (model_type or "unknown").lower()
    if weight is None and kind in ("lora", "locon", "dora", "lycoris"):
        weight = 1.0
    # Civitai's generator injects internal helper resources (e.g. the Safe
    # Helper embeddings, model 222256) that are not downloadable and not
    # part of the visible prompt — mark them so the UI can grey them out.
    is_system = ("safe helper" in (model.get("name") or "").lower()
                 or info.get("modelId") == 222256)
    return {
        "version_id": vid,
        "model_id": info.get("modelId"),
        "kind": kind,
        "system": is_system,
        "folder": folder_for_type(model_type),
        "model_name": model.get("name"),
        "version_name": info.get("name"),
        "base_model": info.get("baseModel"),
        "weight": weight,
        "trained_words": info.get("trainedWords") or [],
        "air": info.get("air"),
        "file": None if primary is None else {
            "name": primary.get("name"),
            "size_kb": primary.get("sizeKB"),
            "hashes": primary.get("hashes") or {},
            "download_url": primary.get("downloadUrl") or info.get("downloadUrl"),
        },
    }


def build_recipe(item: dict, version_infos: dict[int, dict], source_url: str | None = None) -> dict:
    """Normalize hop-1 item + hop-2 version details into one recipe dict.

    `version_infos` maps modelVersionId -> the /model-versions/{id} response
    (value may be an Exception if that fetch failed; it is reported as a warning).
    """
    meta = item.get("meta") or {}
    warnings: list[str] = []

    # --- resource list ----------------------------------------------------
    entries = []
    civitai_resources = meta.get("civitaiResources") or []
    if civitai_resources:
        for res in civitai_resources:
            vid = res.get("modelVersionId")
            if vid is None:
                continue
            entries.append({"version_id": int(vid),
                            "hint_type": (res.get("type") or "").lower() or None,
                            "weight": res.get("weight")})
    elif item.get("modelVersionIds"):
        # Deduped top-level list; no per-resource type/weight in this path.
        for vid in item["modelVersionIds"]:
            entries.append({"version_id": int(vid), "hint_type": None, "weight": None})
        warnings.append("Resource weights were not present in the metadata; LoRA strengths default to 1.0.")
    else:
        warnings.append(
            "No Civitai-linked resources and no A1111 file hashes in this image's metadata, "
            "so no models could be identified."
        )

    resources = []
    seen = set()
    for entry in entries:
        vid = entry["version_id"]
        if vid in seen:
            continue
        seen.add(vid)

        info = version_infos.get(vid)
        if info is None or isinstance(info, Exception):
            warnings.append(f"Could not fetch details for model version {vid}"
                            + (f": {info}" if info else "."))
            resources.append({
                "version_id": vid,
                "kind": entry["hint_type"] or "unknown",
                "folder": folder_for_type(entry["hint_type"]),
                "model_name": None, "version_name": None, "base_model": None,
                "weight": entry["weight"], "trained_words": [],
                "file": None, "error": str(info) if info else "fetch failed",
            })
            continue

        resources.append(_shape_resource(vid, info, hint_type=entry["hint_type"],
                                         weight=entry["weight"]))

    checkpoints = [r for r in resources if r["kind"] == "checkpoint"]
    if not checkpoints and resources:
        warnings.append("No checkpoint among the linked resources — the graph builder will need a manual pick.")
    if len(checkpoints) > 1:
        warnings.append("Multiple checkpoints linked; the first one will be used.")

    # --- generation parameters -------------------------------------------
    sampler = map_sampler(meta.get("sampler"))
    if not sampler["mapped"] and meta.get("sampler"):
        warnings.append(
            f"Sampler '{meta.get('sampler')}' has no direct ComfyUI equivalent; "
            f"falling back to {sampler['comfy_sampler']}/{sampler['comfy_scheduler']}."
        )

    has_upscaler_resource = any(r.get("kind") == "upscaler" for r in resources)

    width, height, size_source = _parse_size(meta, item)
    uploaded_w, uploaded_h = item.get("width"), item.get("height")
    if size_source == "image" and has_upscaler_resource:
        # The uploaded file is the POST-upscale image, so its dimensions are
        # not the generation size. Snap back to a standard bucket.
        width, height = _snap_to_bucket(width, height, item.get("baseModel"))
        warnings.append(
            f"Uploaded image is {uploaded_w}×{uploaded_h} (already upscaled); base generation "
            f"size snapped to the standard {width}×{height} for this model family."
        )
    elif size_source == "image":
        warnings.append("Generation size not in metadata; using the uploaded image's dimensions.")
    elif size_source is None:
        warnings.append("No size information found; the graph builder will default to the base model's native size.")

    clip_skip = _num(meta.get("clipSkip", meta.get("Clip skip")), int, None)

    upscale_hint = _extract_upscale_hint(meta)
    if meta.get("Hires upscaler") and not has_upscaler_resource:
        warnings.append(
            f"Metadata names hires upscaler '{meta.get('Hires upscaler')}' but it isn't linked as a "
            "resource — not wired (hash lookup comes with step 5)."
        )
    # A1111-style hires-fix hints for the second sampling pass, when present.
    hires_denoise_hint = _num(meta.get("Denoising strength"), float, None) if has_upscaler_resource else None
    hires_steps_hint = _num(meta.get("Hires steps"), int, None) if has_upscaler_resource else None

    if not (meta.get("prompt") or meta.get("negativePrompt") or resources):
        warnings.insert(0,
            "Civitai has no generation data stored for this image — sampler, steps, CFG and "
            "seed shown below are CiviImport defaults, not the image's real settings. "
            "(Check the image page: if it has no 'Generation data' panel, the data was never "
            "uploaded and can't be recovered.)")

    recipe = {
        "source": {
            "image_id": item.get("id"),
            "url": source_url or (f"https://civitai.com/images/{item.get('id')}" if item.get("id") else None),
            "post_id": item.get("postId"),
            "username": item.get("username"),
            "base_model": item.get("baseModel"),
            "process": meta.get("workflow") or "txt2img",
        },
        "prompt": meta.get("prompt") or "",
        "negative_prompt": meta.get("negativePrompt") or "",
        "sampler": sampler,
        "steps": _num(meta.get("steps"), int, 20),
        "cfg": _num(meta.get("cfgScale"), float, 7.0),
        "seed": _num(meta.get("seed"), int, -1),
        "clip_skip": clip_skip,
        "width": width,
        "height": height,
        "uploaded_width": uploaded_w,
        "uploaded_height": uploaded_h,
        "quantity": _num(meta.get("quantity"), int, 1),
        "upscale_factor_hint": upscale_hint,
        "hires_steps_hint": hires_steps_hint,
        "hires_denoise_hint": hires_denoise_hint,
        "resources": resources,
        "warnings": warnings,
        "raw_meta": meta,
    }
    return recipe


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------

def _has_meta_size(meta: dict) -> bool:
    size = meta.get("Size") or meta.get("size")
    if isinstance(size, str) and "x" in size:
        return True
    return bool(_num(meta.get("width"), int, None) and _num(meta.get("height"), int, None))


def _meta_is_thin(meta: dict) -> bool:
    """True when a REST item's meta has nothing we can build a recipe from —
    no prompt, no linked resources, no A1111 hashes. REST sometimes returns an
    image successfully but with an empty meta (older images especially), even
    though the site's generation-data panel has the full thing."""
    if not meta:
        return True
    return not (meta.get("prompt") or meta.get("civitaiResources")
                or meta.get("resources") or meta.get("hashes"))


def _merge_meta(item: dict, gd_item: dict) -> bool:
    """Fill a REST item's meta from the site's generation-data payload, never
    overwriting what REST already provided. Returns True when something we can
    actually use (prompt or resources) was recovered."""
    src = (gd_item or {}).get("meta") or {}
    if not src:
        return False
    meta = item.setdefault("meta", {})
    for k, v in src.items():
        if meta.get(k) in (None, "", [], {}):
            meta[k] = v
    if not item.get("baseModel"):
        item["baseModel"] = gd_item.get("baseModel")
    if not item.get("username"):
        item["username"] = gd_item.get("username")
    return bool(meta.get("prompt") or meta.get("civitaiResources"))


def _merge_meta_size(item: dict, gd_item: dict) -> bool:
    """Copy generation width/height from a generation-data item into the REST
    item's meta (without overwriting anything already there)."""
    src = (gd_item or {}).get("meta") or {}
    w = _num(src.get("width"), int, None)
    h = _num(src.get("height"), int, None)
    if not (w and h):
        return False
    meta = item.setdefault("meta", {})
    meta.setdefault("width", w)
    meta.setdefault("height", h)
    return True


def _finish_recipe(item: dict, host: str, api_key: str | None = None,
                   source_url: str | None = None, skip_enrich: bool = False,
                   note: str | None = None,
                   preset_versions: dict[int, dict] | None = None) -> dict:
    """Shared back half of resolution for any hop-1-style item (from an image
    URL or a model page's showcase gallery): optional size enrichment, resource
    version fetches — including the off-site hash path — build_recipe, and
    host stamping. `preset_versions` short-circuits hop-2 fetches for versions
    already in hand (the model-page flow fetched its own version for the card).
    """
    enriched = False
    recovered = False
    image_id = item.get("id")
    if not skip_enrich and image_id:
        meta_now = item.get("meta") or {}
        thin = _meta_is_thin(meta_now)
        # REST can answer with the image but an empty meta. The site endpoint
        # (what the Remix button reads) often still has it, so fall through on
        # thin meta as well as on a missing size — not only on REST errors.
        if thin or not _has_meta_size(meta_now):
            try:
                gd = fetch_generation_data(image_id, api_key=api_key, host=host)
            except CivitaiError:
                gd = None
            if gd:
                if thin:
                    recovered = _merge_meta(item, gd)
                if not recovered:
                    enriched = _merge_meta_size(item, gd)

    meta = item.get("meta") or {}
    version_ids: list[int] = []
    for res in meta.get("civitaiResources") or []:
        if res.get("modelVersionId") is not None:
            version_ids.append(int(res["modelVersionId"]))
    if not version_ids:
        version_ids = [int(v) for v in item.get("modelVersionIds") or []]

    version_infos: dict[int, dict] = dict(preset_versions or {})
    hash_notes: list[str] = []

    # Off-site images (A1111 uploads): no linked resources, only file hashes.
    # Resolve via /model-versions/by-hash and synthesize civitaiResources so
    # build_recipe treats them exactly like on-site images.
    if not version_ids:
        synth = []
        for e in _hash_entries_from_meta(meta):
            info = None
            for h in (host, _other_host(host)):
                try:
                    info = fetch_version_by_hash(e["hash"], api_key=api_key, host=h)
                    break
                except CivitaiError:
                    pass
            if not info or not info.get("id"):
                hash_notes.append(f"Hash {e['hash'][:10]}… "
                                  f"({e.get('name') or e.get('type_hint') or 'resource'}) "
                                  "was not found on Civitai; skipped.")
                continue
            vid = int(info["id"])
            version_infos[vid] = info
            entry = {"modelVersionId": vid, "type": _HASH_TYPE_MAP.get(e.get("type_hint"))}
            if e.get("weight") is not None:
                entry["weight"] = e["weight"]
            synth.append(entry)
        if synth:
            item.setdefault("meta", {})["civitaiResources"] = synth
            version_ids = [s["modelVersionId"] for s in synth]
            hash_notes.insert(0, "Resources resolved from A1111 hash metadata (off-site image).")

    for vid in dict.fromkeys(version_ids):  # dedupe, keep order
        if vid in version_infos:
            continue
        try:
            version_infos[vid] = fetch_model_version_any(vid, api_key=api_key, host=host)
        except CivitaiError as e:
            version_infos[vid] = e

    recipe = build_recipe(item, version_infos, source_url=source_url)
    recipe["source"]["api_host"] = host
    if note:
        recipe["warnings"].append(note)
    recipe["warnings"].extend(hash_notes)
    if recovered:
        recipe["warnings"].append(
            "The REST API returned this image without generation data; recovered it from the "
            "site's generation-data endpoint (the same source the Remix button uses)."
        )
    if enriched:
        m = item.get("meta") or {}
        recipe["warnings"].append(
            f"Generation size {m.get('width')}×{m.get('height')} taken from the site's "
            "generation data (the same source the Remix button uses)."
        )
    return recipe


def resolve_image(url_or_id: str, api_key: str | None = None) -> dict:
    """URL -> recipe. Network I/O happens here (blocking; run off the event loop).

    Host routing: hop 1 goes to the pasted URL's domain (civitai.red keeps its
    own view of the shared database; .com/.green are the SFW view). If the
    image is invisible there, the other host is tried once before giving up.
    Hop 2 follows the host that answered, with the same per-version fallback
    (an SFW image can still reference a model hidden on civitai.com).
    """
    image_id, host = parse_image_ref(url_or_id)

    fallback_note = None
    item = None
    tried: list[str] = []
    for h in (host, _other_host(host)):
        try:
            item = fetch_image_item(image_id, api_key=api_key, host=h)
            if h != host:
                fallback_note = f"Image not visible via {host}; resolved through {h} instead."
                host = h
            break
        except CivitaiError as e:
            tried.append(f"{h} REST: {e}")
    if item is None:
        for h in (host, _other_host(host)):
            try:
                item = fetch_generation_data(image_id, api_key=api_key, host=h)
                fallback_note = (f"REST image lookup failed; used the {h} site endpoint "
                                 "(same lookup the image page uses).")
                host = h
                via_site = True
                break
            except CivitaiError as e:
                tried.append(f"{h} site: {e}")
    else:
        via_site = False
    if item is None:
        print("[CiviImport] all lookups failed for image", image_id, "->", " | ".join(tried))
        raise CivitaiError(
            f"Image {image_id} was not found on any Civitai endpoint (tried civitai.com and "
            "civitai.red, REST and site APIs). It may be deleted, the URL may be wrong, or it "
            "sits behind a browsing level your API key's account hasn't enabled."
        )

    # REST meta sometimes omits the generation size even though the site has
    # it (this is what the Remix button reads) — _finish_recipe enriches it.
    return _finish_recipe(
        item, host, api_key=api_key,
        source_url=url_or_id if not str(url_or_id).isdigit() else None,
        skip_enrich=via_site, note=fallback_note)


# --------------------------------------------------------------------------
# Model-page resolution  (STEP 6)
#
# A pasted /models/... URL yields two things: the model itself (card with a
# presence light + download) and a starter recipe imported from one of the
# version's showcase images — the creator's own examples, which carry full
# generation meta just like any other Civitai image.
# --------------------------------------------------------------------------

def _usable_showcase_meta(meta) -> bool:
    """Meta that the existing pipeline can turn into a recipe: a prompt or any
    resource linkage (on-site civitaiResources or off-site hashes)."""
    if not isinstance(meta, dict) or not meta:
        return False
    return bool(meta.get("prompt") or meta.get("civitaiResources")
                or meta.get("resources") or meta.get("hashes"))


def _pick_showcase(images: list, creator: str | None = None) -> dict | None:
    """First usable showcase image: videos and meta-less entries are skipped.
    When the creator's name is known (images-API fallback), their own posts win
    over community ones; the version payload keeps the page's carousel order,
    so 'first' there is the cover image."""
    usable = [im for im in images or []
              if (im.get("type") or "image").lower() == "image"
              and _usable_showcase_meta(im.get("meta"))]
    if not usable:
        return None
    if creator:
        for im in usable:
            if (im.get("username") or "").lower() == creator.lower():
                return im
    return usable[0]


def _item_from_showcase(img: dict) -> dict:
    """Shape a showcase-gallery entry into a hop-1-style item. Version-payload
    entries usually have no image id (no size enrichment then — but they do
    ship width/height, which the size-priority chain already handles)."""
    return {
        "id": img.get("id"),
        "meta": dict(img.get("meta") or {}),
        "width": img.get("width"),
        "height": img.get("height"),
        "username": img.get("username"),
        "postId": img.get("postId"),
        "baseModel": img.get("baseModel"),
    }


def _ensure_model_in_recipe(recipe: dict, version_id: int, version_info: dict) -> None:
    """Wire-in guarantee: the pasted model must appear in a showcase-derived
    recipe. Showcase meta normally links it, but off-site-style uploads may
    not. Skipped when the recipe already references the model (any version)."""
    model_id = version_info.get("modelId")
    for r in recipe.get("resources") or []:
        if r.get("version_id") == version_id:
            return
        if model_id is not None and r.get("model_id") == model_id:
            return
    shaped = _shape_resource(version_id, version_info)
    name = ((version_info.get("model") or {}).get("name")) or f"version {version_id}"
    if shaped["kind"] == "checkpoint":
        # The graph builder uses the first checkpoint in the list.
        recipe["resources"].insert(0, shaped)
        recipe["warnings"] = [w for w in recipe["warnings"]
                              if "No checkpoint among the linked resources" not in w]
    else:
        recipe["resources"].append(shaped)
    recipe["warnings"].append(
        f"'{name}' wasn't linked in the showcase image's metadata — added it to the "
        "recipe so the pasted model is wired into the workflow."
    )


def resolve_model_page(url_or_id: str, api_key: str | None = None) -> dict:
    """Model-page URL -> model card + version list + a starter recipe from the
    version's showcase gallery. Blocking network I/O (run off the event loop).

    Version choice: an explicit ?modelVersionId= wins; otherwise the model's
    first listed version (what the page shows by default). The version itself
    goes through the generalist hop-2 chain, so red-only and REST-quirky
    versions resolve the same way they do for images.
    """
    ref = parse_any_ref(url_or_id)
    if ref["kind"] != "model":
        raise CivitaiError("Not a Civitai model-page URL.")
    host = ref["host"]
    model_id, version_id = ref["model_id"], ref["version_id"]
    notes: list[str] = []

    model_data = None
    if model_id is not None:
        try:
            model_data = fetch_model_any(model_id, api_key=api_key, host=host)
        except CivitaiError as e:
            if version_id is None:
                raise  # nothing else identifies a version
            notes.append(f"Model lookup failed ({e}) — continuing with the version alone.")

    if version_id is None:
        listed = (model_data or {}).get("modelVersions") or []
        if not listed or listed[0].get("id") is None:
            raise CivitaiError(f"Model {model_id} has no published versions.")
        version_id = int(listed[0]["id"])  # the page's default version

    version = fetch_model_version_any(version_id, api_key=api_key, host=host)
    if model_id is None:
        model_id = version.get("modelId")
    if model_data is None and model_id is not None:
        try:
            model_data = fetch_model_any(int(model_id), api_key=api_key, host=host)
        except CivitaiError:
            pass  # the card still renders from the version payload

    model_node = version.get("model") or {}
    model_name = (model_data or {}).get("name") or model_node.get("name")
    model_type = (model_data or {}).get("type") or model_node.get("type")
    creator = ((model_data or {}).get("creator") or {}).get("username")

    versions_list = [{"id": int(v["id"]), "name": v.get("name"),
                      "base_model": v.get("baseModel")}
                     for v in (model_data or {}).get("modelVersions") or []
                     if v.get("id") is not None]
    if not any(v["id"] == version_id for v in versions_list):
        versions_list.insert(0, {"id": version_id, "name": version.get("name"),
                                 "base_model": version.get("baseModel")})

    primary = _pick_primary_file(version.get("files") or [])
    selected = {
        "version_id": version_id,
        "model_id": model_id,
        "name": version.get("name"),
        "base_model": version.get("baseModel"),
        "kind": (model_type or "unknown").lower(),
        "folder": folder_for_type(model_type),
        "trained_words": version.get("trainedWords") or [],
        "file": None if primary is None else {
            "name": primary.get("name"),
            "size_kb": primary.get("sizeKB"),
            "hashes": primary.get("hashes") or {},
            "download_url": primary.get("downloadUrl") or version.get("downloadUrl"),
        },
    }

    # --- showcase image -> starter recipe ---------------------------------
    recipe = None
    showcase = None
    img = _pick_showcase(version.get("images") or [])
    source = "version_gallery"
    if img is None:
        gallery = []
        for h in (host, _other_host(host)):
            try:
                gallery = fetch_version_gallery(version_id, api_key=api_key, host=h)
                if gallery:
                    break
            except CivitaiError:
                pass
        img = _pick_showcase(gallery, creator=creator)
        source = "images_api"
    if img is not None:
        item = _item_from_showcase(img)
        item["baseModel"] = item.get("baseModel") or version.get("baseModel")
        if item.get("id"):
            source_url = f"https://{host}/images/{item['id']}"
        elif model_id is not None:
            source_url = f"https://{host}/models/{model_id}?modelVersionId={version_id}"
        else:
            source_url = None
        recipe = _finish_recipe(item, host, api_key=api_key, source_url=source_url,
                                preset_versions={version_id: version})
        _ensure_model_in_recipe(recipe, version_id, version)
        showcase = {"image_id": item.get("id"), "image_url": img.get("url"),
                    "username": img.get("username") or creator, "source": source}
    else:
        notes.append(
            "No generation data found on this version's showcase images — the model "
            "can still be downloaded; paste one of its gallery image URLs for a full recipe."
        )

    return {
        "model": {
            "model_id": model_id,
            "name": model_name,
            "type": model_type,
            "creator": creator,
            "url": f"https://{host}/models/{model_id}" if model_id is not None else None,
        },
        "versions": versions_list,
        "selected_version": selected,
        "api_host": host,
        "recipe": recipe,
        "showcase": showcase,
        "notes": notes,
    }
