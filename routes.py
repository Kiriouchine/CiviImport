# CiviImport — HTTP endpoints exposed to the ComfyUI frontend
#
# Copyright (c) 2026 Sev Kiriouchine
# SPDX-License-Identifier: MIT
# Part of CiviImport: https://github.com/Kiriouchine/CiviImport

"""CiviImport — HTTP endpoints on the running ComfyUI server.

Registered via @PromptServer.instance.routes.* so they live on the same
host/port as ComfyUI itself (the frontend reaches them with api.fetchApi).

Endpoints:
  GET  /civiimport/ping       health check (step 1 smoke test)
  POST /civiimport/resolve    {url} -> {ok, kind:'image', recipe} for image URLs (step 2)
                              or {ok, kind:'model', model_page, recipe?} for /models/... URLs (step 6)
  GET  /civiimport/apikey     {set, masked, source}
  POST /civiimport/apikey     {key} ('' clears) -> {ok, set, masked}
  POST /civiimport/build      {recipe, options?} -> {ok, workflow, info} (steps 3, 7)
  POST /civiimport/download   {version_ids, api_host?} -> starts background downloads (step 4)
  POST /civiimport/hashscan   {folders?} -> hashes uncached local files (step 5)

Security note: these handlers never accept filesystem paths from the client.
Download targets (step 4) will be derived server-side from folder_paths only.
"""

import asyncio
import threading
import traceback

from aiohttp import web
from server import PromptServer

from . import civitai_api, config, downloader, graph_builder, local_models

VERSION = "1.3.1"
routes = PromptServer.instance.routes


def _err(message: str, status: int = 200) -> web.Response:
    # App-level errors ride on 200 with ok=false so the frontend handles one shape.
    return web.json_response({"ok": False, "error": message}, status=status)


@routes.get("/civiimport/ping")
async def ping(request: web.Request) -> web.Response:
    return web.json_response({
        "ok": True,
        "name": "civiimport",
        "version": VERSION,
        "local_check": "folder_paths" if local_models.HAVE_COMFY else "unavailable",
    })


@routes.post("/civiimport/resolve")
async def resolve(request: web.Request) -> web.Response:
    try:
        data = await request.json()
    except Exception:
        return _err("Invalid request body (expected JSON).", status=400)

    url = (data.get("url") or "").strip()
    if not url:
        return _err("Paste a Civitai image or model URL first.")

    api_key = config.get_api_key()
    loop = asyncio.get_event_loop()
    try:
        kind = civitai_api.parse_any_ref(url)["kind"]
    except civitai_api.CivitaiError as e:
        return _err(str(e))
    try:
        # requests is blocking; keep it off the server's event loop.
        if kind == "model":
            page = await loop.run_in_executor(
                None, civitai_api.resolve_model_page, url, api_key)
            recipe = page.pop("recipe", None)
            if recipe:
                local_models.annotate_local_status(recipe)
            sel = page.get("selected_version") or {}
            sel["local"] = local_models.local_status_for_file(
                sel.get("folder"), sel.get("file"))
            return web.json_response({"ok": True, "kind": "model", "model_page": page,
                                      "recipe": recipe, "used_api_key": bool(api_key)})
        recipe = await loop.run_in_executor(None, civitai_api.resolve_image, url, api_key)
        local_models.annotate_local_status(recipe)
        return web.json_response({"ok": True, "kind": "image", "recipe": recipe,
                                  "used_api_key": bool(api_key)})
    except civitai_api.CivitaiError as e:
        return _err(str(e))
    except Exception as e:
        traceback.print_exc()
        return _err(f"Unexpected error while resolving: {e}")


@routes.get("/civiimport/apikey")
async def apikey_get(request: web.Request) -> web.Response:
    key = config.get_api_key()
    return web.json_response({
        "ok": True,
        "set": bool(key),
        "masked": config.masked_key(key),
        "source": config.get_api_key_source(),
    })


@routes.post("/civiimport/apikey")
async def apikey_set(request: web.Request) -> web.Response:
    try:
        data = await request.json()
    except Exception:
        return _err("Invalid request body (expected JSON).", status=400)
    try:
        config.set_api_key(data.get("key") or "")
    except Exception as e:
        return _err(f"Could not save the key: {e}")
    key = config.get_api_key()
    return web.json_response({
        "ok": True,
        "set": bool(key),
        "masked": config.masked_key(key),
        "source": config.get_api_key_source(),
    })


@routes.post("/civiimport/build")
async def build(request: web.Request) -> web.Response:
    try:
        data = await request.json()
    except Exception:
        return _err("Invalid request body (expected JSON).", status=400)

    recipe = data.get("recipe")
    if not recipe:
        return _err("No recipe to build from — fetch an image first.")
    try:
        options = data.get("options") or {}
        build_opts = {
            "upscale_factor": options.get("upscale_factor"),
            # Synthesized hires chain for upscaler-less recipes: on by default.
            "add_upscaler": bool(options.get("add_upscaler", True)),
        }
        if build_opts["add_upscaler"] and isinstance(recipe, dict) and not any(
                r.get("kind") == "upscaler" for r in recipe.get("resources") or []):
            build_opts["upscaler_file"] = local_models.pick_upscaler_file()
        workflow, info = graph_builder.build_workflow(recipe, build_opts)
        return web.json_response({"ok": True, "workflow": workflow, "info": info})
    except graph_builder.BuildError as e:
        return _err(str(e))
    except Exception as e:
        traceback.print_exc()
        return _err(f"Unexpected error while building the workflow: {e}")


@routes.post("/civiimport/download")
async def download(request: web.Request) -> web.Response:
    try:
        data = await request.json()
    except Exception:
        return _err("Invalid request body (expected JSON).", status=400)

    raw_ids = data.get("version_ids") or ([data["version_id"]] if data.get("version_id") else [])
    try:
        version_ids = [int(v) for v in raw_ids]
    except (TypeError, ValueError):
        return _err("version_ids must be integers.")
    if not version_ids:
        return _err("Nothing to download — no version ids given.")

    host = data.get("api_host") or civitai_api.DEFAULT_HOST
    if host not in (civitai_api.DEFAULT_HOST, civitai_api.RED_HOST):
        host = civitai_api.DEFAULT_HOST
    api_key = config.get_api_key()

    def fetch(vid, key, h):
        return civitai_api.fetch_model_version_any(vid, api_key=key, host=h)

    def send(payload):
        try:
            PromptServer.instance.send_sync("civiimport.progress", payload)
        except Exception:
            pass

    accepted, skipped = downloader.start_downloads(
        version_ids, api_key=api_key, host=host, fetch_version=fetch, send=send,
        type_to_folder=civitai_api.CIVITAI_TYPE_TO_FOLDER,
        dir_hint=(config.load().get("download_dir_hint") or None),
    )
    return web.json_response({"ok": True, "accepted": accepted, "skipped": skipped,
                              "api_key": bool(api_key)})


_hashscan_running = {"flag": False}


@routes.post("/civiimport/hashscan")
async def hashscan(request: web.Request) -> web.Response:
    try:
        data = await request.json()
    except Exception:
        return _err("Invalid request body (expected JSON).", status=400)
    valid = sorted(set(civitai_api.CIVITAI_TYPE_TO_FOLDER.values()))
    folders = [f for f in (data.get("folders") or []) if f in valid] or valid
    if _hashscan_running["flag"]:
        return _err("A hash scan is already running.")
    _hashscan_running["flag"] = True

    def send(payload):
        try:
            PromptServer.instance.send_sync("civiimport.hashscan", payload)
        except Exception:
            pass

    def work():
        try:
            local_models.deep_hash_scan(folders, progress=send)
        except Exception as e:
            traceback.print_exc()
            send({"status": "done", "hashed": 0, "total": 0, "error": str(e)})
        finally:
            _hashscan_running["flag"] = False

    threading.Thread(target=work, daemon=True, name="CiviImport-hashscan").start()
    return web.json_response({"ok": True, "folders": folders})


print(f"[CiviImport] v{VERSION} routes registered (/civiimport/ping, /resolve, /apikey, ...)")
