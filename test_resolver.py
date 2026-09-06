#!/usr/bin/env python3
# CiviImport — self-tests for the resolver
#
# Copyright (c) 2026 Sev Kiriouchine
# SPDX-License-Identifier: MIT
# Part of CiviImport: https://github.com/Kiriouchine/CiviImport

"""Standalone test for the CiviImport resolver — no ComfyUI needed.

Online mode (default) hits the real Civitai API:
    python test_resolver.py                      # the three reference images
    python test_resolver.py <url|id> [...]       # your own (image or /models/ URLs)
    python test_resolver.py --json <url>         # dump the full recipe JSON

Optional environment variables:
    CIVITAI_API_KEY          used for gated images / higher browsing level
    CIVIIMPORT_MODELS_ROOT   e.g. C:\\Users\\maste\\ComfyUI-Shared\\models
                             enables the local-presence check outside ComfyUI

Offline mode runs the normalizer against canned fixtures (no network):
    python test_resolver.py --offline
"""

import argparse
import copy
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import civitai_api  # noqa: E402
import local_models  # noqa: E402

DEFAULT_URLS = [
    "https://civitai.com/images/135890833",
    "https://civitai.com/images/135891361",
    "https://civitai.com/images/24499094",
]

GREEN, RED, GRAY, RESET = "\033[92m", "\033[91m", "\033[90m", "\033[0m"


def summarize(recipe: dict) -> None:
    src = recipe["source"]
    print(f"\n=== Image #{src['image_id']} by {src.get('username')} "
          f"({src.get('base_model')}, {src.get('process')}) ===")
    s = recipe["sampler"]
    print(f"  sampler : {s['civitai']} -> {s['comfy_sampler']}/{s['comfy_scheduler']}"
          + ("" if s["mapped"] else "  [FALLBACK]"))
    print(f"  steps/cfg: {recipe['steps']} / {recipe['cfg']}   seed: {recipe['seed']}"
          f"   clip skip: {recipe['clip_skip']}   size: {recipe['width']}x{recipe['height']}")
    print(f"  prompt  : {recipe['prompt'][:90]!r}{'…' if len(recipe['prompt']) > 90 else ''}")
    print(f"  negative: {recipe['negative_prompt'][:90]!r}{'…' if len(recipe['negative_prompt']) > 90 else ''}")
    print(f"  resources ({len(recipe['resources'])}), local check: {recipe.get('local_check')}")
    for r in recipe["resources"]:
        loc = r.get("local") or {}
        light = f"{GREEN}●{RESET}" if loc.get("present") else (
            f"{GRAY}●{RESET}" if r.get("error") else f"{RED}●{RESET}")
        w = f" @ {r['weight']}" if r.get("weight") is not None and r["kind"] != "checkpoint" else ""
        fname = (r.get("file") or {}).get("name")
        print(f"    {light} [{r['kind']:<10}] {r.get('model_name')} ({r.get('version_name')}){w}")
        print(f"        file: {fname}  ->  local: {loc.get('filename') or 'missing'}")
    for w in recipe.get("warnings", []):
        print(f"  ! {w}")
    print(f"  missing locally: {recipe.get('missing_count')} / {len(recipe['resources'])}")


def summarize_model_page(page: dict, as_json: bool = False) -> None:
    m, sel = page["model"], page["selected_version"]
    sel["local"] = local_models.local_status_for_file(sel.get("folder"), sel.get("file"))
    loc = sel["local"]
    light = f"{GREEN}●{RESET}" if loc.get("present") else f"{RED}●{RESET}"
    f = sel.get("file") or {}
    print(f"\n=== Model #{m.get('model_id')} {m.get('name')} [{m.get('type')}] "
          f"by {m.get('creator') or '?'} (via {page.get('api_host')}) ===")
    print(f"  version : {sel.get('name')} (id {sel['version_id']}, {sel.get('base_model')})"
          f" — {len(page.get('versions') or [])} listed")
    print(f"  {light} file: {f.get('name')} ({(f.get('size_kb') or 0) / 1024:.0f} MB)"
          f"  ->  local: {loc.get('filename') or 'missing'}")
    for n in page.get("notes") or []:
        print(f"  ! {n}")
    recipe = page.get("recipe")
    if recipe:
        sc = page.get("showcase") or {}
        print(f"  starter recipe from showcase ({sc.get('source')},"
              f" image {sc.get('image_id') or '—'}):")
        local_models.annotate_local_status(recipe)
        if as_json:
            recipe.pop("raw_meta", None)
            print(json.dumps(recipe, indent=2, ensure_ascii=False))
        else:
            summarize(recipe)
    else:
        print("  (no showcase recipe)")


def run_online(urls: list[str], as_json: bool) -> int:
    api_key = os.environ.get("CIVITAI_API_KEY") or None
    print(f"API key: {'set' if api_key else 'NOT set'}   "
          f"models root: {os.environ.get('CIVIIMPORT_MODELS_ROOT') or '(none — local check disabled)'}")
    failures = 0
    for url in urls:
        try:
            if civitai_api.parse_any_ref(url)["kind"] == "model":
                summarize_model_page(civitai_api.resolve_model_page(url, api_key=api_key),
                                     as_json)
                continue
            recipe = civitai_api.resolve_image(url, api_key=api_key)
            local_models.annotate_local_status(recipe)
            if as_json:
                recipe.pop("raw_meta", None)
                print(json.dumps(recipe, indent=2, ensure_ascii=False))
            else:
                summarize(recipe)
        except civitai_api.CivitaiError as e:
            failures += 1
            print(f"\n=== {url} ===\n  ERROR: {e}")
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# Offline fixtures — synthetic data shaped like the real API responses,
# with parameter values taken from reference image 135891361
# (MoonArt Cauldron Mix + 2 LoRAs, Euler, cfg 5.5, steps 23, clip skip 2).
# ---------------------------------------------------------------------------
FIXTURE_ITEM = {
    "id": 135891361,
    "width": 832, "height": 1216,  # synthetic; page does not expose these
    "postId": 29619336,
    "username": "OptimalOyster",
    "baseModel": "SDXL 1.0",
    "meta": {
        "prompt": "masterpiece, best quality, highly detailed, aidmaImageUpgrader, cs-l3ath3r, "
                  "art printed in decorative leather, intricate embossing, rich textures",
        "negativePrompt": "low quality, blurry, distorted, deformed, watermark, text",
        "sampler": "Euler", "cfgScale": 5.5, "steps": 23, "clipSkip": 2,
        "seed": 285342747, "quantity": 5, "workflow": "txt2img",
        "civitaiResources": [
            {"type": "checkpoint", "modelVersionId": 1970825},
            {"type": "lora", "weight": 0.8, "modelVersionId": 634488},
            {"type": "lora", "weight": 0.6, "modelVersionId": 1269325},
        ],
    },
}

FIXTURE_VERSIONS = {
    1970825: {
        "id": 1970825, "modelId": 1741447, "name": "SDXL_v1.0", "baseModel": "SDXL 1.0",
        "trainedWords": [], "downloadUrl": "https://civitai.com/api/download/models/1970825",
        "model": {"name": "MoonArt Cauldron Mix", "type": "Checkpoint"},
        "files": [{"name": "moonartCauldronMix_sdxlV10.safetensors", "sizeKB": 6775430.0,
                   "primary": True, "hashes": {"AutoV2": "AAAA1111BB"},
                   "downloadUrl": "https://civitai.com/api/download/models/1970825"}],
    },
    634488: {
        "id": 634488, "modelId": 569267, "name": "SDXL", "baseModel": "SDXL 1.0",
        "trainedWords": ["cs-l3ath3r"], "downloadUrl": "https://civitai.com/api/download/models/634488",
        "model": {"name": "Decorative Leather & Mixed Media", "type": "LORA"},
        "files": [{"name": "decorative_leather_sdxl.safetensors", "sizeKB": 223000.0,
                   "primary": True, "hashes": {"AutoV2": "BBBB2222CC"},
                   "downloadUrl": "https://civitai.com/api/download/models/634488"}],
    },
    1269325: {
        "id": 1269325, "modelId": 562866, "name": "SDXL v0.3", "baseModel": "SDXL 1.0",
        "trainedWords": ["aidmaImageUpgrader"], "downloadUrl": "https://civitai.com/api/download/models/1269325",
        "model": {"name": "FLUX Image Upgrader / Detail Maximizer", "type": "LORA"},
        "files": [{"name": "aidmaImageUpgrader_sdxl_v03.safetensors", "sizeKB": 165000.0,
                   "primary": True, "hashes": {"AutoV2": "CCCC3333DD"},
                   "downloadUrl": "https://civitai.com/api/download/models/1269325"}],
    },
}


def run_offline() -> int:
    print("Running offline self-test (fixtures, no network)…")

    # Sampler mapping spot checks
    m = civitai_api.map_sampler("Euler a")
    assert (m["comfy_sampler"], m["comfy_scheduler"], m["mapped"]) == ("euler_ancestral", "normal", True), m
    m = civitai_api.map_sampler("DPM++ 2M Karras")
    assert (m["comfy_sampler"], m["comfy_scheduler"], m["mapped"]) == ("dpmpp_2m", "karras", True), m
    m = civitai_api.map_sampler("TotallyNewSampler")
    assert m["mapped"] is False and m["comfy_sampler"] == "euler", m

    # URL parsing (multi-domain: .com / .green / .red share one ID space)
    assert civitai_api.parse_image_id("https://civitai.com/images/135890833?period=AllTime") == 135890833
    assert civitai_api.parse_image_id("24499094") == 24499094
    assert civitai_api.parse_image_ref("https://civitai.red/images/129103401") == (129103401, "civitai.red")
    assert civitai_api.parse_image_ref("civitai.green/images/136859929") == (136859929, "civitai.com")
    assert civitai_api.parse_image_ref("https://www.civitai.com/images/135890833") == (135890833, "civitai.com")
    assert civitai_api.parse_image_ref("129073947") == (129073947, "civitai.com")

    # URL parsing — model pages and download links (step 6)
    ref = civitai_api.parse_any_ref(
        "https://civitai.red/models/424242/example-lora?modelVersionId=616161")
    assert ref == {"kind": "model", "model_id": 424242, "version_id": 616161,
                   "host": "civitai.red"}, ref
    ref = civitai_api.parse_any_ref("https://civitai.com/models/424242")
    assert ref == {"kind": "model", "model_id": 424242, "version_id": None,
                   "host": "civitai.com"}, ref
    ref = civitai_api.parse_any_ref("civitai.green/models/12345/some-slug")
    assert (ref["kind"], ref["model_id"], ref["host"]) == ("model", 12345, "civitai.com"), ref
    ref = civitai_api.parse_any_ref(
        "https://civitai.com/api/download/models/616161?type=Model&format=SafeTensor")
    assert ref == {"kind": "model", "model_id": None, "version_id": 616161,
                   "host": "civitai.com"}, ref
    ref = civitai_api.parse_any_ref("https://civitai.red/images/129103401")
    assert ref == {"kind": "image", "image_id": 129103401, "host": "civitai.red"}, ref
    assert civitai_api.parse_any_ref("24499094")["kind"] == "image"
    try:
        civitai_api.parse_any_ref("https://example.com/whatever")
        raise AssertionError("expected CivitaiError for a non-Civitai URL")
    except civitai_api.CivitaiError:
        pass

    # Showcase picking: videos and meta-less entries skipped; creator preferred
    imgs = [
        {"type": "video", "url": "clip.mp4", "meta": {"prompt": "x"}},
        {"type": "image", "url": "a.jpg", "meta": None},
        {"type": "image", "url": "b.jpg", "meta": {"prompt": "community"}, "username": "fan"},
        {"type": "image", "url": "c.jpg", "meta": {"prompt": "own"}, "username": "Maker"},
    ]
    assert civitai_api._pick_showcase(imgs)["url"] == "b.jpg"
    assert civitai_api._pick_showcase(imgs, creator="maker")["url"] == "c.jpg"
    assert civitai_api._pick_showcase([{"type": "image", "meta": {}}]) is None
    assert civitai_api._pick_showcase([]) is None
    assert civitai_api._usable_showcase_meta({"hashes": {"model": "ab12"}})  # off-site style

    # Site-endpoint (generation-data) fallback mapping
    node = {"meta": {"prompt": "p", "negativePrompt": "n", "sampler": "Euler a",
                     "steps": 30, "cfgScale": 7, "seed": 5, "clipSkip": 2,
                     "width": 832, "height": 1216},
            "resources": [
                {"versionId": 1970825, "modelType": "Checkpoint", "baseModel": "SDXL 1.0"},
                {"versionId": 634488, "modelType": "LORA", "strength": 0.8},
            ]}
    item_gd = civitai_api._item_from_generation_data(999, node)
    civ = item_gd["meta"]["civitaiResources"]
    assert civ[0] == {"modelVersionId": 1970825, "type": "checkpoint"}
    assert civ[1] == {"modelVersionId": 634488, "type": "lora", "weight": 0.8}
    assert item_gd["baseModel"] == "SDXL 1.0" and item_gd["width"] is None
    r_gd = civitai_api.build_recipe(item_gd, copy.deepcopy(FIXTURE_VERSIONS))
    # meta width/height are the true generation size — preferred, no snapping/warning
    assert (r_gd["width"], r_gd["height"]) == (832, 1216)
    assert not any("uploaded image's dimensions" in w for w in r_gd["warnings"])
    assert [x["weight"] for x in r_gd["resources"]] == [None, 0.8]

    # trpc model-version mapper: list-shaped hashes, synthesized download URL
    vnode = {"id": 616161, "modelId": 424242, "name": "V1", "baseModel": "Pony",
             "trainedWords": ["examplestyle"],
             "model": {"name": "Example LoRA [SDXL]", "type": "LORA"},
             "files": [{"name": "ExampleLora_v1.safetensors", "sizeKB": 218000.0,
                        "primary": True,
                        "hashes": [{"type": "AutoV2", "hash": "ABCD1234EF"}]}]}
    v = civitai_api._version_from_trpc(616161, vnode, "civitai.red")
    assert v["model"]["type"] == "LORA" and v["files"][0]["hashes"] == {"AutoV2": "ABCD1234EF"}
    assert v["files"][0]["downloadUrl"] == "https://civitai.red/api/download/models/616161"
    assert v["downloadUrl"].endswith("/api/download/models/616161")

    # size enrichment helpers
    assert civitai_api._has_meta_size({"Size": "832x1216"})
    assert civitai_api._has_meta_size({"width": 1216, "height": 832})
    assert not civitai_api._has_meta_size({"seed": 1})
    rest_item = {"id": 1, "meta": {"prompt": "p"}}
    assert civitai_api._merge_meta_size(rest_item, {"meta": {"width": 1216, "height": 832}})
    assert (rest_item["meta"]["width"], rest_item["meta"]["height"]) == (1216, 832)

    # size sanity: degenerate values are ignored, next source wins
    assert civitai_api._parse_size({"width": 8, "height": 8},
                                   {"width": 832, "height": 1216}) == (832, 1216, "image")
    assert civitai_api._parse_size({}, {"width": 64, "height": 64}) == (None, None, None)

    # internal system resources (safe helpers): flagged, never counted missing
    item_sys = copy.deepcopy(FIXTURE_ITEM)
    item_sys["meta"]["civitaiResources"].append({"type": "textualinversion",
                                                 "modelVersionId": 250708})
    vers_sys = copy.deepcopy(FIXTURE_VERSIONS)
    vers_sys[250708] = {
        "id": 250708, "modelId": 222256, "name": "safe_pos", "baseModel": "SD 1.5",
        "trainedWords": [],
        "model": {"name": "Civitai Safe Helper (Minor)", "type": "TextualInversion"},
        "files": [{"name": "safe_pos.pt", "sizeKB": 25.0, "primary": True, "hashes": {}}],
    }
    r_sys = civitai_api.build_recipe(item_sys, vers_sys)
    sysres = next(r for r in r_sys["resources"] if r["version_id"] == 250708)
    assert sysres["system"] is True and sysres["kind"] == "textualinversion"
    assert r_sys["resources"][0]["system"] is False
    local_models.annotate_local_status(r_sys)  # no models root set here
    assert r_sys["missing_count"] == 3, r_sys["missing_count"]  # ckpt + 2 loras; system excluded
    assert "internal" in (sysres["local"].get("note") or "")

    # Recipe build
    recipe = civitai_api.build_recipe(FIXTURE_ITEM, FIXTURE_VERSIONS,
                                      source_url="https://civitai.com/images/135891361")
    assert recipe["steps"] == 23 and recipe["cfg"] == 5.5 and recipe["seed"] == 285342747
    assert recipe["clip_skip"] == 2 and recipe["width"] == 832 and recipe["height"] == 1216
    kinds = [r["kind"] for r in recipe["resources"]]
    assert kinds == ["checkpoint", "lora", "lora"], kinds
    assert [r["weight"] for r in recipe["resources"]] == [None, 0.8, 0.6]
    assert recipe["resources"][0]["folder"] == "checkpoints"
    assert recipe["resources"][1]["folder"] == "loras"
    assert recipe["resources"][1]["trained_words"] == ["cs-l3ath3r"]

    # Local matching: create a fake models root containing ONLY the checkpoint,
    # nested in a subfolder to verify relative-path reporting.
    with tempfile.TemporaryDirectory() as root:
        ckpt_dir = os.path.join(root, "checkpoints", "sdxl")
        os.makedirs(ckpt_dir)
        open(os.path.join(ckpt_dir, "moonartCauldronMix_sdxlV10.safetensors"), "wb").close()
        os.makedirs(os.path.join(root, "loras"))
        os.environ["CIVIIMPORT_MODELS_ROOT"] = root
        try:
            local_models.annotate_local_status(recipe)
        finally:
            os.environ.pop("CIVIIMPORT_MODELS_ROOT", None)

    assert recipe["resources"][0]["local"]["present"] is True
    assert recipe["resources"][0]["local"]["filename"] == "sdxl/moonartCauldronMix_sdxlV10.safetensors"
    assert recipe["resources"][1]["local"]["present"] is False
    assert recipe["missing_count"] == 2, recipe["missing_count"]

    # ------------------------------------------------------------------
    # Thin REST meta: recover the whole thing from the site endpoint
    # (regression — REST can return an image with an empty meta)
    # ------------------------------------------------------------------
    rest_thin = {"id": 273488, "meta": {}, "width": 768, "height": 768,
                 "username": "SomeUser", "postId": 1}
    gd_payload = {"meta": {"prompt": "a wizard", "negativePrompt": "blurry",
                           "sampler": "DPM++ 2M Karras", "steps": 30, "cfgScale": 6.5,
                           "seed": 12345, "Size": "512x768"},
                  "resources": [{"modelVersionId": 1970825, "modelType": "checkpoint",
                                 "baseModel": "SDXL 1.0"}]}
    saved_gd = civitai_api.fetch_generation_data
    saved_ver = civitai_api.fetch_model_version_any
    civitai_api.fetch_generation_data = (
        lambda i, api_key=None, host=civitai_api.DEFAULT_HOST:
        civitai_api._item_from_generation_data(i, gd_payload))
    civitai_api.fetch_model_version_any = (
        lambda v, api_key=None, host=civitai_api.DEFAULT_HOST:
        copy.deepcopy(FIXTURE_VERSIONS[1970825]))
    try:
        rec = civitai_api._finish_recipe(copy.deepcopy(rest_thin), "civitai.com")
        assert rec["prompt"] == "a wizard", rec["prompt"]
        assert rec["negative_prompt"] == "blurry"
        assert rec["steps"] == 30 and rec["cfg"] == 6.5 and rec["seed"] == 12345
        assert rec["sampler"]["comfy_sampler"] == "dpmpp_2m"
        assert (rec["width"], rec["height"]) == (512, 768)   # meta Size beats uploaded dims
        assert [r["kind"] for r in rec["resources"]] == ["checkpoint"]
        assert any("recovered it from the site" in w for w in rec["warnings"])
        assert not any("no generation data stored" in w for w in rec["warnings"])

        # ... and when the site endpoint has nothing either, say so honestly
        civitai_api.fetch_generation_data = (
            lambda i, api_key=None, host=civitai_api.DEFAULT_HOST:
            civitai_api._item_from_generation_data(i, {}))   # raises
        rec2 = civitai_api._finish_recipe(copy.deepcopy(rest_thin), "civitai.com")
        assert rec2["prompt"] == "" and not rec2["resources"]
        assert any("no generation data stored" in w for w in rec2["warnings"])
        assert any("no A1111 file hashes" in w for w in rec2["warnings"])
        assert not any("planned for a later step" in w for w in rec2["warnings"])
        # thin-meta recovery must not fire when REST already gave us a prompt
        rich = {"id": 1, "meta": {"prompt": "p", "seed": 3, "Size": "512x512"}}
        rec3 = civitai_api._finish_recipe(copy.deepcopy(rich), "civitai.com")
        assert rec3["prompt"] == "p" and rec3["seed"] == 3
        assert not any("recovered it from the site" in w for w in rec3["warnings"])
    finally:
        civitai_api.fetch_generation_data = saved_gd
        civitai_api.fetch_model_version_any = saved_ver
    print("  thin-meta recovery OK (site fallback, honest no-data path)")

    # ------------------------------------------------------------------
    # Step 6: model-page import
    # ------------------------------------------------------------------
    lora_info = {
        "id": 616161, "modelId": 424242, "name": "V1", "baseModel": "Pony",
        "trainedWords": ["examplestyle"],
        "model": {"name": "Example LoRA [SDXL]", "type": "LORA"},
        "files": [{"name": "ExampleLora_v1.safetensors", "sizeKB": 218000.0, "primary": True,
                   "hashes": {"AutoV2": "ABCD1234EF"},
                   "downloadUrl": "https://civitai.red/api/download/models/616161"}],
    }

    # Wire-in guarantee: pasted model injected when the showcase meta lacks it
    r_inj = civitai_api.build_recipe(copy.deepcopy(FIXTURE_ITEM), copy.deepcopy(FIXTURE_VERSIONS))
    civitai_api._ensure_model_in_recipe(r_inj, 616161, lora_info)
    added = r_inj["resources"][-1]
    assert added["version_id"] == 616161 and added["kind"] == "lora" and added["weight"] == 1.0
    assert added["file"]["name"] == "ExampleLora_v1.safetensors"
    assert any("wasn't linked" in w for w in r_inj["warnings"])
    n_res = len(r_inj["resources"])
    civitai_api._ensure_model_in_recipe(r_inj, 616161, lora_info)   # no duplicates
    assert len(r_inj["resources"]) == n_res
    other_version = dict(lora_info, id=616162)                      # same model, other version
    civitai_api._ensure_model_in_recipe(r_inj, 616162, other_version)
    assert len(r_inj["resources"]) == n_res

    # Checkpoint injection goes first and clears the no-checkpoint warning
    lora_only = {"id": 1, "meta": {"prompt": "p", "sampler": "Euler", "steps": 20,
                                   "cfgScale": 7, "seed": 1, "width": 832, "height": 1216,
                                   "civitaiResources": [{"type": "lora", "weight": 0.8,
                                                         "modelVersionId": 634488}]}}
    r_ck = civitai_api.build_recipe(lora_only, copy.deepcopy(FIXTURE_VERSIONS))
    assert any("No checkpoint among" in w for w in r_ck["warnings"])
    civitai_api._ensure_model_in_recipe(r_ck, 1970825, FIXTURE_VERSIONS[1970825])
    assert r_ck["resources"][0]["kind"] == "checkpoint"
    assert not any("No checkpoint among" in w for w in r_ck["warnings"])
    assert any("wasn't linked" in w for w in r_ck["warnings"])

    # resolve_model_page orchestration — monkeypatched fetchers, no network
    model_payload = {
        "id": 424242, "name": "Example LoRA [SDXL]", "type": "LORA",
        "creator": {"username": "Maker"},
        "modelVersions": [
            {"id": 616161, "name": "V1", "baseModel": "Pony"},
            {"id": 700001, "name": "V0.9", "baseModel": "Pony"},
        ],
    }
    lora_showcase = dict(lora_info)
    lora_showcase["images"] = [
        {"type": "video", "url": "clip.mp4"},
        {"type": "image", "url": "https://image.civitai.com/x/width=450/1.jpeg",
         "width": 832, "height": 1216,
         "meta": {"prompt": "examplestyle, portrait", "negativePrompt": "bad",
                  "sampler": "Euler a", "steps": 25, "cfgScale": 6, "seed": 42,
                  "Size": "832x1216",
                  "civitaiResources": [
                      {"type": "checkpoint", "modelVersionId": 1970825},
                      {"type": "lora", "weight": 0.15, "modelVersionId": 616161},
                  ]}},
    ]
    version_store = {616161: lora_showcase, 1970825: FIXTURE_VERSIONS[1970825]}
    calls = {"model": [], "version": []}

    def fake_fetch_model(mid, api_key=None, host=civitai_api.DEFAULT_HOST):
        calls["model"].append((mid, host))
        if mid != 424242:
            raise civitai_api.CivitaiError("no such model")
        return copy.deepcopy(model_payload)

    def fake_fetch_version(vid, api_key=None, host=civitai_api.DEFAULT_HOST):
        calls["version"].append((vid, host))
        if vid not in version_store:
            raise civitai_api.CivitaiError("no such version")
        return copy.deepcopy(version_store[vid])

    def no_gallery(vid, api_key=None, host=civitai_api.DEFAULT_HOST, limit=20):
        return []

    saved = (civitai_api.fetch_model, civitai_api.fetch_model_version,
             civitai_api.fetch_model_version_trpc, civitai_api.fetch_version_gallery)
    civitai_api.fetch_model = fake_fetch_model
    civitai_api.fetch_model_version = fake_fetch_version
    civitai_api.fetch_model_version_trpc = fake_fetch_version
    civitai_api.fetch_version_gallery = no_gallery
    try:
        page = civitai_api.resolve_model_page(
            "https://civitai.red/models/424242/example-lora?modelVersionId=616161")
        assert page["model"]["name"] == "Example LoRA [SDXL]"
        assert page["model"]["creator"] == "Maker"
        assert page["api_host"] == "civitai.red"
        assert [v["id"] for v in page["versions"]] == [616161, 700001]
        sel = page["selected_version"]
        assert sel["folder"] == "loras" and sel["file"]["name"] == "ExampleLora_v1.safetensors"
        rec = page["recipe"]
        assert rec is not None and rec["source"]["api_host"] == "civitai.red"
        assert sorted(r["kind"] for r in rec["resources"]) == ["checkpoint", "lora"]
        lora_res = next(r for r in rec["resources"] if r["version_id"] == 616161)
        assert lora_res["weight"] == 0.15                       # showcase meta wins
        assert (rec["width"], rec["height"]) == (832, 1216)
        assert not any("wasn't linked" in w for w in rec["warnings"])  # linked, no injection
        # the card's own version fetch is reused — hop 2 never refetches 616161
        assert [c[0] for c in calls["version"]].count(616161) == 1, calls["version"]
        assert page["showcase"]["source"] == "version_gallery"
        assert rec["source"]["url"].endswith("/models/424242?modelVersionId=616161")

        # no ?modelVersionId -> the page's default (first listed) version
        page2 = civitai_api.resolve_model_page("https://civitai.red/models/424242")
        assert page2["selected_version"]["version_id"] == 616161

        # showcase that doesn't link the model -> injected at weight 1.0
        unlinked = copy.deepcopy(lora_showcase)
        unlinked["images"][1]["meta"]["civitaiResources"] = [
            {"type": "checkpoint", "modelVersionId": 1970825}]
        version_store[616161] = unlinked
        page3 = civitai_api.resolve_model_page(
            "https://civitai.red/models/424242?modelVersionId=616161")
        rec3 = page3["recipe"]
        inj = next(r for r in rec3["resources"] if r["version_id"] == 616161)
        assert inj["weight"] == 1.0
        assert any("wasn't linked" in w for w in rec3["warnings"])

        # no usable showcase at all -> download-only mode with a note
        bare = dict(lora_info)
        bare["images"] = [{"type": "video", "url": "clip.mp4"}]
        version_store[616161] = bare
        page4 = civitai_api.resolve_model_page(
            "https://civitai.red/models/424242?modelVersionId=616161")
        assert page4["recipe"] is None and page4["showcase"] is None
        assert any("No generation data" in n for n in page4["notes"])

        # bare API download link -> model recovered from the version payload
        version_store[616161] = lora_showcase
        page5 = civitai_api.resolve_model_page(
            "https://civitai.red/api/download/models/616161")
        assert page5["model"]["model_id"] == 424242
        assert page5["model"]["name"] == "Example LoRA [SDXL]"
        assert [v["id"] for v in page5["versions"]] == [616161, 700001]
        assert page5["recipe"] is not None
    finally:
        (civitai_api.fetch_model, civitai_api.fetch_model_version,
         civitai_api.fetch_model_version_trpc, civitai_api.fetch_version_gallery) = saved

    summarize(recipe)
    print("\nOFFLINE SELF-TEST PASSED ✔")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("urls", nargs="*", help="Civitai image URLs or bare IDs")
    ap.add_argument("--json", action="store_true", help="dump full recipe JSON instead of the summary")
    ap.add_argument("--offline", action="store_true", help="run the no-network fixture self-test")
    args = ap.parse_args()

    if args.offline:
        return run_offline()
    return run_online(args.urls or DEFAULT_URLS, args.json)


if __name__ == "__main__":
    raise SystemExit(main())
