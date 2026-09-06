#!/usr/bin/env python3
# CiviImport — self-tests for the graph builder
#
# Copyright (c) 2026 Sev Kiriouchine
# SPDX-License-Identifier: MIT
# Part of CiviImport: https://github.com/Kiriouchine/CiviImport

"""Offline structural tests for graph_builder — no ComfyUI, no network.

    python test_builder.py
"""

import copy
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import civitai_api  # noqa: E402
import graph_builder  # noqa: E402
import local_models  # noqa: E402
from test_resolver import FIXTURE_ITEM, FIXTURE_VERSIONS  # noqa: E402

EXPECTED_WIDGETS = {
    "CheckpointLoaderSimple": 1, "LoraLoader": 3, "CLIPSetLastLayer": 1,
    "CLIPTextEncode": 1, "EmptyLatentImage": 3, "KSampler": 7,
    "VAEDecode": 0, "SaveImage": 1, "Note": 1, "VAEEncode": 0,
    "UpscaleModelLoader": 1, "ImageUpscaleWithModel": 0, "ImageScale": 4,
}

REMACRI = {
    "version_id": 164821, "kind": "upscaler", "folder": "upscale_models",
    "model_name": "Remacri", "version_name": "Original", "weight": None,
    "trained_words": [],
    "file": {"name": "4x_foolhardy_Remacri.safetensors", "size_kb": 65300,
             "hashes": {"AutoV2": "AC3E6CC5B5"}, "download_url": None},
    "local": {"present": False, "filename": None, "method": None},
}

REMACRI_VERSION = {  # /model-versions/164821 response shape
    "id": 164821, "modelId": 147759, "name": "Original", "baseModel": "Other",
    "trainedWords": [], "downloadUrl": "https://civitai.com/api/download/models/164821",
    "model": {"name": "Remacri", "type": "Upscaler"},
    "files": [{"name": "4x_foolhardy_Remacri.safetensors", "sizeKB": 65300.0,
               "primary": True, "hashes": {"AutoV2": "AC3E6CC5B5"},
               "downloadUrl": "https://civitai.com/api/download/models/164821"}],
}


def validate(wf: dict) -> None:
    """Structural invariants any loadable graph-format workflow must satisfy."""
    ids = {}
    for n in wf["nodes"]:
        assert n["id"] not in ids, f"duplicate node id {n['id']}"
        ids[n["id"]] = n
        exp = EXPECTED_WIDGETS.get(n["type"])
        assert exp is not None, f"unexpected node type {n['type']}"
        got = len(n.get("widgets_values", []))
        assert got == exp, f"{n['type']}: {got} widgets, expected {exp}"
    assert wf["last_node_id"] == max(ids)

    seen = set()
    for lid, a, aslot, b, bslot, ltype in wf["links"]:
        assert lid not in seen, f"duplicate link id {lid}"
        seen.add(lid)
        out = ids[a]["outputs"][aslot]
        inp = ids[b]["inputs"][bslot]
        assert lid in out["links"], f"link {lid} missing from origin outputs"
        assert inp["link"] == lid, f"link {lid} missing from target input"
        assert out["type"] == ltype == inp["type"], f"type mismatch on link {lid}"
    assert wf["last_link_id"] == max(seen)

    for n in wf["nodes"]:
        for inp in n.get("inputs", []):
            assert inp["link"] is not None, f"unlinked input {n['type']}.{inp['name']}"

    json.dumps(wf)  # must be serializable


def fixture_recipe(local_ckpt: bool = True) -> dict:
    recipe = civitai_api.build_recipe(
        copy.deepcopy(FIXTURE_ITEM), copy.deepcopy(FIXTURE_VERSIONS),
        source_url="https://civitai.com/images/135891361",
    )
    with tempfile.TemporaryDirectory() as root:
        if local_ckpt:
            d = os.path.join(root, "checkpoints")
            os.makedirs(d)
            open(os.path.join(d, "moonartCauldronMix_sdxlV10.safetensors"), "wb").close()
        os.environ["CIVIIMPORT_MODELS_ROOT"] = root
        try:
            local_models.annotate_local_status(recipe)
        finally:
            os.environ.pop("CIVIIMPORT_MODELS_ROOT", None)
    return recipe


def main() -> int:
    print("Running builder self-test (fixtures, no network)…")

    # --- base case: ckpt (present) + 2 LoRAs (missing) + clip skip 2 --------
    recipe = fixture_recipe()
    wf, info = graph_builder.build_workflow(recipe)
    validate(wf)

    types = [n["type"] for n in wf["nodes"]]
    assert types.count("LoraLoader") == 2 and types.count("CLIPSetLastLayer") == 1

    by_type = lambda t: next(n for n in wf["nodes"] if n["type"] == t)  # noqa: E731
    ks = by_type("KSampler")
    assert ks["widgets_values"] == [285342747, "fixed", 23, 5.5, "euler", "normal", 1.0], ks["widgets_values"]
    assert by_type("EmptyLatentImage")["widgets_values"] == [832, 1216, 1]
    assert by_type("CLIPSetLastLayer")["widgets_values"] == [-2]
    assert by_type("CheckpointLoaderSimple")["widgets_values"] == ["moonartCauldronMix_sdxlV10.safetensors"]

    loras = [n for n in wf["nodes"] if n["type"] == "LoraLoader"]
    assert loras[0]["widgets_values"][1:] == [0.8, 0.8]
    assert loras[1]["widgets_values"][1:] == [0.6, 0.6]

    link_by_id = {l[0]: l for l in wf["links"]}
    # KSampler.model must come from the LAST LoRA in the chain
    assert link_by_id[ks["inputs"][0]["link"]][1] == loras[-1]["id"]
    # both encoders must be fed from CLIPSetLastLayer
    skip_id = by_type("CLIPSetLastLayer")["id"]
    for n in wf["nodes"]:
        if n["type"] == "CLIPTextEncode":
            assert link_by_id[n["inputs"][0]["link"]][1] == skip_id
    # VAE comes straight from the checkpoint
    dec = by_type("VAEDecode")
    assert link_by_id[dec["inputs"][1]["link"]][1] == by_type("CheckpointLoaderSimple")["id"]

    assert info["missing_files"] == [
        "decorative_leather_sdxl.safetensors",
        "aidmaImageUpgrader_sdxl_v03.safetensors",
    ], info["missing_files"]
    assert info["upscale"] is None
    print(f"  base case OK — {info['node_count']} nodes, {len(wf['links'])} links, "
          f"{len(info['missing_files'])} missing file(s)")

    # --- no clip skip, no seed ------------------------------------------
    r2 = fixture_recipe()
    r2["clip_skip"] = None
    r2["seed"] = -1
    wf2, _info2 = graph_builder.build_workflow(r2)
    validate(wf2)
    assert not any(n["type"] == "CLIPSetLastLayer" for n in wf2["nodes"])
    ks2 = next(n for n in wf2["nodes"] if n["type"] == "KSampler")
    assert ks2["widgets_values"][0] == 0 and ks2["widgets_values"][1] == "randomize"
    print("  no-clip-skip / no-seed case OK")

    # --- embedding token rewrite ------------------------------------------
    r3 = fixture_recipe()
    r3["negative_prompt"] += ", zPDXL3, (zPDXL3:0.8)"
    r3["resources"].append({
        "version_id": 1, "kind": "textualinversion", "folder": "embeddings",
        "model_name": "zPDXL3", "version_name": "v3", "weight": None, "trained_words": [],
        "file": {"name": "zPDXL3.safetensors", "size_kb": 100, "hashes": {}, "download_url": None},
        "local": {"present": False, "filename": None, "method": None},
    })
    wf3, info3 = graph_builder.build_workflow(r3)
    validate(wf3)
    neg = next(n for n in wf3["nodes"] if n.get("title") == "Negative prompt")
    assert "embedding:zPDXL3, (embedding:zPDXL3:0.8)" in neg["widgets_values"][0], neg["widgets_values"][0]
    assert any(t["type"] == "CLIPTextEncode" for t in wf3["nodes"])
    assert any("Rewrote 'zPDXL3'" in w for w in info3["warnings"])
    print("  embedding rewrite OK")

    # --- no checkpoint -> BuildError ---------------------------------------
    r4 = fixture_recipe()
    r4["resources"] = [r for r in r4["resources"] if r["kind"] != "checkpoint"]
    try:
        graph_builder.build_workflow(r4)
        raise AssertionError("expected BuildError for missing checkpoint")
    except graph_builder.BuildError:
        print("  no-checkpoint error OK")

    # --- upscaler wiring (Remacri) -----------------------------------------
    # hint extraction happens in the resolver:
    item = copy.deepcopy(FIXTURE_ITEM)
    item["meta"]["Hires upscale"] = "2"
    rh = civitai_api.build_recipe(item, copy.deepcopy(FIXTURE_VERSIONS))
    assert rh["upscale_factor_hint"] == 2.0

    r5 = fixture_recipe()
    r5["resources"].append(copy.deepcopy(REMACRI))
    wf5, info5 = graph_builder.build_workflow(r5)
    validate(wf5)
    t5 = [n["type"] for n in wf5["nodes"]]
    for t in ("UpscaleModelLoader", "ImageUpscaleWithModel", "ImageScale", "VAEEncode"):
        assert t in t5, f"missing {t}"
    assert t5.count("KSampler") == 2 and t5.count("VAEDecode") == 2 and t5.count("SaveImage") == 2

    resize = next(n for n in wf5["nodes"] if n["type"] == "ImageScale")
    # base 832×1216 × default 1.2 → 998.4×1459.2 → snapped to the /8 grid = 1000×1456
    assert resize["widgets_values"] == ["bilinear", 1000, 1456, "center"], resize["widgets_values"]

    lb5 = {l[0]: l for l in wf5["links"]}
    by_id5 = {n["id"]: n for n in wf5["nodes"]}
    upm = next(n for n in wf5["nodes"] if n["type"] == "ImageUpscaleWithModel")
    uml5 = next(n for n in wf5["nodes"] if n["type"] == "UpscaleModelLoader")
    venc = next(n for n in wf5["nodes"] if n["type"] == "VAEEncode")
    ckpt5 = next(n for n in wf5["nodes"] if n["type"] == "CheckpointLoaderSimple")
    assert uml5["widgets_values"] == ["4x_foolhardy_Remacri.safetensors"]
    assert lb5[upm["inputs"][0]["link"]][1] == uml5["id"]
    assert lb5[resize["inputs"][0]["link"]][1] == upm["id"]
    assert lb5[venc["inputs"][0]["link"]][1] == resize["id"]      # pixels from resize
    assert lb5[venc["inputs"][1]["link"]][1] == ckpt5["id"]       # vae from checkpoint

    # second KSampler = the one whose latent comes from VAEEncode
    samplers = [n for n in wf5["nodes"] if n["type"] == "KSampler"]
    ks2 = next(n for n in samplers if lb5[n["inputs"][3]["link"]][1] == venc["id"])
    ks1 = next(n for n in samplers if n is not ks2)
    assert ks1["widgets_values"] == [285342747, "fixed", 23, 5.5, "euler", "normal", 1.0]
    # seed 0 fixed, steps 23//3=7, cfg 5.5-0.8=4.7, same sampler/scheduler, denoise 0.14
    assert ks2["widgets_values"] == [0, "fixed", 7, 4.7, "euler", "normal", 0.14], ks2["widgets_values"]
    loras5 = [n for n in wf5["nodes"] if n["type"] == "LoraLoader"]
    assert lb5[ks2["inputs"][0]["link"]][1] == loras5[-1]["id"]   # same lora-chained model
    # same conditioning as pass 1
    assert lb5[ks2["inputs"][1]["link"]][1] == lb5[ks1["inputs"][1]["link"]][1]
    assert lb5[ks2["inputs"][2]["link"]][1] == lb5[ks1["inputs"][2]["link"]][1]

    # upscaled image goes through decode2 into the _hires save; base save stays
    dec2 = by_id5[lb5[next(n for n in wf5["nodes"] if n["type"] == "SaveImage"
                           and n["widgets_values"][0].endswith("_hires"))["inputs"][0]["link"]][1]]
    assert dec2["type"] == "VAEDecode" and lb5[dec2["inputs"][0]["link"]][1] == ks2["id"]
    base_save = next(n for n in wf5["nodes"] if n["type"] == "SaveImage"
                     and not n["widgets_values"][0].endswith("_hires"))
    assert by_id5[lb5[base_save["inputs"][0]["link"]][1]]["type"] == "VAEDecode"

    # Branding: outputs land in a CiviImport/ subfolder and the workflow
    # carries a CiviImport provenance block (regression — a bulk rename once
    # left these lower-cased and out of step with the pack name).
    for save in (n for n in wf5["nodes"] if n["type"] == "SaveImage"):
        assert save["widgets_values"][0].startswith("CiviImport/civitai_"), \
            save["widgets_values"]
    assert "CiviImport" in (wf5.get("extra") or {}), list((wf5.get("extra") or {}))

    assert "4x_foolhardy_Remacri.safetensors" in info5["missing_files"]
    assert info5["upscale"] == {"model": "4x_foolhardy_Remacri.safetensors",
                                "factor": 1.2, "target": [1000, 1456],
                                "second_pass": {"seed": 0, "steps": 7, "cfg": 4.7, "denoise": 0.14}}
    assert any("defaulted to ×1.2" in w for w in info5["warnings"])

    # explicit factor + metadata hints for the refine pass
    item6 = copy.deepcopy(FIXTURE_ITEM)
    item6["meta"]["civitaiResources"].append({"type": "upscaler", "modelVersionId": 164821})
    item6["meta"]["Hires upscale"] = "2"
    item6["meta"]["Hires steps"] = "12"
    item6["meta"]["Denoising strength"] = "0.3"
    versions6 = copy.deepcopy(FIXTURE_VERSIONS)
    versions6[164821] = copy.deepcopy(REMACRI_VERSION)
    r6 = civitai_api.build_recipe(item6, versions6)
    assert r6["upscale_factor_hint"] == 2.0
    assert r6["hires_steps_hint"] == 12 and r6["hires_denoise_hint"] == 0.3
    local_models.annotate_local_status(r6)
    wf6, info6 = graph_builder.build_workflow(r6)
    validate(wf6)
    resize6 = next(n for n in wf6["nodes"] if n["type"] == "ImageScale")
    assert resize6["widgets_values"] == ["bilinear", 1664, 2432, "center"], resize6["widgets_values"]
    assert info6["upscale"]["second_pass"] == {"seed": 0, "steps": 12, "cfg": 4.7, "denoise": 0.3}
    assert not any("defaulted" in w for w in info6["warnings"])
    print("  upscaler refine chain OK (default ×1.2 snapped and hinted ×2/12 steps/0.3 denoise)")

    # --- post-upscale uploaded size snaps back to a standard bucket ---------
    item = copy.deepcopy(FIXTURE_ITEM)
    item["width"], item["height"] = 2688, 3840  # uploaded = already-upscaled
    item["meta"]["civitaiResources"].append({"type": "upscaler", "modelVersionId": 164821})
    versions = copy.deepcopy(FIXTURE_VERSIONS)
    versions[164821] = copy.deepcopy(REMACRI_VERSION)
    r7 = civitai_api.build_recipe(item, versions)
    assert (r7["width"], r7["height"]) == (832, 1216), (r7["width"], r7["height"])
    assert (r7["uploaded_width"], r7["uploaded_height"]) == (2688, 3840)
    assert any("snapped" in w for w in r7["warnings"]), r7["warnings"]

    assert civitai_api._snap_to_bucket(1856, 1280, "Illustrious") == (1216, 832)
    assert civitai_api._snap_to_bucket(2048, 2048, "Pony") == (1024, 1024)
    assert civitai_api._snap_to_bucket(1536, 1024, "SD 1.5") == (768, 512)

    local_models.annotate_local_status(r7)  # no models root set → all missing, fine
    wf7, _info7 = graph_builder.build_workflow(r7)
    validate(wf7)
    assert next(n for n in wf7["nodes"] if n["type"] == "EmptyLatentImage")["widgets_values"] == [832, 1216, 1]
    assert next(n for n in wf7["nodes"] if n["type"] == "ImageScale")["widgets_values"] == ["bilinear", 1000, 1456, "center"]
    print("  post-upscale size snap OK (2688×3840 uploaded → 832×1216 base → 1000×1456 target)")

    # --- step 7: user factor override + synthesized hires chain -------------
    # panel factor wins over hint/default (×2 of 832×1216 → 1664×2432, already /8)
    wf8, info8 = graph_builder.build_workflow(copy.deepcopy(r5), {"upscale_factor": 2.0})
    resize8 = next(n for n in wf8["nodes"] if n["type"] == "ImageScale")
    assert resize8["widgets_values"] == ["bilinear", 1664, 2432, "center"], resize8["widgets_values"]
    assert info8["upscale"]["factor"] == 2.0
    assert not any("defaulted" in w for w in info8["warnings"])

    # out-of-range user factor falls back to the metadata/default logic
    _wf8b, info8b = graph_builder.build_workflow(copy.deepcopy(r5), {"upscale_factor": 0.5})
    assert info8b["upscale"]["factor"] == 1.2
    assert any("out of range" in w for w in info8b["warnings"])

    # synthesized chain: recipe without an upscaler + explicit local file
    r9 = fixture_recipe()
    assert not any(r["kind"] == "upscaler" for r in r9["resources"])
    wf9, info9 = graph_builder.build_workflow(
        copy.deepcopy(r9),
        {"add_upscaler": True, "upscaler_file": "4x_Remacri.pth", "upscale_factor": 2.0})
    validate(wf9)
    t9 = [n["type"] for n in wf9["nodes"]]
    for t in ("UpscaleModelLoader", "ImageUpscaleWithModel", "ImageScale", "VAEEncode"):
        assert t in t9, f"missing {t}"
    assert t9.count("KSampler") == 2 and t9.count("SaveImage") == 2
    uml9 = next(n for n in wf9["nodes"] if n["type"] == "UpscaleModelLoader")
    assert uml9["widgets_values"] == ["4x_Remacri.pth"]
    resize9 = next(n for n in wf9["nodes"] if n["type"] == "ImageScale")
    assert resize9["widgets_values"] == ["bilinear", 1664, 2432, "center"]
    assert info9["upscale"]["synthetic"] is True
    assert info9["upscale"]["second_pass"]["denoise"] == 0.14
    assert any("added by request" in w for w in info9["warnings"])
    assert "4x_Remacri.pth" not in info9["missing_files"]

    # ... placeholder + missing when the caller found no local upscale model
    _wf10, info10 = graph_builder.build_workflow(copy.deepcopy(r9), {"add_upscaler": True})
    assert info10["upscale"]["model"] == "4x_foolhardy_Remacri.pth"
    assert "4x_foolhardy_Remacri.pth" in info10["missing_files"]
    assert info10["upscale"]["factor"] == 1.2
    assert info10["upscale"]["target"] == [1000, 1456]
    assert any("No upscale model found" in w for w in info10["warnings"])

    # ... and library behavior without options is unchanged (no synthesis)
    wf11, info11 = graph_builder.build_workflow(copy.deepcopy(r9))
    assert info11["upscale"] is None
    assert not any(n["type"] == "UpscaleModelLoader" for n in wf11["nodes"])
    print("  step 7 OK (factor override, /8 snap, synthesized chain, defaults preserved)")

    # --- base-pass denoise guard (grey-square protection) --------------------
    r8 = fixture_recipe()
    r8["raw_meta"]["denoise"] = 0.3
    wf8, info8 = graph_builder.build_workflow(r8)
    ks8 = next(n for n in wf8["nodes"] if n["type"] == "KSampler")
    assert ks8["widgets_values"][6] == 1.0, ks8["widgets_values"]
    assert any("forced to denoise 1.0" in w for w in info8["warnings"])
    r9 = fixture_recipe()
    r9["raw_meta"]["denoise"] = 0.97
    wf9, _info9 = graph_builder.build_workflow(r9)
    ks9 = next(n for n in wf9["nodes"] if n["type"] == "KSampler")
    assert ks9["widgets_values"][6] == 0.97, ks9["widgets_values"]

    # system embeddings produce no token-not-found warnings
    r10 = fixture_recipe()
    r10["resources"].append({
        "version_id": 250708, "kind": "textualinversion", "system": True,
        "folder": "embeddings", "model_name": "Civitai Safe Helper (Minor)",
        "version_name": "safe_pos", "weight": None, "trained_words": [],
        "file": {"name": "safe_pos.pt", "size_kb": 25, "hashes": {}, "download_url": None},
        "local": {"present": False, "filename": None, "method": None},
    })
    _wf10, info10 = graph_builder.build_workflow(r10)
    assert not any("safe_pos" in w for w in info10["warnings"]), info10["warnings"]
    print("  denoise guard + system-resource handling OK")

    print("\nBUILDER SELF-TEST PASSED ✔")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
