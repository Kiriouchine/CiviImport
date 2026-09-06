# CiviImport — recipe to ComfyUI workflow graph
#
# Copyright (c) 2026 Sev Kiriouchine
# SPDX-License-Identifier: MIT
# Part of CiviImport: https://github.com/Kiriouchine/CiviImport

"""CiviImport — recipe -> ComfyUI workflow JSON (graph format).  (STEP 3)

Emits the format the canvas itself saves/loads, so the frontend can hand the
result straight to app.loadGraphData() and the whole graph appears, editable.

Node chain (vanilla core nodes only — no runtime dependency on this pack):

  Note (provenance + trigger words + warnings)
  CheckpointLoaderSimple
    -> LoraLoader × N                (strength_model = strength_clip = weight)
    -> CLIPSetLastLayer              (only if clip_skip > 1; value = -clip_skip)
    -> CLIPTextEncode ("Positive prompt") / CLIPTextEncode ("Negative prompt")
  EmptyLatentImage (width, height, batch 1)
  KSampler (seed fixed, steps, cfg, mapped sampler/scheduler, denoise)
  VAEDecode (checkpoint VAE) -> SaveImage (CiviImport/civitai_<id>)

Widget values use resource['local']['filename'] when the file exists locally
(the exact string ComfyUI's combo expects, subfolders included); otherwise the
Civitai file name, which is what the file will be called after download.

Graph-format details that matter here: `widgets_values` are POSITIONAL per
node type (KSampler includes the seed's `control_after_generate` slot), and
every link is bookkept three times — in the top-level `links` array, in the
origin's `outputs[slot].links`, and in the target's `inputs[slot].link`.
"""

import os
import re

LORA_KINDS = {"lora", "locon", "dora", "lycoris"}

# Hires default when neither the metadata nor the panel supplies a factor:
# a cheap but visible bump (×1.2 ≈ 1.44× the pixels, vs ×1.5's 2.25×).
DEFAULT_HIRES_FACTOR = 1.2

_X0 = 30      # left margin
_COL = 360    # horizontal step for the loader chain


class BuildError(Exception):
    """User-facing build failure."""


# ---------------------------------------------------------------------------
# Minimal litegraph serializer
# ---------------------------------------------------------------------------

class _Graph:
    def __init__(self):
        self.nodes: list[dict] = []
        self.links: list[list] = []
        self._next_node = 0
        self._next_link = 0

    def node(self, type_, pos, size, widgets=None, inputs=(), outputs=(),
             title=None, color=None, bgcolor=None, props=None):
        self._next_node += 1
        n = {
            "id": self._next_node,
            "type": type_,
            "pos": [int(pos[0]), int(pos[1])],
            "size": [int(size[0]), int(size[1])],
            "flags": {},
            "order": len(self.nodes),
            "mode": 0,
            "inputs": [{"name": nm, "type": t, "link": None} for nm, t in inputs],
            "outputs": [{"name": nm, "type": t, "links": [], "slot_index": i}
                        for i, (nm, t) in enumerate(outputs)],
            "properties": {"Node name for S&R": type_} if props is None else props,
            "widgets_values": list(widgets) if widgets is not None else [],
        }
        if title:
            n["title"] = title
        if color:
            n["color"] = color
        if bgcolor:
            n["bgcolor"] = bgcolor
        self.nodes.append(n)
        return n

    def link(self, src, src_slot, dst, dst_slot):
        self._next_link += 1
        lid = self._next_link
        ltype = src["outputs"][src_slot]["type"]
        self.links.append([lid, src["id"], src_slot, dst["id"], dst_slot, ltype])
        src["outputs"][src_slot]["links"].append(lid)
        dst["inputs"][dst_slot]["link"] = lid

    def to_workflow(self, extra=None):
        return {
            "last_node_id": self._next_node,
            "last_link_id": self._next_link,
            "nodes": self.nodes,
            "links": self.links,
            "groups": [],
            "config": {},
            "extra": extra or {},
            "version": 0.4,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_size(base_model: str | None) -> tuple[int, int]:
    bm = (base_model or "").lower()
    if any(k in bm for k in ("sdxl", "pony", "illustrious", "noobai", "flux")):
        return 1024, 1024
    return 512, 512


def _resource_widget_name(res: dict) -> tuple[str | None, bool]:
    """(widget value, present locally). Local listing string wins."""
    local = res.get("local") or {}
    if local.get("present") and local.get("filename"):
        return local["filename"], True
    return (res.get("file") or {}).get("name"), False


def _apply_embedding_syntax(recipe: dict, warnings: list[str]) -> tuple[str, str]:
    """A1111 references embeddings by bare token; ComfyUI needs 'embedding:<file stem>'.
    Rewrites exact-token occurrences for every linked TextualInversion resource."""
    prompt = recipe.get("prompt") or ""
    negative = recipe.get("negative_prompt") or ""
    for res in recipe.get("resources") or []:
        if res.get("kind") != "textualinversion" or res.get("system"):
            continue
        name, _present = _resource_widget_name(res)
        if not name:
            continue
        stem = os.path.splitext(os.path.basename(name))[0]
        pat = re.compile(r"(?<!embedding:)\b" + re.escape(stem) + r"\b", re.IGNORECASE)
        prompt, n1 = pat.subn(f"embedding:{stem}", prompt)
        negative, n2 = pat.subn(f"embedding:{stem}", negative)
        if n1 + n2:
            warnings.append(f"Rewrote '{stem}' as 'embedding:{stem}' ({n1 + n2}×) for ComfyUI.")
        else:
            warnings.append(
                f"Embedding '{res.get('model_name') or stem}' is linked but its token wasn't "
                f"found in the prompts — add 'embedding:{stem}' manually if needed."
            )
    return prompt, negative


def _note_text(recipe: dict, sampler: dict, upscale_plan: dict | None = None) -> str:
    src = recipe.get("source") or {}
    triggers: list[str] = []
    for r in recipe.get("resources") or []:
        for w in r.get("trained_words") or []:
            if w not in triggers:
                triggers.append(w)
    lines = [
        f"Civitai import — image #{src.get('image_id') or '?'} by {src.get('username') or '?'}",
        src.get("url") or "",
        f"Base model: {src.get('base_model') or '?'} · sampler: "
        f"{sampler.get('civitai') or sampler.get('comfy_sampler')} · "
        f"original quantity: {recipe.get('quantity') or 1}",
    ]
    if upscale_plan:
        sp = upscale_plan.get("second_pass") or {}
        lines.append(
            f"Upscale: {upscale_plan['label']} ×{upscale_plan['factor']:g} → "
            f"{upscale_plan['target'][0]}×{upscale_plan['target'][1]}"
            + (f" + refine pass ({sp['steps']} steps @ denoise {sp['denoise']:g})" if sp else "")
            + (" · added by request (not in the source metadata)"
               if upscale_plan.get("synthetic") else "")
        )
    if triggers:
        lines.append("Trigger words: " + ", ".join(triggers))
    for w in recipe.get("warnings") or []:
        lines.append(f"! {w}")
    return "\n".join(l for l in lines if l)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_workflow(recipe: dict, options: dict | None = None) -> tuple[dict, dict]:
    """Returns (workflow, info). Raises BuildError with a friendly message.

    `options` (all optional; the panel sends them, step 7):
      upscale_factor  float 1–8 — hires target = original size × factor,
                      snapped to the /8 latent grid; wins over metadata hints
      add_upscaler    bool — synthesize the reference hires chain when the
                      recipe has no upscaler resource
      upscaler_file   str — local upscale-model filename for that chain
    """
    if not isinstance(recipe, dict):
        raise BuildError("Invalid recipe payload.")
    warnings: list[str] = []
    resources = recipe.get("resources") or []

    checkpoints = [r for r in resources if r.get("kind") == "checkpoint"]
    if not checkpoints:
        raise BuildError(
            "This recipe has no checkpoint resource, so a runnable graph can't be built yet. "
            "Hash-based resolution for off-site images arrives with step 5."
        )
    ckpt = checkpoints[0]
    if len(checkpoints) > 1:
        warnings.append("Multiple checkpoints linked; used the first.")

    loras = [r for r in resources if r.get("kind") in LORA_KINDS]
    for r in resources:
        if r.get("kind") not in LORA_KINDS | {"checkpoint", "textualinversion", "upscaler"}:
            warnings.append(
                f"Resource '{r.get('model_name') or r.get('version_id')}' "
                f"({r.get('kind')}) isn't wired automatically yet."
            )

    prompt_text, negative_text = _apply_embedding_syntax(recipe, warnings)

    # Parameters -------------------------------------------------------------
    seed = recipe.get("seed")
    seed = int(seed) if isinstance(seed, (int, float)) and seed >= 0 else -1
    seed_control = "fixed" if seed >= 0 else "randomize"
    if seed < 0:
        seed = 0
        warnings.append("No seed in metadata; seed set to 0 with control 'randomize'.")

    width, height = recipe.get("width"), recipe.get("height")
    if not width or not height:
        width, height = _default_size((recipe.get("source") or {}).get("base_model"))
        warnings.append(f"No size in metadata; defaulted to {width}×{height}.")

    steps = int(recipe.get("steps") or 20)
    cfg = float(recipe.get("cfg") or 7.0)
    sampler = recipe.get("sampler") or {}
    sampler_name = sampler.get("comfy_sampler") or "euler"
    scheduler = sampler.get("comfy_scheduler") or "normal"
    raw_denoise = (recipe.get("raw_meta") or {}).get("denoise")
    try:
        denoise = float(raw_denoise)
        if not (0.5 <= denoise <= 1.0):
            raise ValueError
    except (TypeError, ValueError):
        if raw_denoise is not None:
            warnings.append(
                f"Metadata denoise {raw_denoise!r} looks like a hires/img2img value; "
                "base pass forced to denoise 1.0 (low denoise on an empty latent "
                "yields a flat grey image)."
            )
        denoise = 1.0

    clip_skip = recipe.get("clip_skip")
    use_clip_skip = isinstance(clip_skip, int) and clip_skip > 1

    # Nodes -------------------------------------------------------------
    src = recipe.get("source") or {}
    missing_files: list[str] = []

    # Upscale stage plan (resource kind 'upscaler', e.g. Remacri). The final
    # size is the ORIGINAL size × factor, snapped to the /8 latent grid and
    # computed here rather than in-graph: vanilla ComfyUI has no arithmetic
    # nodes, ImageScale with explicit ints stays correct whatever the model's
    # native scale is, and VAEEncode silently crops non-/8 sizes anyway.
    opt = options or {}
    user_factor = opt.get("upscale_factor")
    if user_factor is not None:
        try:
            user_factor = float(user_factor)
            if not (1.0 <= user_factor <= 8.0):
                raise ValueError
        except (TypeError, ValueError):
            warnings.append(f"Requested upscale factor {opt.get('upscale_factor')!r} is "
                            "out of range (1–8); used the metadata/default value instead.")
            user_factor = None

    def _snap8(v: float) -> int:
        return max(8, int(round(v / 8)) * 8)

    def _hires_plan(model_name: str, label: str, factor: float) -> dict:
        # Second (refine) sampling pass, following the reference hires
        # workflow: seed 0, ~1/3 of the base steps, cfg slightly below the
        # base, same sampler/scheduler, low denoise. Metadata hints win.
        steps2 = recipe.get("hires_steps_hint")
        try:
            steps2 = int(steps2)
            if steps2 < 1:
                raise ValueError
        except (TypeError, ValueError):
            steps2 = max(4, steps // 3)
        denoise2 = recipe.get("hires_denoise_hint")
        try:
            denoise2 = float(denoise2)
            if not (0.0 < denoise2 <= 1.0):
                raise ValueError
        except (TypeError, ValueError):
            denoise2 = 0.14
        return {
            "model": model_name,
            "label": label,
            "factor": factor,
            "target": [_snap8(int(width) * factor), _snap8(int(height) * factor)],
            "second_pass": {
                "seed": 0,
                "steps": steps2,
                "cfg": round(max(1.0, cfg - 0.8), 1),
                "denoise": denoise2,
            },
        }

    upscale_plan = None
    upscalers = [r for r in resources if r.get("kind") == "upscaler"]
    if upscalers:
        if len(upscalers) > 1:
            warnings.append("Multiple upscalers linked; used the first.")
        up = upscalers[0]
        up_name, up_present = _resource_widget_name(up)
        if not up_name:
            warnings.append(f"Upscaler '{up.get('model_name') or up.get('version_id')}' "
                            "has no file info; upscale stage skipped.")
        else:
            if not up_present:
                missing_files.append(up_name)
            factor = user_factor
            if factor is None:
                factor = recipe.get("upscale_factor_hint")
                try:
                    factor = float(factor)
                    if factor <= 1.0:
                        raise ValueError
                except (TypeError, ValueError):
                    factor = DEFAULT_HIRES_FACTOR
                    warnings.append(
                        f"No upscale factor in metadata; defaulted to "
                        f"×{DEFAULT_HIRES_FACTOR:g} — adjust it in the panel or "
                        "edit the Resize node."
                    )
            upscale_plan = _hires_plan(up_name, up.get("model_name") or up_name, factor)
    elif opt.get("add_upscaler"):
        # The URL had no upscaler — the panel asked for the reference hires
        # chain anyway (toggle, on by default). The upscale model is whatever
        # the caller found locally (Remacri preferred); a placeholder if none.
        up_name = opt.get("upscaler_file")
        if up_name:
            label = str(up_name).replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
        else:
            up_name = "4x_foolhardy_Remacri.pth"
            label = "Remacri (not installed)"
            warnings.append(
                "No upscale model found in your upscale_models folder — the loader "
                "points at '4x_foolhardy_Remacri.pth'; drop a model in (or toggle "
                "the hires chain off) and regenerate."
            )
            missing_files.append(up_name)
        factor = user_factor if user_factor is not None else DEFAULT_HIRES_FACTOR
        upscale_plan = _hires_plan(up_name, label, factor)
        upscale_plan["synthetic"] = True
        warnings.append(f"Hires chain added by request (×{factor:g}, {label}) — "
                        "this image wasn't upscaled on-site.")

    g = _Graph()

    g.node("Note", (_X0, 40), (470, 180), widgets=[_note_text(recipe, sampler, upscale_plan)],
           title="Civitai import", color="#432", bgcolor="#653", props={})

    ckpt_name, ckpt_present = _resource_widget_name(ckpt)
    if not ckpt_name:
        raise BuildError("The checkpoint resource has no file information; cannot fill the loader.")
    if not ckpt_present:
        missing_files.append(ckpt_name)
    n_ckpt = g.node("CheckpointLoaderSimple", (_X0, 260), (340, 100), widgets=[ckpt_name],
                    outputs=[("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")])

    model_src, model_slot = n_ckpt, 0
    clip_src, clip_slot = n_ckpt, 1
    n_added = 0
    for lr in loras:
        name, present = _resource_widget_name(lr)
        if not name:
            warnings.append(f"LoRA '{lr.get('model_name') or lr.get('version_id')}' has no file info; skipped.")
            continue
        if not present:
            missing_files.append(name)
        try:
            weight = float(lr.get("weight"))
        except (TypeError, ValueError):
            weight = 1.0
        n_added += 1
        n_lora = g.node("LoraLoader", (_X0 + _COL * n_added, 260), (330, 130),
                        widgets=[name, weight, weight],
                        inputs=[("model", "MODEL"), ("clip", "CLIP")],
                        outputs=[("MODEL", "MODEL"), ("CLIP", "CLIP")],
                        title=f"LoRA · {lr.get('model_name') or name}")
        g.link(model_src, model_slot, n_lora, 0)
        g.link(clip_src, clip_slot, n_lora, 1)
        model_src, model_slot = n_lora, 0
        clip_src, clip_slot = n_lora, 1

    x_after = _X0 + _COL * (n_added + 1)
    if use_clip_skip:
        n_skip = g.node("CLIPSetLastLayer", (x_after, 300), (315, 60), widgets=[-int(clip_skip)],
                        inputs=[("clip", "CLIP")], outputs=[("CLIP", "CLIP")])
        g.link(clip_src, clip_slot, n_skip, 0)
        clip_src, clip_slot = n_skip, 0
        x_enc = x_after + _COL
    else:
        x_enc = x_after

    n_pos = g.node("CLIPTextEncode", (x_enc, 110), (420, 210), widgets=[prompt_text],
                   inputs=[("clip", "CLIP")], outputs=[("CONDITIONING", "CONDITIONING")],
                   title="Positive prompt", color="#232", bgcolor="#353")
    n_neg = g.node("CLIPTextEncode", (x_enc, 370), (420, 210), widgets=[negative_text],
                   inputs=[("clip", "CLIP")], outputs=[("CONDITIONING", "CONDITIONING")],
                   title="Negative prompt", color="#322", bgcolor="#533")
    g.link(clip_src, clip_slot, n_pos, 0)
    g.link(clip_src, clip_slot, n_neg, 0)

    n_latent = g.node("EmptyLatentImage", (x_enc, 630), (315, 110),
                      widgets=[int(width), int(height), 1],
                      outputs=[("LATENT", "LATENT")])

    x_ks = x_enc + 470
    n_ks = g.node("KSampler", (x_ks, 260), (315, 262),
                  widgets=[seed, seed_control, steps, cfg, sampler_name, scheduler, denoise],
                  inputs=[("model", "MODEL"), ("positive", "CONDITIONING"),
                          ("negative", "CONDITIONING"), ("latent_image", "LATENT")],
                  outputs=[("LATENT", "LATENT")])
    g.link(model_src, model_slot, n_ks, 0)
    g.link(n_pos, 0, n_ks, 1)
    g.link(n_neg, 0, n_ks, 2)
    g.link(n_latent, 0, n_ks, 3)

    x_out = x_ks + 360
    prefix = f"CiviImport/civitai_{src.get('image_id') or 'import'}"

    n_dec = g.node("VAEDecode", (x_out, 140), (210, 46),
                   inputs=[("samples", "LATENT"), ("vae", "VAE")],
                   outputs=[("IMAGE", "IMAGE")])
    g.link(n_ks, 0, n_dec, 0)
    g.link(n_ckpt, 2, n_dec, 1)

    n_save = g.node("SaveImage", (x_out, 240), (320, 280), widgets=[prefix],
                    inputs=[("images", "IMAGE")])
    g.link(n_dec, 0, n_save, 0)

    if upscale_plan:
        sp = upscale_plan["second_pass"]
        n_uml = g.node("UpscaleModelLoader", (x_out, 570), (315, 60),
                       widgets=[upscale_plan["model"]],
                       outputs=[("UPSCALE_MODEL", "UPSCALE_MODEL")],
                       title=f"Upscaler · {upscale_plan['label']}")
        n_upm = g.node("ImageUpscaleWithModel", (x_out + 360, 140), (240, 60),
                       inputs=[("upscale_model", "UPSCALE_MODEL"), ("image", "IMAGE")],
                       outputs=[("IMAGE", "IMAGE")])
        g.link(n_uml, 0, n_upm, 0)
        g.link(n_dec, 0, n_upm, 1)

        n_resize = g.node("ImageScale", (x_out + 360, 250), (315, 130),
                          widgets=["bilinear", upscale_plan["target"][0],
                                   upscale_plan["target"][1], "center"],
                          inputs=[("image", "IMAGE")],
                          outputs=[("IMAGE", "IMAGE")],
                          title=f"Resize to ×{upscale_plan['factor']:g} of original")
        g.link(n_upm, 0, n_resize, 0)

        n_venc = g.node("VAEEncode", (x_out + 360, 430), (210, 46),
                        inputs=[("pixels", "IMAGE"), ("vae", "VAE")],
                        outputs=[("LATENT", "LATENT")])
        g.link(n_resize, 0, n_venc, 0)
        g.link(n_ckpt, 2, n_venc, 1)

        n_ks2 = g.node("KSampler", (x_out + 720, 140), (315, 262),
                       widgets=[sp["seed"], "fixed", sp["steps"], sp["cfg"],
                                sampler_name, scheduler, sp["denoise"]],
                       inputs=[("model", "MODEL"), ("positive", "CONDITIONING"),
                               ("negative", "CONDITIONING"), ("latent_image", "LATENT")],
                       outputs=[("LATENT", "LATENT")],
                       title="Hires refine")
        g.link(model_src, model_slot, n_ks2, 0)
        g.link(n_pos, 0, n_ks2, 1)
        g.link(n_neg, 0, n_ks2, 2)
        g.link(n_venc, 0, n_ks2, 3)

        n_dec2 = g.node("VAEDecode", (x_out + 1080, 140), (210, 46),
                        inputs=[("samples", "LATENT"), ("vae", "VAE")],
                        outputs=[("IMAGE", "IMAGE")])
        g.link(n_ks2, 0, n_dec2, 0)
        g.link(n_ckpt, 2, n_dec2, 1)

        n_save2 = g.node("SaveImage", (x_out + 1080, 240), (320, 280),
                         widgets=[prefix + "_hires"],
                         inputs=[("images", "IMAGE")])
        g.link(n_dec2, 0, n_save2, 0)

    workflow = g.to_workflow(extra={
        "CiviImport": {"image_id": src.get("image_id"), "source_url": src.get("url"), "builder": 1},
    })
    info = {
        "node_count": len(workflow["nodes"]),
        "lora_count": n_added,
        "missing_files": missing_files,
        "seed_control": seed_control,
        "upscale": None if upscale_plan is None else {
            k: upscale_plan[k]
            for k in ("model", "factor", "target", "second_pass", "synthetic")
            if k in upscale_plan
        },
        "warnings": warnings,
    }
    return workflow, info
