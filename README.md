# CiviImport

Import a Civitai image's **entire generation setup** into ComfyUI: paste an image
URL, see the checkpoint / LoRAs / sampler settings it used, check which model
files you already have, download the missing ones, and generate the full
workflow on your canvas.

**Status: v1.3.1 — all 7 steps complete.** Paste an image or model URL,
fetch the recipe, download missing models with live progress and hash
verification (renamed files recognized by hash), and generate the full
workflow — with a tunable hires upscale — on your canvas.

| Step | What | Status |
|---|---|---|
| 1 | Package scaffold: routes, topbar menu, sidebar panel | ✅ |
| 2 | Resolver: URL → recipe + local presence lights | ✅ |
| 3 | Graph builder → `app.loadGraphData` | ✅ |
| 4 | Downloader with progress + folder placement | ✅ |
| 5 | Hash matching, off-site images, target preference, publishing prep | ✅ |
| 6 | Model-page import (card + downloads + showcase starter recipe), verified downloads | ✅ |
| 7 | Hires factor control (/8-snapped, default ×1.2), synthesized upscale chain toggle | ✅ |

## Install

**Recommended — ComfyUI Manager:** search for `CiviImport` in the Manager and
install it. This is the reliable route: the Manager places the pack in the
`custom_nodes` folder of the instance you are actually running, installs
`requests` into that instance's Python environment, and handles updates.

**Manual:** copy (or `git clone`) the `CiviImport` folder into that instance's
`custom_nodes` directory and restart ComfyUI.

> **Desktop users, read this.** Desktop keeps several installs side by side
> (`...\ComfyUI-Installs\<name>\ComfyUI`) and points input, output and model
> paths at shared folders — but **`custom_nodes` is per-instance and is not
> redirected**. A pack dropped in `Documents\ComfyUI\custom_nodes` will be
> silently ignored by an instance living elsewhere. The startup log names the
> right folder:
>
> ```
> ** ComfyUI Base Folder Path: C:\Users\you\ComfyUI-Installs\a\ComfyUI
> ```
>
> so the pack belongs in `<that path>\custom_nodes\CiviImport`. If the sidebar
> button doesn't appear, run `python diagnose.py` (see Troubleshooting) — it
> works out which install is running and tells you exactly what to move.

Dependencies: only `requests` (`pip install -r requirements.txt` using the same
Python your ComfyUI runs — for Desktop that is the instance's
`.venv\Scripts\python.exe`, not a system-wide Python).

On boot you should see `[CiviImport] v1.3.1 routes registered` in the log.

## Use

1. Click the **cloud icon in the left sidebar** (or menu: **Civitai → Import
   from Civitai URL…**).
2. Paste an image URL like `https://civitai.com/images/135891361` — or a
   model page like `https://civitai.com/models/12345?modelVersionId=67890`
   (see step 4) — and hit **Fetch**.
3. The panel shows the recipe (sampler mapped to ComfyUI names, steps, CFG,
   seed, clip skip, size) and each resource with a light:
   🟢 file found in your model folders · 🔴 missing · ⚪ lookup failed.
   Each resource also links to its **model page** on Civitai, pinned to the
   exact version the image used, so you can check licences, trigger words or
   sample images before downloading.
   Each missing resource carries its **own Download button**, so you can pull
   just the checkpoint (or one LoRA, embedding, VAE…) and leave the rest;
   **Download all** above the list fetches every remaining one. Downloads
   run sequentially with live progress bars: the target folder is derived
   server-side from the model's type via `folder_paths` (so `extra_model_paths.yaml` is respected; the
   first existing configured folder for that type is used), files stream to
   `<name>.part` and are atomically renamed, then **verified against
   Civitai's published SHA-256/AutoV2** when available — a corrupt or
   swapped file is deleted and reported instead of poisoning your models
   folder, and the computed hash goes straight into the cache so hash
   matching recognizes the download immediately. Lights flip green as each
   file lands, and the loader dropdowns are refreshed automatically — **no
   ComfyUI restart needed** for new models. Most Civitai downloads require
   an API key.
   **Scan by hash** covers renamed files: it SHA-256-hashes local models
   (cached in `hash_cache.json`, validated by size+mtime, so multi-GB files
   are hashed once ever) and re-checks the lights — a model saved under a
   different name counts as present. Off-site A1111 uploads that carry only
   file hashes are resolved automatically through Civitai's by-hash lookup.
4. **Model pages** work too — paste any `/models/...` link (with or without
   `?modelVersionId=`; bare API download links also count). The panel shows
   a model card with a version dropdown, a presence light, and a
   **Download model** button, making it a general-purpose Civitai
   downloader. Below the card, the panel imports a **starter recipe** from
   the version's showcase gallery (the creator's own example images carry
   full generation data), guaranteeing the pasted model is wired into the
   resulting workflow even when the showcase metadata didn't link it. If no
   showcase image carries generation data, the card alone appears and the
   model remains downloadable.
5. **Set API key** stores your Civitai key (Account settings → API Keys) in
   `config.json` next to the pack (gitignored, plain text — treat the folder
   accordingly). The `CIVITAI_API_KEY` environment variable works as a
   fallback. A key is needed for browsing-level-gated images and, later, for
   most downloads.
6. **Generate workflow** builds the graph and loads it onto the canvas
   (after a confirm — it replaces the current graph). You get:
   `Note (provenance/trigger words) → Checkpoint → LoRA chain →
   CLIPSetLastLayer (if clip skip > 1) → prompts → EmptyLatent → KSampler
   (seed fixed) → VAEDecode → SaveImage (CiviImport/…)`. Vanilla core nodes
   only. Embedding tokens are rewritten to ComfyUI's `embedding:name` syntax.
   If the image lists an **upscaler** (e.g. Remacri), the tail becomes a
   hires-fix refine chain modeled on a reference workflow:
   `VAEDecode → SaveImage (base)` plus
   `→ Load Upscale Model + Upscale Image (using Model) → Upscale Image
   (bilinear, center; width/height = base × factor, /8-snapped) → VAEEncode →
   KSampler "Hires refine" → VAEDecode → SaveImage (…_hires)`.
   The refine sampler uses seed 0 (fixed), ~⅓ of the base steps, cfg − 0.8,
   the same sampler/scheduler, and denoise 0.14; `Hires steps`,
   `Denoising strength` and `Hires upscale` metadata override these when
   present. The panel's **Hires ×** field sets the factor directly and wins
   over metadata; without either, it defaults to **×1.2** (a cheap but
   visible bump — ~1.44× the pixels). Targets are `original size × factor`
   snapped to the /8 latent grid (`VAEEncode` would silently crop anything
   else). Core ComfyUI has no arithmetic nodes, so the multiply happens at
   build time; the resize stays correct regardless of the model's native
   scale (Remacri is ×4).
   For images that **weren't** upscaled on-site, an **add hires upscale**
   toggle (on by default) appends the same chain using your local Remacri —
   or the first upscale model found in `upscale_models`; if none exists, the
   loader points at `4x_foolhardy_Remacri.pth` as a placeholder and the
   build warns. Untick it for a plain faithful graph.
   Missing model files load anyway — their loader widgets show as invalid
   until the files exist (step 4 automates that).

## Testing the resolver without ComfyUI

```
python test_resolver.py                 # the three reference images
python test_resolver.py --json <url>    # full recipe JSON
python test_resolver.py --offline       # fixture self-test, no network
python test_downloader.py               # downloader test against a localhost fake CDN
python test_hashing.py                  # hash cache, renamed-file matching, deep scan
python test_builder.py                  # graph-builder structural tests
```

Set `CIVIIMPORT_MODELS_ROOT` (e.g. `C:\Users\maste\ComfyUI-Shared\models`) to
enable the local-presence check outside ComfyUI, and `CIVITAI_API_KEY` for
gated images.

## Endpoints

`GET /civiimport/ping` · `POST /civiimport/resolve {url}` (returns
`kind: image|model`; model pages come back as a `model_page` card plus an
optional showcase `recipe`) · `GET|POST /civiimport/apikey` ·
`POST /civiimport/build {recipe}` ·
`POST /civiimport/download {version_ids, api_host?}` ·
`POST /civiimport/hashscan {folders?}`

## Configuration (`config.json` next to the pack)

* `api_key` — set via the panel button; the `CIVITAI_API_KEY` env var is the
  fallback.
* `download_dir_hint` — optional substring. When several folders are
  configured for a model type (e.g. a shared models directory via
  `extra_model_paths.yaml`), downloads go to the first existing path whose
  full path contains this text (case-insensitive), e.g.
  `"download_dir_hint": "ComfyUI-Shared"`. Unset = first configured path.

## Fidelity expectations

Reproduction across different backends is *approximate by nature* — set
expectations before filing "the image is different" issues:

* A seed is an index into a noise generator, not an image. Different stacks
  produce different noise from the same number, and ancestral samplers
  ("Euler a") re-inject RNG noise every step, compounding divergence. Civitai
  also generates in batches: the posted image may be sample 3 of 4 under one
  recorded seed, and the metadata doesn't say which.
* What the *conditioning* controls transfers reliably: style, character,
  wardrobe, palette, quality. What the *noise* controls diverges: layout,
  subject count, camera, background specifics. Constrained recipes (character
  LoRA + plain background + portrait aspect) reproduce as near-twins; open
  scenes on square canvases reproduce as siblings.
* Civitai's pipeline includes pieces that never appear in metadata: injected
  internal embeddings (the Safe Helper pair shows greyed as "internal"), a
  hi-res refine pass (reproduced here), and a face-fix pass (not yet). Their
  displayed previews are CDN-downscaled, which reads as extra sharpness —
  compare the `_hires` output at 1:1 zoom.
* A **flat grey/black output** is not a fidelity issue: it means broken
  numbers — typically a corrupt/truncated checkpoint file, occasionally fp16
  NaN. Re-download the checkpoint and check the console for NaN warnings.
* Pony Diffusion V6 XL is famously CFG-sensitive; if colors wash out, nudge
  CFG down ~0.5 — the standard community remedy.

## Notes

* **Domains & lookup chain:** since the April 2026 split, civitai.com is the
  SFW-only view (the role civitai.green used to play) and civitai.red shows
  everything — one shared database and ID space. URLs from all three domains
  are accepted. The resolver tries, in order: the pasted domain's REST API,
  the other domain's REST API, then the site's own generation-data endpoint
  on both — the same lookup the image page uses, which also covers images
  the documented REST endpoint fails to return (a known quirk, mostly older
  IDs). If an image is viewable in a browser, it should resolve. NSFW content
  generally needs your API key plus matching browsing settings on your
  civitai.red account. The panel shows "via civitai.red" when that host
  answered.
* When an image was **upscaled on-site**, the uploaded file's dimensions are
  the post-upscale size — the resolver snaps the base generation size back to
  the standard bucket (1024×1024 / 1216×832 / 832×1216 for the SDXL family;
  512-class for SD 1.5) chosen by orientation, and the resize target is
  `base × factor`. The panel shows both sizes.
* Generated workflows (step 3) use **vanilla core nodes only** — no
  runtime dependency on this pack.
* Off-site images that carry only A1111 hash metadata (no `civitaiResources`)
  are resolved automatically through Civitai's by-hash lookup — this applies
  to showcase images on model pages too.
* **Showcase source order (model pages):** the version payload's own
  `images` array is the page carousel and usually carries per-image `meta`;
  when it doesn't (or the version resolved via the site endpoint, which has
  no images array), the resolver falls back to
  `/api/v1/images?modelVersionId=` and prefers the creator's own posts.
  Videos and meta-less entries are skipped.
* Requires a ComfyUI frontend with the sidebar-tab extension API (any recent
  Desktop build qualifies).

## Troubleshooting — no sidebar button?

Run the bundled diagnostic. In **ComfyUI Desktop** open the built-in terminal
(it is already inside the right Python environment) and run:

```
python custom_nodes/CiviImport/diagnose.py
```

It uses the standard library only, so it works even when the pack itself
fails to load, and it never modifies anything. It reports: every
`custom_nodes` folder on the machine and whether CiviImport is in each (and
whether the copy is complete), whether the modules import — with the real
traceback if not — and, for the ComfyUI server that is actually running,
which install it is, whether our backend answered `/civiimport/ping`, and
whether our frontend file appears in the extension list the browser loads.

Two checks you can do in any browser without a terminal, replacing `8000`
with your port:

* `http://127.0.0.1:8000/civiimport/ping` — JSON means the backend loaded;
  404 means this install doesn't have the pack.
* `http://127.0.0.1:8000/api/extensions` — the list of frontend files
  ComfyUI serves. If no `CiviImport` entry appears, no button can appear.

The usual cause is **the pack sitting in a different install than the one
you launch**. Desktop supports several installs, and its bundled
`resources/ComfyUI` folder is replaced on update — anything dropped there
is lost. Keep the pack in the install's own `custom_nodes` folder (Help
menu → open the install folder).

## Author & licence

Created by **Sev Kiriouchine**. Released under the MIT licence (see `LICENSE`).
Source: <https://github.com/Kiriouchine/CiviImport>

Civitai is a trademark of its respective owner; this project is an
independent, unofficial client for their public API.
