/**
 * CiviImport — sidebar panel: fetch, review, download, generate.
 *
 * Copyright (c) 2026 Sev Kiriouchine
 * SPDX-License-Identifier: MIT
 * Part of CiviImport: https://github.com/Kiriouchine/CiviImport
 */

// CiviImport — frontend panel.
// Registers: a "Civitai" topbar menu command and a left-sidebar tab.
// Talks to the backend routes in routes.py via api.fetchApi.
//
// Design note: the panel deliberately inherits ComfyUI's own theme variables
// so it reads as native UI. The only color it adds is the status lights.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TAB_ID = "civiimport";

// ---------------------------------------------------------------------------
// Styles (theme-following; injected once)
// ---------------------------------------------------------------------------
const CSS = `
.civi-panel { display:flex; flex-direction:column; gap:10px; padding:10px;
  color: var(--input-text, #ddd); font-size: 12px; height:100%; overflow-y:auto; }
.civi-title { font-size: 14px; font-weight: 600; margin: 0; }
.civi-sub { color: var(--descrip-text, #999); margin: 0; }
.civi-row { display:flex; gap:6px; align-items:center; }
.civi-input { flex:1; min-width:0; padding:6px 8px; border-radius:6px;
  border:1px solid var(--border-color,#444);
  background: var(--comfy-input-bg,#222); color: var(--input-text,#ddd); }
.civi-btn { padding:6px 10px; border-radius:6px; cursor:pointer;
  border:1px solid var(--border-color,#444);
  background: var(--comfy-input-bg,#222); color: var(--input-text,#ddd);
  white-space:nowrap; }
.civi-btn:hover:not(:disabled) { filter: brightness(1.25); }
.civi-btn:disabled { opacity:.45; cursor:not-allowed; }
.civi-btn.mini { padding:1px 7px; font-size:11px; border-radius:5px;
  margin-left:auto; align-self:center; }
.civi-card { border:1px solid var(--border-color,#444); border-radius:8px;
  background: var(--comfy-menu-bg, #202020); padding:8px; display:flex;
  flex-direction:column; gap:6px; }
.civi-kv { display:grid; grid-template-columns: 82px 1fr; gap:2px 8px; }
.civi-k { color: var(--descrip-text,#999); }
.civi-dot { display:inline-block; width:9px; height:9px; border-radius:50%;
  margin-right:6px; flex:none; }
.civi-dot.green { background:#3fb950; } .civi-dot.red { background:#f85149; }
.civi-dot.gray { background:#8b8b8b; }
.civi-res { display:flex; align-items:baseline; gap:6px; line-height:1.35; }
.civi-res .name { overflow-wrap:anywhere; }
.civi-badge { font-size:10px; padding:0 5px; border-radius:4px;
  border:1px solid var(--border-color,#444); color:var(--descrip-text,#999);
  text-transform:uppercase; flex:none; }
.civi-file { color: var(--descrip-text,#999); font-size:11px; overflow-wrap:anywhere; }
.civi-warn { border:1px solid #d29922; border-radius:6px; padding:6px 8px;
  color:#d29922; }
.civi-error { border:1px solid var(--error-text,#f85149); border-radius:6px;
  padding:6px 8px; color: var(--error-text,#f85149); overflow-wrap:anywhere; }
.civi-status { color: var(--descrip-text,#999); min-height:14px; }
.civi-panel details > summary { cursor:pointer; color: var(--descrip-text,#999); }
.civi-prompt { white-space:pre-wrap; overflow-wrap:anywhere; margin:4px 0 0 0;
  max-height:180px; overflow-y:auto; }
.civi-link { color:#6ea8fe; text-decoration:none; }
.civi-link:hover { text-decoration:underline; }
.civi-bar { height:6px; border-radius:4px; background: var(--comfy-input-bg,#222);
  border:1px solid var(--border-color,#444); overflow:hidden; }
.civi-bar-fill { height:100%; background:#6ea8fe; transition:width .3s; }
.civi-bar-fill.done { background:#3fb950; }
.civi-dl-row { display:flex; flex-direction:column; gap:3px; }
.civi-thumb { max-width:100%; max-height:200px; border-radius:8px;
  object-fit:contain; align-self:flex-start; }
`;

// ---------------------------------------------------------------------------
// Tiny DOM + API helpers
// ---------------------------------------------------------------------------
function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") node.className = v;
    else if (k === "dataset") Object.assign(node.dataset, v);
    else if (k.startsWith("on") && typeof v === "function")
      node.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null) node.setAttribute(k, v);
  }
  for (const c of children) {
    if (c === null || c === undefined) continue;
    node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return node;
}

async function apiJSON(route, options = {}) {
  const res = await api.fetchApi(route, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  let data;
  try { data = await res.json(); }
  catch { data = { ok: false, error: `Bad response (HTTP ${res.status})` }; }
  return data;
}

function toast(severity, summary, detail, life = 4000) {
  try {
    app.extensionManager?.toast?.add?.({ severity, summary, detail, life });
  } catch (e) { console.log(`[CiviImport] ${summary}: ${detail}`); }
}

function fmtSize(kb) {
  if (!kb && kb !== 0) return "";
  const mb = kb / 1024;
  return mb >= 1024 ? `${(mb / 1024).toFixed(2)} GB` : `${mb.toFixed(0)} MB`;
}

async function copyText(text, label) {
  try { await navigator.clipboard.writeText(text); toast("success", "Copied", label, 1500); }
  catch { toast("warn", "Copy failed", "Clipboard unavailable in this context."); }
}

// ---------------------------------------------------------------------------
// Panel state + rendering
// ---------------------------------------------------------------------------
const state = { busy: false, recipe: null, modelPage: null, downloads: {},
                buildOpts: { factor: "1.2", addUpscaler: true },
                key: { set: false, masked: null, source: null } };
let refs = {}; // live DOM references for the current mount

function setStatus(text) { if (refs.status) refs.status.textContent = text || ""; }
function setBusy(b) {
  state.busy = b;
  if (refs.fetchBtn) refs.fetchBtn.disabled = b;
  if (refs.url) refs.url.disabled = b;
}

function keyStatusText() {
  if (!state.key.set) return "API key: not set (needed for gated images & most downloads)";
  const src = state.key.source === "env" ? "env var" : "saved";
  return `API key: ${state.key.masked} (${src})`;
}

async function refreshKeyStatus() {
  const data = await apiJSON("/civiimport/apikey");
  if (data.ok) state.key = { set: data.set, masked: data.masked, source: data.source };
  if (refs.keyStatus) refs.keyStatus.textContent = keyStatusText();
}

async function setApiKeyFlow() {
  let key = null;
  try {
    key = await app.extensionManager.dialog.prompt({
      title: "Civitai API key",
      message: "Paste your Civitai API key (Account settings → API Keys). Leave empty to clear.",
    });
  } catch (e) {
    key = window.prompt?.("Civitai API key (empty clears):") ?? null; // non-Desktop fallback
  }
  if (key === null || key === undefined) return; // cancelled
  const data = await apiJSON("/civiimport/apikey", { method: "POST", body: JSON.stringify({ key }) });
  if (data.ok) {
    state.key = { set: data.set, masked: data.masked, source: data.source };
    if (refs.keyStatus) refs.keyStatus.textContent = keyStatusText();
    toast(data.set ? "success" : "info", "Civitai API key", data.set ? "Key saved." : "Key cleared.");
  } else {
    toast("error", "Civitai API key", data.error || "Could not save the key.");
  }
}

function defaultFactor(recipe) {
  const h = parseFloat(recipe?.upscale_factor_hint);
  return Number.isFinite(h) && h > 1 ? String(h) : "1.2";
}

async function resolveUrl() {
  const url = refs.url?.value?.trim();
  if (!url) { setStatus("Paste a Civitai image or model URL first."); return; }
  setBusy(true);
  setStatus("Fetching data from Civitai…");
  refs.result.replaceChildren();
  try {
    const data = await apiJSON("/civiimport/resolve", { method: "POST", body: JSON.stringify({ url }) });
    if (!data.ok) {
      setStatus("");
      refs.result.replaceChildren(el("div", { class: "civi-error" }, data.error || "Unknown error."));
      return;
    }
    setStatus("");
    state.buildOpts = { factor: defaultFactor(data.recipe), addUpscaler: true };
    if (data.kind === "model") {
      state.modelPage = data.model_page || null;
      state.recipe = data.recipe || null;
      if (state.recipe) renderRecipe(state.recipe);
      else renderModelCardOnly();
      return;
    }
    state.modelPage = null;
    state.recipe = data.recipe;
    renderRecipe(data.recipe);
  } finally {
    setBusy(false);
  }
}

function rerenderResults() {
  if (state.recipe) renderRecipe(state.recipe);
  else if (state.modelPage) renderModelCardOnly();
}

function switchVersion(versionId) {
  const mp = state.modelPage;
  if (!mp?.model?.model_id || state.busy) return;
  const url = `https://${mp.api_host || "civitai.com"}/models/${mp.model.model_id}?modelVersionId=${versionId}`;
  if (refs.url) refs.url.value = url;
  resolveUrl();
}

// A resource the server can actually fetch: real version, known folder, a
// file to get, not one of Civitai's internal system resources, not already
// on disk. Shared by the per-resource buttons and "Download all".
function isDownloadable(r) {
  return !!(r && r.version_id && r.folder && r.file && !r.system
            && !(r.local && r.local.present));
}

// Live state per version — the backend dedupes in-flight ids, and the panel
// mirrors that so one running download only disables its own button.
function inFlight(versionId) {
  const d = state.downloads[versionId];
  return !!d && d.status !== "done" && d.status !== "error";
}

function anyInFlight() {
  return Object.values(state.downloads).some(
    (d) => d.status !== "done" && d.status !== "error");
}

/** Start one version; `title` labels the toasts. Optimistically marks the
 *  version in flight so a double-click can't fire two requests. */
async function downloadVersion(versionId, host, title, filename) {
  if (!versionId || inFlight(versionId)) return;
  if (!state.key.set) {
    toast("warn", title,
      "No API key set — Civitai refuses many downloads without one.", 6000);
  }
  state.downloads[versionId] = { version_id: versionId, status: "queued", filename };
  rerenderResults();
  renderDownloads();
  const data = await apiJSON("/civiimport/download", {
    method: "POST",
    body: JSON.stringify({ version_ids: [versionId], api_host: host }),
  });
  if (!data.ok) {
    delete state.downloads[versionId];
    rerenderResults();
    renderDownloads();
    toast("error", title, data.error || "Could not start the download.");
  }
}

function downloadModel() {
  const mp = state.modelPage;
  const v = mp?.selected_version;
  if (!v) return;
  return downloadVersion(v.version_id, mp.api_host, "Download model", v.file?.name);
}

function downloadResource(r) {
  return downloadVersion(r.version_id, state.recipe?.source?.api_host,
                         r.model_name || "Download", r.file?.name);
}

async function downloadMissing() {
  if (!state.recipe) return;
  const missing = (state.recipe.resources || []).filter(
    (r) => isDownloadable(r) && !inFlight(r.version_id));
  if (!missing.length) {
    toast("info", "Download all", "Nothing left to download.");
    return;
  }
  if (!state.key.set) {
    toast("warn", "Download all",
      "No API key set — Civitai refuses many downloads without one.", 6000);
  }
  for (const r of missing) {  // queued until the worker reaches each one
    state.downloads[r.version_id] = {
      version_id: r.version_id, status: "queued", filename: r.file?.name };
  }
  renderRecipe(state.recipe); // re-render to disable the buttons
  renderDownloads();
  const data = await apiJSON("/civiimport/download", {
    method: "POST",
    body: JSON.stringify({
      version_ids: missing.map((r) => r.version_id),
      api_host: state.recipe.source?.api_host,
    }),
  });
  if (!data.ok) {
    for (const r of missing) delete state.downloads[r.version_id];
    renderRecipe(state.recipe);
    renderDownloads();
    toast("error", "Download all", data.error || "Could not start downloads.");
    return;
  }
  toast("info", "Download all",
    `Started ${data.accepted.length} download(s)` +
    (data.skipped.length ? `, ${data.skipped.length} already running.` : "."));
}

function onProgress(p) {
  if (p.status === "all_done") {
    refreshCombos();
    // Individual buttons each start their own worker, so this can arrive
    // while another one is still going — only wrap up when all are done.
    if (anyInFlight()) return;
    toast("success", "Downloads", "All downloads finished.");
    rerenderResults();
    return;
  }
  if (p.version_id === null || p.version_id === undefined) return;
  state.downloads[p.version_id] = { ...(state.downloads[p.version_id] || {}), ...p };
  if (p.status === "done") {
    const v = state.modelPage?.selected_version;
    if (v && v.version_id === p.version_id) {
      v.local = { present: true, filename: p.filename, method: "download" };
    }
    if (state.recipe) {
      for (const r of state.recipe.resources || []) {
        if (r.version_id === p.version_id) {
          r.local = { present: true, filename: p.filename, method: "download" };
        }
      }
      state.recipe.missing_count = (state.recipe.resources || [])
        .filter((r) => !r.system && !(r.local && r.local.present)).length;
    }
  }
  if (p.status === "done" || p.status === "error") rerenderResults();
  renderDownloads();
}

function renderDownloads() {
  if (!refs.downloads) return;
  const entries = Object.values(state.downloads);
  if (!entries.length) {
    refs.downloads.replaceChildren();
    return;
  }
  const rows = entries.map((d) => {
    if (d.status === "error") {
      return el("div", { class: "civi-dl-row" },
        el("div", { class: "civi-error" }, `${d.filename || d.version_id}: ${d.error}`));
    }
    const pct = d.pct ?? (d.status === "done" ? 100 : 0);
    const label = d.status === "done"
      ? `✓ ${d.filename}${d.note ? ` (${d.note})` : ""}`
      : d.status === "verifying"
        ? `Verifying ${d.filename} (sha-256)…`
        : d.status === "queued"
          ? `Queued: ${d.filename || d.version_id}`
          : `${d.filename || d.version_id} — ${d.downloaded_mb ?? 0}${d.total_mb ? ` / ${d.total_mb}` : ""} MB`;
    return el("div", { class: "civi-dl-row" },
      el("div", { class: "civi-file" }, label),
      el("div", { class: "civi-bar" },
        el("div", {
          class: "civi-bar-fill" + (d.status === "done" ? " done" : ""),
          style: `width:${Math.max(2, Math.min(100, pct))}%`,
        })),
    );
  });
  refs.downloads.replaceChildren(el("div", { class: "civi-card" },
    el("div", { class: "civi-sub" }, "Downloads"), ...rows));
}

async function refreshCombos() {
  try {
    if (typeof app.refreshComboInNodes === "function") {
      await app.refreshComboInNodes();
      toast("success", "Model lists refreshed", "Loader dropdowns now include the new files.");
      return;
    }
  } catch (e) {
    console.warn("[CiviImport] combo refresh failed:", e);
  }
  toast("info", "Refresh needed",
    "Press R (or the Refresh button) so loaders pick up the new files.");
}

async function hashScan() {
  if (!state.recipe) return;
  const folders = [...new Set((state.recipe.resources || [])
    .filter((r) => r.folder && !r.system && !(r.local && r.local.present))
    .map((r) => r.folder))];
  const data = await apiJSON("/civiimport/hashscan", {
    method: "POST", body: JSON.stringify({ folders }),
  });
  if (!data.ok) {
    toast("error", "Hash scan", data.error || "Could not start the scan.");
    return;
  }
  toast("info", "Hash scan",
    `Hashing local files in: ${data.folders.join(", ")} — renamed models will be recognized.`);
}

function onHashScan(p) {
  if (p.status === "hashing" && refs.status) {
    refs.status.textContent = `Hashing ${p.index}/${p.total} — ${p.file}`
      + (p.pct != null ? ` (${p.pct}%)` : "");
  } else if (p.status === "done") {
    if (refs.status) refs.status.textContent = "";
    toast("success", "Hash scan", `Hashed ${p.hashed} new file(s).`);
    // Re-resolve so hash matches flip the lights.
    if (refs.url?.value?.trim() && !state.busy) resolveUrl();
  } else if (p.status === "error") {
    console.warn("[CiviImport] hash scan:", p);
  }
}

async function confirmDialog(title, message) {
  try {
    const r = await app.extensionManager.dialog.confirm({ title, message });
    return r === true;
  } catch (e) {
    // Older frontends without dialog.confirm; window.confirm may not exist on Desktop.
    return window.confirm ? window.confirm(message) : true;
  }
}

async function generateWorkflow() {
  if (!state.recipe) return;
  const ok = await confirmDialog(
    "Generate workflow",
    "Load the generated workflow onto the canvas? This replaces the current graph.",
  );
  if (!ok) return;

  const f = parseFloat(state.buildOpts.factor);
  const data = await apiJSON("/civiimport/build", {
    method: "POST",
    body: JSON.stringify({
      recipe: state.recipe,
      options: {
        upscale_factor: Number.isFinite(f) ? f : undefined,
        add_upscaler: !!state.buildOpts.addUpscaler,
      },
    }),
  });
  if (!data.ok) {
    toast("error", "Generate workflow", data.error || "Build failed.");
    return;
  }

  try {
    await app.loadGraphData(data.workflow);
  } catch (e) {
    console.error("[CiviImport] loadGraphData failed:", e);
    toast("error", "Generate workflow", `Could not load the graph: ${e}`);
    return;
  }

  const info = data.info || {};
  const missing = (info.missing_files || []).length;
  toast(
    missing ? "warn" : "success",
    "Workflow loaded",
    missing
      ? `${missing} model file(s) missing — their loader widgets stay invalid until downloaded.`
      : `Ready to queue (${info.node_count} nodes, seed ${info.seed_control}).`,
    6000,
  );
  renderBuildInfo(info);
}

function renderBuildInfo(info) {
  if (!refs.result) return;
  const w = info.warnings || [];
  if (!w.length) return;
  refs.result.appendChild(el("div", { class: "civi-card" },
    el("div", { class: "civi-sub" }, "Build notes"),
    ...w.map((t) => el("div", { class: "civi-warn" }, t)),
  ));
}

function buildModelCard() {
  const mp = state.modelPage;
  if (!mp) return null;
  const m = mp.model || {};
  const v = mp.selected_version || {};
  const file = v.file || {};
  const present = v.local?.present;
  const busy = inFlight(v.version_id);

  const dot = el("span", {
    class: `civi-dot ${present ? "green" : "red"}`,
    title: present ? `Found locally: ${v.local.filename}` : "Not found in your model folders",
  });
  let versionCtl;
  if ((mp.versions || []).length > 1) {
    versionCtl = el("select", {
      class: "civi-input",
      title: "Switch version — re-fetches the card and showcase recipe.",
      onchange: (e) => switchVersion(e.target.value),
    }, ...mp.versions.map((x) => el("option", {
      value: String(x.id),
      selected: x.id === v.version_id ? "selected" : undefined,
    }, `${x.name || `version ${x.id}`}${x.base_model ? ` · ${x.base_model}` : ""}`)));
  } else {
    versionCtl = el("span", {}, `${v.name || `version ${v.version_id}`}${v.base_model ? ` · ${v.base_model}` : ""}`);
  }
  const triggers = (v.trained_words || []).length
    ? el("div", { class: "civi-file" }, `triggers: ${v.trained_words.join(", ")}`)
    : null;

  return el("div", { class: "civi-card" },
    el("div", { class: "civi-res" },
      el("span", { class: "civi-badge" }, (v.kind || m.type || "?").toUpperCase()),
      el("span", { class: "name", style: "font-weight:600;" }, m.name || `model ${m.model_id ?? "?"}`),
      m.url ? el("a", { class: "civi-link", style: "margin-left:auto;", href: m.url, target: "_blank", rel: "noopener" }, "open ↗") : null,
    ),
    el("div", { class: "civi-row" }, el("span", { class: "civi-k" }, "Version"), versionCtl),
    el("div", { class: "civi-res" },
      dot,
      el("span", { class: "civi-file" },
        file.name ? `${file.name}${file.size_kb ? ` · ${fmtSize(file.size_kb)}` : ""}` : "no file info"),
    ),
    triggers,
    el("div", { class: "civi-row" },
      el("button", {
        class: "civi-btn",
        disabled: (present || busy || !file.download_url) ? "disabled" : undefined,
        title: present
          ? "Already in your model folders."
          : "Downloads this model version into the right ComfyUI folder (verified by hash).",
        onclick: downloadModel,
      }, present ? "✓ Present" : (busy ? "Downloading…" : "Download model")),
      m.creator ? el("span", { class: "civi-sub", style: "margin-left:auto;" }, `by ${m.creator}`) : null,
    ),
  );
}

/** Civitai page for a resource, pinned to the exact version the image used
 *  (`?modelVersionId=`). The host follows the recipe's source, so a .red
 *  import links back to .red rather than a page that 404s on .com. */
function resourceUrl(r) {
  if (!r?.model_id) return null;
  const host = state.recipe?.source?.api_host || "civitai.com";
  return `https://${host}/models/${r.model_id}`
       + (r.version_id ? `?modelVersionId=${r.version_id}` : "");
}

/** Per-resource download control. Present and internal resources get none —
 *  the dot already says so; the rest get their own button so you can pull a
 *  single checkpoint/LoRA/embedding without fetching the whole recipe. */
function resourceButton(r) {
  if (r.system || r.local?.present) return null;
  if (inFlight(r.version_id)) {
    const st = state.downloads[r.version_id]?.status;
    return el("button", { class: "civi-btn mini", disabled: "disabled" },
      st === "verifying" ? "Verifying…" : st === "queued" ? "Queued…" : "Downloading…");
  }
  if (!isDownloadable(r)) {
    return el("button", {
      class: "civi-btn mini", disabled: "disabled",
      title: r.error ? `Lookup failed: ${r.error}`
                     : "No downloadable file for this resource.",
    }, "Download");
  }
  return el("button", {
    class: "civi-btn mini",
    title: `Download only this ${r.kind || "model"} → ${r.folder}/`,
    onclick: () => downloadResource(r),
  }, "Download");
}

function renderModelCardOnly() {
  if (!refs.result || !state.modelPage) return;
  const out = [buildModelCard()];
  for (const n of state.modelPage.notes || []) out.push(el("div", { class: "civi-warn" }, n));
  refs.result.replaceChildren(...out.filter(Boolean));
}

function renderRecipe(recipe) {
  const out = [];

  // Model card (model-page imports come with one) --------------------------
  const card = buildModelCard();
  if (card) {
    out.push(card);
    for (const n of state.modelPage?.notes || []) out.push(el("div", { class: "civi-warn" }, n));
  }

  // Source line -------------------------------------------------------------
  const src = recipe.source || {};
  if (state.modelPage) {
    out.push(el("div", { class: "civi-sub" },
      `Starter recipe from ${src.username ? `${src.username}'s` : "a"} showcase image`,
      src.api_host === "civitai.red" ? " · via civitai.red" : "",
      "  ",
      src.url ? el("a", { class: "civi-link", href: src.url, target: "_blank", rel: "noopener" }, "open ↗") : null,
    ));
    const shot = state.modelPage.showcase;
    if (shot?.image_url) {
      out.push(el("img", {
        class: "civi-thumb", src: shot.image_url, alt: "showcase image",
        onerror: (e) => e.target.remove(),
      }));
    }
  } else {
    out.push(el("div", { class: "civi-sub" },
      `Image #${src.image_id ?? "?"}`,
      src.username ? ` by ${src.username}` : "",
      src.api_host === "civitai.red" ? " · via civitai.red" : "",
      "  ",
      src.url ? el("a", { class: "civi-link", href: src.url, target: "_blank", rel: "noopener" }, "open ↗") : null,
    ));
  }

  // Warnings ------------------------------------------------------------
  for (const w of recipe.warnings || []) out.push(el("div", { class: "civi-warn" }, w));

  // Parameters ------------------------------------------------------------
  const s = recipe.sampler || {};
  const samplerText = s.civitai
    ? `${s.civitai} → ${s.comfy_sampler}/${s.comfy_scheduler}${s.mapped ? "" : " (fallback)"}`
    : `${s.comfy_sampler}/${s.comfy_scheduler}`;
  const kv = (k, v) => [el("span", { class: "civi-k" }, k), el("span", {}, String(v))];
  const upscaler = (recipe.resources || []).find((r) => r.kind === "upscaler");
  out.push(el("div", { class: "civi-card" },
    el("div", { class: "civi-kv" },
      ...kv("Sampler", samplerText),
      ...kv("Steps / CFG", `${recipe.steps} / ${recipe.cfg}`),
      ...kv("Seed", recipe.seed >= 0 ? recipe.seed : "—"),
      ...kv("Clip skip", recipe.clip_skip ?? "—"),
      ...kv("Size", recipe.width && recipe.height
        ? `${recipe.width}×${recipe.height}`
          + (recipe.uploaded_width && (recipe.uploaded_width !== recipe.width || recipe.uploaded_height !== recipe.height)
            ? ` (uploaded ${recipe.uploaded_width}×${recipe.uploaded_height})` : "")
        : "—"),
      ...kv("Base model", src.base_model ?? "—"),
      ...(upscaler
        ? kv("Upscale", `${recipe.upscale_factor_hint ? "×" + recipe.upscale_factor_hint : "×1.5 (default)"} · ${upscaler.model_name ?? "?"}`)
        : []),
    ),
  ));

  // Resources ------------------------------------------------------------
  const resources = recipe.resources || [];
  const rows = resources.map((r) => {
    const present = r.local?.present;
    const sys = !!r.system;
    const dot = el("span", {
      class: `civi-dot ${present ? "green" : (sys || r.error ? "gray" : "red")}`,
      title: present
        ? `Found locally: ${r.local.filename}`
        : sys
          ? "Civitai internal resource (auto-injected by their generator) — not downloadable, skipped."
          : (r.error ? `Lookup failed: ${r.error}` : "Not found in your model folders"),
    });
    const weight = r.weight !== null && r.weight !== undefined && r.kind !== "checkpoint"
      ? ` @ ${r.weight}` : "";
    const fileText = r.file?.name
      ? `${r.file.name}${r.file.size_kb ? ` · ${fmtSize(r.file.size_kb)}` : ""}`
      : null;
    const pageUrl = resourceUrl(r);
    const fileLine = (fileText || pageUrl)
      ? el("div", { class: "civi-file" },
          fileText,
          fileText && pageUrl ? " · " : null,
          pageUrl ? el("a", {
            class: "civi-link", href: pageUrl, target: "_blank", rel: "noopener",
            title: "Open this model's page on Civitai",
          }, "model page ↗") : null)
      : null;
    return el("div", {},
      el("div", { class: "civi-res" },
        dot,
        el("span", { class: "civi-badge" }, (r.kind || "?").toUpperCase()),
        el("span", { class: "name" }, `${r.model_name ?? `version ${r.version_id}`}${r.version_name ? ` (${r.version_name})` : ""}${weight}${sys ? " · internal" : ""}`),
        resourceButton(r),
      ),
      fileLine,
    );
  });
  const pending = resources.filter((r) => isDownloadable(r) && !inFlight(r.version_id)).length;
  out.push(el("div", { class: "civi-card" },
    el("div", { class: "civi-row" },
      el("div", { class: "civi-sub" }, `Resources (${resources.length})`),
      el("button", {
        class: "civi-btn mini",
        disabled: pending === 0 ? "disabled" : undefined,
        title: "Downloads every missing model listed below, one after another.",
        onclick: downloadMissing,
      }, pending === 0 && anyInFlight() ? "Downloading…" : `Download all (${pending})`),
    ),
    ...(rows.length ? rows : [el("div", { class: "civi-sub" }, "None linked in metadata.")]),
  ));

  // Workflow options (step 7) ----------------------------------------------
  const hasUpscaler = (recipe.resources || []).some((r) => r.kind === "upscaler");
  const factorDisabled = !hasUpscaler && !state.buildOpts.addUpscaler;
  const optChildren = [
    el("span", { class: "civi-k", title: "Hires resize = original size × this factor (snapped to the /8 latent grid)." }, "Hires ×"),
    el("input", {
      class: "civi-input", type: "number", step: "0.05", min: "1", max: "8",
      style: "width:76px;flex:0 0 auto;",
      value: state.buildOpts.factor,
      disabled: factorDisabled ? "disabled" : undefined,
      title: "Hires resize = original size × this factor (snapped to the /8 latent grid).",
      oninput: (e) => { state.buildOpts.factor = e.target.value; },
    }),
  ];
  if (factorDisabled) {
    optChildren.push(el("span", { class: "civi-sub" }, "hires chain off"));
  }
  if (!hasUpscaler) {
    optChildren.push(el("label", {
      class: "civi-sub",
      style: "display:flex;align-items:center;gap:5px;margin-left:auto;cursor:pointer;",
      title: "This image wasn't upscaled on-site — add the Remacri hires-refine chain anyway (uses your local upscale model).",
    },
      el("input", {
        type: "checkbox",
        ...(state.buildOpts.addUpscaler ? { checked: "checked" } : {}),
        onchange: (e) => { state.buildOpts.addUpscaler = e.target.checked; renderRecipe(state.recipe); },
      }),
      "add hires upscale"));
  }
  out.push(el("div", { class: "civi-row" }, ...optChildren));

  // Missing banner + actions ----------------------------------------------
  const missing = recipe.missing_count ?? 0;
  if (missing > 0) {
    const note = recipe.local_check === "unavailable"
      ? " (local check unavailable — running outside ComfyUI?)" : "";
    out.push(el("div", { class: "civi-warn" },
      `⚠ ${missing} of ${resources.length} model file(s) missing locally${note}.`));
  }
  out.push(el("div", { class: "civi-row" },
    el("button", {
      class: "civi-btn",
      disabled: missing === 0 ? "disabled" : undefined,
      title: "Hashes local model files (cached) so renamed downloads are recognized as present.",
      onclick: hashScan,
    }, "Scan by hash"),
    el("button", {
      class: "civi-btn",
      title: "Builds the workflow and loads it onto the canvas.",
      onclick: generateWorkflow,
    }, "Generate workflow"),
  ));

  // Prompts ------------------------------------------------------------
  const promptBlock = (label, text) => el("details", {},
    el("summary", {}, `${label} (${text ? text.length : 0} chars)`),
    el("pre", { class: "civi-prompt" }, text || "—"),
    text ? el("button", { class: "civi-btn", onclick: () => copyText(text, label) }, `Copy ${label.toLowerCase()}`) : null,
  );
  out.push(el("div", { class: "civi-card" },
    promptBlock("Prompt", recipe.prompt),
    promptBlock("Negative prompt", recipe.negative_prompt),
  ));

  refs.result.replaceChildren(...out);
}

function renderPanel(root) {
  root.replaceChildren();
  refs = {};

  refs.keyStatus = el("div", { class: "civi-sub" }, keyStatusText());
  refs.url = el("input", {
    class: "civi-input", type: "text",
    placeholder: "https://civitai.com/images/… or /models/… (red & green ok)",
    onkeydown: (e) => { if (e.key === "Enter" && !state.busy) resolveUrl(); },
  });
  refs.fetchBtn = el("button", { class: "civi-btn", onclick: resolveUrl }, "Fetch");
  refs.status = el("div", { class: "civi-status" });
  refs.result = el("div", { style: "display:flex;flex-direction:column;gap:10px;" });
  refs.downloads = el("div", {});

  root.appendChild(el("div", { class: "civi-panel" },
    el("p", { class: "civi-title" }, "Civitai Import"),
    el("p", { class: "civi-sub" },
      "Paste an image URL to pull its full generation setup — or a model page for the model plus a starter recipe from its showcase."),
    el("div", { class: "civi-row" },
      refs.keyStatus,
      el("button", { class: "civi-btn", style: "margin-left:auto;", onclick: setApiKeyFlow }, "Set API key"),
    ),
    el("div", { class: "civi-row" }, refs.url, refs.fetchBtn),
    refs.status,
    refs.result,
    refs.downloads,
  ));

  refreshKeyStatus();
  rerenderResults(); // keep results across tab remounts
  renderDownloads();
}

// ---------------------------------------------------------------------------
// Opening the panel from the menu (sidebar APIs vary a bit across versions)
// ---------------------------------------------------------------------------
function openPanel() {
  const em = app.extensionManager;
  try {
    if (em?.sidebarTab?.toggleSidebarTab) return em.sidebarTab.toggleSidebarTab(TAB_ID);
    if (typeof em?.toggleSidebarTab === "function") return em.toggleSidebarTab(TAB_ID);
  } catch (e) { console.warn("[CiviImport] toggleSidebarTab failed:", e); }
  toast("info", "Civitai Import", "Open the Civitai tab (cloud icon) in the left sidebar.");
}

// ---------------------------------------------------------------------------
// Extension registration
// ---------------------------------------------------------------------------
/** Register the sidebar tab, retrying while the frontend finishes booting.
 *  Some frontend builds create app.extensionManager after extensions' setup()
 *  runs; a single attempt there fails silently and the tab never appears. */
function registerSidebar(attempt = 0) {
  const em = app.extensionManager;
  const reg = em?.registerSidebarTab;
  if (typeof reg === "function") {
    try {
      reg.call(em, {
        id: TAB_ID,
        icon: "pi pi-cloud-download",
        title: "Civitai Import",
        tooltip: "Import from a Civitai image or model URL",
        type: "custom",
        render: (elmt) => renderPanel(elmt),
      });
      console.log(`[CiviImport] sidebar tab registered (attempt ${attempt + 1})`);
    } catch (e) {
      console.error("[CiviImport] registerSidebarTab threw:", e);
    }
    return;
  }
  if (attempt < 20) {           // ~10 s of retries, then give up loudly
    setTimeout(() => registerSidebar(attempt + 1), 500);
    return;
  }
  console.error("[CiviImport] app.extensionManager.registerSidebarTab never became " +
                "available — this frontend may not support sidebar tabs. Use the " +
                "Civitai entry in the top menu instead.");
}

app.registerExtension({
  name: "civiimport",
  commands: [
    {
      id: "civiimport.open",
      label: "Import from Civitai URL…",
      icon: "pi pi-cloud-download",
      function: openPanel,
    },
  ],
  menuCommands: [
    { path: ["Civitai"], commands: ["civiimport.open"] },
  ],
  setup() {
    console.log(`[CiviImport] frontend loaded (extension ${TAB_ID})`);

    try {
      const style = document.createElement("style");
      style.textContent = CSS;
      document.head.appendChild(style);
    } catch (e) {
      console.warn("[CiviImport] could not inject styles:", e);
    }

    registerSidebar();

    try {
      api.addEventListener("civiimport.progress", (ev) => onProgress(ev.detail || {}));
      api.addEventListener("civiimport.hashscan", (ev) => onHashScan(ev.detail || {}));
    } catch (e) {
      console.warn("[CiviImport] could not attach progress listener:", e);
    }

    // Sanity ping so problems show up in the browser console early.
    apiJSON("/civiimport/ping").then((d) => {
      if (!d.ok) console.warn("[CiviImport] backend ping failed:", d);
      else console.log(`[CiviImport] backend v${d.version} reachable (local check: ${d.local_check})`);
    });
  },
});
