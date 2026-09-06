#!/usr/bin/env python3
# CiviImport — standalone installation diagnostics
#
# Copyright (c) 2026 Sev Kiriouchine
# SPDX-License-Identifier: MIT
# Part of CiviImport: https://github.com/Kiriouchine/CiviImport

"""CiviImport diagnostics — answers "why is the sidebar button missing?".

Run it in ComfyUI Desktop's built-in terminal (Help/Terminal panel — it is
already inside the right Python environment):

    python diagnose.py

Or from anywhere with any Python 3.8+:

    python path\\to\\CiviImport\\diagnose.py

Standard library only, on purpose: it has to run even when the pack itself
is failing to load. It never writes anything — it only looks and reports.

What it checks, in order:
  1. which Python is running it
  2. every ComfyUI install / custom_nodes folder it can find on this machine
  3. whether CiviImport is in each one, and whether the copy is complete
  4. whether the pack's modules actually import (with the real traceback)
  5. the running ComfyUI server: which install it is, whether our backend
     answered, and whether our frontend file is in the list the browser loads

Paste the whole output when asking for help.
"""

import glob
import json
import os
import platform
import sys
import traceback
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PORTS = [8000, 8188, 8189, 8288, 7860, 8080, 8001]
REQUIRED = [
    "__init__.py", "routes.py", "civitai_api.py", "local_models.py",
    "graph_builder.py", "downloader.py", "hashing.py", "config.py",
    os.path.join("web", "js", "civiimport.js"),
]
PACK = "CiviImport"


def find_pack_dir(custom_nodes):
    """Locate the pack inside a custom_nodes folder, case-insensitively —
    the directory may be CiviImport, civiimport or any casing in between,
    and Linux/macOS care about the difference even though Windows doesn't."""
    try:
        for entry in os.listdir(custom_nodes):
            if entry.lower() == PACK.lower():
                full = os.path.join(custom_nodes, entry)
                if os.path.isdir(full):
                    return full
    except OSError:
        pass
    return None


def head(title):
    print()
    print("=" * 68)
    print(title)
    print("=" * 68)


def get_json(url, timeout=4):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


# ---------------------------------------------------------------- 1. python
def section_python():
    head("1. Python running this script")
    print("version   :", sys.version.split()[0], f"({platform.platform()})")
    print("executable:", sys.executable)
    print("script at :", HERE)


# --------------------------------------------------------------- 2. installs
def candidate_roots():
    """Every place a ComfyUI install or custom_nodes folder might live."""
    home = os.path.expanduser("~")
    env = os.environ
    roots = [
        env.get("LOCALAPPDATA", ""), env.get("APPDATA", ""),
        os.path.join(env.get("LOCALAPPDATA", ""), "Comfy-Desktop"),
        os.path.join(home, "ComfyUI-Installs"),
        os.path.join(home, "Documents"),
        os.path.join(home, "Library", "Application Support"),
        home,
    ]
    pats = []
    for r in [x for x in roots if x and os.path.isdir(x)]:
        pats += [
            os.path.join(r, "custom_nodes"),
            os.path.join(r, "ComfyUI", "custom_nodes"),
            os.path.join(r, "*", "custom_nodes"),
            os.path.join(r, "*", "ComfyUI", "custom_nodes"),
            os.path.join(r, "*", "*", "custom_nodes"),
            os.path.join(r, "*", "*", "ComfyUI", "custom_nodes"),
        ]
    # the folder this script lives in is probably already a custom_nodes child
    pats.append(os.path.join(os.path.dirname(HERE)))
    found, seen = [], set()
    for p in pats:
        for hit in glob.glob(p):
            real = os.path.normcase(os.path.abspath(hit))
            if real in seen or not os.path.isdir(hit):
                continue
            if os.path.basename(real.rstrip("\\/")) != "custom_nodes":
                continue
            seen.add(real)
            found.append(os.path.abspath(hit))
    return sorted(found)


def check_copy(pack_dir):
    missing = [f for f in REQUIRED if not os.path.isfile(os.path.join(pack_dir, f))]
    return missing


def section_installs():
    head("2. custom_nodes folders found on this machine")
    dirs = candidate_roots()
    if not dirs:
        print("!! none found automatically — check the Desktop Help menu for the")
        print("   install folder, and look for a 'custom_nodes' directory inside.")
        return []
    results, seen_real = [], {}
    for d in dirs:
        # Windows keeps legacy junctions ("My Documents" -> "Documents"); the
        # same folder must not be reported twice as if it were two installs.
        real = os.path.normcase(os.path.realpath(d))
        if real in seen_real:
            print(f"\n  {d}\n      -> same folder as {seen_real[real]} (junction/symlink), skipped")
            continue
        seen_real[real] = d
        pack_dir = find_pack_dir(d)
        present = pack_dir is not None
        pack_dir = pack_dir or os.path.join(d, PACK)
        print(f"\n  {d}\n      -> {'CiviImport PRESENT' if present else 'no CiviImport'}")
        try:
            packs = sorted(x for x in os.listdir(d)
                           if os.path.isdir(os.path.join(d, x)) and not x.startswith((".", "__")))
        except OSError as e:
            packs = []
            print("      (could not list:", e, ")")
        print(f"      packs here ({len(packs)}): "
              + (", ".join(packs[:8]) + (" …" if len(packs) > 8 else "") if packs else "(none)"))
        version = None
        if present:
            missing = check_copy(pack_dir)
            if missing:
                print("      !! INCOMPLETE COPY — missing:", ", ".join(missing))
            else:
                version = "?"
                try:
                    with open(os.path.join(pack_dir, "routes.py"), encoding="utf-8") as f:
                        for line in f:
                            if line.startswith("VERSION"):
                                version = line.split("=", 1)[1].strip().strip('"\'')
                                break
                except OSError:
                    pass
                print(f"      all required files present, version {version}")
        results.append({"dir": d, "packs": packs, "has_pack": present,
                        "version": version, "pack_dir": pack_dir})
    have = [r for r in results if r["has_pack"]]
    if not have:
        print("\n  >>> CiviImport is not in ANY custom_nodes folder found above.")
        print("      That alone explains a missing sidebar button.")
    elif len(have) > 1:
        print("\n  >>> CiviImport exists in MORE THAN ONE install. Only the install")
        print("      you actually launch will show the button (see section 4).")
    return results


# ----------------------------------------------------------------- 3. import
def section_import(installs):
    head("3. Do the pack's modules import?")
    hits = [r["pack_dir"] for r in installs if r["has_pack"]]
    target = hits[0] if hits else HERE
    print("testing:", target)
    sys.path.insert(0, target)
    for mod in ("civitai_api", "graph_builder", "local_models", "downloader", "hashing"):
        try:
            __import__(mod)
            print(f"  ok      {mod}")
        except Exception as e:
            print(f"  FAILED  {mod}: {type(e).__name__}: {e}")
            traceback.print_exc()
    try:
        import routes  # noqa: F401
        print("  ok      routes")
    except Exception as e:
        # aiohttp/server only exist inside ComfyUI, so this is expected outside it
        print(f"  n/a     routes: {type(e).__name__}: {e}")
        print("          (normal unless you ran this in ComfyUI's own environment)")
    try:
        import requests  # noqa: F401
    except ImportError:
        print("\n  note: 'requests' is missing from THIS Python. If you ran the script")
        print("  with a system Python rather than ComfyUI's own environment, that is")
        print("  expected and harmless — ComfyUI's environment is what matters.")


def served_pack_names(base):
    """Pack folder names ComfyUI is serving frontend files for, e.g.
    '/extensions/civicomfy/x.js' -> 'civicomfy'. Used to work out which
    install is running, since two installs can differ only by custom_nodes."""
    for path in ("/api/extensions", "/extensions"):
        try:
            data = get_json(base + path, timeout=3)
        except Exception:
            continue
        if isinstance(data, list):
            names = set()
            for entry in data:
                parts = str(entry).replace("\\", "/").split("/")
                if "extensions" in parts:
                    i = parts.index("extensions")
                    if i + 1 < len(parts):
                        names.add(parts[i + 1].lower())
            return names, len(data), path
    return None, 0, None


def section_server(installs):
    head("4. The running ComfyUI server")
    print("(start ComfyUI first, then re-run this, or these will all fail)")
    found_any = False
    for port in PORTS:
        base = f"http://127.0.0.1:{port}"
        try:
            stats = get_json(base + "/system_stats", timeout=2)
        except Exception:
            continue
        found_any = True
        sysinfo = stats.get("system", {}) if isinstance(stats, dict) else {}
        print(f"\n  server on {base}")
        for k, v in sysinfo.items():
            if k in ("argv", "required_frontend_version", "embedded_python", "devices"):
                continue
            print(f"    {k:16}: {v}")

        loaded = False
        try:
            ping = get_json(base + "/civiimport/ping", timeout=3)
            loaded = True
            print("    CiviImport ping : OK, version", ping.get("version"),
                  "| local check:", ping.get("local_check"))
        except urllib.error.HTTPError as e:
            print(f"    CiviImport ping : HTTP {e.code} — backend NOT loaded in this install")
        except Exception as e:
            print(f"    CiviImport ping : failed ({e}) — backend NOT loaded in this install")

        names, total, path = served_pack_names(base)
        if names is None:
            print("    extension list  : could not read (endpoint name may have changed)")
            continue
        print(f"    {path:16}: {total} file(s) served;",
              "CiviImport IS listed" if PACK.lower() in names else "CiviImport NOT listed")
        if loaded and PACK.lower() in names:
            print("\n    >>> This install has the pack. If the button is still missing,")
            print("        the problem is frontend-side, not installation.")
            continue

        # Work out which candidate folder is this server's custom_nodes by
        # matching the packs it serves against what each folder contains.
        best, best_score = None, 0
        for r in installs:
            score = len(names & {p.lower() for p in r["packs"]})
            if score > best_score:
                best, best_score = r, score
        donors = [r for r in installs if r["has_pack"] and r is not best]
        print("\n    >>> The pack is NOT loaded in the install that is running.")
        if best:
            print(f"        Running install's custom_nodes looks like:\n          {best['dir']}")
        if donors and best:
            src = donors[0]["pack_dir"]
            dst = os.path.join(best["dir"], PACK)
            print(f"        Your copy (v{donors[0]['version'] or '?'}) is at:\n          {src}")
            print("\n        Fix — copy it across, then restart ComfyUI:")
            if os.name == "nt":
                print(f'          xcopy /E /I /Y "{src}" "{dst}"')
            else:
                print(f'          cp -r "{src}" "{dst}"')
            print("        (afterwards, rename the old copy so you don't edit a dead one)")
        elif not donors:
            print("        No complete copy was found elsewhere either — install the pack")
            print("        into the folder above, or use ComfyUI Manager.")
    if not found_any:
        print("\n  no ComfyUI server answered on ports:", ", ".join(map(str, PORTS)))
        print("  If yours uses another port, open the app and check its address bar,")
        print("  then try: http://127.0.0.1:<port>/civiimport/ping in a browser.")


def main():
    print(__doc__.strip().splitlines()[0])
    section_python()
    installs = section_installs()
    section_import(installs)
    section_server(installs)
    head("Done — copy everything above when reporting the problem")


if __name__ == "__main__":
    main()
