# CiviImport — package entry point (no graph nodes by design)
#
# Copyright (c) 2026 Sev Kiriouchine
# SPDX-License-Identifier: MIT
# Part of CiviImport: https://github.com/Kiriouchine/CiviImport

"""CiviImport — import a Civitai image's full generation setup into ComfyUI.

This pack registers no graph nodes on purpose: workflows it generates use
only vanilla core nodes, so they stay portable. What it ships instead:
  * backend routes on the ComfyUI server (routes.py)
  * a frontend panel + menu entry (web/js/civiimport.js)

Author: Sev Kiriouchine.  MIT licensed — see LICENSE.
"""

import os
import traceback

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./web/js"

_HERE = os.path.dirname(os.path.abspath(__file__))
_JS = os.path.join(_HERE, "web", "js", "civiimport.js")

# Printed before anything can fail, so the startup log always shows whether
# ComfyUI found this pack at all — the first thing to check when the sidebar
# button is missing.
print(f"[CiviImport] loading from {_HERE}")
if not os.path.isfile(_JS):
    print(f"[CiviImport] WARNING: frontend file missing at {_JS} — the sidebar "
          "tab cannot appear. The folder was probably copied incompletely.")

try:
    from . import routes  # noqa: F401  (import side effect: registers endpoints)
except Exception as e:  # keep ComfyUI booting even if we fail
    print(f"[CiviImport] FAILED to register routes: {e}")
    traceback.print_exc()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
