#!/usr/bin/env python3
# CiviImport — self-tests for the hash cache
#
# Copyright (c) 2026 Sev Kiriouchine
# SPDX-License-Identifier: MIT
# Part of CiviImport: https://github.com/Kiriouchine/CiviImport

"""Offline tests for hashing.py, hash matching, and deep scan — no network.

    python test_hashing.py
"""

import hashlib
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import civitai_api  # noqa: E402
import hashing  # noqa: E402
import local_models  # noqa: E402


def main() -> int:
    print("Running hashing self-test (no network)…")

    with tempfile.TemporaryDirectory() as tmp:
        hashing.CACHE_PATH = os.path.join(tmp, "hash_cache.json")

        # --- sha256_of matches hashlib; progress fires -----------------------
        payload = os.urandom(2 * 1024 * 1024 + 777)
        f1 = os.path.join(tmp, "model_a.safetensors")
        with open(f1, "wb") as fh:
            fh.write(payload)
        ticks = []
        digest = hashing.sha256_of(f1, progress=lambda d, t: ticks.append((d, t)))
        assert digest == hashlib.sha256(payload).hexdigest().upper()
        assert ticks and ticks[-1][0] == ticks[-1][1] == len(payload)
        print("  sha256_of OK")

        # --- cache behavior: cache-only miss, compute, hit, invalidation -----
        assert hashing.cached_sha256(f1, compute=False) is None
        assert hashing.cached_sha256(f1, compute=True) == digest
        assert hashing.cached_sha256(f1, compute=False) == digest  # cache hit
        with open(f1, "ab") as fh:  # size change invalidates
            fh.write(b"x")
        assert hashing.cached_sha256(f1, compute=False) is None
        print("  cache store/hit/invalidation OK")

        # --- hash matching recognizes a RENAMED lora -------------------------
        root = os.path.join(tmp, "models")
        lora_dir = os.path.join(root, "loras")
        os.makedirs(lora_dir)
        lora_bytes = os.urandom(512 * 1024)
        renamed = os.path.join(lora_dir, "my_renamed_lora.safetensors")
        with open(renamed, "wb") as fh:
            fh.write(lora_bytes)
        lora_sha = hashing.cached_sha256(renamed, compute=True)

        recipe = {"resources": [{
            "version_id": 1, "kind": "lora", "folder": "loras", "weight": 0.8,
            "model_name": "Renamed", "version_name": "v1", "trained_words": [],
            "file": {"name": "original_upstream_name.safetensors",
                     "hashes": {"AutoV2": lora_sha[:10]}, "download_url": None},
        }]}
        os.environ["CIVIIMPORT_MODELS_ROOT"] = root
        try:
            local_models.annotate_local_status(recipe)
        finally:
            os.environ.pop("CIVIIMPORT_MODELS_ROOT", None)
        loc = recipe["resources"][0]["local"]
        assert loc["present"] is True and loc["method"] == "hash", loc
        assert loc["filename"] == "my_renamed_lora.safetensors"
        assert recipe["missing_count"] == 0
        print("  renamed-file hash match OK (AutoV2 prefix)")

        # --- deep scan: hashes only uncached files, emits progress + done ----
        extra = os.path.join(lora_dir, "uncached.safetensors")
        with open(extra, "wb") as fh:
            fh.write(os.urandom(256 * 1024))
        events = []
        os.environ["CIVIIMPORT_MODELS_ROOT"] = root
        try:
            stats = local_models.deep_hash_scan(["loras"], progress=events.append)
            assert stats == {"hashed": 1, "candidates": 1}, stats  # renamed one is cached
            assert events[-1]["status"] == "done" and events[-1]["hashed"] == 1
            stats2 = local_models.deep_hash_scan(["loras"])
            assert stats2 == {"hashed": 0, "candidates": 0}, stats2
        finally:
            os.environ.pop("CIVIIMPORT_MODELS_ROOT", None)
        print("  deep scan OK (only uncached, idempotent)")

    # --- A1111 hash-metadata parsing ----------------------------------------
    meta = {
        "resources": [{"name": "coolLora", "type": "lora", "weight": 0.7, "hash": "AAAA1111BB"}],
        "hashes": {"model": "CCDDEEFF00", "lora:coolLora": "aaaa1111bb", "embed:zPDXL3": "1234567890"},
    }
    entries = civitai_api._hash_entries_from_meta(meta)
    assert [e["hash"] for e in entries] == ["AAAA1111BB", "CCDDEEFF00", "1234567890"], entries
    assert entries[0]["weight"] == 0.7 and entries[0]["type_hint"] == "lora"
    assert entries[1]["type_hint"] == "model" and entries[2]["type_hint"] == "embed"
    assert civitai_api._HASH_TYPE_MAP["model"] == "checkpoint"
    assert civitai_api._HASH_TYPE_MAP["embed"] == "textualinversion"
    print("  A1111 hash-meta parsing OK (dedupe, hints, weights)")

    print("\nHASHING SELF-TEST PASSED ✔")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
