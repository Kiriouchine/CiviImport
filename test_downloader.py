#!/usr/bin/env python3
# CiviImport — self-tests for the downloader
#
# Copyright (c) 2026 Sev Kiriouchine
# SPDX-License-Identifier: MIT
# Part of CiviImport: https://github.com/Kiriouchine/CiviImport

"""Offline tests for downloader.py — serves a fake CDN on localhost, no Civitai.

    python test_downloader.py
"""

import http.server
import hashlib
import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import downloader  # noqa: E402
import hashing  # noqa: E402

PAYLOAD = os.urandom(3 * 1024 * 1024 + 12345)  # ~3 MB
TYPE_MAP = {"upscaler": "upscale_models", "checkpoint": "checkpoints", "lora": "loras"}


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/deny"):
            self.send_response(403)
            self.end_headers()
            return
        if self.path.startswith("/html"):
            body = b"<html>please log in</html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(PAYLOAD)))
        self.end_headers()
        self.wfile.write(PAYLOAD)

    def log_message(self, *args):  # keep the test output clean
        pass


def make_version(url, name="4x_test_model.safetensors", mtype="Upscaler"):
    return {"id": 1, "model": {"name": "Test", "type": mtype},
            "downloadUrl": url,
            "files": [{"name": name, "sizeKB": len(PAYLOAD) / 1024,
                       "primary": True, "downloadUrl": url}]}


def main() -> int:
    print("Running downloader self-test (localhost CDN, no Civitai)…")
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    events = []
    send = events.append

    with tempfile.TemporaryDirectory() as root:
        roots = {"upscale_models": os.path.join(root, "upscale_models")}
        os.makedirs(roots["upscale_models"])

        # --- happy path -----------------------------------------------------
        fetch = lambda vid, key, host: make_version(f"{base}/file.bin")  # noqa: E731
        name, folder = downloader.download_one(
            164821, api_key=None, host="civitai.com", fetch_version=fetch,
            send=send, type_to_folder=TYPE_MAP, folder_roots=roots)
        final = os.path.join(roots["upscale_models"], name)
        assert folder == "upscale_models" and os.path.isfile(final)
        with open(final, "rb") as fh:
            assert fh.read() == PAYLOAD
        assert not os.path.exists(final + ".part")
        statuses = [e["status"] for e in events]
        assert statuses[0] == "starting" and statuses[-1] == "done", statuses
        print(f"  happy path OK ({len(PAYLOAD) // 1024} KB, events: {statuses})")

        # --- already present -> immediate done with note ---------------------
        events.clear()
        downloader.download_one(164821, api_key=None, host="civitai.com",
                                fetch_version=fetch, send=send,
                                type_to_folder=TYPE_MAP, folder_roots=roots)
        assert events == [{"version_id": 164821, "status": "done",
                           "filename": name, "folder": "upscale_models",
                           "note": "already present"}], events
        print("  already-present skip OK")

        # --- 403 -> API-key message, no partial left -------------------------
        fetch403 = lambda vid, key, host: make_version(f"{base}/deny", name="denied.safetensors")  # noqa: E731
        try:
            downloader.download_one(2, api_key=None, host="civitai.com",
                                    fetch_version=fetch403, send=send,
                                    type_to_folder=TYPE_MAP, folder_roots=roots)
            raise AssertionError("expected DownloadError for 403")
        except downloader.DownloadError as e:
            assert "API key" in str(e), e
        assert not os.path.exists(os.path.join(roots["upscale_models"], "denied.safetensors.part"))
        print("  403 handling OK")

        # --- html consent wall ----------------------------------------------
        fetch_html = lambda vid, key, host: make_version(f"{base}/html", name="wall.safetensors")  # noqa: E731
        try:
            downloader.download_one(3, api_key=None, host="civitai.com",
                                    fetch_version=fetch_html, send=send,
                                    type_to_folder=TYPE_MAP, folder_roots=roots)
            raise AssertionError("expected DownloadError for html wall")
        except downloader.DownloadError as e:
            assert "web page" in str(e), e
        print("  html-wall handling OK")

        # --- post-download hash verification (step 6) -------------------------
        hashing.CACHE_PATH = os.path.join(root, "hash_cache.json")  # keep the repo clean
        good_sha = hashlib.sha256(PAYLOAD).hexdigest().upper()

        def make_hashed(name, hashes):
            v = make_version(f"{base}/file.bin", name=name)
            v["files"][0]["hashes"] = hashes
            return v

        events.clear()
        downloader.download_one(
            21, api_key=None, host="civitai.com",
            fetch_version=lambda v, k, h: make_hashed("verified.safetensors", {"SHA256": good_sha}),
            send=send, type_to_folder=TYPE_MAP, folder_roots=roots)
        done_ev = [e for e in events if e["status"] == "done"][-1]
        assert done_ev.get("note") == "sha-256 verified", done_ev
        assert any(e["status"] == "verifying" for e in events), events
        assert os.path.isfile(os.path.join(roots["upscale_models"], "verified.safetensors"))
        # the computed hash landed in the shared cache (hash matching sees it)
        cached = hashing.cached_sha256(
            os.path.join(roots["upscale_models"], "verified.safetensors"), compute=False)
        assert cached == good_sha, cached

        events.clear()  # AutoV2 prefix (first 10 hex of the SHA-256) also accepted
        downloader.download_one(
            22, api_key=None, host="civitai.com",
            fetch_version=lambda v, k, h: make_hashed("verified_auto.safetensors",
                                                      {"AutoV2": good_sha[:10]}),
            send=send, type_to_folder=TYPE_MAP, folder_roots=roots)
        assert [e for e in events if e["status"] == "done"][-1].get("note") == "sha-256 verified"

        events.clear()  # mismatch -> file removed, friendly error
        try:
            downloader.download_one(
                23, api_key=None, host="civitai.com",
                fetch_version=lambda v, k, h: make_hashed("corrupt.safetensors",
                                                          {"SHA256": "0" * 64}),
                send=send, type_to_folder=TYPE_MAP, folder_roots=roots)
            raise AssertionError("expected DownloadError for hash mismatch")
        except downloader.DownloadError as e:
            assert "hash verification" in str(e), e
        assert not os.path.exists(os.path.join(roots["upscale_models"], "corrupt.safetensors"))
        print("  hash verification OK (sha256, autov2 prefix, mismatch removal)")

        # --- name sanitization + unsupported type ----------------------------
        for bad in ("../../evil.bin", "sub/dir.bin", "", ".hidden"):
            try:
                got = downloader._safe_name(bad)
                assert "/" not in got and "\\" not in got and got not in ("", ".", "..") \
                    and not got.startswith("."), (bad, got)
            except downloader.DownloadError:
                pass  # rejection is also fine
        try:
            downloader.download_one(
                4, api_key=None, host="civitai.com",
                fetch_version=lambda v, k, h: make_version(f"{base}/f", mtype="MotionModule"),
                send=send, type_to_folder=TYPE_MAP, folder_roots=roots)
            raise AssertionError("expected unsupported-type error")
        except downloader.DownloadError as e:
            assert "Unsupported" in str(e), e
        print("  sanitization + type guard OK")

        # --- background worker: sequential + all_done ------------------------
        events.clear()
        fetch_two = lambda vid, key, host: make_version(  # noqa: E731
            f"{base}/file.bin", name=f"worker_{vid}.safetensors")
        accepted, skipped = downloader.start_downloads(
            [11, 12], api_key=None, host="civitai.com", fetch_version=fetch_two,
            send=send, type_to_folder=TYPE_MAP, folder_roots=roots)
        assert accepted == [11, 12] and skipped == []
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if any(e.get("status") == "all_done" for e in events):
                break
            time.sleep(0.05)
        else:
            raise AssertionError(f"worker did not finish in time: {events}")
        assert os.path.isfile(os.path.join(roots["upscale_models"], "worker_11.safetensors"))
        assert os.path.isfile(os.path.join(roots["upscale_models"], "worker_12.safetensors"))
        dones = [e for e in events if e["status"] == "done"]
        assert [d["version_id"] for d in dones] == [11, 12], dones
        print("  background worker OK (sequential, all_done emitted)")

    srv.shutdown()
    print("\nDOWNLOADER SELF-TEST PASSED ✔")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
