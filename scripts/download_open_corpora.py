"""Download open speech corpora into the repo workspace."""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from html import unescape
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
REGISTRY_PATH = REPO_ROOT / "datasets" / "registry.json"
USER_AGENT = "MomentZipSpeechWaveTraining/1.0"


def _load_registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _http_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request) as response:
        return response.read().decode("utf-8", errors="replace")


def _download(url: str, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return {"url": url, "path": str(destination), "status": "existing", "bytes": destination.stat().st_size}
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request) as response, destination.open("wb") as handle:
        total = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            total += len(chunk)
    return {"url": url, "path": str(destination), "status": "downloaded", "bytes": total}


def _download_speech_commands(entry: dict, downloads_root: Path) -> list[dict]:
    url = str(entry["download_url"])
    filename = Path(url).name
    return [_download(url, downloads_root / "speech_commands" / filename)]


def _download_voxforge(entry: dict, downloads_root: Path, limit: int) -> list[dict]:
    listing_url = str(entry["listing_url"])
    html = _http_text(listing_url)
    hrefs = re.findall(r'href="([^"]+?\.tgz)"', html, flags=re.IGNORECASE)
    unique_hrefs: list[str] = []
    seen: set[str] = set()
    for href in hrefs:
        clean = unescape(href)
        if clean in seen:
            continue
        seen.add(clean)
        unique_hrefs.append(clean)
    selected = unique_hrefs[:limit] if limit > 0 else unique_hrefs
    results: list[dict] = []
    for href in selected:
        url = href if href.startswith("http") else f"{listing_url}{href}"
        results.append(_download(url, downloads_root / "voxforge" / Path(href).name))
    return results


def _download_common_voice(downloads_root: Path, urls: list[str]) -> list[dict]:
    results: list[dict] = []
    for url in urls:
        results.append(_download(url, downloads_root / "common_voice" / Path(url).name))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Download open speech corpora into the repo workspace.")
    parser.add_argument("--source", action="append", choices=["speech_commands", "voxforge", "common_voice"], required=True)
    parser.add_argument("--voxforge-limit", type=int, default=50, help="Max VoxForge archives to download; use 0 for all.")
    parser.add_argument("--common-voice-url", action="append", default=[], help="Direct Common Voice locale archive URL.")
    args = parser.parse_args()

    downloads_root = REPO_ROOT / "downloads"
    downloads_root.mkdir(parents=True, exist_ok=True)
    registry = {entry["id"]: entry for entry in _load_registry().get("sources", [])}
    manifest = {"version": 1, "results": []}

    for source in args.source:
        if source == "speech_commands":
            manifest["results"].append({"source": source, "downloads": _download_speech_commands(registry[source], downloads_root)})
        elif source == "voxforge":
            manifest["results"].append({"source": source, "downloads": _download_voxforge(registry[source], downloads_root, args.voxforge_limit)})
        elif source == "common_voice":
            manifest["results"].append({"source": source, "downloads": _download_common_voice(downloads_root, args.common_voice_url)})

    manifest_path = downloads_root / "corpus_fetch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(str(manifest_path))


if __name__ == "__main__":
    main()
