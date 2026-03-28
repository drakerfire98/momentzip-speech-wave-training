"""Extract downloaded archives into repo-managed extraction folders."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent


def _extract_archive(archive_path: Path, extract_root: Path) -> Path:
    target_root = extract_root / archive_path.stem.replace(".tar", "")
    if target_root.exists():
        return target_root
    target_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:*") as handle:
        handle.extractall(target_root)
    return target_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract downloaded speech corpora archives.")
    parser.add_argument("--downloads-root", type=Path, default=REPO_ROOT / "downloads")
    parser.add_argument("--extract-root", type=Path, default=REPO_ROOT / "downloads" / "extracted")
    args = parser.parse_args()

    args.extract_root.mkdir(parents=True, exist_ok=True)
    extracted = []
    for archive_path in sorted(args.downloads_root.rglob("*.tar.gz")) + sorted(args.downloads_root.rglob("*.tgz")):
        extracted.append(str(_extract_archive(archive_path, args.extract_root)))
    print("\n".join(extracted))


if __name__ == "__main__":
    main()
