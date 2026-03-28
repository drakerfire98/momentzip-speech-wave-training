"""Build emotional contour artifacts from staged words and VoxForge sentences."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from momentzip_speech_wave_training.emotional_contour_training import build_emotional_contour_bank


def main() -> None:
    parser = argparse.ArgumentParser(description="Build emotional contour and word overlay artifacts.")
    parser.add_argument("--staged-root", type=Path, default=REPO_ROOT / "staged")
    parser.add_argument("--voxforge-root", type=Path, default=REPO_ROOT / "downloads" / "voxforge")
    parser.add_argument("--artifacts-root", type=Path, default=REPO_ROOT / "artifacts")
    args = parser.parse_args()

    bank_path, overlay_path = build_emotional_contour_bank(
        args.staged_root,
        sorted(args.voxforge_root.glob("*.tgz")),
        args.artifacts_root / "emotional_contour_bank.json",
        args.artifacts_root / "word_emotional_overlay.json",
    )
    print(bank_path)
    print(overlay_path)


if __name__ == "__main__":
    main()
