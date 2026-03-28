"""Build a continuous-speech transition bank from downloaded VoxForge archives."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from momentzip_speech_wave_training.transition_learning import build_transition_bank


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a word-transition bank from VoxForge archives.")
    parser.add_argument("--voxforge-root", type=Path, default=REPO_ROOT / "downloads" / "voxforge")
    parser.add_argument("--artifacts-root", type=Path, default=REPO_ROOT / "artifacts")
    args = parser.parse_args()

    archives = sorted(args.voxforge_root.glob("*.tgz"))
    transition_path, sentence_path = build_transition_bank(
        archives,
        args.artifacts_root / "voxforge_transition_bank.json",
        args.artifacts_root / "voxforge_sentence_blends.json",
    )
    print(transition_path)
    print(sentence_path)


if __name__ == "__main__":
    main()
