"""Build unsupervised proto-phoneme and speaker-invariant artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from momentzip_speech_wave_training.proto_phoneme_training import (
    build_proto_phoneme_index,
    build_speaker_invariant_bank,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build proto-phoneme and speaker-invariant shape banks.")
    parser.add_argument("--staged-root", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    args = parser.parse_args()

    proto_path = build_proto_phoneme_index(args.staged_root, args.artifacts_root / "proto_phoneme_index.json")
    invariant_path = build_speaker_invariant_bank(args.staged_root, args.artifacts_root / "speaker_invariant_bank.json")
    print(proto_path)
    print(invariant_path)


if __name__ == "__main__":
    main()
