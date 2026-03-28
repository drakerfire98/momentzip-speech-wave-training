"""Build a word-shape index and label bank from staged WAV clips."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from momentzip_speech_wave_training.word_shape_training import (
    build_label_bank,
    build_three_layer_indexes,
    build_word_shape_index,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build waveform indexes from staged speech clips.")
    parser.add_argument("--staged-root", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    args = parser.parse_args()

    index_path = build_word_shape_index(args.staged_root, args.artifacts_root)
    label_bank_path = build_label_bank(index_path, args.artifacts_root / "label_bank.json")
    layer_paths = build_three_layer_indexes(index_path, label_bank_path, args.artifacts_root)
    print(index_path)
    print(label_bank_path)
    print(layer_paths["layer_v1_raw_index"])
    print(layer_paths["layer_v2_signal_index"])
    print(layer_paths["layer_v3_semantic_index"])


if __name__ == "__main__":
    main()
