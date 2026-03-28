"""Build the waveform-to-concept bridge and optionally recognize a WAV clip."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from momentzip_speech_wave_training.concept_bridge import build_concept_bridge, recognize


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or query the waveform-to-concept bridge.")
    parser.add_argument("--artifacts-root", type=Path, default=REPO_ROOT / "artifacts")
    parser.add_argument("--recognize", type=Path, help="Optional WAV clip to recognize after building the bridge.")
    args = parser.parse_args()

    bridge_path = build_concept_bridge(args.artifacts_root, args.artifacts_root / "concept_bridge.json")
    print(bridge_path)
    if args.recognize:
        result = recognize(args.recognize, args.artifacts_root)
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
