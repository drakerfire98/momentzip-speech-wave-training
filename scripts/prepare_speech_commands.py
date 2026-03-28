"""Stage isolated-word speech clips for MomentZip waveform training."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable


DEFAULT_LABELS = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "go",
    "stop",
]


def stage_speech_commands(
    *,
    source_root: Path,
    output_root: Path,
    labels: Iterable[str],
    limit_per_label: int = 250,
) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
      "source": "speech_commands",
      "source_root": str(source_root),
      "labels": [],
    }

    for label in labels:
        source_dir = source_root / label
        if not source_dir.exists():
            continue
        target_dir = output_root / label
        if target_dir.exists():
            # Re-stage each label cleanly so repeated runs do not silently mix
            # synthetic smoke data or older imports into the current corpus.
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for clip in sorted(source_dir.glob("*.wav")):
            if copied >= limit_per_label:
                break
            shutil.copy2(clip, target_dir / clip.name)
            copied += 1
        manifest["labels"].append({"label": label, "count": copied})

    (output_root / "staging_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage Speech Commands clips for waveform training.")
    parser.add_argument("--source-root", type=Path, required=True, help="Unpacked Speech Commands dataset root.")
    parser.add_argument("--output-root", type=Path, required=True, help="Local staged dataset directory.")
    parser.add_argument("--labels", nargs="*", default=DEFAULT_LABELS, help="Word labels to stage.")
    parser.add_argument("--limit-per-label", type=int, default=250, help="Max clips to copy for each label.")
    args = parser.parse_args()

    stage_speech_commands(
        source_root=args.source_root,
        output_root=args.output_root,
        labels=args.labels,
        limit_per_label=args.limit_per_label,
    )


if __name__ == "__main__":
    main()
