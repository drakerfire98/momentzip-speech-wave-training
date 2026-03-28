"""Bootstrap a local word-shape training workspace."""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
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
from prepare_speech_commands import DEFAULT_LABELS, stage_speech_commands


def _extract_archive(archive_path: Path, destination_root: Path) -> Path:
    extract_root = destination_root / archive_path.stem.replace(".tar", "")
    extract_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:*") as handle:
        handle.extractall(extract_root)
    nested_dirs = [path for path in extract_root.iterdir() if path.is_dir()]
    if len(nested_dirs) == 1:
        return nested_dirs[0]
    return extract_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap the speech-wave training repo.")
    parser.add_argument("--source-root", type=Path, help="Unpacked Speech Commands dataset root.")
    parser.add_argument("--archive-file", type=Path, help="Optional Speech Commands archive to extract first.")
    parser.add_argument("--labels", nargs="*", default=DEFAULT_LABELS, help="Word labels to stage.")
    parser.add_argument("--limit-per-label", type=int, default=250, help="Max clips to stage for each label.")
    args = parser.parse_args()

    downloads_root = REPO_ROOT / "downloads"
    staged_root = REPO_ROOT / "staged"
    artifacts_root = REPO_ROOT / "artifacts"
    downloads_root.mkdir(parents=True, exist_ok=True)
    staged_root.mkdir(parents=True, exist_ok=True)
    artifacts_root.mkdir(parents=True, exist_ok=True)

    source_root = args.source_root
    if args.archive_file:
        source_root = _extract_archive(args.archive_file, downloads_root)

    if source_root:
        stage_manifest = stage_speech_commands(
            source_root=source_root,
            output_root=staged_root,
            labels=args.labels,
            limit_per_label=args.limit_per_label,
        )
    else:
        stage_manifest = {
            "source": "none",
            "labels": [],
            "note": "No dataset source was provided. Existing staged clips, if any, were reused.",
        }

    index_path = build_word_shape_index(staged_root, artifacts_root)
    label_bank_path = build_label_bank(index_path, artifacts_root / "label_bank.json")
    layer_paths = build_three_layer_indexes(index_path, label_bank_path, artifacts_root)

    report_path = artifacts_root / "bootstrap_report.json"
    report_path.write_text(
        json.dumps(
            {
                "version": 1,
                "source_root": str(source_root) if source_root else "",
                "stage_manifest": stage_manifest,
                "word_shape_index_path": str(index_path),
                "label_bank_path": str(label_bank_path),
                "three_layer_paths": layer_paths,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(str(report_path))


if __name__ == "__main__":
    main()
