"""Local word-shape indexing and aggregation helpers."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from .audio_archive_runtime import archive_wav_with_tokens


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_word_shape_index(staged_root: Path, artifacts_root: Path) -> Path:
    artifacts_root.mkdir(parents=True, exist_ok=True)
    entries: List[dict] = []
    for label_dir in sorted(path for path in staged_root.iterdir() if path.is_dir()):
        label_artifacts_root = artifacts_root / label_dir.name
        for wav_path in sorted(label_dir.glob("*.wav")):
            build = archive_wav_with_tokens(
                wav_path,
                runtime_root=label_artifacts_root,
                stem_prefix=wav_path.stem,
            )
            entries.append(
                {
                    "label": label_dir.name,
                    "clip_path": str(wav_path),
                    "wavtxt_path": build.wavtxt_path,
                    "restored_path": build.restored_path,
                    "signal_tokens_path": build.signal_tokens_path,
                    "waveform_meaning_path": build.waveform_meaning_path,
                    "wav_sha256": build.wav_sha256,
                    "duration_seconds": build.duration_seconds,
                    "silence_ratio": build.silence_ratio,
                    "waveform_summary": build.waveform_summary,
                }
            )
    index_path = artifacts_root / "word_shape_index.json"
    index_path.write_text(
        json.dumps({"version": 1, "entries": entries}, indent=2),
        encoding="utf-8",
    )
    return index_path


def build_label_bank(index_path: Path, output_path: Path) -> Path:
    index_payload = _read_json(index_path)
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for entry in index_payload.get("entries", []):
        grouped[str(entry["label"])].append(entry)

    label_bank: Dict[str, dict] = {}
    for label, entries in grouped.items():
        waveform_counter: Counter[str] = Counter()
        total_silence_ratio = 0.0
        total_duration = 0.0
        unique_hashes = set()
        for entry in entries:
            total_silence_ratio += float(entry.get("silence_ratio", 0.0))
            total_duration += float(entry.get("duration_seconds", 0.0))
            unique_hashes.add(str(entry.get("wav_sha256", "")))
            waveform_payload = _read_json(Path(entry["waveform_meaning_path"]))
            for frame_entry in waveform_payload.get("frames", []):
                for label_name in frame_entry.get("labels", []):
                    waveform_counter[str(label_name)] += 1

        clip_count = len(entries)
        label_bank[label] = {
            "clip_count": clip_count,
            "unique_wave_hashes": len(unique_hashes),
            "average_silence_ratio": round(total_silence_ratio / clip_count, 4) if clip_count else 0.0,
            "average_duration_seconds": round(total_duration / clip_count, 4) if clip_count else 0.0,
            "dominant_wave_labels": [name for name, _count in waveform_counter.most_common(5)],
            "example_summaries": [entry["waveform_summary"] for entry in entries[:3]],
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"version": 1, "labels": label_bank}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path
