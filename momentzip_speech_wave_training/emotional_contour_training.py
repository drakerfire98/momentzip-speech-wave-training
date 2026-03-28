"""Emotional contour extraction and overlay training."""

from __future__ import annotations

import json
import tarfile
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev
from typing import Iterable, List, Sequence

from .audio_archive_runtime import build_signal_tokens
from .transition_learning import _build_frame_view, _trim_silence


EMOTION_ANCHORS = {
    "calm": {"onset_sharpness": 0.15, "amplitude_variance": 0.18, "pace": 2.2, "decay_shape": 0.2},
    "urgent": {"onset_sharpness": 0.85, "amplitude_variance": 0.72, "pace": 4.8, "decay_shape": 0.85},
    "hesitant": {"onset_sharpness": 0.18, "amplitude_variance": 0.24, "pace": 1.8, "decay_shape": 0.6},
    "confident": {"onset_sharpness": 0.55, "amplitude_variance": 0.32, "pace": 3.4, "decay_shape": 0.4},
    "stressed": {"onset_sharpness": 0.7, "amplitude_variance": 0.82, "pace": 5.4, "decay_shape": 0.7},
}


def _normalize(value: float, *, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


def _burst_count(rms_values: Sequence[float], threshold: float) -> int:
    count = 0
    above = False
    for value in rms_values:
        if not above and value >= threshold:
            count += 1
            above = True
        elif above and value < threshold * 0.6:
            above = False
    return count


def _extract_signature_from_frames(frames: Sequence[dict]) -> dict:
    voiced = _trim_silence(frames)
    if not voiced:
        return {
            "onset_sharpness": 0.0,
            "amplitude_variance": 0.0,
            "pace": 0.0,
            "decay_shape": 0.0,
            "duration_seconds": 0.0,
        }

    rms_values = [float(frame["rms"]) for frame in voiced]
    duration_seconds = max(0.001, (voiced[-1]["end_ms"] - voiced[0]["start_ms"]) / 1000.0)
    peak_rms = max(rms_values) or 1e-9
    onset_window = rms_values[: min(5, len(rms_values))]
    onset_sharpness = 0.0
    if len(onset_window) >= 2:
        onset_sharpness = max(
            0.0,
            max((right - left) for left, right in zip(onset_window, onset_window[1:])) / peak_rms,
        )
    mean_rms = fmean(rms_values)
    amplitude_variance = pstdev(rms_values) / max(1e-9, mean_rms)
    burst_threshold = max(0.05, mean_rms * 1.05)
    pace = _burst_count(rms_values, burst_threshold) / duration_seconds
    tail = rms_values[max(0, len(rms_values) - max(3, len(rms_values) // 5)) :]
    tail_mean = fmean(tail)
    decay_shape = 1.0 - min(1.0, tail_mean / peak_rms)
    return {
        "onset_sharpness": round(_normalize(onset_sharpness, low=0.0, high=1.0), 5),
        "amplitude_variance": round(_normalize(amplitude_variance, low=0.0, high=1.2), 5),
        "pace": round(pace, 5),
        "pace_normalized": round(_normalize(pace, low=0.5, high=7.0), 5),
        "decay_shape": round(_normalize(decay_shape, low=0.0, high=1.0), 5),
        "duration_seconds": round(duration_seconds, 5),
    }


def classify_emotion(signature: dict) -> str:
    pace_value = float(signature.get("pace_normalized", signature.get("pace", 0.0)))
    best_name = "calm"
    best_distance = float("inf")
    for name, anchor in EMOTION_ANCHORS.items():
        distance = (
            abs(float(signature["onset_sharpness"]) - anchor["onset_sharpness"])
            + abs(float(signature["amplitude_variance"]) - anchor["amplitude_variance"])
            + abs(pace_value - _normalize(anchor["pace"], low=0.5, high=7.0))
            + abs(float(signature["decay_shape"]) - anchor["decay_shape"])
        )
        if distance < best_distance:
            best_distance = distance
            best_name = name
    return best_name


def classify_emotion_from_wav_path(wav_path: Path) -> tuple[str, dict]:
    frames = [frame.__dict__ for frame in build_signal_tokens(wav_path)]
    signature = _extract_signature_from_frames(frames)
    return classify_emotion(signature), signature


def _aggregate_ranges(records: Sequence[dict]) -> dict:
    features = ["onset_sharpness", "amplitude_variance", "pace", "decay_shape"]
    ranges = {}
    for feature in features:
        values = [float(record["signature"][feature]) for record in records]
        ranges[feature] = {
            "min": round(min(values), 5) if values else 0.0,
            "max": round(max(values), 5) if values else 0.0,
            "mean": round(fmean(values), 5) if values else 0.0,
        }
    return ranges


def build_emotional_contour_bank(
    staged_root: Path,
    voxforge_archives: Iterable[Path],
    bank_output_path: Path,
    overlay_output_path: Path,
) -> tuple[Path, Path]:
    emotion_records: dict[str, List[dict]] = defaultdict(list)
    word_overlay: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for label_dir in sorted(path for path in staged_root.iterdir() if path.is_dir()):
        for wav_path in sorted(label_dir.glob("*.wav")):
            emotion_name, signature = classify_emotion_from_wav_path(wav_path)
            record = {
                "source": "staged",
                "label": label_dir.name,
                "clip_path": str(wav_path),
                "signature": signature,
            }
            emotion_records[emotion_name].append(record)
            word_overlay[label_dir.name][emotion_name] += 1

    for archive_path in voxforge_archives:
        try:
            with tarfile.open(archive_path, "r:gz") as handle:
                for member in handle.getmembers():
                    if not member.isfile() or not member.name.lower().endswith(".wav"):
                        continue
                    wav_file = handle.extractfile(member)
                    if wav_file is None:
                        continue
                    signature = _extract_signature_from_frames(_build_frame_view(wav_file.read()))
                    emotion_name = classify_emotion(signature)
                    emotion_records[emotion_name].append(
                        {
                            "source": "voxforge",
                            "archive_path": str(archive_path),
                            "clip_path": member.name,
                            "signature": signature,
                        }
                    )
        except tarfile.TarError:
            continue

    bank_payload = {
        "version": 1,
        "kind": "emotional_contour_bank",
        "emotions": {
            emotion_name: {
                "clip_count": len(records),
                "ranges": _aggregate_ranges(records),
                "examples": records[:5],
            }
            for emotion_name, records in sorted(emotion_records.items())
        },
    }
    overlay_payload = {
        "version": 1,
        "kind": "word_emotional_overlay",
        "words": {
            word: {
                "variants": dict(sorted(variants.items(), key=lambda item: item[1], reverse=True)),
                "dominant_emotion": max(variants.items(), key=lambda item: item[1])[0] if variants else "calm",
            }
            for word, variants in sorted(word_overlay.items())
        },
    }

    bank_output_path.parent.mkdir(parents=True, exist_ok=True)
    bank_output_path.write_text(json.dumps(bank_payload, indent=2, sort_keys=True), encoding="utf-8")
    overlay_output_path.write_text(json.dumps(overlay_payload, indent=2, sort_keys=True), encoding="utf-8")
    return bank_output_path, overlay_output_path
