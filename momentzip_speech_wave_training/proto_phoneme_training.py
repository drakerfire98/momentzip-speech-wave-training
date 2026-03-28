"""Unsupervised proto-phoneme and speaker-invariant shape builders.

This is an honest first step toward phoneme learning. It does not claim true
IPA alignment yet. Instead, it:

- trims each clip down to its voiced frames
- splits the voiced span into change-based subword segments
- clusters those segments into reusable proto-phoneme shapes
- builds speaker-invariant per-word contours by normalizing away loudness and
  duration differences across speakers
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List, Sequence

from .audio_archive_runtime import SignalFrame, build_signal_tokens


def _trim_silence(frames: Sequence[SignalFrame]) -> List[SignalFrame]:
    start = 0
    end = len(frames)
    while start < end and frames[start].silence:
        start += 1
    while end > start and frames[end - 1].silence:
        end -= 1
    trimmed = list(frames[start:end])
    return trimmed if trimmed else list(frames)


def _change_scores(frames: Sequence[SignalFrame]) -> List[tuple[int, float]]:
    scores: List[tuple[int, float]] = []
    for index in range(1, len(frames)):
        left = frames[index - 1]
        right = frames[index]
        score = (
            abs(right.rms - left.rms) * 2.4
            + abs(right.peak - left.peak) * 1.6
            + abs(right.zero_crossing_rate - left.zero_crossing_rate) * 1.2
        )
        scores.append((index, score))
    return scores


def _segment_count(frame_count: int) -> int:
    if frame_count <= 6:
        return 1
    if frame_count <= 12:
        return 2
    if frame_count <= 22:
        return 3
    return 4


def _select_breakpoints(frames: Sequence[SignalFrame]) -> List[int]:
    desired_breaks = max(0, _segment_count(len(frames)) - 1)
    if desired_breaks <= 0:
        return []
    selected: List[int] = []
    for candidate, _score in sorted(_change_scores(frames), key=lambda item: item[1], reverse=True):
        if candidate < 2 or candidate > len(frames) - 2:
            continue
        if any(abs(candidate - existing) < 2 for existing in selected):
            continue
        selected.append(candidate)
        if len(selected) >= desired_breaks:
            break
    return sorted(selected)


def _segment_frames(frames: Sequence[SignalFrame]) -> List[List[SignalFrame]]:
    if not frames:
        return []
    breakpoints = _select_breakpoints(frames)
    segments: List[List[SignalFrame]] = []
    previous = 0
    for breakpoint in breakpoints:
        segments.append(list(frames[previous:breakpoint]))
        previous = breakpoint
    segments.append(list(frames[previous:]))
    return [segment for segment in segments if segment]


def _quantize(value: float, buckets: int) -> int:
    clamped = max(0.0, min(0.999999, value))
    return int(clamped * buckets)


def _shape_key(*, duration_ratio: float, rms_ratio: float, peak_ratio: float, zcr_ratio: float, edge_ratio: float) -> str:
    parts = [
        f"d{_quantize(duration_ratio, 10)}",
        f"r{_quantize(rms_ratio, 12)}",
        f"p{_quantize(peak_ratio, 12)}",
        f"z{_quantize(zcr_ratio, 12)}",
        f"e{_quantize(edge_ratio, 8)}",
    ]
    return "-".join(parts)


def _summarize_segment(segment: Sequence[SignalFrame], clip_frames: Sequence[SignalFrame]) -> dict:
    total_duration = max(1e-9, clip_frames[-1].end_ms - clip_frames[0].start_ms)
    segment_duration = segment[-1].end_ms - segment[0].start_ms
    clip_max_rms = max(frame.rms for frame in clip_frames) or 1e-9
    clip_max_peak = max(frame.peak for frame in clip_frames) or 1e-9
    clip_max_zcr = max(frame.zero_crossing_rate for frame in clip_frames) or 1e-9
    mean_rms = sum(frame.rms for frame in segment) / len(segment)
    mean_peak = sum(frame.peak for frame in segment) / len(segment)
    mean_zcr = sum(frame.zero_crossing_rate for frame in segment) / len(segment)
    rises = 0
    falls = 0
    for left, right in zip(segment, segment[1:]):
        if right.rms > left.rms:
            rises += 1
        elif right.rms < left.rms:
            falls += 1
    edge_ratio = rises / max(1, rises + falls)
    return {
        "duration_ms": round(segment_duration, 2),
        "duration_ratio": round(segment_duration / total_duration, 4),
        "mean_rms_ratio": round(mean_rms / clip_max_rms, 4),
        "mean_peak_ratio": round(mean_peak / clip_max_peak, 4),
        "mean_zcr_ratio": round(mean_zcr / clip_max_zcr, 4),
        "edge_ratio": round(edge_ratio, 4),
    }


def build_proto_phoneme_index(staged_root: Path, output_path: Path) -> Path:
    segment_entries: List[dict] = []
    cluster_counter: Counter[str] = Counter()
    cluster_labels: dict[str, Counter[str]] = defaultdict(Counter)

    for label_dir in sorted(path for path in staged_root.iterdir() if path.is_dir()):
        for wav_path in sorted(label_dir.glob("*.wav")):
            voiced_frames = _trim_silence(build_signal_tokens(wav_path))
            for segment_index, segment in enumerate(_segment_frames(voiced_frames)):
                summary = _summarize_segment(segment, voiced_frames)
                shape_key = _shape_key(
                    duration_ratio=float(summary["duration_ratio"]),
                    rms_ratio=float(summary["mean_rms_ratio"]),
                    peak_ratio=float(summary["mean_peak_ratio"]),
                    zcr_ratio=float(summary["mean_zcr_ratio"]),
                    edge_ratio=float(summary["edge_ratio"]),
                )
                segment_entries.append(
                    {
                        "label": label_dir.name,
                        "clip_path": str(wav_path),
                        "segment_index": segment_index,
                        "frame_span": [segment[0].index, segment[-1].index],
                        "shape_key": shape_key,
                        "summary": summary,
                    }
                )
                cluster_counter[shape_key] += 1
                cluster_labels[shape_key][label_dir.name] += 1

    clusters = []
    for shape_key, count in cluster_counter.most_common():
        clusters.append(
            {
                "shape_key": shape_key,
                "count": count,
                "labels": dict(cluster_labels[shape_key].most_common()),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "proto_phoneme_index",
                "segments": segment_entries,
                "clusters": clusters,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path


def _resample(values: Sequence[float], bins: int) -> List[float]:
    if not values:
        return [0.0] * bins
    if len(values) == 1:
        return [round(values[0], 5)] * bins
    result: List[float] = []
    for bin_index in range(bins):
        start = int((bin_index / bins) * len(values))
        end = int(((bin_index + 1) / bins) * len(values))
        if end <= start:
            end = min(len(values), start + 1)
        window = values[start:end]
        result.append(round(sum(window) / len(window), 5))
    return result


def _contour_signature(frames: Sequence[SignalFrame], bins: int = 24) -> dict:
    voiced_frames = _trim_silence(frames)
    rms_values = [frame.rms for frame in voiced_frames]
    peak_values = [frame.peak for frame in voiced_frames]
    zcr_values = [frame.zero_crossing_rate for frame in voiced_frames]
    max_rms = max(rms_values) if rms_values else 1.0
    max_peak = max(peak_values) if peak_values else 1.0
    max_zcr = max(zcr_values) if zcr_values else 1.0
    normalized_rms = [value / max_rms if max_rms else 0.0 for value in rms_values]
    normalized_peak = [value / max_peak if max_peak else 0.0 for value in peak_values]
    normalized_zcr = [value / max_zcr if max_zcr else 0.0 for value in zcr_values]
    return {
        "rms_bins": _resample(normalized_rms, bins),
        "peak_bins": _resample(normalized_peak, bins),
        "zcr_bins": _resample(normalized_zcr, bins),
    }


def _mean_vectors(signatures: Iterable[dict]) -> dict:
    signatures = list(signatures)
    if not signatures:
        return {"rms_bins": [], "peak_bins": [], "zcr_bins": []}
    bins = len(signatures[0]["rms_bins"])
    output = {"rms_bins": [], "peak_bins": [], "zcr_bins": []}
    for key in output:
        for index in range(bins):
            output[key].append(round(sum(signature[key][index] for signature in signatures) / len(signatures), 5))
    return output


def _similarity(left: dict, right: dict) -> float:
    deltas: List[float] = []
    for key in ("rms_bins", "peak_bins", "zcr_bins"):
        for left_value, right_value in zip(left[key], right[key]):
            deltas.append(abs(left_value - right_value))
    if not deltas:
        return 0.0
    return round(max(0.0, 1.0 - (sum(deltas) / len(deltas))), 5)


def build_clip_signature(wav_path: Path, bins: int = 24) -> dict:
    return _contour_signature(build_signal_tokens(wav_path), bins=bins)


def signature_similarity(left: dict, right: dict) -> float:
    return _similarity(left, right)


def extract_proto_sequence(wav_path: Path) -> List[str]:
    voiced_frames = _trim_silence(build_signal_tokens(wav_path))
    sequence: List[str] = []
    for segment in _segment_frames(voiced_frames):
        summary = _summarize_segment(segment, voiced_frames)
        sequence.append(
            _shape_key(
                duration_ratio=float(summary["duration_ratio"]),
                rms_ratio=float(summary["mean_rms_ratio"]),
                peak_ratio=float(summary["mean_peak_ratio"]),
                zcr_ratio=float(summary["mean_zcr_ratio"]),
                edge_ratio=float(summary["edge_ratio"]),
            )
        )
    return sequence


def build_speaker_invariant_bank(staged_root: Path, output_path: Path) -> Path:
    label_bank: dict[str, dict] = {}
    for label_dir in sorted(path for path in staged_root.iterdir() if path.is_dir()):
        clip_signatures = []
        for wav_path in sorted(label_dir.glob("*.wav")):
            signature = _contour_signature(build_signal_tokens(wav_path))
            clip_signatures.append({"clip_path": str(wav_path), "signature": signature})
        prototype = _mean_vectors([entry["signature"] for entry in clip_signatures])
        prototype_hash = hashlib.sha256(
            json.dumps(prototype, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        similarities = [_similarity(entry["signature"], prototype) for entry in clip_signatures]
        label_bank[label_dir.name] = {
            "clip_count": len(clip_signatures),
            "prototype_hash": prototype_hash,
            "average_similarity": round(sum(similarities) / len(similarities), 5) if similarities else 0.0,
            "prototype": prototype,
            "example_clips": [entry["clip_path"] for entry in clip_signatures[:5]],
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "speaker_invariant_bank",
                "labels": label_bank,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return output_path
