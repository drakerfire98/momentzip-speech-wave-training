"""Sentence transition learning from VoxForge-style archives.

This is an honest first pass at continuous speech learning. VoxForge packs
sentence-level prompts and WAV files together, but they do not provide precise
word timestamps here. So this builder:

- reads prompts and WAV clips directly from the downloaded tar archives
- trims silence and normalizes each sentence contour
- approximates word spans across the voiced duration
- extracts boundary windows for adjacent word pairs
- aggregates reusable transition signatures for blends like "go left"

That gives the AH real transition memory without pretending it already has
forced alignment.
"""

from __future__ import annotations

import audioop
import io
import json
import math
import re
import tarfile
import wave
from array import array
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


WORD_PATTERN = re.compile(r"[A-Z']+")


def _read_wav_samples_from_bytes(raw_bytes: bytes) -> tuple[array, int]:
    with wave.open(io.BytesIO(raw_bytes), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frame_count = handle.getnframes()
        raw = handle.readframes(frame_count)
    if channels > 1:
        raw = audioop.tomono(raw, sample_width, 0.5, 0.5)
    if sample_width != 2:
        raw = audioop.lin2lin(raw, sample_width, 2)
    samples = array("h")
    samples.frombytes(raw)
    return samples, sample_rate


def _zero_crossing_rate(samples: Sequence[int]) -> float:
    if len(samples) < 2:
        return 0.0
    crossings = 0
    previous = samples[0]
    for sample in samples[1:]:
        if (previous < 0 <= sample) or (previous >= 0 > sample):
            crossings += 1
        previous = sample
    return crossings / len(samples)


def _build_frame_view(raw_bytes: bytes, frame_ms: int = 20) -> List[dict]:
    samples, sample_rate = _read_wav_samples_from_bytes(raw_bytes)
    frame_size = max(32, int(sample_rate * (frame_ms / 1000.0)))
    rms_values: List[float] = []
    provisional: List[tuple[int, int, float, float, float]] = []
    for start in range(0, len(samples), frame_size):
        frame = samples[start : start + frame_size]
        if not frame:
            continue
        rms = math.sqrt(sum(sample * sample for sample in frame) / len(frame)) / 32768.0
        peak = max(abs(sample) for sample in frame) / 32768.0
        zcr = _zero_crossing_rate(frame)
        rms_values.append(rms)
        provisional.append((start, len(frame), rms, peak, zcr))
    silence_threshold = max(0.01, (sum(rms_values) / len(rms_values)) * 0.22 if rms_values else 0.01)
    frames: List[dict] = []
    for index, (start, length, rms, peak, zcr) in enumerate(provisional):
        frames.append(
            {
                "index": index,
                "start_ms": round((start / sample_rate) * 1000.0, 2),
                "end_ms": round(((start + length) / sample_rate) * 1000.0, 2),
                "rms": round(rms, 5),
                "peak": round(peak, 5),
                "zcr": round(zcr, 5),
                "silence": rms < silence_threshold,
            }
        )
    return frames


def _trim_silence(frames: Sequence[dict]) -> List[dict]:
    start = 0
    end = len(frames)
    while start < end and frames[start]["silence"]:
        start += 1
    while end > start and frames[end - 1]["silence"]:
        end -= 1
    trimmed = list(frames[start:end])
    return trimmed if trimmed else list(frames)


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


def _normalize_window(frames: Sequence[dict], bins: int = 16) -> dict:
    if not frames:
        return {"rms_bins": [0.0] * bins, "peak_bins": [0.0] * bins, "zcr_bins": [0.0] * bins}
    rms_values = [float(frame["rms"]) for frame in frames]
    peak_values = [float(frame["peak"]) for frame in frames]
    zcr_values = [float(frame["zcr"]) for frame in frames]
    max_rms = max(rms_values) or 1.0
    max_peak = max(peak_values) or 1.0
    max_zcr = max(zcr_values) or 1.0
    return {
        "rms_bins": _resample([value / max_rms for value in rms_values], bins),
        "peak_bins": _resample([value / max_peak for value in peak_values], bins),
        "zcr_bins": _resample([value / max_zcr for value in zcr_values], bins),
    }


def _mean_signature(signatures: Sequence[dict]) -> dict:
    if not signatures:
        return {"rms_bins": [], "peak_bins": [], "zcr_bins": []}
    bin_count = len(signatures[0]["rms_bins"])
    result = {"rms_bins": [], "peak_bins": [], "zcr_bins": []}
    for key in result:
        for index in range(bin_count):
            result[key].append(round(sum(signature[key][index] for signature in signatures) / len(signatures), 5))
    return result


def _similarity(left: dict, right: dict) -> float:
    deltas: List[float] = []
    for key in ("rms_bins", "peak_bins", "zcr_bins"):
        for left_value, right_value in zip(left[key], right[key]):
            deltas.append(abs(left_value - right_value))
    if not deltas:
        return 0.0
    return round(max(0.0, 1.0 - (sum(deltas) / len(deltas))), 5)


def _parse_prompts(text: str) -> Dict[str, str]:
    prompts: Dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        key = Path(parts[0]).name.replace(".mfc", "")
        prompts[key] = parts[1].strip()
    return prompts


def _tokenize_prompt(prompt: str) -> List[str]:
    return WORD_PATTERN.findall(prompt.upper())


def _window_for_boundary(frames: Sequence[dict], word_count: int, boundary_index: int) -> List[dict]:
    if not frames or word_count < 2:
        return []
    left_edge = int((boundary_index / word_count) * len(frames))
    right_edge = int(((boundary_index + 1) / word_count) * len(frames))
    left_start = max(0, left_edge - max(2, len(frames) // max(10, word_count * 3)))
    right_end = min(len(frames), right_edge + max(2, len(frames) // max(10, word_count * 3)))
    return list(frames[left_start:right_end])


def build_transition_bank(archive_paths: Iterable[Path], output_path: Path, sentence_output_path: Path) -> tuple[Path, Path]:
    pair_signatures: dict[str, List[dict]] = defaultdict(list)
    pair_examples: dict[str, List[dict]] = defaultdict(list)
    sentence_records: List[dict] = []

    for archive_path in archive_paths:
        try:
            with tarfile.open(archive_path, "r:gz") as handle:
                prompt_member = next((member for member in handle.getmembers() if member.name.endswith("/etc/PROMPTS")), None)
                if prompt_member is None:
                    continue
                prompt_file = handle.extractfile(prompt_member)
                if prompt_file is None:
                    continue
                prompts = _parse_prompts(prompt_file.read().decode("utf-8", errors="replace"))
                wav_members = {
                    Path(member.name).stem: member
                    for member in handle.getmembers()
                    if member.isfile() and member.name.lower().endswith(".wav")
                }
                for clip_id, prompt in prompts.items():
                    member = wav_members.get(clip_id)
                    if member is None:
                        continue
                    wav_file = handle.extractfile(member)
                    if wav_file is None:
                        continue
                    words = _tokenize_prompt(prompt)
                    if len(words) < 2:
                        continue
                    frames = _trim_silence(_build_frame_view(wav_file.read()))
                    sentence_signature = _normalize_window(frames, bins=24)
                    sentence_records.append(
                        {
                            "archive_path": str(archive_path),
                            "clip_id": clip_id,
                            "prompt": prompt,
                            "word_count": len(words),
                            "signature": sentence_signature,
                        }
                    )
                    for boundary_index in range(len(words) - 1):
                        pair = f"{words[boundary_index]} {words[boundary_index + 1]}"
                        boundary_window = _window_for_boundary(frames, len(words), boundary_index + 1)
                        signature = _normalize_window(boundary_window, bins=16)
                        pair_signatures[pair].append(signature)
                        if len(pair_examples[pair]) < 5:
                            pair_examples[pair].append(
                                {
                                    "archive_path": str(archive_path),
                                    "clip_id": clip_id,
                                    "prompt": prompt,
                                    "boundary_index": boundary_index,
                                }
                            )
        except tarfile.TarError:
            continue

    transitions = []
    for pair, signatures in sorted(pair_signatures.items(), key=lambda item: len(item[1]), reverse=True):
        prototype = _mean_signature(signatures)
        similarities = [_similarity(signature, prototype) for signature in signatures]
        transitions.append(
            {
                "pair": pair,
                "count": len(signatures),
                "average_similarity": round(sum(similarities) / len(similarities), 5) if similarities else 0.0,
                "prototype": prototype,
                "examples": pair_examples[pair],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "transition_bank",
                "transitions": transitions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    sentence_output_path.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "sentence_blend_signatures",
                "sentences": sentence_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path, sentence_output_path
