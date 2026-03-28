"""Standalone raw-audio archiving helpers for the training repo."""

from __future__ import annotations

import audioop
import hashlib
import json
import lzma
import math
import statistics
import wave
from array import array
from base64 import b85decode, b85encode
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence


@dataclass
class SignalFrame:
    index: int
    start_ms: float
    end_ms: float
    rms: float
    peak: float
    mean_abs: float
    zero_crossing_rate: float
    silence: bool


@dataclass
class WavArchiveBuild:
    clip_path: str
    wavtxt_path: str
    restored_path: str
    signal_tokens_path: str
    waveform_meaning_path: str
    wav_sha256: str
    restored_sha256: str
    frame_count: int
    duration_seconds: float
    archive_chars: int
    silence_ratio: float
    waveform_summary: str


WAVEFORM_GLOSSARY = {
    "silence": "very low energy, usually a pause or room tone",
    "onset": "a rising edge where a voiced event starts",
    "sustain": "stable voiced energy holding in place",
    "decay": "energy dropping off after a voiced event",
    "transient": "a sharp short event like a click, consonant edge, or attack",
    "noisy": "busy high-crossing texture, often hiss, friction, or roughness",
    "clipped_risk": "peaks are close to the ceiling and may distort",
}


def _read_wave_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _read_wave_samples(path: Path) -> tuple[array, int]:
    with wave.open(str(path), "rb") as handle:
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


def _window_rms(samples: Sequence[int], frame_size: int) -> List[float]:
    values: List[float] = []
    for start in range(0, len(samples), frame_size):
        frame = samples[start : start + frame_size]
        if not frame:
            continue
        values.append(math.sqrt(sum(sample * sample for sample in frame) / len(frame)) / 32768.0)
    return values


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


def encode_wav_to_text(path: Path) -> str:
    raw = _read_wave_bytes(path)
    compressed = lzma.compress(raw, preset=6)
    payload = {
        "version": 1,
        "codec": "wav+b85+lzma",
        "byte_length": len(raw),
        "wav_sha256": hashlib.sha256(raw).hexdigest(),
        "payload_b85": b85encode(compressed).decode("ascii"),
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def decode_text_to_wav(text: str, output_path: Path) -> Path:
    payload = json.loads(text)
    compressed = b85decode(payload["payload_b85"].encode("ascii"))
    restored = lzma.decompress(compressed)
    expected_hash = str(payload["wav_sha256"])
    restored_hash = hashlib.sha256(restored).hexdigest()
    if restored_hash != expected_hash:
        raise ValueError("restored wav hash did not match the archived hash")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(restored)
    return output_path


def build_signal_tokens(path: Path, *, frame_ms: int = 20) -> List[SignalFrame]:
    samples, sample_rate = _read_wave_samples(path)
    frame_size = max(32, int(sample_rate * (frame_ms / 1000.0)))
    rms_values = _window_rms(samples, frame_size)
    silence_threshold = max(0.01, statistics.median(rms_values) * 0.35 if rms_values else 0.01)
    frames: List[SignalFrame] = []
    for index, start in enumerate(range(0, len(samples), frame_size)):
        frame = samples[start : start + frame_size]
        if not frame:
            continue
        rms = math.sqrt(sum(sample * sample for sample in frame) / len(frame)) / 32768.0
        peak = max(abs(sample) for sample in frame) / 32768.0
        mean_abs = sum(abs(sample) for sample in frame) / len(frame) / 32768.0
        frames.append(
            SignalFrame(
                index=index,
                start_ms=round((start / sample_rate) * 1000.0, 2),
                end_ms=round(((start + len(frame)) / sample_rate) * 1000.0, 2),
                rms=round(rms, 5),
                peak=round(peak, 5),
                mean_abs=round(mean_abs, 5),
                zero_crossing_rate=round(_zero_crossing_rate(frame), 5),
                silence=rms < silence_threshold,
            )
        )
    return frames


def _frame_labels(frames: Sequence[SignalFrame]) -> List[dict]:
    labeled: List[dict] = []
    for index, frame in enumerate(frames):
        previous = frames[index - 1] if index > 0 else None
        delta = frame.rms - previous.rms if previous is not None else frame.rms
        labels: List[str] = []
        if frame.silence:
            labels.append("silence")
        else:
            if previous is None or previous.silence or delta > 0.03:
                labels.append("onset")
            if abs(delta) <= 0.02:
                labels.append("sustain")
            if previous is not None and delta < -0.03:
                labels.append("decay")
            if frame.peak > max(0.12, frame.rms * 1.8):
                labels.append("transient")
            if frame.zero_crossing_rate >= 0.15:
                labels.append("noisy")
            if frame.peak >= 0.92:
                labels.append("clipped_risk")
        labeled.append(
            {
                "frame": asdict(frame),
                "delta_rms": round(delta, 5),
                "labels": labels or ["sustain"],
            }
        )
    return labeled


def _waveform_summary(frame_entries: Sequence[dict]) -> str:
    counts = {name: 0 for name in WAVEFORM_GLOSSARY}
    for entry in frame_entries:
        for label in entry["labels"]:
            counts[label] = counts.get(label, 0) + 1
    active_labels = [name for name, count in counts.items() if count > 0]
    if not active_labels:
        return "The waveform does not contain enough signal to summarize yet."
    ordered = sorted(active_labels, key=lambda name: counts[name], reverse=True)
    top = ordered[:3]
    bits = [f"{label}={counts[label]}" for label in top]
    meaning_bits = [f"{label} means {WAVEFORM_GLOSSARY[label]}" for label in top]
    return f"Waveform pattern summary: {', '.join(bits)}. " + " ".join(meaning_bits)


def archive_wav_with_tokens(path: Path, *, runtime_root: Path, stem_prefix: str | None = None) -> WavArchiveBuild:
    runtime_root.mkdir(parents=True, exist_ok=True)
    stem = stem_prefix or path.stem
    wavtxt_path = runtime_root / f"{stem}.wavtxt"
    restored_path = runtime_root / f"{stem}.restored.wav"
    signal_tokens_path = runtime_root / f"{stem}.signal_tokens.json"
    waveform_meaning_path = runtime_root / f"{stem}.waveform_meaning.json"

    wav_text = encode_wav_to_text(path)
    wavtxt_path.write_text(wav_text, encoding="utf-8")
    decode_text_to_wav(wav_text, restored_path)

    original_bytes = _read_wave_bytes(path)
    restored_bytes = _read_wave_bytes(restored_path)
    original_hash = hashlib.sha256(original_bytes).hexdigest()
    restored_hash = hashlib.sha256(restored_bytes).hexdigest()

    frames = build_signal_tokens(path)
    frame_entries = _frame_labels(frames)
    waveform_summary = _waveform_summary(frame_entries)
    silence_ratio = sum(1 for frame in frames if frame.silence) / len(frames) if frames else 0.0

    with wave.open(str(path), "rb") as handle:
        duration_seconds = handle.getnframes() / handle.getframerate()

    signal_tokens_path.write_text(
        json.dumps(
            {
                "version": 1,
                "clip_path": str(path),
                "wav_sha256": original_hash,
                "frame_ms": 20,
                "duration_seconds": round(duration_seconds, 3),
                "frames": [asdict(frame) for frame in frames],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    waveform_meaning_path.write_text(
        json.dumps(
            {
                "version": 1,
                "clip_path": str(path),
                "wav_sha256": original_hash,
                "summary": waveform_summary,
                "glossary": WAVEFORM_GLOSSARY,
                "frames": frame_entries,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return WavArchiveBuild(
        clip_path=str(path),
        wavtxt_path=str(wavtxt_path),
        restored_path=str(restored_path),
        signal_tokens_path=str(signal_tokens_path),
        waveform_meaning_path=str(waveform_meaning_path),
        wav_sha256=original_hash,
        restored_sha256=restored_hash,
        frame_count=len(frames),
        duration_seconds=round(duration_seconds, 3),
        archive_chars=len(wav_text),
        silence_ratio=round(silence_ratio, 4),
        waveform_summary=waveform_summary,
    )
