"""Smoke-test the standalone speech-wave training repo."""

from __future__ import annotations

import math
import struct
import subprocess
import sys
import wave
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent


def _write_tone_word(path: Path, *, sample_rate: int, segments: list[tuple[float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = bytearray()
    for duration_seconds, frequency_hz, amplitude in segments:
        frame_count = int(sample_rate * duration_seconds)
        for index in range(frame_count):
            if frequency_hz <= 0:
                value = 0
            else:
                phase = 2.0 * math.pi * frequency_hz * (index / sample_rate)
                value = int(max(-1.0, min(1.0, math.sin(phase) * amplitude)) * 32767)
            samples.extend(struct.pack("<h", value))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(bytes(samples))


def _build_synthetic_source(source_root: Path) -> None:
    sample_rate = 16000
    # The smoke test uses repeatable synthetic shapes so the standalone repo can
    # prove the archive/index pipeline without shipping real speech data.
    words = {
        "yes": [
            [(0.12, 420.0, 0.35), (0.05, 0.0, 0.0), (0.18, 540.0, 0.45)],
            [(0.09, 430.0, 0.32), (0.04, 0.0, 0.0), (0.2, 560.0, 0.48)],
        ],
        "no": [
            [(0.18, 240.0, 0.30), (0.03, 0.0, 0.0), (0.12, 180.0, 0.40)],
            [(0.15, 220.0, 0.28), (0.04, 0.0, 0.0), (0.14, 170.0, 0.42)],
        ],
    }
    for label, shapes in words.items():
        for index, segments in enumerate(shapes, start=1):
            _write_tone_word(source_root / label / f"{label}_{index}.wav", sample_rate=sample_rate, segments=segments)


def main() -> None:
    smoke_root = REPO_ROOT / "artifacts" / "smoke"
    source_root = smoke_root / "source"
    _build_synthetic_source(source_root)

    command = [
        sys.executable,
        str(THIS_DIR / "bootstrap_training_repo.py"),
        "--source-root",
        str(source_root),
        "--labels",
        "yes",
        "no",
        "--limit-per-label",
        "2",
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    print(completed.stdout.strip())


if __name__ == "__main__":
    main()
