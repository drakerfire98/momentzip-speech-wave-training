# MomentZip Speech Wave Training

Local-first training workspace for teaching the AH what spoken words look like in:

- raw WAV bytes
- reversible `wavtxt` archives
- structural signal tokens
- waveform-meaning labels
- higher-level voice and word-shape features

## Goals

- Learn how isolated words vary across speakers.
- Preserve exact raw audio truth for later replay or verification.
- Build word-shape indexes from waveform structure instead of only plain transcripts.
- Keep source attribution and dataset-license boundaries explicit.

## Initial Dataset Plan

- `Speech Commands` for isolated spoken words and short commands.
- `VoxForge` for broader open speech variation.
- `Common Voice` only for broader STT-style variation, not for speaker-identity or cloning work.

Details and credits live in [`DATASETS.md`](DATASETS.md).

## Layout

- `datasets/registry.json`
  - source registry and license notes
- `scripts/prepare_speech_commands.py`
  - stages isolated-word clips into this repo
- `scripts/build_word_shape_index.py`
  - turns staged clips into waveform and signal-token indexes
- `artifacts/`
  - local outputs, indexes, manifests, and generated archives

## Current Status

The core MomentZip stack already knows how to:

- archive WAV as `wavtxt`
- restore WAV exactly
- emit signal tokens
- emit waveform-meaning summaries

This repo is the training surface that will feed those layers at scale.

## Quick Start

If you already have an unpacked Speech Commands dataset:

```powershell
python scripts/bootstrap_training_repo.py --source-root C:\path\to\speech_commands
```

If you only want to prove the repo itself works before using real clips:

```powershell
python scripts/smoke_test_training_repo.py
```

The main outputs are:

- `artifacts/word_shape_index.json`
- `artifacts/label_bank.json`
- `artifacts/bootstrap_report.json`

## GitHub Publish Note

This scaffold is ready for a dedicated GitHub repo, but the current `gh` login on this machine is invalid, so the publish step is blocked until GitHub auth is fixed.

See `PUBLISHING.md` for the clean push flow once auth is repaired.
