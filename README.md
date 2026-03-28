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
- `datasets/commands.json`
  - command shortcuts for mass fetch and rebuild flows
- `downloads/`
  - raw source archives kept inside the repo workspace
- `staged/`
  - label-organized WAV clips ready for isolated-word learning
- `momentzip_speech_wave_training/`
  - standalone archive and 3-layer indexing code
- `scripts/prepare_speech_commands.py`
  - stages isolated-word clips into this repo
- `scripts/download_open_corpora.py`
  - downloads official open-source corpora into `downloads/`
- `scripts/extract_downloads.py`
  - extracts repo-managed archives into `downloads/extracted/`
- `scripts/build_word_shape_index.py`
  - turns staged clips into waveform, hash, and 3-layer indexes
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
- `artifacts/layer_v1_raw_index.json`
- `artifacts/layer_v2_signal_index.json`
- `artifacts/layer_v3_semantic_index.json`
- `artifacts/bootstrap_report.json`

## Three-Layer Compression Model

- `v1 raw`
  - exact clip truth and reversible `wavtxt`
- `v2 signal`
  - structural frame tokens, waveform meaning, and signal hashes
- `v3 semantic`
  - aggregated label banks, dominant waveform traits, and semantic hashes

This is the repo form of the AH speech 3-version system and hash system.

## GitHub Publish Note

This scaffold is now published as its own GitHub repo. `PUBLISHING.md` still
documents the local publish flow if the repo ever needs to be recreated.
