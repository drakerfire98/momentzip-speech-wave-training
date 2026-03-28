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
- `momentzip_speech_wave_training/proto_phoneme_training.py`
  - unsupervised subword segmentation and speaker-invariant prototype building
- `momentzip_speech_wave_training/transition_learning.py`
  - continuous-speech transition learning from VoxForge sentence archives
- `momentzip_speech_wave_training/emotional_contour_training.py`
  - emotional signature extraction and word-level emotion overlays
- `momentzip_speech_wave_training/concept_bridge.py`
  - direct waveform-to-concept bridge plus recognition
- `scripts/prepare_speech_commands.py`
  - stages isolated-word clips into this repo
- `scripts/download_open_corpora.py`
  - downloads official open-source corpora into `downloads/`
- `scripts/extract_downloads.py`
  - extracts repo-managed archives into `downloads/extracted/`
- `scripts/build_word_shape_index.py`
  - turns staged clips into waveform, hash, and 3-layer indexes
- `scripts/build_proto_phoneme_bank.py`
  - builds proto-phoneme clusters and speaker-invariant word shapes
- `scripts/build_transition_bank.py`
  - builds sentence-blend signatures and word-to-word transition prototypes
- `scripts/build_emotional_contour_bank.py`
  - builds emotional contour prototypes and per-word emotion overlays
- `scripts/build_concept_bridge.py`
  - builds the waveform-to-concept bridge and can recognize a new WAV clip
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
- `artifacts/proto_phoneme_index.json`
- `artifacts/speaker_invariant_bank.json`
- `artifacts/voxforge_transition_bank.json`
- `artifacts/voxforge_sentence_blends.json`
- `artifacts/emotional_contour_bank.json`
- `artifacts/word_emotional_overlay.json`
- `artifacts/concept_bridge.json`
- `artifacts/bootstrap_report.json`

## Three-Layer Compression Model

- `v1 raw`
  - exact clip truth and reversible `wavtxt`
- `v2 signal`
  - structural frame tokens, waveform meaning, and signal hashes
- `v3 semantic`
  - aggregated label banks, dominant waveform traits, and semantic hashes

This is the repo form of the AH speech 3-version system and hash system.

## Next Speech Layer

The repo now also supports:

- `proto phoneme` shapes
  - unsupervised subword-like segments learned from waveform changes
- `speaker invariant` shapes
  - per-word prototypes normalized across different speakers

These are the honest first steps toward phoneme decomposition and
speaker-invariant recognition. They are not full IPA-aligned phoneme learning
yet.

## Continuous Speech Layer

The repo also supports a first continuous-speech pass from VoxForge sentence
packs:

- sentence-level blend signatures
- approximate word-to-word transition prototypes

This is not forced alignment yet. It uses prompt text plus normalized sentence
timing to learn likely boundary shapes between adjacent words.

## Emotion And Concept Bridge

The repo now also supports:

- emotional contour overlays like `calm`, `urgent`, `hesitant`, `confident`,
  and `stressed`
- direct waveform-to-concept bridging for the trained words
- a `recognize(wav_path)` pipeline that returns:
  - matched word
  - speaker similarity
  - emotion classification
  - suggested concept node

## GitHub Publish Note

This scaffold is now published as its own GitHub repo. `PUBLISHING.md` still
documents the local publish flow if the repo ever needs to be recreated.
