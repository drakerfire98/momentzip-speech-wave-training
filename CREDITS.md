# Credits

## Core Repo Purpose

This repository is for local-first waveform and word-shape training work that
feeds the wider MomentZip / AH speech stack.

The training repo is designed to preserve:

- exact raw WAV truth
- reversible `wavtxt` archives
- structural signal tokens
- waveform-meaning labels
- reusable per-word shape banks

## Dataset Sources

### Speech Commands

- Dataset page: https://www.tensorflow.org/datasets/catalog/speech_commands
- Paper: https://arxiv.org/abs/1804.03209
- Why it is used here:
  - isolated spoken words
  - many speakers saying the same short commands
  - best first source for teaching what a word looks like in waveform form

Attribution note:
- This repo does not claim ownership of Speech Commands or its source clips.
- Keep the upstream license and citation with any local staging manifest.

### VoxForge

- Homepage: https://www.voxforge.org/
- About: https://www.voxforge.org/home/about
- Why it is used here:
  - broader open speech variation
  - more acoustic diversity than tiny keyword-only sets

Attribution note:
- VoxForge discusses GPL-oriented licensing for its speech corpus.
- Keep redistribution conservative and avoid bundling derived data without
  checking the exact upstream terms that apply to the downloaded material.

### Mozilla Common Voice

- Homepage: https://www.mozillafoundation.org/en/common-voice
- Dataset page: https://datacollective.mozillafoundation.org/datasets/cmj8u3p1w0075nxxbe8bedl00
- Why it is used here:
  - accent variation
  - phrase variation
  - broader speech robustness work

Attribution note:
- Use Common Voice here for word and phrase variation, not as the default
  speaker-identity or cloning source.

## Local Project Credit

This training scaffold was assembled inside the OpenClaw research workspace to
support MomentZip v15 speech-wave research and future mass training.

Credit boundaries:

- upstream datasets keep their own licenses and authorship
- this repo only owns the local staging, archiving, and indexing code
- source URLs and attribution should stay attached to every derived manifest
