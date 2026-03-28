# Dataset Credits And Use Notes

## 1. Speech Commands

- Source: TensorFlow Datasets catalog
  - https://www.tensorflow.org/datasets/catalog/speech_commands
- Paper:
  - https://arxiv.org/abs/1804.03209
- Best use here:
  - isolated spoken words
  - keyword shape learning
  - multi-speaker variation for the same word

Why it fits:
- one-second word clips
- many speakers
- cleanest starting point for teaching the AH what a single word looks like in waveform form

## 2. VoxForge

- Source:
  - https://www.voxforge.org/
- About and licensing context:
  - https://www.voxforge.org/home/about

Best use here:
- broader open speech variation
- acoustic variation beyond tiny keyword sets

Important note:
- VoxForge states that submitted speech is made available under GPL-style terms, so derivative packaging needs to respect that boundary.

## 3. Mozilla Common Voice

- Source:
  - https://www.mozillafoundation.org/common-voice/platform-and-dataset/
- TFDS catalog:
  - https://www.tensorflow.org/datasets/catalog/common_voice

Best use here:
- accent and phrasing variation
- broader sentence speech

Important restriction for this repo:
- do not position Common Voice as the primary source for speaker-identity training
- use it for broader word and phrase variation instead

## Repo Policy

- Keep dataset sources external unless redistribution terms are explicitly cleared.
- Keep source URLs and citations with every ingestion manifest.
- Treat identity, cloning, and speaker-sensitive work as opt-in and more restricted than generic speech-pattern learning.

## Practical Training Order

1. Start with `Speech Commands`.
Why:
- it gives the cleanest repeated word shapes across many speakers
- it is the best source for building a first reusable `label_bank.json`

2. Expand with `VoxForge`.
Why:
- it broadens acoustic variation once the isolated-word path is stable

3. Use `Common Voice` later and more selectively.
Why:
- it is better for phrase and accent robustness than for the very first
  isolated-word bank

## Processing Notes

- `Speech Commands`
  - direct fit for the full 3-layer system
  - WAV files can be promoted straight into raw, signal, and semantic layers

- `VoxForge`
  - strong mass open corpus
  - best used for raw and signal learning first, then promoted further as
    segmentation and transcript mapping improve

- `Common Voice`
  - good broader source, but many distributions are MP3-based
  - keep it organized in the repo workspace and only promote it into the
    WAV-based 3-layer system after an explicit conversion route exists
