# Publishing Notes

## Current State

This repo scaffold is ready to be pushed to a dedicated GitHub repository, but
the current local `gh` login on this machine is invalid.

Until GitHub auth is fixed, treat this directory as the source-of-truth draft.

## Suggested Publish Flow

1. Fix GitHub authentication on the machine.
2. Create a new repository, for example `momentzip-speech-wave-training`.
3. Push this directory as the initial commit.
4. Keep dataset downloads and staged artifacts out of Git.

## Suggested Commands

```powershell
cd research/autoresearch/test_training/momentzip_v15/momentzip-speech-wave-training
git init
git add .
git commit -m "Initial speech-wave training scaffold"
gh repo create momentzip-speech-wave-training --public --source . --remote origin --push
```

If you want the repo private first, replace `--public` with `--private`.
