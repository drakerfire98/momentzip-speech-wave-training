"""Waveform-to-concept bridge for the speech-wave training repo."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from .emotional_contour_training import classify_emotion_from_wav_path
from .proto_phoneme_training import build_clip_signature, extract_proto_sequence, signature_similarity


def _levenshtein(left: List[str], right: List[str]) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)
    rows = len(left) + 1
    cols = len(right) + 1
    table = [[0] * cols for _ in range(rows)]
    for row in range(rows):
        table[row][0] = row
    for col in range(cols):
        table[0][col] = col
    for row in range(1, rows):
        for col in range(1, cols):
            cost = 0 if left[row - 1] == right[col - 1] else 1
            table[row][col] = min(
                table[row - 1][col] + 1,
                table[row][col - 1] + 1,
                table[row - 1][col - 1] + cost,
            )
    return table[-1][-1]


def _sequence_similarity(left: List[str], right: List[str]) -> float:
    if not left and not right:
        return 1.0
    distance = _levenshtein(left, right)
    scale = max(len(left), len(right), 1)
    return max(0.0, 1.0 - (distance / scale))


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_concept_bridge(
    artifacts_root: Path,
    output_path: Path,
) -> Path:
    label_bank = _load_json(artifacts_root / "label_bank.json")
    proto_index = _load_json(artifacts_root / "proto_phoneme_index.json")
    invariant_bank = _load_json(artifacts_root / "speaker_invariant_bank.json")
    emotional_bank = _load_json(artifacts_root / "word_emotional_overlay.json")
    transition_bank = _load_json(artifacts_root / "voxforge_transition_bank.json")

    cluster_id_map = {
        cluster["shape_key"]: f"cluster_{index + 1}"
        for index, cluster in enumerate(proto_index.get("clusters", []))
    }

    clip_sequences: Dict[str, List[str]] = {}
    clip_labels: Dict[str, str] = {}
    for segment in proto_index.get("segments", []):
        clip_path = str(segment["clip_path"])
        clip_sequences.setdefault(clip_path, [])
        clip_sequences[clip_path].append(cluster_id_map.get(segment["shape_key"], segment["shape_key"]))
        clip_labels[clip_path] = str(segment["label"])

    label_sequences: Dict[str, Dict[str, int]] = defaultdict(dict)
    label_clip_examples: Dict[str, List[dict]] = defaultdict(list)
    for clip_path, sequence in clip_sequences.items():
        label = clip_labels[clip_path]
        sequence_key = "|".join(sequence)
        label_sequences[label][sequence_key] = label_sequences[label].get(sequence_key, 0) + 1
        label_clip_examples[label].append(
            {
                "clip_path": clip_path,
                "proto_phoneme_sequence": sequence,
            }
        )

    transition_contexts: Dict[str, List[str]] = {}
    for word in label_bank.get("labels", {}):
        contexts = [
            transition["pair"]
            for transition in transition_bank.get("transitions", [])
            if f" {word.upper()}" in transition["pair"] or transition["pair"].startswith(f"{word.upper()} ")
        ]
        transition_contexts[word] = contexts[:12]

    bridge = {
        "version": 2,
        "kind": "concept_bridge",
        "shape_key_to_cluster_id": dict(sorted(cluster_id_map.items())),
        "entries": {},
    }
    for word in sorted(label_bank.get("labels", {})):
        invariant_entry = invariant_bank["labels"][word]
        overlay_entry = emotional_bank["words"].get(word, {"variants": {}, "dominant_emotion": "calm"})
        sequence_variants = label_sequences.get(word, {})
        best_sequence_key = max(sequence_variants.items(), key=lambda item: item[1])[0] if sequence_variants else ""
        ranked_variants = sorted(sequence_variants.items(), key=lambda item: item[1], reverse=True)
        bridge["entries"][word] = {
            "word": word,
            "text_concept_id": word,
            "proto_phoneme_sequence": [token for token in best_sequence_key.split("|") if token],
            "proto_phoneme_variants": [
                {
                    "sequence": [token for token in sequence_key.split("|") if token],
                    "count": count,
                }
                for sequence_key, count in ranked_variants[:32]
            ],
            "speaker_invariant_prototype": invariant_entry["prototype"],
            "speaker_prototype_hash": invariant_entry["prototype_hash"],
            "emotional_variants": [f"{emotion}_{word}" for emotion in overlay_entry["variants"].keys()],
            "dominant_emotion": overlay_entry["dominant_emotion"],
            "transition_contexts": transition_contexts.get(word, []),
            "example_clip_paths": [example["clip_path"] for example in label_clip_examples.get(word, [])[:8]],
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bridge, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def recognize(wav_path: Path, artifacts_root: Path) -> dict:
    bridge = _load_json(artifacts_root / "concept_bridge.json")
    cluster_map = bridge.get("shape_key_to_cluster_id", {})
    proto_sequence = [cluster_map.get(token, token) for token in extract_proto_sequence(wav_path)]
    clip_signature = build_clip_signature(wav_path)
    emotion_name, emotion_signature = classify_emotion_from_wav_path(wav_path)

    best_word = ""
    best_score = -1.0
    best_sequence_score = 0.0
    best_speaker_score = 0.0
    for word, entry in bridge.get("entries", {}).items():
        sequence_variants = entry.get("proto_phoneme_variants") or [
            {"sequence": list(entry.get("proto_phoneme_sequence", [])), "count": 1}
        ]
        sequence_score = 0.0
        for variant in sequence_variants:
            sequence_score = max(sequence_score, _sequence_similarity(proto_sequence, list(variant["sequence"])))
        speaker_score = signature_similarity(clip_signature, entry["speaker_invariant_prototype"])
        score = sequence_score * 0.72 + speaker_score * 0.28
        if score > best_score:
            best_score = score
            best_word = word
            best_sequence_score = sequence_score
            best_speaker_score = speaker_score

    return {
        "wav_path": str(wav_path),
        "matched_word": best_word,
        "sequence_similarity": round(best_sequence_score, 5),
        "speaker_similarity": round(best_speaker_score, 5),
        "emotional_classification": emotion_name,
        "emotion_signature": emotion_signature,
        "suggested_concept_graph_node": best_word,
    }
