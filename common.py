"""Shared helpers for the Cascade vs Direct challenge."""
import json
import math
import os
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CKPT = ROOT / "checkpoints"
SR = 16000

EVAL_MANIFEST = DATA / "eval" / "eval.jsonl"
TRAIN_AUDIO_MANIFEST = DATA / "train_audio" / "train.jsonl"
TEXT_TRAIN = DATA / "text" / "train.jsonl"
TEXT_DEV = DATA / "text" / "dev.jsonl"
LABELS = DATA / "labels.json"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_labels():
    with open(LABELS, encoding="utf-8") as f:
        labels = json.load(f)
    return labels, {l: i for i, l in enumerate(labels)}


_PUNCT = re.compile(r"[\.,\?\!~\"'“”‘’…·:;()\[\]{}<>-]")


def normalize_text(s: str) -> str:
    """Light normalization applied to BOTH gold text and ASR hypotheses
    before classification (removes punctuation Whisper tends to add)."""
    s = _PUNCT.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_for_cer(s: str) -> str:
    """CER is computed on characters without spaces/punctuation."""
    return normalize_text(s).replace(" ", "")


def load_audio(path):
    import soundfile as sf
    wav, sr = sf.read(str(ROOT / path) if not os.path.isabs(path) else path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    assert sr == SR, f"expected {SR} Hz, got {sr} ({path})"
    return wav


def expected_calibration_error(conf, correct, n_bins=15):
    """Standard top-label ECE with equal-width bins."""
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def n_encoder_frames(num_samples: int) -> int:
    """Whisper encoder emits 50 frames per second (max 1500 for 30 s)."""
    return max(1, min(1500, math.ceil(num_samples / SR * 50)))
