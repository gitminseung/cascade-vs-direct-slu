"""Cascade baseline: Whisper ASR -> text intent classifier.

  python baseline/cascade.py --asr openai/whisper-small
  python baseline/cascade.py --asr openai/whisper-large-v3-turbo
  python baseline/cascade.py --oracle          # gold transcript -> classifier (upper bound)

Writes results/preds_<system>.jsonl with per-utterance probabilities and latency.
Latency = end-to-end wall-clock per utterance (batch size 1, fp16 on GPU),
covering feature extraction + ASR decoding + text classification.
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          WhisperForConditionalGeneration, WhisperProcessor)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (CKPT, EVAL_MANIFEST, RESULTS, SR, load_audio, load_labels,  # noqa
                    normalize_text, read_jsonl, write_jsonl)


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr", default="openai/whisper-small")
    ap.add_argument("--clf", default=str(CKPT / "text_clf"))
    ap.add_argument("--manifest", default=str(EVAL_MANIFEST))
    ap.add_argument("--oracle", action="store_true", help="use gold transcripts instead of ASR")
    ap.add_argument("--name", default=None)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    labels, l2i = load_labels()
    rows = read_jsonl(args.manifest)

    tok = AutoTokenizer.from_pretrained(args.clf)
    clf = AutoModelForSequenceClassification.from_pretrained(args.clf, torch_dtype=dtype).to(device).eval()
    assert clf.config.num_labels == len(labels)

    if not args.oracle:
        proc = WhisperProcessor.from_pretrained(args.asr)
        asr = WhisperForConditionalGeneration.from_pretrained(args.asr, torch_dtype=dtype).to(device).eval()

    name = args.name or ("oracle_text" if args.oracle else "cascade_" + args.asr.split("/")[-1])

    @torch.no_grad()
    def run(r):
        if args.oracle:
            hyp = r["utt"]
        else:
            wav = load_audio(r["audio"])
            feats = proc(wav, sampling_rate=SR, return_tensors="pt").input_features.to(device, dtype)
            ids = asr.generate(feats, language="ko", task="transcribe", max_new_tokens=128)
            hyp = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        enc = tok(normalize_text(hyp), truncation=True, max_length=64, return_tensors="pt").to(device)
        probs = torch.softmax(clf(**enc).logits.float(), -1)[0].cpu()
        return hyp, probs

    for r in rows[: args.warmup]:  # warm-up (not timed)
        run(r)

    out = []
    for r in tqdm(rows, desc=name):
        sync(device); t0 = time.perf_counter()
        hyp, probs = run(r)
        sync(device); ms = (time.perf_counter() - t0) * 1000
        out.append({
            "id": r["id"], "gold": r["intent"], "pred": labels[int(probs.argmax())],
            "conf": float(probs.max()), "probs": [round(float(p), 5) for p in probs],
            "latency_ms": round(ms, 2), "ref": r["utt"], "hyp": hyp,
        })
    path = RESULTS / f"preds_{name}.jsonl"
    write_jsonl(path, out)
    acc = sum(o["gold"] == o["pred"] for o in out) / len(out)
    print(f"[{name}] acc={acc:.4f} -> {path}")


if __name__ == "__main__":
    main()
