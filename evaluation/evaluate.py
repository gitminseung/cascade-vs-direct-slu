"""Score one system's predictions.

  python evaluation/evaluate.py results/preds_cascade_whisper-small.jsonl

Metrics
  accuracy   (primary)  top-1 intent accuracy
  macro_f1              F1 averaged over intents present in the eval set
  ece                   expected calibration error, 15 equal-width bins on top-1 confidence
  latency_ms            median / p90 end-to-end per-utterance latency (batch 1)
  cer                   ASR character error rate (cascade systems only; spaces & punctuation removed)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RESULTS, expected_calibration_error, normalize_for_cer, read_jsonl  # noqa

SYSTEM_INFO = {
    "oracle_text": ("Oracle", "Gold transcript → KLUE-RoBERTa", True),
    "cascade_whisper-small": ("Cascade", "Whisper-small → KLUE-RoBERTa", False),
    "cascade_whisper-large-v3-turbo": ("Cascade", "Whisper-large-v3-turbo → KLUE-RoBERTa", False),
    "direct_whisper-small": ("Direct", "Whisper-small encoder + attentive pooling", False),
}


def score(path):
    rows = read_jsonl(path)
    name = Path(path).stem.replace("preds_", "")
    gold = [r["gold"] for r in rows]
    pred = [r["pred"] for r in rows]
    correct = np.array([g == p for g, p in zip(gold, pred)])
    lat = np.array([r["latency_ms"] for r in rows])
    track, desc, reference = SYSTEM_INFO.get(name, ("Other", name, False))
    m = {
        "system": name, "track": track, "description": desc, "reference_only": reference,
        "n": len(rows),
        "accuracy": round(float(correct.mean()), 4),
        "macro_f1": round(float(f1_score(gold, pred, labels=sorted(set(gold)), average="macro",
                                         zero_division=0)), 4),
        "ece": round(expected_calibration_error([r["conf"] for r in rows], correct), 4),
        "latency_ms_median": round(float(np.median(lat)), 1),
        "latency_ms_p90": round(float(np.percentile(lat, 90)), 1),
        "cer": None,
    }
    if all("hyp" in r for r in rows) and not reference:
        import jiwer
        refs = [normalize_for_cer(r["ref"]) for r in rows]
        hyps = [normalize_for_cer(r["hyp"]) for r in rows]
        keep = [i for i, x in enumerate(refs) if x]
        m["cer"] = round(float(jiwer.cer([refs[i] for i in keep], [hyps[i] for i in keep])), 4)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+")
    args = ap.parse_args()
    for p in args.preds:
        m = score(p)
        out = RESULTS / f"metrics_{m['system']}.json"
        out.write_text(json.dumps(m, indent=1, ensure_ascii=False))
        print(json.dumps(m, ensure_ascii=False))


if __name__ == "__main__":
    main()
