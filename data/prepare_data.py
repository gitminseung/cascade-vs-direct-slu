"""Build the challenge splits.

Outputs
  data/eval/eval.jsonl + data/eval/audio/*.flac   <- Evaluation set (committed to the repo)
  data/train_audio/train.jsonl + audio/*.flac     <- audio training split for Direct models (not committed)
  data/text/train.jsonl, data/text/dev.jsonl      <- MASSIVE ko-KR text for the Cascade text classifier
  data/labels.json                                <- 60 intent labels

Source
  * Speech-MASSIVE ko-KR `validation` split (2,033 utt., CC BY-NC-SA 4.0)
    https://huggingface.co/datasets/FBK-MT/Speech-MASSIVE
  * MASSIVE 1.1 ko-KR text (train / dev partitions)
    https://github.com/alexa/massive

Split rule
  The Speech-MASSIVE ko-KR dev split is divided SPEAKER-DISJOINTLY (seed 42):
  ~70% of speakers -> train_audio, ~30% of speakers -> eval.
  Eval utterance IDs are also removed from the text dev set used for model selection,
  so no system ever sees the gold transcript of an eval utterance during training.
"""
import argparse
import io
import json
import sys
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, SR, write_jsonl  # noqa: E402

MASSIVE_URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"


def load_massive_text():
    print("[text] downloading MASSIVE 1.1 ...")
    raw = urllib.request.urlopen(MASSIVE_URL).read()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith("ko-KR.jsonl"))
        lines = tar.extractfile(member).read().decode("utf-8").splitlines()
    rows = [json.loads(l) for l in lines if l.strip()]
    return rows


def speaker_disjoint_split(groups, test_size, seed):
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx = np.arange(len(groups))
    train_idx, test_idx = next(gss.split(idx, groups=groups))
    return train_idx, test_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_speaker_ratio", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from datasets import load_dataset

    print("[audio] loading Speech-MASSIVE ko-KR validation ...")
    ds = load_dataset("FBK-MT/Speech-MASSIVE", "ko-KR", split="validation")
    cols = ds.column_names
    spk_col = "speaker_id" if "speaker_id" in cols else None
    if spk_col is None:
        print("[warn] no speaker_id column -> falling back to utterance-level split")
        groups = np.arange(len(ds))
    else:
        groups = np.array([str(s) for s in ds[spk_col]])

    tr_idx, ev_idx = speaker_disjoint_split(groups, args.eval_speaker_ratio, args.seed)

    def dump(indices, subdir, manifest_name):
        out_dir = DATA / subdir / "audio"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for i in indices:
            ex = ds[int(i)]
            wav = np.asarray(ex["audio"]["array"], dtype=np.float32)
            assert ex["audio"]["sampling_rate"] == SR
            rel = f"data/{subdir}/audio/{ex['id']}.flac"
            sf.write(DATA.parent / rel, wav, SR, format="FLAC")
            rows.append({
                "id": str(ex["id"]),
                "audio": rel,
                "utt": ex["utt"],
                "intent": ex["intent_str"],
                "scenario": ex.get("scenario_str"),
                "speaker": str(ex[spk_col]) if spk_col else None,
                "duration": round(len(wav) / SR, 3),
            })
        write_jsonl(DATA / subdir / manifest_name, rows)
        return rows

    eval_rows = dump(ev_idx, "eval", "eval.jsonl")
    train_rows = dump(tr_idx, "train_audio", "train.jsonl")

    # ---- text for the cascade classifier ----
    text = load_massive_text()
    eval_ids = {r["id"] for r in eval_rows}
    tr_text = [{"id": r["id"], "utt": r["utt"], "intent": r["intent"]}
               for r in text if r["partition"] == "train"]
    dev_text = [{"id": r["id"], "utt": r["utt"], "intent": r["intent"]}
                for r in text if r["partition"] == "dev" and r["id"] not in eval_ids]
    write_jsonl(DATA / "text" / "train.jsonl", tr_text)
    write_jsonl(DATA / "text" / "dev.jsonl", dev_text)

    labels = sorted({r["intent"] for r in tr_text})
    assert len(labels) == 60, len(labels)
    with open(DATA / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=1)

    def summary(name, rows):
        spk = len({r["speaker"] for r in rows})
        hrs = sum(r["duration"] for r in rows) / 3600
        print(f"  {name:12s} {len(rows):5d} utt  {spk:3d} speakers  {hrs:.2f} h  "
              f"{len(Counter(r['intent'] for r in rows))} intents")

    print("[done]")
    summary("train_audio", train_rows)
    summary("eval", eval_rows)
    print(f"  text train   {len(tr_text)} | text dev (eval ids removed) {len(dev_text)}")


if __name__ == "__main__":
    main()
