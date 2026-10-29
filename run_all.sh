#!/usr/bin/env bash
# Full reproduction on a single GPU (Colab T4 ~ 40–60 min incl. downloads).
set -euo pipefail
cd "$(dirname "$0")"

# 1) data (skips if the eval set is already in the repo and train audio exists)
if [ ! -f data/train_audio/train.jsonl ] || [ ! -f data/text/train.jsonl ]; then
  python data/prepare_data.py
fi

# 2) cascade stage-2 text classifier
python baseline/train_text_clf.py

# 3) systems
python baseline/cascade.py --oracle
python baseline/cascade.py --asr openai/whisper-small
python baseline/cascade.py --asr openai/whisper-large-v3-turbo
python baseline/train_direct.py --encoder openai/whisper-small
python baseline/direct_infer.py --ckpt checkpoints/direct_whisper-small

# 4) evaluation + leaderboard
python evaluation/evaluate.py results/preds_*.jsonl
python evaluation/make_leaderboard.py "$@"
