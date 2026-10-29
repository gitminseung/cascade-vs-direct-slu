"""Cascade stage 2: Korean text intent classifier (KLUE-RoBERTa-base).

Trained on MASSIVE ko-KR text `train` (11,514 utt.), model selection on
MASSIVE ko-KR `dev` with eval-set IDs removed.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CKPT, TEXT_DEV, TEXT_TRAIN, load_labels, normalize_text, read_jsonl  # noqa


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def accuracy(model, tok, rows, l2i, device, bs=128):
    model.eval()
    correct = 0
    for i in range(0, len(rows), bs):
        batch = rows[i:i + bs]
        enc = tok([normalize_text(r["utt"]) for r in batch], padding=True, truncation=True,
                  max_length=64, return_tensors="pt").to(device)
        pred = model(**enc).logits.argmax(-1).cpu().tolist()
        correct += sum(p == l2i[r["intent"]] for p, r in zip(pred, batch))
    return correct / len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="klue/roberta-base")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(CKPT / "text_clf"))
    args = ap.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    labels, l2i = load_labels()
    train, dev = read_jsonl(TEXT_TRAIN), read_jsonl(TEXT_DEV)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(labels),
        id2label=dict(enumerate(labels)), label2id=l2i).to(device)

    def collate(batch):
        enc = tok([normalize_text(r["utt"]) for r in batch], padding=True, truncation=True,
                  max_length=64, return_tensors="pt")
        enc["labels"] = torch.tensor([l2i[r["intent"]] for r in batch])
        return enc

    dl = DataLoader(train, batch_size=args.bs, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")

    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        for step, batch in enumerate(dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = model(**batch).loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            if step % 100 == 0:
                print(f"ep{ep} step{step}/{len(dl)} loss {loss.item():.4f}")
        acc = accuracy(model, tok, dev, l2i, device)
        print(f"[ep{ep}] dev acc = {acc:.4f}")
        if acc > best:
            best = acc
            model.save_pretrained(args.out); tok.save_pretrained(args.out)
            print(f"  saved -> {args.out}")
    with open(Path(args.out) / "train_info.json", "w") as f:
        json.dump({"base": args.model, "best_dev_acc": best, **vars(args)}, f, indent=1)
    print(f"best dev acc {best:.4f}")


if __name__ == "__main__":
    main()
