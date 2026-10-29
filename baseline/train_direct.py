"""Train the Direct baseline head on the speaker-disjoint audio training split.

Encoder states are computed once and cached (encoder is frozen), then only the
attentive-pooling head is trained. ~1.4k utterances -> a few minutes on a Colab T4.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import WhisperModel, WhisperProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CKPT, TRAIN_AUDIO_MANIFEST, load_audio, load_labels, read_jsonl  # noqa
from direct_model import AttentivePoolingHead, encode  # noqa


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def batches(items, bs, shuffle):
    idx = list(range(len(items)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), bs):
        sel = [items[j] for j in idx[i:i + bs]]
        h = pad_sequence([x[0] for x in sel], batch_first=True)
        lens = torch.tensor([x[0].shape[0] for x in sel])
        mask = torch.arange(h.shape[1])[None, :] < lens[:, None]
        y = torch.tensor([x[1] for x in sel])
        yield h, mask, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="openai/whisper-small")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    out = Path(args.out or CKPT / f"direct_{args.encoder.split('/')[-1]}")
    out.mkdir(parents=True, exist_ok=True)

    labels, l2i = load_labels()
    rows = read_jsonl(TRAIN_AUDIO_MANIFEST)
    proc = WhisperProcessor.from_pretrained(args.encoder)
    encoder = WhisperModel.from_pretrained(args.encoder, torch_dtype=dtype).encoder.to(device).eval()

    cache = []
    for r in tqdm(rows, desc="caching encoder states"):
        h = encode(encoder, proc, load_audio(r["audio"]), device, dtype).float().cpu()
        cache.append((h, l2i[r["intent"]]))
    del encoder; torch.cuda.empty_cache()

    random.shuffle(cache)
    n_val = max(1, int(len(cache) * args.val_ratio))
    val, train = cache[:n_val], cache[n_val:]
    d_model = train[0][0].shape[-1]

    head = AttentivePoolingHead(d_model, len(labels)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = -1.0
    for ep in range(1, args.epochs + 1):
        head.train()
        for h, m, y in batches(train, args.bs, True):
            loss = F.cross_entropy(head(h.to(device), m.to(device)), y.to(device), label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        head.eval(); correct = 0
        with torch.no_grad():
            for h, m, y in batches(val, 64, False):
                correct += (head(h.to(device), m.to(device)).argmax(-1).cpu() == y).sum().item()
        acc = correct / len(val)
        if acc > best:
            best = acc
            torch.save(head.state_dict(), out / "head.pt")
        print(f"ep{ep:02d} loss {loss.item():.3f} val acc {acc:.4f} (best {best:.4f})")

    with open(out / "config.json", "w") as f:
        json.dump({"encoder": args.encoder, "d_model": d_model, "n_classes": len(labels),
                   "best_val_acc": best, "n_train": len(train), "n_val": len(val), **vars(args)},
                  f, indent=1)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
