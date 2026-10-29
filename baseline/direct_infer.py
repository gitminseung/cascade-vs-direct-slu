"""Run the Direct baseline on the evaluation set.

Latency = end-to-end wall-clock per utterance (batch 1, fp16): feature extraction
+ Whisper encoder forward + head. No autoregressive decoding.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import WhisperModel, WhisperProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CKPT, EVAL_MANIFEST, RESULTS, load_audio, load_labels, read_jsonl, write_jsonl  # noqa
from direct_model import AttentivePoolingHead, encode  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CKPT / "direct_whisper-small"))
    ap.add_argument("--manifest", default=str(EVAL_MANIFEST))
    ap.add_argument("--name", default=None)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    labels, _ = load_labels()
    cfg = json.load(open(Path(args.ckpt) / "config.json"))
    proc = WhisperProcessor.from_pretrained(cfg["encoder"])
    encoder = WhisperModel.from_pretrained(cfg["encoder"], torch_dtype=dtype).encoder.to(device).eval()
    head = AttentivePoolingHead(cfg["d_model"], cfg["n_classes"]).to(device).eval()
    head.load_state_dict(torch.load(Path(args.ckpt) / "head.pt", map_location=device))
    name = args.name or "direct_" + cfg["encoder"].split("/")[-1]
    rows = read_jsonl(args.manifest)

    @torch.no_grad()
    def run(r):
        h = encode(encoder, proc, load_audio(r["audio"]), device, dtype).float()
        mask = torch.ones(1, h.shape[0], dtype=torch.bool, device=device)
        return torch.softmax(head(h[None], mask), -1)[0].cpu()

    for r in rows[: args.warmup]:
        run(r)

    out = []
    for r in tqdm(rows, desc=name):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        probs = run(r)
        if device == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        out.append({
            "id": r["id"], "gold": r["intent"], "pred": labels[int(probs.argmax())],
            "conf": float(probs.max()), "probs": [round(float(p), 5) for p in probs],
            "latency_ms": round(ms, 2),
        })
    path = RESULTS / f"preds_{name}.jsonl"
    write_jsonl(path, out)
    acc = sum(o["gold"] == o["pred"] for o in out) / len(out)
    print(f"[{name}] acc={acc:.4f} -> {path}")


if __name__ == "__main__":
    main()
