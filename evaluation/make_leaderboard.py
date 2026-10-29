"""Collect results/metrics_*.json into a leaderboard.

  python evaluation/make_leaderboard.py
  python evaluation/make_leaderboard.py --html ../Speech-Recognition-Challenge-2026/submission/Team_KimMinseung_202020760.html

Ranking: accuracy (desc), ties broken by median latency (asc).
The oracle system is listed as a reference row and is not ranked.
With --html, the JSON block inside <script id="lb-data"> of the challenge page is replaced in place.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RESULTS  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", default=None)
    args = ap.parse_args()

    ms = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("metrics_*.json"))]
    if not ms:
        sys.exit("no results/metrics_*.json found — run evaluation/evaluate.py first")
    ranked = sorted([m for m in ms if not m["reference_only"]],
                    key=lambda m: (-m["accuracy"], m["latency_ms_median"]))
    refs = [m for m in ms if m["reference_only"]]
    for i, m in enumerate(ranked, 1):
        m["rank"] = i
    for m in refs:
        m["rank"] = None
    board = ranked + refs

    (RESULTS / "leaderboard.json").write_text(json.dumps(board, indent=1, ensure_ascii=False))
    fmt = lambda v, p=4: "—" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))
    lines = ["| Rank | System | Track | Acc ↑ | Macro-F1 ↑ | ECE ↓ | Latency (ms, median) ↓ | CER ↓ |",
             "|---|---|---|---|---|---|---|---|"]
    for m in board:
        lines.append(f"| {m['rank'] or 'ref'} | {m['description']} | {m['track']} | {fmt(m['accuracy'])} | "
                     f"{fmt(m['macro_f1'])} | {fmt(m['ece'])} | {fmt(m['latency_ms_median'], 1)} | {fmt(m['cer'])} |")
    md = "\n".join(lines)
    (RESULTS / "leaderboard.md").write_text(md + "\n")
    print(md)

    if args.html:
        html = Path(args.html).read_text(encoding="utf-8")
        payload = json.dumps(board, ensure_ascii=False, indent=1)
        new, n = re.subn(r'(<script id="lb-data" type="application/json">)(.*?)(</script>)',
                         lambda mo: mo.group(1) + "\n" + payload + "\n" + mo.group(3), html, flags=re.S)
        if n != 1:
            sys.exit("could not find <script id=\"lb-data\"> block in the html")
        Path(args.html).write_text(new, encoding="utf-8")
        print(f"updated leaderboard in {args.html}")


if __name__ == "__main__":
    main()
