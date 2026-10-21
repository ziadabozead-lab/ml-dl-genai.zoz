"""
Download an English-Arabic parallel corpus (OPUS-100, via Hugging Face) and
save it in the format train_translate.py expects: one "english<TAB>arabic"
sentence pair per line.

    pip install datasets
    python scripts/prepare_translate_data.py                     # ~40,000 pairs (default cap)
    python scripts/prepare_translate_data.py --max-pairs 15000    # smaller / faster to train
    python scripts/prepare_translate_data.py --max-pairs 0        # everything (~1 million pairs - slow)

If you don't have internet access to Hugging Face from this machine, you can
instead hand-write your own rdf/translate_data.txt with lines like:
    hello<TAB>مرحبا
    how are you<TAB>كيف حالك
"""
import argparse
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WS_RE = re.compile(r"\s+")


def clean(s):
    return WS_RE.sub(" ", s.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "rdf" / "translate_data.txt"))
    ap.add_argument("--max-pairs", type=int, default=40000, help="0 = keep everything")
    ap.add_argument("--max-len", type=int, default=25, help="skip sentences longer than this many words")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Please run:  pip install datasets")

    print("Downloading OPUS-100 (en-ar) ... this can take a few minutes the first time.")
    ds = load_dataset("Helsinki-NLP/opus-100", "ar-en")

    rng = random.Random(args.seed)
    seen, rows = set(), []
    for split in ds:                       # train + validation + test; the trainer re-splits anyway
        for row in ds[split]:
            pair = row["translation"]
            en, ar = clean(pair["en"]), clean(pair["ar"])
            if not en or not ar:
                continue
            if len(en.split()) > args.max_len or len(ar.split()) > args.max_len:
                continue
            key = (en.lower(), ar)
            if key in seen:
                continue
            seen.add(key)
            rows.append((en, ar))

    rng.shuffle(rows)
    if args.max_pairs:
        rows = rows[:args.max_pairs]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for en, ar in rows:
            f.write(f"{en}\t{ar}\n")
    print(f"Wrote {len(rows)} pairs -> {out}")


if __name__ == "__main__":
    main()
