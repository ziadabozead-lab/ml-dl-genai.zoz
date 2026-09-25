"""
Download the "emotion" dataset from Hugging Face and save it in the format
train_emotion.py expects (one "label<TAB>text" per line).

    pip install datasets
    python scripts/prepare_emotion_data.py                        # ~20k tweets, 6 emotions
    python scripts/prepare_emotion_data.py --max-per-class 1000   # SMALL: 1000 per emotion (~6k rows)
    python scripts/prepare_emotion_data.py --config unsplit       # ~416k tweets (big, slow)

--max-per-class keeps the emotions balanced and makes training much faster.
Emotions: sadness, joy, love, anger, fear, surprise.
"""
import argparse
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="split", choices=["split", "unsplit"])
    ap.add_argument("--out", default=str(ROOT / "rdf" / "emotion_data.txt"))
    ap.add_argument("--max-per-class", type=int, default=0,
                    help="keep at most this many examples per emotion (0 = keep everything)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Please run:  pip install datasets")

    ds = load_dataset("dair-ai/emotion", args.config)
    names = ds["train"].features["label"].names
    by_label = defaultdict(list)
    for split in ds:                          # train + validation + test; the trainer re-splits
        for row in ds[split]:
            text = " ".join(row["text"].split())
            if text:
                by_label[names[row["label"]]].append(text)

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for label, texts in by_label.items():
            rng.shuffle(texts)
            if args.max_per_class:
                texts = texts[:args.max_per_class]
            for t in texts:
                f.write(f"{label}\t{t}\n")
            n += len(texts)
            print(f"  {label:<9} {len(texts)}")
    print(f"Wrote {n} examples -> {out}")


if __name__ == "__main__":
    main()