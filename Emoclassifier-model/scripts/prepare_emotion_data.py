"""
Download the "emotion" dataset from Hugging Face and save it in the format
train_emotion.py expects (one "label<TAB>text" per line).

    pip install datasets
    python scripts/prepare_emotion_data.py                     # ~20k tweets, 6 emotions (clean, good start)
    python scripts/prepare_emotion_data.py --config unsplit    # ~416k tweets (bigger, noisier, slower)

Emotions: sadness, joy, love, anger, fear, surprise.
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="split", choices=["split", "unsplit"])
    ap.add_argument("--out", default=str(ROOT / "rdf" / "emotion_data.txt"))
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Please run:  pip install datasets")

    ds = load_dataset("dair-ai/emotion", args.config)
    names = ds["train"].features["label"].names
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for split in ds:                      # train + validation + test; the trainer re-splits
            for row in ds[split]:
                text = " ".join(row["text"].split())
                if text:
                    f.write(f"{names[row['label']]}\t{text}\n")
                    n += 1
    print(f"Wrote {n} examples ({', '.join(names)}) -> {out}")


if __name__ == "__main__":
    main()
