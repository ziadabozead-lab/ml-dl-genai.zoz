"""
Sentiment classifier (positive / negative) - small enough to train on a laptop CPU.

Project layout this script expects (paths are found automatically, run from anywhere):
    no/
      scripts/train_sentiment.py     <- this file
      rdf/sentiment_data.txt         <- data (lines:  pos<TAB>text  or  neg<TAB>text)
      weights/sentiment_model.pt     <- created by training

Run from the "no" folder:
    Train:        python scripts/train_sentiment.py
    Try a text:   python scripts/train_sentiment.py --predict "I love this so much"
    Type many:    python scripts/train_sentiment.py --interactive
    Own data:     python scripts/train_sentiment.py --data path/to/yourfile.txt
"""
import argparse
import random
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader

torch.backends.mkldnn.enabled = False  # avoids a Windows LSTM crash on some builds

PAD, UNK = 0, 1
MAXLEN = 60
ROOT = Path(__file__).resolve().parent.parent          # the "no" folder
DEFAULT_DATA = ROOT / "rdf" / "sentiment_data.txt"
DEFAULT_OUT = ROOT / "weights" / "sentiment_model.pt"
URL_RE = re.compile(r"https?://\S+|www\.\S+")
USER_RE = re.compile(r"@\w+")
TOKEN_RE = re.compile(r"[a-z0-9']+|[!?]")
# A lone "(" / ")" run (what is left of ":(" / ":)" once the colon is stripped).
# The tokenizer throws these away, so in such rows the label came from a symbol
# the model never sees and the remaining words are essentially random.
EMOTICON_RE = re.compile(r"(?:^|\s)[()]+(?=\s|$)")


# ------------------------------- data ---------------------------------- #
def tokenize(text):
    text = URL_RE.sub(" url ", text.lower())
    text = USER_RE.sub(" user ", text)
    text = text.replace("#", " ")
    return TOKEN_RE.findall(text)


def read_data(path, drop_emoticon=True):
    rows, seen = [], set()
    n_emo = n_dup = 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if not (len(parts) == 2 and parts[0] in ("pos", "neg") and parts[1].strip()):
                continue
            if drop_emoticon and EMOTICON_RE.search(parts[1]):
                n_emo += 1
                continue
            toks = tokenize(parts[1])
            if not toks:
                continue
            # Near-duplicate key: spam like "... x26" / "... x39" differs only by digits.
            key = tuple(t for t in (re.sub(r"\d+", "", t) for t in toks) if t)
            if key in seen:
                n_dup += 1
                continue
            seen.add(key)
            rows.append((toks, 1 if parts[0] == "pos" else 0))
    print(f"Dropped {n_emo} emoticon-labelled rows and {n_dup} near-duplicates; kept {len(rows)}")
    return rows


def build_vocab(rows, min_freq=2):
    counts = Counter(t for toks, _ in rows for t in toks)
    w2i = {"<pad>": PAD, "<unk>": UNK}
    for w, c in counts.items():
        if c >= min_freq:
            w2i[w] = len(w2i)
    return w2i


def make_collate(w2i):
    def collate(batch):
        seqs = [torch.tensor([w2i.get(t, UNK) for t in toks[:MAXLEN]]) for toks, _ in batch]
        lens = torch.tensor([len(s) for s in seqs])
        x = pad_sequence(seqs, batch_first=True, padding_value=PAD)
        y = torch.tensor([lab for _, lab in batch], dtype=torch.float32)
        return x, lens, y
    return collate


# ------------------------------- model --------------------------------- #
class Classifier(nn.Module):
    def __init__(self, n_words, emb=100, hidden=64, dropout=0.4):
        super().__init__()
        self.emb = nn.Embedding(n_words, emb, padding_idx=PAD)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, x, lens):
        e = self.drop(self.emb(x))
        packed = pack_padded_sequence(e, lens.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        mask = (torch.arange(x.size(1))[None, :] < lens[:, None]).unsqueeze(-1)
        out = out.masked_fill(~mask, -1e9).max(dim=1).values
        return self.fc(self.drop(out)).squeeze(-1)


def evaluate(model, loader, loss_fn):
    """Returns mean loss, accuracy, and (confidence, was_correct) for every example."""
    model.eval()
    loss_sum, correct, total, conf_correct = 0.0, 0, 0, []
    with torch.no_grad():
        for x, lens, y in loader:
            logits = model(x, lens)
            loss_sum += loss_fn(logits, y).item() * len(y)
            p = torch.sigmoid(logits)
            hit = ((p > 0.5).float() == y)
            correct += hit.sum().item()
            total += len(y)
            conf_correct += list(zip(torch.maximum(p, 1 - p).tolist(), hit.tolist()))
    total = max(total, 1)
    return {"loss": loss_sum / total, "acc": correct / total, "conf": conf_correct}


def confidence_table(conf_correct):
    """Accuracy vs. confidence: use this to pick UNSURE_BELOW in sentiment_gui.py."""
    print("  min confidence | share of examples | accuracy on those")
    for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
        kept = [ok for c, ok in conf_correct if c >= thr]
        if kept:
            print(f"  {thr:>13.0%} | {len(kept) / len(conf_correct):>17.0%} | {sum(kept) / len(kept):>.0%}")


# ------------------------------ predict -------------------------------- #
def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = Classifier(len(ck["w2i"]), **ck["hparams"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck["w2i"]


def predict_text(model, w2i, text):
    toks = tokenize(text)
    if not toks:
        return None, 0.0
    x, lens, _ = make_collate(w2i)([(toks, 0)])
    with torch.no_grad():
        p = torch.sigmoid(model(x, lens)).item()
    return ("POSITIVE" if p > 0.5 else "NEGATIVE"), max(p, 1 - p)


# -------------------------------- main --------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--min-delta", type=float, default=1e-3,
                    help="smallest drop in val loss that counts as improvement")
    ap.add_argument("--keep-emoticon-rows", action="store_true",
                    help="keep rows whose label came only from a stripped :( / :) emoticon")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--predict", default=None)
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()

    if args.predict or args.interactive:
        model, w2i = load_model(args.out)
        if args.predict:
            label, conf = predict_text(model, w2i, args.predict)
            print(f"{label} ({conf:.0%})" if label else "No words found.")
        else:
            print("Type a sentence (empty line to quit).")
            while True:
                s = input("> ").strip()
                if not s:
                    break
                label, conf = predict_text(model, w2i, s)
                print(f"   {label} ({conf:.0%})" if label else "   No words found.")
        return

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)

    rows = read_data(args.data, drop_emoticon=not args.keep_emoticon_rows)
    random.shuffle(rows)
    n = len(rows)
    n_val = n_test = int(n * 0.1)
    val, test, train = rows[:n_val], rows[n_val:n_val + n_test], rows[n_val + n_test:]
    print(f"Examples: {n} (train {len(train)}, val {len(val)}, test {len(test)})")

    w2i = build_vocab(train)
    print(f"Vocab: {len(w2i)} words")
    coll = make_collate(w2i)
    tr = DataLoader(train, batch_size=args.batch, shuffle=True, collate_fn=coll)
    va = DataLoader(val, batch_size=256, collate_fn=coll)
    te = DataLoader(test, batch_size=256, collate_fn=coll)

    hparams = dict(emb=100, hidden=64)
    model = Classifier(len(w2i), **hparams)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    # Early stopping on validation LOSS (smooth) instead of accuracy (coarse, noisy).
    best_loss, best_epoch, best_state, wait = float("inf"), 0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, lens, y in tr:
            opt.zero_grad()
            loss = loss_fn(model(x, lens), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item()
        m = evaluate(model, va, loss_fn)
        improved = m["loss"] < best_loss - args.min_delta
        if improved:
            best_loss, best_epoch, wait = m["loss"], epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        print(f"Epoch {epoch:2d} | train loss {total / len(tr):.4f} | val loss {m['loss']:.4f} | "
              f"val acc {m['acc']:.4f}{'  *' if improved else ''}")
        if wait >= args.patience:
            print(f"Early stopping: val loss has not improved for {args.patience} epochs.")
            break
    print(f"Best epoch: {best_epoch}")

    model.load_state_dict(best_state)
    t = evaluate(model, te, loss_fn)
    pos_rate = sum(lab for _, lab in test) / max(len(test), 1)
    print(f"\nTest accuracy: {t['acc']:.4f}  (always-guess-majority baseline: {max(pos_rate, 1 - pos_rate):.4f})")
    confidence_table(t["conf"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "w2i": w2i, "hparams": hparams}, args.out)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()