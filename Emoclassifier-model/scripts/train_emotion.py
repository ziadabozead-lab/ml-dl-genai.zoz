"""
Emotion classifier - a BiLSTM that reads a sentence and predicts which emotion
it expresses (sadness, joy, love, anger, fear, surprise, ...).

Same idea as your sentiment model, but with MORE THAN TWO classes, so:
  - the output layer has one score per emotion and uses softmax + CrossEntropy
  - accuracy alone is misleading (some emotions are far more common), so we
    also report macro-F1, per-emotion F1 and a confusion matrix

Data format (rdf/emotion_data.txt):
    One line per example:  "label<TAB>text"      e.g.  "joy<TAB>i feel so happy today"
    The label set is read from the file. Get the data with:
        python scripts/prepare_emotion_data.py

Train:      python scripts/train_emotion.py
Balanced:   python scripts/train_emotion.py --balanced      (weights rare emotions more)
Try a text: python scripts/train_emotion.py --predict "i am so scared of tomorrow"
Type many:  python scripts/train_emotion.py --interactive
"""
import argparse
import random
import re
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader

torch.backends.mkldnn.enabled = False  # avoids a Windows LSTM crash on some builds

PAD, UNK = 0, 1
MAXLEN = 60
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "rdf" / "emotion_data.txt"
DEFAULT_OUT = ROOT / "weights" / "emotion_model.pt"
URL_RE = re.compile(r"https?://\S+|www\.\S+")
USER_RE = re.compile(r"@\w+")
TOKEN_RE = re.compile(r"[a-z0-9']+|[!?]")


# ------------------------------- data ---------------------------------- #
def tokenize(text):
    text = URL_RE.sub(" url ", text.lower())
    text = USER_RE.sub(" user ", text)
    text = text.replace("#", " ")
    return TOKEN_RE.findall(text)


def read_data(path):
    """Returns [(tokens, label_name)], with duplicate sentences removed."""
    rows, seen, n_dup = [], set(), 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
                continue
            toks = tokenize(parts[1])
            if not toks:
                continue
            key = tuple(toks)
            if key in seen:          # also stops the same sentence leaking into train AND test
                n_dup += 1
                continue
            seen.add(key)
            rows.append((toks, parts[0].strip().lower()))
    print(f"Read {len(rows)} examples ({n_dup} duplicates dropped)")
    return rows


def build_vocab(rows, min_freq=2):
    from collections import Counter
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
        y = torch.tensor([lab for _, lab in batch], dtype=torch.long)
        return x, lens, y
    return collate


# ------------------------------- model --------------------------------- #
class Classifier(nn.Module):
    def __init__(self, n_words, n_classes, emb=100, hidden=128, dropout=0.4):
        super().__init__()
        self.emb = nn.Embedding(n_words, emb, padding_idx=PAD)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, x, lens):
        e = self.drop(self.emb(x))
        packed = pack_padded_sequence(e, lens.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        mask = (torch.arange(x.size(1))[None, :] < lens[:, None]).unsqueeze(-1)
        out = out.masked_fill(~mask, -1e9).max(dim=1).values
        return self.fc(self.drop(out))            # [batch, n_classes] raw scores


def evaluate(model, loader, loss_fn, n_classes):
    model.eval()
    loss_sum, preds, golds, confs = 0.0, [], [], []
    with torch.no_grad():
        for x, lens, y in loader:
            logits = model(x, lens)
            loss_sum += loss_fn(logits, y).item() * len(y)
            conf, pred = torch.softmax(logits, dim=-1).max(dim=-1)
            preds += pred.tolist()
            golds += y.tolist()
            confs += conf.tolist()
    n = max(len(golds), 1)
    f1s = []
    for c in range(n_classes):
        tp = sum(1 for p, g in zip(preds, golds) if p == c and g == c)
        fp = sum(1 for p, g in zip(preds, golds) if p == c and g != c)
        fn = sum(1 for p, g in zip(preds, golds) if p != c and g == c)
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return {"loss": loss_sum / n, "acc": sum(p == g for p, g in zip(preds, golds)) / n,
            "macro_f1": sum(f1s) / n_classes, "f1s": f1s,
            "preds": preds, "golds": golds, "confs": confs}


def report(t, labels):
    n = len(labels)
    print("\nPer-emotion F1 (test set):")
    for name, f1 in zip(labels, t["f1s"]):
        print(f"  {name:<10} {f1:.3f}")
    print("\nConfusion matrix (rows = true emotion, columns = predicted):")
    print("  " + " " * 10 + "".join(f"{l[:7]:>8}" for l in labels))
    grid = [[0] * n for _ in range(n)]
    for p, g in zip(t["preds"], t["golds"]):
        grid[g][p] += 1
    for i, l in enumerate(labels):
        print(f"  {l:<10}" + "".join(f"{v:>8}" for v in grid[i]))
    print("\n  min confidence | share of examples | accuracy on those")
    pairs = [(c, p == g) for c, p, g in zip(t["confs"], t["preds"], t["golds"])]
    for thr in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        kept = [ok for c, ok in pairs if c >= thr]
        if kept:
            print(f"  {thr:>13.0%} | {len(kept) / len(pairs):>17.0%} | {sum(kept) / len(kept):>.0%}")


# ------------------------------ predict -------------------------------- #
def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = Classifier(len(ck["w2i"]), len(ck["labels"]), **ck["hparams"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck["w2i"], ck["labels"]


def predict_text(model, w2i, labels, text, top=None):
    """[(emotion, probability), ...] best first, or [] if the text has no words."""
    toks = tokenize(text)
    if not toks:
        return []
    x, lens, _ = make_collate(w2i)([(toks, 0)])
    with torch.no_grad():
        probs = torch.softmax(model(x, lens), dim=-1)[0].tolist()
    ranked = sorted(zip(labels, probs), key=lambda p: p[1], reverse=True)
    return ranked[:top] if top else ranked


def show(ranked, indent=""):
    if not ranked:
        print(f"{indent}No words found.")
        return
    print(indent + "  ".join(f"{l} {p:.0%}" for l, p in ranked[:3]))


# -------------------------------- main --------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--min-delta", type=float, default=1e-3)
    ap.add_argument("--balanced", action="store_true",
                    help="weight the loss so rare emotions count as much as common ones")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--predict", default=None)
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()

    if args.predict or args.interactive:
        model, w2i, labels = load_model(args.out)
        if args.predict:
            show(predict_text(model, w2i, labels, args.predict))
        else:
            print("Type a sentence (empty line to quit).")
            while True:
                s = input("> ").strip()
                if not s:
                    break
                show(predict_text(model, w2i, labels, s), indent="   ")
        return

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)

    rows = read_data(args.data)
    labels = sorted({l for _, l in rows})
    l2i = {l: i for i, l in enumerate(labels)}
    rows = [(t, l2i[l]) for t, l in rows]
    random.shuffle(rows)
    n = len(rows)
    n_val = n_test = int(n * 0.1)
    val, test, train = rows[:n_val], rows[n_val:n_val + n_test], rows[n_val + n_test:]
    print(f"Emotions ({len(labels)}): {', '.join(labels)}")
    print(f"Examples: {n} (train {len(train)}, val {len(val)}, test {len(test)})")
    counts = [sum(1 for _, y in train if y == c) for c in range(len(labels))]
    print("Train class counts: " + ", ".join(f"{l}={c}" for l, c in zip(labels, counts)))

    w2i = build_vocab(train)
    print(f"Vocab: {len(w2i)} words")
    coll = make_collate(w2i)
    tr = DataLoader(train, batch_size=args.batch, shuffle=True, collate_fn=coll)
    va = DataLoader(val, batch_size=256, collate_fn=coll)
    te = DataLoader(test, batch_size=256, collate_fn=coll)

    hparams = dict(emb=100, hidden=128)
    model = Classifier(len(w2i), len(labels), **hparams)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    weight = None
    if args.balanced:
        weight = torch.tensor([len(train) / (len(labels) * max(c, 1)) for c in counts])
    train_loss_fn = nn.CrossEntropyLoss(weight=weight)
    eval_loss_fn = nn.CrossEntropyLoss()      # plain loss, so epochs are comparable

    best_loss, best_epoch, best_state, wait = float("inf"), 0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, lens, y in tr:
            opt.zero_grad()
            loss = train_loss_fn(model(x, lens), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item()
        m = evaluate(model, va, eval_loss_fn, len(labels))
        improved = m["loss"] < best_loss - args.min_delta
        if improved:
            best_loss, best_epoch, wait = m["loss"], epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        print(f"Epoch {epoch:2d} | train loss {total / len(tr):.4f} | val loss {m['loss']:.4f} | "
              f"val acc {m['acc']:.4f} | val macro-F1 {m['macro_f1']:.4f}{'  *' if improved else ''}")
        if wait >= args.patience:
            print(f"Early stopping: val loss has not improved for {args.patience} epochs.")
            break
    print(f"Best epoch: {best_epoch}")

    model.load_state_dict(best_state)
    t = evaluate(model, te, eval_loss_fn, len(labels))
    majority = max(counts) / max(len(train), 1)
    print(f"\nTest accuracy: {t['acc']:.4f} | macro-F1: {t['macro_f1']:.4f} "
          f"(always-guess-most-common baseline accuracy: {majority:.4f})")
    report(t, labels)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "w2i": w2i, "labels": labels, "hparams": hparams}, args.out)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
