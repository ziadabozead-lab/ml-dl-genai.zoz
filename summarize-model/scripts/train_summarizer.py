"""
Extractive summarizer - trains a small BiLSTM model to learn which sentences
belong in a summary, using labeled examples instead of guessing from word
frequency alone.

Data format (rdf/summary_data.txt):
    One line per sentence:  "1<TAB>sentence text"   (1 = keep in summary)
                             "0<TAB>sentence text"   (0 = skip)
    A blank line separates one document/article from the next.

Train:    python scripts/train_summarizer.py
Predict:  used internally by chatbot/summarizer_gui.py

Before training finishes for the first time (or if no weights file exists
yet), summarize_freq() below gives a reasonable fallback based on plain word
frequency, so the GUI still works from the very first run.
"""
import argparse
import random
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

torch.backends.mkldnn.enabled = False  # avoids a Windows LSTM crash on some builds

PAD, UNK = 0, 1
MAX_SENT_LEN = 40           # words kept per sentence
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "rdf" / "summary_data.txt"
DEFAULT_OUT = ROOT / "weights" / "summarizer_model.pt"

WORD_RE = re.compile(r"[a-z0-9']+")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "once", "here", "there", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "s", "t", "can",
    "will", "just", "don", "should", "now", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "of", "as", "it", "its", "this", "that", "these",
    "those", "i", "you", "he", "she", "we", "they", "them", "his", "her",
    "their", "our", "your", "my", "me", "him", "us", "am", "what", "which",
    "who", "whom",
}


# ---------------------------- shared text utils ---------------------------- #
def split_sentences(text):
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    return [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]


def tokenize(sentence):
    return WORD_RE.findall(sentence.lower())


def score_words(sentences):
    """Word -> normalized importance (0..1), ignoring stopwords."""
    counts = Counter()
    for s in sentences:
        for w in tokenize(s):
            if w not in STOPWORDS and len(w) > 1:
                counts[w] += 1
    if not counts:
        return {}
    top = max(counts.values())
    return {w: c / top for w, c in counts.items()}


# ---------------------------- frequency fallback ---------------------------- #
def summarize_freq(text, n_sentences=3, lead_bonus=0.15):
    """No training required - used before a model has been trained."""
    sentences = split_sentences(text)
    if not sentences:
        return [], []
    word_scores = score_words(sentences)
    n = len(sentences)

    raw_scores = []
    for i, s in enumerate(sentences):
        words = tokenize(s)
        if not words:
            raw_scores.append(0.0)
            continue
        base = sum(word_scores.get(w, 0.0) for w in words) / len(words)
        position_boost = lead_bonus * max(0.0, 1 - i / max(n - 1, 1))
        raw_scores.append(base + position_boost)

    n_sentences = max(1, min(n_sentences, n))
    ranked_idx = sorted(range(n), key=lambda i: raw_scores[i], reverse=True)[:n_sentences]
    chosen = set(ranked_idx)
    summary = [sentences[i] for i in sorted(chosen)]
    scored_all = [(sentences[i], raw_scores[i], i in chosen) for i in range(n)]
    return summary, scored_all


# -------------------------------- data (training) -------------------------------- #
def read_docs(path, max_docs=0):
    docs, sents, labels = [], [], []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if sents:
                    docs.append((sents, labels))
                    sents, labels = [], []
                    if max_docs and len(docs) >= max_docs:
                        break
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2 and parts[0] in ("0", "1"):
                labels.append(int(parts[0]))
                sents.append(parts[1])
    if sents:
        docs.append((sents, labels))
    return docs


def build_vocab(docs, min_freq=2):
    counts = Counter()
    for sents, _ in docs:
        for s in sents:
            counts.update(tokenize(s))
    w2i = {"<pad>": PAD, "<unk>": UNK}
    for w, c in counts.items():
        if c >= min_freq:
            w2i[w] = len(w2i)
    return w2i


def encode_doc(sents, w2i):
    """Turn one document's sentences into model-ready tensors."""
    tokenized = [tokenize(s)[:MAX_SENT_LEN] for s in sents]
    lens = [max(len(t), 1) for t in tokenized]
    maxw = max(lens)
    n = len(sents)
    x = torch.zeros(n, maxw, dtype=torch.long)
    for i, toks in enumerate(tokenized):
        for j, w in enumerate(toks):
            x[i, j] = w2i.get(w, UNK)
    lens_t = torch.tensor(lens, dtype=torch.long)

    freq_scores = score_words(sents)
    feats = torch.zeros(n, 3)
    for i, toks in enumerate(tokenized):
        pos_ratio = i / (n - 1) if n > 1 else 0.0
        length_ratio = min(len(toks) / MAX_SENT_LEN, 1.0)
        fscore = sum(freq_scores.get(w, 0.0) for w in toks) / len(toks) if toks else 0.0
        feats[i, 0] = pos_ratio
        feats[i, 1] = length_ratio
        feats[i, 2] = fscore
    return x, lens_t, feats


# ------------------------------- model --------------------------------- #
class SentenceScorer(nn.Module):
    def __init__(self, n_words, emb=80, hidden=64, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(n_words, emb, padding_idx=PAD)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2 + 3, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x, lens, feats):
        e = self.drop(self.emb(x))
        packed = pack_padded_sequence(e, lens.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        mask = (torch.arange(x.size(1))[None, :] < lens[:, None]).unsqueeze(-1)
        pooled = out.masked_fill(~mask, -1e9).max(dim=1).values   # [n_sents, hidden*2]
        combined = torch.cat([pooled, feats], dim=-1)
        return self.fc(combined).squeeze(-1)   # [n_sents] logits


def evaluate(model, docs, w2i, loss_fn):
    """Validation metrics.

    loss     - mean BCE per sentence (smooth, best signal for early stopping)
    f1       - F1 on the positive class at threshold 0.5
    topk     - of the k sentences the model ranks highest (k = number of gold
               sentences in that doc), the fraction that are gold. This is what
               the GUI actually does, so it's the number you care about.
    baseline - accuracy of predicting "0" for everything (acc must beat this)
    """
    model.eval()
    loss_sum = 0.0
    total = correct = positives = 0
    tp = fp = fn = 0
    topk_hit = topk_total = 0
    with torch.no_grad():
        for sents, labels in docs:
            if not sents:
                continue
            x, lens, feats = encode_doc(sents, w2i)
            logits = model(x, lens, feats)
            y = torch.tensor(labels, dtype=torch.float32)
            n = len(y)

            loss_sum += loss_fn(logits, y).item() * n
            pred = (torch.sigmoid(logits) > 0.5).float()
            correct += (pred == y).sum().item()
            tp += ((pred == 1) & (y == 1)).sum().item()
            fp += ((pred == 1) & (y == 0)).sum().item()
            fn += ((pred == 0) & (y == 1)).sum().item()
            total += n
            positives += int(y.sum().item())

            k = int(y.sum().item())
            if k > 0:
                top = torch.topk(logits, min(k, n)).indices
                topk_hit += y[top].sum().item()
                topk_total += k

    total = max(total, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return {
        "loss": loss_sum / total,
        "acc": correct / total,
        "f1": f1,
        "topk": topk_hit / max(topk_total, 1),
        "baseline": 1 - positives / total,
    }


# ------------------------------ predict (trained model) -------------------------------- #
def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = SentenceScorer(len(ck["w2i"]), **ck["hparams"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck["w2i"]


def score_document(model, w2i, text, n_sentences=3):
    """Same return shape as summarize_freq(), but using the trained model."""
    sentences = split_sentences(text)
    if not sentences:
        return [], []
    x, lens, feats = encode_doc(sentences, w2i)
    with torch.no_grad():
        probs = torch.sigmoid(model(x, lens, feats)).tolist()

    n = len(sentences)
    n_sentences = max(1, min(n_sentences, n))
    ranked_idx = sorted(range(n), key=lambda i: probs[i], reverse=True)[:n_sentences]
    chosen = set(ranked_idx)
    summary = [sentences[i] for i in sorted(chosen)]
    scored_all = [(sentences[i], probs[i], i in chosen) for i in range(n)]
    return summary, scored_all


# -------------------------------- train (CLI) -------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--patience", type=int, default=6,
                    help="epochs without improvement before stopping")
    ap.add_argument("--min-delta", type=float, default=1e-3,
                    help="smallest change in the monitored metric that counts as improvement")
    ap.add_argument("--monitor", choices=["loss", "topk", "f1"], default="loss",
                    help="metric used to pick the checkpoint and stop early")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)

    docs = read_docs(args.data)
    if len(docs) < 10:
        print(f"Only {len(docs)} labeled documents found in {args.data}.")
        print("You need at least ~50 for this to learn anything useful - "
              "see the format guide at the top of this file.")
        if len(docs) == 0:
            return

    random.shuffle(docs)
    n = len(docs)
    n_val = max(1, int(n * args.val_frac))
    val, train = docs[:n_val], docs[n_val:]
    print(f"Documents: {n} (train {len(train)}, val {len(val)})")

    w2i = build_vocab(train)
    print(f"Vocab: {len(w2i)} words")

    hparams = dict(emb=80, hidden=64)
    model = SentenceScorer(len(w2i), **hparams)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    # Early stopping: lower loss is better, higher topk/f1 is better. Flip the
    # sign for loss so "bigger score = better" holds for every option.
    def score_of(m):
        return -m["loss"] if args.monitor == "loss" else m[args.monitor]

    best_score, best_epoch, best_state, wait = float("-inf"), 0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train)
        total, steps = 0.0, 0
        for sents, labels in train:
            if not sents:
                continue
            x, lens, feats = encode_doc(sents, w2i)
            y = torch.tensor(labels, dtype=torch.float32)
            opt.zero_grad()
            loss = loss_fn(model(x, lens, feats), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item()
            steps += 1

        m = evaluate(model, val, w2i, loss_fn)
        score = score_of(m)
        improved = score > best_score + args.min_delta
        if improved:
            best_score, best_epoch, wait = score, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1

        print(f"Epoch {epoch:2d} | train loss {total / max(steps, 1):.4f} | "
              f"val loss {m['loss']:.4f} | top-k {m['topk']:.3f} | F1 {m['f1']:.3f} | "
              f"acc {m['acc']:.3f} (all-zeros {m['baseline']:.3f})"
              f"{'  *' if improved else ''}")

        if wait >= args.patience:
            print(f"Early stopping: no improvement in {args.monitor} for {args.patience} epochs.")
            break

    print(f"Best epoch: {best_epoch} (monitor={args.monitor})")

    if best_state is not None:
        model.load_state_dict(best_state)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "w2i": w2i, "hparams": hparams}, args.out)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()