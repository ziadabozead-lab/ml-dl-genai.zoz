"""
Language identifier - a small character-level BiGRU that reads text one
character at a time and predicts which language it is written in.
 
Because it works on characters (not words), it handles any script - Latin,
Arabic, Cyrillic, ... - and can guess from just a few letters.
 
Data format (rdf/langid_data.txt):
    One line per example:  "LanguageName<TAB>sentence text"
    e.g.  "French<TAB>Il fait très beau aujourd'hui."
    Add as many languages as you like - the label set is read from the file.
 
Train:    python scripts/train_langid.py
          python scripts/train_langid.py --data rdf/langid_data.txt --epochs 50 --patience 50
Predict:  used internally by chatbot/langid_gui.py
 
How it trains: sentences are split into train/val FIRST. During training each
sentence is also shown as random short word-windows (1-8 words), so the model
learns to identify short inputs like "bonjour" and not only full sentences.
"""
import argparse
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
 
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
 
torch.backends.mkldnn.enabled = False  # avoids a Windows LSTM/GRU crash on some builds
 
PAD, UNK = 0, 1
MAX_LEN = 120                # characters kept per example
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "rdf" / "langid_data.txt"
DEFAULT_OUT = ROOT / "weights" / "langid_model.pt"
 
STRIP_PUNCT_RE = re.compile(r"[^\w\s']", re.UNICODE)
 
 
# ------------------------------- text utils ------------------------------- #
def clean(text):
    return re.sub(r"\s+", " ", text.lower().strip())
 
 
def encode(text, c2i):
    ids = [c2i.get(ch, UNK) for ch in clean(text)[:MAX_LEN]]
    return ids or [UNK]
 
 
def make_batch(texts, c2i):
    seqs = [encode(t, c2i) for t in texts]
    lens = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    x = torch.zeros(len(seqs), int(lens.max()), dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
    return x, lens
 
 
def random_window(text, rng):
    """Random slice of 1-8 consecutive words, sometimes with punctuation removed."""
    words = text.split()
    if len(words) <= 1:
        return text
    size = rng.randint(1, min(8, len(words)))
    start = rng.randint(0, len(words) - size)
    piece = " ".join(words[start:start + size])
    if rng.random() < 0.4:
        piece = STRIP_PUNCT_RE.sub("", piece).strip() or piece
    return piece
 
 
# --------------------------------- data ---------------------------------- #
def read_examples(path):
    ex = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                ex.append((parts[0].strip(), parts[1].strip()))
    return ex
 
 
def build_char_vocab(texts):
    counts = Counter()
    for t in texts:
        counts.update(clean(t))
    c2i = {"<pad>": PAD, "<unk>": UNK}
    for ch in counts:
        c2i[ch] = len(c2i)
    return c2i
 
 
# --------------------------------- model --------------------------------- #
class LangID(nn.Module):
    def __init__(self, n_chars, n_langs, emb=48, hidden=96, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(n_chars, emb, padding_idx=PAD)
        self.rnn = nn.GRU(emb, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, n_langs)
 
    def forward(self, x, lens):
        e = self.drop(self.emb(x))
        packed = pack_padded_sequence(e, lens.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        mask = (torch.arange(x.size(1))[None, :] < lens[:, None]).unsqueeze(-1)
        pooled = out.masked_fill(~mask, -1e9).max(dim=1).values
        return self.fc(self.drop(pooled))
 
 
# ----------------------------- predict (trained) ----------------------------- #
def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = LangID(len(ck["c2i"]), len(ck["langs"]), **ck["hparams"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck["c2i"], ck["langs"]
 
 
def predict(model, c2i, langs, text, top=3):
    """Return [(language, probability), ...] best first. Long text is scored in
    chunks of MAX_LEN characters and the probabilities are averaged."""
    text = clean(text)
    if not text:
        return []
    chunks = [text[i:i + MAX_LEN] for i in range(0, len(text), MAX_LEN)]
    if len(chunks) > 1 and len(chunks[-1]) < 15:
        chunks.pop()
    x, lens = make_batch(chunks, c2i)
    with torch.no_grad():
        probs = torch.softmax(model(x, lens), dim=-1).mean(dim=0).tolist()
    ranked = sorted(zip(langs, probs), key=lambda p: p[1], reverse=True)
    return ranked[:top]
 
 
# --------------------------------- evaluate --------------------------------- #
def evaluate(model, examples, c2i, l2i, loss_fn, bs=128):
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    per_lang = defaultdict(lambda: [0, 0])
    with torch.no_grad():
        for i in range(0, len(examples), bs):
            chunk = examples[i:i + bs]
            x, lens = make_batch([t for _, t in chunk], c2i)
            y = torch.tensor([l2i[l] for l, _ in chunk])
            logits = model(x, lens)
            loss_sum += loss_fn(logits, y).item() * len(chunk)
            pred = logits.argmax(-1)
            for (lang, _), p, t in zip(chunk, pred.tolist(), y.tolist()):
                per_lang[lang][1] += 1
                per_lang[lang][0] += int(p == t)
            correct += (pred == y).sum().item()
            total += len(chunk)
    total = max(total, 1)
    return {"loss": loss_sum / total, "acc": correct / total, "per_lang": dict(per_lang)}
 
 
# ---------------------------------- train ---------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--patience", type=int, default=8,
                    help="epochs without improvement before stopping")
    ap.add_argument("--min-delta", type=float, default=1e-3)
    ap.add_argument("--monitor", choices=["loss", "acc"], default="loss")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--windows", type=int, default=3,
                    help="random short word-windows generated per sentence each epoch")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
 
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
 
    examples = read_examples(args.data)
    if not examples:
        print(f"No examples found in {args.data}")
        return
    langs = sorted({l for l, _ in examples})
    l2i = {l: i for i, l in enumerate(langs)}
    print(f"Examples: {len(examples)} | languages ({len(langs)}): {', '.join(langs)}")
 
    # stratified split by sentence, so every language appears in validation
    by_lang = defaultdict(list)
    for e in examples:
        by_lang[e[0]].append(e)
    train, val_full = [], []
    for lang, items in by_lang.items():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * args.val_frac))
        val_full += items[:n_val]
        train += items[n_val:]
    # fixed short fragments from validation sentences (harder test)
    frag_rng = random.Random(123)
    val_short = [(l, random_window(t, frag_rng)) for l, t in val_full for _ in range(4)]
    print(f"Train sentences: {len(train)} | val sentences: {len(val_full)} | val fragments: {len(val_short)}")
 
    c2i = build_char_vocab([t for _, t in train])
    print(f"Character vocab: {len(c2i)}")
 
    hparams = dict(emb=48, hidden=96)
    model = LangID(len(c2i), len(langs), **hparams)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
 
    def score_of(m):
        return -m["loss"] if args.monitor == "loss" else m["acc"]
 
    best_score, best_epoch, best_state, wait = float("-inf"), 0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        # full sentences + random short windows, reshuffled every epoch
        epoch_data = list(train)
        for l, t in train:
            for _ in range(args.windows):
                epoch_data.append((l, random_window(t, rng)))
        rng.shuffle(epoch_data)
 
        total, steps = 0.0, 0
        for i in range(0, len(epoch_data), args.batch):
            chunk = epoch_data[i:i + args.batch]
            x, lens = make_batch([t for _, t in chunk], c2i)
            y = torch.tensor([l2i[l] for l, _ in chunk])
            opt.zero_grad()
            loss = loss_fn(model(x, lens), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item()
            steps += 1
 
        m_full = evaluate(model, val_full, c2i, l2i, loss_fn)
        m_short = evaluate(model, val_short, c2i, l2i, loss_fn)
        # monitor the average of both, so short inputs count too
        m = {"loss": (m_full["loss"] + m_short["loss"]) / 2, "acc": (m_full["acc"] + m_short["acc"]) / 2}
        score = score_of(m)
        improved = score > best_score + args.min_delta
        if improved:
            best_score, best_epoch, wait = score, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
 
        print(f"Epoch {epoch:2d} | train loss {total / max(steps, 1):.4f} | "
              f"val loss {m['loss']:.4f} | sentences acc {m_full['acc']:.3f} | "
              f"short-fragment acc {m_short['acc']:.3f}{'  *' if improved else ''}")
 
        if wait >= args.patience:
            print(f"Early stopping: no improvement in {args.monitor} for {args.patience} epochs.")
            break
 
    print(f"Best epoch: {best_epoch} (monitor={args.monitor})")
    if best_state is not None:
        model.load_state_dict(best_state)
 
    final = evaluate(model, val_short, c2i, l2i, loss_fn)
    print("Per-language accuracy on short fragments:")
    for lang in langs:
        ok, n = final["per_lang"].get(lang, [0, 0])
        print(f"  {lang:<12} {ok}/{n}  ({ok / max(n, 1):.0%})")
 
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "c2i": c2i, "langs": langs, "hparams": hparams}, args.out)
    print(f"Saved -> {args.out}")
 
 
if __name__ == "__main__":
    main()
 