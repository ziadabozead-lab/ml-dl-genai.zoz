# -*- coding: utf-8 -*-
"""
Arabic nahw (grammar) tagger: learns each word's i'rab role (مبتدأ، خبر،
فاعل، مفعول به، مضاف إليه، حال، تمييز، حرف جر، فعل مضارع منصوب/مجزوم ...)
from a fully-diacritized (مُشكّل) sentence.

Same architecture as lang-model/scripts/text.py (word + character BiLSTM
tagger) - only the data and tagset are Arabic/nahw-specific.

KNOWN LIMITATION: attached one-letter prefixes (و العطف / ب و ل الجر /
ك) are not separated by whitespace in real Arabic, so a word like
"والمعلمُ" is tokenized (and must be trained) as a single surface token,
not as "و" + "المعلمُ". The bundled generator avoids this by only using
free-standing particles. Feed it real, unfiltered Arabic text and it WILL
mislabel attached-prefix words - that needs morphological segmentation
first, which this project does not attempt.

Reads rdf/nahw_data.txt (one "word<TAB>TAG" per line, blank line between
sentences - same format as lang-model, made by scripts/generate_nahw_data.py
or by hand).

Train:    python scripts/train_nahw.py
Quick:    python scripts/train_nahw.py --epochs 15
Predict:  python scripts/train_nahw.py --predict "ذهبَ الطالبُ إلى المدرسةِ."
"""
import argparse
import random
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

MAXC = 20  # max characters kept per word
PAD, UNK = 0, 1
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "rdf" / "nahw_data.txt"
DEFAULT_OUT = ROOT / "weights" / "nahw_model.pt"

torch.backends.mkldnn.enabled = False  # avoids a Windows LSTM crash on some builds


# ------------------------------- data ---------------------------------- #
def read_corpus(path, max_sents=0):
    sents, words, tags = [], [], []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if words:
                    sents.append((words, tags))
                    words, tags = [], []
                    if max_sents and len(sents) >= max_sents:
                        break
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                words.append(parts[0])
                tags.append(parts[1])
    if words:
        sents.append((words, tags))
    return sents


class Vocab:
    def __init__(self, train_sents, min_freq=1):
        wc = Counter(w for ws, _ in train_sents for w in ws)
        self.w2i = {"<pad>": PAD, "<unk>": UNK}
        for w, c in wc.items():
            if c >= min_freq:
                self.w2i[w] = len(self.w2i)
        chars = sorted({ch for ws, _ in train_sents for w in ws for ch in w})
        self.c2i = {"<pad>": PAD, "<unk>": UNK}
        for ch in chars:
            self.c2i[ch] = len(self.c2i)
        tags = sorted({t for _, ts in train_sents for t in ts})
        self.t2i = {t: i for i, t in enumerate(tags)}
        self.i2t = {i: t for t, i in self.t2i.items()}

    def word_id(self, w):
        return self.w2i.get(w, UNK)   # no .lower() - Arabic has no case

    def char_ids(self, w):
        ids = [self.c2i.get(ch, UNK) for ch in w[:MAXC]]
        return ids + [PAD] * (MAXC - len(ids))


def make_collate(vocab):
    def collate(batch):
        B = len(batch)
        T = max(len(ws) for ws, _ in batch)
        words = torch.zeros(B, T, dtype=torch.long)
        chars = torch.zeros(B, T, MAXC, dtype=torch.long)
        tags = torch.full((B, T), -100, dtype=torch.long)
        lens = torch.tensor([len(ws) for ws, _ in batch], dtype=torch.long)
        for b, (ws, ts) in enumerate(batch):
            for i, w in enumerate(ws):
                words[b, i] = vocab.word_id(w)
                chars[b, i] = torch.tensor(vocab.char_ids(w))
                if ts is not None:
                    tags[b, i] = vocab.t2i.get(ts[i], -100)
        return words, chars, tags, lens
    return collate


# ------------------------------- model --------------------------------- #
class Tagger(nn.Module):
    def __init__(self, n_words, n_chars, n_tags, emb=100, char_emb=30,
                 char_filters=50, hidden=128, dropout=0.4):
        super().__init__()
        self.word_emb = nn.Embedding(n_words, emb, padding_idx=PAD)
        self.char_emb = nn.Embedding(n_chars, char_emb, padding_idx=PAD)
        self.char_cnn = nn.Conv1d(char_emb, char_filters, kernel_size=3, padding=1)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb + char_filters, hidden, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, n_tags)

    def forward(self, words, chars, lens):
        B, T, C = chars.shape
        ce = self.char_emb(chars.view(B * T, C)).transpose(1, 2)   # [BT, E, C]
        ce = torch.relu(self.char_cnn(ce)).max(dim=2).values       # [BT, F]
        ce = ce.view(B, T, -1)
        x = self.drop(torch.cat([self.word_emb(words), ce], dim=-1))
        packed = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=T)
        return self.fc(self.drop(out))


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    per_tag = Counter()
    per_tag_total = Counter()
    with torch.no_grad():
        for words, chars, tags, lens in loader:
            words, chars, tags = words.to(device), chars.to(device), tags.to(device)
            pred = model(words, chars, lens).argmax(-1)
            mask = tags != -100
            correct += ((pred == tags) & mask).sum().item()
            total += mask.sum().item()
    return correct / max(total, 1)


# ------------------------------ predict -------------------------------- #
def tokenize(text):
    """Whitespace-based; keeps Arabic word forms (with diacritics) intact,
    and drops punctuation entirely (training data has no punctuation tag,
    so tagging it would just be noise). See the module docstring for the
    attached-prefix limitation this implies."""
    toks = []
    for raw in text.strip().split():
        word = raw.strip(".,!?؟،:؛")
        if word:
            toks.append(word)
    return toks


def predict(text, ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    vocab = ck["vocab"]
    model = Tagger(len(vocab.w2i), len(vocab.c2i), len(vocab.t2i), **ck["hparams"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    tokens = tokenize(text)
    if not tokens:
        print("No words found.")
        return
    words, chars, _, lens = make_collate(vocab)([(tokens, None)])
    with torch.no_grad():
        pred = model(words, chars, lens).argmax(-1)[0].tolist()
    for tok, p in zip(tokens, pred):
        print(f"{tok:<15} {vocab.i2t[p]}")


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    vocab = ck["vocab"]
    model = Tagger(len(vocab.w2i), len(vocab.c2i), len(vocab.t2i), **ck["hparams"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, vocab


def tag_sentence(model, vocab, text):
    tokens = tokenize(text)
    if not tokens:
        return []
    words, chars, _, lens = make_collate(vocab)([(tokens, None)])
    with torch.no_grad():
        pred = model(words, chars, lens).argmax(-1)[0].tolist()
    return list(zip(tokens, [vocab.i2t[p] for p in pred]))


# -------------------------------- main --------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--emb", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--predict", default=None, help="tag a sentence with a saved model")
    args = ap.parse_args()

    if args.predict:
        predict(args.predict, args.out)
        return

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cpu"
    torch.set_num_threads(4)

    sents = read_corpus(args.data)
    random.shuffle(sents)
    n = len(sents)
    n_val = max(1, int(n * args.val_frac))
    val, train = sents[:n_val], sents[n_val:]
    print(f"Sentences: {n} (train {len(train)}, val {len(val)}) | Device: {device}")

    vocab = Vocab(train)
    print(f"Vocab: {len(vocab.w2i)} words, {len(vocab.c2i)} chars, "
          f"{len(vocab.t2i)} tags: {list(vocab.t2i)}")
    coll = make_collate(vocab)
    tr_dl = torch.utils.data.DataLoader(train, batch_size=args.batch, shuffle=True, collate_fn=coll)
    va_dl = torch.utils.data.DataLoader(val, batch_size=64, collate_fn=coll)

    hparams = dict(emb=args.emb, hidden=args.hidden)
    model = Tagger(len(vocab.w2i), len(vocab.c2i), len(vocab.t2i), **hparams).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    best_acc, best_state = 0.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for words, chars, tags, lens in tr_dl:
            words, chars, tags = words.to(device), chars.to(device), tags.to(device)
            opt.zero_grad()
            logits = model(words, chars, lens)
            loss = loss_fn(logits.view(-1, logits.size(-1)), tags.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item()
        acc = evaluate(model, va_dl, device)
        print(f"Epoch {epoch:2d} | train loss {total / len(tr_dl):.4f} | val accuracy {acc:.4f}")
        if acc >= best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    print(f"\nBest val accuracy: {best_acc:.4f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "vocab": vocab, "hparams": hparams}, args.out)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
