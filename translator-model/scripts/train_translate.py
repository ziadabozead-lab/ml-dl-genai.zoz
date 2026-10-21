"""
English <-> Arabic phrase translator - a word-level sequence-to-sequence model
with attention, same idea as the spelling corrector but with two separate
vocabularies (source language, target language) instead of one alphabet.

Data format (rdf/translate_data.txt):
    One line per sentence pair:  "english sentence<TAB>arabic sentence"
    Get data with:  python scripts/prepare_translate_data.py

Train (English -> Arabic):   python scripts/train_translate.py
Train (Arabic -> English):   python scripts/train_translate.py --direction ar2en
Quick test run:               python scripts/train_translate.py --max-pairs 5000 --epochs 8
Try a sentence:                python scripts/train_translate.py --predict "how are you today"
Type many:                     python scripts/train_translate.py --interactive

This is a PHRASEBOOK-scale translator, not a production one: it does well on
short, common sentences similar to its training data, and gets shaky on long
or unusual ones. That's normal for a model this size trained on a laptop CPU.
"""
import argparse
import random
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

torch.backends.mkldnn.enabled = False  # avoids a Windows GRU crash on some builds

PAD, UNK, SOS, EOS = 0, 1, 2, 3
SPECIALS = ["<pad>", "<unk>", "<sos>", "<eos>"]
MAXLEN = 30                                    # words kept per sentence (source and target)
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "rdf" / "translate_data.txt"
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)   # \w matches Arabic letters too


# ------------------------------- text utils ------------------------------- #
def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def read_pairs(path):
    rows, seen, n_dup = [], set(), 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
                continue
            en, ar = tokenize(parts[0]), tokenize(parts[1])
            if not en or not ar or len(en) > MAXLEN or len(ar) > MAXLEN:
                continue
            key = (tuple(en), tuple(ar))
            if key in seen:
                n_dup += 1
                continue
            seen.add(key)
            rows.append((en, ar))
    print(f"Read {len(rows)} pairs ({n_dup} duplicates dropped)")
    return rows


def build_vocab(token_lists, min_freq=2):
    counts = Counter(t for toks in token_lists for t in toks)
    w2i = {s: i for i, s in enumerate(SPECIALS)}
    for w, c in counts.items():
        if c >= min_freq:
            w2i[w] = len(w2i)
    return w2i


def encode(toks, w2i):
    return [w2i.get(t, UNK) for t in toks]


def pad_batch(seqs):
    lens = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    x = torch.zeros(len(seqs), int(lens.max()), dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
    return x, lens


def make_batch(pairs, src_w2i, tgt_w2i):
    """pairs = [(src_toks, tgt_toks)] -> source ids/lens, target ids (with SOS/EOS)."""
    src, lens = pad_batch([encode(s, src_w2i) for s, _ in pairs])
    tgt, _ = pad_batch([[SOS] + encode(t, tgt_w2i) + [EOS] for _, t in pairs])
    return src, lens, tgt


# --------------------------------- model --------------------------------- #
class Seq2Seq(nn.Module):
    """Bidirectional GRU encoder + attention decoder. Source and target use
    separate embeddings/vocabularies (translation, not correction)."""

    def __init__(self, n_src, n_tgt, emb=128, enc_hidden=256, dec_hidden=256, dropout=0.3):
        super().__init__()
        H2 = enc_hidden * 2
        self.src_emb = nn.Embedding(n_src, emb, padding_idx=PAD)
        self.enc = nn.GRU(emb, enc_hidden, batch_first=True, bidirectional=True)
        self.bridge = nn.Linear(H2, dec_hidden)
        self.tgt_emb = nn.Embedding(n_tgt, emb, padding_idx=PAD)
        self.dec = nn.GRUCell(emb + H2, dec_hidden)
        self.attn_q = nn.Linear(dec_hidden, H2, bias=False)
        self.out = nn.Linear(dec_hidden + H2, n_tgt)
        self.drop = nn.Dropout(dropout)

    def encode(self, src, lens):
        e = self.drop(self.src_emb(src))
        packed = pack_padded_sequence(e, lens.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.enc(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=src.size(1))
        mask = src != PAD
        m = mask.unsqueeze(-1).float()
        mean = (out * m).sum(1) / m.sum(1).clamp(min=1)
        return out, torch.tanh(self.bridge(mean)), mask

    def step(self, prev, h, enc_out, mask):
        q = self.attn_q(h)
        scores = (enc_out * q.unsqueeze(1)).sum(-1).masked_fill(~mask, -1e9)
        ctx = (torch.softmax(scores, dim=-1).unsqueeze(-1) * enc_out).sum(1)
        h = self.dec(torch.cat([self.drop(self.tgt_emb(prev)), ctx], dim=-1), h)
        return self.out(torch.cat([self.drop(h), ctx], dim=-1)), h

    def forward(self, src, lens, tgt_in):
        enc_out, h, mask = self.encode(src, lens)
        logits = []
        for t in range(tgt_in.size(1)):        # teacher forcing
            lg, h = self.step(tgt_in[:, t], h, enc_out, mask)
            logits.append(lg)
        return torch.stack(logits, dim=1)

    @torch.no_grad()
    def decode(self, src, lens, max_len=MAXLEN + 5):
        enc_out, h, mask = self.encode(src, lens)
        tok = torch.full((src.size(0),), SOS, dtype=torch.long)
        done = torch.zeros(src.size(0), dtype=torch.bool)
        outs = []
        for _ in range(max_len):
            lg, h = self.step(tok, h, enc_out, mask)
            lg[:, [PAD, UNK, SOS]] = -1e9
            tok = lg.argmax(-1)
            outs.append(tok)
            done |= tok == EOS
            if done.all():
                break
        result = []
        for row in torch.stack(outs, dim=1).tolist():
            ids = []
            for i in row:
                if i == EOS:
                    break
                ids.append(i)
            result.append(ids)
        return result


# ------------------------------- inference ------------------------------- #
def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = Seq2Seq(len(ck["src_w2i"]), len(ck["tgt_w2i"]), **ck["hparams"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    i2w = {i: w for w, i in ck["tgt_w2i"].items()}
    return model, ck["src_w2i"], i2w, ck["direction"]


def translate(model, src_w2i, tgt_i2w, text):
    toks = tokenize(text)[:MAXLEN]
    if not toks:
        return ""
    src, lens = pad_batch([encode(toks, src_w2i)])
    ids = model.decode(src, lens)[0]
    words = [tgt_i2w.get(i, "?") for i in ids]
    # Arabic has no spaces around most punctuation the way English does;
    # this is a light touch-up, not real detokenization rules.
    out, sep = [], ""
    for w in words:
        out.append(sep + w if not (len(w) == 1 and not w.isalnum()) else w)
        sep = " "
    return " ".join(words)


# -------------------------------- evaluate ------------------------------- #
def evaluate(model, pairs, src_w2i, tgt_w2i, loss_fn, bs=64):
    """Returns mean loss (teacher-forced) and exact-match sentence accuracy (greedy decode)."""
    model.eval()
    loss_sum, n_tok, correct = 0.0, 0, 0
    with torch.no_grad():
        for i in range(0, len(pairs), bs):
            chunk = pairs[i:i + bs]
            src, lens, tgt = make_batch(chunk, src_w2i, tgt_w2i)
            logits = model(src, lens, tgt[:, :-1])
            gold = tgt[:, 1:]
            k = (gold != PAD).sum().item()
            loss_sum += loss_fn(logits.reshape(-1, logits.size(-1)), gold.reshape(-1)).item() * k
            n_tok += k
            for ids, (_, t) in zip(model.decode(src, lens), chunk):
                correct += int(ids == encode(t, tgt_w2i))
    return {"loss": loss_sum / max(n_tok, 1), "acc": correct / max(len(pairs), 1)}


# ---------------------------------- main --------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--direction", choices=["en2ar", "ar2en"], default="en2ar")
    ap.add_argument("--out", default=None, help="defaults to weights/translate_<direction>.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-patience", type=int, default=2)
    ap.add_argument("--max-pairs", type=int, default=0, help="0 = use every pair in the file")
    ap.add_argument("--min-freq", type=int, default=3, help="a word must appear this often to get its own id")
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--min-delta", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--predict", default=None)
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()
    out_path = Path(args.out) if args.out else ROOT / "weights" / f"translate_{args.direction}.pt"

    if args.predict or args.interactive:
        model, src_w2i, tgt_i2w, direction = load_model(out_path)
        label = "English -> Arabic" if direction == "en2ar" else "Arabic -> English"
        if args.predict:
            print(f"[{label}]", translate(model, src_w2i, tgt_i2w, args.predict))
        else:
            print(f"[{label}] Type a sentence (empty line to quit).")
            while True:
                s = input("> ").strip()
                if not s:
                    break
                print("  " + translate(model, src_w2i, tgt_i2w, s))
        return

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)

    rows = read_pairs(args.data)
    if args.direction == "ar2en":
        rows = [(ar, en) for en, ar in rows]      # (src, tgt) = (arabic, english)
    rng.shuffle(rows)
    if args.max_pairs:
        rows = rows[:args.max_pairs]
    n = len(rows)
    n_val = n_test = max(50, int(n * 0.1))
    val, test, train = rows[:n_val], rows[n_val:n_val + n_test], rows[n_val + n_test:]
    print(f"Direction: {args.direction} | pairs: {n} (train {len(train)}, val {len(val)}, test {len(test)})")

    src_w2i = build_vocab([s for s, _ in train], args.min_freq)
    tgt_w2i = build_vocab([t for _, t in train], args.min_freq)
    print(f"Source vocab: {len(src_w2i)} | target vocab: {len(tgt_w2i)}")

    hparams = dict(emb=128, enc_hidden=256, dec_hidden=256)
    model = Seq2Seq(len(src_w2i), len(tgt_w2i), **hparams)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=args.lr_patience)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    demo = [p[0] for p in val[:4]]

    best_score, best_epoch, best_state, wait = float("-inf"), 0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        data = list(train)
        rng.shuffle(data)
        total, steps = 0.0, 0
        for i in range(0, len(data), args.batch):
            src, lens, tgt = make_batch(data[i:i + args.batch], src_w2i, tgt_w2i)
            opt.zero_grad()
            logits = model(src, lens, tgt[:, :-1])
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item()
            steps += 1
            if steps % 200 == 0:
                print(f"  epoch {epoch} batch {steps}/{-(-len(data) // args.batch)} loss {total / steps:.4f}")

        m = evaluate(model, val, src_w2i, tgt_w2i, loss_fn)
        sched.step(m["loss"])
        improved = -m["loss"] > best_score + args.min_delta
        if improved:
            best_score, best_epoch, wait = -m["loss"], epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        lr_now = opt.param_groups[0]["lr"]
        print(f"Epoch {epoch:2d} | lr {lr_now:.2e} | train loss {total / max(steps, 1):.4f} | "
              f"val loss {m['loss']:.4f} | exact-match {m['acc']:.3f}{'  *' if improved else ''}")
        tgt_i2w = {i: w for w, i in tgt_w2i.items()}
        print("   demo:")
        for s in demo:
            print(f"     {' '.join(s)}  ->  {translate(model, src_w2i, tgt_i2w, ' '.join(s))}")
        if wait >= args.patience:
            print(f"Early stopping: no improvement for {args.patience} epochs.")
            break

    print(f"Best epoch: {best_epoch}")
    model.load_state_dict(best_state)
    t = evaluate(model, test, src_w2i, tgt_w2i, loss_fn)
    print(f"\nTest loss: {t['loss']:.4f} | test exact-match accuracy: {t['acc']:.3f}")
    print("(Exact match is strict - a translation that means the same thing but phrased\n"
          " differently still counts as wrong. Read the demo lines above for a better feel.)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "src_w2i": src_w2i, "tgt_w2i": tgt_w2i,
               "hparams": hparams, "direction": args.direction}, out_path)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
