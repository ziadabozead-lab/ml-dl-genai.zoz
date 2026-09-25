"""
Spelling corrector - a character-level sequence-to-sequence model with attention.
It reads a (possibly misspelled) word one letter at a time and WRITES the
corrected word one letter at a time.

No labeled data needed: take any English text, list its words, and make your own
training pairs by corrupting them the way fingers do (dropped letters, doubled
letters, swapped neighbours, keys next to each other...).
    "receive"  ->  "recieve"   (learns:  recieve -> receive)

How it is trained honestly: the WORD LIST is split first, so validation words
were never seen (even in corrupted form) during training. Fresh random typos
are generated every epoch, so the model can't just memorise them.

How it is used (see correct_text): only words NOT already in the known-word list
are sent to the model, and a fix is accepted only if it lands on a known word.
That stops it from "fixing" names and correct words into nonsense.

Data: any plain .txt (default rdf/spell_corpus.txt), English letters only.
      Your tagged corpus works too:  --data path/to/language_corpus.txt --tagged

Train:      python scripts/train_spell.py
Quick:      python scripts/train_spell.py --max-words 8000 --epochs 8
Try a text: python scripts/train_spell.py --predict "I recieve teh mesage tommorow"
Type many:  python scripts/train_spell.py --interactive

If val loss bounces around instead of settling (watch the "demo" line - it
should keep getting better, not flicker), the LR is dropped automatically
(see --lr / --lr-patience) - no need to restart with a different --lr by hand.
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
MAXLEN = 20                                   # longest word handled
LETTERS = "abcdefghijklmnopqrstuvwxyz"
ALPHABET = LETTERS + "'"
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "rdf" / "spell_corpus.txt"
DEFAULT_OUT = ROOT / "weights" / "spell_model.pt"
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

C2I = {"<pad>": PAD, "<unk>": UNK, "<sos>": SOS, "<eos>": EOS}
for _ch in ALPHABET:
    C2I[_ch] = len(C2I)
I2C = {i: c for c, i in C2I.items()}


# ---------------------------- typo generator ----------------------------- #
def _neighbors():
    rows = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
    nb = {}
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            near = [row[k] for k in (c - 1, c + 1) if 0 <= k < len(row)]
            for rr in (r - 1, r + 1):
                if 0 <= rr < len(rows):
                    near += [rows[rr][k] for k in (c - 1, c, c + 1) if 0 <= k < len(rows[rr])]
            nb[ch] = near
    return nb


NEIGH = _neighbors()


def _one_typo(w, i, rng):
    op = rng.choice(("del", "ins", "sub", "swap", "dup"))
    if op == "del" and len(w) > 3:
        return w[:i] + w[i + 1:]
    if op == "ins":
        return w[:i] + rng.choice(NEIGH.get(w[i], LETTERS)) + w[i:]
    if op == "sub":
        return w[:i] + rng.choice(NEIGH.get(w[i], LETTERS)) + w[i + 1:]
    if op == "swap" and i < len(w) - 1:
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    if op == "dup":
        return w[:i + 1] + w[i] + w[i:]
    return w


def corrupt(word, rng, two_typo_chance=0.15):
    """Apply 1 typo (mostly) or 2 typos (sometimes) so training starts on the easy case."""
    w = word
    n = 2 if rng.random() < two_typo_chance else 1
    for _ in range(n):
        w = _one_typo(w, rng.randrange(len(w)), rng)
    return w[:MAXLEN]


# --------------------------------- data ---------------------------------- #
def read_counts(path, tagged=False):
    """Word -> frequency from a text file (lowercased, letters and inner apostrophes only)."""
    counts = Counter()
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if tagged:
                line = line.split("\t")[0]        # "word<TAB>TAG" -> keep only the word
            counts.update(w.lower() for w in WORD_RE.findall(line))
    return counts


def enc(word):
    return [C2I.get(ch, UNK) for ch in word] or [UNK]


def pad_batch(seqs):
    lens = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    x = torch.zeros(len(seqs), int(lens.max()), dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
    return x, lens


def make_batch(pairs):
    """pairs = [(typo, correct)] -> source ids, source lengths, target ids (with SOS/EOS)."""
    src, lens = pad_batch([enc(s) for s, _ in pairs])
    tgt, _ = pad_batch([[SOS] + enc(t) + [EOS] for _, t in pairs])
    return src, lens, tgt


# --------------------------------- model --------------------------------- #
class Seq2Seq(nn.Module):
    def __init__(self, n_chars, emb=48, enc_hidden=128, dec_hidden=256, dropout=0.2):
        super().__init__()
        H2 = enc_hidden * 2
        self.emb = nn.Embedding(n_chars, emb, padding_idx=PAD)
        self.enc = nn.GRU(emb, enc_hidden, batch_first=True, bidirectional=True)
        self.bridge = nn.Linear(H2, dec_hidden)
        self.dec_emb = nn.Embedding(n_chars, emb, padding_idx=PAD)
        self.dec = nn.GRUCell(emb + H2, dec_hidden)
        self.attn_q = nn.Linear(dec_hidden, H2, bias=False)
        self.out = nn.Linear(dec_hidden + H2, n_chars)
        self.drop = nn.Dropout(dropout)

    def encode(self, src, lens):
        e = self.drop(self.emb(src))
        packed = pack_padded_sequence(e, lens.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.enc(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=src.size(1))
        mask = src != PAD
        m = mask.unsqueeze(-1).float()
        mean = (out * m).sum(1) / m.sum(1).clamp(min=1)
        return out, torch.tanh(self.bridge(mean)), mask

    def step(self, prev, h, enc_out, mask):
        """One output letter: look at the input letters (attention), update state, score letters."""
        q = self.attn_q(h)                                         # [B, 2H]
        scores = (enc_out * q.unsqueeze(1)).sum(-1).masked_fill(~mask, -1e9)
        ctx = (torch.softmax(scores, dim=-1).unsqueeze(-1) * enc_out).sum(1)
        h = self.dec(torch.cat([self.drop(self.dec_emb(prev)), ctx], dim=-1), h)
        return self.out(torch.cat([self.drop(h), ctx], dim=-1)), h

    def forward(self, src, lens, tgt_in):
        enc_out, h, mask = self.encode(src, lens)
        logits = []
        for t in range(tgt_in.size(1)):                            # teacher forcing
            lg, h = self.step(tgt_in[:, t], h, enc_out, mask)
            logits.append(lg)
        return torch.stack(logits, dim=1)                          # [B, T, vocab]

    @torch.no_grad()
    def decode(self, src, lens):
        """Greedy decoding -> list of id lists (without EOS)."""
        enc_out, h, mask = self.encode(src, lens)
        tok = torch.full((src.size(0),), SOS, dtype=torch.long)
        done = torch.zeros(src.size(0), dtype=torch.bool)
        outs = []
        for _ in range(MAXLEN + 2):
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
    model = Seq2Seq(len(C2I), **ck["hparams"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, set(ck["known"])


def fix_words(model, words):
    model.eval()
    src, lens = pad_batch([enc(w) for w in words])
    return ["".join(I2C[i] for i in ids) for ids in model.decode(src, lens)]


def match_case(orig, fixed):
    if len(orig) > 1 and orig.isupper():
        return fixed.upper()
    if orig[0].isupper():
        return fixed.capitalize()
    return fixed


def correct_text(model, known, text, strict=True):
    """Returns (corrected_text, changes) where changes = [(old, new, position_in_new_text)].
    Words already in `known` are left alone. With strict=True a fix is only accepted
    if the result is itself a known word."""
    todo = [(m, m.group(0).lower()) for m in WORD_RE.finditer(text)]
    todo = [(m, w) for m, w in todo if 3 <= len(w) <= MAXLEN and w not in known]
    fixes = fix_words(model, [w for _, w in todo]) if todo else []
    out, changes, last, length = [], [], 0, 0
    for (m, w), fx in zip(todo, fixes):
        if fx == w or (strict and fx not in known):
            continue
        new = match_case(m.group(0), fx)
        prefix = text[last:m.start()]
        out += [prefix, new]
        changes.append((m.group(0), new, length + len(prefix)))
        length += len(prefix) + len(new)
        last = m.end()
    out.append(text[last:])
    return "".join(out), changes


# -------------------------------- evaluate ------------------------------- #
def evaluate(model, pairs, loss_fn, bs=256):
    """Returns mean loss and exact-match accuracy (whole word must be right)."""
    model.eval()
    loss_sum, n_tok, correct = 0.0, 0, 0
    with torch.no_grad():
        for i in range(0, len(pairs), bs):
            chunk = pairs[i:i + bs]
            src, lens, tgt = make_batch(chunk)
            logits = model(src, lens, tgt[:, :-1])
            gold = tgt[:, 1:]
            k = (gold != PAD).sum().item()
            loss_sum += loss_fn(logits.reshape(-1, logits.size(-1)), gold.reshape(-1)).item() * k
            n_tok += k
            for ids, (_, t) in zip(model.decode(src, lens), chunk):
                correct += int(ids == enc(t))
    return {"loss": loss_sum / max(n_tok, 1), "acc": correct / max(len(pairs), 1)}


# ---------------------------------- main --------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--tagged", action="store_true", help='file lines look like "word<TAB>TAG"')
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="lowered from 2e-3: the higher rate made val loss bounce instead of settling")
    ap.add_argument("--lr-patience", type=int, default=2,
                    help="halve the LR after this many epochs with no val-loss improvement")
    ap.add_argument("--max-words", type=int, default=30000, help="use the N most frequent words")
    ap.add_argument("--min-count", type=int, default=2, help="a word must appear this often to train on it")
    ap.add_argument("--known-min", type=int, default=3, help="count needed to treat a word as correctly spelled")
    ap.add_argument("--typos-per-word", type=int, default=3, help="fresh typos generated per word each epoch")
    ap.add_argument("--two-typo-chance", type=float, default=0.15,
                    help="chance a training example gets 2 typos instead of 1 (harder)")
    ap.add_argument("--clean-frac", type=float, default=0.1, help="share of examples that are already correct")
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--min-delta", type=float, default=1e-3)
    ap.add_argument("--monitor", choices=["acc", "loss"], default="loss",
                    help="loss is smoother/less noisy than exact-match acc on a small val set")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--predict", default=None)
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()

    if args.predict or args.interactive:
        model, known = load_model(args.out)
        if args.predict:
            fixed, ch = correct_text(model, known, args.predict)
            print(fixed)
            for old, new, _ in ch:
                print(f"  {old} -> {new}")
        else:
            print("Type a sentence (empty line to quit).")
            while True:
                s = input("> ").strip()
                if not s:
                    break
                print("  " + correct_text(model, known, s)[0])
        return

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)

    counts = read_counts(args.data, args.tagged)
    known = sorted(w for w, c in counts.items() if c >= args.known_min)
    words = [w for w, c in counts.most_common(args.max_words)
             if c >= args.min_count and 3 <= len(w) <= MAXLEN]
    print(f"Distinct words: {len(counts)} | training words: {len(words)} | known-word list: {len(known)}")
    if len(words) < 3000:
        print("Warning: small word list. 5,000+ distinct words works much better - use more text.")

    rng.shuffle(words)
    n_val = max(50, int(len(words) * 0.1))
    val_words, train_words = words[:n_val], words[n_val:]
    vr = random.Random(123)                                    # fixed typos -> comparable epochs
    val_typo = [(corrupt(w, vr, args.two_typo_chance), w) for w in val_words for _ in range(2)]
    val_typo = [(s, t) for s, t in val_typo if s != t]
    val_clean = [(w, w) for w in val_words]
    print(f"Train words: {len(train_words)} | val words: {len(val_words)} (never seen in training) "
          f"| val typos: {len(val_typo)}")

    hparams = dict(emb=48, enc_hidden=128, dec_hidden=256)
    model = Seq2Seq(len(C2I), **hparams)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=args.lr_patience)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    demo = ["teh", "recieve", "becuase", "wich", "freind", "definately"]

    def score_of(m):
        return -m["loss"] if args.monitor == "loss" else m["acc"]

    best_score, best_epoch, best_state, wait = float("-inf"), 0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        data = []
        for w in train_words:
            for _ in range(args.typos_per_word):
                data.append((w if rng.random() < args.clean_frac else corrupt(w, rng, args.two_typo_chance), w))
        rng.shuffle(data)
        total, steps = 0.0, 0
        for i in range(0, len(data), args.batch):
            src, lens, tgt = make_batch(data[i:i + args.batch])
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

        m_typo = evaluate(model, val_typo, loss_fn)
        m_clean = evaluate(model, val_clean, loss_fn)
        sched.step(m_typo["loss"])
        score = score_of(m_typo)
        improved = score > best_score + args.min_delta
        if improved:
            best_score, best_epoch, wait = score, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        lr_now = opt.param_groups[0]["lr"]
        print(f"Epoch {epoch:2d} | lr {lr_now:.2e} | train loss {total / max(steps, 1):.4f} | "
              f"val loss {m_typo['loss']:.4f} | typo->fixed {m_typo['acc']:.3f} | "
              f"correct words kept {m_clean['acc']:.3f}{'  *' if improved else ''}")
        print("   demo:", ", ".join(f"{a}->{b}" for a, b in zip(demo, fix_words(model, demo))))
        if wait >= args.patience:
            print(f"Early stopping: no improvement in {args.monitor} for {args.patience} epochs.")
            break

    print(f"Best epoch: {best_epoch} (monitor={args.monitor})")
    model.load_state_dict(best_state)
    print("\nSample validation results (typo -> model output | truth):")
    for s, t in rng.sample(val_typo, min(10, len(val_typo))):
        out = fix_words(model, [s])[0]
        print(f"  {s:<15} -> {out:<15} | {t}  {'ok' if out == t else 'WRONG'}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "hparams": hparams, "known": known}, args.out)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()