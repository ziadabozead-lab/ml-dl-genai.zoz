"""
Character-level autocomplete - a 2-layer LSTM that reads text one character at
a time and learns to predict the NEXT character. Once trained, it can continue
whatever you type, like the suggestions on a phone keyboard.

Data: any plain .txt file (default rdf/charlm_corpus.txt). The more text and the
more similar it is to what you'll type, the better the suggestions. You can
point it at your own file, e.g. rdf/language_corpus.txt:
    python scripts/train_charlm.py --data rdf/language_corpus.txt

Train:    python scripts/train_charlm.py
          python scripts/train_charlm.py --epochs 50 --patience 50
Predict:  used internally by chatbot/autocomplete_gui.py

Metrics printed each epoch:
    val loss   - cross-entropy per character (lower = better)
    val ppl    - perplexity = exp(loss): "how many characters is it torn
                 between" on average. Random guessing = vocab size.
    val bpc    - bits per character
A short sample is printed every epoch so you can watch it learn to spell.
"""
import argparse
import math
import random
from pathlib import Path

import torch
import torch.nn as nn

torch.backends.mkldnn.enabled = False  # avoids a Windows LSTM crash on some builds

UNK = 0
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "rdf" / "charlm_corpus.txt"
DEFAULT_OUT = ROOT / "weights" / "charlm_model.pt"
MAX_CONTEXT = 400            # prompt characters used when predicting


# --------------------------------- model --------------------------------- #
class CharLM(nn.Module):
    def __init__(self, n_chars, emb=64, hidden=256, layers=2, dropout=0.25):
        super().__init__()
        self.emb = nn.Embedding(n_chars, emb)
        self.lstm = nn.LSTM(emb, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, n_chars)

    def forward(self, x, state=None):
        out, state = self.lstm(self.drop(self.emb(x)), state)
        return self.fc(self.drop(out)), state


# ------------------------------- data utils ------------------------------- #
def build_vocab(text, min_count=2):
    from collections import Counter
    counts = Counter(text)
    chars = ["<unk>"] + sorted(c for c, n in counts.items() if n >= min_count)
    return {c: i for i, c in enumerate(chars)}


def to_ids(text, c2i):
    return torch.tensor([c2i.get(c, UNK) for c in text], dtype=torch.long)


def batchify(ids, batch):
    """Cut one long id stream into `batch` parallel streams -> [batch, steps]."""
    steps = ids.size(0) // batch
    return ids[:steps * batch].view(batch, steps)


def detach(state):
    return tuple(s.detach() for s in state)


# ------------------------------- generation ------------------------------- #
def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = CharLM(len(ck["c2i"]), **ck["hparams"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck["c2i"]


def sample_next(logits, temperature, top_k):
    logits = logits / max(temperature, 1e-3)
    logits[..., UNK] = float("-inf")                      # never emit <unk>
    if top_k and top_k < logits.size(-1):
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    return torch.multinomial(torch.softmax(logits, dim=-1), 1)


def generate(model, c2i, prompt, n_chars=200, temperature=0.8, top_k=20, n_samples=1):
    """Continue `prompt` by n_chars characters. Returns a list of n_samples strings
    (only the NEW characters, not the prompt)."""
    i2c = {i: c for c, i in c2i.items()}
    prompt = (prompt or "")[-MAX_CONTEXT:] or "\n"
    ids = to_ids(prompt, c2i).unsqueeze(0)
    with torch.no_grad():
        logits, state = model(ids)
        # copy the encoded prompt state for each independent sample
        state = tuple(s.repeat(1, n_samples, 1).contiguous() for s in state)
        last = logits[:, -1, :].repeat(n_samples, 1)
        outs = [[] for _ in range(n_samples)]
        for _ in range(n_chars):
            nxt = sample_next(last.clone(), temperature, top_k)          # [n_samples, 1]
            for k in range(n_samples):
                outs[k].append(i2c[int(nxt[k, 0])])
            logits, state = model(nxt, state)
            last = logits[:, -1, :]
    return ["".join(o) for o in outs]


def suggest(model, c2i, prompt, n_suggestions=3, n_chars=30, temperature=0.6, top_k=15):
    """Autocomplete-style suggestions: short continuations, cut at a word
    boundary, duplicates removed."""
    raw = generate(model, c2i, prompt, n_chars=n_chars, temperature=temperature,
                   top_k=top_k, n_samples=n_suggestions + 2)
    seen, out = set(), []
    for s in raw:
        s = s.split("\n")[0]
        if len(s) > 12 and " " in s.strip():
            s = s[:s.rstrip().rfind(" ")]            # drop a half-written last word
        s = s.rstrip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out[:n_suggestions]


# ---------------------------------- train ---------------------------------- #
def evaluate(model, data, seq_len, loss_fn):
    model.eval()
    state, total, count = None, 0.0, 0
    with torch.no_grad():
        for i in range(0, data.size(1) - 1, seq_len):
            L = min(seq_len, data.size(1) - 1 - i)
            x, y = data[:, i:i + L], data[:, i + 1:i + 1 + L]
            logits, state = model(x, state)
            total += loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item() * y.numel()
            count += y.numel()
    return total / max(count, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--patience", type=int, default=5,
                    help="epochs without improvement before stopping")
    ap.add_argument("--min-delta", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)

    with open(args.data, encoding="utf-8", errors="ignore") as f:
        text = f.read().replace("\r\n", "\n")
    if len(text) < 20000:
        print(f"Warning: only {len(text):,} characters of text. 100,000+ works much better.")
    cut = int(len(text) * (1 - args.val_frac))
    train_text, val_text = text[:cut], text[cut:]
    print(f"Corpus: {len(text):,} characters (train {len(train_text):,}, val {len(val_text):,})")

    c2i = build_vocab(train_text)
    print(f"Character vocab: {len(c2i)}")
    train = batchify(to_ids(train_text, c2i), args.batch)
    val = batchify(to_ids(val_text, c2i), max(1, min(16, len(val_text) // (args.seq_len * 2))))
    print(f"Training streams: {tuple(train.shape)} | val streams: {tuple(val.shape)}")

    hparams = dict(emb=64, hidden=args.hidden, layers=args.layers)
    model = CharLM(len(c2i), **hparams)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
    loss_fn = nn.CrossEntropyLoss()

    best_loss, best_epoch, best_state, wait = float("inf"), 0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        state, total, steps = None, 0.0, 0
        for i in range(0, train.size(1) - 1, args.seq_len):
            L = min(args.seq_len, train.size(1) - 1 - i)
            x, y = train[:, i:i + L], train[:, i + 1:i + 1 + L]
            if state is not None:
                state = detach(state)
            opt.zero_grad()
            logits, state = model(x, state)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            steps += 1

        val_loss = evaluate(model, val, args.seq_len, loss_fn)
        sched.step(val_loss)
        improved = val_loss < best_loss - args.min_delta
        if improved:
            best_loss, best_epoch, wait = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1

        print(f"Epoch {epoch:2d} | train loss {total / max(steps, 1):.4f} | val loss {val_loss:.4f} | "
              f"val ppl {math.exp(min(val_loss, 20)):.2f} | val bpc {val_loss / math.log(2):.3f}"
              f"{'  *' if improved else ''}")
        model.eval()
        print("   sample:", repr(generate(model, c2i, "The ", n_chars=70, temperature=0.7)[0]))

        if wait >= args.patience:
            print(f"Early stopping: no improvement for {args.patience} epochs.")
            break

    print(f"Best epoch: {best_epoch}")
    if best_state is not None:
        model.load_state_dict(best_state)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "c2i": c2i, "hparams": hparams}, args.out)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()