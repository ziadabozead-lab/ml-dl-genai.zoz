"""
Full fine-tune of DistilGPT-2 on a plain .txt file (CPU-friendly).

Install:   pip install torch transformers
Run:       python finetune_full.py --data corpus.txt
Then chat: python finetune_full.py --generate "Your prompt here"
"""
import argparse
import math
import os
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

p = argparse.ArgumentParser()
p.add_argument("--data", default="rdf/llm_corpus-v3.txt", help="path to training .txt")
p.add_argument("--model", default="distilgpt2", help="HF model name (distilgpt2, gpt2)")
p.add_argument("--out", default="weights/full_fine_tune/llm_model_full-v3")
p.add_argument("--block", type=int, default=32, help="sequence length (smaller = faster)")
p.add_argument("--batch", type=int, default=4)
p.add_argument("--accum", type=int, default=8, help="gradient accumulation steps")
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--lr", type=float, default=5e-5)
p.add_argument("--val_split", type=float, default=0.05)
p.add_argument("--threads", type=int, default=os.cpu_count())
p.add_argument("--generate", default=None, help="skip training, generate from this prompt")
p.add_argument("--max_new", type=int, default=100)
args = p.parse_args()

torch.set_num_threads(args.threads)
torch.manual_seed(42)
random.seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- generate-only mode ----------
if args.generate is not None:
    tok = AutoTokenizer.from_pretrained(args.out)
    model = AutoModelForCausalLM.from_pretrained(args.out).to(device).eval()
    ids = tok(args.generate, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=args.max_new, do_sample=True, top_k=50, top_p=0.95,
            temperature=0.8, repetition_penalty=1.1, pad_token_id=tok.eos_token_id,
        )
    print(tok.decode(out[0], skip_special_tokens=True))
    raise SystemExit

# ---------- data ----------
tok = AutoTokenizer.from_pretrained(args.model)
tok.pad_token = tok.eos_token

with open(args.data, "r", encoding="utf-8") as f:
    text = f.read()

ids = tok(text, return_tensors="pt", truncation=False).input_ids[0]
n_blocks = len(ids) // args.block
if n_blocks < 10:
    raise SystemExit(f"Corpus too small: only {n_blocks} blocks of {args.block} tokens. Add more text.")
blocks = ids[: n_blocks * args.block].view(n_blocks, args.block)
perm = torch.randperm(n_blocks)
blocks = blocks[perm]
n_val = max(1, int(n_blocks * args.val_split))
val, train = blocks[:n_val], blocks[n_val:]
print(f"tokens={len(ids):,}  train blocks={len(train)}  val blocks={len(val)}  device={device}")


def batches(data, bs, shuffle):
    idx = torch.randperm(len(data)) if shuffle else torch.arange(len(data))
    for i in range(0, len(data), bs):
        yield data[idx[i : i + bs]]


# ---------- model ----------
model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
steps_per_epoch = math.ceil(len(train) / args.batch / args.accum)
total_steps = steps_per_epoch * args.epochs
sched = get_linear_schedule_with_warmup(opt, int(0.05 * total_steps), total_steps)


@torch.no_grad()
def evaluate():
    model.eval()
    losses = []
    for b in batches(val, args.batch, False):
        b = b.to(device)
        losses.append(model(input_ids=b, labels=b).loss.item())
    model.train()
    return sum(losses) / len(losses)


# ---------- train ----------
model.train()
best = float("inf")
step = 0
for epoch in range(1, args.epochs + 1):
    running, n = 0.0, 0
    opt.zero_grad()
    for i, b in enumerate(batches(train, args.batch, True), 1):
        b = b.to(device)
        loss = model(input_ids=b, labels=b).loss
        (loss / args.accum).backward()
        running += loss.item()
        n += 1
        if i % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 10 == 0:
                print(f"epoch {epoch} step {step}/{total_steps} train_loss {running/n:.4f}")
    val_loss = evaluate()
    print(f"== epoch {epoch}: train {running/max(n,1):.4f}  val {val_loss:.4f}  ppl {math.exp(val_loss):.1f}")
    if val_loss < best:
        best = val_loss
        os.makedirs(args.out, exist_ok=True)
        model.save_pretrained(args.out)
        tok.save_pretrained(args.out)
        print(f"   saved best model to {args.out}")

print("done. Test it with:  python finetune_full.py --generate \"your prompt\"")
