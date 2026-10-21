"""
SEMI fine-tune steps + time calculator. Only calculates (no training, no model).

Run:  python semi_calc.py --data rdf/llm_corpus-v8.txt --sec_per_step 3.5
"""
import argparse
import math
import os

from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--data", default="rdf/llm_corpus-v4.txt")
p.add_argument("--model", default="distilgpt2")   # tokenizer only
p.add_argument("--block", type=int, default=64)
p.add_argument("--batch", type=int, default=4)
p.add_argument("--accum", type=int, default=8)
p.add_argument("--epochs", type=int, default=2)
p.add_argument("--val_split", type=float, default=0.05)
p.add_argument("--sec_per_step", type=float, default=3.0)
args = p.parse_args()


def fmt(sec):
    sec = int(sec)
    d, r = divmod(sec, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    return (f"{d}d " if d else "") + f"{h}h {m}m {s}s"


tok = AutoTokenizer.from_pretrained(args.model)
size = os.path.getsize(args.data)

n_tokens, done = 0, 0
with open(args.data, "r", encoding="utf-8", errors="ignore") as f:
    while chunk := f.read(2_000_000):
        n_tokens += len(tok(chunk).input_ids)
        done += len(chunk.encode("utf-8"))
        print(f"\rcounting tokens... {100*done/size:5.1f}%", end="", flush=True)
print()

n_blocks = n_tokens // args.block
n_val = max(1, int(n_blocks * args.val_split))
n_train = n_blocks - n_val
steps_per_epoch = math.ceil(n_train / args.batch / args.accum)
total_steps = steps_per_epoch * args.epochs

print(f"tokens          : {n_tokens:,}")
print(f"train blocks    : {n_train:,}")
print(f"steps per epoch : {steps_per_epoch:,}")
print(f"epochs          : {args.epochs}")
print(f"TOTAL STEPS     : {total_steps:,}")
print(f"TIME @ {args.sec_per_step}s/step : {fmt(args.sec_per_step * total_steps)}")
