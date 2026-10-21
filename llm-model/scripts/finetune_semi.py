"""
SEMI (partial) fine-tune of a small pretrained LLM on a plain .txt file (CPU-friendly).

Freezes the embeddings and the lower transformer blocks, and trains only the top N blocks
plus the final layer norm. Compared with a full fine-tune this is faster per step, uses less
memory, and forgets less of what the model already knows - but it adapts less.

Install:   pip install torch transformers
Run:       python scripts/finetune_semi.py --data rdf/llm_corpus-v1.txt --out weights/llm_model-semi-v1
Generate:  python scripts/finetune_semi.py --out weights/llm_model-semi-v1 --generate "Your prompt"

The output is a normal model folder, so chatbot/llm_gui.py can load it.
Knob to play with: --train_layers (DistilGPT-2 has 6 blocks; 2 is a good start, 6 = almost full).
"""
import argparse
import math
import os
import random
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

p = argparse.ArgumentParser()
p.add_argument("--data", default="rdf/llm_corpus-v4.txt", help="path to training .txt")
p.add_argument("--model", default="distilgpt2", help="HF model name (distilgpt2, gpt2, Qwen/Qwen2.5-0.5B ...)")
p.add_argument("--out", default="weights/semi_fine_tune/llm_model-semi-v4")
p.add_argument("--train_layers", type=int, default=5, help="how many of the TOP blocks to train")
p.add_argument("--block", type=int, default=32, help="sequence length (smaller = faster)")
p.add_argument("--batch", type=int, default=64)
p.add_argument("--accum", type=int, default=16, help="gradient accumulation steps")
p.add_argument("--epochs", type=int, default=1)
p.add_argument("--lr", type=float, default=1e-4, help="higher than full fine-tune since fewer weights train")
p.add_argument("--val_split", type=float, default=0.05)
p.add_argument("--threads", type=int, default=os.cpu_count())
p.add_argument("--save_every", type=int, default=100, help="save a checkpoint every N optimizer steps (0 = off)")
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
            ids, attention_mask=torch.ones_like(ids), max_new_tokens=args.max_new, do_sample=True,
            top_k=50, top_p=0.95, temperature=0.8, repetition_penalty=1.1, pad_token_id=tok.eos_token_id,
        )
    print(tok.decode(out[0], skip_special_tokens=True))
    raise SystemExit

# ---------- data ----------
tok = AutoTokenizer.from_pretrained(args.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

with open(args.data, "r", encoding="utf-8") as f:
    text = f.read()

ids = tok(text, return_tensors="pt", truncation=False).input_ids[0]
n_blocks = len(ids) // args.block
if n_blocks < 10:
    raise SystemExit(f"Corpus too small: only {n_blocks} blocks of {args.block} tokens. Add more text.")
blocks = ids[: n_blocks * args.block].view(n_blocks, args.block)
blocks = blocks[torch.randperm(n_blocks)]
n_val = max(1, int(n_blocks * args.val_split))
val, train = blocks[:n_val], blocks[n_val:]
print(f"tokens={len(ids):,}  train blocks={len(train)}  val blocks={len(val)}  device={device}")


def batches(data, bs, shuffle):
    idx = torch.randperm(len(data)) if shuffle else torch.arange(len(data))
    for i in range(0, len(data), bs):
        yield data[idx[i : i + bs]]


# ---------- model: freeze everything, then unfreeze the top blocks ----------
model = AutoModelForCausalLM.from_pretrained(args.model).to(device)

layer_re = re.compile(r"\.(?:h|layers)\.(\d+)\.")           # GPT-2: transformer.h.N.   Qwen/Llama: model.layers.N.
final_norm_names = ("transformer.ln_f", "model.norm", "gpt_neox.final_layer_norm")

n_layers = 1 + max(
    (int(m.group(1)) for n, _ in model.named_parameters() if (m := layer_re.search(n))), default=-1
)
if n_layers == 0:
    raise SystemExit("Could not detect transformer blocks in this model - use finetune_full.py instead.")
first_trainable = max(0, n_layers - args.train_layers)

for name, param in model.named_parameters():
    m = layer_re.search(name)
    in_top_block = m is not None and int(m.group(1)) >= first_trainable
    in_final_norm = any(name.startswith(n) for n in final_norm_names)
    param.requires_grad = in_top_block or in_final_norm

trainable = [p_ for p_ in model.parameters() if p_.requires_grad]
n_train = sum(p_.numel() for p_ in trainable)
n_total = sum(p_.numel() for p_ in model.parameters())
print(f"blocks={n_layers}  training blocks {first_trainable}..{n_layers - 1} + final norm")
print(f"trainable params: {n_train/1e6:.1f}M of {n_total/1e6:.1f}M ({100*n_train/n_total:.0f}%)")

opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
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


def save_checkpoint(step):
    path = os.path.join(args.out, f"checkpoint-{step}")
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tok.save_pretrained(path)
    print(f"   checkpoint saved: {path}")


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
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if args.save_every and step % args.save_every == 0:
                save_checkpoint(step)
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

print(f'done. Test it with:  python scripts/finetune_semi.py --out {args.out} --generate "your prompt"')