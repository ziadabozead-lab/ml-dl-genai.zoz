"""
get_reasoning_data.py - build a step-by-step REASONING corpus in Qwen's chat format (ChatML), one .txt.

Sources (downloaded with the `datasets` package you already have, internet needed once):
  * openai/gsm8k           ~7.4k grade-school math word problems with worked solutions (MIT licence)
  * meta-math/MetaMathQA   optional extra: bigger set of math Q&A with step-by-step answers (--metamath N)

Every example looks like:
    <|im_start|>system ... <|im_end|>
    <|im_start|>user   question <|im_end|>
    <|im_start|>assistant   step-by-step solution ... The answer is 18.<|im_end|>
which is the format Qwen-Instruct models use, so the fine-tune teaches the model to
"think step by step, then give the answer" in exactly the way the chat GUI will ask for it.

Place in:  llm-model/scripts/get_reasoning_data.py
Run from the llm-model folder:

    python scripts/get_reasoning_data.py                  # -> rdf/reasoning_qwen-v6.txt  (GSM8K only)
    python scripts/get_reasoning_data.py --metamath 8000  # + 8000 MetaMathQA examples

Then fine-tune Qwen with LoRA on a free Colab/Kaggle GPU (see finetune_lora.py):
    python finetune_lora.py --data reasoning_qwen-v6.txt --model Qwen/Qwen2.5-0.5B-Instruct --out lora-reasoning-v6
"""
import argparse
import os
import random
import re

p = argparse.ArgumentParser()
p.add_argument("--out", default="rdf/reasoning_qwen-v6.txt")
p.add_argument("--max", type=int, default=7500, help="max GSM8K examples")
p.add_argument("--metamath", type=int, default=0, help="also add this many MetaMathQA examples")
# keep identical to DEFAULT_SYSTEM in chatbot/llm_chat_rag_gui.py
p.add_argument("--system", default="You are a helpful assistant. Think step by step before answering.")
args = p.parse_args()

from datasets import load_dataset  # noqa: E402


def wrap(system, question, answer):
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n{answer}<|im_end|>\n")


examples = []

print("loading openai/gsm8k ...")
gsm = load_dataset("openai/gsm8k", "main", split="train")
for row in gsm:
    q = row["question"].strip()
    body, _, final = row["answer"].partition("####")
    body = re.sub(r"<<[^>]*>>", "", body).strip()          # drop the calculator annotations <<48/2=24>>
    final = final.strip()
    if not q or not body or not final:
        continue
    examples.append(wrap(args.system, q, f"{body}\nThe answer is {final}."))
examples = examples[: args.max]
print(f"  GSM8K examples: {len(examples):,}")

if args.metamath > 0:
    try:
        print("loading meta-math/MetaMathQA ...")
        mm = load_dataset("meta-math/MetaMathQA", split="train").shuffle(seed=42).select(range(args.metamath))
        added = 0
        for row in mm:
            q, a = row["query"].strip(), row["response"].strip()
            if q and a:
                examples.append(wrap(args.system, q, a))
                added += 1
        print(f"  MetaMathQA examples: {added:,}")
    except Exception as e:
        print(f"[!] could not load MetaMathQA ({e}) - continuing with GSM8K only")

random.seed(42)
random.shuffle(examples)

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
with open(args.out, "w", encoding="utf-8") as f:
    f.write("".join(examples))

size = os.path.getsize(args.out)
print(f"wrote {args.out}: {len(examples):,} examples, {size/1e6:.2f} MB, ~{size//4:,} tokens (rough)")
print("\nsample:\n" + examples[0])
