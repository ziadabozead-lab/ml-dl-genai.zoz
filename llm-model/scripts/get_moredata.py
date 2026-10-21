# build_corpus.py
# pip install datasets
import re
from pathlib import Path
from datasets import load_dataset

OUT_DIR = Path(r"D:\MyWork\training\no\llm-model\rdf")
EOS = "<|endoftext|>"
LANGS = ("en", "ar")          # use ("en",) for English only
SMOLTALK_LIMIT = 100_000      # smoltalk is ~1M rows, so take a subset

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------- formatting ----------
ROLE = {"system": "### System:", "user": "### User:", "assistant": "### Assistant:"}

def format_chat(messages):
    parts = []
    for m in messages:
        text = (m.get("content") or "").strip()
        if not text:
            continue
        block = f"{ROLE.get(m['role'], '### User:')}\n{text}"
        if m["role"] == "assistant":
            block += EOS
        parts.append(block)
    return "\n\n".join(parts)

def single_turn(prompt, answer):
    return format_chat([
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ])


# ---------- dataset generators ----------
def gen_alpaca():
    for x in load_dataset("yahma/alpaca-cleaned", split="train"):
        prompt = x["instruction"] + ("\n" + x["input"] if x["input"] else "")
        yield single_turn(prompt, x["output"])

def gen_dolly():
    for x in load_dataset("databricks/databricks-dolly-15k", split="train"):
        prompt = x["instruction"] + ("\n" + x["context"] if x["context"] else "")
        yield single_turn(prompt, x["response"])

def gen_no_robots():
    for x in load_dataset("HuggingFaceH4/no_robots", split="train"):
        yield format_chat(x["messages"])

def gen_oasst1():
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    msgs = {m["message_id"]: m for m in ds if not m["deleted"] and m["lang"] in LANGS}
    has_child = {m["parent_id"] for m in msgs.values() if m["parent_id"]}
    for m in msgs.values():
        if m["role"] != "assistant" or m["message_id"] in has_child:
            continue
        chain, cur = [], m
        while cur:
            chain.append({
                "role": "user" if cur["role"] == "prompter" else "assistant",
                "content": cur["text"],
            })
            cur = msgs.get(cur["parent_id"])
        chain.reverse()
        if chain[0]["role"] == "user":
            yield format_chat(chain)

def gen_ultrachat():
    for x in load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft"):
        yield format_chat(x["messages"])

def gen_smoltalk():
    ds = load_dataset("HuggingFaceTB/smoltalk", "all", split="train", streaming=True)
    for i, x in enumerate(ds):
        if i >= SMOLTALK_LIMIT:
            break
        yield format_chat(x["messages"])


# Comment out any line you don't want
DATASETS = [
    ("alpaca-cleaned", gen_alpaca),
    ("dolly-15k",      gen_dolly),
    ("no_robots",      gen_no_robots),
    ("oasst1",         gen_oasst1),
    ("ultrachat_200k", gen_ultrachat),
    ("smoltalk",       gen_smoltalk),
]


def next_version():
    nums = [int(m.group(1))
            for p in OUT_DIR.glob("llm_corpus-v*.txt")
            if (m := re.fullmatch(r"llm_corpus-v(\d+)\.txt", p.name))]
    return max(nums, default=0) + 1


def main():
    for name, gen in DATASETS:
        fname = f"llm_corpus-v{next_version()}.txt"
        path = OUT_DIR / fname
        tmp = path.with_suffix(".part")
        print(f"[run ] {name} -> {fname}")

        count = 0
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            for ex in gen():
                if ex.strip():
                    f.write(ex + "\n\n")
                    count += 1
        tmp.replace(path)
        print(f"[done] {name}: {count:,} examples")

    print("All finished.")

if __name__ == "__main__":
    main()