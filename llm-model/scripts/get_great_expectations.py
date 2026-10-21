"""
get_great_expectations.py - download "Great Expectations" (Charles Dickens, public domain) from
Project Gutenberg and write it as ONE clean .txt, ready for fine-tuning.

Place in:  llm-model/scripts/get_great_expectations.py
Run from the llm-model folder (needs internet, no extra packages):

    python scripts/get_great_expectations.py

Output: rdf/llm_corpus-v4.txt  (your v1 file is left untouched). Use --out to write somewhere else.
Train on it with:  python scripts/finetune_semi.py --data rdf/llm_corpus-v4.txt --out weights/llm_model-semi-v4

No internet on this machine? Download the book in a browser from
    https://www.gutenberg.org/ebooks/1400.txt.utf-8
save it as ge_raw.txt and run:   python scripts/get_great_expectations.py --file ge_raw.txt
"""
import argparse
import os
import re
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("--out", default="rdf/llm_corpus-v4.txt")
p.add_argument("--file", default=None, help="use an already-downloaded Gutenberg .txt instead of downloading")
args = p.parse_args()

URLS = [
    "https://www.gutenberg.org/ebooks/1400.txt.utf-8",
    "https://www.gutenberg.org/cache/epub/1400/pg1400.txt",
]


def download():
    last = None
    for url in URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (corpus builder)"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            print(f"downloaded {len(data)/1e6:.2f} MB from {url}")
            return data.decode("utf-8-sig", errors="replace")
        except Exception as e:
            last = e
            print(f"[!] {url} failed: {e}")
    raise SystemExit(f"Could not download the book ({last}). Use --file with a manually downloaded copy.")


if args.file:
    with open(args.file, "r", encoding="utf-8-sig", errors="replace") as f:
        raw = f.read()
else:
    raw = download()

raw = raw.replace("\r\n", "\n").replace("\r", "\n")

# keep only the book itself (drop the Project Gutenberg header and licence footer)
m = re.search(r"\*\*\* ?START OF[^\n]*\n", raw)
if m:
    raw = raw[m.end():]
m = re.search(r"\*\*\* ?END OF", raw)
if m:
    raw = raw[:m.start()]

paragraphs = []
for block in re.split(r"\n\s*\n", raw):
    para = " ".join(line.strip() for line in block.split("\n"))   # join hard-wrapped lines
    para = re.sub(r"\[Illustration[^\]]*\]", "", para)
    para = para.replace("_", "")                                    # Gutenberg italics markers
    para = re.sub(r" {2,}", " ", para).strip()
    if not para:
        continue
    if para.lower() == "contents":
        continue
    if para.count("Chapter") > 1 and re.fullmatch(r"(Chapter [IVXLCDM]+\.?\s*)+", para):
        continue                                                    # the table of contents
    paragraphs.append(para)

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
with open(args.out, "w", encoding="utf-8") as f:
    f.write("\n".join(paragraphs) + "\n")

size = os.path.getsize(args.out)
print(f"wrote {args.out}: {len(paragraphs):,} paragraphs, {size/1e6:.2f} MB, ~{size//4:,} tokens (rough)")
if size < 500_000:
    print("[!] smaller than expected - the download may be incomplete.")
