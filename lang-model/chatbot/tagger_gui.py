"""
Sentence Tagger GUI: type or paste a sentence, press Enter, see each word's grammar tag.

Project layout this script expects:
    no/
      chatbot/tagger_gui.py          <- this file
      scripts/text.py                <- the training script it imports
      weights/language_model.pt      <- trained weights

Run from the "no" folder:
    python chatbot/tagger_gui.py

Enter = tag the sentence | Shift+Enter = new line
"""
import os
import re
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import __main__
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the "no" folder
sys.path.insert(0, os.path.join(ROOT, "scripts"))                    # so "import text" finds scripts/text.py
import text  # scripts/text.py (the training script)

# language_model.pt was saved by a script run as __main__, so the pickled Vocab
# points at __main__.Vocab. Point that name at the real class.
__main__.Vocab = text.Vocab

BASE = os.path.join(ROOT, "weights")   # where language_model.pt lives
TOKEN_RE = re.compile(r"\w+(?:'\w+)?|[^\w\s]")

MEANING = {
    "NOUN": "noun", "VERB": "verb", "ADJ": "adjective", "ADV": "adverb",
    "DET": "determiner / article", "PRON": "pronoun", "ADP": "preposition",
    "CONJ": "conjunction", "NUM": "number", "PRT": "particle",
    ".": "punctuation", "X": "other / unknown",
}
COLOR = {
    "NOUN": "#1565c0", "VERB": "#c62828", "ADJ": "#2e7d32", "ADV": "#ef6c00",
    "DET": "#6a1b9a", "PRON": "#00838f", "ADP": "#795548", "CONJ": "#ad1457",
    "NUM": "#455a64", "PRT": "#9e9d24", ".": "#616161", "X": "#000000",
}


class App:
    def __init__(self, root):
        self.root = root
        self.model = None
        self.vocab = None
        root.title("Sentence Tagger")
        root.geometry("780x640")

        top = ttk.Frame(root, padding=10)
        top.pack(fill="x")
        ttk.Label(top, text="Type or paste a sentence, then press Enter:").pack(anchor="w")
        self.inp = tk.Text(top, height=3, wrap="word", font=("Segoe UI", 12))
        self.inp.pack(fill="x", pady=(4, 6))
        self.inp.bind("<Return>", self.on_enter)
        self.inp.focus_set()

        btns = ttk.Frame(top)
        btns.pack(fill="x")
        ttk.Button(btns, text="Tag sentence", command=self.run).pack(side="left")
        ttk.Button(btns, text="Clear", command=self.clear).pack(side="left", padx=6)
        ttk.Button(btns, text="Load weights...", command=self.pick_model).pack(side="right")

        # coloured sentence view
        ttk.Label(root, text="Result:", padding=(10, 4, 0, 0)).pack(anchor="w")
        self.view = tk.Text(root, height=4, wrap="word", font=("Segoe UI", 14, "bold"),
                            state="disabled", relief="flat", bg="#f4f4f4", padx=8, pady=8)
        self.view.pack(fill="x", padx=10)
        for tag, color in COLOR.items():
            self.view.tag_configure(tag, foreground=color)

        # table
        table = ttk.Frame(root, padding=10)
        table.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(table, columns=("word", "tag", "meaning"),
                                 show="headings", height=10)
        for col, w in (("word", 220), ("tag", 100), ("meaning", 300)):
            self.tree.heading(col, text={"word": "Word", "tag": "Tag", "meaning": "Grammar type"}[col])
            self.tree.column(col, width=w, anchor="w")
        for tag, color in COLOR.items():
            self.tree.tag_configure(tag, foreground=color)
        scroll = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        # legend
        legend = ttk.Frame(root, padding=(10, 0, 10, 4))
        legend.pack(fill="x")
        for tag, color in COLOR.items():
            tk.Label(legend, text=f"{tag} = {MEANING[tag]}", fg=color,
                     font=("Segoe UI", 8)).pack(side="left", padx=4)

        self.status = ttk.Label(root, text="No weights loaded", anchor="w", padding=(10, 2))
        self.status.pack(fill="x")

    # ------------------------------ model ------------------------------ #
    def load_model(self, path):
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            vocab = ck["vocab"]
            model = text.Tagger(len(vocab.w2i), len(vocab.c2i), len(vocab.t2i), **ck["hparams"])
            model.load_state_dict(ck["state_dict"])
            model.eval()
        except Exception as e:
            messagebox.showerror("Could not load weights", f"{path}\n\n{e}")
            return False
        self.model, self.vocab = model, vocab
        self.status.config(text=f"Loaded: {path}")
        return True

    def pick_model(self):
        path = filedialog.askopenfilename(title="Choose language_model.pt",
                                          initialdir=BASE,
                                          filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")])
        if path:
            self.load_model(path)

    def tag(self, tokens):
        words, chars, _, lens = text.make_collate(self.vocab)([(tokens, None)])
        with torch.no_grad():
            pred = self.model(words, chars, lens).argmax(-1)[0].tolist()
        return [self.vocab.i2t[p] for p in pred]

    # ------------------------------- UI -------------------------------- #
    def on_enter(self, event):
        if event.state & 0x1:      # Shift held: allow a new line
            return None
        self.run()
        return "break"

    def run(self):
        sentence = self.inp.get("1.0", "end").strip()
        if not sentence:
            return
        if self.model is None:
            messagebox.showwarning("No weights", "Load language_model.pt first.")
            return
        tokens = TOKEN_RE.findall(sentence)
        if not tokens:
            return
        tags = self.tag(tokens)
        self.show(tokens, tags)

    def show(self, tokens, tags):
        self.view.config(state="normal")
        self.view.delete("1.0", "end")
        for tok, tg in zip(tokens, tags):
            if tg != "." and self.view.index("end-1c") != "1.0":
                self.view.insert("end", " ")
            self.view.insert("end", tok, tg)
        self.view.config(state="disabled")

        self.tree.delete(*self.tree.get_children())
        for tok, tg in zip(tokens, tags):
            self.tree.insert("", "end", values=(tok, tg, MEANING.get(tg, tg)), tags=(tg,))
        self.status.config(text=f"{len(tokens)} tokens tagged")

    def clear(self):
        self.inp.delete("1.0", "end")
        self.view.config(state="normal")
        self.view.delete("1.0", "end")
        self.view.config(state="disabled")
        self.tree.delete(*self.tree.get_children())
        self.inp.focus_set()


def main():
    root = tk.Tk()
    app = App(root)
    default = os.path.join(BASE, "language_model.pt")
    if os.path.exists(default):
        app.load_model(default)
    else:
        app.pick_model()
    root.mainloop()


if __name__ == "__main__":
    main()