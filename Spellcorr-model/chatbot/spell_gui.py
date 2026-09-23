"""
Spelling Corrector GUI: type or paste text, and the trained model fixes your
typos live. Corrected words are highlighted; the table lists every change.

Project layout this script expects:
    no/
      chatbot/spell_gui.py           <- this file
      scripts/train_spell.py         <- model + training
      rdf/spell_corpus.txt           <- any English text the model learns words from
      weights/spell_model.pt         <- trained weights (create with train_spell.py)

Run from the "no" folder:
    python chatbot/spell_gui.py

"Only accept fixes to known words" (recommended) stops the model from turning
names or rare words into nonsense; untick it to see the raw model behaviour.
"""
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the "no" folder
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import train_spell as ts  # noqa: E402

WEIGHTS_DIR = os.path.join(ROOT, "weights")
DEFAULT_WEIGHTS = os.path.join(WEIGHTS_DIR, "spell_model.pt")
DELAY_MS = 400
HIGHLIGHT_BG = "#fff59d"


class App:
    def __init__(self, root):
        self.root = root
        self.model = self.known = None
        self._after = None
        self.corrected = ""
        root.title("Spelling Corrector")
        root.geometry("760x680")

        top = ttk.Frame(root, padding=10)
        top.pack(fill="x")
        ttk.Label(top, text="Type or paste text (typos welcome):").pack(anchor="w")
        self.inp = tk.Text(top, height=7, wrap="word", font=("Segoe UI", 12), undo=True)
        self.inp.pack(fill="x", pady=(4, 6))
        self.inp.bind("<KeyRelease>", self.schedule)
        self.inp.focus_set()

        row = ttk.Frame(top)
        row.pack(fill="x")
        self.strict = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="Only accept fixes to known words", variable=self.strict,
                        command=self.analyze).pack(side="left")
        ttk.Button(row, text="Load weights...", command=self.pick_model).pack(side="right")
        ttk.Button(row, text="Clear", command=self.clear).pack(side="right", padx=6)

        ttk.Label(root, text="Corrected:", padding=(10, 8, 0, 0)).pack(anchor="w")
        self.out = tk.Text(root, height=7, wrap="word", font=("Segoe UI", 12), state="disabled",
                           relief="flat", bg="#f4f4f4", padx=8, pady=8)
        self.out.tag_configure("fixed", background=HIGHLIGHT_BG)
        self.out.pack(fill="x", padx=10)

        actions = ttk.Frame(root, padding=(10, 6))
        actions.pack(fill="x")
        ttk.Button(actions, text="Copy corrected", command=self.copy).pack(side="left")
        ttk.Button(actions, text="Use corrected as input", command=self.use_corrected).pack(side="left", padx=6)

        ttk.Label(root, text="Changes:", padding=(10, 4, 0, 0)).pack(anchor="w")
        table = ttk.Frame(root, padding=(10, 0, 10, 6))
        table.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(table, columns=("old", "new"), show="headings", height=6)
        self.tree.heading("old", text="You typed")
        self.tree.heading("new", text="Corrected to")
        self.tree.column("old", width=240, anchor="w")
        self.tree.column("new", width=240, anchor="w")
        scroll = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        self.status = ttk.Label(root, text="", anchor="w", padding=(10, 2))
        self.status.pack(fill="x")

        if os.path.exists(DEFAULT_WEIGHTS):
            self.load_model(DEFAULT_WEIGHTS)
        else:
            self.status.config(text="No weights yet - run: python scripts/train_spell.py")

    # ------------------------------ model ------------------------------ #
    def load_model(self, path):
        try:
            self.model, self.known = ts.load_model(path)
        except Exception as e:
            messagebox.showerror("Could not load weights", f"{path}\n\n{e}")
            return False
        self.status.config(text=f"Loaded: {path}  ({len(self.known):,} known words)")
        return True

    def pick_model(self):
        path = filedialog.askopenfilename(title="Choose spell_model.pt", initialdir=WEIGHTS_DIR,
                                          filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")])
        if path and self.load_model(path):
            self.analyze()

    # ------------------------------- UI -------------------------------- #
    def schedule(self, _event=None):
        if self._after is not None:
            self.root.after_cancel(self._after)
        self._after = self.root.after(DELAY_MS, self.analyze)

    def show(self, text, changes):
        self.out.config(state="normal")
        self.out.delete("1.0", "end")
        self.out.insert("1.0", text)
        for _old, new, pos in changes:
            self.out.tag_add("fixed", f"1.0+{pos}c", f"1.0+{pos + len(new)}c")
        self.out.config(state="disabled")
        self.tree.delete(*self.tree.get_children())
        for old, new, _ in changes:
            self.tree.insert("", "end", values=(old, new))

    def analyze(self):
        self._after = None
        if self.model is None:
            return
        text = self.inp.get("1.0", "end-1c")
        if not text.strip():
            self.corrected = ""
            self.show("", [])
            return
        try:
            self.corrected, changes = ts.correct_text(self.model, self.known, text, strict=self.strict.get())
        except Exception as e:
            self.status.config(text=f"Error: {e}")
            return
        self.show(self.corrected, changes)
        self.status.config(text=f"{len(changes)} word(s) corrected")

    def copy(self):
        if self.corrected:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.corrected)
            self.status.config(text="Corrected text copied to clipboard")

    def use_corrected(self):
        if self.corrected:
            self.inp.delete("1.0", "end")
            self.inp.insert("1.0", self.corrected)
            self.analyze()

    def clear(self):
        self.inp.delete("1.0", "end")
        self.corrected = ""
        self.show("", [])
        self.inp.focus_set()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
