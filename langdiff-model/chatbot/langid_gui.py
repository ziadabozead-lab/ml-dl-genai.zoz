"""
Language Identifier GUI: type or paste text in any language and see, live,
which language the trained model thinks it is - with confidence bars.

Project layout this script expects:
    no/
      chatbot/langid_gui.py          <- this file
      scripts/train_langid.py        <- model + training
      rdf/langid_data.txt            <- labeled examples
      weights/langid_model.pt        <- trained weights (create with train_langid.py)

Run from the "no" folder:
    python chatbot/langid_gui.py
"""
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the "no" folder
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import train_langid as tl  # noqa: E402

WEIGHTS_DIR = os.path.join(ROOT, "weights")
DEFAULT_WEIGHTS = os.path.join(WEIGHTS_DIR, "langid_model.pt")
DELAY_MS = 250          # wait this long after the last keystroke before analysing


class App:
    def __init__(self, root):
        self.root = root
        self.model = self.c2i = self.langs = None
        self._after = None
        root.title("Language Identifier")
        root.geometry("720x560")

        top = ttk.Frame(root, padding=10)
        top.pack(fill="x")
        ttk.Label(top, text="Type or paste text in any language:").pack(anchor="w")
        self.inp = tk.Text(top, height=8, wrap="word", font=("Segoe UI", 12))
        self.inp.pack(fill="x", pady=(4, 6))
        self.inp.bind("<KeyRelease>", self.schedule)
        self.inp.focus_set()

        buttons = ttk.Frame(top)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Clear", command=self.clear).pack(side="left")
        ttk.Button(buttons, text="Load weights...", command=self.pick_model).pack(side="right")

        self.result = ttk.Label(root, text="-", font=("Segoe UI", 30, "bold"), anchor="center")
        self.result.pack(fill="x", pady=(14, 2))
        self.hint = ttk.Label(root, text="", anchor="center", foreground="#666666")
        self.hint.pack(fill="x")

        bars = ttk.Frame(root, padding=(20, 12))
        bars.pack(fill="x")
        bars.columnconfigure(1, weight=1)
        self.names, self.bars, self.pcts = [], [], []
        for r in range(3):
            name = ttk.Label(bars, text="", width=14, font=("Segoe UI", 11))
            bar = ttk.Progressbar(bars, maximum=100, length=360)
            pct = ttk.Label(bars, text="", width=7, anchor="e", font=("Segoe UI", 11))
            name.grid(row=r, column=0, sticky="w", pady=5)
            bar.grid(row=r, column=1, sticky="ew", padx=10)
            pct.grid(row=r, column=2, sticky="e")
            self.names.append(name)
            self.bars.append(bar)
            self.pcts.append(pct)

        self.status = ttk.Label(root, text="", anchor="w", padding=(10, 2))
        self.status.pack(side="bottom", fill="x")

        if os.path.exists(DEFAULT_WEIGHTS):
            self.load_model(DEFAULT_WEIGHTS)
        else:
            self.status.config(text="No weights yet - run: python scripts/train_langid.py")

    # ------------------------------ model ------------------------------ #
    def load_model(self, path):
        try:
            self.model, self.c2i, self.langs = tl.load_model(path)
        except Exception as e:
            messagebox.showerror("Could not load weights", f"{path}\n\n{e}")
            return False
        self.status.config(text=f"Loaded: {path}  ({len(self.langs)} languages: {', '.join(self.langs)})")
        return True

    def pick_model(self):
        path = filedialog.askopenfilename(title="Choose langid_model.pt", initialdir=WEIGHTS_DIR,
                                          filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")])
        if path and self.load_model(path):
            self.analyze()

    # ------------------------------- UI -------------------------------- #
    def schedule(self, _event=None):
        if self._after is not None:
            self.root.after_cancel(self._after)
        self._after = self.root.after(DELAY_MS, self.analyze)

    def reset_view(self):
        self.result.config(text="-")
        self.hint.config(text="")
        for name, bar, pct in zip(self.names, self.bars, self.pcts):
            name.config(text="")
            pct.config(text="")
            bar["value"] = 0

    def analyze(self):
        self._after = None
        if self.model is None:
            return
        text = self.inp.get("1.0", "end").strip()
        if not text:
            self.reset_view()
            return
        ranked = tl.predict(self.model, self.c2i, self.langs, text, top=3)
        self.reset_view()
        best_lang, best_p = ranked[0]
        self.result.config(text=best_lang)
        if best_p < 0.6:
            self.hint.config(text="Not very sure - try a longer piece of text")
        elif len(text) < 6:
            self.hint.config(text="Very short text - the guess may be unreliable")
        for i, (lang, p) in enumerate(ranked):
            self.names[i].config(text=lang)
            self.bars[i]["value"] = p * 100
            self.pcts[i].config(text=f"{p:.1%}")

    def clear(self):
        self.inp.delete("1.0", "end")
        self.reset_view()
        self.inp.focus_set()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
