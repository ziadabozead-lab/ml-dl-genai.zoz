"""
Emotion GUI: type or paste a sentence and see which emotions the model hears in
it, with a coloured bar for every emotion. Updates live as you type.

Project layout this script expects:
    no/
      chatbot/emotion_gui.py         <- this file
      scripts/train_emotion.py       <- model + training
      rdf/emotion_data.txt           <- data (create with scripts/prepare_emotion_data.py)
      weights/emotion_model.pt       <- trained weights (create with train_emotion.py)

Run from the "no" folder:
    python chatbot/emotion_gui.py
"""
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the "no" folder
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import train_emotion as te  # noqa: E402

WEIGHTS_DIR = os.path.join(ROOT, "weights")
DEFAULT_WEIGHTS = os.path.join(WEIGHTS_DIR, "emotion_model.pt")
DELAY_MS = 300
UNSURE_BELOW = 0.45   # with 6 emotions, a top score under this means "mixed feelings"

COLOR = {"sadness": "#1565c0", "joy": "#f9a825", "love": "#ad1457",
         "anger": "#c62828", "fear": "#6a1b9a", "surprise": "#00838f"}
FALLBACK = "#616161"


class App:
    def __init__(self, root):
        self.root = root
        self.model = self.w2i = self.labels = None
        self._after = None
        self.rows = []
        root.title("Emotion Detector")
        root.geometry("720x600")

        top = ttk.Frame(root, padding=10)
        top.pack(fill="x")
        ttk.Label(top, text="Type or paste a sentence:").pack(anchor="w")
        self.inp = tk.Text(top, height=5, wrap="word", font=("Segoe UI", 12))
        self.inp.pack(fill="x", pady=(4, 6))
        self.inp.bind("<KeyRelease>", self.schedule)
        self.inp.focus_set()
        btns = ttk.Frame(top)
        btns.pack(fill="x")
        ttk.Button(btns, text="Clear", command=self.clear).pack(side="left")
        ttk.Button(btns, text="Load weights...", command=self.pick_model).pack(side="right")

        self.verdict = tk.Label(root, text="-", font=("Segoe UI", 30, "bold"), fg=FALLBACK)
        self.verdict.pack(fill="x", pady=(14, 0))
        self.hint = ttk.Label(root, text="", anchor="center", foreground="#666666")
        self.hint.pack(fill="x")

        self.bars = ttk.Frame(root, padding=(20, 12))
        self.bars.pack(fill="both", expand=True)
        self.bars.columnconfigure(1, weight=1)

        self.status = ttk.Label(root, text="", anchor="w", padding=(10, 2))
        self.status.pack(side="bottom", fill="x")

        if os.path.exists(DEFAULT_WEIGHTS):
            self.load_model(DEFAULT_WEIGHTS)
        else:
            self.status.config(text="No weights yet - run: python scripts/train_emotion.py")

    # ------------------------------ model ------------------------------ #
    def load_model(self, path):
        try:
            self.model, self.w2i, self.labels = te.load_model(path)
        except Exception as e:
            messagebox.showerror("Could not load weights", f"{path}\n\n{e}")
            return False
        self.build_rows(len(self.labels))
        self.status.config(text=f"Loaded: {path}  ({', '.join(self.labels)})")
        return True

    def pick_model(self):
        path = filedialog.askopenfilename(title="Choose emotion_model.pt", initialdir=WEIGHTS_DIR,
                                          filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")])
        if path and self.load_model(path):
            self.analyze()

    # ------------------------------- UI -------------------------------- #
    def build_rows(self, n):
        for w in self.bars.winfo_children():
            w.destroy()
        self.rows = []
        for r in range(n):
            name = ttk.Label(self.bars, text="", width=10, font=("Segoe UI", 11))
            canvas = tk.Canvas(self.bars, height=22, highlightthickness=0, bg="#eeeeee")
            pct = ttk.Label(self.bars, text="", width=6, anchor="e", font=("Segoe UI", 11))
            name.grid(row=r, column=0, sticky="w", pady=4)
            canvas.grid(row=r, column=1, sticky="ew", padx=10)
            pct.grid(row=r, column=2, sticky="e")
            self.rows.append((name, canvas, pct))

    def schedule(self, _event=None):
        if self._after is not None:
            self.root.after_cancel(self._after)
        self._after = self.root.after(DELAY_MS, self.analyze)

    def reset_view(self):
        self.verdict.config(text="-", fg=FALLBACK)
        self.hint.config(text="")
        for name, canvas, pct in self.rows:
            name.config(text="")
            pct.config(text="")
            canvas.delete("all")

    def analyze(self):
        self._after = None
        if self.model is None:
            return
        text = self.inp.get("1.0", "end").strip()
        if not text:
            self.reset_view()
            return
        ranked = te.predict_text(self.model, self.w2i, self.labels, text)
        self.reset_view()
        if not ranked:
            self.hint.config(text="No words found - try a real sentence")
            return
        best, p = ranked[0]
        if p < UNSURE_BELOW:
            self.verdict.config(text="mixed / unsure", fg="#ef6c00")
            self.hint.config(text=f"closest: {best} ({p:.0%})")
        else:
            self.verdict.config(text=best, fg=COLOR.get(best, FALLBACK))
            if len(text.split()) < 3:
                self.hint.config(text="Very short text - the guess may be unreliable")
        for (name, canvas, pct), (label, prob) in zip(self.rows, ranked):
            name.config(text=label)
            pct.config(text=f"{prob:.0%}")
            canvas.update_idletasks()
            width = max(canvas.winfo_width(), 100)
            canvas.create_rectangle(0, 0, width * prob, 22, fill=COLOR.get(label, FALLBACK), width=0)

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
