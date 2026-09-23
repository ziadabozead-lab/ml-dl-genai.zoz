"""
Summarizer GUI: paste an article or paragraph, press Enter, see the extracted summary.

If weights/summarizer_model.pt exists (trained with scripts/train_summarizer.py
on your labeled examples in rdf/summary_data.txt), it uses the trained model's
judgment of which sentences matter. If no weights are found yet, it falls back
to plain word-frequency scoring so the tool still works before you've trained
anything - just less accurately.

Project layout this script expects:
    no/
      chatbot/summarizer_gui.py      <- this file
      scripts/train_summarizer.py    <- all the logic (model + fallback)
      rdf/summary_data.txt           <- your labeled training examples
      weights/summarizer_model.pt    <- trained weights (optional)

Run from the "no" folder:
    python chatbot/summarizer_gui.py

Enter = summarize | Shift+Enter = new line
"""
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the "no" folder
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import train_summarizer as tsum  # noqa: E402

WEIGHTS_DIR = os.path.join(ROOT, "weights")
DEFAULT_WEIGHTS = os.path.join(WEIGHTS_DIR, "summarizer_model.pt")
HIGHLIGHT_BG = "#fff59d"   # soft yellow for chosen sentences


class App:
    def __init__(self, root):
        self.root = root
        self.model = None
        self.w2i = None
        root.title("Text Summarizer")
        root.geometry("780x700")

        top = ttk.Frame(root, padding=10)
        top.pack(fill="x")
        ttk.Label(top, text="Paste an article or paragraph, then press Enter:").pack(anchor="w")
        self.inp = tk.Text(top, height=10, wrap="word", font=("Segoe UI", 11))
        self.inp.pack(fill="both", pady=(4, 6))
        self.inp.bind("<Return>", self.on_enter)
        self.inp.focus_set()

        controls = ttk.Frame(top)
        controls.pack(fill="x")
        ttk.Label(controls, text="Sentences to keep:").pack(side="left")
        self.n_var = tk.IntVar(value=3)
        ttk.Spinbox(controls, from_=1, to=20, width=4, textvariable=self.n_var).pack(side="left", padx=(6, 12))
        ttk.Button(controls, text="Summarize", command=self.run).pack(side="left")
        ttk.Button(controls, text="Clear", command=self.clear).pack(side="left", padx=6)
        ttk.Button(controls, text="Load weights...", command=self.pick_model).pack(side="right")
        ttk.Button(controls, text="Copy summary", command=self.copy_summary).pack(side="right", padx=6)

        ttk.Label(root, text="Summary:", padding=(10, 8, 0, 0)).pack(anchor="w")
        self.summary_box = tk.Text(root, height=6, wrap="word", font=("Segoe UI", 12),
                                   state="disabled", relief="flat", bg="#f4f4f4", padx=8, pady=8)
        self.summary_box.pack(fill="x", padx=10)

        ttk.Label(root, text="Original (chosen sentences highlighted):",
                 padding=(10, 8, 0, 0)).pack(anchor="w")
        view_frame = ttk.Frame(root, padding=(10, 0, 10, 6))
        view_frame.pack(fill="both", expand=True)
        self.view = tk.Text(view_frame, wrap="word", font=("Segoe UI", 10),
                            state="disabled", relief="flat")
        self.view.tag_configure("chosen", background=HIGHLIGHT_BG)
        scroll = ttk.Scrollbar(view_frame, orient="vertical", command=self.view.yview)
        self.view.configure(yscrollcommand=scroll.set)
        self.view.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        self.status = ttk.Label(root, text="No trained weights found - using frequency fallback",
                                anchor="w", padding=(10, 2))
        self.status.pack(fill="x")

        self.last_summary = ""
        if os.path.exists(DEFAULT_WEIGHTS):
            self.load_model(DEFAULT_WEIGHTS)

    # ------------------------------ model ------------------------------ #
    def load_model(self, path):
        try:
            self.model, self.w2i = tsum.load_model(path)
        except Exception as e:
            messagebox.showerror("Could not load weights", f"{path}\n\n{e}")
            return False
        self.status.config(text=f"Loaded trained model: {path}")
        return True

    def pick_model(self):
        path = filedialog.askopenfilename(title="Choose summarizer_model.pt", initialdir=WEIGHTS_DIR,
                                          filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")])
        if path:
            self.load_model(path)

    # ------------------------------- UI -------------------------------- #
    def on_enter(self, event):
        if event.state & 0x1:      # Shift held: allow a new line
            return None
        self.run()
        return "break"

    def run(self):
        text = self.inp.get("1.0", "end").strip()
        if not text:
            return
        n = self.n_var.get()

        if self.model is not None:
            summary, scored_all = tsum.score_document(self.model, self.w2i, text, n_sentences=n)
            mode = "trained model"
        else:
            summary, scored_all = tsum.summarize_freq(text, n_sentences=n)
            mode = "frequency fallback"

        if not summary:
            self.status.config(text="No sentences found - try a longer passage.")
            return

        self.last_summary = " ".join(summary)
        self.summary_box.config(state="normal")
        self.summary_box.delete("1.0", "end")
        self.summary_box.insert("1.0", self.last_summary)
        self.summary_box.config(state="disabled")

        self.view.config(state="normal")
        self.view.delete("1.0", "end")
        for i, (sentence, _score, chosen) in enumerate(scored_all):
            tag = ("chosen",) if chosen else ()
            self.view.insert("end", sentence, tag)
            if i < len(scored_all) - 1:
                self.view.insert("end", "  ")
        self.view.config(state="disabled")

        self.status.config(text=f"{len(summary)} of {len(scored_all)} sentences kept ({mode})")

    def clear(self):
        self.inp.delete("1.0", "end")
        self.summary_box.config(state="normal")
        self.summary_box.delete("1.0", "end")
        self.summary_box.config(state="disabled")
        self.view.config(state="normal")
        self.view.delete("1.0", "end")
        self.view.config(state="disabled")
        self.inp.focus_set()

    def copy_summary(self):
        if not self.last_summary:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(self.last_summary)
        self.status.config(text="Summary copied to clipboard")


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
