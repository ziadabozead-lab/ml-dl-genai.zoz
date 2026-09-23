"""
Sentiment GUI: type or paste a sentence, press Enter, see POSITIVE / NEGATIVE + confidence.

Project layout this script expects:
    no/
      chatbot/sentiment_gui.py        <- this file
      scripts/train_sentiment.py      <- the training script it imports
      weights/sentiment_model.pt      <- trained weights

Run from the "no" folder:
    python chatbot/sentiment_gui.py

Enter = analyze | Shift+Enter = new line
"""
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the "no" folder
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import train_sentiment as ts  # noqa: E402

WEIGHTS_DIR = os.path.join(ROOT, "weights")
UNSURE_BELOW = 0.65   # below this the model is close to a coin flip; tune with the table printed by training

COLOR = {"POSITIVE": "#2e7d32", "NEGATIVE": "#c62828", "UNSURE": "#ef6c00", "NO WORDS": "#616161"}


class App:
    def __init__(self, root):
        self.root = root
        self.model = None
        self.w2i = None
        root.title("Sentiment Analyzer")
        root.geometry("760x640")

        top = ttk.Frame(root, padding=10)
        top.pack(fill="x")
        ttk.Label(top, text="Type or paste a sentence, then press Enter:").pack(anchor="w")
        self.inp = tk.Text(top, height=3, wrap="word", font=("Segoe UI", 12))
        self.inp.pack(fill="x", pady=(4, 6))
        self.inp.bind("<Return>", self.on_enter)
        self.inp.focus_set()

        btns = ttk.Frame(top)
        btns.pack(fill="x")
        ttk.Button(btns, text="Analyze", command=self.run).pack(side="left")
        ttk.Button(btns, text="Clear", command=self.clear).pack(side="left", padx=6)
        ttk.Button(btns, text="Clear history", command=self.clear_history).pack(side="left")
        ttk.Button(btns, text="Load weights...", command=self.pick_model).pack(side="right")

        # verdict panel
        panel = tk.Frame(root, bg="#f4f4f4", padx=12, pady=12)
        panel.pack(fill="x", padx=10, pady=(6, 0))
        self.verdict = tk.Label(panel, text="-", font=("Segoe UI", 30, "bold"), bg="#f4f4f4", fg="#616161")
        self.verdict.pack()
        self.detail = tk.Label(panel, text="Waiting for a sentence", font=("Segoe UI", 11),
                               bg="#f4f4f4", fg="#444444")
        self.detail.pack()
        gauge = tk.Frame(panel, bg="#f4f4f4")
        gauge.pack(fill="x", pady=(8, 0))
        tk.Label(gauge, text="Negative", fg=COLOR["NEGATIVE"], bg="#f4f4f4",
                 font=("Segoe UI", 9, "bold")).pack(side="left")
        tk.Label(gauge, text="Positive", fg=COLOR["POSITIVE"], bg="#f4f4f4",
                 font=("Segoe UI", 9, "bold")).pack(side="right")
        self.bar = ttk.Progressbar(gauge, maximum=100, value=50)
        self.bar.pack(fill="x", padx=10, pady=4)

        # history
        ttk.Label(root, text="History (newest first):", padding=(10, 8, 0, 0)).pack(anchor="w")
        table = ttk.Frame(root, padding=(10, 0, 10, 6))
        table.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(table, columns=("text", "verdict", "conf"), show="headings", height=8)
        for col, title, w in (("text", "Sentence", 460), ("verdict", "Verdict", 110), ("conf", "Confidence", 90)):
            self.tree.heading(col, text=title)
            self.tree.column(col, width=w, anchor="w")
        for name, color in COLOR.items():
            self.tree.tag_configure(name, foreground=color)
        scroll = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        self.status = ttk.Label(root, text="No weights loaded", anchor="w", padding=(10, 2))
        self.status.pack(fill="x")

    # ------------------------------ model ------------------------------ #
    def load_model(self, path):
        try:
            self.model, self.w2i = ts.load_model(path)
        except Exception as e:
            messagebox.showerror("Could not load weights", f"{path}\n\n{e}")
            return False
        self.status.config(text=f"Loaded: {path}")
        return True

    def pick_model(self):
        path = filedialog.askopenfilename(title="Choose sentiment_model.pt", initialdir=WEIGHTS_DIR,
                                          filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")])
        if path:
            self.load_model(path)

    # ------------------------------- UI -------------------------------- #
    def on_enter(self, event):
        if event.state & 0x1:      # Shift held: allow a new line
            return None
        self.run()
        return "break"

    def set_verdict(self, name, detail, positive_pct):
        self.verdict.config(text=name, fg=COLOR[name])
        self.detail.config(text=detail)
        self.bar["value"] = positive_pct

    def run(self):
        sentence = self.inp.get("1.0", "end").strip()
        if not sentence:
            return
        if self.model is None:
            messagebox.showwarning("No weights", "Load sentiment_model.pt first.")
            return
        label, conf = ts.predict_text(self.model, self.w2i, sentence)
        if label is None:
            self.set_verdict("NO WORDS", "Try a sentence with real words.", 50)
            return
        positive = conf if label == "POSITIVE" else 1 - conf
        verdict = label if conf >= UNSURE_BELOW else "UNSURE"
        detail = f"{conf:.0%} confident" if verdict != "UNSURE" else f"only {conf:.0%} sure - could go either way"
        self.set_verdict(verdict, detail, positive * 100)
        shown = " ".join(sentence.split())
        if len(shown) > 90:
            shown = shown[:87] + "..."
        self.tree.insert("", 0, values=(shown, verdict, f"{conf:.0%}"), tags=(verdict,))
        self.status.config(text="Analyzed 1 sentence")

    def clear(self):
        self.inp.delete("1.0", "end")
        self.verdict.config(text="-", fg="#616161")
        self.detail.config(text="Waiting for a sentence")
        self.bar["value"] = 50
        self.inp.focus_set()

    def clear_history(self):
        self.tree.delete(*self.tree.get_children())


def main():
    root = tk.Tk()
    app = App(root)
    default = os.path.join(WEIGHTS_DIR, "sentiment_model.pt")
    if os.path.exists(default):
        app.load_model(default)
    else:
        app.pick_model()
    root.mainloop()


if __name__ == "__main__":
    main()