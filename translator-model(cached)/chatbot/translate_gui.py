"""
Translator GUI: type an English or Arabic sentence, press Enter, see the
translation. Switch direction with the dropdown - it loads whichever weights
file matches (translate_en2ar.pt / translate_ar2en.pt).

Project layout this script expects:
    no/
      chatbot/translate_gui.py       <- this file
      scripts/train_translate.py     <- model + training
      rdf/translate_data.txt         <- data (create with scripts/prepare_translate_data.py)
      weights/translate_en2ar.pt     <- trained weights for English -> Arabic
      weights/translate_ar2en.pt     <- trained weights for Arabic -> English

Run from the "no" folder:
    python chatbot/translate_gui.py

Enter = translate | Shift+Enter = new line
You only need ONE of the two weight files to use the GUI - the dropdown will
just show whichever directions are actually trained.
"""
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the "no" folder
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import train_translate as tt  # noqa: E402

WEIGHTS_DIR = os.path.join(ROOT, "weights")
DIRECTIONS = {"English -> Arabic": "en2ar", "Arabic -> English": "ar2en"}


class App:
    def __init__(self, root):
        self.root = root
        self.model = self.src_w2i = self.tgt_i2w = None
        root.title("Translator")
        root.geometry("760x600")

        top = ttk.Frame(root, padding=10)
        top.pack(fill="x")
        row = ttk.Frame(top)
        row.pack(fill="x")
        ttk.Label(row, text="Direction:").pack(side="left")
        self.dir_var = tk.StringVar(value="English -> Arabic")
        self.dir_combo = ttk.Combobox(row, textvariable=self.dir_var, state="readonly",
                                      values=list(DIRECTIONS), width=22)
        self.dir_combo.pack(side="left", padx=6)
        self.dir_combo.bind("<<ComboboxSelected>>", lambda _e: self.load_current())
        ttk.Button(row, text="Load weights...", command=self.pick_model).pack(side="right")

        ttk.Label(top, text="Type or paste a sentence, then press Enter:").pack(anchor="w", pady=(8, 0))
        self.inp = tk.Text(top, height=4, wrap="word", font=("Segoe UI", 13))
        self.inp.pack(fill="x", pady=(4, 6))
        self.inp.bind("<Return>", self.on_enter)
        self.inp.focus_set()

        btns = ttk.Frame(top)
        btns.pack(fill="x")
        ttk.Button(btns, text="Translate", command=self.run).pack(side="left")
        ttk.Button(btns, text="Swap direction", command=self.swap).pack(side="left", padx=6)
        ttk.Button(btns, text="Clear", command=self.clear).pack(side="left")

        ttk.Label(root, text="Translation:", padding=(10, 8, 0, 0)).pack(anchor="w")
        self.out = tk.Text(root, height=6, wrap="word", font=("Segoe UI", 14), state="disabled",
                           relief="flat", bg="#f4f4f4", padx=8, pady=8)
        self.out.pack(fill="x", padx=10)
        ttk.Button(root, text="Copy translation", command=self.copy).pack(anchor="e", padx=10, pady=6)

        self.status = ttk.Label(root, text="", anchor="w", padding=(10, 2))
        self.status.pack(side="bottom", fill="x")

        self.load_current(warn_if_missing=True)

    # ------------------------------ model ------------------------------ #
    def default_path(self, direction):
        return os.path.join(WEIGHTS_DIR, f"translate_{direction}.pt")

    def load_current(self, warn_if_missing=False):
        direction = DIRECTIONS[self.dir_var.get()]
        path = self.default_path(direction)
        if os.path.exists(path):
            self.load_model(path)
        else:
            self.model = None
            msg = f"No weights for this direction yet - run: python scripts/train_translate.py --direction {direction}"
            self.status.config(text=msg)
            if warn_if_missing:
                other = "ar2en" if direction == "en2ar" else "en2ar"
                if os.path.exists(self.default_path(other)):
                    other_label = [k for k, v in DIRECTIONS.items() if v == other][0]
                    self.dir_var.set(other_label)
                    self.load_current()

    def load_model(self, path):
        try:
            self.model, self.src_w2i, self.tgt_i2w, direction = tt.load_model(path)
        except Exception as e:
            messagebox.showerror("Could not load weights", f"{path}\n\n{e}")
            return False
        label = [k for k, v in DIRECTIONS.items() if v == direction]
        if label:
            self.dir_var.set(label[0])
        self.status.config(text=f"Loaded: {path}")
        return True

    def pick_model(self):
        path = filedialog.askopenfilename(title="Choose a translate_*.pt file", initialdir=WEIGHTS_DIR,
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
        if self.model is None:
            messagebox.showwarning("No weights", "Train or load weights for this direction first.")
            return
        result = tt.translate(self.model, self.src_w2i, self.tgt_i2w, text)
        self.out.config(state="normal")
        self.out.delete("1.0", "end")
        self.out.insert("1.0", result or "(no output - try a shorter, simpler sentence)")
        self.out.config(state="disabled")
        self.status.config(text="Translated")

    def swap(self):
        current = self.dir_var.get()
        other = "Arabic -> English" if current == "English -> Arabic" else "English -> Arabic"
        self.dir_var.set(other)
        self.load_current()
        self.clear()

    def copy(self):
        text = self.out.get("1.0", "end").strip()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status.config(text="Translation copied to clipboard")

    def clear(self):
        self.inp.delete("1.0", "end")
        self.out.config(state="normal")
        self.out.delete("1.0", "end")
        self.out.config(state="disabled")
        self.inp.focus_set()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
