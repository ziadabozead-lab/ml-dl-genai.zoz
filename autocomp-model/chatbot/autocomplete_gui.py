"""
Autocomplete GUI: start typing and the trained character-level model suggests
how to continue. Press Tab to accept the first suggestion, or click one.

Project layout this script expects:
    no/
      chatbot/autocomplete_gui.py    <- this file
      scripts/train_charlm.py        <- model + training
      rdf/charlm_corpus.txt          <- text the model learns from
      weights/charlm_model.pt        <- trained weights (create with train_charlm.py)

Run from the "no" folder:
    python chatbot/autocomplete_gui.py

Tab = accept first suggestion | "Write more" = let the model keep writing
Temperature: low = safe and repetitive, high = creative and error-prone.
Weights dropdown: pick any .pt file already sitting in weights/. Hit the
refresh button (⟳) if you trained a new one while this was open.
"""
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the "no" folder
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import train_charlm as tc  # noqa: E402

WEIGHTS_DIR = os.path.join(ROOT, "weights")
DEFAULT_WEIGHTS = os.path.join(WEIGHTS_DIR, "charlm_model.pt")
DELAY_MS = 450          # wait this long after the last keystroke before suggesting
N_SUGGESTIONS = 3


def list_weight_files():
    """Basenames of *.pt files in weights/, newest-modified first."""
    if not os.path.isdir(WEIGHTS_DIR):
        return []
    files = [f for f in os.listdir(WEIGHTS_DIR) if f.lower().endswith(".pt")]
    files.sort(key=lambda f: os.path.getmtime(os.path.join(WEIGHTS_DIR, f)), reverse=True)
    return files


class App:
    def __init__(self, root):
        self.root = root
        self.model = self.c2i = None
        self._after = None
        self.suggestions = []
        root.title("Autocomplete")
        root.geometry("760x650")

        top = ttk.Frame(root, padding=10)
        top.pack(fill="both", expand=True)

        self.status = ttk.Label(root, text="", anchor="w", padding=(10, 2))
        self.status.pack(side="bottom", fill="x")

        # -------------------------------------------------------------- #
        # Everything below is packed side="bottom", in REVERSE of the    #
        # order it should visually appear (last packed = topmost of the #
        # group). This pins the whole control stack to the bottom of    #
        # the window so it can never get pushed off-screen, no matter   #
        # how big the text box or the window itself gets.                #
        # -------------------------------------------------------------- #
        buttons = ttk.Frame(top)
        buttons.pack(side="bottom", fill="x", pady=(8, 0))
        ttk.Button(buttons, text="Write more", command=self.write_more).pack(side="left")
        ttk.Button(buttons, text="Clear", command=self.clear).pack(side="left", padx=6)

        controls = ttk.Frame(top)
        controls.pack(side="bottom", fill="x", pady=(10, 0))
        ttk.Label(controls, text="Temperature:").pack(side="left")
        self.temp = tk.DoubleVar(value=0.6)
        ttk.Scale(controls, from_=0.2, to=1.2, variable=self.temp, length=140,
                  command=lambda _v: self.temp_label.config(text=f"{self.temp.get():.2f}")).pack(side="left", padx=6)
        self.temp_label = ttk.Label(controls, text="0.60", width=5)
        self.temp_label.pack(side="left")
        ttk.Label(controls, text="Suggestion length:").pack(side="left", padx=(14, 0))
        self.length = tk.IntVar(value=30)
        ttk.Spinbox(controls, from_=10, to=80, width=4, textvariable=self.length).pack(side="left", padx=6)

        # ------------------------- weights selector ------------------------- #
        weights_row = ttk.Frame(top)
        weights_row.pack(side="bottom", fill="x", pady=(10, 0))
        ttk.Label(weights_row, text="Weights:").pack(side="left")
        self.weights_var = tk.StringVar()
        self.weights_combo = ttk.Combobox(weights_row, textvariable=self.weights_var,
                                          state="readonly", width=32)
        self.weights_combo.pack(side="left", padx=6)
        self.weights_combo.bind("<<ComboboxSelected>>", self.on_weights_selected)
        ttk.Button(weights_row, text="\u27f3", width=3, command=self.refresh_weights).pack(side="left")
        ttk.Button(weights_row, text="Browse...", command=self.pick_model).pack(side="left", padx=(6, 0))

        self.sug_frame = ttk.Frame(top)
        self.sug_frame.pack(side="bottom", fill="x")
        self.sug_buttons = []
        for i in range(N_SUGGESTIONS):
            b = ttk.Button(self.sug_frame, text="", command=lambda k=i: self.accept(k))
            b.pack(fill="x", pady=2)
            self.sug_buttons.append(b)

        ttk.Label(top, text="Suggestions:").pack(side="bottom", anchor="w", pady=(4, 2))

        # The text box takes whatever space is left in the middle, however
        # much or little that is - it will never crowd out the controls above.
        ttk.Label(top, text="Start typing (Tab accepts the first suggestion):").pack(side="top", anchor="w")
        box = ttk.Frame(top)
        box.pack(side="top", fill="both", expand=True, pady=(4, 6))
        self.inp = tk.Text(box, wrap="word", font=("Segoe UI", 13), undo=True)
        scroll = ttk.Scrollbar(box, orient="vertical", command=self.inp.yview)
        self.inp.configure(yscrollcommand=scroll.set)
        self.inp.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.inp.bind("<KeyRelease>", self.on_key)
        self.inp.bind("<Tab>", self.on_tab)
        self.inp.focus_set()

        self.refresh_weights()
        if os.path.exists(DEFAULT_WEIGHTS):
            self.load_model(DEFAULT_WEIGHTS)
        elif self.weights_combo["values"]:
            # nothing named charlm_model.pt, but something's in weights/ - load the newest
            first = self.weights_combo["values"][0]
            self.weights_var.set(first)
            self.load_model(os.path.join(WEIGHTS_DIR, first))
        else:
            self.status.config(text="No weights yet - run: python scripts/train_charlm.py")

    # ------------------------------ model ------------------------------ #
    def load_model(self, path):
        try:
            self.model, self.c2i = tc.load_model(path)
        except Exception as e:
            messagebox.showerror("Could not load weights", f"{path}\n\n{e}")
            return False
        self.status.config(text=f"Loaded: {path}")
        return True

    def refresh_weights(self):
        """Rescan weights/ and repopulate the dropdown, keeping the current
        selection if it's still present."""
        files = list_weight_files()
        current = self.weights_var.get()
        self.weights_combo["values"] = files
        if current in files:
            self.weights_var.set(current)
        elif files:
            self.weights_var.set(files[0])
        else:
            self.weights_var.set("")

    def on_weights_selected(self, _event):
        name = self.weights_var.get()
        if name:
            self.load_model(os.path.join(WEIGHTS_DIR, name))

    def pick_model(self):
        path = filedialog.askopenfilename(title="Choose a .pt weights file", initialdir=WEIGHTS_DIR,
                                          filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")])
        if path:
            if self.load_model(path):
                # if it's inside weights/, reflect it in the dropdown too
                if os.path.dirname(os.path.abspath(path)) == os.path.abspath(WEIGHTS_DIR):
                    self.refresh_weights()
                    self.weights_var.set(os.path.basename(path))
                else:
                    self.weights_var.set(os.path.basename(path))

    # ------------------------------- UI -------------------------------- #
    def prompt(self):
        """Everything typed before the cursor."""
        return self.inp.get("1.0", "insert")

    def on_key(self, event):
        if event.keysym in ("Tab", "Shift_L", "Shift_R", "Control_L", "Control_R", "Alt_L", "Alt_R"):
            return
        self.clear_suggestions()
        if self._after is not None:
            self.root.after_cancel(self._after)
        self._after = self.root.after(DELAY_MS, self.update_suggestions)

    def clear_suggestions(self):
        self.suggestions = []
        for b in self.sug_buttons:
            b.config(text="")

    def update_suggestions(self):
        self._after = None
        if self.model is None:
            return
        prompt = self.prompt()
        if not prompt.strip():
            self.clear_suggestions()
            return
        try:
            self.suggestions = tc.suggest(self.model, self.c2i, prompt, n_suggestions=N_SUGGESTIONS,
                                          n_chars=int(self.length.get()), temperature=float(self.temp.get()))
        except Exception as e:
            self.status.config(text=f"Suggestion error: {e}")
            return
        for i, b in enumerate(self.sug_buttons):
            b.config(text=self.suggestions[i] if i < len(self.suggestions) else "")

    def accept(self, k):
        if k >= len(self.suggestions):
            return
        self.inp.insert("insert", self.suggestions[k])
        self.inp.focus_set()
        self.clear_suggestions()
        self.update_suggestions()

    def on_tab(self, _event):
        if self.suggestions:
            self.accept(0)
        return "break"          # never insert a real tab character

    def write_more(self):
        if self.model is None:
            return
        prompt = self.prompt() or "\n"
        text = tc.generate(self.model, self.c2i, prompt, n_chars=200,
                           temperature=float(self.temp.get()))[0]
        self.inp.insert("insert", text)
        self.inp.see("insert")
        self.clear_suggestions()

    def clear(self):
        self.inp.delete("1.0", "end")
        self.clear_suggestions()
        self.inp.focus_set()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()