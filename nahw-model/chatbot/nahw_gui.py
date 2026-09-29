# -*- coding: utf-8 -*-
"""
Nahw Tagger GUI: type or paste a fully-diacritized (مُشكّل) Arabic sentence,
press Enter, see each word's grammatical role (مبتدأ، خبر، فاعل، مفعول به...).

IMPORTANT: the model expects diacritics (تشكيل). Undiacritized text will
tag much worse - the case ending IS most of the signal it learned from.
It also can't split attached prefixes (و العطف / ب-ل-ك الجر): a word like
"والمعلمُ" is one surface token and will be tagged as a single word, not as
"و" + "المعلمُ".

Project layout this script expects:
    no/
      chatbot/nahw_gui.py             <- this file
      scripts/train_nahw.py           <- the training script it imports
      weights/nahw_model.pt           <- trained weights

Run from the "no" folder:
    python chatbot/nahw_gui.py

Enter = tag the sentence | Shift+Enter = new line
"""
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import __main__

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the "no" folder
sys.path.insert(0, os.path.join(ROOT, "scripts"))                    # so "import train_nahw" finds it
import train_nahw as tn  # noqa: E402

# nahw_model.pt was saved by a script run as __main__, so the pickled Vocab
# points at __main__.Vocab. Point that name at the real class.
__main__.Vocab = tn.Vocab

WEIGHTS_DIR = os.path.join(ROOT, "weights")
DEFAULT_WEIGHTS = os.path.join(WEIGHTS_DIR, "nahw_model.pt")

# Tags grouped by grammatical family so related roles share a color family.
MEANING = {
    "مبتدأ": "مبتدأ (مرفوع)", "خبر": "خبر (مرفوع)",
    "فاعل": "فاعل (مرفوع)",
    "مفعول_به": "مفعول به (منصوب)", "مفعول_فيه_ظرف": "مفعول فيه / ظرف (منصوب)",
    "مفعول_لأجله": "مفعول لأجله (منصوب)",
    "حال": "حال (منصوب)", "تمييز": "تمييز (منصوب)",
    "نعت_منصوب": "نعت (تابع)", "توكيد_مرفوع": "توكيد (تابع)", "معطوف_مرفوع": "معطوف (تابع)",
    "مضاف": "مضاف", "مضاف_إليه": "مضاف إليه (مجرور)",
    "اسم_مجرور": "اسم مجرور", "حرف_جر": "حرف جر (مبني)",
    "حرف_عطف": "حرف عطف (مبني)", "حرف_نصب": "حرف نصب (مبني)",
    "حرف_نفي_جازم": "حرف جزم/نفي (مبني)",
    "فعل_ماض": "فعل ماضٍ (مبني)",
    "فعل_مضارع_مرفوع": "فعل مضارع مرفوع", "فعل_مضارع_منصوب": "فعل مضارع منصوب",
    "فعل_مضارع_مجزوم": "فعل مضارع مجزوم",
}
COLOR = {
    "مبتدأ": "#1565c0", "خبر": "#1976d2",
    "فاعل": "#2e7d32",
    "مفعول_به": "#c62828", "مفعول_فيه_ظرف": "#d84315", "مفعول_لأجله": "#ad1457",
    "حال": "#ef6c00", "تمييز": "#f9a825",
    "نعت_منصوب": "#6a1b9a", "توكيد_مرفوع": "#8e24aa", "معطوف_مرفوع": "#7b1fa2",
    "مضاف": "#00695c", "مضاف_إليه": "#00838f",
    "اسم_مجرور": "#4e342e", "حرف_جر": "#795548",
    "حرف_عطف": "#616161", "حرف_نصب": "#757575", "حرف_نفي_جازم": "#9e9d24",
    "فعل_ماض": "#b71c1c",
    "فعل_مضارع_مرفوع": "#e65100", "فعل_مضارع_منصوب": "#e64a19", "فعل_مضارع_مجزوم": "#bf360c",
}
FALLBACK_COLOR = "#000000"
ARABIC_FONT = ("Tahoma", 13)  # decent Arabic glyph coverage cross-platform


class App:
    def __init__(self, root):
        self.root = root
        self.model = None
        self.vocab = None
        root.title("محلل النحو - Nahw Tagger")
        root.geometry("820x680")

        top = ttk.Frame(root, padding=10)
        top.pack(fill="x")
        ttk.Label(top, text="اكتب أو الصق جملة مُشكّلة (بالتشكيل الكامل)، ثم اضغط Enter:",
                 font=ARABIC_FONT).pack(anchor="e", fill="x")
        self.inp = tk.Text(top, height=3, wrap="word", font=ARABIC_FONT)
        self.inp.pack(fill="x", pady=(4, 6))
        self.inp.bind("<Return>", self.on_enter)
        self.inp.focus_set()

        btns = ttk.Frame(top)
        btns.pack(fill="x")
        ttk.Button(btns, text="حلّل الجملة", command=self.run).pack(side="left")
        ttk.Button(btns, text="مسح", command=self.clear).pack(side="left", padx=6)
        ttk.Button(btns, text="Load weights...", command=self.pick_model).pack(side="right")

        # colored sentence view
        ttk.Label(root, text="النتيجة:", padding=(10, 4, 0, 0)).pack(anchor="e")
        self.view = tk.Text(root, height=4, wrap="word", font=(ARABIC_FONT[0], 15, "bold"),
                            state="disabled", relief="flat", bg="#f4f4f4", padx=8, pady=8)
        self.view.pack(fill="x", padx=10)
        for tag, color in COLOR.items():
            self.view.tag_configure(tag, foreground=color)

        # table
        table = ttk.Frame(root, padding=10)
        table.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(table, columns=("word", "tag", "meaning"),
                                 show="headings", height=10)
        for col, title, w in (("word", "الكلمة", 200), ("tag", "الوسم", 160), ("meaning", "المعنى", 260)):
            self.tree.heading(col, text=title)
            self.tree.column(col, width=w, anchor="e")
        for tag, color in COLOR.items():
            self.tree.tag_configure(tag, foreground=color)
        scroll = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        # legend (wraps across a couple of rows)
        legend = ttk.Frame(root, padding=(10, 0, 10, 4))
        legend.pack(fill="x")
        row = ttk.Frame(legend)
        row.pack(fill="x")
        for i, (tag, color) in enumerate(COLOR.items()):
            if i and i % 6 == 0:
                row = ttk.Frame(legend)
                row.pack(fill="x")
            tk.Label(row, text=tag, fg=color, font=("Tahoma", 8)).pack(side="left", padx=4)

        self.status = ttk.Label(root, text="No weights loaded", anchor="w", padding=(10, 2))
        self.status.pack(fill="x")

    # ------------------------------ model ------------------------------ #
    def load_model(self, path):
        try:
            self.model, self.vocab = tn.load_model(path)
        except Exception as e:
            messagebox.showerror("Could not load weights", f"{path}\n\n{e}")
            return False
        self.status.config(text=f"Loaded: {path}  ({len(self.vocab.t2i)} tags)")
        return True

    def pick_model(self):
        path = filedialog.askopenfilename(title="Choose nahw_model.pt", initialdir=WEIGHTS_DIR,
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
        sentence = self.inp.get("1.0", "end").strip()
        if not sentence:
            return
        if self.model is None:
            messagebox.showwarning("No weights", "Load nahw_model.pt first.")
            return
        pairs = tn.tag_sentence(self.model, self.vocab, sentence)
        if not pairs:
            messagebox.showinfo("No words", "لم يتم العثور على كلمات صالحة.")
            return
        self.show(pairs)
        self.status.config(text=f"تم تحليل {len(pairs)} كلمة")

    def show(self, pairs):
        self.view.config(state="normal")
        self.view.delete("1.0", "end")
        for i, (word, tag) in enumerate(pairs):
            if i:
                self.view.insert("end", "   ")
            self.view.insert("end", word, (tag,) if tag in COLOR else ())
        self.view.config(state="disabled")

        self.tree.delete(*self.tree.get_children())
        for word, tag in pairs:
            self.tree.insert("", "end", values=(word, tag, MEANING.get(tag, tag)), tags=(tag,))

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
    if os.path.exists(DEFAULT_WEIGHTS):
        app.load_model(DEFAULT_WEIGHTS)
    else:
        app.pick_model()
    root.mainloop()


if __name__ == "__main__":
    main()
