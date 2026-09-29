"""
Image Classifier GUI: pick a photo, see which of the 10 CIFAR-10 classes it
thinks it is, with a confidence bar and a top-3 breakdown.

Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

HONEST SCOPE NOTE: this model was trained on tiny 32x32 CIFAR-10 images.
It works best on simple photos with one centered subject and a plain-ish
background - it will struggle on busy real-world scenes, unusual angles,
or anything outside its 10 classes (it will still confidently pick one of
the 10, since that's all it knows how to output).

Project layout this script expects:
    no/
      chatbot/imgclf_gui.py           <- this file
      scripts/train_imgclf.py         <- the training script it imports
      weights/imgclf_model.pt         <- trained weights

Run from the "no" folder:
    python chatbot/imgclf_gui.py

Requires: pip install torchvision pillow
"""
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the "no" folder
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import train_imgclf as ti  # noqa: E402

WEIGHTS_DIR = os.path.join(ROOT, "weights")
DEFAULT_WEIGHTS = os.path.join(WEIGHTS_DIR, "imgclf_model.pt")
THUMB_SIZE = 260

CLASS_AR = {  # optional Arabic labels alongside the English class names
    "airplane": "طائرة", "automobile": "سيارة", "bird": "طائر", "cat": "قطة",
    "deer": "غزال", "dog": "كلب", "frog": "ضفدع", "horse": "حصان",
    "ship": "سفينة", "truck": "شاحنة",
}


class App:
    def __init__(self, root):
        self.root = root
        self.model = None
        self.classes = None
        self.current_path = None
        root.title("Image Classifier")
        root.geometry("760x620")

        top = ttk.Frame(root, padding=10)
        top.pack(fill="x")
        ttk.Label(top, text="Pick a photo to classify:").pack(anchor="w")
        btns = ttk.Frame(top)
        btns.pack(fill="x", pady=(6, 0))
        ttk.Button(btns, text="Choose image...", command=self.pick_image).pack(side="left")
        ttk.Button(btns, text="Load weights...", command=self.pick_model).pack(side="right")

        # image preview + verdict side by side
        mid = ttk.Frame(root, padding=10)
        mid.pack(fill="both", expand=True)

        preview_frame = tk.Frame(mid, width=THUMB_SIZE, height=THUMB_SIZE, bg="#e0e0e0")
        preview_frame.pack(side="left", padx=(0, 16))
        preview_frame.pack_propagate(False)
        self.preview = tk.Label(preview_frame, text="No image", bg="#e0e0e0")
        self.preview.pack(fill="both", expand=True)

        panel = tk.Frame(mid, bg="#f4f4f4", padx=12, pady=12)
        panel.pack(side="left", fill="both", expand=True)
        self.verdict = tk.Label(panel, text="-", font=("Segoe UI", 26, "bold"),
                                bg="#f4f4f4", fg="#616161")
        self.verdict.pack(anchor="w")
        self.detail = tk.Label(panel, text="Choose an image to classify", font=("Segoe UI", 11),
                               bg="#f4f4f4", fg="#444444")
        self.detail.pack(anchor="w", pady=(2, 10))

        self.bar_rows = []
        for i in range(3):
            row = tk.Frame(panel, bg="#f4f4f4")
            row.pack(fill="x", pady=3)
            name = tk.Label(row, text="", width=14, anchor="w", bg="#f4f4f4", font=("Segoe UI", 10))
            name.pack(side="left")
            bar = ttk.Progressbar(row, maximum=100, length=180)
            bar.pack(side="left", padx=8)
            pct = tk.Label(row, text="", width=6, anchor="e", bg="#f4f4f4", font=("Segoe UI", 10))
            pct.pack(side="left")
            self.bar_rows.append((name, bar, pct))

        self.status = ttk.Label(root, text="No weights loaded", anchor="w", padding=(10, 2))
        self.status.pack(fill="x")

    # ------------------------------ model ------------------------------ #
    def load_model(self, path):
        try:
            self.model, self.classes = ti.load_model(path)
        except Exception as e:
            messagebox.showerror("Could not load weights", f"{path}\n\n{e}")
            return False
        self.status.config(text=f"Loaded: {path}  ({len(self.classes)} classes)")
        return True

    def pick_model(self):
        path = filedialog.askopenfilename(title="Choose imgclf_model.pt", initialdir=WEIGHTS_DIR,
                                          filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")])
        if path and self.load_model(path) and self.current_path:
            self.classify(self.current_path)

    # ------------------------------- UI -------------------------------- #
    def pick_image(self):
        path = filedialog.askopenfilename(
            title="Choose a photo",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.gif"), ("All files", "*.*")])
        if not path:
            return
        self.current_path = path
        self.show_preview(path)
        if self.model is None:
            messagebox.showwarning("No weights", "Load imgclf_model.pt first.")
            return
        self.classify(path)

    def show_preview(self, path):
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Could not open image", f"{path}\n\n{e}")
            return
        img.thumbnail((THUMB_SIZE, THUMB_SIZE))
        self._thumb = ImageTk.PhotoImage(img)   # keep a reference, tkinter needs it
        self.preview.config(image=self._thumb, text="")

    def classify(self, path):
        try:
            ranked = ti.predict_image(self.model, self.classes, path, top_k=3)
        except Exception as e:
            messagebox.showerror("Could not classify image", f"{e}")
            return
        top_name, top_p = ranked[0]
        label_ar = CLASS_AR.get(top_name, "")
        self.verdict.config(text=top_name.upper())
        self.detail.config(text=f"{top_p:.0%} confident" + (f"  ({label_ar})" if label_ar else ""))
        for (name_lbl, bar, pct_lbl), (cname, p) in zip(self.bar_rows, ranked):
            name_lbl.config(text=cname)
            bar["value"] = p * 100
            pct_lbl.config(text=f"{p:.0%}")
        for i in range(len(ranked), len(self.bar_rows)):
            name_lbl, bar, pct_lbl = self.bar_rows[i]
            name_lbl.config(text="")
            bar["value"] = 0
            pct_lbl.config(text="")
        self.status.config(text=f"Classified: {os.path.basename(path)}")


def main():
    root = tk.Tk()
    app = App(root)
    if os.path.exists(DEFAULT_WEIGHTS):
        app.load_model(DEFAULT_WEIGHTS)
    else:
        app.status.config(text="No weights yet - train first: python scripts/train_imgclf.py")
    root.mainloop()


if __name__ == "__main__":
    main()
