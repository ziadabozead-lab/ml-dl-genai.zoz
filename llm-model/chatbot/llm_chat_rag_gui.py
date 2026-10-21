"""
Chat GUI with RAG: ask questions and the model answers using YOUR documents.

How RAG works here: your .txt / .md files are cut into small chunks and indexed (BM25 keyword
search, pure Python, nothing to install). When you send a question, the best-matching chunks are
pasted into the prompt as context, so the model can answer from them instead of guessing.
RAG supplies FACTS; it does not make a model smarter at reasoning.

ONE file. Needs:  pip install torch transformers      (LoRA adapters also need: pip install peft)
    python chatbot/llm_chat_rag_gui.py            (from the llm-model folder; works from anywhere too)

Setup for RAG: put documents in a "knowledge" folder next to weights/ (or Browse to any folder),
press "Index", tick "Use documents". Sources used for each answer are listed in the chat.
Enter = send | Shift+Enter = new line | Stop = cut the reply short.

Models: the dropdown lists every model folder under weights/ (full models and LoRA adapters,
even nested ones) plus Hugging Face presets. You can also type any Hugging Face name and press Load.
"""
import json
import math
import os
import queue
import re
import threading
import tkinter as tk
from collections import Counter
from tkinter import filedialog, messagebox, ttk

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, StoppingCriteria,
                          StoppingCriteriaList)
from transformers.generation.streamers import BaseStreamer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) if os.path.basename(HERE).lower() == "chatbot" else HERE
WEIGHTS_DIR = os.path.join(ROOT, "weights")
KNOWLEDGE_DIR = os.path.join(ROOT, "knowledge")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRESETS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct"]
INTRO = "The following is a conversation between a User and a friendly, helpful AI assistant."
# keep identical to --system in scripts/get_reasoning_data.py
DEFAULT_SYSTEM = "You are a helpful assistant. Think step by step before answering."
RAG_PROMPT = ("Answer the question using the context below. If the context does not contain the answer, "
              "say you don't know.\n\nContext:\n{ctx}\n\nQuestion: {q}")


# ====================================================================== #
#                         RAG: tiny BM25 search index                    #
# ====================================================================== #
def tokenize(text):
    return re.findall(r"\w+", text.lower())


class Index:
    def __init__(self):
        self.chunks, self.tfs, self.lens, self.df = [], [], [], Counter()
        self.avg = 0.0

    @property
    def ready(self):
        return bool(self.chunks)

    def build(self, folder, chunk_words=120, overlap=30):
        self.__init__()
        step = max(1, chunk_words - overlap)
        files = 0
        for dirpath, _, names in os.walk(folder):
            for name in sorted(names):
                if not name.lower().endswith((".txt", ".md")):
                    continue
                files += 1
                with open(os.path.join(dirpath, name), "r", encoding="utf-8-sig", errors="ignore") as f:
                    words = f.read().split()
                for i in range(0, len(words), step):
                    piece = words[i:i + chunk_words]
                    if len(piece) < 20 and i > 0:
                        continue
                    text = " ".join(piece)
                    tf = Counter(tokenize(text))
                    if not tf:
                        continue
                    self.chunks.append((name, text))
                    self.tfs.append(tf)
                    self.lens.append(sum(tf.values()))
                    self.df.update(tf.keys())
        self.avg = sum(self.lens) / len(self.lens) if self.lens else 0.0
        return files, len(self.chunks)

    def search(self, query, k=3, min_score=1.0):
        terms = set(tokenize(query))
        n = len(self.chunks)
        if not n or not terms:
            return []
        k1, b = 1.5, 0.75
        idf = {t: math.log(1 + (n - self.df[t] + 0.5) / (self.df[t] + 0.5)) for t in terms if t in self.df}
        scored = []
        for i, tf in enumerate(self.tfs):
            s = 0.0
            for t, w in idf.items():
                f = tf.get(t)
                if f:
                    s += w * f * (k1 + 1) / (f + k1 * (1 - b + b * self.lens[i] / self.avg))
            if s >= min_score:
                scored.append((s, i))
        scored.sort(reverse=True)
        return [(self.chunks[i][0], self.chunks[i][1], s) for s, i in scored[:k]]


# ====================================================================== #
#                              model helpers                             #
# ====================================================================== #
def list_local_models():
    """Model folders (full models or LoRA adapters) anywhere under weights/, newest first."""
    found = []
    for dirpath, dirnames, files in os.walk(WEIGHTS_DIR):
        if "config.json" in files or "adapter_config.json" in files:
            found.append(dirpath)
            dirnames[:] = []
    found.sort(key=os.path.getmtime, reverse=True)
    return [os.path.relpath(d, WEIGHTS_DIR).replace(os.sep, "/") for d in found]


def load_any(path):
    adapter_cfg = os.path.join(path, "adapter_config.json")
    if os.path.exists(adapter_cfg):                       # LoRA adapter (from finetune_lora.py)
        from peft import PeftModel
        with open(adapter_cfg, "r", encoding="utf-8") as f:
            base_name = json.load(f)["base_model_name_or_path"]
        model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base_name), path)
        model = model.merge_and_unload()
        tok_src = path if os.path.exists(os.path.join(path, "tokenizer_config.json")) else base_name
    else:
        model = AutoModelForCausalLM.from_pretrained(path)
        tok_src = path
    tok = AutoTokenizer.from_pretrained(tok_src)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model.to(DEVICE).eval(), tok


def build_inputs(tok, history, system, use_template, budget, override=None):
    """Chat history -> model inputs. `override` replaces the text of the last user turn (RAG context)."""
    turns = list(history)                                  # always ends with a user turn
    while True:
        eff = turns[:-1] + [("user", override)] if override else turns
        if use_template:
            msgs = ([{"role": "system", "content": system}] if system.strip() else [])
            msgs += [{"role": r, "content": t} for r, t in eff]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else:
            lines = [INTRO] + [("User: " if r == "user" else "AI: ") + t for r, t in eff]
            text = "\n".join(lines) + "\nAI:"
        enc = tok(text, return_tensors="pt", add_special_tokens=not use_template)
        if enc.input_ids.shape[1] <= budget or len(turns) <= 1:
            ids = enc.input_ids[:, -budget:]
            return {"input_ids": ids.to(DEVICE), "attention_mask": torch.ones_like(ids).to(DEVICE)}
        turns = turns[2:] if len(turns) > 2 else turns[1:]


class QStreamer(BaseStreamer):
    def __init__(self, tok, q):
        self.tok, self.q = tok, q
        self.ids, self.shown, self.first = [], "", True

    def put(self, value):
        if self.first:
            self.first = False
            return
        self.ids.extend(value.view(-1).tolist())
        text = self.tok.decode(self.ids, skip_special_tokens=True)
        if text.endswith("\ufffd"):
            return
        new = text[len(self.shown):]
        self.shown = text
        if new:
            self.q.put(("tok", new))

    def end(self):
        pass


class StopFlag(StoppingCriteria):
    def __init__(self, event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full((input_ids.shape[0],), self.event.is_set(), dtype=torch.bool, device=input_ids.device)


# ====================================================================== #
#                                   GUI                                  #
# ====================================================================== #
class App:
    def __init__(self, root):
        self.root = root
        self.model = self.tok = None
        self.use_template = False
        self.history = []
        self.buffer = ""
        self.generating = self.loading = False
        self.index = Index()
        self.q = queue.Queue()
        self.stop_flag = threading.Event()

        root.title("Chat with AI + RAG")
        root.geometry("820x780")
        top = ttk.Frame(root, padding=10)
        top.pack(fill="both", expand=True)
        self.status = ttk.Label(root, text="", anchor="w", padding=(10, 2))
        self.status.pack(side="bottom", fill="x")

        # bottom stack, packed in reverse so it stays pinned to the bottom
        send_row = ttk.Frame(top)
        send_row.pack(side="bottom", fill="x", pady=(8, 0))
        self.inp = tk.Text(send_row, height=3, wrap="word", font=("Segoe UI", 12), undo=True)
        self.inp.pack(side="left", fill="x", expand=True)
        side = ttk.Frame(send_row)
        side.pack(side="left", padx=(8, 0))
        self.send_btn = ttk.Button(side, text="Send", command=self.send)
        self.send_btn.pack(fill="x")
        self.stop_btn = ttk.Button(side, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(fill="x", pady=(4, 0))
        self.inp.bind("<Return>", self.send)
        self.inp.bind("<Shift-Return>", lambda _e: None)

        sys_row = ttk.Frame(top)
        sys_row.pack(side="bottom", fill="x", pady=(10, 0))
        ttk.Label(sys_row, text="System prompt:").pack(side="left")
        self.system = tk.StringVar(value=DEFAULT_SYSTEM)
        ttk.Entry(sys_row, textvariable=self.system).pack(side="left", fill="x", expand=True, padx=6)

        # ---- RAG row ----
        rag_row = ttk.Frame(top)
        rag_row.pack(side="bottom", fill="x", pady=(10, 0))
        self.rag_on = tk.BooleanVar(value=True)
        ttk.Checkbutton(rag_row, text="Use documents (RAG)", variable=self.rag_on).pack(side="left")
        self.kfolder = tk.StringVar(value=KNOWLEDGE_DIR)
        ttk.Entry(rag_row, textvariable=self.kfolder, width=34).pack(side="left", padx=6)
        ttk.Button(rag_row, text="Browse...", command=self.pick_folder).pack(side="left")
        ttk.Button(rag_row, text="Index", command=self.build_index).pack(side="left", padx=6)
        ttk.Label(rag_row, text="Chunks:").pack(side="left", padx=(8, 2))
        self.topk = tk.IntVar(value=3)
        ttk.Spinbox(rag_row, from_=1, to=8, width=3, textvariable=self.topk).pack(side="left")
        self.index_label = ttk.Label(rag_row, text="not indexed", foreground="#666666")
        self.index_label.pack(side="left", padx=8)

        controls = ttk.Frame(top)
        controls.pack(side="bottom", fill="x", pady=(10, 0))
        ttk.Label(controls, text="Temperature:").pack(side="left")
        self.temp = tk.DoubleVar(value=0.5)
        ttk.Scale(controls, from_=0.1, to=1.3, variable=self.temp, length=140,
                  command=lambda _v: self.temp_label.config(text=f"{self.temp.get():.2f}")).pack(side="left", padx=6)
        self.temp_label = ttk.Label(controls, text="0.50", width=5)
        self.temp_label.pack(side="left")
        ttk.Label(controls, text="Max reply (tokens):").pack(side="left", padx=(14, 0))
        self.max_new = tk.IntVar(value=300)
        ttk.Spinbox(controls, from_=20, to=1000, increment=20, width=5, textvariable=self.max_new).pack(side="left", padx=6)
        ttk.Label(controls, text="Repeat penalty:").pack(side="left", padx=(8, 0))
        self.rep = tk.DoubleVar(value=1.05)       # set to 1.0 for models trained from scratch (they repeat digits)
        ttk.Spinbox(controls, from_=1.0, to=1.5, increment=0.05, width=5, textvariable=self.rep).pack(side="left", padx=6)
        ttk.Button(controls, text="New chat", command=self.new_chat).pack(side="right")

        model_row = ttk.Frame(top)
        model_row.pack(side="bottom", fill="x", pady=(10, 0))
        ttk.Label(model_row, text="Model:").pack(side="left")
        self.model_var = tk.StringVar(value=PRESETS[0])
        self.model_combo = ttk.Combobox(model_row, textvariable=self.model_var, width=42)
        self.model_combo.pack(side="left", padx=6)
        self.load_btn = ttk.Button(model_row, text="Load", command=self.load)
        self.load_btn.pack(side="left")
        ttk.Button(model_row, text="\u27f3", width=3, command=self.refresh_models).pack(side="left", padx=(6, 0))

        box = ttk.Frame(top)
        box.pack(side="top", fill="both", expand=True)
        self.chat = tk.Text(box, wrap="word", font=("Segoe UI", 12), state="disabled",
                            relief="flat", bg="#ffffff", padx=10, pady=10)
        scroll = ttk.Scrollbar(box, orient="vertical", command=self.chat.yview)
        self.chat.configure(yscrollcommand=scroll.set)
        self.chat.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.chat.tag_configure("user_name", foreground="#1565c0", font=("Segoe UI", 9, "bold"), lmargin1=60, lmargin2=60)
        self.chat.tag_configure("user_msg", background="#dbeafe", lmargin1=60, lmargin2=60, rmargin=6, spacing1=3, spacing3=3)
        self.chat.tag_configure("bot_name", foreground="#555555", font=("Segoe UI", 9, "bold"), rmargin=60)
        self.chat.tag_configure("bot_msg", background="#f0f0f0", lmargin1=6, lmargin2=6, rmargin=60, spacing1=3, spacing3=3)
        self.chat.tag_configure("note", foreground="#888888", font=("Segoe UI", 10, "italic"))

        self.refresh_models()
        local = list_local_models()
        self.model_var.set(local[0] if local else PRESETS[0])
        self.note("Loading model ...")
        if os.path.isdir(KNOWLEDGE_DIR):
            self.build_index(quiet=True)
        self.inp.focus_set()
        self.poll()
        self.root.after(300, self.load)

    # ----------------------------- transcript ----------------------------- #
    def note(self, text):
        self.chat.config(state="normal")
        self.chat.insert("end-1c", text + "\n\n", "note")
        self.chat.see("end")
        self.chat.config(state="disabled")

    def add_message(self, who, text):
        name, tag = ("You", "user") if who == "user" else ("AI", "bot")
        self.chat.config(state="normal")
        self.chat.insert("end-1c", name + "\n", f"{tag}_name")
        self.chat.insert("end-1c", text + "\n\n", f"{tag}_msg")
        self.chat.see("end")
        self.chat.config(state="disabled")

    def begin_bot(self):
        self.chat.config(state="normal")
        self.chat.insert("end-1c", "AI\n", "bot_name")
        self.chat.mark_set("bot_start", "end-1c")
        self.chat.mark_gravity("bot_start", "left")
        self.chat.insert("end-1c", "\u2026", "bot_msg")
        self.chat.see("end")
        self.chat.config(state="disabled")

    @staticmethod
    def clean(text):
        for marker in ("\nUser:", "\nAI:", "\nYou:"):
            i = text.find(marker)
            if i != -1:
                text = text[:i]
        return text.strip()

    def render_reply(self, final=False):
        text = self.clean(self.buffer) or ("(no reply)" if final else "\u2026")
        self.chat.config(state="normal")
        self.chat.delete("bot_start", "end-1c")
        self.chat.insert("end-1c", text, "bot_msg")
        if final:
            self.chat.insert("end-1c", "\n\n")
        self.chat.see("end")
        self.chat.config(state="disabled")
        return text

    # ------------------------------- RAG -------------------------------- #
    def pick_folder(self):
        d = filedialog.askdirectory(title="Folder with your .txt / .md documents", initialdir=ROOT)
        if d:
            self.kfolder.set(d)
            self.build_index()

    def build_index(self, quiet=False):
        folder = self.kfolder.get().strip()
        if not os.path.isdir(folder):
            if not quiet:
                messagebox.showinfo("No folder", f"Folder not found:\n{folder}\n\nCreate it and put .txt / .md files inside.")
            return
        self.index_label.config(text="indexing ...")
        self.root.update_idletasks()
        files, chunks = self.index.build(folder)
        self.index_label.config(text=f"{files} file(s), {chunks} chunks" if chunks else "no .txt/.md found")
        if not quiet:
            self.status.config(text=f"Indexed {files} file(s) into {chunks} chunks")

    # ------------------------------ model ------------------------------- #
    def refresh_models(self):
        self.model_combo["values"] = list_local_models() + PRESETS

    def load(self):
        if self.loading or self.generating:
            return
        name = self.model_var.get().strip()
        if not name:
            return
        local = os.path.join(WEIGHTS_DIR, name)
        path = local if os.path.isdir(local) else name
        self.loading = True
        self.load_btn.config(state="disabled")
        self.send_btn.config(state="disabled")
        self.status.config(text=f"Loading {name} ... (first time downloads it)")

        def worker():
            try:
                model, tok = load_any(path)
                self.q.put(("loaded", (model, tok, name)))
            except Exception as e:
                self.q.put(("loaderr", e))

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------- chat ------------------------------- #
    def send(self, _event=None):
        if self.generating or self.loading:
            return "break"
        if self.model is None:
            messagebox.showinfo("No model", "Pick a model and press Load first.")
            return "break"
        question = self.inp.get("1.0", "end").strip()
        if not question:
            return "break"
        self.inp.delete("1.0", "end")

        override, sources = None, []
        if self.rag_on.get() and self.index.ready:
            hits = self.index.search(question, k=int(self.topk.get()))
            if hits:
                ctx = "\n\n".join(f"[{i + 1}] ({src}) {txt}" for i, (src, txt, _s) in enumerate(hits))
                override = RAG_PROMPT.format(ctx=ctx, q=question)
                sources = sorted({src for src, _t, _s in hits})

        self.history.append(("user", question))
        self.add_message("user", question)
        if override:
            self.note("Sources: " + ", ".join(sources))
        elif self.rag_on.get() and self.index.ready:
            self.note("No matching passage found in your documents - answering without them.")
        self.begin_bot()
        self.buffer = ""
        self.generating = True
        self.stop_flag.clear()
        self.send_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status.config(text="Thinking ...")

        model, tok, use_template = self.model, self.tok, self.use_template
        history, system = list(self.history), self.system.get()
        temp, max_new = float(self.temp.get()), int(self.max_new.get())
        rep = float(self.rep.get())
        max_pos = getattr(model.config, "max_position_embeddings", 2048) or 2048
        max_new = min(max_new, max(32, max_pos // 2))             # leave room in small-context models
        budget = max(64, min(3000, max_pos - max_new - 8))

        def worker():
            try:
                inputs = build_inputs(tok, history, system, use_template, budget, override)
                kwargs = dict(
                    **inputs, max_new_tokens=max_new, do_sample=True, temperature=max(temp, 0.05),
                    top_p=0.9, top_k=50, repetition_penalty=rep if use_template else max(rep, 1.15),
                    streamer=QStreamer(tok, self.q), pad_token_id=tok.pad_token_id,
                    stopping_criteria=StoppingCriteriaList([StopFlag(self.stop_flag)]),
                )
                if not use_template:
                    kwargs.update(stop_strings=["\nUser:"], tokenizer=tok)
                with torch.no_grad():
                    model.generate(**kwargs)
            except Exception as e:
                self.q.put(("err", e))
            finally:
                self.q.put(("done", None))

        threading.Thread(target=worker, daemon=True).start()
        return "break"

    def stop(self):
        self.stop_flag.set()

    def new_chat(self):
        if self.generating:
            return
        self.history = []
        self.chat.config(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.config(state="disabled")
        self.note("New chat." if self.model else "Pick a model above and press Load.")

    # ------------------------------ plumbing ---------------------------- #
    def poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "tok":
                    self.buffer += payload
                    self.render_reply()
                    self.status.config(text="Replying ...")
                elif kind == "done":
                    reply = self.render_reply(final=True)
                    if self.buffer.strip():
                        self.history.append(("assistant", reply))
                    elif self.history:
                        self.history.pop()
                    self.generating = False
                    self.send_btn.config(state="normal")
                    self.stop_btn.config(state="disabled")
                    self.status.config(text=f"Ready  ({self.model_var.get()}, {DEVICE})")
                    self.inp.focus_set()
                elif kind == "err":
                    self.status.config(text=f"Error: {payload}")
                    self.note(f"Error: {payload}")
                elif kind == "loaded":
                    self.model, self.tok, name = payload
                    self.use_template = bool(getattr(self.tok, "chat_template", None))
                    self.loading = False
                    self.load_btn.config(state="normal")
                    self.send_btn.config(state="normal")
                    self.history = []
                    self.chat.config(state="normal")
                    self.chat.delete("1.0", "end")
                    self.chat.config(state="disabled")
                    self.note(f"Loaded {name}. Say hi!" if self.use_template else
                              f"Loaded {name}. Type something and it will continue it.")
                    self.status.config(text=f"Ready  ({name}, {DEVICE})")
                    self.refresh_models()
                    self.inp.focus_set()
                elif kind == "loaderr":
                    self.loading = False
                    self.load_btn.config(state="normal")
                    self.send_btn.config(state="normal" if self.model else "disabled")
                    self.status.config(text="Load failed")
                    messagebox.showerror("Could not load model", str(payload))
        except queue.Empty:
            pass
        self.root.after(50, self.poll)


def main():
    root = tk.Tk()
    app = App(root)

    def on_close():
        app.stop_flag.set()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
