"""
Chat GUI: talk to a language model. Replies stream in word by word.
Enter = send | Shift+Enter = new line | Stop = cut the reply short.

ONE file, nothing else needed:
    pip install torch transformers
    python chatbot/llm_chat_gui.py          (from the "no" folder; works from anywhere too)

Model box: pick a folder from weights/ (your fine-tuned models) OR type any Hugging Face
model name and press Load, e.g.
    Qwen/Qwen2.5-0.5B-Instruct        <- small, real chat model, fine on CPU (downloads once)
    HuggingFaceTB/SmolLM2-360M-Instruct

IMPORTANT: a model fine-tuned on plain text (like DistilGPT-2 on your .txt) is NOT a chat model.
It only continues text, so this GUI shows the conversation to it as a "User: / AI:" transcript
and it will ramble. Instruct models (names ending in -Instruct) answer properly.
"""
import os
import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, StoppingCriteria,
                          StoppingCriteriaList)
from transformers.generation.streamers import BaseStreamer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) if os.path.basename(HERE).lower() == "chatbot" else HERE
WEIGHTS_DIR = os.path.join(ROOT, "weights")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRESETS = ["Qwen/Qwen2.5-0.5B-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct"]
PROMPT_BUDGET = 700           # max prompt tokens (older turns are dropped first)
INTRO = "The following is a conversation between a User and a friendly, helpful AI assistant."
DEFAULT_SYSTEM = "You are a helpful, concise assistant."


def list_local_models():
    if not os.path.isdir(WEIGHTS_DIR):
        return []
    names = [n for n in os.listdir(WEIGHTS_DIR) if os.path.exists(os.path.join(WEIGHTS_DIR, n, "config.json"))]
    names.sort(key=lambda n: os.path.getmtime(os.path.join(WEIGHTS_DIR, n)), reverse=True)
    return names


def build_inputs(tok, history, system, use_template):
    """Turn the chat history into model inputs, dropping the oldest turns if it gets too long."""
    turns = list(history)                      # [(role, text), ...] always ends with a user turn
    while True:
        if use_template:
            msgs = ([{"role": "system", "content": system}] if system.strip() else [])
            msgs += [{"role": r, "content": t} for r, t in turns]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else:
            lines = [INTRO] + [("User: " if r == "user" else "AI: ") + t for r, t in turns]
            text = "\n".join(lines) + "\nAI:"
        enc = tok(text, return_tensors="pt", add_special_tokens=not use_template)
        if enc.input_ids.shape[1] <= PROMPT_BUDGET or len(turns) <= 1:
            ids = enc.input_ids[:, -PROMPT_BUDGET:]
            return {"input_ids": ids.to(DEVICE), "attention_mask": torch.ones_like(ids).to(DEVICE)}
        turns = turns[2:] if len(turns) > 2 else turns[1:]


class QStreamer(BaseStreamer):
    """Pushes newly generated text to the GUI queue as tokens arrive."""

    def __init__(self, tok, q):
        self.tok, self.q = tok, q
        self.ids, self.shown, self.first = [], "", True

    def put(self, value):
        if self.first:                         # the first call is the prompt itself
            self.first = False
            return
        self.ids.extend(value.view(-1).tolist())
        text = self.tok.decode(self.ids, skip_special_tokens=True)
        if text.endswith("\ufffd"):            # half-finished character, wait for the next token
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


class App:
    def __init__(self, root):
        self.root = root
        self.model = self.tok = None
        self.use_template = False
        self.history = []
        self.buffer = ""
        self.generating = False
        self.loading = False
        self.q = queue.Queue()
        self.stop_flag = threading.Event()

        root.title("Chat with AI")
        root.geometry("780x720")

        top = ttk.Frame(root, padding=10)
        top.pack(fill="both", expand=True)

        self.status = ttk.Label(root, text="", anchor="w", padding=(10, 2))
        self.status.pack(side="bottom", fill="x")

        # Bottom stack, packed in reverse so it stays pinned to the bottom of the window.
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
        self.inp.bind("<Shift-Return>", lambda _e: None)      # default behaviour: newline

        sys_row = ttk.Frame(top)
        sys_row.pack(side="bottom", fill="x", pady=(10, 0))
        ttk.Label(sys_row, text="System prompt:").pack(side="left")
        self.system = tk.StringVar(value=DEFAULT_SYSTEM)
        ttk.Entry(sys_row, textvariable=self.system).pack(side="left", fill="x", expand=True, padx=6)

        controls = ttk.Frame(top)
        controls.pack(side="bottom", fill="x", pady=(10, 0))
        ttk.Label(controls, text="Temperature:").pack(side="left")
        self.temp = tk.DoubleVar(value=0.7)
        ttk.Scale(controls, from_=0.1, to=1.3, variable=self.temp, length=140,
                  command=lambda _v: self.temp_label.config(text=f"{self.temp.get():.2f}")).pack(side="left", padx=6)
        self.temp_label = ttk.Label(controls, text="0.70", width=5)
        self.temp_label.pack(side="left")
        ttk.Label(controls, text="Max reply (tokens):").pack(side="left", padx=(14, 0))
        self.max_new = tk.IntVar(value=200)
        ttk.Spinbox(controls, from_=20, to=600, increment=20, width=5, textvariable=self.max_new).pack(side="left", padx=6)
        ttk.Button(controls, text="New chat", command=self.new_chat).pack(side="right")

        model_row = ttk.Frame(top)
        model_row.pack(side="bottom", fill="x", pady=(10, 0))
        ttk.Label(model_row, text="Model:").pack(side="left")
        self.model_var = tk.StringVar(value=PRESETS[0])
        self.model_combo = ttk.Combobox(model_row, textvariable=self.model_var, width=38)
        self.model_combo.pack(side="left", padx=6)
        self.load_btn = ttk.Button(model_row, text="Load", command=self.load)
        self.load_btn.pack(side="left")
        ttk.Button(model_row, text="\u27f3", width=3, command=self.refresh_models).pack(side="left", padx=(6, 0))

        # Chat transcript fills the middle.
        box = ttk.Frame(top)
        box.pack(side="top", fill="both", expand=True)
        self.chat = tk.Text(box, wrap="word", font=("Segoe UI", 12), state="disabled",
                            relief="flat", bg="#ffffff", padx=10, pady=10)
        scroll = ttk.Scrollbar(box, orient="vertical", command=self.chat.yview)
        self.chat.configure(yscrollcommand=scroll.set)
        self.chat.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.chat.tag_configure("user_name", foreground="#1565c0", font=("Segoe UI", 9, "bold"), lmargin1=60, lmargin2=60)
        self.chat.tag_configure("user_msg", background="#dbeafe", lmargin1=60, lmargin2=60, rmargin=6,
                                spacing1=3, spacing3=3)
        self.chat.tag_configure("bot_name", foreground="#555555", font=("Segoe UI", 9, "bold"), rmargin=60)
        self.chat.tag_configure("bot_msg", background="#f0f0f0", lmargin1=6, lmargin2=6, rmargin=60,
                                spacing1=3, spacing3=3)
        self.chat.tag_configure("note", foreground="#888888", font=("Segoe UI", 10, "italic"))

        self.refresh_models()
        local = list_local_models()
        self.model_var.set(local[0] if local else PRESETS[0])      # newest model of yours, else Qwen
        self.note("Loading model ...")
        self.inp.focus_set()
        self.poll()
        self.root.after(300, self.load)      # load automatically so the chat is ready to use

    # --------------------------- transcript helpers --------------------- #
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

    # ------------------------------ model ------------------------------ #
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
                tok = AutoTokenizer.from_pretrained(path)
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                model = AutoModelForCausalLM.from_pretrained(path).to(DEVICE).eval()
                self.q.put(("loaded", (model, tok, name)))
            except Exception as e:
                self.q.put(("loaderr", e))

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------- chat ------------------------------ #
    def send(self, _event=None):
        if self.generating or self.loading:
            return "break"
        if self.model is None:
            messagebox.showinfo("No model", "Pick a model and press Load first.")
            return "break"
        text = self.inp.get("1.0", "end").strip()
        if not text:
            return "break"
        self.inp.delete("1.0", "end")
        self.history.append(("user", text))
        self.add_message("user", text)
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

        def worker():
            try:
                inputs = build_inputs(tok, history, system, use_template)
                kwargs = dict(
                    **inputs, max_new_tokens=max_new, do_sample=True, temperature=max(temp, 0.05),
                    top_p=0.9, top_k=50, repetition_penalty=1.05 if use_template else 1.15,
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

    # ------------------------------ plumbing --------------------------- #
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
                    else:
                        self.history.pop()          # no answer: forget the unanswered question
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
                    if self.use_template:
                        self.note(f"Loaded {name}. Say hi!")
                    else:
                        self.note(f"Loaded {name}. Type something and it will continue it.")
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