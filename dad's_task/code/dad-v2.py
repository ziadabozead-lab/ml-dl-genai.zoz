# install the required libraries
!pip -q install -U transformers accelerate bitsandbytes huggingface_hub hf_transfer gradio pypdf
!pip -q install -U flash-linear-attention

# our model
coder_model = "Qwen/Qwen3.8-27B"  # if you want to use a different model, change this to the model ID on Hugging Face

# set the cache location and enable fast downloads
import os, shutil
os.environ["HF_HOME"] = "/tmp/hf_cache" # it can be /kaggle/working/ if you wanna download it and keep it for the next session
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download, login

# log in to Hugging Face with the Kaggle secret
try:
    from kaggle_secrets import UserSecretsClient
    login(UserSecretsClient().get_secret("HF_TOKEN"))
    print("Logged in to Hugging Face")
except Exception as e:
    print("No HF_TOKEN secret found, continuing anonymously:", e)

# check that there is enough free disk space
free_gb = shutil.disk_usage("/tmp").free / 1e9
print(f"Free disk in /tmp: {free_gb:.0f} GB")

# download the model files
MODEL_DIR = snapshot_download(
    coder_model,
    cache_dir="/tmp/hf_cache", # it can be /kaggle/working/ if you wanna download it and keep it for the next session
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
)
print(MODEL_DIR)

# load the tokenizer
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

tok = AutoTokenizer.from_pretrained(MODEL_DIR)

# configure 4-bit quantization for the T4 GPUs
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_skip_modules=["visual", "lm_head"],
)

# set how the model is split across both GPUs
kwargs = dict(
    quantization_config=bnb,
    torch_dtype=torch.float16,
    device_map="auto",
    max_memory={0: "14GiB", 1: "14GiB", "cpu": "20GiB"},
    low_cpu_mem_usage=True,
)

# load the model, falling back to the text-only loader if needed
try:
    from transformers import AutoModelForMultimodalLM
    model = AutoModelForMultimodalLM.from_pretrained(MODEL_DIR, **kwargs)
except Exception as e:
    print("Multimodal loader failed, trying causal LM:", e)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, **kwargs)

# switch to inference mode and print GPU memory use
model.eval()
print({k: round(v / 1e9, 1) for k, v in
       {f"gpu{i}": torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())}.items()}, "GB used")

# import the chat UI and streaming tools
import re, time
from threading import Thread
from transformers import TextIteratorStreamer
import gradio as gr

# the Gradio major version (file cards inside the chat need version 5 or newer)
try:
    GRADIO_MAJOR = int(gr.__version__.split(".")[0])
except Exception:
    GRADIO_MAJOR = 0

# ---------- file helpers: start ----------

# storage folders: uploaded files and files created by the model are kept in 2 separate folders
STORAGE_DIR = "/kaggle/working/storage" if os.path.isdir("/kaggle/working") else os.path.abspath("storage")
UPLOAD_DIR = os.path.join(STORAGE_DIR, "uploads")
OUTPUT_DIR = os.path.join(STORAGE_DIR, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# limits on how much uploaded text is given to the model (characters)
TEXT_LIMIT = 30000   # per file
TOTAL_LIMIT = 80000  # all files together

# make a file name safe (no folders, no strange characters)
def safe_name(name):
    name = os.path.basename(str(name).replace("\\", "/")).strip()
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip(". ")
    return name[-120:] or "file.txt"

# find a free path in a folder, adding _1, _2 ... if the name already exists
def unique_path(folder, name):
    base, ext = os.path.splitext(safe_name(name))
    path = os.path.join(folder, base + ext)
    i = 1
    while os.path.exists(path):
        path = os.path.join(folder, f"{base}_{i}{ext}")
        i += 1
    return path

# list the files in a folder, newest first (used by the download panels)
def list_files(folder):
    paths = [os.path.join(folder, f) for f in os.listdir(folder)]
    paths = [p for p in paths if os.path.isfile(p)]
    return sorted(paths, key=os.path.getmtime, reverse=True)

# read an uploaded file as text (returns None for binary or unreadable files)
def read_file_text(path):
    try:
        if path.lower().endswith(".pdf"):
            from pypdf import PdfReader
            text = "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
        else:
            with open(path, "rb") as f:
                raw = f.read(2_000_000)
            if b"\x00" in raw[:4096]:
                return None
            text = raw.decode("utf-8", errors="replace")
    except Exception:
        return None
    if len(text) > TEXT_LIMIT:
        text = text[:TEXT_LIMIT] + "\n...[truncated]"
    return text

# uploaded files of this session: name -> text (or None if the content can't be read)
UPLOADED = {}

# copy uploaded files into the uploads folder and remember their content
def save_uploads(files):
    names = []
    for f in files or []:
        src = f.get("path") if isinstance(f, dict) else f
        if not src or not os.path.isfile(src):
            continue
        dst = unique_path(UPLOAD_DIR, os.path.basename(src))
        shutil.copy(src, dst)
        UPLOADED[os.path.basename(dst)] = read_file_text(dst)
        names.append(os.path.basename(dst))
    return names

# build the text that gives the model the content of the uploaded files
def files_context():
    if not UPLOADED:
        return ""
    parts = ["The user has uploaded these files:"]
    total = 0
    for name, text in UPLOADED.items():
        if text is None:
            parts.append(f"\n### {name}\n(binary or unreadable file, its content is not available)")
        elif total + len(text) > TOTAL_LIMIT:
            parts.append(f"\n### {name}\n(skipped: too much file content is already included)")
        else:
            total += len(text)
            parts.append(f"\n### {name}\n```\n{text}\n```")
    return "\n".join(parts)

# tell the model how to hand back a file
FILE_RULES = (
    "When the user asks you to create a file (code, script, text, csv, json, markdown, etc.), "
    "write the complete file in exactly this format:\n"
    '<file name="example.py">\nfile content here\n</file>\n'
    "Never put the file tag inside a code block. "
    "For short snippets that the user only wants to read, use a normal markdown code block instead."
)

# patterns that find file blocks in the model's reply
FILE_RE = re.compile(r"<file\s+name\s*=\s*[\"']([^\"']+)[\"']\s*>\n?(.*?)\n?</file>", re.S)
OPEN_RE = re.compile(r"<file\s+name\s*=\s*[\"']([^\"']+)[\"']\s*>\n?(.*)$", re.S)

# remove a ``` fence the model may have put around the file content
def strip_fence(s):
    m = re.match(r"^\s*```[^\n]*\n(.*?)\n?```\s*$", s, re.S)
    return m.group(1) if m else s

# turn the model's reply into chat text: file blocks become code previews, and are saved if save=True
def render(text, save=False):
    saved = []
    if save and text.count("<file") > text.count("</file>"):
        text += "\n</file>"  # the reply was cut off in the middle of a file, keep what we have

    def done(m):
        name, body = m.group(1), strip_fence(m.group(2))
        shown = safe_name(name)
        if save:
            path = unique_path(OUTPUT_DIR, name)
            with open(path, "w", encoding="utf-8") as f:
                f.write(body)
            shown = os.path.basename(path)
            saved.append(shown)
        lang = os.path.splitext(shown)[1].lstrip(".")
        return f"\n📎 **{shown}**\n````{lang}\n{body}\n````\n"

    text = FILE_RE.sub(done, text)
    m = OPEN_RE.search(text)  # a file that is still being written while streaming
    if m:
        text = text[:m.start()] + f"\n📎 **{safe_name(m.group(1))}** (writing...)\n````\n{m.group(2)}"
    return text, saved

# ---------- file helpers: end ----------

# pull the plain text out of Gradio message content (string, dict, list of dicts, or a file entry)
def content_to_text(c):
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, dict):
        return (c.get("text") or "") if "text" in c else ""  # file entries have no text
    if isinstance(c, tuple):
        return ""  # a file entry in the older Gradio format
    if isinstance(c, list):
        return "".join(content_to_text(x) for x in c)
    return ""

# remove the "saved" line and the speed line we add to replies, so the model never sees them in the history
def clean_reply(text):
    text = re.sub(r"\n*✅ Saved to outputs:.*$", "", text, flags=re.S)
    text = re.sub(r"\n*\*⏱.*\*\s*$", "", text)
    return text

# convert Gradio chat history into the model's message format
def to_messages(history):
    msgs = []
    for h in history:
        if isinstance(h, dict):
            text = content_to_text(h["content"])
            if h["role"] == "assistant":
                text = clean_reply(text)
            if text.strip():
                msgs.append({"role": h["role"], "content": text})
        else:
            u, a = h
            if content_to_text(u).strip(): msgs.append({"role": "user", "content": content_to_text(u)})
            if clean_reply(content_to_text(a)).strip(): msgs.append({"role": "assistant", "content": clean_reply(content_to_text(a))})
    return msgs

# let the reply be cut off when the user presses stop or closes the page
from transformers import StoppingCriteria, StoppingCriteriaList

class StopFlag(StoppingCriteria):
    def __init__(self):
        self.stop = False
    def __call__(self, input_ids, scores, **kwargs):
        return self.stop

# run generation in the background and show any error in the chat instead of hanging
def run_generate(gen_kwargs, streamer):
    try:
        model.generate(**gen_kwargs)
    except Exception as e:
        import traceback
        traceback.print_exc()
        streamer.on_finalized_text(f"\n\n[Generation error: {type(e).__name__}: {e}]", stream_end=True)

# generate a streamed reply to the user's message (the message can include uploaded files)
def chat(message, history, system_prompt, thinking, temperature, max_tokens):
    # split the message into text and files, and store the files in the uploads folder
    if isinstance(message, dict):
        user_text = message.get("text") or ""
        new_files = save_uploads(message.get("files"))
    else:
        user_text, new_files = content_to_text(message), []
    if new_files:
        user_text = f"[uploaded: {', '.join(new_files)}]\n" + (user_text or "Please look at the uploaded file(s).")

    # build the chat: system prompt + file rules + uploaded file content, then history, then the new message
    system = "\n\n".join(p for p in [system_prompt.strip(), FILE_RULES, files_context()] if p)
    msgs = [{"role": "system", "content": system}]
    msgs += to_messages(history)
    msgs.append({"role": "user", "content": user_text})

    # turn the conversation into model input
    text = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=bool(thinking)
    )
    inputs = tok(text, return_tensors="pt").to(model.device)
    inputs = {k: v for k, v in inputs.items() if k != "token_type_ids"}

    # start generation in a background thread
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    flag = StopFlag()
    gen_kwargs = dict(
        **inputs, streamer=streamer, max_new_tokens=max(1, int(max_tokens or 32768)),
        do_sample=temperature > 0, temperature=max(temperature, 0.01), top_p=0.95,
        stopping_criteria=StoppingCriteriaList([flag]),
    )
    Thread(target=run_generate, args=(gen_kwargs, streamer), daemon=True).start()

    # stream the reply back as it is produced, and stop generating if the user leaves
    out, first = "", None
    t0 = time.time()
    try:
        for piece in streamer:
            if first is None:
                first = time.time() - t0
            out += piece
            yield render(out)[0]
        # the reply is finished: save any files the model created
        final, saved = render(out, save=True)
        if saved:
            final += "\n\n✅ Saved to outputs: " + ", ".join(saved)
        # add a small speed line (first token delay, total time, tokens per second)
        secs = time.time() - t0
        n_tokens = len(tok(out, add_special_tokens=False).input_ids)
        first = first or 0.0
        final += f"\n\n*⏱ {n_tokens} tokens · {secs:.1f}s total · first token after {first:.1f}s · {n_tokens / max(secs - first, 0.01):.1f} tokens/s*"
        # files the model created are shown as file cards in the chat (needs Gradio 5 or newer)
        if saved and GRADIO_MAJOR >= 5:
            yield [{"role": "assistant", "content": final}] + [
                {"role": "assistant", "content": {"path": os.path.join(OUTPUT_DIR, n)}} for n in saved
            ]
        else:
            yield final
    finally:
        flag.stop = True

# define the settings controls shown in the UI
inputs_ui = [
    gr.Textbox("You are a helpful coding assistant. Keep answers short and direct unless the user asks for detail.", label="System prompt"),
    gr.Checkbox(False, label="Thinking (much slower)"),
    gr.Slider(0, 1.5, 0.6, label="Temperature"),
    gr.Number(32768, label="Max new tokens (type a number)", precision=0, minimum=1),
]

# refresh the two download panels
def refresh_lists():
    return list_files(OUTPUT_DIR), list_files(UPLOAD_DIR)

# make the model forget the uploaded files (they stay in the uploads folder)
def forget_uploads():
    UPLOADED.clear()
    gr.Info("The model no longer sees the uploaded files.")

# build the chat window: tall, like a normal chat app
def make_chatbot():
    # render=False: the chat interface draws it itself, otherwise it shows up twice
    try:
        return gr.Chatbot(type="messages", height="72vh", render=False)
    except TypeError:
        return gr.Chatbot(height="72vh", render=False)

# create the chat interface (the paperclip uploads files, created files appear as cards in the chat)
with gr.Blocks(title=coder_model) as demo:
    gr.Markdown(f"**{coder_model}**")
    try:
        chat_ui = gr.ChatInterface(chat, chatbot=make_chatbot(), multimodal=True, type="messages", additional_inputs=inputs_ui)
    except TypeError:
        chat_ui = gr.ChatInterface(chat, chatbot=make_chatbot(), multimodal=True, additional_inputs=inputs_ui)

    # all stored files live in a small closed section, so they take almost no space
    with gr.Accordion("📁 Files", open=False):
        with gr.Row():
            out_files = gr.File(label="Outputs (created by the model)", file_count="multiple", interactive=False, height=120)
            up_files = gr.File(label="Uploads (sent by you)", file_count="multiple", interactive=False, height=120)
        with gr.Row():
            refresh_btn = gr.Button("Refresh", size="sm")
            forget_btn = gr.Button("Forget uploaded files", size="sm")

    refresh_btn.click(refresh_lists, None, [out_files, up_files])
    forget_btn.click(forget_uploads, None, None)
    demo.load(refresh_lists, None, [out_files, up_files])
    try:
        chat_ui.chatbot.change(refresh_lists, None, [out_files, up_files], show_progress="hidden")
    except Exception as e:
        print("Auto refresh is not available, use the Refresh button:", e)

# launch the app with a public share link
demo.queue().launch(share=True, allowed_paths=[STORAGE_DIR])