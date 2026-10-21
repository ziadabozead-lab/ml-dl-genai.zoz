# install the required libraries
!pip -q install -U transformers accelerate bitsandbytes huggingface_hub hf_transfer gradio
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
from threading import Thread
from transformers import TextIteratorStreamer
import gradio as gr

# pull the plain text out of Gradio message content (it can be a string, a dict, or a list of dicts)
def content_to_text(c):
    if isinstance(c, str):
        return c
    if isinstance(c, dict):
        return c.get("text", "") or ""
    if isinstance(c, (list, tuple)):
        return "".join(content_to_text(x) for x in c)
    return str(c)

# convert Gradio chat history into the model's message format
def to_messages(history):
    msgs = []
    for h in history:
        if isinstance(h, dict):
            msgs.append({"role": h["role"], "content": content_to_text(h["content"])})
        else:
            u, a = h
            if u: msgs.append({"role": "user", "content": u})
            if a: msgs.append({"role": "assistant", "content": a})
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

# generate a streamed reply to the user's message
def chat(message, history, system_prompt, thinking, temperature, max_tokens):
    # build the chat with history
    msgs = []
    if system_prompt.strip():
        msgs.append({"role": "system", "content": system_prompt})
    msgs += to_messages(history)
    msgs.append({"role": "user", "content": content_to_text(message)})

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
    out = ""
    try:
        for piece in streamer:
            out += piece
            yield out
    finally:
        flag.stop = True

# define the settings controls shown in the UI
inputs_ui = [
    gr.Textbox("You are a helpful coding assistant.", label="System prompt"),
    gr.Checkbox(False, label="Thinking (much slower)"),
    gr.Slider(0, 1.5, 0.6, label="Temperature"),
    gr.Number(32768, label="Max new tokens (type a number)", precision=0, minimum=1),
]

# create the chat interface
try:
    demo = gr.ChatInterface(chat, type="messages", additional_inputs=inputs_ui, title=coder_model)
except TypeError:
    demo = gr.ChatInterface(chat, additional_inputs=inputs_ui, title=coder_model)

# launch the app with a public share link
demo.queue().launch(share=True)