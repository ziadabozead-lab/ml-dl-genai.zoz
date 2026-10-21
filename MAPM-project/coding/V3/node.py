# node_v3.py - MAPM node, "short hop" version.
# Paste into ONE cell of each Kaggle notebook (set ROLE first) and run it.
# Notebook settings: Accelerator = GPU T4 x2, Internet = On,
# secrets HF_TOKEN and TUNNEL_PASSWORD attached.
#
# How one token travels (every request is answered "ok" at once; the work runs
# in a background thread, so no request stays open while a model computes):
#   device -> N1 /start     N1 runs the embedding + its layers
#   N1 -> N2 /step -> N3 /step   activations hop forward
#   N3 picks the token, keeps the text, and calls N1 /token (callback)
#   N1 feeds that token back in; repeat until N3 says done
#   device -> N3 /poll      the device reads the text from notebook 3
# You can paste the whole file in one cell, or split it at the CHUNK lines
# (run the chunks in order; the last chunk never returns).

# ================================================================ CHUNK 1: settings
import os
os.environ["HF_HOME"] = "/tmp/hf"              # big scratch disk; NOT counted in the 20 GB /kaggle/working limit

ROLE = 0                                # 0 = first, 1 = middle, 2 = last
N_NODES = 3
MODEL = "Qwen/Qwen2.5-14B-Instruct"     # 48 layers -> 16 per notebook
PORT = 8000
MAX_SESSIONS = 8                        # KV caches kept per node (oldest evicted)

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "fastapi", "uvicorn", "requests"], check=True)

import re, gc, json, time, hmac, uuid, asyncio, inspect, threading, traceback
import urllib.request
from collections import OrderedDict
import numpy as np, requests, torch, uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from starlette.concurrency import run_in_threadpool
from kaggle_secrets import UserSecretsClient
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import (AutoModelForCausalLM, AutoTokenizer, AutoConfig,
                          GenerationConfig, DynamicCache)

secrets = UserSecretsClient()
PASSWORD = secrets.get_secret("TUNNEL_PASSWORD")
HF_TOKEN = secrets.get_secret("HF_TOKEN")

# ================================================================ CHUNK 2: load my layers
tok = AutoTokenizer.from_pretrained(MODEL, token=HF_TOKEN)
cfg = AutoConfig.from_pretrained(MODEL, token=HF_TOKEN)
n_layers = cfg.num_hidden_layers
per = n_layers // N_NODES
start = ROLE * per
end = n_layers if ROLE == N_NODES - 1 else start + per

# empty skeleton on the meta device: costs no RAM and no VRAM
torch.set_default_dtype(torch.float16)
with torch.device("meta"):
    full = AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa")
torch.set_default_dtype(torch.float32)
full = full.to(torch.float16)
full.eval()
inner = full.model

# which file holds which tensor (sharded repos have an index, small ones do not)
try:
    FILE_OF = json.load(open(hf_hub_download(
        MODEL, "model.safetensors.index.json", token=HF_TOKEN)))["weight_map"]
except Exception:
    _p = hf_hub_download(MODEL, "model.safetensors", token=HF_TOKEN)
    with safe_open(_p, framework="pt") as _f:
        FILE_OF = {k: "model.safetensors" for k in _f.keys()}


def load_into(module, prefix, device):
    """Fill `module` from the checkpoint tensors whose name starts with `prefix`."""
    names = [n for n in FILE_OF if n.startswith(prefix)]
    sd = {}
    for fname in sorted({FILE_OF[n] for n in names}):
        path = hf_hub_download(MODEL, fname, token=HF_TOKEN)   # only shards I need
        with safe_open(path, framework="pt") as f:
            for n in names:
                if FILE_OF[n] == fname:
                    sd[n[len(prefix):]] = f.get_tensor(n).to(device=device,
                                                             dtype=torch.float16)
    module.to_empty(device=device)
    module.load_state_dict(sd)


n_gpu = max(1, min(2, torch.cuda.device_count()))
devs = [f"cuda:{i}" for i in range(n_gpu)]
layers = [inner.layers[i] for i in range(start, end)]
half = -(-len(layers) // n_gpu)                 # my layers split over my GPUs
for i, layer in enumerate(layers):
    load_into(layer, f"model.layers.{start + i}.", devs[i // half])
rotary = type(inner.rotary_emb)(config=cfg, device=devs[0])   # real buffers, not meta
embed = None
if ROLE == 0:
    embed = inner.embed_tokens
    load_into(embed, "model.embed_tokens.", devs[0])
if ROLE == N_NODES - 1:
    norm = inner.norm
    load_into(norm, "model.norm.", devs[-1])
    head = full.lm_head
    load_into(head, "model.embed_tokens." if cfg.tie_word_embeddings
              else "lm_head.", devs[-1])
_gc = GenerationConfig.from_pretrained(MODEL, token=HF_TOKEN)
eos = _gc.eos_token_id or tok.eos_token_id
STOP = set(eos if isinstance(eos, list) else [eos])
for i in range(n_layers):                        # drop the layers I do not own
    if not (start <= i < end):
        inner.layers[i] = torch.nn.Identity()
gc.collect()
print(f"loaded layers {start}-{end - 1} of {n_layers}; "
      f"VRAM {[round(torch.cuda.memory_allocated(i)/2**30, 1) for i in range(n_gpu)]} GB")

# transformers renamed past_key_value -> past_key_values in newer versions
_params = inspect.signature(layers[0].forward).parameters
PKV_NAME = "past_key_values" if "past_key_values" in _params else "past_key_value"

# ================================================================ CHUNK 3: sessions, links, http
# One DynamicCache per generation request, indexed by the layer's global index.
SESSIONS = OrderedDict()      # sid -> {"cache", "past", "acc"}
SESS_LOCK = threading.Lock()
GPU_LOCK = threading.Lock()   # one forward pass on the GPUs at a time


def get_session(sid):
    with SESS_LOCK:
        if sid not in SESSIONS:
            SESSIONS[sid] = {"cache": DynamicCache(), "past": 0, "acc": 0}
            while len(SESSIONS) > MAX_SESSIONS:
                SESSIONS.popitem(last=False)
        SESSIONS.move_to_end(sid)
        return SESSIONS[sid]


def drop_session(sid):
    with SESS_LOCK:
        SESSIONS.pop(sid, None)


def accept(sid, step):
    """True only the first time `step` arrives for this session.
    A retried request that was already processed is dropped here."""
    st = get_session(sid)
    with SESS_LOCK:
        if st["acc"] != step:
            return False
        st["acc"] = step + 1
        return True


# Set by the controller through /set_links.
NEXT_URL = None     # the next notebook (None on the last one)
OUT_URL = None      # notebook 3: keeps the text, receives errors
FIRST_URL = None    # notebook 1: receives the token callback
SESSION = requests.Session()

# Notebook 3: the text of each request (the device polls this).
OUT = OrderedDict()
OUT_LOCK = threading.Lock()
# Notebook 1: bookkeeping of each request.
JOBS = {}
JOBS_LOCK = threading.Lock()


def get_out(sid):
    with OUT_LOCK:
        if sid not in OUT:
            OUT[sid] = {"tokens": [], "text": "", "done": False, "error": None}
            while len(OUT) > 20:
                OUT.popitem(last=False)
        return OUT[sid]


def post_json(url, path, payload=None, data=None, headers=None, tries=3):
    """One short POST to another notebook. Retries only on network or 5xx errors."""
    h = {"X-Password": PASSWORD}
    if headers:
        h.update(headers)
    last = None
    for i in range(tries):
        try:
            r = SESSION.post(url + path, json=payload, data=data,
                             headers=h, timeout=60)
            if r.ok:
                return r
            last = RuntimeError(f"{path} on {url} returned {r.status_code}: "
                                f"{r.text[:300]}")
            if r.status_code in (400, 401, 404, 422):
                break
        except Exception as e:
            last = e
        time.sleep(1.5 * (i + 1))
    raise last


# ================================================================ CHUNK 4: the math
@torch.no_grad()
def run_block(h, state):
    past = state["past"]
    new_len = h.shape[1]
    pos = torch.arange(past, past + new_len, device=devs[0]).unsqueeze(0)
    h = h.to(devs[0], dtype=torch.float16)
    cos, sin = rotary(h, pos)
    for i, layer in enumerate(layers):
        d = devs[i // half]
        if h.device != torch.device(d):
            h = h.to(d)
        out = layer(h, attention_mask=None, position_ids=pos.to(d),
                    position_embeddings=(cos.to(d), sin.to(d)),
                    cache_position=pos[0].to(d), use_cache=True,
                    **{PKV_NAME: state["cache"]})
        h = out[0] if isinstance(out, tuple) else out
    state["past"] = past + new_len
    return h


def send_next(h, sid, max_new, step):
    arr = h.to(torch.float16).cpu().numpy()
    post_json(NEXT_URL, "/step", data=arr.tobytes(),
              headers={"X-Session": sid, "X-Shape": ",".join(map(str, arr.shape)),
                       "X-Max-New": str(max_new), "X-Step": str(step),
                       "Content-Type": "application/octet-stream"})


def end_session(sid):
    """Free this session's cache here and on every node after me."""
    drop_session(sid)
    if NEXT_URL is not None and ROLE != N_NODES - 1:
        try:
            post_json(NEXT_URL, "/end", {"session": sid}, tries=1)
        except Exception:
            pass


def set_error(sid, msg):
    """Notebook 3: mark the request as failed so the device sees it, free caches."""
    o = get_out(sid)
    o["error"] = msg
    o["done"] = True
    try:
        if ROLE == 0:
            end_session(sid)
        elif FIRST_URL:
            post_json(FIRST_URL, "/end", {"session": sid}, tries=1)
    except Exception:
        pass


def report_error(sid, msg):
    try:
        if ROLE == N_NODES - 1:
            set_error(sid, msg)
        else:
            post_json(OUT_URL, "/error", {"session": sid, "error": msg})
    except Exception:
        print("could not report error:", msg)


def finish_token(sid, max_new, step, tokid):
    """Notebook 3: store the token's text, then tell notebook 1."""
    o = get_out(sid)
    done = tokid in STOP
    if not done:
        o["tokens"].append(tokid)
        done = len(o["tokens"]) >= max_new
    text = tok.decode(o["tokens"], skip_special_tokens=True)
    if not done:
        text = text.rstrip("\ufffd")           # wait for a complete character
    o["text"] = text
    o["done"] = done                           # set last: the poll reads text first
    post_json(FIRST_URL, "/token", {"session": sid, "step": step,
                                    "token": tokid, "done": done})


@torch.no_grad()
def run_step(sid, max_new, step, h=None, ids=None):
    """One pipeline step on this notebook, then hand over (runs in a thread)."""
    try:
        if not accept(sid, step):
            return                              # duplicate delivery
        tokid = None
        with GPU_LOCK:
            if ids is not None:                 # notebook 1: embed the new ids
                h = embed(ids.to(devs[0]))
            h = run_block(h, get_session(sid))
            if ROLE == N_NODES - 1:             # last notebook picks the next word
                h = norm(h.to(devs[-1]))
                tokid = int(head(h[:, -1, :]).argmax(dim=-1).item())
        if ROLE == N_NODES - 1:
            finish_token(sid, max_new, step, tokid)
        else:
            send_next(h, sid, max_new, step)
    except Exception as e:
        traceback.print_exc()
        report_error(sid, f"node {ROLE}: {type(e).__name__}: {e}")


# ================================================================ CHUNK 5: the endpoints
app = FastAPI()


def check(pw):
    if not pw or not hmac.compare_digest(pw, PASSWORD):
        raise HTTPException(401, "bad password")


def clean(u):
    return u.rstrip("/") if u else None


@app.get("/health")
def health(x_password: str = Header(None)):
    check(x_password)
    return {"role": ROLE, "layers": f"{start}-{end - 1}", "gpus": n_gpu,
            "next": NEXT_URL, "out": OUT_URL, "first": FIRST_URL,
            "sessions": len(SESSIONS)}


@app.post("/set_links")
async def set_links(request: Request, x_password: str = Header(None)):
    global NEXT_URL, OUT_URL, FIRST_URL
    check(x_password)
    req = await request.json()
    NEXT_URL, OUT_URL, FIRST_URL = (clean(req.get("next")), clean(req.get("out")),
                                    clean(req.get("first")))
    return {"ok": True}


@app.post("/start")                              # device -> notebook 1
async def start(request: Request, x_password: str = Header(None)):
    check(x_password)
    if ROLE != 0:
        raise HTTPException(400, "send prompts to the first notebook")
    if not (NEXT_URL and OUT_URL and FIRST_URL):
        raise HTTPException(400, "links not set: run controller.py first")
    req = await request.json()
    messages = req["messages"]
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    max_new = int(req.get("max_new_tokens", 100))
    text = tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids
    sid = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[sid] = {"max_new": max_new, "cb": 0}
        while len(JOBS) > 20:
            JOBS.pop(next(iter(JOBS)))
    threading.Thread(target=run_step, args=(sid, max_new, 0),
                     kwargs={"ids": ids}, daemon=True).start()
    return {"job": sid, "prompt_tokens": int(ids.shape[1])}


@app.post("/step")                               # notebook N -> notebook N+1
async def step(request: Request, x_password: str = Header(None),
               x_shape: str = Header(None), x_session: str = Header(None),
               x_max_new: str = Header(None), x_step: str = Header(None)):
    check(x_password)
    if not (x_shape and x_session and x_max_new and x_step):
        raise HTTPException(400, "missing X-Shape / X-Session / X-Max-New / X-Step")
    shape = tuple(int(x) for x in x_shape.split(","))
    arr = np.frombuffer(await request.body(), dtype=np.float16).reshape(shape).copy()
    threading.Thread(target=run_step,
                     args=(x_session, int(x_max_new), int(x_step)),
                     kwargs={"h": torch.from_numpy(arr)}, daemon=True).start()
    return {"ok": True}


@app.post("/token")                              # notebook 3 -> notebook 1 (callback)
async def token(request: Request, x_password: str = Header(None)):
    check(x_password)
    if ROLE != 0:
        raise HTTPException(400, "tokens go back to the first notebook")
    req = await request.json()
    sid, s = req["session"], int(req["step"])
    with JOBS_LOCK:
        job = JOBS.get(sid)
        if job is None or s != job["cb"]:
            return {"ok": True, "ignored": True}   # unknown or duplicate delivery
        job["cb"] += 1
        max_new = job["max_new"]
        if req.get("done"):
            JOBS.pop(sid, None)
    if req.get("done"):
        threading.Thread(target=end_session, args=(sid,), daemon=True).start()
    else:
        ids = torch.tensor([[int(req["token"])]])
        threading.Thread(target=run_step, args=(sid, max_new, s + 1),
                         kwargs={"ids": ids}, daemon=True).start()
    return {"ok": True}


@app.post("/poll")                               # device -> notebook 3
async def poll(request: Request, x_password: str = Header(None)):
    check(x_password)
    if ROLE != N_NODES - 1:
        raise HTTPException(400, "poll the last notebook")
    req = await request.json()
    since = int(req.get("since", 0))
    final_only = bool(req.get("final_only", False))   # True: text only when finished
    o = get_out(req["session"])
    for _ in range(100):                         # wait up to ~50 s
        if o["done"] or (not final_only and len(o["text"]) > since):
            break
        await asyncio.sleep(0.5)
    done = o["done"]                             # read done first, then text
    text = o["text"]
    if final_only and not done:                  # not finished: send no text yet
        return {"text": "", "len": since, "done": False,
                "error": o["error"], "tokens": len(o["tokens"])}
    return {"text": text[since:], "len": len(text), "done": done,
            "error": o["error"], "tokens": len(o["tokens"])}


@app.post("/error")                              # any notebook -> notebook 3
async def error_route(request: Request, x_password: str = Header(None)):
    check(x_password)
    req = await request.json()
    threading.Thread(target=set_error, args=(req["session"], req["error"]),
                     daemon=True).start()
    return {"ok": True}


@app.post("/end")
async def end_route(request: Request, x_password: str = Header(None)):
    check(x_password)
    sid = (await request.json())["session"]
    await run_in_threadpool(end_session, sid)
    return {"ok": True}


# ================================================================ CHUNK 6: server + tunnel (run last)
server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORT,
                                       log_level="warning"))
server.install_signal_handlers = lambda: None
threading.Thread(target=server.run, daemon=True).start()
for _ in range(100):
    if server.started:
        break
    time.sleep(0.1)
else:
    raise RuntimeError(f"server did not start on port {PORT}: an old run is still "
                       "alive in this kernel. Restart the session and run again.")

if not os.path.exists("cloudflared"):
    urllib.request.urlretrieve(
        "https://github.com/cloudflare/cloudflared/releases/latest/download/"
        "cloudflared-linux-amd64", "cloudflared")
os.chmod("cloudflared", 0o755)                   # always, not only after download
proc = subprocess.Popen(
    ["./cloudflared", "tunnel", "--url", f"http://127.0.0.1:{PORT}",
     "--no-autoupdate"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
url = None
for line in proc.stdout:
    m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
    if m and m.group(0) != "https://api.trycloudflare.com":
        url = m.group(0)
        break
threading.Thread(target=lambda: [None for _ in proc.stdout], daemon=True).start()

print(f"NODE {ROLE} READY: layers {start}-{end - 1} on {devs} (short-hop version)")
print(f"TUNNEL ADDRESS: {url}")
while True:                                      # keep this cell running
    time.sleep(60)