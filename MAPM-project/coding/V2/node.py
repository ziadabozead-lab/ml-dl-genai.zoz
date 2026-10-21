# node.py - MAPM node (with KV cache). Paste into ONE cell of each Kaggle notebook and run it.
# Notebook settings: Accelerator = GPU T4 x2, Internet = On,
# secrets HF_TOKEN and TUNNEL_PASSWORD attached.
# Change ROLE in each notebook: 0 = first, 1 = middle, 2 = last.
ROLE = 0
N_NODES = 3
MODEL = "Qwen/Qwen2.5-14B-Instruct"   # 48 layers -> 16 per notebook (use the full "org/name")
PORT = 8000
MAX_SESSIONS = 8                        # KV caches kept per node (oldest evicted)

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "fastapi", "uvicorn", "requests"], check=True)

import os, re, gc, time, hmac, uuid, inspect, threading, traceback, urllib.request
from collections import OrderedDict
import numpy as np, requests, torch, uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from starlette.concurrency import run_in_threadpool
from kaggle_secrets import UserSecretsClient
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

secrets = UserSecretsClient()
PASSWORD = secrets.get_secret("TUNNEL_PASSWORD")
HF_TOKEN = secrets.get_secret("HF_TOKEN")

# ---------------------------------------------------------------- load my slice
# Only the tensors this node owns are downloaded and loaded (no full model in RAM).
import json
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig, GenerationConfig

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

# ---------------------------------------------------------------- KV cache sessions
# One DynamicCache per generation request. Each cache is indexed by the layer's
# global index, so a node only fills the entries for the layers it owns.
SESSIONS = OrderedDict()      # sid -> {"cache": DynamicCache, "past": int}
SESS_LOCK = threading.Lock()


def get_session(sid):
    with SESS_LOCK:
        if sid not in SESSIONS:
            SESSIONS[sid] = {"cache": DynamicCache(), "past": 0}
            while len(SESSIONS) > MAX_SESSIONS:
                SESSIONS.popitem(last=False)
        SESSIONS.move_to_end(sid)
        return SESSIONS[sid]


def drop_session(sid):
    with SESS_LOCK:
        SESSIONS.pop(sid, None)


# ---------------------------------------------------------------- the math
NEXT_URL = None
SESSION = requests.Session()   # reuse one connection to the next notebook


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


def send_next(h, sid):
    arr = h.to(torch.float16).cpu().numpy()
    r = SESSION.post(
        NEXT_URL + "/step", data=arr.tobytes(), timeout=300,
        headers={"X-Password": PASSWORD, "X-Session": sid,
                 "X-Shape": ",".join(map(str, arr.shape)),
                 "Content-Type": "application/octet-stream"})
    if not r.ok:                                  # surface the remote node's real error
        raise RuntimeError(f"next node failed ({r.status_code}): {r.text[:800]}")
    return r.json()["token"]


@torch.no_grad()
def process(h, sid):
    h = run_block(h, get_session(sid))
    if ROLE == N_NODES - 1:                      # last notebook picks the next word
        h = norm(h.to(devs[-1]))
        return int(head(h[:, -1, :]).argmax(dim=-1).item())
    return send_next(h, sid)                     # otherwise pass activations on


def end_session(sid):
    """Free this session's cache here and on every node after me."""
    drop_session(sid)
    if NEXT_URL is not None and ROLE != N_NODES - 1:
        try:
            SESSION.post(NEXT_URL + "/end", json={"session": sid},
                         headers={"X-Password": PASSWORD}, timeout=30)
        except Exception:
            pass


@torch.no_grad()
def do_generate(messages, max_new):
    if NEXT_URL is None and ROLE != N_NODES - 1:
        raise RuntimeError("next notebook address not set yet")
    if isinstance(messages, str):                      # plain prompt still works
        messages = [{"role": "user", "content": messages}]
    text = tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids     # step 0: whole prompt
    sid = uuid.uuid4().hex
    new, t0 = [], time.time()
    try:
        for _ in range(max_new):
            nxt = process(embed(ids.to(devs[0])), sid)
            if nxt in STOP:
                break
            new.append(nxt)
            ids = torch.tensor([[nxt]])                # later steps: ONLY the new token
    finally:
        end_session(sid)
    return {"text": tok.decode(new, skip_special_tokens=True),
            "tokens": len(new), "seconds": round(time.time() - t0, 1)}


# Single-notebook smoke test: set N_NODES = 1 to run the whole model here first.
if N_NODES == 1:
    print("SMOKE TEST:", do_generate("What is the capital of France?", 30))

# ---------------------------------------------------------------- the server
app = FastAPI()


def fail(e):
    traceback.print_exc()                         # full traceback in this notebook's log
    raise HTTPException(500, f"node {ROLE}: {type(e).__name__}: {e}")


def check(pw):
    if not pw or not hmac.compare_digest(pw, PASSWORD):
        raise HTTPException(401, "bad password")


@app.get("/health")
def health(x_password: str = Header(None)):
    check(x_password)
    return {"role": ROLE, "layers": f"{start}-{end - 1}", "gpus": n_gpu,
            "next": NEXT_URL, "kv_cache": True, "sessions": len(SESSIONS)}


@app.post("/set_next")
async def set_next(request: Request, x_password: str = Header(None)):
    global NEXT_URL
    check(x_password)
    NEXT_URL = (await request.json())["url"].rstrip("/")
    return {"ok": True}


@app.post("/step")
async def step(request: Request, x_password: str = Header(None),
               x_shape: str = Header(None), x_session: str = Header(None)):
    check(x_password)
    if not x_session:
        raise HTTPException(400, "missing X-Session")
    shape = tuple(int(x) for x in x_shape.split(","))
    arr = np.frombuffer(await request.body(), dtype=np.float16).reshape(shape).copy()
    try:
        token = await run_in_threadpool(process, torch.from_numpy(arr), x_session)
    except Exception as e:
        fail(e)
    return {"token": token}


@app.post("/end")
async def end_route(request: Request, x_password: str = Header(None)):
    check(x_password)
    sid = (await request.json())["session"]
    await run_in_threadpool(end_session, sid)
    return {"ok": True}


@app.post("/generate")
async def generate(request: Request, x_password: str = Header(None)):
    check(x_password)
    if ROLE != 0:
        raise HTTPException(400, "send prompts to the first notebook")
    req = await request.json()
    try:
        return await run_in_threadpool(
            do_generate, req.get("messages") or req["prompt"],
            int(req.get("max_new_tokens", 100)))
    except Exception as e:
        fail(e)


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

# ---------------------------------------------------------------- the tunnel
if not os.path.exists("cloudflared"):
    urllib.request.urlretrieve(
        "https://github.com/cloudflare/cloudflared/releases/latest/download/"
        "cloudflared-linux-amd64", "cloudflared")
    os.chmod("cloudflared", 0o755)
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

print(f"NODE {ROLE} READY: layers {start}-{end - 1} on {devs} (KV cache on)")
print(f"TUNNEL ADDRESS: {url}")
while True:                                      # keep this cell running
    time.sleep(60)
