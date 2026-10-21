# node.py - MAPM node. Paste into ONE cell of each Kaggle notebook and run it.
# Notebook settings: Accelerator = GPU T4 x2, Internet = On,
# secrets HF_TOKEN and TUNNEL_PASSWORD attached.
# Change ROLE in each notebook: 0 = first, 1 = middle, 2 = last.
ROLE = 0
N_NODES = 3
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"   # 24 layers -> 8 layers per notebook
PORT = 8000

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "fastapi", "uvicorn", "requests"], check=True)

import os, re, gc, time, hmac, threading, urllib.request
import numpy as np, requests, torch, uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from starlette.concurrency import run_in_threadpool
from kaggle_secrets import UserSecretsClient
from transformers import AutoModelForCausalLM, AutoTokenizer

secrets = UserSecretsClient()
PASSWORD = secrets.get_secret("TUNNEL_PASSWORD")
HF_TOKEN = secrets.get_secret("HF_TOKEN")

# ---------------------------------------------------------------- load my slice
tok = AutoTokenizer.from_pretrained(MODEL, token=HF_TOKEN)
full = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, attn_implementation="sdpa", token=HF_TOKEN)
inner = full.model
n_layers = len(inner.layers)
per = n_layers // N_NODES
start = ROLE * per
end = n_layers if ROLE == N_NODES - 1 else start + per

n_gpu = max(1, min(2, torch.cuda.device_count()))
devs = [f"cuda:{i}" for i in range(n_gpu)]
layers = [inner.layers[i] for i in range(start, end)]
half = -(-len(layers) // n_gpu)                 # my layers split over my GPUs
for i, layer in enumerate(layers):
    layer.to(devs[i // half])
rotary = inner.rotary_emb.to(devs[0])
embed = inner.embed_tokens.to(devs[0]) if ROLE == 0 else None
if ROLE == N_NODES - 1:
    norm = inner.norm.to(devs[-1])
    head = full.lm_head.to(devs[-1])
eos = full.generation_config.eos_token_id or tok.eos_token_id
STOP = set(eos if isinstance(eos, list) else [eos])
for i in range(n_layers):                        # drop the layers I do not own
    if not (start <= i < end):
        inner.layers[i] = torch.nn.Identity()
gc.collect()

# ---------------------------------------------------------------- the math
NEXT_URL = None
SESSION = requests.Session()   # reuse one connection to the next notebook


@torch.no_grad()
def run_block(h):
    pos = torch.arange(h.shape[1], device=devs[0]).unsqueeze(0)
    h = h.to(devs[0])
    cos, sin = rotary(h, pos)
    for i, layer in enumerate(layers):
        d = devs[i // half]
        if h.device != torch.device(d):
            h = h.to(d)
        out = layer(h, attention_mask=None, position_ids=pos.to(d),
                    position_embeddings=(cos.to(d), sin.to(d)))
        h = out[0] if isinstance(out, tuple) else out
    return h


def send_next(h):
    arr = h.to(torch.float16).cpu().numpy()
    r = SESSION.post(
        NEXT_URL + "/step", data=arr.tobytes(), timeout=300,
        headers={"X-Password": PASSWORD,
                 "X-Shape": ",".join(map(str, arr.shape)),
                 "Content-Type": "application/octet-stream"})
    r.raise_for_status()
    return r.json()["token"]


@torch.no_grad()
def process(h):
    h = run_block(h)
    if ROLE == N_NODES - 1:                      # last notebook picks the next word
        h = norm(h.to(devs[-1]))
        return int(head(h[:, -1, :]).argmax(dim=-1).item())
    return send_next(h)                          # otherwise pass activations on


@torch.no_grad()
def do_generate(prompt, max_new):
    if NEXT_URL is None and ROLE != N_NODES - 1:
        raise RuntimeError("next notebook address not set yet")
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids
    new, t0 = [], time.time()
    for _ in range(max_new):
        nxt = process(embed(ids.to(devs[0])))
        if nxt in STOP:
            break
        new.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
    return {"text": tok.decode(new, skip_special_tokens=True),
            "tokens": len(new), "seconds": round(time.time() - t0, 1)}


# Single-notebook smoke test: set N_NODES = 1 to run the whole model here first.
if N_NODES == 1:
    print("SMOKE TEST:", do_generate("What is the capital of France?", 30))

# ---------------------------------------------------------------- the server
app = FastAPI()


def check(pw):
    if not pw or not hmac.compare_digest(pw, PASSWORD):
        raise HTTPException(401, "bad password")


@app.get("/health")
def health(x_password: str = Header(None)):
    check(x_password)
    return {"role": ROLE, "layers": f"{start}-{end - 1}", "gpus": n_gpu,
            "next": NEXT_URL}


@app.post("/set_next")
async def set_next(request: Request, x_password: str = Header(None)):
    global NEXT_URL
    check(x_password)
    NEXT_URL = (await request.json())["url"].rstrip("/")
    return {"ok": True}


@app.post("/step")
async def step(request: Request, x_password: str = Header(None),
               x_shape: str = Header(None)):
    check(x_password)
    shape = tuple(int(x) for x in x_shape.split(","))
    arr = np.frombuffer(await request.body(), dtype=np.float16).reshape(shape).copy()
    token = await run_in_threadpool(process, torch.from_numpy(arr))
    return {"token": token}


@app.post("/generate")
async def generate(request: Request, x_password: str = Header(None)):
    check(x_password)
    if ROLE != 0:
        raise HTTPException(400, "send prompts to the first notebook")
    req = await request.json()
    return await run_in_threadpool(
        do_generate, req["prompt"], int(req.get("max_new_tokens", 100)))


server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORT,
                                       log_level="warning"))
server.install_signal_handlers = lambda: None
threading.Thread(target=server.run, daemon=True).start()

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

print(f"NODE {ROLE} READY: layers {start}-{end - 1} on {devs}")
print(f"TUNNEL ADDRESS: {url}")
while True:                                      # keep this cell running
    time.sleep(60)
