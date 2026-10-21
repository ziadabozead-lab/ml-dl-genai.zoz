# controller_v3.py - run on your device (VS Code terminal). Needs only: pip install requests
# Works with node_v3.py. Usage:  python controller_v3.py            (asks for the 3 addresses)
#                           or:  python controller_v3.py URL1 URL2 URL3
#
# The prompt goes to notebook 1 (/start). The answer is read from notebook 3 (/poll).
import os, sys, json, time, requests

# Secrets live in secrets.json next to this file (keep it out of git).
_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "secrets.json")
if not os.path.exists(_path):
    sys.exit("secrets.json not found next to controller_v3.py")
with open(_path, encoding="utf-8") as f:
    PASSWORD = json.load(f)["tunnel_password"]   # must match the Kaggle secret TUNNEL_PASSWORD
MAX_NEW_TOKENS = 6000      # no Cloudflare limit any more: every request is short
IDLE_LIMIT = 300          # give up if no new text arrives for this many seconds
H = {"X-Password": PASSWORD}

urls = [u.strip().rstrip("/") for u in sys.argv[1:4]]
if len(urls) != 3:
    print("Paste the tunnel address printed by each notebook:")
    urls = [input(f"  notebook {i + 1}: ").strip().rstrip("/") for i in range(3)]

# 1) check that every notebook answers (a new tunnel can take a few seconds)
for i, u in enumerate(urls):
    for attempt in range(10):
        try:
            r = requests.get(u + "/health", headers=H, timeout=30)
            r.raise_for_status()
            print(f"notebook {i + 1} OK:", r.json())
            break
        except Exception as e:
            if attempt == 9:
                sys.exit(f"notebook {i + 1} not reachable: {e}")
            time.sleep(3)

# 2) tell every notebook the links: next notebook, notebook 3 (output), notebook 1 (callback)
for i, u in enumerate(urls):
    links = {"next": urls[i + 1] if i < 2 else None,
             "out": urls[2], "first": urls[0]}
    requests.post(u + "/set_links", json=links, headers=H, timeout=30).raise_for_status()
print("linked: notebook 1 -> 2 -> 3, output read from notebook 3\n")

# 3) chat (type exit to stop, "reset" to forget the conversation)
history = []          # the whole conversation is resent every turn
while True:
    prompt = input("You: ").strip()
    if prompt.lower() in ("", "exit", "quit"):
        break
    if prompt.lower() == "reset":
        history.clear()
        print("(conversation cleared)\n")
        continue
    history.append({"role": "user", "content": prompt})
    t0 = time.time()

    # start the request on notebook 1 (answers at once)
    try:
        r = requests.post(urls[0] + "/start", headers=H, timeout=60,
                          json={"messages": history, "max_new_tokens": MAX_NEW_TOKENS})
    except Exception as e:
        print("error: could not reach notebook 1:", e)
        history.pop()
        continue
    if r.status_code != 200:
        print("error:", r.status_code, r.text[:500])
        history.pop()
        continue
    job = r.json()["job"]

    # ask notebook 3 "is it finished?" in short requests (each waits up to ~50 s).
    # It sends no text until the whole answer is ready, then sends it all at once.
    text, final, fails = "", None, 0
    last_tokens, last_progress = -1, time.time()
    while final is None:
        try:
            p = requests.post(urls[2] + "/poll", headers=H, timeout=90,
                              json={"session": job, "since": 0, "final_only": True})
            p.raise_for_status()
            fails = 0
        except Exception as e:                   # dropped connection: ask again
            fails += 1
            if fails >= 30:
                final = {"error": f"lost contact with notebook 3: {e}",
                         "done": True, "tokens": 0}
                break
            time.sleep(2)
            continue
        m = p.json()
        if m["done"]:
            final = m
            text = m["text"]
            break
        if m["tokens"] != last_tokens:           # progress = new tokens on notebook 3
            last_tokens, last_progress = m["tokens"], time.time()
            print(f"\r(working... {m['tokens']} tokens so far)", end="", flush=True)
        elif time.time() - last_progress > IDLE_LIMIT:
            final = {"error": f"no progress for {IDLE_LIMIT}s (a notebook may have stopped)",
                     "done": True, "tokens": 0}
    print("\r" + " " * 40 + "\r", end="")        # clear the status line
    if final.get("error"):
        print("error:", final["error"])
        history.pop()
        continue
    print(f"Model: {text}")
    history.append({"role": "assistant", "content": text})
    secs = time.time() - t0
    per = secs / max(final["tokens"], 1)
    print(f"({final['tokens']} tokens in {secs:.1f}s = {per:.2f}s per token)\n")
