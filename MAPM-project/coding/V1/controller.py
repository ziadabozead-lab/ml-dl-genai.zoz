# controller.py - run on your device (VS Code terminal). Needs only: pip install requests
# Usage:  python controller.py            (it asks for the 3 tunnel addresses)
#     or  python controller.py URL1 URL2 URL3
import os, sys, json, time, requests



# Secrets live in secrets.json next to this file (keep it out of git).
_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "secrets.json")
if not os.path.exists(_path):
    sys.exit("secrets.json not found next to controller.py")
with open(_path, encoding="utf-8") as f:
    PASSWORD = json.load(f)["tunnel_password"]   # must match the Kaggle secret TUNNEL_PASSWORD
MAX_NEW_TOKENS = 60   # lower = faster answers, higher = longer answers
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

# 2) tell each notebook where the next one is: 1 -> 2 -> 3
for i in range(2):
    requests.post(urls[i] + "/set_next", json={"url": urls[i + 1]},
                  headers=H, timeout=30).raise_for_status()
print("linked: notebook 1 -> 2 -> 3\n")

# 3) chat (type exit to stop)
while True:
    prompt = input("You: ").strip()
    if prompt.lower() in ("", "exit", "quit"):
        break
    r = requests.post(urls[0] + "/generate", headers=H, timeout=900,
                      json={"prompt": prompt, "max_new_tokens": MAX_NEW_TOKENS})
    if r.status_code != 200:
        print("error:", r.status_code, r.text)
        continue
    out = r.json()
    per = out["seconds"] / max(out["tokens"], 1)
    print(f"Model: {out['text']}\n({out['tokens']} tokens in {out['seconds']}s = {per:.2f}s per token)\n")
