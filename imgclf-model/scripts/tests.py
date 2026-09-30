import torch, glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # imgclf-model folder
for p in sorted(glob.glob(str(ROOT / "weights" / "*.pt"))):
    ck = torch.load(p, map_location="cpu", weights_only=True)
    print(Path(p).name, len(ck["classes"]), ck.get("img_size"))