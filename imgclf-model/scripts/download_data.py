"""
Download script for all three imgclf-model datasets:
    v1 - CIFAR-10      (10 classes)   -> rdf/data-v1/cifar10/
    v2 - CIFAR-100      (100 classes)  -> rdf/data-v2/cifar100/
    v3 - Tiny-ImageNet-200 (200 classes) -> rdf/data-v3/tiny-imagenet-200/
    v4 - ImageNet-1k at 64x64 (1000 classes, ~1.7 GB train + val) -> rdf/data-v4/imagenet-1k-64x64/
         (NOT included in the default "download everything" run - ask for it with --v4)

Does ONLY the downloading/extracting - no training. Run this first, then
use the matching training script (train_imgclf.py for v1/v2,
train_imgclf_transfer_only.py for v3) once the data is in place.

Requires:  pip install torchvision pillow
           pip install huggingface_hub pyarrow        (only for --v4)

Usage (run from the project root, imgclf-model/):
    python scripts/download_data.py            # downloads all three
    python scripts/download_data.py --v1        # just CIFAR-10
    python scripts/download_data.py --v2        # just CIFAR-100
    python scripts/download_data.py --v3        # just Tiny-ImageNet-200
    python scripts/download_data.py --v4        # just ImageNet-1k 64x64 (1000 classes)
    python scripts/download_data.py --v1 --v2   # any combination
"""
import argparse
import time
import urllib.request
import zipfile
from pathlib import Path

from torchvision import datasets

ROOT = Path(__file__).resolve().parent.parent          # the project folder

V1_DIR = ROOT / "rdf" / "data-v1" / "cifar10"
V2_DIR = ROOT / "rdf" / "data-v2" / "cifar100"
V3_DIR = ROOT / "rdf" / "data-v3" / "tiny-imagenet-200"

V4_DIR = ROOT / "rdf" / "data-v4" / "imagenet-1k-64x64"
V4_REPO = "benjamin-paine/imagenet-1k-64x64"      # Hugging Face dataset (parquet files)

TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


# --------------------------------- v1 ------------------------------------ #
def download_v1():
    print(f"[v1] Downloading CIFAR-10 (~170 MB) to {V1_DIR} ...")
    V1_DIR.mkdir(parents=True, exist_ok=True)
    datasets.CIFAR10(root=str(V1_DIR), train=True, download=True)
    datasets.CIFAR10(root=str(V1_DIR), train=False, download=True)
    print(f"[v1] Done -> {V1_DIR}")


# --------------------------------- v2 ------------------------------------ #
def download_v2():
    print(f"[v2] Downloading CIFAR-100 (~170 MB) to {V2_DIR} ...")
    V2_DIR.mkdir(parents=True, exist_ok=True)
    datasets.CIFAR100(root=str(V2_DIR), train=True, download=True)
    datasets.CIFAR100(root=str(V2_DIR), train=False, download=True)
    print(f"[v2] Done -> {V2_DIR}")


# --------------------------------- v3 ------------------------------------ #
def download_v3():
    if (V3_DIR / "wnids.txt").exists():
        print(f"[v3] Already present at {V3_DIR}, skipping download.")
        return
    data_parent = V3_DIR.parent
    data_parent.mkdir(parents=True, exist_ok=True)
    zip_path = data_parent / "tiny-imagenet-200.zip"

    if not zip_path.exists():
        print(f"[v3] Downloading Tiny-ImageNet-200 (~237 MB) to {zip_path} ...")
        last = [0.0]
        def hook(count, block_size, total_size):
            pct = count * block_size / max(total_size, 1) * 100
            if time.time() - last[0] > 1:
                print(f"\r  {pct:5.1f}%", end="", flush=True)
                last[0] = time.time()
        urllib.request.urlretrieve(TINY_IMAGENET_URL, zip_path, reporthook=hook)
        print("\r  100.0% - done")
    else:
        print(f"[v3] Zip already downloaded at {zip_path}, skipping download.")

    print("[v3] Extracting...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_parent)
    print(f"[v3] Done -> {V3_DIR}")


# --------------------------------- v4 ------------------------------------ #
def download_v4():
    """ImageNet-1k downsampled to 64x64 (1.28M train / 50k val), stored as parquet.
    Only the train + validation shards are fetched (the 100k 'test' split has no labels)."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("v4 needs huggingface_hub:  pip install huggingface_hub pyarrow")
    print(f"[v4] Downloading ImageNet-1k 64x64 (~1.7 GB for train + val) to {V4_DIR} ...")
    V4_DIR.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(repo_id=V4_REPO, repo_type="dataset", local_dir=str(V4_DIR),
                          allow_patterns=["data/train-*.parquet", "data/validation-*.parquet", "README.md"])
    except Exception as e:
        raise SystemExit(f"[v4] Download failed: {e}\n"
                         f"If it says 401/gated: log in with `huggingface-cli login` and accept the "
                         f"terms on https://huggingface.co/datasets/{V4_REPO}, then run again.")
    print(f"[v4] Done -> {V4_DIR}")


# -------------------------------- main ------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", action="store_true", help="download CIFAR-10 only")
    ap.add_argument("--v2", action="store_true", help="download CIFAR-100 only")
    ap.add_argument("--v3", action="store_true", help="download Tiny-ImageNet-200 only")
    ap.add_argument("--v4", action="store_true", help="download ImageNet-1k 64x64 (1000 classes)")
    args = ap.parse_args()

    # if none of the flags are given, download v1-v3 (v4 is big, so it is opt-in)
    run_all = not (args.v1 or args.v2 or args.v3 or args.v4)

    if args.v1 or run_all:
        download_v1()
    if args.v2 or run_all:
        download_v2()
    if args.v3 or run_all:
        download_v3()
    if args.v4:
        download_v4()

    print("\nAll requested datasets are ready.")


if __name__ == "__main__":
    main()