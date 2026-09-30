"""
Image classifier v6 - ImageNet-1k (64x64), CPU-friendly transfer learning tuned for accuracy.

Same idea as v5 (frozen pretrained backbone -> cached features -> small trained head) with:
  * a choice of backbones (mobilenet_v2, efficientnet_b0, resnet50, convnext_tiny)
  * feature caching PER PARQUET SHARD, so a crash / sleep / Ctrl+C never loses hours of work:
    just run the same command again and it resumes
  * optional horizontal-flip augmentation of the training images (--flip-train)
  * a better head (input BatchNorm, AdamW, cosine LR, dropout, label smoothing)
  * flip test-time augmentation when classifying a single photo
  * class-name fix: trims a stray 1001st name if the dataset metadata has one
  * --benchmark: measures how fast each backbone runs on YOUR CPU before you commit

Honest note: the backbones were pretrained on ImageNet-1k, so this is transfer learning, not
from-scratch training. The reported test score is still honest: it is measured on official
ImageNet validation images that were never used to train the backbone or the head.

Project layout:
    imgclf-model/
      scripts/train_imgclf_v6.py
      rdf/data-v4/imagenet-1k-64x64/data/*.parquet
      weights/imgclf-model-v6.pt            <- created by training

Run from the project root (imgclf-model/):
    1) See speeds:       python scripts/train_imgclf_v6.py --benchmark --img-size 128
    2) Train:            python scripts/train_imgclf_v6.py --backbone efficientnet_b0 --img-size 128 --per-class 400
       Max accuracy:     python scripts/train_imgclf_v6.py --backbone convnext_tiny --img-size 128 --per-class 400 --flip-train
    3) Tune head only:   same command with different --hidden/--dropout/... (features are cached)
    4) Classify:         python scripts/train_imgclf_v6.py --predict photo.jpg

GUI: in chatbot/imgclf_gui.py change  import train_imgclf as ti  ->  import train_imgclf_v6 as ti
(load_model / predict_image keep the same signatures, and old v5 checkpoints still load).

Requires: pip install torchvision pillow pyarrow
"""
import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torchvision.models as tvm
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

ROOT = Path(__file__).resolve().parent.parent          # the project folder
DEFAULT_DATA = ROOT / "rdf" / "data-v4" / "imagenet-1k-64x64"
DEFAULT_OUT = ROOT / "weights" / "imgclf-model-v6.pt"

# Reuse the pretrained weights already downloaded for earlier versions if they are there.
_HUB = ROOT / "rdf" / "data-v3" / "torch_cache"
torch.hub.set_dir(str(_HUB if _HUB.exists() else DEFAULT_DATA.parent / "torch_cache"))

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SRC_SIZE = 64
N_CLASSES = 1000
BACKBONES = {            # name -> feature width
    "mobilenet_v2": 1280,
    "efficientnet_b0": 1280,
    "resnet50": 2048,
    "convnext_tiny": 768,
}


# -------------------------------- transforms -------------------------------- #
def make_tf(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


# ---------------------------------- backbone -------------------------------- #
def build_backbone(name, pretrained=True):
    if name not in BACKBONES:
        raise SystemExit(f"Unknown backbone '{name}'. Choose from: {', '.join(BACKBONES)}")
    net = getattr(tvm, name)(weights="DEFAULT" if pretrained else None)
    if name == "mobilenet_v2":
        net = nn.Sequential(net.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
    elif name == "efficientnet_b0":
        net.classifier = nn.Identity()
    elif name == "resnet50":
        net.fc = nn.Identity()
    elif name == "convnext_tiny":
        net.classifier[2] = nn.Identity()       # keep the final LayerNorm + Flatten, drop the Linear
    for p in net.parameters():
        p.requires_grad = False
    return net.eval()


# ------------------------------------ data ---------------------------------- #
def parquet_files(data_dir: Path, split):
    return sorted((data_dir / "data").glob(f"{split}-*.parquet"))


def load_class_names(first_file):
    meta = (pq.read_schema(first_file).metadata or {}).get(b"huggingface")
    if meta:
        try:
            names = json.loads(meta)["info"]["features"]["label"]["names"]
            return [n.split(",")[0].strip() for n in names]
        except (KeyError, ValueError):
            pass
    print("Warning: class names not found in the parquet metadata - using numbers instead.")
    return [str(i) for i in range(N_CLASSES)]


@torch.no_grad()
def extract_shard(path, backbone, tf, counts, per_class, flip, dim, batch=128):
    """Run one parquet shard through the backbone. Returns a dict that is saved as the shard cache."""
    local = counts.copy()
    feats, labels, buf_x, buf_y = [], [], [], []

    def flush():
        if not buf_x:
            return
        xb = torch.stack(buf_x)
        yb = torch.tensor(buf_y, dtype=torch.long)
        feats.append(backbone(xb).half())
        labels.append(yb)
        if flip:
            feats.append(backbone(torch.flip(xb, dims=[3])).half())
            labels.append(yb)
        buf_x.clear()
        buf_y.clear()

    for rb in pq.ParquetFile(path).iter_batches(batch_size=512, columns=["image", "label"]):
        for im, y in zip(rb.column("image").to_pylist(), rb.column("label").to_pylist()):
            if y < 0 or y >= N_CLASSES:
                continue
            if per_class and local[y] >= per_class:
                continue
            local[y] += 1
            buf_x.append(tf(Image.open(io.BytesIO(im["bytes"])).convert("RGB")))
            buf_y.append(y)
            if len(buf_x) >= batch:
                flush()
    flush()
    kept = torch.from_numpy(local - counts)
    if feats:
        return {"features": torch.cat(feats), "labels": torch.cat(labels), "kept": kept}
    return {"features": torch.empty(0, dim, dtype=torch.float16),
            "labels": torch.empty(0, dtype=torch.long), "kept": kept}


def build_features(files, cache_folder: Path, backbone, tf, dim, label, per_class=None, flip=False):
    """One cache file per shard: rerunning the command resumes where it stopped."""
    cache_folder.mkdir(parents=True, exist_ok=True)
    counts = np.zeros(N_CLASSES, dtype=np.int64)
    parts_f, parts_y = [], []
    t0, new_imgs = time.time(), 0
    for i, path in enumerate(files):
        if per_class and counts.min() >= per_class:
            break
        shard_cache = cache_folder / f"shard{i:04d}.pt"
        if shard_cache.exists():
            d = torch.load(shard_cache, weights_only=True)
        else:
            d = extract_shard(path, backbone, tf, counts, per_class, flip, dim)
            torch.save(d, shard_cache)
            new_imgs += int(d["kept"].sum().item()) * (2 if flip else 1)
        counts += d["kept"].numpy()
        parts_f.append(d["features"])
        parts_y.append(d["labels"])
        speed = new_imgs / max(time.time() - t0, 1e-9)
        print(f"\r  {label}: shard {i + 1}/{len(files)} | {int(counts.sum())} images kept | "
              f"{speed:.0f} img/s (new work)", end="", flush=True)
    print()
    if not parts_f or sum(len(y) for y in parts_y) == 0:
        raise SystemExit(f"No images read for {label} - is the download complete?")
    return torch.cat(parts_f), torch.cat(parts_y)


# ------------------------------------ model --------------------------------- #
class ClassifierHead(nn.Module):
    """norm=False, in_drop=0 reproduces the old v5 layout exactly (old checkpoints still load)."""
    def __init__(self, n_classes, in_dim=1280, hidden=1024, dropout=0.4, norm=False, in_drop=0.0):
        super().__init__()
        layers = []
        if norm:
            layers.append(nn.BatchNorm1d(in_dim))
        if in_drop > 0:
            layers.append(nn.Dropout(in_drop))
        if hidden:
            layers += [nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
                       nn.Dropout(dropout), nn.Linear(hidden, n_classes)]
        else:                                   # plain linear probe
            layers += [nn.Dropout(dropout), nn.Linear(in_dim, n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TransferModel(nn.Module):
    def __init__(self, backbone, head, img_size):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.img_size = img_size

    def forward(self, x):
        with torch.no_grad():
            f = self.backbone(x)
        return self.head(f)


def evaluate_head(head, feats, labels, loss_fn, bs=4096):
    head.eval()
    loss_sum, top1, top5 = 0.0, 0, 0
    with torch.no_grad():
        for i in range(0, len(labels), bs):
            x, y = feats[i:i + bs].float(), labels[i:i + bs]
            logits = head(x)
            loss_sum += loss_fn(logits, y).item() * len(y)
            best5 = logits.topk(5, dim=-1).indices
            top1 += (best5[:, 0] == y).sum().item()
            top5 += (best5 == y[:, None]).any(dim=1).sum().item()
    n = max(len(labels), 1)
    return {"loss": loss_sum / n, "acc": top1 / n, "top5": top5 / n}


# ------------------------------ predict / load ------------------------------ #
def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    name = ck.get("backbone", "mobilenet_v2")            # v5 and older checkpoints: MobileNetV2
    hp = dict(ck["hparams"])
    hp.setdefault("in_dim", BACKBONES[name])
    head = ClassifierHead(len(ck["classes"]), **hp)
    head.load_state_dict(ck["state_dict"])
    head.eval()
    model = TransferModel(build_backbone(name), head, ck.get("img_size", 96))
    model.eval()
    return model, ck["classes"]


def predict_image(model, classes, path, top_k=3):
    """Returns [(class_name, probability), ...] best first. Averages the photo and its mirror image."""
    img = Image.open(path).convert("RGB")
    img = img.resize((SRC_SIZE, SRC_SIZE), Image.BICUBIC)   # match the 64x64 training images
    x = make_tf(model.img_size)(img).unsqueeze(0)
    x = torch.cat([x, torch.flip(x, dims=[3])])
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=-1).mean(dim=0).tolist()
    return sorted(zip(classes, probs), key=lambda p: p[1], reverse=True)[:top_k]


# --------------------------------- benchmark -------------------------------- #
def benchmark(args):
    x = torch.randn(32, 3, args.img_size, args.img_size)
    n_train = args.per_class * N_CLASSES * (2 if args.flip_train else 1)
    print(f"Speed test at {args.img_size}x{args.img_size} (random data, {torch.get_num_threads()} threads)")
    print(f"Projected train extraction for --per-class {args.per_class}"
          f"{' --flip-train' if args.flip_train else ''}: {n_train:,} forward passes\n")
    for name in BACKBONES:
        net = build_backbone(name, pretrained=False)
        with torch.no_grad():
            net(x[:8])                                   # warm-up
            t = time.time()
            net(x)
            dt = time.time() - t
        ips = len(x) / dt
        print(f"  {name:<16} {ips:6.1f} img/s  ->  ~{n_train / ips / 3600:5.1f} h   "
              f"(features {BACKBONES[name]}-d, cache ~{n_train * BACKBONES[name] * 2 / 1e9:.1f} GB)")


# ------------------------------------ main ----------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--backbone", default="efficientnet_b0", choices=list(BACKBONES))
    ap.add_argument("--img-size", type=int, default=128, help="size the 64x64 images are upscaled to")
    ap.add_argument("--per-class", type=int, default=300, help="training images per class (max 1300)")
    ap.add_argument("--flip-train", action="store_true", help="also use mirrored training images (2x extraction time)")
    ap.add_argument("--val-per-class", type=int, default=50, help="val images per class to extract (max 50)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--hidden", type=int, default=2048, help="0 = plain linear head")
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--in-drop", type=float, default=0.1, help="dropout applied to the input features")
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--min-delta", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--extract-only", action="store_true", help="only build the feature caches, then stop")
    ap.add_argument("--benchmark", action="store_true", help="measure backbone speeds and exit")
    ap.add_argument("--predict", default=None, help="path to an image file to classify")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)

    if args.benchmark:
        benchmark(args)
        return
    if args.predict:
        model, classes = load_model(args.out)
        for name, p in predict_image(model, classes, args.predict, top_k=5):
            print(f"{name:<28} {p:.1%}")
        return

    torch.manual_seed(args.seed)
    data_dir = Path(args.data_dir)
    train_files = parquet_files(data_dir, "train")
    val_files = parquet_files(data_dir, "validation")
    if not train_files or not val_files:
        raise SystemExit(f"No parquet files under {data_dir / 'data'}.\n"
                         f"Run first:  python scripts/download_data.py --v4")

    classes = load_class_names(train_files[0])
    if len(classes) != N_CLASSES:
        print(f"Warning: metadata lists {len(classes)} names but labels are 0..{N_CLASSES - 1}; "
              f"keeping the first {N_CLASSES}. Dropped: {classes[N_CLASSES:]}")
        classes = classes[:N_CLASSES]
    print(f"Class names: first {classes[:3]} ... last {classes[-3:]}")
    if not classes[0].lower().startswith("tench"):
        print("WARNING: the first class is not 'tench' - labels may be shifted. Check the dataset before trusting results.")

    dim = BACKBONES[args.backbone]
    n_est = args.per_class * N_CLASSES * (2 if args.flip_train else 1)
    print(f"Backbone {args.backbone} ({dim}-d) | {SRC_SIZE} -> {args.img_size} px | {args.per_class}/class"
          f"{' + flips' if args.flip_train else ''} | train features ~{n_est * dim * 2 / 1e9:.1f} GB (x2 while loading)")

    tf = make_tf(args.img_size)
    backbone = build_backbone(args.backbone)
    cache_root = data_dir.parent / "feature_cache_v6"
    base = f"{args.backbone}_s{args.img_size}"
    train_f, train_y = build_features(
        train_files, cache_root / f"train_{base}_n{args.per_class}_f{int(args.flip_train)}",
        backbone, tf, dim, "train features", per_class=args.per_class, flip=args.flip_train)
    val_all_f, val_all_y = build_features(
        val_files, cache_root / f"val_{base}_n{args.val_per_class}",
        backbone, tf, dim, "val features", per_class=args.val_per_class, flip=False)
    if args.extract_only:
        print("Features cached. Run again without --extract-only to train the head.")
        return

    # official val images are split in two: model selection (val) and the final honest score (test)
    perm = torch.randperm(len(val_all_y), generator=torch.Generator().manual_seed(args.seed))
    half = len(perm) // 2
    val_f, val_y = val_all_f[perm[:half]], val_all_y[perm[:half]]
    test_f, test_y = val_all_f[perm[half:]], val_all_y[perm[half:]]
    print(f"Examples: train {len(train_y)}, val {len(val_y)}, test {len(test_y)}")

    hparams = dict(in_dim=dim, hidden=args.hidden, dropout=args.dropout, norm=True, in_drop=args.in_drop)
    head = ClassifierHead(len(classes), **hparams)
    print(f"Trainable head parameters: {sum(p.numel() for p in head.parameters()):,}  (backbone frozen)")
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    train_loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    eval_loss_fn = nn.CrossEntropyLoss()

    best_loss, best_epoch, best_state, wait = float("inf"), 0, None, 0
    for epoch in range(1, args.epochs + 1):
        head.train()
        total, nb = 0.0, 0
        for idx in torch.randperm(len(train_y)).split(args.batch):
            x, y = train_f[idx].float(), train_y[idx]
            opt.zero_grad(set_to_none=True)
            loss = train_loss_fn(head(x), y)
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        sched.step()
        m = evaluate_head(head, val_f, val_y, eval_loss_fn)
        improved = m["loss"] < best_loss - args.min_delta
        if improved:
            best_loss, best_epoch, wait = m["loss"], epoch, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            wait += 1
        print(f"Epoch {epoch:2d} | train loss {total / nb:.4f} | val loss {m['loss']:.4f} | "
              f"val top-1 {m['acc']:.4f} | top-5 {m['top5']:.4f}{'  *' if improved else ''}")
        if wait >= args.patience:
            print(f"Early stopping: val loss has not improved for {args.patience} epochs.")
            break
    print(f"Best epoch: {best_epoch}")

    head.load_state_dict(best_state)
    t = evaluate_head(head, test_f, test_y, eval_loss_fn)
    print(f"\nTest top-1: {t['acc']:.4f} | top-5: {t['top5']:.4f}  "
          f"(chance: {1 / len(classes):.4f} / {5 / len(classes):.4f})")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "classes": classes, "hparams": hparams,
                "img_size": args.img_size, "backbone": args.backbone}, args.out)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()