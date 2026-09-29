"""
Image classifier (200 classes) - transfer learning, TRAINING-ONLY version.

Assumes rdf/data-v3/tiny-imagenet-200/ is already downloaded and extracted
(train/, val/, test/, wnids.txt, words.txt all present). No download or
extraction code here - if that folder is missing or incomplete, this will
error out rather than fetching it for you.

Same approach as before:
  1. Loads a frozen, pretrained MobileNetV2 backbone (feature extractor).
  2. Runs every image through it ONCE, caching the resulting features.
  3. Trains a small classifier head on those cached features - fast, since
     it's just matrix math on short feature vectors, not full CNN passes.

Requires:  pip install torchvision pillow

Project layout expected:
    imgclf-model/
      chatbot/imgclf_gui.py                  <- unchanged, works as-is
      scripts/train_imgclf_transfer_only.py  <- this file
      rdf/data-v3/tiny-imagenet-200/         <- already downloaded/extracted
      weights/imgclf_model.pt                <- created by training

Run from the project root folder:
    Train:        python scripts/train_imgclf_transfer_only.py
    Classify:     python scripts/train_imgclf_transfer_only.py --predict path/to/photo.jpg
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

ROOT = Path(__file__).resolve().parent.parent          # the project folder
DEFAULT_DATA = ROOT / "rdf" / "data-v3" / "tiny-imagenet-200"
DEFAULT_OUT = ROOT / "weights" / "imgclf_model.pt"

# Keep the pretrained backbone cache inside data-v3, same as before.
torch.hub.set_dir(str(DEFAULT_DATA.parent / "torch_cache"))

# ImageNet's standard mean/std - required to match what MobileNetV2 was
# pretrained on, NOT CIFAR's stats.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMG_SIZE = 96
FEATURE_DIM = 1280      # MobileNetV2's output feature width


# -------------------------------- data ------------------------------------ #
def load_class_names(data_dir: Path):
    wnids = [line.strip() for line in (data_dir / "wnids.txt").read_text().splitlines() if line.strip()]
    id_to_word = {}
    for line in (data_dir / "words.txt").read_text(encoding="utf-8").splitlines():
        wid, names = line.split("\t", 1)
        id_to_word[wid] = names.split(",")[0].strip()
    classes = [id_to_word.get(w, w) for w in wnids]
    wnid_to_idx = {w: i for i, w in enumerate(wnids)}
    return classes, wnid_to_idx


class TinyImageNetTrain(Dataset):
    """train/<wnid>/images/<wnid>_<n>.JPEG"""
    def __init__(self, data_dir: Path, wnid_to_idx, transform):
        self.samples = []
        self.transform = transform
        for wnid, idx in wnid_to_idx.items():
            img_dir = data_dir / "train" / wnid / "images"
            for p in img_dir.glob("*.JPEG"):
                self.samples.append((p, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        from PIL import Image
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


class TinyImageNetVal(Dataset):
    """val/images/val_<n>.JPEG, labeled via val_annotations.txt"""
    def __init__(self, data_dir: Path, wnid_to_idx, transform):
        self.samples = []
        self.transform = transform
        ann = (data_dir / "val" / "val_annotations.txt").read_text().splitlines()
        img_dir = data_dir / "val" / "images"
        for line in ann:
            fname, wnid = line.split("\t")[:2]
            if wnid in wnid_to_idx:
                self.samples.append((img_dir / fname, wnid_to_idx[wnid]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        from PIL import Image
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf


# ---------------------------- feature extraction --------------------------- #
def build_backbone():
    weights = MobileNet_V2_Weights.DEFAULT
    net = mobilenet_v2(weights=weights)
    backbone = nn.Sequential(
        net.features,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()
    return backbone


@torch.no_grad()
def extract_features(backbone, loader, device, label=""):
    feats, labels = [], []
    n_done = 0
    for x, y in loader:
        x = x.to(device)
        f = backbone(x)
        feats.append(f.cpu())
        labels.append(y)
        n_done += len(y)
        print(f"\r  {label}: {n_done} images", end="", flush=True)
    print()
    return torch.cat(feats), torch.cat(labels)


def get_or_build_features(cache_path: Path, backbone, dataset, device, label):
    if cache_path.exists():
        print(f"Using cached features: {cache_path}")
        d = torch.load(cache_path, weights_only=True)
        return d["features"], d["labels"]
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    feats, labels = extract_features(backbone, loader, device, label)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": feats, "labels": labels}, cache_path)
    return feats, labels


# -------------------------------- head model -------------------------------- #
class ClassifierHead(nn.Module):
    def __init__(self, n_classes, in_dim=FEATURE_DIM, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class TransferModel(nn.Module):
    """backbone (frozen) + trained head, used together at inference time."""
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        with torch.no_grad():
            f = self.backbone(x)
        return self.head(f)


def evaluate_head(head, loader, loss_fn, device):
    head.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = head(x)
            loss_sum += loss_fn(logits, y).item() * len(y)
            correct += (logits.argmax(-1) == y).sum().item()
            total += len(y)
    total = max(total, 1)
    return {"loss": loss_sum / total, "acc": correct / total}


# ------------------------------ predict / load ------------------------------ #
def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    backbone = build_backbone()
    head = ClassifierHead(len(ck["classes"]), **ck["hparams"])
    head.load_state_dict(ck["state_dict"])
    head.eval()
    model = TransferModel(backbone, head)
    model.eval()
    return model, ck["classes"]


def predict_image(model, classes, path, top_k=3):
    """Returns [(class_name, probability), ...] sorted best-first."""
    from PIL import Image
    _, eval_tf = build_transforms()
    img = Image.open(path).convert("RGB")
    x = eval_tf(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=-1)[0].tolist()
    ranked = sorted(zip(classes, probs), key=lambda p: p[1], reverse=True)
    return ranked[:top_k]


# -------------------------------- main --------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=20, help="set >= epochs to disable early stopping")
    ap.add_argument("--min-delta", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--predict", default=None, help="path to an image file to classify")
    args = ap.parse_args()

    if args.predict:
        model, classes = load_model(args.out)
        for name, p in predict_image(model, classes, args.predict):
            print(f"{name:<25} {p:.1%}")
        return

    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = "cpu"
    data_dir = Path(args.data_dir)

    if not (data_dir / "wnids.txt").exists():
        raise SystemExit(f"Dataset not found at {data_dir} - expected wnids.txt there. "
                          f"This script assumes the data is already downloaded and extracted.")

    classes, wnid_to_idx = load_class_names(data_dir)
    print(f"Classes: {len(classes)}")

    _, eval_tf = build_transforms()
    train_ds = TinyImageNetTrain(data_dir, wnid_to_idx, eval_tf)  # no aug needed for one-time features
    val_full = TinyImageNetVal(data_dir, wnid_to_idx, eval_tf)
    n_val = len(val_full) // 2
    val_ds = torch.utils.data.Subset(val_full, range(0, n_val))
    test_ds = torch.utils.data.Subset(val_full, range(n_val, len(val_full)))
    print(f"Examples: train {len(train_ds)}, val {len(val_ds)}, test {len(test_ds)}")

    backbone = build_backbone()

    cache_dir = data_dir.parent / "feature_cache"
    train_f, train_y = get_or_build_features(cache_dir / "train.pt", backbone, train_ds, device, "train features")
    val_f, val_y = get_or_build_features(cache_dir / "val.pt", backbone, val_ds, device, "val features")
    test_f, test_y = get_or_build_features(cache_dir / "test.pt", backbone, test_ds, device, "test features")

    tr = DataLoader(TensorDataset(train_f, train_y), batch_size=args.batch, shuffle=True)
    va = DataLoader(TensorDataset(val_f, val_y), batch_size=256)
    te = DataLoader(TensorDataset(test_f, test_y), batch_size=256)

    hparams = dict(hidden=256, dropout=0.3)
    head = ClassifierHead(len(classes), **hparams).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"Trainable head parameters: {n_params:,}  (backbone is frozen, not counted)")
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    loss_fn = nn.CrossEntropyLoss()

    best_loss, best_epoch, best_state, wait = float("inf"), 0, None, 0
    for epoch in range(1, args.epochs + 1):
        head.train()
        total = 0.0
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(head(x), y)
            loss.backward()
            opt.step()
            total += loss.item()
        m = evaluate_head(head, va, loss_fn, device)
        sched.step(m["loss"])
        improved = m["loss"] < best_loss - args.min_delta
        if improved:
            best_loss, best_epoch, wait = m["loss"], epoch, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            wait += 1
        print(f"Epoch {epoch:2d} | train loss {total / len(tr):.4f} | val loss {m['loss']:.4f} | "
              f"val acc {m['acc']:.4f}{'  *' if improved else ''}")
        if wait >= args.patience:
            print(f"Early stopping: val loss has not improved for {args.patience} epochs.")
            break
    print(f"Best epoch: {best_epoch}")

    head.load_state_dict(best_state)
    t = evaluate_head(head, te, loss_fn, device)
    print(f"\nTest accuracy: {t['acc']:.4f}  (chance level for {len(classes)} classes: {1/len(classes):.4f})")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "classes": classes, "hparams": hparams}, args.out)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()