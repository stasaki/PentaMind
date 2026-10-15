#!/usr/bin/env python
"""
PentaMind: predict global cognition from drawings of intersecting pentagons.

Runs the 10 released VGG19-bn models on every image in a folder and averages them.
Works on CPU or GPU with a current PyTorch. Model weights are downloaded from
Hugging Face (stasaking/pentamind) on first use, or read from --models_dir.

Examples:
    # score a folder of images (pentagon already fills most of the image)
    python pentamind_predict.py --images my_images/ --out scores.csv

    # whole-page scans: crop to the drawing first
    python pentamind_predict.py --images scans/ --out scores.csv --crop

    # same settings as the paper (30 test-time augmentations; 30x slower)
    python pentamind_predict.py --images my_images/ --out scores.csv --n_tta 30

    # also save image features for your own model (e.g. ridge regression)
    python pentamind_predict.py --images my_images/ --out scores.csv --features features.npz

Output CSV columns:
    file_name        image file name
    pentamind_score  mean of the 10 models = predicted global cognition (higher = better)
    sd_models        standard deviation across the 10 models (disagreement)
    model_1..10      score of each model (only with --per_model)

Feature file (--features, numpy .npz), each array has shape (n_models, n_images, dim):
    gap4  conv block 4 output, averaged over the image (512)
    gap5  conv block 5 output, averaged over the image (512)
    fc2   second-to-last dense layer (4096)
    file_names  image order
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch import nn

HF_REPO = "stasaking/pentamind"
N_MODELS = 10
INPUT_SIZE = 224
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)  # ImageNet normalisation,
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)   # as in training

Image.MAX_IMAGE_PIXELS = None  # page scans can be large


# ---------------------------------------------------------------- images

def to_rgb(im):
    """Any PIL mode -> 8-bit RGB on a white background."""
    if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        a = np.asarray(im, dtype=np.float32)
        a = 255 * (a - a.min()) / max(a.max() - a.min(), 1e-6)
        im = Image.fromarray(a.astype(np.uint8))
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        im = Image.alpha_composite(Image.new("RGBA", im.size, (255, 255, 255, 255)), im)
    return im.convert("RGB")


def drawing_box(im, ink_threshold=200, edge_ignore=0.02):
    """Box (x0, y0, x1, y1) around dark pixels, ignoring page edges and rare specks."""
    g = np.asarray(im.convert("L"))
    ink = g < ink_threshold
    h, w = ink.shape
    ey, ex = int(h * edge_ignore), int(w * edge_ignore)
    ink[:ey], ink[h - ey:], ink[:, :ex], ink[:, w - ex:] = False, False, False, False
    ys, xs = np.nonzero(ink)
    if len(xs) < 20:
        return None
    x0, x1 = np.percentile(xs, [0.2, 99.8])
    y0, y1 = np.percentile(ys, [0.2, 99.8])
    return int(x0), int(y0), int(x1) + 1, int(y1) + 1


def pad_square(im, box=None, margin=0.10):
    """Pad (or crop to box + margin, then pad) with white to a square."""
    x0, y0, x1, y1 = box if box else (0, 0, im.width, im.height)
    side = max(x1 - x0, y1 - y0)
    if box:
        side = int(side * (1 + 2 * margin))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left, top = int(cx - side / 2), int(cy - side / 2)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    src = im.crop((max(left, 0), max(top, 0), min(left + side, im.width), min(top + side, im.height)))
    canvas.paste(src, (max(-left, 0), max(-top, 0)))
    return canvas


def load_image(path, crop):
    """Read one image and return a 224x224 RGB uint8 array (white background)."""
    im = to_rgb(Image.open(path))
    box = drawing_box(im) if crop else None
    if crop and box is None:
        print(f"  warning: no drawing found in {path.name}; using whole image", file=sys.stderr)
    im = pad_square(im, box)
    # torchvision Resize((224, 224)) on a PIL image, as in training = PIL bilinear
    return np.asarray(im.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR))


def to_tensor(batch_uint8):
    x = torch.from_numpy(np.ascontiguousarray(batch_uint8)).permute(0, 3, 1, 2).float() / 255
    return (x - MEAN) / STD


def build_augmenter():
    """Test-time augmentation, same settings as functions/PentaMind-pred_ave.py."""
    if not hasattr(np, "sctypes"):  # imgaug 0.4 still uses np.sctypes (removed in numpy 2)
        np.sctypes = {"float": [np.float16, np.float32, np.float64],
                      "int": [np.int8, np.int16, np.int32, np.int64],
                      "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
                      "complex": [np.complex64, np.complex128], "others": [bool, object, bytes, str]}
    try:
        import imgaug
        from imgaug import augmenters as iaa
    except ImportError:
        sys.exit("--n_tta > 1 needs imgaug:  pip install imgaug")
    geom = iaa.Sequential([
        iaa.Fliplr(0.5), iaa.Flipud(0.5), iaa.Rot90((0, 3)),
        iaa.Sometimes(0.9, iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                                      rotate=(-10, 10), cval=255)),
    ], random_order=True)
    look = iaa.SomeOf((0, 3), [iaa.GammaContrast((0, 2.0)), iaa.GaussianBlur(sigma=(0, 3.0)),
                               iaa.Sharpen((0, 1))], random_order=True)

    def augment(images, seed):
        imgaug.random.seed(seed)
        return np.stack([look.augment_image(geom.augment_image(im)) for im in images])
    return augment


# ---------------------------------------------------------------- models

def model_path(i, models_dir):
    """Local file if present (fp32 or fp16 copy), else download from Hugging Face."""
    if models_dir:
        for name in (f"vgg19_bn-{i}_model.pth", f"vgg19_bn-{i}_model_fp16.pth"):
            if (Path(models_dir) / name).exists():
                return Path(models_dir) / name
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit(f"Model {i} not found in --models_dir. Install huggingface_hub to download it, "
                 f"or download it from https://huggingface.co/{HF_REPO}")
    return Path(hf_hub_download(HF_REPO, f"models/vgg19_bn-{i}_model.pth"))


def build_net():
    """VGG19 with batch norm; last layer replaced by one output (regression)."""
    net = torchvision.models.vgg19_bn(weights=None)
    net.classifier[6] = nn.Linear(net.classifier[6].in_features, 1)
    return net


def load_weights(net, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = {k.removeprefix("module."): v.float() for k, v in state.items()}
    net.load_state_dict(state)


@torch.inference_mode()
def run(net, x, want_features):
    """Return score (n,) and, if asked, gap4 / gap5 / fc2 features."""
    pools = [i for i, m in enumerate(net.features) if isinstance(m, nn.MaxPool2d)]
    feats = {}
    for i, layer in enumerate(net.features):
        x = layer(x)
        if want_features and i == pools[3]:
            feats["gap4"] = x.mean(dim=(2, 3))
        elif want_features and i == pools[4]:
            feats["gap5"] = x.mean(dim=(2, 3))
    x = torch.flatten(net.avgpool(x), 1)
    for i, layer in enumerate(net.classifier):
        x = layer(x)
        if want_features and i == 4:  # Linear, ReLU, Dropout, Linear, ReLU(4)
            feats["fc2"] = x
    return x[:, 0], feats


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="folder with images (png/jpg/tif/bmp)")
    ap.add_argument("--out", required=True, help="output CSV file")
    ap.add_argument("--models_dir", default=None,
                    help="folder with vgg19_bn-<i>_model.pth; default: download from Hugging Face")
    ap.add_argument("--crop", action="store_true",
                    help="crop each image to the drawing (dark pixels) before padding to a square")
    ap.add_argument("--n_tta", type=int, default=1,
                    help="passes per image: 1 = original image only (default); the paper used 30 "
                         "(1 original + 29 randomly augmented copies, averaged)")
    ap.add_argument("--features", default=None, help="also save gap4/gap5/fc2 features to this .npz")
    ap.add_argument("--per_model", action="store_true", help="add one column per model to the CSV")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    files = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in IMAGE_EXT)
    if not files:
        sys.exit(f"No images found in {args.images}")
    print(f"{len(files)} images, device {args.device}, test-time augmentation passes: {args.n_tta}")
    images = np.stack([load_image(p, args.crop) for p in files])
    augment = build_augmenter() if args.n_tta > 1 else None

    net = build_net().to(args.device).eval()
    scores = np.zeros((N_MODELS, len(files)), dtype=np.float32)
    feats = {k: [] for k in ("gap4", "gap5", "fc2")} if args.features else None
    for m in range(N_MODELS):
        path = model_path(m + 1, args.models_dir)
        print(f"model {m + 1}/{N_MODELS}: {path.name}", flush=True)
        load_weights(net, path)
        net.to(args.device)
        passes, model_feats = [], {k: [] for k in ("gap4", "gap5", "fc2")}
        for t in range(args.n_tta):
            # pass 0 = original image; passes 1.. = augmented with a fixed seed (reproducible)
            imgs = images if t == 0 else augment(images, seed=t)
            out = []
            for s in range(0, len(files), args.batch_size):
                y, f = run(net, to_tensor(imgs[s:s + args.batch_size]).to(args.device),
                           want_features=feats is not None and t == 0)
                out.append(y.float().cpu().numpy())
                for k, v in f.items():
                    model_feats[k].append(v.float().cpu().numpy())
            passes.append(np.concatenate(out))
        scores[m] = np.mean(passes, axis=0)
        if feats is not None:
            for k in feats:
                feats[k].append(np.concatenate(model_feats[k]))

    res = pd.DataFrame({"file_name": [p.name for p in files],
                        "pentamind_score": scores.mean(axis=0),
                        "sd_models": scores.std(axis=0)})
    if args.per_model:
        for m in range(N_MODELS):
            res[f"model_{m + 1}"] = scores[m]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    print(f"wrote {args.out}")
    if feats is not None:
        np.savez_compressed(args.features, file_names=np.array([p.name for p in files]),
                            **{k: np.stack(v) for k, v in feats.items()})
        print(f"wrote {args.features} (features of the original image, no augmentation)")


if __name__ == "__main__":
    main()
