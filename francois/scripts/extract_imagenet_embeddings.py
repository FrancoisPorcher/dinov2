import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dinov2.hub import dinov2_vitb14
from dinov2.data.datasets import ImageNet
from dinov2.data.transforms import make_classification_eval_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract ImageNet embeddings with DINOv2")
    parser.add_argument(
        "--imagenet-root",
        required=True,
        help="Path to the ImageNet root directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("features_output"),
        help="Directory where features will be stored",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size used for the DataLoader",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    transform = make_classification_eval_transform()
    dataset = ImageNet(
        split=ImageNet.Split.VAL,
        root=args.imagenet_root,
        extra=args.imagenet_root,
        transform=transform,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = dinov2_vitb14(pretrained=True)
    model.to(device)
    model.eval()

    features = []
    labels = []
    with torch.no_grad():
        for imgs, target in dataloader:
            imgs = imgs.to(device)
            feats = model(imgs)
            features.append(feats.cpu())
            labels.append(target)

    features = torch.cat(features)
    labels = torch.cat(labels)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"features": features, "labels": labels}, args.output_dir / "imagenet_features.pt")


if __name__ == "__main__":
    main(parse_args())
