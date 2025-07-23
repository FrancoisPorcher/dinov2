import torch
import torchvision.datasets as datasets
from torchvision.transforms.v2 import Compose, RandomHorizontalFlip, ToDtype, Lambda, Resize
from torchvision.transforms.v2.functional import pil_to_tensor
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

import argparse
import os


def get_train_transform():
    return Compose(
        [
            pil_to_tensor,  # PIL → uint8 Tensor in [0..255]
            Resize((args.image_resolution, args.image_resolution)),
            RandomHorizontalFlip(p=0.5),  # 50% chance horizontal flip
            ToDtype(torch.float32),  # uint8 → float32, still [0..255]
            Lambda(lambda t: t / 255.0),  # now [0..1]
        ]
    )


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.eval()
    model.to(device)

    print(f"device used {device}")
    imagenet_dir = "/datasets01/imagenet_full_size/061417"
    transform_train = get_train_transform()
    dataset_train = datasets.ImageFolder(imagenet_dir, transform=transform_train)
    train_loader = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    
    output_dir = os.path.join(args.output_dir, f"resolution_{args.image_resolution}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving files at {output_dir}")

    with torch.no_grad():
        for batch_idx, (tensors, labels) in enumerate(
            tqdm(train_loader, desc="Extracting embeddings", total=len(train_loader))
        ):
            tensors = tensors.to(device)
            outputs = model.forward_features(tensors)
            x_norm_patchtokens = outputs["x_norm_patchtokens"].cpu()
            labels = labels.cpu()
            
            print(f"x_norm_patchtokens device {x_norm_patchtokens.device}")
            print(f"labels device {labels.device}")
            
            batch_file = os.path.join(output_dir, f"batch_{batch_idx:05d}.pt")
            print(f"batch file {batch_file}")
            torch.save({'embeddings': x_norm_patchtokens, 'labels': labels}, batch_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_resolution", type = int, default = 56, help= "Image resolution for resizing")
    parser.add_argument("--batch_size", type=int, default = 128, help = "Batch size to process embeddings")
    parser.add_argument("--output_dir", type=str, default = "/private/home/francoisporcher/imagenet_embeddings", help = "Path where to save image embeddings")
    args = parser.parse_args()
    # run with python -m francois.scripts.extract_imagenet_embeddings
    main(args)
