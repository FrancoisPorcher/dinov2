import torch
torch.backends.cudnn.benchmark = True
import torchvision.datasets as datasets
from torchvision.transforms.v2 import Compose, RandomHorizontalFlip, ToDtype, Lambda, Resize
from torchvision.transforms.v2.functional import pil_to_tensor
from torch.utils.data import DataLoader, DistributedSampler

from tqdm.auto import tqdm

import argparse
import os
import shutil

# distributed inference
import torch.distributed as dist

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


def main_worker(args):
    local_rank=int(os.environ.get("LOCAL_RANK"))
    world_size=int(os.environ.get("WORLD_SIZE"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend = "nccl")
    print(f"[RANK {local_rank}] World size: {world_size}")
    device = torch.device(f"cuda:{local_rank}")
    print(f"device used {device}")

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.eval()
    model.to(device)

    imagenet_dir = "/datasets01/imagenet_full_size/061417"
    transform_train = get_train_transform()
    dataset_train = datasets.ImageFolder(imagenet_dir, transform=transform_train)
    sampler = DistributedSampler(dataset = dataset_train, num_replicas=world_size, rank=local_rank, shuffle=False)
    train_loader = DataLoader(dataset=dataset_train, batch_size=args.batch_size, pin_memory=True, sampler=sampler)
    
    # Output dir: add rank for uniqueness, remove if it exists
    output_dir = os.path.join(args.output_dir, f"resolution_{args.image_resolution}")
    if local_rank==0:
        if os.path.exists(output_dir):
            print(f"[RANK {local_rank}] Removing existing output dir: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            os.makedirs(output_dir, exist_ok=True)
    print(f"[RANK {local_rank}] Saving files at {output_dir}")

    with torch.no_grad():
        for batch_idx, (tensors, labels) in enumerate(
            tqdm(train_loader, desc="Extracting embeddings", total=len(train_loader))
        ):
            tensors = tensors.to(device)
            outputs = model.forward_features(tensors)
            x_norm_patchtokens = outputs["x_norm_patchtokens"].cpu()
            labels = labels.cpu()
                        
            batch_file = os.path.join(output_dir, f"batch_{batch_idx:05d}_rank_{local_rank}.pt")
            torch.save({'embeddings': x_norm_patchtokens, 'labels': labels}, batch_file)
            
    dist.barrier()              # make sure everyone finished writing
    dist.destroy_process_group()
    if local_rank == 0:
        print("✓ Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_resolution", type = int, default = 56, help= "Image resolution for resizing")
    parser.add_argument("--batch_size", type=int, default = 128, help = "Batch size to process embeddings")
    parser.add_argument("--output_dir", type=str, default = "/private/home/francoisporcher/imagenet_embeddings", help = "Path where to save image embeddings")
    args = parser.parse_args()
    # or with torchrun --nproc_per_node=2 francois/scripts/extract_imagenet_embeddings.py

    main_worker(args)
