import os
import torch
import random
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
from torchvision.transforms.functional import InterpolationMode
from torchvision.io.image import ImageReadMode
from tqdm import tqdm

def resize_dataset(raw_dir: str = "./images/raw", target_dir: str = "./images/1024x1024") -> None:
    """
    Reads images from the raw directory, resizes them to 1024x1024,
    and saves them to the target directory.
    """
    os.makedirs(target_dir, exist_ok=True)
    valid_extensions = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    files = [f for f in os.listdir(raw_dir) if f.lower().endswith(valid_extensions)]

    trans_resize = torchvision.transforms.Resize((1024, 1024), InterpolationMode.BICUBIC, antialias=True)

    for filename in tqdm(files, desc="Resizing images"):
        img_path: str = os.path.join(raw_dir, filename)
        raw_image = torchvision.io.read_image(img_path, ImageReadMode.RGB_ALPHA)
        
        alpha = raw_image[3:4, :, :].float() / 255.0
        rgb = raw_image[:3, :, :].float()
        white_bg = torch.ones_like(rgb) * 255.0
        
        flat_image = (rgb * alpha + white_bg * (1.0 - alpha)).to(torch.uint8)

        output_image: torch.Tensor = trans_resize(flat_image).cpu() 
        torchvision.io.write_png(output_image, os.path.join(target_dir, filename), 0)


def load_dataset(image_dir: str = "./images/1024x1024") -> torch.Tensor:
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),  # Scales pixels to [0, 1]
        torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Scales to [-1, 1]
    ])

    image_tensors = []
    valid_extensions = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

    for filename in os.listdir(image_dir):
        if filename.lower().endswith(valid_extensions):
            img_path = os.path.join(image_dir, filename)
            img = Image.open(img_path).convert("RGB")
            image_tensors.append(transform(img))

    if not image_tensors:
        raise ValueError(f"No valid images found in {image_dir}")

    return torch.stack(image_tensors)

def test_dataset(image_dir: str = "./images/1024x1024") -> None:
    images = load_dataset(image_dir)
    idx = random.randint(0, len(images) - 1)
    img_tensor = images[idx]

    img_to_show = (img_tensor * 0.5 + 0.5).permute(1, 2, 0).numpy()

    plt.imshow(img_to_show)
    plt.title(f"Random Pokémon Sample (Index: {idx})")
    plt.axis("off")
    plt.show()

test_dataset()