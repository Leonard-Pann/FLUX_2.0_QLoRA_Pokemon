Welcome to the **Pokémon Flux LoRA Trainer**. This project provides a complete Python pipeline for preparing a dataset, captioning, and fine-tuning a Low-Rank Adaptation (LoRA) on the Flux text-to-image architecture to generate high-quality, stylistically accurate Pokémon.

Because the Flux model has a massive parameter count (12B+), this repository leverages `accelerate` and quantization techniques to make LoRA training accessible on consumer GPUs (32GB VRAM recommended).

## ✨ Features
* **Optimized for Flux:** Custom config files specifically tuned for Flux's Flow Matching and transformer architecture.
* **Memory Efficient:** Supports 8-bit AdamW and gradient checkpointing to squeeze training into a 24GB RTX 3090/4090.
* **Plug-and-Play Inference:** Clean Python scripts to test your LoRA immediately after training.

## 💻 Prerequisites & Hardware
* **OS:** Windiws 11
* **GPU:** RTX 5090 32GB
* **Python:** 3.13.x
* **CUDA:** 12.8+

## 🛠️ Installation

Clone the repository and set up your virtual environment:
```bash
git clone [https://github.com/yourusername/pokemon-flux-lora.git](https://github.com/yourusername/pokemon-flux-lora.git)
cd pokemon-flux-lora

python -m venv .venv
.venv\Scripts\activate
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -e .