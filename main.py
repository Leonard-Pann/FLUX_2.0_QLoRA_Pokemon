import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
from diffusers import AutoencoderKL, FluxTransformer2DModel
from peft import LoraConfig, get_peft_model
from typing import List, Dict, Any

def train_pokemon_flux_lora(
    images: torch.Tensor,
    captions: List[str],
    vae_id: str = "black-forest-labs/flux-2-vae",
    text_encoder_id: str = "Qwen/Qwen3-4B",
    transformer_id: str = "black-forest-labs/flux-2-8b-fp8",
    batch_size: int = 4,
    epochs: int = 20,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    lora_rank: int = 16,
    lora_alpha: int = 32
) -> None:
    
    # 1. Device and Precision Setup
    # The RTX 5090 excels at bfloat16 computation.
    weight_dtype: torch.dtype = torch.bfloat16
    torch_device: torch.device = torch.device(device)
    
    print("Loading tokenizer and text encoder (Qwen 3)...")
    # 2. Load Text Encoder and Tokenizer
    tokenizer: Any = AutoTokenizer.from_pretrained(text_encoder_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # We load the text encoder in bfloat16 but keep it frozen.
    text_encoder: nn.Module = AutoModel.from_pretrained(
        text_encoder_id, 
        torch_dtype=weight_dtype, 
        trust_remote_code=True
    ).to(torch_device)
    text_encoder.eval()
    text_encoder.requires_grad_(False)

    print("Loading VAE...")
    # 3. Load VAE
    vae: nn.Module = AutoencoderKL.from_pretrained(
        vae_id, 
        torch_dtype=weight_dtype
    ).to(torch_device)
    vae.eval()
    vae.requires_grad_(False)

    print("Loading Flux 2.0 Transformer in 8-bit...")
    # 4. Load Flux DiT Backbone in 8-bit
    # We use load_in_8bit to leverage bitsandbytes Q8 quantization natively via Hugging Face
    transformer: nn.Module = FluxTransformer2DModel.from_pretrained(
        transformer_id,
        load_in_8bit=True,
        device_map=device
    )
    transformer.requires_grad_(False) # Freeze base model

    print("Injecting LoRA adapters...")
    # 5. Configure and Apply LoRA
    # We target the attention projections and feed-forward linear layers of the DiT blocks
    lora_config: LoraConfig = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0", "proj_mlp", "proj_out"],
        init_lora_weights="gaussian"
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.train()

    # 6. Optimizer Setup
    # We only pass the parameters that require gradients (the LoRA weights)
    trainable_params: List[nn.Parameter] = [p for p in transformer.parameters() if p.requires_grad]
    optimizer: torch.optim.AdamW = torch.optim.AdamW(
        trainable_params, 
        lr=learning_rate, 
        weight_decay=1e-4
    )

    print("Preparing dataset and dataloader...")
    # 7. Dataset Preparation
    # images is expected to be shape (N, 3, 1024, 1024) and normalized between [-1, 1]
    dataset: TensorDataset = TensorDataset(images)
    dataloader: DataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Starting Training Loop...")
    # 8. Training Loop
    global_step: int = 0
    
    for epoch in range(epochs):
        epoch_loss: float = 0.0
        
        for step, (batch_images,) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Current batch size dynamically extracted
            bsz: int = batch_images.shape[0]
            
            # Move images to device and cast to bf16
            batch_images = batch_images.to(torch_device, dtype=weight_dtype)
            
            # --- A. Encode Images to Latents ---
            # Flux VAE expects inputs in [-1, 1] and outputs latents. We apply the standard scaling factor.
            vae_scaling_factor: float = 0.18215 # Default flux scaling, adjust if Flux 2.0 differs
            with torch.no_grad():
                latents: torch.Tensor = vae.encode(batch_images).latent_dist.sample()
                latents = latents * vae_scaling_factor
            
            # --- B. Process Text Captions ---
            # Slice captions for the current batch
            batch_captions: List[str] = captions[step * batch_size : step * batch_size + bsz]
            
            text_inputs: Dict[str, torch.Tensor] = tokenizer(
                batch_captions,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt"
            ).to(torch_device)
            
            with torch.no_grad():
                # Extract hidden states from Qwen text encoder
                encoder_outputs: Any = text_encoder(**text_inputs, output_hidden_states=True)
                # Usually, diffusion models use the penultimate hidden state or pooled output
                encoder_hidden_states: torch.Tensor = encoder_outputs.hidden_states[-2] 
                
                # Create a mock pooled projection if the specific Flux pipeline requires it
                # (Flux often concatenates pooled CLIP embeddings and T5/Qwen sequences)
                pooled_projections: torch.Tensor = encoder_hidden_states[:, 0, :] 
            
            # --- C. Flow Matching Setup ---
            # Sample pure noise x_0
            noise: torch.Tensor = torch.randn_like(latents)
            
            # Sample random timesteps t from Uniform(0, 1)
            timesteps: torch.Tensor = torch.rand((bsz,), device=torch_device, dtype=weight_dtype)
            
            # Reshape timesteps for broadcasting: (bsz, 1, 1, 1)
            t_expand: torch.Tensor = timesteps.view(bsz, 1, 1, 1)
            
            # Interpolate between noise (x_0) and data (x_1)
            # x_t = (1 - t) * x_0 + t * x_1
            x_t: torch.Tensor = (1.0 - t_expand) * noise + t_expand * latents
            
            # Target velocity vector v
            # v = x_1 - x_0
            target_velocity: torch.Tensor = latents - noise
            
            # --- D. Forward Pass ---
            # Predict the velocity field
            model_pred: torch.Tensor = transformer(
                hidden_states=x_t,
                timestep=timesteps, # Usually raw float timesteps in DiTs
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections
            ).sample
            
            # --- E. Loss Calculation and Backprop ---
            # Mean Squared Error between predicted velocity and target velocity
            loss: torch.Tensor = nn.functional.mse_loss(model_pred, target_velocity, reduction="mean")
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            if global_step % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Step {global_step} | Loss: {loss.item():.4f}")
                
    print("Training complete! Saving LoRA weights...")
    transformer.save_pretrained("./pokemon_flux2_lora")

# Example invocation (assuming you have populated `my_pokemon_images` and `my_captions`):
# train_pokemon_flux_lora(images=my_pokemon_images, captions=my_captions)