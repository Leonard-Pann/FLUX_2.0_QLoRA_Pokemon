import os
import base64
from pathlib import Path
from openai import OpenAI
import json

client = OpenAI(
    base_url="http://192.168.130.184:8000/v1",
    api_key="sk-rJ606F3zlzJMTCkLPNk9N5d7xRSqcPzr"
)

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct" 

IMAGE_DIRECTORY = "./images/1024x1024"
CAPTION_DIRECTORY = "./captions.json"
PROMPT = (
    "Describe this image concisely in 2 to 3 sentences. "
    "First, start the caption exactly with this phrase: 'pkmnstyle, a cel-shaded anime-style illustration on a plain white background with flat colors and clean outlines.' "
    "Next, explicitly state what real-world animal, object, food, or mythical entity the subject most closely resembles (e.g., 'a dog-like creature', 'a floating fish', 'a humanoid blob of whipped cream', 'a mushroom-like creature'). "
    "Provide a literal, objective description of its anatomy. Describe its body type or posture carefully—state if it is bipedal, quadrupedal, floating, swimming, serpentine, or an amorphous shape. "
    "CRITICAL RULE: Do NOT hallucinate limbs. Look closely at the image. If it has no legs, state that it is legless. If it stands on a single base, tail, or stem, do not call it bipedal or quadrupedal. "
    "Detail its primary colors and prominent features like tails, horns, fins, or extra appendages. "
    "Keep it highly factual and direct."
)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_images():
    valid_extensions = { ".jpg", ".jpeg", ".png", ".webp" }
    image_dir_path = Path(IMAGE_DIRECTORY)

    if not image_dir_path.exists():
        print(f"Error: Directory {IMAGE_DIRECTORY} not found.")
        return

    captions: dict[str, str] = {}
    for image_path in image_dir_path.iterdir():
        if image_path.suffix.lower() in valid_extensions:
            try:
                base64_image = encode_image(image_path)
                mime_type: str = "image/png"

                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": PROMPT},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{base64_image}"
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens= 250,
                    temperature= 0.2
                )
                
                img_name: str = image_path.name.removesuffix(".png")
                caption = f"{response.choices[0].message.content.strip()}" 
                captions[img_name] = caption

                print(f"Processed {image_path.name}.")
                with open(CAPTION_DIRECTORY, encoding="utf-8", mode="w") as f:
                    f.write(json.dumps(captions, indent=4))

            except Exception as e:
                print(f"Failed to process {image_path.name}. Error: {e}")

process_images()