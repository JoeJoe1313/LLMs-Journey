from mlx_vlm import apply_chat_template, generate, load
from mlx_vlm.utils import load_image

model_id = "mlx-community/medgemma-4b-it-bf16"

model, processor = load(model_id)
config = model.config

image_url = (
    "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
)
image = load_image(image_url)

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are an expert radiologist."}],
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this X-ray"},
        ],
    },
]

formatted_prompt = apply_chat_template(processor, config, messages, num_images=1)

response = generate(model, processor, formatted_prompt, image, verbose=False)
print(response)
