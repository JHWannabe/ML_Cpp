import numpy as np
import cv2
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

pipe = None

def load_model():
    global pipe
    if pipe is None:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16,
            safety_checker=None,              # NSFW 체크 비활성화 :contentReference[oaicite:0]{index=0}
            requires_safety_checker=False     # 추가 경고 억제 :contentReference[oaicite:1]{index=1}
        ).to("cuda")
        pipe.set_progress_bar_config(disable=True)

def anomaly_source(file_path) -> np.ndarray:
    load_model()

    img = cv2.imread(file_path)
    cv2.resize(img, (512, 512), img)  # 이미지 크기를 512x512로 조정
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB로 변환

    img = img.astype(np.uint8)
    pil_img = Image.fromarray(img).convert("RGB")
    w, h = pil_img.size
    pil_img = pil_img.resize((w // 2, h //2))  

    prompt = "Crack, Scratch"
    negative_prompt = "clean, smooth, flawless"

    result = pipe(
        prompt=prompt,
        image=pil_img,
        strength=0.75,
        guidance_scale=7.5,
        negative_prompt=negative_prompt
    ).images[0]

    result = result.resize((img.shape[1], img.shape[0]))
    result_np = np.array(result)

    return result_np
