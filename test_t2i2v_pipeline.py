import time
import os
import requests
import json
from gradio_client import Client, handle_file

print("=== TESTING TEXT -> IMAGE -> VIDEO PIPELINE ===")
t0 = time.time()

prompt = "A majestic dragon flying over snow-covered mountain peaks during sunset"

# Step 1: Text to Image via Pollinations / Flux (takes ~3s)
print("\n[Step 1] Generating reference image from text prompt...")
t1 = time.time()
encoded_prompt = requests.utils.quote(f"{prompt}, 8k masterpiece, cinematic")
img_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true&model=flux"

img_path = "temp_t2v_ref.jpg"
res = requests.get(img_url, timeout=30)
if res.status_code == 200:
    with open(img_path, "wb") as f:
        f.write(res.content)
    print(f"-> Reference Image generated in {time.time() - t1:.2f}s | Saved: {img_path}")
else:
    print(f"-> Image gen failed: {res.status_code}")
    exit(1)

# Step 2: Image to Video via Wan 2.2 Lightning (takes ~35s)
print("\n[Step 2] Animating image into video via Wan 2.2 Lightning...")
t2 = time.time()
token = os.getenv("HF_TOKEN", "").strip()
space_url = "https://saravutw-wan2-2-i2v-lightning-4-8step-custom.hf.space"

client = Client(space_url, token=token, httpx_kwargs={"timeout": 180})
video_res = client.predict(
    handle_file(img_path),
    handle_file(img_path),
    prompt,
    4,
    "blurry, low quality, chaotic, deformed, watermark, bad anatomy, shaky camera view point",
    3.5,
    1.0,
    1.0,
    42,
    True,
    5,
    "UniPCMultistep",
    3.0,
    16,
    False,
    True,
    api_name="/generate_video"
)

t3 = time.time()
print(f"-> Video Animation finished in {t3 - t2:.2f}s!")
print(f"-> Video Output: {video_res}")
print(f"\nTOTAL PIPELINE ELAPSED TIME: {t3 - t0:.2f}s!")
