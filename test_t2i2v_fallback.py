import time
import os
import requests
from gradio_client import Client, handle_file

print("=== TESTING T2I2V FALLBACK PIPELINE ===")
prompt = "A majestic eagle flying over snowy mountains at sunset, cinematic"
job_id = "test_t2i2v_job"
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Step 1: Generate reference image from Pollinations (3s)
print("\n[Step 1] Generating reference image via Pollinations...")
t0 = time.time()
encoded_prompt = requests.utils.quote(prompt)
img_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true&model=flux"

ref_img_path = os.path.join(output_dir, f"ref_{job_id}.jpg")
res = requests.get(img_url, timeout=30)
if res.status_code == 200:
    with open(ref_img_path, "wb") as f:
        f.write(res.content)
    print(f" -> Reference Image Created in {time.time() - t0:.2f}s | Path: {ref_img_path}")
else:
    print(f" -> Failed to generate reference image: HTTP {res.status_code}")
    exit(1)

# Step 2: Animate reference image via Wan 2.2 Lightning Space (35s)
print("\n[Step 2] Animating reference image via Wan 2.2 Lightning Space...")
t1 = time.time()
space_url = "https://saravutw-wan2-2-i2v-lightning-4-8step-custom.hf.space"

try:
    client = Client(space_url, token=None, httpx_kwargs={"timeout": 180})
    result = client.predict(
        handle_file(ref_img_path),
        handle_file(ref_img_path),
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
    print(f" -> Video Generation Succeeded in {time.time() - t1:.2f}s!")
    print(f" -> Result: {result}")
except Exception as e:
    print(f" -> Error during Wan 2.2 Lightning call: {e}")
