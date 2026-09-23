import time
import os
import requests
import json
import base64
from gradio_client import Client, handle_file

print("=== STARTING IMAGE-TO-VIDEO PROFILING TEST ===")
start_time = time.time()

image_path = "test-prompt-image.jpg"
prompt = "A person walking on a mountain trail, cinematic motion"

if not os.path.exists(image_path):
    print(f"Error: image {image_path} not found")
    exit(1)

token = os.getenv("HF_TOKEN", "").strip()
replicate_key = os.getenv("REPLICATE_API_KEY") or "r8_8UHA1bKEwSXNOudUYXcep18keIY87Yd2gRW4F"

# Test Wan 2.2 Lightning Space
i2v_space_url = "https://saravutw-wan2-2-i2v-lightning-4-8step-custom.hf.space"
print(f"\n[Stage 1] Connecting to Space: {i2v_space_url}")
t1 = time.time()

try:
    client = Client(i2v_space_url, token=token, httpx_kwargs={"timeout": 300})
    print(f"-> Space Connected! Elapsed: {time.time() - t1:.2f}s")

    # Inspect endpoints
    print("\n[Endpoints Info]")
    try:
        endpoints = client.view_api(return_format="dict")
        print(json.dumps(endpoints, indent=2))
    except Exception as e:
        print(f"view_api error: {e}")

    print(f"\n[Stage 2] Submitting Job to Wan 2.2 Lightning...")
    t2 = time.time()

    # Try calling predict
    res = client.predict(
        handle_file(image_path),
        handle_file(image_path),
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
    print(f"-> Job Returned! Elapsed for generation: {t3 - t2:.2f}s")
    print(f"-> Result: {res}")

except Exception as e:
    print(f"-> ERROR during Wan 2.2 Lightning test: {type(e).__name__}: {e}")
    t3 = time.time()
    print(f"-> Total time until error: {t3 - t1:.2f}s")

print(f"\nTotal elapsed: {time.time() - start_time:.2f}s")
