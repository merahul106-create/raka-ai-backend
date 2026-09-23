import requests
import json
import time

fal_key = "c4d27282-74e9-40ca-b19d-4588a427af11:9d633e9d2dcf4fccc1170f5334f8058c"
headers = {
    "Authorization": f"Key {fal_key}",
    "Content-Type": "application/json"
}

print("Testing Fal.ai LTX Video model...")
t0 = time.time()
try:
    res = requests.post(
        "https://fal.run/fal-ai/ltx-video",
        headers=headers,
        json={"prompt": "A majestic dragon flying over mountains during sunset"},
        timeout=60
    )
    print(f"Status: {res.status_code} in {time.time() - t0:.2f}s")
    print(res.text[:500])
except Exception as e:
    print(f"Error: {e}")
