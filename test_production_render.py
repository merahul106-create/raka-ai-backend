import requests
import json
import time
import os
import cv2
import numpy as np

PROD_URL = "https://raka-ai-backend.onrender.com"

print("==================================================")
print(" TESTING LIVE DEPLOYED PRODUCTION BACKEND ON RENDER")
print(f" URL: {PROD_URL}")
print("==================================================")

# 1. Health check
print("\n--- 1. Testing GET /api/health ---")
res_health = requests.get(f"{PROD_URL}/api/health", timeout=15)
print(f"Status: {res_health.status_code} | Body: {res_health.json()}")

# 2. Root check
print("\n--- 2. Testing GET /api/ ---")
res_root = requests.get(f"{PROD_URL}/api/", timeout=15)
print(f"Status: {res_root.status_code} | Body: {res_root.json()}")

# 3. Prompt to Image
print("\n--- 3. Testing POST /api/prompt-to-image ---")
res_p2i = requests.post(
    f"{PROD_URL}/api/prompt-to-image",
    data={"prompt": "A beautiful sunset over snow-covered mountains", "style": "Realistic"},
    timeout=40
)
print(f"Status: {res_p2i.status_code} | Body: {res_p2i.json()}")
p2i_url = res_p2i.json().get("url")
if p2i_url:
    img_check = requests.get(f"{PROD_URL}{p2i_url}")
    print(f" Generated Image Accessibility: HTTP {img_check.status_code} | Size: {len(img_check.content)} bytes")

# 4. Create sample test image and video
sample_img_path = "prod_sample.jpg"
cv2.imwrite(sample_img_path, np.zeros((300, 300, 3), np.uint8))

sample_vid_path = "prod_sample.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(sample_vid_path, fourcc, 10.0, (300, 300))
for _ in range(20): out.write(np.zeros((300, 300, 3), np.uint8))
out.release()

# 5. Image -> Prompt
print("\n--- 4. Testing POST /api/image-to-prompt ---")
with open(sample_img_path, "rb") as f:
    res_i2p = requests.post(f"{PROD_URL}/api/image-to-prompt", files={"image": ("sample.jpg", f, "image/jpeg")}, timeout=30)
print(f"Status: {res_i2p.status_code} | Body: {res_i2p.json()}")

# 6. Video -> Prompt
print("\n--- 5. Testing POST /api/video-to-prompt ---")
with open(sample_vid_path, "rb") as f:
    res_v2p = requests.post(f"{PROD_URL}/api/video-to-prompt", files={"video": ("sample.mp4", f, "video/mp4")}, timeout=30)
print(f"Status: {res_v2p.status_code} | Body: {res_v2p.json()}")

# 7. Text to Video
print("\n--- 6. Testing POST /api/text-to-video ---")
res_t2v = requests.post(
    f"{PROD_URL}/api/text-to-video",
    data={"text": "A dragon flying over mountains", "style": "Cinematic"},
    timeout=30
)
print(f"Status: {res_t2v.status_code} | Body: {res_t2v.json()}")
t2v_job = res_t2v.json().get("job_id") if res_t2v.status_code == 200 else None

# 8. Image to Video
print("\n--- 7. Testing POST /api/image-to-video ---")
with open(sample_img_path, "rb") as f:
    res_i2v = requests.post(
        f"{PROD_URL}/api/image-to-video",
        files={"image": ("sample.jpg", f, "image/jpeg")},
        data={"prompt": "Cinematic motion", "style": "Realistic"},
        timeout=30
    )
print(f"Status: {res_i2v.status_code} | Body: {res_i2v.json()}")
i2v_job = res_i2v.json().get("job_id") if res_i2v.status_code == 200 else None

# 9. Poll Jobs
print("\n--- 8. Polling Jobs on Deployed Production ---")
for name, jid in [("TextToVideo", t2v_job), ("ImageToVideo", i2v_job)]:
    if jid:
        print(f"\nPolling {name} Job: {jid}")
        for poll_i in range(12):
            time.sleep(5)
            res_job = requests.get(f"{PROD_URL}/api/video-job/{jid}", timeout=15)
            print(f" Poll {poll_i+1} ({res_job.status_code}) | Stage: {res_job.json().get('stage')} | Status: {res_job.json().get('status')}")
            if res_job.status_code == 200:
                st = res_job.json().get("status")
                if st in ["completed", "failed"]:
                    print(f" Final Job Response: {json.dumps(res_job.json(), indent=2)}")
                    break

for f in [sample_img_path, sample_vid_path]:
    if os.path.exists(f): os.remove(f)

print("\n==================================================")
print(" LIVE PRODUCTION TEST FINISHED!")
print("==================================================")
