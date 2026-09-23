import time
import os
import requests
import json
from gradio_client import Client

print("=== STARTING TEXT-TO-VIDEO PROFILING TEST ===")
start_time = time.time()

prompt = "A cinematic view of a sunset over the ocean, high quality, 4k"
token = os.getenv("HF_TOKEN", "").strip()
replicate_key = os.getenv("REPLICATE_API_KEY") or "r8_8UHA1bKEwSXNOudUYXcep18keIY87Yd2gRW4F"

t2v_candidates = [
    {"url": "Lightricks/ltx-video-distilled", "type": "ltx"},
    {"url": "Wan-AI/Wan2.1", "type": "wan21"},
]

for cand in t2v_candidates:
    url = cand["url"]
    ctype = cand["type"]
    print(f"\n--- Testing Candidate: {url} ({ctype}) ---")
    t1 = time.time()
    try:
        client = Client(url, token=token, httpx_kwargs={"timeout": 120})
        print(f"Connected to {url} in {time.time() - t1:.2f}s")

        t2 = time.time()
        if ctype == "ltx":
            res = client.predict(
                prompt,
                "worst quality, blurry, low resolution, jittery, distorted",
                None,
                None,
                480,
                704,
                "text-to-video",
                2.0,
                9,
                42,
                True,
                3.0,
                False,
                api_name="/text_to_video"
            )
            print(f"Generation finished in {time.time() - t2:.2f}s | Result: {res}")
            break
        elif ctype == "wan21":
            res = client.predict(
                prompt,
                "1280*720",
                False,
                -1,
                api_name="/t2v_generation_async"
            )
            print(f"Submitted async in {time.time() - t2:.2f}s | Result: {res}")
            # Poll async status
            task_id = res[0] if isinstance(res, (list, tuple)) else res
            print(f"Task ID: {task_id}")
            t_poll_start = time.time()
            for i in range(60):
                time.sleep(5)
                status_res = client.predict(task_id, "t2v", False, api_name="/status_refresh")
                print(f"Poll {i+1} ({time.time() - t_poll_start:.1f}s): {status_res}")
                if status_res and (status_res[0] if isinstance(status_res, (list, tuple)) else status_res):
                    print(f"SUCCESS in {time.time() - t2:.2f}s!")
                    break
            break
    except Exception as e:
        print(f"Failed on {url}: {type(e).__name__}: {e}")

print("\n--- Testing Replicate Fallback for T2V ---")
t_rep_start = time.time()
try:
    headers = {
        "Authorization": f"Bearer {replicate_key}",
        "Content-Type": "application/json",
        "User-Agent": "RakaAI/1.0"
    }
    payload = {"input": {"prompt": prompt}}
    res = requests.post(
        "https://api.replicate.com/v1/models/minimax/video-01/predictions",
        headers=headers,
        json=payload,
        timeout=30
    )
    print(f"Replicate init status: {res.status_code} in {time.time() - t_rep_start:.2f}s")
    if res.status_code == 201:
        prediction = res.json()
        p_id = prediction.get("id")
        p_url = prediction.get("urls", {}).get("get")
        print(f"Job created: {p_id}")
        for i in range(20):
            time.sleep(5)
            poll_res = requests.get(p_url, headers=headers, timeout=15)
            if poll_res.status_code == 200:
                p = poll_res.json()
                st = p.get("status")
                print(f"Poll {i+1} ({time.time() - t_rep_start:.1f}s): status={st}")
                if st == "succeeded":
                    print(f"Replicate SUCCESS in {time.time() - t_rep_start:.2f}s | Output: {p.get('output')}")
                    break
                elif st in ["failed", "canceled"]:
                    print(f"Replicate FAILED: {p.get('error')}")
                    break
except Exception as e:
    print(f"Replicate Error: {e}")

print(f"\nTotal elapsed: {time.time() - start_time:.2f}s")
