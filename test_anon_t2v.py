import time
from gradio_client import Client

print("Connecting to LTX Video Distilled anonymously (no token)...")
t0 = time.time()
try:
    client = Client("Lightricks/ltx-video-distilled", token=None, httpx_kwargs={"timeout": 120})
    print(f"Connected in {time.time() - t0:.2f}s")
    t1 = time.time()
    res = client.predict(
        "A majestic dragon flying over mountains during sunset",
        "worst quality, blurry, low resolution",
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
    print(f"SUCCESS in {time.time() - t1:.2f}s | Result: {res}")
except Exception as e:
    print(f"ERROR: {e}")
