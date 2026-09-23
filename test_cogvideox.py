import time
import os
from gradio_client import Client

token = os.getenv("HF_TOKEN", "").strip()
space = "zai-org/CogVideoX-5B-space"

print("Connecting to CogVideoX-5B-space...")
t0 = time.time()
client = Client(space, token=token, httpx_kwargs={"timeout": 120})
print(f"Connected in {time.time() - t0:.2f}s")

t1 = time.time()
print("Generating video...")
try:
    res = client.predict(
        "A majestic dragon flying over mountains, sunset, 4k",
        None,
        None,
        0.8,
        -1,
        False,
        False,
        api_name="/generate"
    )
    print(f"SUCCESS in {time.time() - t1:.2f}s!")
    print(f"Result: {res}")
except Exception as e:
    print(f"ERROR: {e}")
