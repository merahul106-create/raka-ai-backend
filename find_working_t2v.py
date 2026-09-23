import time
import os
from gradio_client import Client

token = os.getenv("HF_TOKEN", "").strip()
prompt = "A majestic dragon flying over mountains, high quality"

candidates = [
    "Lightricks/ltx-video-distilled",
    "strangerzonehf/Wan-2.1-T2V-1.3B",
    "zai-org/CogVideoX-5B-space",
    "ali-vilab/i2vgen-xl",
    "multimodalart/cosetter",
    "mediasynthesismuseum/modelscope-text-to-video-synthesis"
]

for space in candidates:
    print(f"\n--- Testing Space: {space} ---")
    t0 = time.time()
    try:
        client = Client(space, token=token, httpx_kwargs={"timeout": 60})
        print(f"Connected in {time.time() - t0:.2f}s")
        print("View API endpoints:")
        try:
            apis = client.view_api(return_format="dict")
            for ep_name in apis.get("named_endpoints", {}):
                print(f"  Endpoint: {ep_name}")
        except Exception as ve:
            print(f"  view_api err: {ve}")
    except Exception as e:
        print(f"Failed to connect ({time.time() - t0:.2f}s): {e}")
