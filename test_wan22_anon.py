import time
from gradio_client import Client, handle_file

print("Connecting to Wan 2.2 Lightning anonymously (token=None)...")
t0 = time.time()
client = Client("https://saravutw-wan2-2-i2v-lightning-4-8step-custom.hf.space", token=None, httpx_kwargs={"timeout": 180})
print(f"Connected in {time.time() - t0:.2f}s")

t1 = time.time()
try:
    res = client.predict(
        handle_file("test-prompt-image.jpg"),
        handle_file("test-prompt-image.jpg"),
        "A person walking on a mountain trail",
        4,
        "blurry, low quality",
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
    print(f"SUCCESS in {time.time() - t1:.2f}s!")
    print(res)
except Exception as e:
    print(f"ERROR: {e}")
