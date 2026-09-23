import requests
import time

request_data = {"data": ["mountain", 42, True, 1024, 1024, 4]}
response = requests.post("https://black-forest-labs-flux-1-schnell.hf.space/gradio_api/call/infer", json=request_data, timeout=30)
response.raise_for_status()
event_id = response.json()["event_id"]

res = requests.get(f"https://black-forest-labs-flux-1-schnell.hf.space/gradio_api/call/infer/{event_id}", stream=True, timeout=120)
for line in res.iter_lines(decode_unicode=True):
    print("LINE:", line)
