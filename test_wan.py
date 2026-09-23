import requests
import json
import time

res = requests.post('https://wan-video-wan2-1-t2v-14b.hf.space/gradio_api/call/t2v_generation_async', json={'data': ['A majestic dragon', '1280*720', True, -1]})
print(res.json())
event_id = res.json()["event_id"]

res2 = requests.get(f'https://wan-video-wan2-1-t2v-14b.hf.space/gradio_api/call/t2v_generation_async/{event_id}', stream=True)
for line in res2.iter_lines(decode_unicode=True):
    print("LINE:", line)
