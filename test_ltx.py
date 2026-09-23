import requests
import json
import time

res = requests.post('https://lightricks-ltx-video-distilled.hf.space/gradio_api/call/text_to_video', json={'data': ['A majestic dragon flying over a medieval castle', 'worst quality, inconsistent motion', None, None, 512, 704, 'text-to-video', 2, 9, 42, True, 1.0, True]})
event_id = res.json()["event_id"]

res2 = requests.get(f'https://lightricks-ltx-video-distilled.hf.space/gradio_api/call/text_to_video/{event_id}', stream=True)
for line in res2.iter_lines(decode_unicode=True):
    print("LINE:", line)
