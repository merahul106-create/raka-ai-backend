import requests
import json

def joy_caption(file_path):
    space_url = "https://fancyfeast-joy-caption-beta-one.hf.space"
    with open(file_path, "rb") as f:
        up_res = requests.post(f"{space_url}/gradio_api/upload", files={"files": f})
    up_path = up_res.json()[0]

    data = {
        "data": [
            {"path": up_path, "meta": {"_type": "gradio.FileData"}},
            "Write a long detailed description for this image.",
            0.6,
            0.9,
            512,
            True
        ]
    }

    res = requests.post(f"{space_url}/gradio_api/call/chat_joycaption", json=data)
    print("res", res.text)
    event_id = res.json()["event_id"]

    res2 = requests.get(f"{space_url}/gradio_api/call/chat_joycaption/{event_id}")
    print("res2 text length", len(res2.text))
    for line in res2.iter_lines(decode_unicode=True):
        if line.startswith("data:"):
            print("RAW LINE", line)
            try:
                d = json.loads(line[5:])
                if isinstance(d, list) and len(d) > 0:
                    print('Result:', str(d)[:200])
            except Exception as e:
                print("Ex", e)

joy_caption('E:/AF/app fit/fit/android/app/src/main/res/drawable/raka_logo_official.png')
