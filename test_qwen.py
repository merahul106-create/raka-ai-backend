import requests
import json

def qwen_predict(file_path, prompt):
    space_url = "https://qwen-qwen2-5-vl-32b-instruct.hf.space"
    with open(file_path, "rb") as f:
        up_res = requests.post(f"{space_url}/gradio_api/upload", files={"files": f})
    up_path = up_res.json()[0]

    data = {
        "data": [
            [
                [{"file": {"path": up_path}}, None],
                [prompt, None]
            ]
        ]
    }

    res = requests.post(f"{space_url}/gradio_api/call/predict", json=data)
    print("res", res.json())

qwen_predict('E:/AF/app fit/fit/android/raka_test_video.mp4', 'describe video')
