import requests

url = "https://huggingface.co/api/spaces?search=text-to-video&limit=20&sort=likes"
res = requests.get(url)
if res.status_code == 200:
    spaces = res.json()
    print(f"Found {len(spaces)} spaces:")
    for s in spaces:
        print(f" - {s['id']} (Likes: {s.get('likes', 0)}, Running: {s.get('runtime', {}).get('stage')})")
