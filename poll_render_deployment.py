import requests
import time

url = "https://raka-ai-backend.onrender.com/api/health"
print("Polling Render deployment for version 2.2.0-production-hardened-fallback...")

for i in range(30):
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            ver = data.get("version")
            print(f"Poll {i+1} (HTTP 200) -> Version: {ver}")
            if ver == "2.2.0-production-hardened-fallback":
                print("\nRENDER DEPLOYMENT SUCCESSFUL! New version is LIVE!")
                print(f"Health Response: {data}")
                break
    except Exception as e:
        print(f"Poll {i+1} error: {e}")
    time.sleep(10)
