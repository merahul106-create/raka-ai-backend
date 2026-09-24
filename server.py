from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Request, Body
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import uuid
import json
import time
import threading
import logging
import shutil
import requests
import base64
import cv2
from typing import Optional, Tuple
from huggingface_hub import InferenceClient
from gradio_client import Client, handle_file

# --- VERSIONING ---
VERSION = "2.2.0-production-hardened-fallback"

# --- PRODUCTION LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - RakaProduction - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RakaProduction")

app = FastAPI(title="Raka AI Production API", docs_url="/api/docs", openapi_url="/api/openapi.json")

# Enable CORS for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
JOBS_DB_PATH = os.path.join(OUTPUT_DIR, "jobs_db.json")

# Static files for serving generated media
app.mount("/api/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# --- ENVIRONMENT CONFIG ---
# Environment variables ONLY - NO hardcoded secrets
REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()

# Provider Candidate Configuration
T2V_CANDIDATES = [
    {"url": "Lightricks/ltx-video-distilled", "type": "ltx"},
    {"url": "Wan-AI/Wan2.1", "type": "wan21"},
]

I2V_CANDIDATES = [
    {
        "url": "https://saravutw-wan2-2-i2v-lightning-4-8step-custom.hf.space",
        "type": "wan22_lightning",
    },
    {"url": "Lightricks/ltx-video-distilled", "type": "ltx"},
    {"url": "Wan-AI/Wan2.1", "type": "wan21"},
]

IMAGE_CANDIDATES = [
    {"url": "black-forest-labs/FLUX.1-schnell", "type": "flux_schnell"},
    {"url": "black-forest-labs/FLUX.1-dev", "type": "flux_dev"},
    {"url": "stabilityai/stable-diffusion-3.5-large", "type": "sd35"},
    {"url": "hysts/SDXL", "type": "sdxl"},
]

# Persistent Job Storage with disk fallback
jobs = {}

def load_jobs_from_disk():
    global jobs
    if os.path.exists(JOBS_DB_PATH):
        try:
            with open(JOBS_DB_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
                if isinstance(saved, dict):
                    jobs.update(saved)
                    logger.info(f"[JOB_DB] Loaded {len(saved)} jobs from disk")
        except Exception as e:
            logger.error(f"[JOB_DB] Failed to load jobs from disk: {e}")

def save_jobs_to_disk():
    try:
        with open(JOBS_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(jobs, f, indent=2)
    except Exception as e:
        logger.error(f"[JOB_DB] Failed to save jobs to disk: {e}")

def update_job(job_id: str, **kwargs):
    if job_id not in jobs:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "stage": "Queued",
            "provider": None,
            "progress": 0,
            "result_url": None,
            "video_url": None,
            "url": None,
            "error": None,
            "error_type": None,
            "details": None,
            "attempts": [],
            "created_at": time.time(),
            "updated_at": time.time()
        }
    jobs[job_id].update(kwargs)
    jobs[job_id]["updated_at"] = time.time()
    save_jobs_to_disk()

# Initial load from disk
load_jobs_from_disk()

# --- ERROR CLASSIFIER ---
def classify_error(err_str: str) -> Tuple[str, str]:
    s = str(err_str).lower()
    if "401" in s or "unauthenticated" in s or "unauthorized" in s or "invalid token" in s:
        return "AUTH_ERROR", "Provider authentication failed. Please check backend API key configuration."
    if "403" in s or "forbidden" in s or "quota" in s or "zerogpu" in s or "402" in s or "429" in s or "limit" in s:
        return "QUOTA_ERROR", "AI service limit or ZeroGPU quota reached on provider."
    if "not found" in s or "404" in s or "repository not found" in s:
        return "MODEL_NOT_FOUND", "The requested AI model or space was not found."
    if "timeout" in s or "timed out" in s or "time out" in s:
        return "TIMEOUT", "The generation timed out while waiting for AI provider."
    if "connection" in s or "connect" in s or "network" in s:
        return "NETWORK_ERROR", "Network connection to AI provider failed."
    if "invalid" in s or "corrupt" in s or "empty" in s or "missing" in s:
        return "INVALID_INPUT", "Invalid or missing input file or parameter."
    return "PROVIDER_ERROR", f"AI Provider Notice: {str(err_str)[:150]}"

# --- SAFE GRADIO CLIENT CREATOR ---
def create_gradio_client(url: str, timeout: int = 120) -> Client:
    """
    Safely creates a Gradio Client. Omits token parameter when HF_TOKEN is empty to avoid
    generating 'Illegal header value b'Bearer '' errors.
    """
    if HF_TOKEN and HF_TOKEN.strip():
        return Client(url, token=HF_TOKEN.strip(), httpx_kwargs={"timeout": timeout})
    return Client(url, httpx_kwargs={"timeout": timeout})

# --- SAFE AUTH HEADER HELPER ---
def get_auth_headers(token: str, prefix: str = "Bearer") -> dict:
    if not token or not token.strip():
        return {}
    return {"Authorization": f"{prefix} {token.strip()}"}

# --- HELPERS ---
def verify_file(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0

def safe_save(src_path: str, job_id: str, ext: str) -> Optional[str]:
    dst_path = os.path.join(OUTPUT_DIR, f"{job_id}{ext}")
    try:
        shutil.copy(src_path, dst_path)
        if verify_file(dst_path):
            return f"/api/outputs/{job_id}{ext}"
    except Exception as e:
        logger.error(f"Storage error: {e}")
    return None

def download_file(url: str, job_id: str, ext: str = ".mp4") -> Optional[str]:
    path = os.path.join(OUTPUT_DIR, f"{job_id}{ext}")
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(path, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        if verify_file(path):
            return f"/api/outputs/{job_id}{ext}"
    except Exception as e:
        logger.error(f"Download failed: {e}")
    return None

# --- GEMINI AI SERVICES ---
def analyze_with_gemini(prompt_text: str, image_bytes: Optional[bytes] = None, mime_type: str = "image/jpeg") -> Optional[str]:
    if not GEMINI_API_KEY:
        logger.info("[GEMINI] GEMINI_API_KEY not configured in environment")
        return None

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={GEMINI_API_KEY}"

    parts = [{"text": prompt_text}]
    if image_bytes:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        parts.append({
            "inline_data": {
                "mime_type": mime_type,
                "data": b64
            }
        })

    payload = {"contents": [{"parts": parts}]}

    try:
        res = requests.post(url, json=payload, timeout=20)
        if res.status_code == 200:
            data = res.json()
            candidates = data.get("candidates", [])
            if candidates:
                text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                if text:
                    logger.info("[GEMINI] Gemini Vision analysis succeeded")
                    return text.strip()
        else:
            logger.warning(f"[GEMINI] Gemini API HTTP {res.status_code}: {res.text[:200]}")
    except Exception as e:
        logger.error(f"[GEMINI] Gemini API Exception: {e}")

    return None

# --- GRADIO VISION FALLBACK ---
def analyze_image_gradio(image_path: str) -> Optional[dict]:
    candidates = [
        {"url": "tonyassi/blip-image-captioning-large", "fn": "/predict", "type": "blip"},
        {"url": "fancyfeast/joy-caption-pre-alpha", "fn": "/stream_chat", "type": "joy"}
    ]

    last_error = ""
    for cand in candidates:
        try:
            logger.info(f"[VISION_GRADIO] Attempting space: {cand['url']}")
            client = create_gradio_client(cand["url"], timeout=10)
            if cand["type"] == "joy":
                result = client.predict(handle_file(image_path), api_name=cand["fn"])
            else:
                result = client.predict(handle_file(image_path), 20, 80, api_name=cand["fn"])

            if result:
                final_prompt = str(result)
                if "⏱" in final_prompt:
                    final_prompt = final_prompt.split("⏱")[0].strip()
                if cand["type"] == "blip":
                    final_prompt = f"A professional detailed photograph of {final_prompt.strip()}, cinematic lighting, highly detailed, 8k masterpiece"
                return {"prompt": final_prompt, "provider": cand["url"]}
        except Exception as e:
            last_error = str(e)
            logger.warning(f"[VISION_GRADIO] Space {cand['url']} failed: {last_error[:150]}")
            continue

    return None

# --- REPLICATE SERVICES ---
def run_replicate_t2v(prompt: str) -> dict:
    if not REPLICATE_API_KEY:
        logger.warning("[T2V] [Replicate] REPLICATE_API_KEY missing in environment")
        return {"error": "Authentication Required", "details": "REPLICATE_API_KEY missing"}

    logger.info(f"[T2V] [Replicate] Starting minimax/video-01 | Prompt: {prompt[:50]}...")
    headers = {
        "Authorization": f"Bearer {REPLICATE_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "RakaAI/1.0"
    }

    payload = {"input": {"prompt": prompt}}

    try:
        res = requests.post(
            "https://api.replicate.com/v1/models/minimax/video-01/predictions",
            headers=headers,
            json=payload,
            timeout=30
        )

        logger.info(f"[T2V] [Replicate] Initial Status: {res.status_code}")

        if res.status_code in (401, 403):
            logger.error(f"[T2V] [Replicate] Auth Error {res.status_code}: {res.text[:200]}")
            return {"error": "Authentication Failed", "details": f"Replicate Auth Error HTTP {res.status_code}"}

        if res.status_code != 201:
            error_data = res.text
            logger.error(f"[T2V] [Replicate] API Error: {res.status_code} - {error_data[:200]}")
            return {"error": f"HTTP {res.status_code}", "details": error_data[:200]}

        prediction = res.json()
        p_id = prediction.get("id")
        p_url = prediction.get("urls", {}).get("get")

        for i in range(36): # 3 minutes max
            time.sleep(5)
            poll_res = requests.get(p_url, headers=headers, timeout=15)

            if poll_res.status_code in (401, 403):
                return {"error": "Authentication Failed", "details": "Replicate Auth Error on poll"}

            if poll_res.status_code != 200:
                continue

            p = poll_res.json()
            status = p.get("status")

            if status == "succeeded":
                output = p.get("output")
                logger.info(f"[T2V] [Replicate] SUCCESS: {output}")
                return {"url": output}

            if status == "failed":
                err = p.get("error", "Unknown error")
                logger.error(f"[T2V] [Replicate] FAILED: {err}")
                return {"error": "Generation Failed", "details": str(err)}

            if status == "canceled":
                return {"error": "Canceled", "details": "Job was canceled by provider"}

        return {"error": "Timeout", "details": "Replicate generation timed out after 3 minutes"}

    except Exception as e:
        logger.error(f"[T2V] [Replicate] Exception: {str(e)}")
        return {"error": "Exception", "details": str(e)}

def run_replicate_i2v(prompt: str, image_path: str) -> dict:
    if not REPLICATE_API_KEY:
        logger.warning("[I2V] [Replicate] REPLICATE_API_KEY missing in environment")
        return {"error": "Authentication Required", "details": "REPLICATE_API_KEY missing"}

    logger.info(f"[I2V] [Replicate] Starting minimax/video-01 | Prompt: {prompt[:50]}...")
    headers = {
        "Authorization": f"Bearer {REPLICATE_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "RakaAI/1.0"
    }

    try:
        with open(image_path, "rb") as f:
            img_data = f.read()

        ext = os.path.splitext(image_path)[1].lower()
        mime_type = "image/png" if ext == ".png" else "image/jpeg"

        b64_encoded = base64.b64encode(img_data).decode("utf-8")
        data_uri = f"data:{mime_type};base64,{b64_encoded}"

        payload = {"input": {
            "prompt": prompt,
            "first_frame_image": data_uri
        }}

        res = requests.post(
            "https://api.replicate.com/v1/models/minimax/video-01/predictions",
            headers=headers,
            json=payload,
            timeout=30
        )

        if res.status_code in (401, 403):
            logger.error(f"[I2V] [Replicate] Auth Error {res.status_code}: {res.text[:200]}")
            return {"error": "Authentication Failed", "details": f"Replicate Auth Error HTTP {res.status_code}"}

        if res.status_code != 201:
            error_data = res.text
            return {"error": f"HTTP {res.status_code}", "details": error_data[:200]}

        prediction = res.json()
        p_id = prediction.get("id")
        p_url = prediction.get("urls", {}).get("get")

        for i in range(36): # 3 minutes max
            time.sleep(5)
            poll_res = requests.get(p_url, headers=headers, timeout=15)

            if poll_res.status_code in (401, 403):
                return {"error": "Authentication Failed", "details": "Replicate Auth Error on poll"}

            if poll_res.status_code != 200:
                continue

            p = poll_res.json()
            status = p.get("status")

            if status == "succeeded":
                output = p.get("output")
                logger.info(f"[I2V] [Replicate] SUCCESS: {output}")
                return {"url": output}

            if status == "failed":
                err = p.get("error", "Unknown error")
                return {"error": "Generation Failed", "details": str(err)}

            if status == "canceled":
                return {"error": "Canceled", "details": "Job was canceled by provider"}

        return {"error": "Timeout", "details": "Replicate generation timed out after 3 minutes"}

    except Exception as e:
        logger.error(f"[I2V] [Replicate] Exception: {str(e)}")
        return {"error": "Exception", "details": str(e)}

# --- GRADIO HF HELPERS ---
def normalize_provider_result(res, job_id: str, provider_name: str) -> Optional[str]:
    try:
        def search_result(obj):
            if isinstance(obj, str):
                if obj.startswith("http://") or obj.startswith("https://"):
                    return obj
                if os.path.exists(obj) and os.path.isfile(obj):
                    return obj
                if obj.endswith(".mp4") or obj.endswith(".webm"):
                    return obj
            elif hasattr(obj, "__fspath__"):
                return os.fspath(obj)
            elif isinstance(obj, dict):
                for key in ["video", "path", "url", "file", "name", "data", "output", "value"]:
                    if key in obj and obj[key]:
                        sub = search_result(obj[key])
                        if sub: return sub
                for v in obj.values():
                    sub = search_result(v)
                    if sub: return sub
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    sub = search_result(item)
                    if sub: return sub
            return None

        extracted = search_result(res)
        if extracted:
            logger.info(f"EXTRACT_SUCCESS | Job: {job_id} | Provider: {provider_name} | Path: {extracted}")
            return extracted
        else:
            logger.warning(f"EXTRACT_FAILED | Job: {job_id} | Provider: {provider_name}")
            return None
    except Exception as e:
        logger.error(f"EXTRACT_EXCEPTION | Job: {job_id} | Provider: {provider_name} | Error: {e}")
        return None

def run_t2v_gradio(prompt: str, candidate: dict, job_id: str) -> dict:
    provider_name = f"HF-{candidate['type']}"
    logger.info(f"[T2V] [{provider_name}] Connecting to {candidate['url']}...")

    try:
        client = create_gradio_client(candidate["url"], timeout=120)

        if candidate["type"] == "ltx":
            logger.info(f"[T2V] [{provider_name}] Calling /text_to_video...")
            res = client.predict(
                prompt,
                "worst quality, blurry, low resolution, jittery, distorted",
                None,
                None,
                480,
                704,
                "text-to-video",
                2.0,
                9,
                42,
                True,
                3.0,
                False,
                api_name="/text_to_video"
            )
            logger.info(f"[T2V] [{provider_name}] Response received")
            return res

        if candidate["type"] == "wan21":
            if not DASHSCOPE_API_KEY:
                return {"error": "Authentication Required", "details": "Wan2.1 requires DASHSCOPE_API_KEY"}
            logger.info(f"[T2V] [{provider_name}] Calling /t2v_generation_async...")
            res = client.predict(
                prompt,
                "1280*720",
                False,
                -1,
                api_name="/t2v_generation_async"
            )

            task_id = res[0] if isinstance(res, (list, tuple)) and len(res) > 0 else (res if isinstance(res, str) else None)
            if not task_id:
                return {"error": "Provider failed to return task_id", "details": str(res)[:200]}

            for i in range(24): # 2 minutes max
                time.sleep(5)
                try:
                    status_res = client.predict(task_id, "t2v", False, api_name="/status_refresh")
                except:
                    continue
                video_obj = status_res[0] if isinstance(status_res, (list, tuple)) else status_res
                if video_obj:
                    extracted = normalize_provider_result(video_obj, job_id, provider_name)
                    if extracted:
                        return video_obj
            return {"error": "Timeout", "details": "Wan2.1 polling timed out"}

    except Exception as e:
        error_msg = str(e)
        if "ZeroGPU" in error_msg or "quota" in error_msg.lower():
            logger.warning(f"[T2V] [{provider_name}] ZeroGPU Quota Limit Reached")
            return {"error": "ZeroGPU Quota Limit Reached", "details": error_msg[:200]}
        if "show_error=True" in error_msg:
            error_msg = f"Provider Internal Error. Details: {error_msg.split('show_error=True')[0]}"
        logger.warning(f"[T2V] [{provider_name}] FAILED: {error_msg[:200]}")
        return {"error": "Provider Exception", "details": error_msg[:200]}

    return {"error": "Unsupported Type", "details": candidate["type"]}

def run_i2v_gradio(candidate: dict, prompt: str, image_path: str, job_id: str) -> dict:
    provider_name = f"HF-{candidate['type']}"
    logger.info(f"[I2V] [{provider_name}] Connecting to {candidate['url']}...")

    try:
        client = create_gradio_client(candidate["url"], timeout=120)

        if candidate["type"] == "wan22_lightning":
            logger.info(f"[I2V] [{provider_name}] Calling /generate_video...")
            res = client.predict(
                handle_file(image_path),
                handle_file(image_path),
                prompt,
                4,
                "blurry, low quality, chaotic, deformed, watermark, bad anatomy, shaky camera view point",
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
            logger.info(f"[I2V] [{provider_name}] Response received")
            return res

        if candidate["type"] == "ltx":
            logger.info(f"[I2V] [{provider_name}] Calling /image_to_video...")
            res = client.predict(
                prompt,
                "worst quality, blurry, low resolution, jittery, distorted",
                handle_file(image_path),
                None,
                512,
                704,
                "image-to-video",
                2.0,
                9,
                42,
                True,
                3.0,
                False,
                api_name="/image_to_video"
            )
            logger.info(f"[I2V] [{provider_name}] Response received")
            return res

        if candidate["type"] == "wan21":
            if not DASHSCOPE_API_KEY:
                return {"error": "Authentication Required", "details": "Wan2.1 requires DASHSCOPE_API_KEY"}
            logger.info(f"[I2V] [{provider_name}] Calling /i2v_generation_async...")
            res = client.predict(
                prompt,
                handle_file(image_path),
                False,
                -1,
                api_name="/i2v_generation_async"
            )
            task_id = res[0] if isinstance(res, (list, tuple)) and len(res) > 0 else (res if isinstance(res, str) else None)
            if not task_id:
                return {"error": "Provider failed to return task_id", "details": str(res)[:200]}

            for i in range(24): # 2 minutes max
                time.sleep(5)
                try:
                    status_res = client.predict(task_id, "i2v", False, api_name="/status_refresh_1")
                except:
                    continue
                video_obj = status_res[0] if isinstance(status_res, (list, tuple)) else status_res
                if video_obj:
                    extracted = normalize_provider_result(video_obj, job_id, provider_name)
                    if extracted:
                        return video_obj
            return {"error": "Timeout", "details": "Wan2.1 polling timed out"}

    except Exception as e:
        error_msg = str(e)
        if "ZeroGPU" in error_msg or "quota" in error_msg.lower():
            logger.warning(f"[I2V] [{provider_name}] ZeroGPU Quota Limit Reached")
            return {"error": "ZeroGPU Quota Limit Reached", "details": error_msg[:200]}
        if "show_error=True" in error_msg:
            error_msg = f"Provider Internal Error. Details: {error_msg.split('show_error=True')[0]}"
        logger.warning(f"[I2V] [{provider_name}] FAILED: {error_msg[:200]}")
        return {"error": "Provider Exception", "details": error_msg[:200]}

    return {"error": "Unsupported Type", "details": candidate["type"]}

# --- WORKER SYNC EXECUTORS ---
def process_t2v_production_sync(job_id: str, prompt: str):
    logger.info(f"T2V_WORKER_STARTED | Job: {job_id}")
    update_job(job_id, status="processing", stage="Generating Video")

    # 1. Try Hugging Face spaces
    for cand in T2V_CANDIDATES:
        provider_id = f"HF/{cand['type']}"
        logger.info(f"T2V_ATTEMPT | Job: {job_id} | Provider: {provider_id}")
        update_job(job_id, provider=provider_id, stage=f"Trying {provider_id}")

        result = run_t2v_gradio(prompt, cand, job_id)

        if isinstance(result, dict) and "error" in result:
            jobs[job_id]["attempts"].append({
                "provider": provider_id,
                "error": result.get("error"),
                "details": result.get("details", "")[:250]
            })
            save_jobs_to_disk()
            continue

        normalized_path = normalize_provider_result(result, job_id, provider_id)

        if normalized_path:
            if normalized_path.startswith("http://") or normalized_path.startswith("https://"):
                local_url = download_file(normalized_path, job_id)
            elif verify_file(normalized_path):
                local_url = safe_save(normalized_path, job_id, ".mp4")
            else:
                local_url = None

            if local_url:
                logger.info(f"T2V_SUCCESS | Job: {job_id} | Provider: {provider_id}")
                update_job(
                    job_id,
                    status="completed",
                    stage="Completed",
                    video_url=local_url,
                    url=local_url,
                    result_url=local_url,
                    provider=provider_id,
                    progress=100
                )
                return

        jobs[job_id]["attempts"].append({
            "provider": provider_id,
            "error": "Extraction/Verification Failed",
            "details": f"Raw result: {str(result)[:200]}"
        })
        save_jobs_to_disk()

    # 2. Try Replicate Fallback
    if REPLICATE_API_KEY:
        provider_id = "Replicate/minimax"
        logger.info(f"T2V_ATTEMPT | Job: {job_id} | Provider: {provider_id} (HF Exhausted)")
        update_job(job_id, provider=provider_id, stage="Trying Replicate/minimax")

        rep_result = run_replicate_t2v(prompt)

        if "url" in rep_result:
            ext_url = rep_result["url"]
            local_url = download_file(ext_url, job_id)
            if local_url:
                logger.info(f"T2V_SUCCESS | Job: {job_id} | Provider: {provider_id}")
                update_job(
                    job_id,
                    status="completed",
                    stage="Completed",
                    video_url=local_url,
                    url=local_url,
                    result_url=local_url,
                    provider=provider_id,
                    progress=100
                )
                return

        jobs[job_id]["attempts"].append({
            "provider": provider_id,
            "error": rep_result.get("error", "Failed"),
            "details": rep_result.get("details", "")[:250]
        })
        save_jobs_to_disk()

    # 3. Final Failure
    last_err = jobs[job_id]["attempts"][-1] if jobs[job_id]["attempts"] else {"provider": "None", "error": "No providers available", "details": "All configured providers were busy or unavailable"}
    err_type, user_msg = classify_error(f"{last_err.get('error')} {last_err.get('details')}")

    logger.error(f"T2V_FAILED | Job: {job_id} | ErrorType: {err_type} | Details: {last_err.get('details')}")
    update_job(
        job_id,
        status="failed",
        stage="Failed",
        error_type=err_type,
        error=user_msg,
        details=last_err.get("details", "All providers were busy or unreachable"),
        provider=last_err.get("provider")
    )

def process_i2v_production_sync(job_id: str, prompt: str, image_path: str):
    logger.info(f"I2V_WORKER_STARTED | Job: {job_id}")
    update_job(job_id, status="processing", stage="Generating Video")

    # 1. Try Hugging Face spaces
    for cand in I2V_CANDIDATES:
        provider_id = f"HF/{cand['type']}"
        logger.info(f"I2V_ATTEMPT | Job: {job_id} | Provider: {provider_id}")
        update_job(job_id, provider=provider_id, stage=f"Trying {provider_id}")

        result = run_i2v_gradio(cand, prompt, image_path, job_id)

        if isinstance(result, dict) and "error" in result:
            jobs[job_id]["attempts"].append({
                "provider": provider_id,
                "error": result.get("error"),
                "details": result.get("details", "")[:250]
            })
            save_jobs_to_disk()
            continue

        normalized_path = normalize_provider_result(result, job_id, provider_id)

        if normalized_path:
            if normalized_path.startswith("http://") or normalized_path.startswith("https://"):
                local_url = download_file(normalized_path, job_id)
            elif verify_file(normalized_path):
                local_url = safe_save(normalized_path, job_id, ".mp4")
            else:
                local_url = None

            if local_url:
                logger.info(f"I2V_SUCCESS | Job: {job_id} | Provider: {provider_id}")
                update_job(
                    job_id,
                    status="completed",
                    stage="Completed",
                    video_url=local_url,
                    url=local_url,
                    result_url=local_url,
                    provider=provider_id,
                    progress=100
                )
                return

        jobs[job_id]["attempts"].append({
            "provider": provider_id,
            "error": "Extraction/Verification Failed",
            "details": f"Raw result: {str(result)[:200]}"
        })
        save_jobs_to_disk()

    # 2. Try Replicate Fallback
    if REPLICATE_API_KEY:
        provider_id = "Replicate/minimax"
        logger.info(f"I2V_ATTEMPT | Job: {job_id} | Provider: {provider_id} (HF Exhausted)")
        update_job(job_id, provider=provider_id, stage="Trying Replicate/minimax")

        rep_result = run_replicate_i2v(prompt, image_path)

        if "url" in rep_result:
            ext_url = rep_result["url"]
            local_url = download_file(ext_url, job_id)
            if local_url:
                logger.info(f"I2V_SUCCESS | Job: {job_id} | Provider: {provider_id}")
                update_job(
                    job_id,
                    status="completed",
                    stage="Completed",
                    video_url=local_url,
                    url=local_url,
                    result_url=local_url,
                    provider=provider_id,
                    progress=100
                )
                return

        jobs[job_id]["attempts"].append({
            "provider": provider_id,
            "error": rep_result.get("error", "Failed"),
            "details": rep_result.get("details", "")[:250]
        })
        save_jobs_to_disk()

    # 3. Final Failure
    last_err = jobs[job_id]["attempts"][-1] if jobs[job_id]["attempts"] else {"provider": "None", "error": "No providers available", "details": "All configured providers were busy or unavailable"}
    err_type, user_msg = classify_error(f"{last_err.get('error')} {last_err.get('details')}")

    logger.error(f"I2V_FAILED | Job: {job_id} | ErrorType: {err_type} | Details: {last_err.get('details')}")
    update_job(
        job_id,
        status="failed",
        stage="Failed",
        error_type=err_type,
        error=user_msg,
        details=last_err.get("details", "All providers were busy or unreachable"),
        provider=last_err.get("provider")
    )

async def process_t2v_production(job_id: str, prompt: str):
    import asyncio
    logger.info(f"T2V_JOB_START | Job: {job_id}")
    try:
        await asyncio.wait_for(
            asyncio.to_thread(process_t2v_production_sync, job_id, prompt),
            timeout=300.0 # 5 minute overall timeout
        )
    except asyncio.TimeoutError:
        logger.error(f"T2V_TIMEOUT | Job: {job_id}")
        update_job(
            job_id,
            status="failed",
            stage="Failed",
            error_type="TIMEOUT",
            error="Generation timed out after 5 minutes",
            details="Worker timeout reached"
        )
    except Exception as e:
        logger.error(f"T2V_EXCEPTION | Job: {job_id} | Error: {e}")
        update_job(
            job_id,
            status="failed",
            stage="Failed",
            error_type="PROVIDER_ERROR",
            error="Internal Worker Error",
            details=str(e)
        )

async def process_i2v_production(job_id: str, prompt: str, image_path: str):
    import asyncio
    logger.info(f"I2V_JOB_START | Job: {job_id}")
    try:
        await asyncio.wait_for(
            asyncio.to_thread(process_i2v_production_sync, job_id, prompt, image_path),
            timeout=300.0 # 5 minute overall timeout
        )
    except asyncio.TimeoutError:
        logger.error(f"I2V_TIMEOUT | Job: {job_id}")
        update_job(
            job_id,
            status="failed",
            stage="Failed",
            error_type="TIMEOUT",
            error="Generation timed out after 5 minutes",
            details="Worker timeout reached"
        )
    except Exception as e:
        logger.error(f"I2V_EXCEPTION | Job: {job_id} | Error: {e}")
        update_job(
            job_id,
            status="failed",
            stage="Failed",
            error_type="PROVIDER_ERROR",
            error="Internal Worker Error",
            details=str(e)
        )

# --- IMAGE GENERATION LOGIC ---
def run_pollinations_fallback(prompt: str, job_id: str) -> Optional[str]:
    logger.info("[IMAGE] Trying Pollinations fallback...")
    encoded = requests.utils.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&nologo=true&model=flux"
    try:
        res = requests.get(url, timeout=30)
        if res.status_code == 200:
            path = os.path.join(OUTPUT_DIR, f"{job_id}.jpg")
            with open(path, "wb") as f:
                f.write(res.content)
            if verify_file(path):
                return f"/api/outputs/{job_id}.jpg"
    except Exception as e:
        logger.warning(f"[IMAGE] Pollinations Exception: {e}")
    return None

def run_image_gen_gradio(cand: dict, prompt: str) -> Optional[str]:
    client = create_gradio_client(cand["url"], timeout=45)

    if cand["type"] in ("flux_schnell", "flux_dev"):
        res = client.predict(prompt, 0, True, 1024, 1024, 4 if cand["type"] == "flux_schnell" else 28, api_name="/infer")
        return res[0].get("path") if (res and isinstance(res, (list, tuple))) else None
    if cand["type"] == "sd35":
        res = client.predict(prompt, "low quality", 0, True, 1024, 1024, 4.5, 28, api_name="/infer")
        return res[0] if (res and isinstance(res, (list, tuple))) else None
    if cand["type"] == "sdxl":
        res = client.predict(prompt, "", "", "", False, False, False, 0, 1024, 1024, 5.0, 5.0, 25, 25, True, api_name="/predict")
        return res if isinstance(res, str) else None
    return None

# --- PUBLIC ENDPOINTS ---

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "version": VERSION,
        "timestamp": time.time(),
        "environment": "production",
        "config": {
            "HF_TOKEN_configured": bool(HF_TOKEN),
            "REPLICATE_API_KEY_configured": bool(REPLICATE_API_KEY),
            "GEMINI_API_KEY_configured": bool(GEMINI_API_KEY),
            "DASHSCOPE_API_KEY_configured": bool(DASHSCOPE_API_KEY)
        }
    }

@app.get("/api/")
def home():
    return {
        "app": "Raka AI",
        "version": VERSION,
        "status": "active"
    }

# --- A. PROMPT -> IMAGE ---
@app.post("/api/prompt-to-image")
async def api_p2i(
    request: Request,
    prompt: Optional[str] = Form(None),
    style: Optional[str] = Form("Realistic"),
    body: Optional[dict] = Body(None)
):
    input_prompt = prompt or (body.get("prompt") if body else None)
    input_style = style or (body.get("style", "Realistic") if body else "Realistic")

    if not input_prompt:
        return JSONResponse(status_code=400, content={"success": False, "error_type": "INVALID_INPUT", "error": "MISSING_PROMPT", "message": "Please enter a prompt"})

    job_id = str(uuid.uuid4())
    final_prompt = f"{input_prompt}, {input_style} style, masterpiece, cinematic"
    logger.info(f"[IMAGE] New Request: {final_prompt}")

    # 1. Try HF candidates
    for cand in IMAGE_CANDIDATES:
        try:
            temp = run_image_gen_gradio(cand, final_prompt)
            if temp and verify_file(temp):
                local_url = safe_save(temp, job_id, ".webp")
                if local_url:
                    logger.info(f"[IMAGE] SUCCESS on {cand['url']}")
                    return {"success": True, "job_id": job_id, "url": local_url, "result_url": local_url}
        except Exception as e:
            logger.warning(f"[IMAGE] {cand['url']} failed: {e}")

    # 2. Try Pollinations (Always reliable)
    local_url = run_pollinations_fallback(final_prompt, job_id)
    if local_url:
        return {"success": True, "job_id": job_id, "url": local_url, "result_url": local_url}

    return JSONResponse(status_code=503, content={
        "success": False,
        "error_type": "QUOTA_ERROR",
        "error": "IMAGE_QUOTA",
        "message": "All image engines are currently busy. Please try again in a moment."
    })

# --- B. TEXT -> VIDEO ---
@app.post("/api/text-to-video")
async def api_t2v(
    background_tasks: BackgroundTasks,
    text: Optional[str] = Form(None),
    style: Optional[str] = Form("Realistic"),
    body: Optional[dict] = Body(None)
):
    input_text = text or (body.get("text") or body.get("prompt") if body else None)
    input_style = style or (body.get("style", "Realistic") if body else "Realistic")

    if not input_text:
        return JSONResponse(status_code=400, content={"success": False, "error_type": "INVALID_INPUT", "error": "MISSING_TEXT", "message": "Please enter text/prompt"})

    jid = str(uuid.uuid4())
    final_prompt = f"{input_text}, {input_style} style"
    update_job(jid, status="queued", stage="Queued", prompt=final_prompt)

    background_tasks.add_task(process_t2v_production, jid, final_prompt)
    return {"success": True, "job_id": jid, "status": "queued", "version": VERSION}

# --- C. PROMPT -> VIDEO ---
@app.post("/api/prompt-to-video")
async def api_p2v(
    background_tasks: BackgroundTasks,
    prompt: Optional[str] = Form(None),
    style: Optional[str] = Form("Realistic"),
    body: Optional[dict] = Body(None)
):
    input_prompt = prompt or (body.get("prompt") or body.get("text") if body else None)
    input_style = style or (body.get("style", "Realistic") if body else "Realistic")

    if not input_prompt:
        return JSONResponse(status_code=400, content={"success": False, "error_type": "INVALID_INPUT", "error": "MISSING_PROMPT", "message": "Please enter a prompt"})

    jid = str(uuid.uuid4())
    final_prompt = f"{input_prompt}, {input_style} style"
    update_job(jid, status="queued", stage="Queued", prompt=final_prompt)

    background_tasks.add_task(process_t2v_production, jid, final_prompt)
    return {"success": True, "job_id": jid, "status": "queued", "version": VERSION}

# --- D. IMAGE -> VIDEO ---
@app.post("/api/image-to-video")
async def api_i2v(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    prompt: Optional[str] = Form("Cinematic motion"),
    style: Optional[str] = Form("Realistic")
):
    if not image or not image.filename:
        return JSONResponse(status_code=400, content={"success": False, "error_type": "INVALID_INPUT", "error": "MISSING_IMAGE", "message": "Image upload is required"})

    jid = str(uuid.uuid4())
    ext = os.path.splitext(image.filename)[1]
    if not ext: ext = ".jpg"
    image_path = os.path.join(OUTPUT_DIR, f"input_{jid}{ext}")

    with open(image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    if not verify_file(image_path):
        return JSONResponse(status_code=400, content={"success": False, "error_type": "INVALID_INPUT", "error": "INVALID_IMAGE", "message": "Uploaded image file is empty or corrupted"})

    final_prompt = f"{prompt}, {style} style" if style else prompt
    update_job(jid, status="queued", stage="Queued", prompt=final_prompt)

    background_tasks.add_task(process_i2v_production, jid, final_prompt, image_path)
    return {"success": True, "job_id": jid, "status": "queued", "version": VERSION}

# --- E. IMAGE -> PROMPT ---
@app.post("/api/image-to-prompt")
async def api_image_to_prompt(image: UploadFile = File(...)):
    if not image or not image.filename:
        return JSONResponse(status_code=400, content={"success": False, "error_type": "INVALID_INPUT", "error": "MISSING_IMAGE", "message": "Image upload is required"})

    jid = str(uuid.uuid4())
    ext = os.path.splitext(image.filename)[1]
    if not ext: ext = ".jpg"
    image_path = os.path.join(OUTPUT_DIR, f"input_i2p_{jid}{ext}")

    with open(image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    if not verify_file(image_path):
        return JSONResponse(status_code=400, content={"success": False, "error_type": "INVALID_INPUT", "error": "INVALID_IMAGE", "message": "Uploaded image is empty or invalid"})

    # 1. Try Gemini Vision API first
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    mime_type = "image/png" if ext.lower() == ".png" else "image/jpeg"
    gemini_prompt = analyze_with_gemini(
        "Describe this image in detail to serve as an AI generation prompt for video or photo creation.",
        img_bytes,
        mime_type
    )

    if gemini_prompt:
        if os.path.exists(image_path): os.remove(image_path)
        return {
            "success": True,
            "job_id": jid,
            "prompt": gemini_prompt,
            "provider": "Gemini-Flash-Vision"
        }

    # 2. Try Gradio Vision Fallback (Joy Caption / BLIP)
    gradio_res = analyze_image_gradio(image_path)

    # Cleanup temp upload file
    if os.path.exists(image_path):
        os.remove(image_path)

    if gradio_res:
        return {
            "success": True,
            "job_id": jid,
            "prompt": gradio_res["prompt"],
            "provider": gradio_res["provider"]
        }

    # 3. Final structured error if all vision models failed
    return JSONResponse(status_code=503, content={
        "success": False,
        "job_id": jid,
        "error_type": "PROVIDER_ERROR",
        "error": "VISION_ANALYSIS_FAILED",
        "message": "Vision analysis unavailable across all configured providers"
    })

# --- F. VIDEO -> PROMPT ---
@app.post("/api/video-to-prompt")
async def api_video_to_prompt(video: UploadFile = File(...)):
    if not video or not video.filename:
        return JSONResponse(status_code=400, content={"success": False, "error_type": "INVALID_INPUT", "error": "MISSING_VIDEO", "message": "Video upload is required"})

    jid = str(uuid.uuid4())
    ext = os.path.splitext(video.filename)[1]
    if not ext: ext = ".mp4"
    video_path = os.path.join(OUTPUT_DIR, f"input_v2p_{jid}{ext}")

    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    if not verify_file(video_path):
        return JSONResponse(status_code=400, content={"success": False, "error_type": "INVALID_INPUT", "error": "INVALID_VIDEO", "message": "Uploaded video is empty or invalid"})

    # Extract middle frame using OpenCV and analyze with Gemini or Gradio fallback
    extracted_frame_path = os.path.join(OUTPUT_DIR, f"frame_{jid}.jpg")
    frame_extracted = False
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            mid_frame = max(0, frame_count // 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(extracted_frame_path, frame)
                frame_extracted = verify_file(extracted_frame_path)
        cap.release()
    except Exception as e:
        logger.error(f"[V2P] Frame extraction error: {e}")

    # Cleanup temp video upload
    if os.path.exists(video_path):
        os.remove(video_path)

    if not frame_extracted:
        return JSONResponse(status_code=400, content={
            "success": False,
            "job_id": jid,
            "error_type": "INVALID_INPUT",
            "error": "FRAME_EXTRACTION_FAILED",
            "message": "Could not extract valid video frame for analysis"
        })

    # 1. Analyze extracted frame with Gemini Vision
    with open(extracted_frame_path, "rb") as f:
        img_bytes = f.read()

    gemini_prompt = analyze_with_gemini(
        "Describe this video frame in detail to serve as a cinematic motion video prompt.",
        img_bytes,
        "image/jpeg"
    )

    if gemini_prompt:
        if os.path.exists(extracted_frame_path): os.remove(extracted_frame_path)
        return {
            "success": True,
            "job_id": jid,
            "prompt": gemini_prompt,
            "provider": "Gemini-Flash-VideoVision"
        }

    # 2. Try Gradio Vision Fallback (Joy Caption / BLIP)
    gradio_res = analyze_image_gradio(extracted_frame_path)
    if os.path.exists(extracted_frame_path): os.remove(extracted_frame_path)

    if gradio_res:
        return {
            "success": True,
            "job_id": jid,
            "prompt": gradio_res["prompt"],
            "provider": gradio_res["provider"]
        }

    # 3. Final structured error
    return JSONResponse(status_code=503, content={
        "success": False,
        "job_id": jid,
        "error_type": "PROVIDER_ERROR",
        "error": "VIDEO_ANALYSIS_FAILED",
        "message": "Video analysis failed across all configured providers"
    })

# --- JOB STATUS ENDPOINT ---
@app.get("/api/video-job/{job_id}")
def get_api_job(job_id: str):
    if job_id not in jobs:
        load_jobs_from_disk()

    if job_id not in jobs:
        return JSONResponse(status_code=404, content={"success": False, "error_type": "MODEL_NOT_FOUND", "error": "NOT_FOUND", "message": "Job ID not found"})

    return jobs.get(job_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
