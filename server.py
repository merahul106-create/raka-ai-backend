from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Request
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
from huggingface_hub import InferenceClient
from gradio_client import Client, handle_file

# --- VERSIONING ---
VERSION = "2.0.0-production-auth-fixed"

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

# Static files for serving generated media
app.mount("/api/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# --- ENVIRONMENT CONFIG ---
REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY") or "r8_8UHA1bKEwSXNOudUYXcep18keIY87Yd2gRW4F"
# Production hardened token management
p1 = ""
p2 = ""
HF_TOKEN = os.getenv("HF_TOKEN") or (p1 + p2)

# T2V Models (Gradio Spaces)
T2V_CANDIDATES = [
    {"url": "Lightricks/ltx-video-distilled", "type": "ltx"},
    {"url": "Wan-AI/Wan2.1", "type": "wan21"},
]

# I2V CANDIDATES
I2V_CANDIDATES = [
    {
        "url": "https://saravutw-wan2-2-i2v-lightning-4-8step-custom.hf.space",
        "type": "wan22_lightning",
    },
    {"url": "Lightricks/ltx-video-distilled", "type": "ltx"},
    {"url": "Wan-AI/Wan2.1", "type": "wan21"},
]

# Image Gen candidates
IMAGE_CANDIDATES = [
    {"url": "black-forest-labs/FLUX.1-schnell", "type": "flux_schnell"},
    {"url": "black-forest-labs/FLUX.1-dev", "type": "flux_dev"},
    {"url": "stabilityai/stable-diffusion-3.5-large", "type": "sd35"},
    {"url": "hysts/SDXL", "type": "sdxl"},
]

# In-memory job store
jobs = {}

# --- PUBLIC ENDPOINTS ---

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "version": VERSION,
        "timestamp": time.time(),
        "environment": "production"
    }

@app.get("/api/")
def home():
    return {
        "app": "Raka AI",
        "version": VERSION,
        "status": "active"
    }

# --- HELPERS ---

def verify_file(path):
    return os.path.exists(path) and os.path.getsize(path) > 0

def safe_save(src_path, job_id, ext):
    dst_path = os.path.join(OUTPUT_DIR, f"{job_id}{ext}")
    try:
        shutil.copy(src_path, dst_path)
        if verify_file(dst_path):
            return f"/api/outputs/{job_id}{ext}"
    except Exception as e:
        logger.error(f"Storage error: {e}")
    return None

def download_file(url, job_id, ext=".mp4"):
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

# --- AI ENGINES ---

def run_replicate_t2v(prompt):
    logger.info(f"[T2V] [Replicate] Starting minimax/video-01 | Prompt: {prompt[:50]}...")
    headers = {
        "Authorization": f"Bearer {REPLICATE_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "RakaAI/1.0"
    }
    payload = {"input": {"prompt": prompt}}

    try:
        # 1. Create Prediction
        res = requests.post(
            "https://api.replicate.com/v1/models/minimax/video-01/predictions",
            headers=headers,
            json=payload,
            timeout=45
        )

        logger.info(f"[T2V] [Replicate] Initial Response: {res.status_code}")

        if res.status_code != 201:
            error_data = res.text
            logger.error(f"[T2V] [Replicate] API Error: {res.status_code} - {error_data}")
            return {"error": f"HTTP {res.status_code}", "details": error_data}

        prediction = res.json()
        p_id = prediction.get("id")
        p_url = prediction.get("urls", {}).get("get")
        logger.info(f"[T2V] [Replicate] Job Created: {p_id}")

        # 2. Poll for Completion
        max_poll = 180 # 15 minutes max
        for i in range(max_poll):
            time.sleep(5)
            poll_res = requests.get(p_url, headers=headers, timeout=30)

            if poll_res.status_code != 200:
                logger.warning(f"[T2V] [Replicate] Poll Error: {poll_res.status_code}")
                continue

            p = poll_res.json()
            status = p.get("status")

            if i % 4 == 0:
                logger.info(f"[T2V] [Replicate] Status ({p_id}): {status}")

            if status == "succeeded":
                output = p.get("output")
                logger.info(f"[T2V] [Replicate] SUCCESS: {output}")
                return {"url": output}

            if status == "failed":
                err = p.get("error", "Unknown error")
                logger.error(f"[T2V] [Replicate] FAILED ({p_id}): {err}")
                return {"error": "Generation Failed", "details": err}

            if status == "canceled":
                return {"error": "Canceled", "details": "Job was canceled by provider"}

        return {"error": "Timeout", "details": "Generation exceeded 15 minutes"}

    except Exception as e:
        logger.error(f"[T2V] [Replicate] Exception: {str(e)}")
        return {"error": "Exception", "details": str(e)}

def run_replicate_i2v(prompt, image_path):
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
        mime_type = "image/jpeg"
        if ext == ".png": mime_type = "image/png"
        elif ext == ".webp": mime_type = "image/webp"

        b64_encoded = base64.b64encode(img_data).decode("utf-8")
        data_uri = f"data:{mime_type};base64,{b64_encoded}"

        payload = {"input": {
            "prompt": prompt,
            "first_frame_image": data_uri
        }}

        # 1. Create Prediction
        res = requests.post(
            "https://api.replicate.com/v1/models/minimax/video-01/predictions",
            headers=headers,
            json=payload,
            timeout=45
        )

        logger.info(f"[I2V] [Replicate] Initial Response: {res.status_code}")

        if res.status_code != 201:
            error_data = res.text
            logger.error(f"[I2V] [Replicate] API Error: {res.status_code} - {error_data}")
            return {"error": f"HTTP {res.status_code}", "details": error_data}

        prediction = res.json()
        p_id = prediction.get("id")
        p_url = prediction.get("urls", {}).get("get")
        logger.info(f"[I2V] [Replicate] Job Created: {p_id}")

        # 2. Poll for Completion
        max_poll = 180 # 15 minutes max
        for i in range(max_poll):
            time.sleep(5)
            poll_res = requests.get(p_url, headers=headers, timeout=30)

            if poll_res.status_code != 200:
                logger.warning(f"[I2V] [Replicate] Poll Error: {poll_res.status_code}")
                continue

            p = poll_res.json()
            status = p.get("status")

            if i % 4 == 0:
                logger.info(f"[I2V] [Replicate] Status ({p_id}): {status}")

            if status == "succeeded":
                output = p.get("output")
                logger.info(f"[I2V] [Replicate] SUCCESS: {output}")
                return {"url": output}

            if status == "failed":
                err = p.get("error", "Unknown error")
                logger.error(f"[I2V] [Replicate] FAILED ({p_id}): {err}")
                return {"error": "Generation Failed", "details": err}

            if status == "canceled":
                return {"error": "Canceled", "details": "Job was canceled by provider"}

        return {"error": "Timeout", "details": "Generation exceeded 15 minutes"}

    except Exception as e:
        logger.error(f"[I2V] [Replicate] Exception: {str(e)}")
        return {"error": "Exception", "details": str(e)}

def normalize_provider_result(res, job_id, provider_name):
    """
    Safely extracts a local file path or remote URL from a Gradio/HuggingFace result object.
    Logs the structure safely for debugging.
    """
    try:
        # Safe stringification (avoiding large objects if any)
        safe_res_repr = str(res)[:1000]
        logger.info(f"T2V_WAN_RAW_RESULT_TYPE | Job: {job_id} | Provider: {provider_name} | Type: {type(res)}")
        logger.info(f"T2V_WAN_RAW_RESULT_STRUCTURE | Job: {job_id} | Repr: {safe_res_repr}")
        logger.info(f"T2V_WAN_EXTRACT_ATTEMPT | Job: {job_id} | Provider: {provider_name}")

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
                # Check known keys
                for key in ["video", "path", "url", "file", "name", "data", "output", "value"]:
                    if key in obj and obj[key]:
                        sub = search_result(obj[key])
                        if sub: return sub
                # If we couldn't find known keys, iterate all values
                for v in obj.values():
                    sub = search_result(v)
                    if sub: return sub
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    sub = search_result(item)
                    if sub: return sub
            return None

        extracted_path_or_url = search_result(res)

        if extracted_path_or_url:
            logger.info(f"T2V_WAN_EXTRACT_SUCCESS | Job: {job_id} | Provider: {provider_name} | Path: {extracted_path_or_url}")
            return extracted_path_or_url
        else:
            logger.warning(f"T2V_WAN_EXTRACT_FAILED | Job: {job_id} | Provider: {provider_name} | Msg: Provider returned a result, but no usable video file/path/URL could be extracted.")
            return None

    except Exception as e:
        logger.error(f"T2V_WAN_EXTRACT_FAILED | Job: {job_id} | Provider: {provider_name} | Exception during normalization: {str(e)}")
        return None

def run_t2v_gradio(prompt, candidate, job_id):
    provider_name = f"HF-{candidate['type']}"
    logger.info(f"[T2V] [{provider_name}] Connecting to {candidate['url']}...")

    try:
        # Increase timeout for video generation
        client = Client(candidate["url"], token=HF_TOKEN, httpx_kwargs={"timeout": 600})

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
            logger.info(f"[T2V] [{provider_name}] Calling /t2v_generation_async...")
            res = client.predict(
                prompt,
                "1280*720",
                False,
                -1,
                api_name="/t2v_generation_async"
            )

            task_id = None
            if isinstance(res, (list, tuple)) and len(res) > 0:
                task_id = res[0]
            elif isinstance(res, str):
                task_id = res

            if not task_id:
                return {"error": "Provider failed to return task_id", "details": str(res)[:500]}

            for i in range(24): # 10 minutes max
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
        if "show_error=True" in error_msg:
            error_msg = f"Provider Error (Gradio). Details: {error_msg.split('show_error=True')[0]}"
        logger.warning(f"[T2V] [{provider_name}] FAILED: {error_msg}")
        return {"error": "Provider Exception", "details": error_msg}

    return {"error": "Unsupported Type", "details": candidate["type"]}

def process_t2v_production_sync(job_id, prompt):
    logger.info(f"T2V_WORKER_STARTED | Job: {job_id}")

    # Use the same verified candidates as I2V where possible
    for cand in T2V_CANDIDATES:
        provider_id = f"HF/{cand['type']}"
        logger.info(f"T2V_PROVIDER_ATTEMPT | Job: {job_id} | Provider: {provider_id}")

        result = run_t2v_gradio(prompt, cand, job_id)

        if isinstance(result, dict) and "error" in result:
            jobs[job_id]["attempts"].append({
                "provider": provider_id,
                "error": result.get("error"),
                "details": result.get("details", "")[:250]
            })
            continue

        normalized_path = normalize_provider_result(result, job_id, provider_id)

        if normalized_path:
            if normalized_path.startswith("http://") or normalized_path.startswith("https://"):
                local_url = download_file(normalized_path, job_id)
                if local_url:
                    logger.info(f"T2V_PROVIDER_SUCCESS | Job: {job_id} | Provider: {provider_id}")
                    jobs[job_id]["status"] = "completed"
                    jobs[job_id]["video_url"] = local_url
                    jobs[job_id]["provider"] = provider_id
                    jobs[job_id]["end_time"] = time.time()
                    logger.info(f"T2V_JOB_COMPLETED | Job: {job_id} | URL: {local_url}")
                    return
            elif verify_file(normalized_path):
                local_url = safe_save(normalized_path, job_id, ".mp4")
                if local_url:
                    logger.info(f"T2V_PROVIDER_SUCCESS | Job: {job_id} | Provider: {provider_id}")
                    jobs[job_id]["status"] = "completed"
                    jobs[job_id]["video_url"] = local_url
                    jobs[job_id]["provider"] = provider_id
                    jobs[job_id]["end_time"] = time.time()
                    logger.info(f"T2V_JOB_COMPLETED | Job: {job_id} | URL: {local_url}")
                    return

        # Record failure and move to next
        err_msg = "Extraction Failed" if not normalized_path else "File Verification Failed"
        jobs[job_id]["attempts"].append({
            "provider": provider_id,
            "error": err_msg,
            "details": f"Raw result: {str(result)[:250]}"
        })

    # Try Replicate Fallback
    if REPLICATE_API_KEY:
        provider_id = "Replicate/minimax"
        logger.info(f"T2V_PROVIDER_ATTEMPT | Job: {job_id} | Provider: {provider_id} (HF Exhausted)")
        rep_result = run_replicate_t2v(prompt)

        if "url" in rep_result:
            ext_url = rep_result["url"]
            local_url = download_file(ext_url, job_id)
            if local_url:
                logger.info(f"T2V_PROVIDER_SUCCESS | Job: {job_id} | Provider: {provider_id}")
                jobs[job_id]["status"] = "completed"
                jobs[job_id]["video_url"] = local_url
                jobs[job_id]["provider"] = provider_id
                jobs[job_id]["end_time"] = time.time()
                logger.info(f"T2V_JOB_COMPLETED | Job: {job_id} | URL: {local_url}")
                return
            else:
                rep_result["error"] = "Download Failed"
                rep_result["details"] = f"Could not download from {ext_url}"

        logger.warning(f"T2V_PROVIDER_FAILED | Job: {job_id} | Provider: {provider_id} | Error: {rep_result.get('error')}")
        jobs[job_id]["attempts"].append({
            "provider": provider_id,
            "error": rep_result.get("error"),
            "details": rep_result.get("details", "")[:250]
        })

    # Final Failure
    logger.error(f"T2V_JOB_FAILED | Job: {job_id} | Reason: ALL FREE MODELS FAILED")
    jobs[job_id]["status"] = "failed"
    jobs[job_id]["end_time"] = time.time()

    if jobs[job_id]["attempts"]:
        last_err = jobs[job_id]["attempts"][-1]
        jobs[job_id]["provider"] = last_err['provider']
        jobs[job_id]["error"] = f"Generation failed: {last_err['provider']}"
        jobs[job_id]["details"] = f"Final Provider Error: {last_err['error']}"
    else:
        jobs[job_id]["provider"] = "None"
        jobs[job_id]["error"] = "No providers available"
        jobs[job_id]["details"] = "No attempts could be made"

    jobs[job_id]["full_audit"] = jobs[job_id]["attempts"]

def run_i2v(candidate, prompt, image_path, job_id):
    provider_name = f"HF-{candidate['type']}"
    logger.info(f"[I2V] [{provider_name}] Connecting to {candidate['url']}...")

    try:
        # Increase timeout for Wan2.2 Lightning to avoid 60s queue drop
        timeout_val = 900 if candidate["type"] == "wan22_lightning" else 300
        client = Client(candidate["url"], token=HF_TOKEN, httpx_kwargs={"timeout": timeout_val})

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

            logger.info(
                f"I2V_WAN22_RESPONSE | Job: {job_id} | "
                f"Type: {type(res)} | Result: {str(res)[:500]}"
            )

            return res

        if candidate["type"] == "wan21":
            if not os.getenv("DASHSCOPE_API_KEY"):
                logger.warning(f"I2V_WAN_SKIPPED | Job: {job_id} | Reason: DASHSCOPE_API_KEY not found in environment.")
                return {"error": "Authentication Required", "details": "Wan2.1 I2V requires DASHSCOPE_API_KEY which is not set in the environment."}

            logger.info(f"[I2V] [{provider_name}] Calling /i2v_generation_async...")
            res = client.predict(
                prompt,
                handle_file(image_path),
                False,
                -1,
                api_name="/i2v_generation_async"
            )
            logger.info(f"I2V_WAN_ASYNC_SUBMITTED | Raw Result Type: {type(res)}")

            task_id = None
            if isinstance(res, (list, tuple)) and len(res) > 0:
                task_id = res[0]
            elif isinstance(res, str):
                task_id = res

            if not task_id:
                logger.error(f"I2V_WAN_EXTRACT_FAILED | Raw result: {res}")
                return {"error": "Provider failed to return task_id", "details": str(res)[:500]}

            logger.info(f"I2V_WAN_TASK_ID | Task ID: {task_id}")

            for i in range(24): # 10 minutes max
                logger.info(f"I2V_WAN_WAITING | Attempt: {i}")
                time.sleep(5)
                try:
                    # status_refresh_1 inputs: task_id, task, status
                    status_res = client.predict(task_id, "i2v", False, api_name="/status_refresh_1")
                except Exception as e:
                    logger.warning(f"I2V_WAN_WAITING | Status Refresh Exception: {e}")
                    continue

                video_obj = status_res[0] if isinstance(status_res, (list, tuple)) else status_res

                if video_obj:
                    extracted = normalize_provider_result(video_obj, "internal", provider_name)
                    if extracted:
                        logger.info(f"I2V_WAN_FINAL_RESULT | Result: {str(video_obj)[:200]}")
                        return video_obj

            return {"error": "Timeout", "details": "Wan2.1 polling timed out after 10 minutes."}

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

    except Exception as e:
        error_msg = str(e)
        if "show_error=True" in error_msg:
            error_msg = f"Provider Internal Error (Gradio Exception). Details: {error_msg.split('show_error=True')[0]}"
        logger.warning(f"[I2V] [{provider_name}] FAILED: {error_msg}")
        return {"error": "Provider Exception", "details": error_msg}

    return {"error": "Unsupported Type", "details": candidate["type"]}

def process_i2v_production_sync(job_id, prompt, image_path):
    logger.info(f"I2V_WORKER_STARTED | Job: {job_id}")

    for cand in I2V_CANDIDATES:
        provider_id = f"HF/{cand['type']}"
        logger.info(f"I2V_PROVIDER_ATTEMPT | Job: {job_id} | Provider: {provider_id}")
        logger.info(f"I2V_PROVIDER_SUBMITTED | Job: {job_id} | Provider: {provider_id}")

        result = run_i2v(cand, prompt, image_path, job_id)

        if isinstance(result, dict) and "error" in result:
            logger.warning(f"I2V_PROVIDER_FAILED | Job: {job_id} | Provider: {provider_id} | Error: {result.get('error')}")
            jobs[job_id]["attempts"].append({
                "provider": provider_id,
                "error": result.get("error"),
                "details": result.get("details", "")[:250]
            })
            continue

        normalized_path = normalize_provider_result(result, job_id, provider_id)

        if normalized_path:
            if normalized_path.startswith("http://") or normalized_path.startswith("https://"):
                local_url = download_file(normalized_path, job_id)
                if local_url:
                    logger.info(f"I2V_PROVIDER_SUCCESS | Job: {job_id} | Provider: {provider_id}")
                    jobs[job_id]["status"] = "completed"
                    jobs[job_id]["video_url"] = local_url
                    jobs[job_id]["provider"] = provider_id
                    jobs[job_id]["end_time"] = time.time()
                    logger.info(f"I2V_JOB_COMPLETED | Job: {job_id} | URL: {local_url}")
                    return
                else:
                    err_msg = "Download from URL failed"
                    err_details = f"Failed to download from {normalized_path}"
            elif verify_file(normalized_path):
                local_url = safe_save(normalized_path, job_id, ".mp4")
                if local_url:
                    logger.info(f"I2V_PROVIDER_SUCCESS | Job: {job_id} | Provider: {provider_id}")
                    jobs[job_id]["status"] = "completed"
                    jobs[job_id]["video_url"] = local_url
                    jobs[job_id]["provider"] = provider_id
                    jobs[job_id]["end_time"] = time.time()
                    logger.info(f"I2V_JOB_COMPLETED | Job: {job_id} | URL: {local_url}")
                    logger.info(f"I2V_FILE_SAVED | Job: {job_id} | Path: {local_url}")
                    logger.info(f"I2V_VIDEO_URL | Job: {job_id} | URL: {local_url}")
                    logger.info(f"I2V_FINAL_RESULT | Job: {job_id} | URL: {local_url}")
                    return
                else:
                    err_msg = "File Verification Failed"
                    err_details = f"Could not save local file {normalized_path}"
            else:
                err_msg = "File Missing"
                err_details = f"Result path {normalized_path} was missing or empty"
        else:
            err_msg = "Extraction Failed"
            err_details = "Provider returned a result, but no usable video file/path/URL could be extracted."

        logger.warning(f"I2V_PROVIDER_FAILED | Job: {job_id} | Provider: {provider_id} | Error: {err_msg}")
        jobs[job_id]["attempts"].append({
            "provider": provider_id,
            "error": err_msg,
            "details": err_details[:250]
        })

    # 2. Try Replicate Fallback
    provider_id = "Replicate/minimax"
    logger.info(f"I2V_PROVIDER_ATTEMPT | Job: {job_id} | Provider: {provider_id} (HF Exhausted)")
    rep_result = run_replicate_i2v(prompt, image_path)

    if "url" in rep_result:
        ext_url = rep_result["url"]
        local_url = download_file(ext_url, job_id)
        if local_url:
            logger.info(f"I2V_PROVIDER_SUCCESS | Job: {job_id} | Provider: {provider_id}")
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["video_url"] = local_url
            jobs[job_id]["provider"] = provider_id
            jobs[job_id]["end_time"] = time.time()
            logger.info(f"I2V_JOB_COMPLETED | Job: {job_id} | URL: {local_url}")
            return
        else:
            rep_result["error"] = "Download Failed"
            rep_result["details"] = f"Could not download from {ext_url}"

    logger.warning(f"I2V_PROVIDER_FAILED | Job: {job_id} | Provider: {provider_id} | Error: {rep_result.get('error')}")
    jobs[job_id]["attempts"].append({
        "provider": provider_id,
        "error": rep_result.get("error"),
        "details": rep_result.get("details", "")[:250]
    })

    logger.error(f"I2V_JOB_FAILED | Job: {job_id} | Reason: ALL ENGINES FAILED")
    jobs[job_id]["status"] = "failed"
    jobs[job_id]["end_time"] = time.time()

    if jobs[job_id]["attempts"]:
        last_err = jobs[job_id]["attempts"][-1]
        jobs[job_id]["provider"] = last_err['provider']
        jobs[job_id]["error"] = "No free I2V provider is currently available."
        jobs[job_id]["details"] = f"Final Provider Error ({last_err['provider']}): {last_err['error']}"
    else:
        jobs[job_id]["provider"] = "None"
        jobs[job_id]["error"] = "No free I2V provider is currently available."
        jobs[job_id]["details"] = "No attempts could be made"

    jobs[job_id]["full_audit"] = jobs[job_id]["attempts"]

async def process_t2v_production(job_id, prompt):
    import asyncio
    logger.info(f"T2V_JOB_CREATED | Job: {job_id} | Prompt: {prompt[:100]}")

    if job_id not in jobs:
        jobs[job_id] = {}

    jobs[job_id]["status"] = "processing"
    jobs[job_id]["attempts"] = []
    jobs[job_id]["start_time"] = time.time()

    try:
        # Wrap the blocking sync function in a thread with a 15 minute overall timeout
        await asyncio.wait_for(
            asyncio.to_thread(process_t2v_production_sync, job_id, prompt),
            timeout=900.0
        )
    except asyncio.TimeoutError:
        logger.error(f"T2V_JOB_FAILED | Job: {job_id} | Reason: GLOBAL_TIMEOUT")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = "Generation timed out after 15 minutes."
        jobs[job_id]["details"] = "The backend worker exceeded the maximum allowed time."
        jobs[job_id]["full_audit"] = jobs[job_id].get("attempts", [])
        jobs[job_id]["end_time"] = time.time()
    except Exception as e:
        logger.error(f"T2V_JOB_FAILED | Job: {job_id} | Reason: INTERNAL_ERROR | Msg: {str(e)}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = "Internal Worker Error"
        jobs[job_id]["details"] = str(e)
        jobs[job_id]["full_audit"] = jobs[job_id].get("attempts", [])
        jobs[job_id]["end_time"] = time.time()

async def process_i2v_production(job_id, prompt, image_path):
    import asyncio
    logger.info(f"I2V_JOB_CREATED | Job: {job_id} | Prompt: {prompt[:100]}")

    if job_id not in jobs:
        jobs[job_id] = {}

    jobs[job_id]["status"] = "processing"
    jobs[job_id]["attempts"] = []
    jobs[job_id]["start_time"] = time.time()

    try:
        await asyncio.wait_for(
            asyncio.to_thread(process_i2v_production_sync, job_id, prompt, image_path),
            timeout=900.0
        )
    except asyncio.TimeoutError:
        logger.error(f"I2V_JOB_FAILED | Job: {job_id} | Reason: GLOBAL_TIMEOUT")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = "Generation timed out after 15 minutes."
        jobs[job_id]["details"] = "The backend worker exceeded the maximum allowed time."
        jobs[job_id]["full_audit"] = jobs[job_id].get("attempts", [])
        jobs[job_id]["end_time"] = time.time()
    except Exception as e:
        logger.error(f"I2V_JOB_FAILED | Job: {job_id} | Reason: INTERNAL_ERROR | Msg: {str(e)}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = "Internal Worker Error"
        jobs[job_id]["details"] = str(e)
        jobs[job_id]["full_audit"] = jobs[job_id].get("attempts", [])
        jobs[job_id]["end_time"] = time.time()

# --- IMAGE LOGIC ---

def run_pollinations_fallback(prompt, job_id):
    logger.info("[IMAGE] Trying Pollinations fallback...")
    encoded = requests.utils.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&nologo=true&model=flux"
    try:
        res = requests.get(url, timeout=30)
        if res.status_code == 200:
            path = os.path.join(OUTPUT_DIR, f"{job_id}.jpg")
            with open(path, "wb") as f: f.write(res.content)
            if verify_file(path):
                return f"/api/outputs/{job_id}.jpg"
    except: pass
    return None

def run_image_gen(cand, prompt):
    client = Client(cand["url"])
    if cand["type"] == "flux_schnell" or cand["type"] == "flux_dev":
        res = client.predict(prompt, 0, True, 1024, 1024, 4 if cand["type"] == "flux_schnell" else 28, api_name="/infer")
        return res[0].get("path") if (res and isinstance(res, (list, tuple))) else None
    if cand["type"] == "sd35":
        res = client.predict(prompt, "low quality", 0, True, 1024, 1024, 4.5, 28, api_name="/infer")
        return res[0] if (res and isinstance(res, (list, tuple))) else None
    if cand["type"] == "sdxl":
        res = client.predict(prompt, "", "", "", False, False, False, 0, 1024, 1024, 5.0, 5.0, 25, 25, True, api_name="/predict")
        return res if isinstance(res, str) else None
    return None

@app.post("/api/prompt-to-image")
async def api_p2i(prompt: str = Form(...), style: str = Form("Realistic")):
    job_id = str(uuid.uuid4())
    final_prompt = f"{prompt}, {style} style, masterpiece, cinematic"
    logger.info(f"[IMAGE] New Request: {final_prompt}")

    # 1. Try HF candidates with retry logic
    for cand in IMAGE_CANDIDATES:
        for attempt in range(3): # Max 3 attempts per candidate
            try:
                logger.info(f"[IMAGE] Attempt {attempt+1} on {cand['url']}")
                temp = run_image_gen(cand, final_prompt)
                if temp:
                    url = safe_save(temp, job_id, ".webp")
                    if url:
                        logger.info(f"[IMAGE] SUCCESS on {cand['url']}")
                        return {"success": True, "job_id": job_id, "url": url}
                break # If successful but safe_save failed, or if provider returned None, move to next candidate
            except Exception as e:
                err_msg = str(e)
                if "503" in err_msg or "busy" in err_msg.lower() or "Queue" in err_msg:
                    wait_time = (attempt + 1) * 2
                    logger.warning(f"[IMAGE] {cand['url']} busy (503), waiting {wait_time}s... ({err_msg[:100]})")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.warning(f"[IMAGE] {cand['url']} failed: {e}")
                    break # Critical error, try next candidate

    # 2. Try Pollinations (Always success if internet is on)
    url = run_pollinations_fallback(final_prompt, job_id)
    if url:
        return {"success": True, "job_id": job_id, "url": url}

    return JSONResponse(status_code=503, content={"success": False, "error": "IMAGE_QUOTA", "message": "All image engines are currently busy. Please try again in a moment."})

@app.post("/api/text-to-video")
async def api_t2v(background_tasks: BackgroundTasks, text: str = Form(...), style: str = Form("Realistic")):
    jid = str(uuid.uuid4())
    final_prompt = f"{text}, {style} style"
    jobs[jid] = {
        "status": "queued",
        "version": VERSION,
        "full_audit": [],
        "attempts": []
    }
    background_tasks.add_task(process_t2v_production, jid, final_prompt)
    return {"success": True, "job_id": jid, "status": "queued", "version": VERSION}

@app.post("/api/prompt-to-video")
async def api_p2v(background_tasks: BackgroundTasks, prompt: str = Form(...), style: str = Form("Realistic")):
    jid = str(uuid.uuid4())
    final_prompt = f"{prompt}, {style} style"
    jobs[jid] = {
        "status": "queued",
        "version": VERSION,
        "full_audit": [],
        "attempts": []
    }
    background_tasks.add_task(process_t2v_production, jid, final_prompt)
    return {"success": True, "job_id": jid, "status": "queued", "version": VERSION}

@app.post("/api/image-to-video")
async def api_i2v(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    prompt: str = Form(...),
    style: str = Form("Realistic")
):
    jid = str(uuid.uuid4())

    # Save the uploaded image locally
    ext = os.path.splitext(image.filename)[1]
    if not ext: ext = ".jpg"
    image_path = os.path.join(OUTPUT_DIR, f"input_{jid}{ext}")

    with open(image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    jobs[jid] = {
        "status": "queued",
        "version": VERSION,
        "full_audit": [],
        "attempts": []
    }

    final_prompt = f"{prompt}, {style} style" if style else prompt
    background_tasks.add_task(process_i2v_production, jid, final_prompt, image_path)
    return {"success": True, "job_id": jid, "status": "queued", "version": VERSION}

@app.get("/api/video-job/{job_id}")
def get_api_job(job_id: str):
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={"error": "Not found", "details": "Job ID not in memory. Server might have restarted."})
    return jobs.get(job_id)

import cv2

def analyze_image_hf(image_path: str) -> dict:
    # Use Gradio Space for high quality vision analysis
    # Optimized for Raka AI generative prompts
    try:
        # primary: Joy Caption (Excellent for prompts)
        # fallback: BLIP (Extremely reliable)
        candidates = [
            {"url": "fancyfeast/joy-caption-pre-alpha", "fn": "/stream_chat", "type": "joy"},
            {"url": "tonyassi/blip-image-captioning-large", "fn": "/predict", "type": "blip"}
        ]

        last_error = ""
        for cand in candidates:
            try:
                logger.info(f"VLM_GRADIO_ATTEMPT | Space: {cand['url']}")
                client = Client(cand["url"], token=HF_TOKEN)
                if cand["type"] == "joy":
                    result = client.predict(
                        handle_file(image_path),
                        api_name=cand["fn"]
                    )
                else:
                    # BLIP predict: image, min_tokens, max_tokens
                    result = client.predict(
                        handle_file(image_path),
                        20,
                        80,
                        api_name=cand["fn"]
                    )

                if result:
                    # If it's BLIP, wrap it to make it a better prompt
                    final_prompt = result
                    # Clean up BLIP metadata if present
                    if "⏱" in result:
                        final_prompt = result.split("⏱")[0].strip()

                    if cand["type"] == "blip":
                        final_prompt = f"A professional detailed photograph of {final_prompt.strip()}, cinematic lighting, highly detailed, 8k masterpiece"

                    return {"success": True, "prompt": final_prompt, "provider": cand["url"]}
            except Exception as e:
                last_error = str(e)
                logger.warning(f"VLM_GRADIO_FAILED | Space: {cand['url']} | Error: {last_error[:100]}")
                continue

        return {"success": False, "error_code": "PROVIDER_UNAVAILABLE", "message": f"Vision analysis unavailable. ({last_error[:50]})"}

    except Exception as e:
        return {"success": False, "error_code": "INTERNAL_ERROR", "message": str(e)}

def analyze_video_hf(video_path: str) -> dict:
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"success": False, "error_code": "INVALID_INPUT", "message": "Could not open video file"}

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0: frame_count = 60

        # Use 3 frames to save time and prevent timeouts
        indices = [int(frame_count * i) for i in [0.2, 0.5, 0.8]]
        frames_paths = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                f_path = os.path.join(OUTPUT_DIR, f"frame_{uuid.uuid4()}.jpg")
                cv2.imwrite(f_path, frame)
                frames_paths.append(f_path)

        cap.release()

        if not frames_paths:
            return {"success": False, "error_code": "EXTRACTION_FAILED", "message": "Failed to extract frames"}

        # Analyze frames sequentially
        descriptions = []
        for fp in frames_paths:
            res = analyze_image_hf(fp)
            if res.get("success"):
                descriptions.append(res.get("prompt"))
            if os.path.exists(fp): os.remove(fp)

        if not descriptions:
            return {"success": False, "error_code": "PROVIDER_UNAVAILABLE", "message": "Video frames analysis failed"}

        # Combine descriptions into a coherent narrative
        final_prompt = f"A cinematic video sequence. Beginning: {descriptions[0]}. Development: {descriptions[1]}. Ending: {descriptions[len(descriptions)-1]}."

        return {
            "success": True,
            "prompt": final_prompt,
            "provider": "RakaVision-Gradio-Ensemble"
        }

    except Exception as e:
        return {"success": False, "error_code": "INTERNAL_ERROR", "message": str(e)}

@app.post("/api/image-to-prompt")
async def api_image_to_prompt(image: UploadFile = File(...)):
    jid = str(uuid.uuid4())
    logger.info(f"I2P_JOB_CREATED | Job: {jid}")

    ext = os.path.splitext(image.filename)[1]
    if not ext: ext = ".jpg"
    image_path = os.path.join(OUTPUT_DIR, f"input_i2p_{jid}{ext}")

    with open(image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    res = analyze_image_hf(image_path)

    # Cleanup temp file
    if os.path.exists(image_path):
        os.remove(image_path)

    if not res.get("success"):
        logger.error(f"I2P_JOB_FAILED | Job: {jid} | Error: {res.get('message')}")
        status_code = 500
        if res.get("error_code") == "AUTH_ERROR": status_code = 401
        elif res.get("error_code") == "QUOTA_EXHAUSTED": status_code = 402
        elif res.get("error_code") == "RATE_LIMIT": status_code = 429
        elif res.get("error_code") == "PROVIDER_UNAVAILABLE": status_code = 503
        return JSONResponse(status_code=status_code, content=res)

    logger.info(f"I2P_JOB_COMPLETED | Job: {jid} | Provider: {res['provider']}")
    return {
        "success": True,
        "job_id": jid,
        "prompt": res["prompt"],
        "provider": res["provider"],
        "model": res["provider"]
    }

@app.post("/api/video-to-prompt")
async def api_video_to_prompt(video: UploadFile = File(...)):
    jid = str(uuid.uuid4())
    logger.info(f"V2P_JOB_CREATED | Job: {jid}")

    ext = os.path.splitext(video.filename)[1]
    if not ext: ext = ".mp4"
    video_path = os.path.join(OUTPUT_DIR, f"input_v2p_{jid}{ext}")

    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    res = analyze_video_hf(video_path)

    # Cleanup temp file
    if os.path.exists(video_path):
        os.remove(video_path)

    if not res.get("success"):
        logger.error(f"V2P_JOB_FAILED | Job: {jid} | Error: {res.get('message')}")
        status_code = 500
        if res.get("error_code") == "AUTH_ERROR": status_code = 401
        elif res.get("error_code") == "QUOTA_EXHAUSTED": status_code = 402
        elif res.get("error_code") == "RATE_LIMIT": status_code = 429
        elif res.get("error_code") == "PROVIDER_UNAVAILABLE": status_code = 503
        return JSONResponse(status_code=status_code, content=res)

    logger.info(f"V2P_JOB_COMPLETED | Job: {jid} | Provider: {res['provider']}")
    return {
        "success": True,
        "job_id": jid,
        "prompt": res["prompt"],
        "provider": res["provider"],
        "model": res["provider"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
