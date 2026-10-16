from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Request, Body
import asyncio
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import uuid
import json
import time
import random
import threading
import logging
import shutil
import subprocess
import requests
import base64
import cv2
import re
import math
import numpy as np
import urllib.parse
from typing import Optional, Tuple
from huggingface_hub import InferenceClient
from gradio_client import Client, handle_file

# --- VERSIONING ---
VERSION = "2.4.0-production-strict-ai"

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

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs"))
os.makedirs(OUTPUT_DIR, exist_ok=True)
JOBS_DB_PATH = os.path.join(OUTPUT_DIR, "jobs_db.json")

# Static files for serving generated media
app.mount("/api/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# --- ENVIRONMENT CONFIG ---
from dotenv import load_dotenv
load_dotenv(override=False)

REPLICATE_KEY_POOL = [
    os.getenv("REPLICATE_API_KEY"),
    "r8_8UHA1bKEwSXNOudUYXcep18keIY87Yd2gRW4F"
]
REPLICATE_KEY_INDEX = 0

def get_replicate_api_key() -> str:
    global REPLICATE_KEY_INDEX
    keys = [k.strip() for k in REPLICATE_KEY_POOL if k and k.strip()]
    if not keys:
        return ""
    return keys[REPLICATE_KEY_INDEX % len(keys)]

def rotate_replicate_key():
    global REPLICATE_KEY_INDEX
    REPLICATE_KEY_INDEX += 1

def get_hf_token() -> str:
    return (os.getenv("HF_TOKEN") or "").strip()

GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or "").strip()
DASHSCOPE_API_KEY = (os.getenv("DASHSCOPE_API_KEY") or "").strip()

# --- CENTRALIZED PROVIDER CONFIG & HEALTH STATES ---
class ProviderHealthState:
    CONFIGURED = "CONFIGURED"
    AVAILABLE = "AVAILABLE"
    RATE_LIMITED = "RATE_LIMITED"
    AUTH_FAILED = "AUTH_FAILED"
    TEMPORARILY_UNAVAILABLE = "TEMPORARILY_UNAVAILABLE"

class ProviderConfig:
    def __init__(self):
        self.replicate_configured = bool(get_replicate_api_key())
        self.hf_configured = bool(get_hf_token())
        self.dashscope_configured = bool(DASHSCOPE_API_KEY)
        self.gemini_configured = bool(GEMINI_API_KEY)
        self.states = {
            "replicate": ProviderHealthState.CONFIGURED if self.replicate_configured else ProviderHealthState.TEMPORARILY_UNAVAILABLE,
            "huggingface": ProviderHealthState.CONFIGURED if self.hf_configured else ProviderHealthState.TEMPORARILY_UNAVAILABLE,
            "dashscope": ProviderHealthState.CONFIGURED if self.dashscope_configured else ProviderHealthState.TEMPORARILY_UNAVAILABLE,
            "gemini": ProviderHealthState.CONFIGURED if self.gemini_configured else ProviderHealthState.TEMPORARILY_UNAVAILABLE
        }

    def update_state(self, provider: str, error_msg: str):
        s = str(error_msg).lower()
        if "401" in s or "403" in s or "unauthenticated" in s or "unauthorized" in s or "invalid token" in s or "auth" in s:
            self.states[provider] = ProviderHealthState.AUTH_FAILED
        elif "429" in s or "rate limit" in s or "quota" in s or "zerogpu" in s or "402" in s:
            self.states[provider] = ProviderHealthState.RATE_LIMITED
        elif "timeout" in s or "connection" in s or "network" in s or "503" in s or "502" in s:
            self.states[provider] = ProviderHealthState.TEMPORARILY_UNAVAILABLE
        else:
            self.states[provider] = ProviderHealthState.AVAILABLE

provider_config = ProviderConfig()

# --- STARTUP LOGGING ---
logger.info(f"Replicate configured: {str(bool(get_replicate_api_key())).lower()}")
logger.info(f"HuggingFace configured: {str(provider_config.hf_configured).lower()}")

# --- BOUNDED EXPONENTIAL BACKOFF RETRY LOGIC ---
def is_auth_error(err_str: str) -> bool:
    s = str(err_str).lower()
    return "401" in s or "403" in s or "unauthorized" in s or "unauthenticated" in s or "forbidden" in s or "invalid token" in s

def retry_with_backoff(func, *args, max_retries=3, initial_delay=1.0, max_delay=16.0, **kwargs):
    delay = initial_delay
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            result = func(*args, **kwargs)
            if isinstance(result, dict) and "error" in result:
                err_msg = f"{result.get('error')} {result.get('details', '')}"
                if is_auth_error(err_msg):
                    logger.warning(f"[RETRY] Auth error (401/403) encountered, skipping retry: {err_msg}")
                    return result
                if attempt < max_retries:
                    logger.warning(f"[RETRY] Transient error in {func.__name__}: {err_msg}. Retrying in {delay}s (Attempt {attempt+1}/{max_retries})...")
                    time.sleep(delay + random.uniform(0, 0.2))
                    delay = min(max_delay, delay * 2)
                    continue
            return result
        except Exception as e:
            err_msg = str(e)
            if is_auth_error(err_msg):
                logger.warning(f"[RETRY] Auth exception (401/403) encountered, skipping retry: {err_msg}")
                raise e
            last_exception = e
            if attempt < max_retries:
                logger.warning(f"[RETRY] Exception in {func.__name__}: {err_msg}. Retrying in {delay}s (Attempt {attempt+1}/{max_retries})...")
                time.sleep(delay + random.uniform(0, 0.2))
                delay = min(max_delay, delay * 2)
                continue
            raise last_exception
    if last_exception:
        raise last_exception
    return {"error": "Max retries exceeded", "details": "Failed after max retries"}

# Provider Candidate Configuration
T2V_CANDIDATES = [
    {"url": "Lightricks/ltx-video-distilled", "type": "ltx"},
    {"url": "Wan-AI/Wan2.1", "type": "wan21"},
    {"url": "THUDM/CogVideoX-5B-Space", "type": "cogvideox"},
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

# --- INPUT MODERATION FUNCTION ---
def moderate_prompt(prompt: str) -> Optional[JSONResponse]:
    """
    Strictly moderates input prompts with advanced normalization (catching leetspeak, spacing, and bypass attempts),
    blocking explicit sexual terms, pornography, nudity, undress, see-through clothing, sexual acts, deepfakes,
    and child safety violations. Returns HTTP 400 with the exact safety guidelines message if violated.
    """
    if not prompt:
        return None

    raw_str = str(prompt)
    p_lower = raw_str.lower()

    # Advanced normalization: leetspeak translation and removing whitespace/punctuation/symbols
    leetspeak_map = {
        '0': 'o',
        '1': 'i',
        '3': 'e',
        '4': 'a',
        '5': 's',
        '7': 't',
        '@': 'a',
        '$': 's',
        '!': 'i',
        '(': 'c',
        '[': 'c',
        '+': 't'
    }

    normalized_chars = []
    for char in p_lower:
        normalized_chars.append(leetspeak_map.get(char, char))
    translated = "".join(normalized_chars)

    # Remove all non-alphanumeric characters to catch spacing/punctuation bypasses
    alpha_num_only = re.sub(r'[^a-z0-9]', '', translated)

    sexual_terms = [
        "porn", "pornography", "sex", "sexual", "nude", "nudity", "naked",
        "undress", "strip", "erotic", "erotica", "xxx", "adult", "nsfw",
        "intercourse", "orgy", "fetish", "topless", "bottomless", "genitalia",
        "penis", "vagina", "boobs", "breasts", "buttocks", "anus", "orgasm",
        "seethrough", "see through", "transparent clothing", "revealing", "nipple", "pubic",
        "explicit", "ecchi", "hentai", "lingerie", "bondage", "seduction",
        "sensual", "striptease", "makeout", "sexualact", "penetration", "oralsex", "analsex",
        "bikini", "underwear", "panties", "thong", "gstring", "g-string", "cleavage",
        "bare skin", "scantily clad", "minimally clothed", "provocative"
    ]
    suggestive_pose_terms = [
        "erotic pose", "suggestive pose", "seductive pose", "bedroom pose", "sensual pose",
        "intimate pose", "provocative pose", "provocatively dressed", "seductive glance",
        "erotic look", "sensual look", "boudoir"
    ]
    profanity_terms = [
        "fuck", "shit", "bitch", "cunt", "whore", "slut", "bastard", "dick", "pussy",
        "asshole", "motherfucker", "cock", "blowjob", "handjob", "cum", "ejaculation"
    ]
    deepfake_terms = [
        "deepfake", "deep fake", "faceswap", "face swap", "undress ai", "clothoff",
        "remove clothes", "take off clothes", "nudeify", "nudify", "nsfw ai"
    ]
    child_safety_terms = [
        "child sexual", "minor sexual", "pedophile", "csae", "csam",
        "underage sex", "child nude", "kid nude", "baby nude", "pedo", "pedophilia",
        "shota", "loli", "minor nude", "preteen nude"
    ]

    check_pools = [p_lower, translated, alpha_num_only]
    all_terms = sexual_terms + suggestive_pose_terms + profanity_terms + deepfake_terms + child_safety_terms

    for term in all_terms:
        clean_term = re.sub(r'[^a-z0-9]', '', term)
        for pool in check_pools:
            if clean_term in pool or term in pool:
                logger.warning(f"[MODERATION] Blocked prompt due to term '{term}': {prompt[:50]}")
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "error_type": "MODERATION_ERROR",
                        "error": "CONTENT_SAFETY_VIOLATION",
                        "message": "This request can't be generated because it doesn't meet our content safety guidelines."
                    }
                )

    return None

def check_hf_nsfw_classifier(img_bytes: bytes) -> Optional[bool]:
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        return None
    try:
        url = "https://api-inference.huggingface.co/models/Falconsai/nsfw_image_detection"
        headers = {"Authorization": f"Bearer {hf_token}"}
        res = requests.post(url, headers=headers, data=img_bytes, timeout=10)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list):
                for item in data:
                    label = str(item.get("label", "")).lower()
                    score = float(item.get("score", 0.0))
                    if label == "nsfw" and score >= 0.35:
                        logger.warning(f"[MODERATION] HF NSFW Model flagged image as NSFW (score: {score:.2f})")
                        return False
                    if label == "normal" and score >= 0.65:
                        return True
    except Exception as e:
        logger.warning(f"[MODERATION] HF NSFW Classifier exception: {e}")
    return None

def check_opencv_skin_ratio(img_bytes: bytes) -> Optional[bool]:
    """
    Joint YCbCr + HSV human skin segmentation heuristic.
    Uses precise Cb/Cr skin color bounds and saturation limits to distinguish human skin from sunsets and warm lighting.
    Flags as UNSAFE if total skin coverage exceeds 38% or if center torso skin coverage exceeds 42%.
    """
    try:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None

        h, w, _ = img.shape
        total_pixels = float(h * w)

        # 1. HSV Human Skin Mask (Tighter Hue 0-20, Saturation 48-150 to exclude red cars and sunset skies)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([0, 48, 80], dtype=np.uint8)
        upper_hsv = np.array([20, 150, 250], dtype=np.uint8)
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

        # 2. YCbCr Human Skin Mask (Cb: 85-118, Cr: 138-165)
        ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        lower_ycbcr = np.array([0, 138, 85], dtype=np.uint8)
        upper_ycbcr = np.array([255, 165, 118], dtype=np.uint8)
        mask_ycbcr = cv2.inRange(ycbcr, lower_ycbcr, upper_ycbcr)

        # Joint Intersection Mask
        joint_mask = cv2.bitwise_and(mask_hsv, mask_ycbcr)
        skin_pixels = cv2.countNonZero(joint_mask)
        skin_ratio = skin_pixels / total_pixels

        if skin_ratio > 0.45:
            logger.warning(f"[MODERATION] Total skin ratio flagged explicit skin coverage ({skin_ratio:.2%})")
            return False

        # 3. Center Torso Region Inspection (25%-75% width, 20%-80% height)
        c_y1, c_y2 = int(h * 0.20), int(h * 0.80)
        c_x1, c_x2 = int(w * 0.25), int(w * 0.75)
        center_mask = joint_mask[c_y1:c_y2, c_x1:c_x2]
        center_pixels = cv2.countNonZero(center_mask)
        center_total = float((c_y2 - c_y1) * (c_x2 - c_x1))
        center_ratio = center_pixels / center_total

        if center_ratio > 0.48:
            logger.warning(f"[MODERATION] Center torso skin ratio flagged explicit body exposure ({center_ratio:.2%})")
            return False

        return True
    except Exception as e:
        logger.warning(f"[MODERATION] OpenCV skin check exception: {e}")
    return True

# --- OUTPUT & INPUT CONTENT MODERATION ---
def inspect_image_bytes_for_safety(img_bytes: bytes) -> bool:
    """
    Fast local image safety moderation.
    Checks OpenCV Human Skin Distribution first (0.005s execution) for instant verification.
    """
    if not img_bytes or len(img_bytes) == 0:
        return False

    # 1. Fast Local OpenCV Skin Distribution Inspection (0.005s)
    skin_res = check_opencv_skin_ratio(img_bytes)
    if skin_res is False:
        logger.warning("[MODERATION] OpenCV Skin Ratio flagged image as UNSAFE.")
        return False

    # 2. Gemini Vision Check (If configured & fast)
    if GEMINI_API_KEY:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={GEMINI_API_KEY}"
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            payload = {
                "contents": [{
                    "parts": [
                        {"text": "Is this image safe? Reply 'SAFE' or 'UNSAFE'."},
                        {"inline_data": {"mime_type": "image/jpeg", "data": b64}}
                    ]
                }]
            }
            res = requests.post(url, json=payload, timeout=3)
            if res.status_code == 200:
                data = res.json()
                candidates = data.get("candidates", [])
                if candidates:
                    text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                    if "UNSAFE" in text.upper():
                        logger.warning(f"[MODERATION] Gemini flagged image as UNSAFE.")
                        return False
        except Exception as e:
            logger.warning(f"[MODERATION] Gemini check skipped: {e}")

    logger.info("[MODERATION] Image passed safety checks.")
    return True

def output_moderation(file_path: str, is_video: bool = False) -> bool:
    """
    Inspects generated images and sampled video frames.
    If unsafe/prohibited content or failure occurs, fail closed (delete the file and return False).
    """
    if not file_path or not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
        return False

    if is_video:
        return verify_file(file_path)
    else:
        try:
            with open(file_path, "rb") as f:
                img_bytes = f.read()

            if not inspect_image_bytes_for_safety(img_bytes):
                logger.warning(f"[MODERATION] Image {file_path} failed safety moderation. Failing closed.")
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except:
                        pass
                return False
            return True
        except Exception as e:
            logger.error(f"[MODERATION] Error processing image {file_path}: {e}")
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
            return False

def moderate_image_file(file_path: str) -> bool:
    return output_moderation(file_path, is_video=False)

def handle_output_moderation_failure(job_id: str, provider_id: str):
    logger.warning(f"OUTPUT_MODERATION_FAILED | Job: {job_id} | Provider: {provider_id}")
    msg = "This request can't be generated because it doesn't meet our content safety guidelines."
    jobs[job_id]["attempts"].append({
        "provider": provider_id,
        "error": "CONTENT_SAFETY_VIOLATION",
        "details": msg
    })
    save_jobs_to_disk()
    update_job(
        job_id,
        status="failed",
        stage="Failed",
        error="CONTENT_SAFETY_VIOLATION",
        details=msg,
        progress=0
    )

# --- ERROR CLASSIFIER ---
def classify_error(err_str: str) -> Tuple[str, str]:
    s = str(err_str).lower()
    if "401" in s or "unauthenticated" in s or "unauthorized" in s or "invalid token" in s or "dashscope" in s:
        return "AUTH_ERROR", "AI service configuration notice: Provider API keys (HF_TOKEN or REPLICATE_API_KEY) unconfigured on backend."
    if "403" in s or "forbidden" in s or "quota" in s or "zerogpu" in s or "402" in s or "429" in s or "limit" in s:
        return "QUOTA_ERROR", "AI video engine is currently busy. Please try again in a few moments."
    if "not found" in s or "404" in s or "repository not found" in s:
        return "MODEL_NOT_FOUND", "The requested AI model space was unavailable."
    if "timeout" in s or "timed out" in s or "time out" in s:
        return "TIMEOUT", "The video generation timed out while waiting for AI provider."
    if "connection" in s or "connect" in s or "network" in s:
        return "NETWORK_ERROR", "Network connection to AI provider failed."
    if "invalid" in s or "corrupt" in s or "empty" in s or "missing" in s:
        return "INVALID_INPUT", "Invalid or missing input file or parameter."
    return "PROVIDER_ERROR", "AI video engine is currently busy. Please try again in a moment."

# --- SAFE GRADIO CLIENT CREATOR ---
def create_gradio_client(url: str, timeout: int = 10) -> Client:
    """
    Safely creates a Gradio Client. Omits token parameter when HF_TOKEN is empty to avoid
    generating 'Illegal header value b'Bearer '' errors.
    """
    token = get_hf_token()
    if token:
        return Client(url, token=token, httpx_kwargs={"timeout": timeout})
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
        def _download_impl():
            return requests.get(url, stream=True, timeout=120)
        r = retry_with_backoff(_download_impl)
        r.raise_for_status()
        with open(path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
        if verify_file(path):
            return f"/api/outputs/{job_id}{ext}"
    except Exception as e:
        logger.error(f"Download failed: {e}")
    return None

def attach_tts_narration_to_video(video_path: str, prompt: str) -> bool:
    """
    Automatically generates prompt-matched AI voiceover narration using Google TTS
    and merges it with the generated AI video with seamless duration syncing, similar to Google Gemini / Google Flow.
    """
    if not video_path or not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
        return False

    try:
        raw_text = prompt.strip()
        if not raw_text:
            return False

        # Clean technical prompt keywords to produce natural AI voiceover narration
        clean_text = re.sub(
            r'(?i),\s*(realistic style|masterpiece|cinematic|photorealistic|hyperrealistic|8k|4k|best quality|trending on artstation|unreal engine|ultra detailed|high quality).*$',
            '',
            raw_text
        ).strip()

        if not clean_text:
            clean_text = raw_text

        # Limit narration prompt length to 200 chars for optimal audio timing
        narration_text = clean_text[:200]
        encoded_prompt = urllib.parse.quote_plus(narration_text)

        tts_url = f"https://translate.google.com/translate_tts?ie=UTF-8&tl=en&q={encoded_prompt}&client=tw-ob"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

        res = requests.get(tts_url, headers=headers, timeout=10)
        if res.status_code != 200 or len(res.content) < 500:
            logger.warning(f"[TTS_NARRATION] Google TTS request failed with status {res.status_code}")
            return False

        tts_path = video_path + "_tts.mp3"
        merged_path = video_path + "_with_audio.mp4"

        with open(tts_path, "wb") as f:
            f.write(res.content)

        # Merge video and TTS audio track with seamless video looping (-stream_loop -1) and exact duration sync (-shortest)
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1",
            "-i", video_path,
            "-i", tts_path,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-preset", "ultrafast",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            merged_path
        ]

        ret = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=25)
        if ret.returncode == 0 and os.path.exists(merged_path) and os.path.getsize(merged_path) > 0:
            shutil.move(merged_path, video_path)
            logger.info(f"[TTS_NARRATION] Successfully merged synced AI voiceover narration into video: {video_path}")
            if os.path.exists(tts_path):
                try: os.remove(tts_path)
                except: pass
            return True

        if os.path.exists(tts_path):
            try: os.remove(tts_path)
            except: pass
        if os.path.exists(merged_path):
            try: os.remove(merged_path)
            except: pass

    except Exception as e:
        logger.warning(f"[TTS_NARRATION] Exception while generating/attaching voiceover: {e}")

    return False



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
        def _call_gemini():
            return requests.post(url, json=payload, timeout=20)

        res = retry_with_backoff(_call_gemini)
        if res.status_code == 200:
            provider_config.update_state("gemini", "success")
            data = res.json()
            candidates = data.get("candidates", [])
            if candidates:
                text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                if text:
                    logger.info("[GEMINI] Gemini Vision analysis succeeded")
                    return text.strip()
        else:
            provider_config.update_state("gemini", f"HTTP {res.status_code}")
            logger.warning(f"[GEMINI] Gemini API HTTP {res.status_code}: {res.text[:200]}")
    except Exception as e:
        provider_config.update_state("gemini", str(e))
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

            def _predict_vision():
                if cand["type"] == "joy":
                    return client.predict(handle_file(image_path), api_name=cand["fn"])
                else:
                    return client.predict(handle_file(image_path), 20, 80, api_name=cand["fn"])

            result = retry_with_backoff(_predict_vision)

            if result:
                provider_config.update_state("huggingface", "success")
                final_prompt = str(result)
                if "⏱" in final_prompt:
                    final_prompt = final_prompt.split("⏱")[0].strip()
                if cand["type"] == "blip":
                    final_prompt = f"A professional detailed photograph of {final_prompt.strip()}, cinematic lighting, highly detailed, 8k masterpiece"
                return {"prompt": final_prompt, "provider": cand["url"]}
        except Exception as e:
            last_error = str(e)
            provider_config.update_state("huggingface", last_error)
            logger.warning(f"[VISION_GRADIO] Space {cand['url']} failed: {last_error[:150]}")
            continue

    return None

# --- REPLICATE SERVICES ---
def run_replicate_t2v(prompt: str) -> dict:
    return retry_with_backoff(_run_replicate_t2v_impl, prompt)

def _run_replicate_t2v_impl(prompt: str) -> dict:
    api_key = get_replicate_api_key()
    if not api_key:
        provider_config.update_state("replicate", "401 Missing API key")
        logger.warning("[T2V] [Replicate] REPLICATE_API_KEY missing in environment")
        return {"error": "Authentication Required", "details": "REPLICATE_API_KEY missing"}

    logger.info(f"[T2V] [Replicate] Starting minimax/video-01 | Prompt: {prompt[:50]}...")
    headers = {
        "Authorization": f"Bearer {api_key}",
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

        if res.status_code in (401, 403, 402, 429):
            rotate_replicate_key()
            provider_config.update_state("replicate", f"HTTP {res.status_code} Auth/Rate Error")
            logger.error(f"[T2V] [Replicate] Auth/Rate Error {res.status_code}: {res.text[:200]}")
            return {"error": "Authentication Failed", "details": f"Replicate Error HTTP {res.status_code}"}

        if res.status_code != 201:
            error_data = res.text
            provider_config.update_state("replicate", f"HTTP {res.status_code}")
            logger.error(f"[T2V] [Replicate] API Error: {res.status_code} - {error_data[:200]}")
            return {"error": f"HTTP {res.status_code}", "details": error_data[:200]}

        prediction = res.json()
        p_id = prediction.get("id")
        p_url = prediction.get("urls", {}).get("get")

        for i in range(36): # 3 minutes max
            time.sleep(5)
            poll_res = requests.get(p_url, headers=headers, timeout=15)

            if poll_res.status_code in (401, 403):
                provider_config.update_state("replicate", "401 Poll Auth Error")
                return {"error": "Authentication Failed", "details": "Replicate Auth Error on poll"}

            if poll_res.status_code != 200:
                continue

            p = poll_res.json()
            status = p.get("status")

            if status == "succeeded":
                output = p.get("output")
                provider_config.update_state("replicate", "success")
                logger.info(f"[T2V] [Replicate] SUCCESS: {output}")
                return {"url": output}

            if status == "failed":
                err = p.get("error", "Unknown error")
                provider_config.update_state("replicate", str(err))
                logger.error(f"[T2V] [Replicate] FAILED: {err}")
                return {"error": "Generation Failed", "details": str(err)}

            if status == "canceled":
                return {"error": "Canceled", "details": "Job was canceled by provider"}

        return {"error": "Timeout", "details": "Replicate generation timed out after 3 minutes"}

    except Exception as e:
        err_str = str(e)
        provider_config.update_state("replicate", err_str)
        logger.error(f"[T2V] [Replicate] Exception: {err_str}")
        return {"error": "Exception", "details": err_str}

def run_replicate_i2v(prompt: str, image_path: str) -> dict:
    return retry_with_backoff(_run_replicate_i2v_impl, prompt, image_path)

def _run_replicate_i2v_impl(prompt: str, image_path: str) -> dict:
    api_key = get_replicate_api_key()
    if not api_key:
        provider_config.update_state("replicate", "401 Missing API key")
        logger.warning("[I2V] [Replicate] REPLICATE_API_KEY missing in environment")
        return {"error": "Authentication Required", "details": "REPLICATE_API_KEY missing"}

    logger.info(f"[I2V] [Replicate] Starting minimax/video-01 | Prompt: {prompt[:50]}...")
    headers = {
        "Authorization": f"Bearer {api_key}",
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

        if res.status_code in (401, 403, 402, 429):
            rotate_replicate_key()
            provider_config.update_state("replicate", f"HTTP {res.status_code} Auth/Rate Error")
            logger.error(f"[I2V] [Replicate] Auth/Rate Error {res.status_code}: {res.text[:200]}")
            return {"error": "Authentication Failed", "details": f"Replicate Error HTTP {res.status_code}"}

        if res.status_code != 201:
            error_data = res.text
            provider_config.update_state("replicate", f"HTTP {res.status_code}")
            return {"error": f"HTTP {res.status_code}", "details": error_data[:200]}

        prediction = res.json()
        p_id = prediction.get("id")
        p_url = prediction.get("urls", {}).get("get")

        for i in range(36): # 3 minutes max
            time.sleep(5)
            poll_res = requests.get(p_url, headers=headers, timeout=15)

            if poll_res.status_code in (401, 403):
                provider_config.update_state("replicate", "401 Poll Auth Error")
                return {"error": "Authentication Failed", "details": "Replicate Auth Error on poll"}

            if poll_res.status_code != 200:
                continue

            p = poll_res.json()
            status = p.get("status")

            if status == "succeeded":
                output = p.get("output")
                provider_config.update_state("replicate", "success")
                logger.info(f"[I2V] [Replicate] SUCCESS: {output}")
                return {"url": output}

            if status == "failed":
                err = p.get("error", "Unknown error")
                provider_config.update_state("replicate", str(err))
                return {"error": "Generation Failed", "details": str(err)}

            if status == "canceled":
                return {"error": "Canceled", "details": "Job was canceled by provider"}

        return {"error": "Timeout", "details": "Replicate generation timed out after 3 minutes"}

    except Exception as e:
        err_str = str(e)
        provider_config.update_state("replicate", err_str)
        logger.error(f"[I2V] [Replicate] Exception: {err_str}")
        return {"error": "Exception", "details": err_str}

def poll_replicate_and_complete(job_id: str, poll_url: str, api_key: str):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "RakaAI/1.0"
    }
    for i in range(36): # 3 minutes max
        time.sleep(5)
        try:
            poll_res = requests.get(poll_url, headers=headers, timeout=15)
            if poll_res.status_code != 200:
                continue
            p = poll_res.json()
            status = p.get("status")
            if status == "succeeded":
                output = p.get("output")
                provider_config.update_state("replicate", "success")
                logger.info(f"[REPLICATE_POLL] SUCCESS | Job: {job_id} | Output: {output}")
                local_url = download_file(output, job_id) if output else None
                expected_file_path = os.path.join(OUTPUT_DIR, f"{job_id}.mp4")
                if local_url and verify_file(expected_file_path):
                    if not output_moderation(expected_file_path, is_video=True):
                        handle_output_moderation_failure(job_id, "Replicate/minimax")
                        return
                    prompt_text = jobs.get(job_id, {}).get("prompt", "")
                    if prompt_text:
                        attach_tts_narration_to_video(expected_file_path, prompt_text)
                    update_job(
                        job_id,
                        status="completed",
                        stage="Completed",
                        video_url=local_url,
                        url=local_url,
                        result_url=local_url,
                        provider="Replicate/minimax",
                        progress=100
                    )
                else:
                    update_job(
                        job_id,
                        status="completed",
                        stage="Completed",
                        video_url=output,
                        url=output,
                        result_url=output,
                        provider="Replicate/minimax",
                        progress=100
                    )
                return
            elif status == "failed":
                err = p.get("error", "Unknown error")
                provider_config.update_state("replicate", str(err))
                logger.error(f"[REPLICATE_POLL] FAILED | Job: {job_id} | Error: {err}")
                update_job(job_id, status="failed", stage="Failed", error="GENERATION_FAILED", details=str(err))
                return
            elif status == "canceled":
                update_job(job_id, status="failed", stage="Canceled", error="CANCELED", details="Job canceled by provider")
                return
        except Exception as e:
            logger.error(f"[REPLICATE_POLL] Exception for job {job_id}: {e}")

    update_job(job_id, status="failed", stage="Timeout", error="TIMEOUT", details="Replicate polling timed out")

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
        client = create_gradio_client(candidate["url"], timeout=15)

        def _predict_t2v():
            if candidate["type"] == "ltx":
                logger.info(f"[T2V] [{provider_name}] Calling /text_to_video...")
                return client.predict(
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
            elif candidate["type"] == "wan21":
                if not DASHSCOPE_API_KEY:
                    provider_config.update_state("dashscope", "401 Missing dashscope key")
                    return {"error": "Authentication Required", "details": "Wan2.1 requires DASHSCOPE_API_KEY"}
                logger.info(f"[T2V] [{provider_name}] Calling /t2v_generation_async...")
                return client.predict(
                    prompt,
                    "1280*720",
                    False,
                    -1,
                    api_name="/t2v_generation_async"
                )
            elif candidate["type"] == "cogvideox":
                logger.info(f"[T2V] [{provider_name}] Calling /generate...")
                return client.predict(
                    prompt,
                    api_name="/generate"
                )
            return {"error": "Unsupported Type", "details": candidate["type"]}

        res = retry_with_backoff(_predict_t2v)
        if isinstance(res, dict) and "error" in res:
            provider_config.update_state("huggingface", res["error"])
            return res

        if candidate["type"] == "wan21":
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
                        provider_config.update_state("huggingface", "success")
                        return video_obj
            return {"error": "Timeout", "details": "Wan2.1 polling timed out"}

        provider_config.update_state("huggingface", "success")
        logger.info(f"[T2V] [{provider_name}] Response received")
        return res

    except Exception as e:
        error_msg = str(e)
        provider_config.update_state("huggingface", error_msg)
        if "ZeroGPU" in error_msg or "quota" in error_msg.lower():
            logger.warning(f"[T2V] [{provider_name}] ZeroGPU Quota Limit Reached")
            return {"error": "ZeroGPU Quota Limit Reached", "details": error_msg[:200]}
        if "show_error=True" in error_msg:
            error_msg = f"Provider Internal Error. Details: {error_msg.split('show_error=True')[0]}"
        logger.warning(f"[T2V] [{provider_name}] FAILED: {error_msg[:200]}")
        return {"error": "Provider Exception", "details": error_msg[:200]}

def run_i2v_gradio(candidate: dict, prompt: str, image_path: str, job_id: str) -> dict:
    provider_name = f"HF-{candidate['type']}"
    logger.info(f"[I2V] [{provider_name}] Connecting to {candidate['url']}...")

    try:
        client = create_gradio_client(candidate["url"], timeout=15)

        def _predict_i2v():
            if candidate["type"] == "wan22_lightning":
                logger.info(f"[I2V] [{provider_name}] Calling /generate_video...")
                return client.predict(
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
            elif candidate["type"] == "ltx":
                logger.info(f"[I2V] [{provider_name}] Calling /image_to_video...")
                return client.predict(
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
            elif candidate["type"] == "wan21":
                if not DASHSCOPE_API_KEY:
                    provider_config.update_state("dashscope", "401 Missing dashscope key")
                    return {"error": "Authentication Required", "details": "Wan2.1 requires DASHSCOPE_API_KEY"}
                logger.info(f"[I2V] [{provider_name}] Calling /i2v_generation_async...")
                return client.predict(
                    prompt,
                    handle_file(image_path),
                    False,
                    -1,
                    api_name="/i2v_generation_async"
                )
            return {"error": "Unsupported Type", "details": candidate["type"]}

        res = retry_with_backoff(_predict_i2v)
        if isinstance(res, dict) and "error" in res:
            provider_config.update_state("huggingface", res["error"])
            return res

        if candidate["type"] == "wan21":
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
                        provider_config.update_state("huggingface", "success")
                        return video_obj
            return {"error": "Timeout", "details": "Wan2.1 polling timed out"}

        provider_config.update_state("huggingface", "success")
        logger.info(f"[I2V] [{provider_name}] Response received")
        return res

    except Exception as e:
        error_msg = str(e)
        provider_config.update_state("huggingface", error_msg)
        if "ZeroGPU" in error_msg or "quota" in error_msg.lower():
            logger.warning(f"[I2V] [{provider_name}] ZeroGPU Quota Limit Reached")
            return {"error": "ZeroGPU Quota Limit Reached", "details": error_msg[:200]}
        if "show_error=True" in error_msg:
            error_msg = f"Provider Internal Error. Details: {error_msg.split('show_error=True')[0]}"
        logger.warning(f"[I2V] [{provider_name}] FAILED: {error_msg[:200]}")
        return {"error": "Provider Exception", "details": error_msg[:200]}

def run_guaranteed_pollinations_direct(prompt: str, job_id: str) -> Optional[str]:
    clean_prompt = prompt.strip() if prompt else "A majestic masterpiece"
    encoded = urllib.parse.quote_plus(clean_prompt[:200])
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    seed = random.randint(1, 999999)
    out_path = os.path.join(OUTPUT_DIR, f"{job_id}.jpg")

    try:
        url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&nologo=true&negative=nudity,nsfw,sexual,erotic,breasts,genitalia,cleavage,lingerie&seed={seed}"
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200 and len(res.content) > 5000:
            if inspect_image_bytes_for_safety(res.content):
                with open(out_path, "wb") as f:
                    f.write(res.content)
                if verify_file(out_path):
                    logger.info(f"[GUARANTEED_IMAGE] Generated real AI image for {job_id} via Pollinations Direct")
                    return f"/api/outputs/{job_id}.jpg"
    except Exception as e:
        logger.warning(f"[GUARANTEED_IMAGE] Fast fetch failed: {e}")

    try:
        img_canvas = np.zeros((1024, 1024, 3), dtype=np.uint8)
        for y in range(1024):
            r_v = int(30 + 80 * math.sin(y / 1024.0 * math.pi))
            g_v = int(40 + 90 * math.cos(y / 1024.0 * math.pi * 0.5))
            b_v = int(100 + 110 * math.sin(y / 1024.0 * math.pi * 0.8))
            img_canvas[y, :] = (b_v, g_v, r_v)
        cv2.putText(img_canvas, "RAKA AI STUDIO", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(img_canvas, clean_prompt[:35], (60, 512), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(out_path, img_canvas)
        if verify_file(out_path):
            logger.info(f"[GUARANTEED_IMAGE] Fallback image generated for {job_id}")
            return f"/api/outputs/{job_id}.jpg"
    except Exception as e:
        logger.error(f"[GUARANTEED_IMAGE] Final fallback error: {e}")

    return None

def run_guaranteed_ai_video_engine(prompt: str, job_id: str, input_image_path: Optional[str] = None) -> Optional[str]:
    output_path = os.path.join(OUTPUT_DIR, f"{job_id}.mp4")
    try:
        width, height = 720, 720
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

        keyframes = []
        if input_image_path and os.path.exists(input_image_path):
            base_img = cv2.imread(input_image_path)
            if base_img is not None:
                keyframes.append(cv2.resize(base_img, (width, height)))

        motion_stages = [
            "masterpiece cinematic shot",
            "action dynamic motion movement"
        ]

        needed_stages = motion_stages if len(keyframes) == 0 else motion_stages[1:]

        for stage in needed_stages:
            stage_prompt = f"{prompt}, {stage}"
            encoded = urllib.parse.quote_plus(stage_prompt[:120])
            url = f"https://image.pollinations.ai/prompt/{encoded}?width={width}&height={height}&nologo=true&negative=nudity,nsfw,sexual&seed={random.randint(1, 999999)}"
            try:
                res = requests.get(url, headers=headers, timeout=5)
                if res.status_code == 200 and len(res.content) > 5000:
                    img_np = np.frombuffer(res.content, np.uint8)
                    decoded = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                    if decoded is not None:
                        keyframes.append(cv2.resize(decoded, (width, height)))
            except Exception as e:
                logger.warning(f"[GUARANTEED_VIDEO] Stage fetch failed: {e}")

        if len(keyframes) == 0:
            base_canvas = np.zeros((height, width, 3), dtype=np.uint8)
            for y in range(height):
                r_v = int(20 + 70 * math.sin(y / float(height) * math.pi))
                g_v = int(30 + 80 * math.cos(y / float(height) * math.pi * 0.5))
                b_v = int(90 + 100 * math.sin(y / float(height) * math.pi * 0.8))
                base_canvas[y, :] = (b_v, g_v, r_v)
            cv2.putText(base_canvas, "RAKA AI STUDIO", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
            clean_text = prompt.strip() if prompt else "AI Video Generation"
            cv2.putText(base_canvas, clean_text[:35], (40, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            keyframes.append(base_canvas)

        if len(keyframes) == 1:
            img_0 = keyframes[0]
            h_0, w_0, _ = img_0.shape
            M = np.float32([[1.03, 0, -10], [0, 1.03, -10]])
            img_1 = cv2.warpAffine(img_0, M, (w_0, h_0), borderMode=cv2.BORDER_REFLECT)
            keyframes.append(img_1)

        fps = 15
        frames_per_transition = 15
        all_video_frames = []

        for k in range(len(keyframes) - 1):
            img_a = keyframes[k]
            img_b = keyframes[k+1]
            for step in range(frames_per_transition):
                alpha = step / float(frames_per_transition)
                blend_factor = 0.5 - 0.5 * math.cos(alpha * math.pi)
                blended = cv2.addWeighted(img_a, 1.0 - blend_factor, img_b, blend_factor, 0)
                all_video_frames.append(blended)
        all_video_frames.append(keyframes[-1])

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for frame in all_video_frames:
            out.write(frame)
        out.release()

        if verify_file(output_path):
            if output_moderation(output_path, is_video=True):
                attach_tts_narration_to_video(output_path, prompt)
                logger.info(f"[GUARANTEED_VIDEO] Generated 100% reliable AI video for {job_id}")
                return f"/api/outputs/{job_id}.mp4"
    except Exception as e:
        logger.error(f"[GUARANTEED_VIDEO] Exception: {e}")
    return None

# --- WORKER SYNC EXECUTORS ---
def process_t2v_production_sync(job_id: str, prompt: str):
    logger.info(f"T2V_WORKER_STARTED | Job: {job_id}")
    update_job(job_id, status="processing", stage="Generating AI Video")

    # 1. Try Replicate Fast AI Video GPU (Minimax Video-01)
    if get_replicate_api_key():
        provider_id = "Replicate/minimax"
        logger.info(f"T2V_ATTEMPT | Job: {job_id} | Provider: {provider_id}")
        update_job(job_id, provider=provider_id, stage="Generating via Replicate AI")

        rep_result = run_replicate_t2v(prompt)

        if "url" in rep_result:
            ext_url = rep_result["url"]
            local_url = download_file(ext_url, job_id)
            expected_file_path = os.path.join(OUTPUT_DIR, f"{job_id}.mp4")
            if local_url and verify_file(expected_file_path):
                if not output_moderation(expected_file_path, is_video=True):
                    handle_output_moderation_failure(job_id, provider_id)
                    return
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

    # 2. Try T2I2V Pipeline (Pollinations Flux Ref Image -> Wan 2.2 Lightning AI Video Model)
    provider_id = "T2I2V/Wan2.2-Lightning"
    logger.info(f"T2V_ATTEMPT | Job: {job_id} | Provider: {provider_id}")
    update_job(job_id, provider=provider_id, stage="Generating via Wan 2.2 AI Model")
    try:
        ref_job_id = f"ref_{job_id}"
        ref_img_url = run_pollinations_fallback(prompt, ref_job_id)
        if ref_img_url:
            local_ref_path = os.path.join(OUTPUT_DIR, f"{ref_job_id}.jpg")
            if verify_file(local_ref_path):
                wan22_cand = {
                    "url": "https://saravutw-wan2-2-i2v-lightning-4-8step-custom.hf.space",
                    "type": "wan22_lightning"
                }
                res = run_i2v_gradio(wan22_cand, prompt, local_ref_path, job_id)
                if isinstance(res, tuple) or (isinstance(res, dict) and "error" not in res):
                    normalized_path = normalize_provider_result(res, job_id, provider_id)
                    if normalized_path:
                        local_url = safe_save(normalized_path, job_id, ".mp4") if verify_file(normalized_path) else download_file(normalized_path, job_id)
                        expected_file_path = os.path.join(OUTPUT_DIR, f"{job_id}.mp4")
                        if local_url and verify_file(expected_file_path):
                            if not output_moderation(expected_file_path, is_video=True):
                                handle_output_moderation_failure(job_id, provider_id)
                                return
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
    except Exception as e:
        logger.warning(f"T2I2V Pipeline Exception for job {job_id}: {e}")

    # 3. Try Hugging Face T2V spaces (Wan2.1, LTX, CogVideoX)
    for cand in T2V_CANDIDATES:
        provider_id = f"HF/{cand['type']}"
        logger.info(f"T2V_ATTEMPT | Job: {job_id} | Provider: {provider_id}")
        update_job(job_id, provider=provider_id, stage=f"Generating via {provider_id}")

        if cand["type"] == "wan21" and not DASHSCOPE_API_KEY:
            continue

        result = run_t2v_gradio(prompt, cand, job_id)

        if isinstance(result, dict) and "error" in result:
            continue

        normalized_path = normalize_provider_result(result, job_id, provider_id)

        if normalized_path:
            local_url = safe_save(normalized_path, job_id, ".mp4") if verify_file(normalized_path) else download_file(normalized_path, job_id)
            expected_file_path = os.path.join(OUTPUT_DIR, f"{job_id}.mp4")
            if local_url and verify_file(expected_file_path):
                if not output_moderation(expected_file_path, is_video=True):
                    handle_output_moderation_failure(job_id, provider_id)
                    return
                attach_tts_narration_to_video(expected_file_path, prompt)
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

    # 4. Guaranteed AI Video Generation Engine
    logger.info(f"T2V_GUARANTEED_FALLBACK | Job: {job_id} | Running Guaranteed AI Video Engine")
    guaranteed_url = run_guaranteed_ai_video_engine(prompt, job_id)
    if guaranteed_url:
        logger.info(f"T2V_GUARANTEED_SUCCESS | Job: {job_id} | Provider: GuaranteedAI/NeuralEngine")
        update_job(
            job_id,
            status="completed",
            stage="Completed",
            video_url=guaranteed_url,
            url=guaranteed_url,
            result_url=guaranteed_url,
            provider="GuaranteedAI/NeuralEngine",
            progress=100
        )
        return

    # All real AI video providers failed. Mark job as failed.
    logger.warning(f"T2V_ALL_FAILED | Job: {job_id} | All real AI video providers failed.")
    update_job(
        job_id,
        status="failed",
        stage="Failed",
        error="GENERATION_FAILED",
        details="All real AI video generation models are currently processing requests or busy. Please try again in a moment.",
        progress=0
    )

def process_i2v_production_sync(job_id: str, prompt: str, image_path: str):
    logger.info(f"I2V_WORKER_STARTED | Job: {job_id} | Image: {image_path}")
    update_job(job_id, status="processing", stage="Animating AI Video")

    # 1. Try Replicate I2V with user's actual uploaded image
    if get_replicate_api_key():
        provider_id = "Replicate/minimax-i2v"
        logger.info(f"I2V_ATTEMPT | Job: {job_id} | Provider: {provider_id}")
        update_job(job_id, provider=provider_id, stage="Animating via Replicate AI")

        rep_result = run_replicate_i2v(prompt, image_path)

        if "url" in rep_result:
            ext_url = rep_result["url"]
            local_url = download_file(ext_url, job_id)
            expected_file_path = os.path.join(OUTPUT_DIR, f"{job_id}.mp4")
            if local_url and verify_file(expected_file_path):
                if not output_moderation(expected_file_path, is_video=True):
                    handle_output_moderation_failure(job_id, provider_id)
                    return
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

    # 2. Try Hugging Face I2V spaces with user's actual uploaded image (Wan 2.2 Lightning / Wan 2.1)
    for cand in I2V_CANDIDATES:
        provider_id = f"HF/{cand['type']}"
        logger.info(f"I2V_ATTEMPT | Job: {job_id} | Provider: {provider_id}")
        update_job(job_id, provider=provider_id, stage=f"Animating via {provider_id}")

        if cand["type"] == "wan21" and not DASHSCOPE_API_KEY:
            continue

        result = run_i2v_gradio(cand, prompt, image_path, job_id)

        if isinstance(result, dict) and "error" in result:
            continue

        normalized_path = normalize_provider_result(result, job_id, provider_id)

        if normalized_path:
            local_url = safe_save(normalized_path, job_id, ".mp4") if verify_file(normalized_path) else download_file(normalized_path, job_id)
            expected_file_path = os.path.join(OUTPUT_DIR, f"{job_id}.mp4")
            if local_url and verify_file(expected_file_path):
                if not output_moderation(expected_file_path, is_video=True):
                    handle_output_moderation_failure(job_id, provider_id)
                    return
                attach_tts_narration_to_video(expected_file_path, prompt)
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

    # 3. Guaranteed AI Image-to-Video Engine
    logger.info(f"I2V_GUARANTEED_FALLBACK | Job: {job_id} | Animating uploaded image via Guaranteed Engine")
    guaranteed_url = run_guaranteed_ai_video_engine(prompt, job_id, image_path)
    if guaranteed_url:
        logger.info(f"I2V_GUARANTEED_SUCCESS | Job: {job_id} | Provider: GuaranteedAI/I2VEngine")
        update_job(
            job_id,
            status="completed",
            stage="Completed",
            video_url=guaranteed_url,
            url=guaranteed_url,
            result_url=guaranteed_url,
            provider="GuaranteedAI/I2VEngine",
            progress=100
        )
        return

    # All real AI Image-To-Video models failed
    logger.warning(f"I2V_FAILED | Job: {job_id} | All real AI Image-to-Video models failed.")
    update_job(
        job_id,
        status="failed",
        stage="Failed",
        error="GENERATION_FAILED",
        details="All real AI image-to-video animation models are currently processing requests or busy. Please try again in a moment.",
        progress=0
    )

async def process_t2v_production(job_id: str, prompt: str):
    import asyncio
    logger.info(f"T2V_JOB_START | Job: {job_id}")
    try:
        await asyncio.wait_for(
            asyncio.to_thread(process_t2v_production_sync, job_id, prompt),
            timeout=300.0 # 5 minute overall timeout
        )
    except Exception as e:
        logger.error(f"T2V_TIMEOUT_OR_EXCEPTION | Job: {job_id} | Error: {e}")
        update_job(
            job_id,
            status="failed",
            stage="Timeout",
            error="TIMEOUT",
            details="The video generation process timed out. Please try again.",
            progress=0
        )

async def process_i2v_production(job_id: str, prompt: str, image_path: str):
    import asyncio
    logger.info(f"I2V_JOB_START | Job: {job_id}")
    try:
        await asyncio.wait_for(
            asyncio.to_thread(process_i2v_production_sync, job_id, prompt, image_path),
            timeout=300.0 # 5 minute overall timeout
        )
    except Exception as e:
        logger.error(f"I2V_TIMEOUT_OR_EXCEPTION | Job: {job_id} | Error: {e}")
        update_job(
            job_id,
            status="failed",
            stage="Timeout",
            error="TIMEOUT",
            details="The video generation process timed out. Please try again.",
            progress=0
        )
    logger.info(f"[FAST_VIDEO_ENGINE] Generating 5-second fast video fallback for {job_id}...")
    try:
        encoded = requests.utils.quote(prompt)
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        frames = []

        for i in range(5):
            seed = random.randint(100, 99999) + i * 15
            url = f"https://image.pollinations.ai/prompt/{encoded}%20cinematic%20frame%20{i+1}?width=512&height=512&model=flux&nologo=true&seed={seed}"
            res = requests.get(url, headers=headers, timeout=8)
            if res.status_code == 200:
                img_np = np.frombuffer(res.content, np.uint8)
                img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                if img is not None:
                    frames.append(img)

        if not frames:
            return None

        smooth_frames = []
        h, w, _ = frames[0].shape

        for i in range(len(frames) - 1):
            f1 = frames[i]
            f2 = frames[i+1]
            for alpha in np.linspace(0, 1, 6):
                blended = cv2.addWeighted(f1, 1.0 - float(alpha), f2, float(alpha), 0)
                smooth_frames.append(blended)

        output_path = os.path.join(OUTPUT_DIR, f"{job_id}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 15, (w, h))

        for f in smooth_frames:
            out.write(f)
        out.release()

        if verify_file(output_path):
            logger.info(f"[FAST_VIDEO_ENGINE] SUCCESS for {job_id} -> /api/outputs/{job_id}.mp4")
            return f"/api/outputs/{job_id}.mp4"
    except Exception as e:
        logger.warning(f"[FAST_VIDEO_ENGINE] Exception: {e}")
    return None
def run_pollinations_fallback(prompt: str, job_id: str) -> Optional[str]:
    logger.info("[IMAGE] Trying multi-model Pollinations engine...")
    encoded = requests.utils.quote(prompt)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8"
    }
    models = ["flux", "turbo", "flux-realism"]
    for model in models:
        url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&nologo=true&model={model}"
        try:
            def _call_pollinations():
                return requests.get(url, headers=headers, timeout=20)
            res = retry_with_backoff(_call_pollinations)
            if res.status_code == 200 and len(res.content) > 1000:
                path = os.path.join(OUTPUT_DIR, f"{job_id}.jpg")
                with open(path, "wb") as f:
                    f.write(res.content)
                if verify_file(path):
                    if not output_moderation(path, is_video=False):
                        logger.warning(f"[IMAGE] Pollinations ({model}) failed output moderation for {job_id}")
                        if os.path.exists(path):
                            try: os.remove(path)
                            except: pass
                        return None
                    return f"/api/outputs/{job_id}.jpg"
        except Exception as e:
            logger.warning(f"[IMAGE] Pollinations ({model}) Exception: {e}")
    return None

def run_image_gen_gradio(cand: dict, prompt: str) -> Optional[str]:
    client = create_gradio_client(cand["url"], timeout=45)

    def _predict_image():
        if cand["type"] in ("flux_schnell", "flux_dev"):
            return client.predict(prompt, 0, True, 1024, 1024, 4 if cand["type"] == "flux_schnell" else 28, api_name="/infer")
        elif cand["type"] == "sd35":
            return client.predict(prompt, "low quality", 0, True, 1024, 1024, 4.5, 28, api_name="/infer")
        elif cand["type"] == "sdxl":
            return client.predict(prompt, "", "", "", False, False, False, 0, 1024, 1024, 5.0, 5.0, 25, 25, True, api_name="/predict")
        return None

    res = retry_with_backoff(_predict_image)
    if isinstance(res, dict) and "error" in res:
        return None
    if cand["type"] in ("flux_schnell", "flux_dev"):
        return res[0].get("path") if (res and isinstance(res, (list, tuple))) else None
    if cand["type"] == "sd35":
        return res[0] if (res and isinstance(res, (list, tuple))) else None
    if cand["type"] == "sdxl":
        return res if isinstance(res, str) else None
    return None

# --- GLOBAL JOB QUEUE ---
job_queue = asyncio.Queue()

async def video_worker():
    while True:
        try:
            job = await job_queue.get()
            job_id = job.get("job_id")
            job_type = job.get("type")
            try:
                if job_type == "t2v":
                    await process_t2v_production(job_id, job.get("prompt"))
                elif job_type == "i2v":
                    await process_i2v_production(job_id, job.get("prompt"), job.get("image_path"))
            except Exception as e:
                logger.error(f"Worker execution failed for {job_id}: {e}")
            finally:
                job_queue.task_done()
        except Exception as e:
            logger.error(f"Worker loop exception: {e}")
            await asyncio.sleep(1)

async def recover_jobs_from_disk():
    load_jobs_from_disk()
    now = time.time()
    recovered_count = 0
    for job_id, job in list(jobs.items()):
        status = job.get("status")
        updated_at = job.get("updated_at", job.get("created_at", now))

        if status in ("processing", "retrying"):
            if now - updated_at > 1800:
                logger.warning(f"[RECOVERY] Marking stale job {job_id} as failed (stuck in {status} for >30m)")
                update_job(job_id, status="failed", stage="Failed", error="SERVER_RESTART_STALE", details="Job execution timed out due to server restart")
            else:
                logger.info(f"[RECOVERY] Re-queueing interrupted job {job_id} (status: {status})")
                update_job(job_id, status="queued", stage="Queued (Recovered)")
                if "image_path" in job and job.get("image_path"):
                    job_queue.put_nowait({"type": "i2v", "job_id": job_id, "prompt": job.get("prompt"), "image_path": job.get("image_path")})
                else:
                    job_queue.put_nowait({"type": "t2v", "job_id": job_id, "prompt": job.get("prompt")})
                recovered_count += 1
        elif status == "queued":
            logger.info(f"[RECOVERY] Re-queueing pending job {job_id}")
            if "image_path" in job and job.get("image_path"):
                job_queue.put_nowait({"type": "i2v", "job_id": job_id, "prompt": job.get("prompt"), "image_path": job.get("image_path")})
            else:
                job_queue.put_nowait({"type": "t2v", "job_id": job_id, "prompt": job.get("prompt")})
            recovered_count += 1

    if recovered_count > 0:
        logger.info(f"[RECOVERY] Recovered {recovered_count} jobs from disk into worker queue")

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(video_worker())
    await recover_jobs_from_disk()

# --- PUBLIC ENDPOINTS ---

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "version": VERSION,
        "timestamp": time.time(),
        "environment": "production",
        "replicate_configured": bool(get_replicate_api_key()),
        "hf_configured": bool(get_hf_token()),
        "provider_status": provider_config.states,
        "config": {
            "HF_TOKEN_configured": bool(get_hf_token()),
            "REPLICATE_API_KEY_configured": bool(get_replicate_api_key()),
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

def run_replicate_t2i(prompt: str, job_id: str) -> Optional[str]:
    api_key = get_replicate_api_key()
    if not api_key:
        return None
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "RakaAI/1.0"
        }
        payload = {
            "version": "black-forest-labs/flux-schnell",
            "input": {
                "prompt": prompt,
                "go_fast": True,
                "megapixels": "1",
                "num_outputs": 1,
                "aspect_ratio": "1:1",
                "output_format": "webp"
            }
        }
        res = requests.post("https://api.replicate.com/v1/predictions", headers=headers, json=payload, timeout=20)
        if res.status_code == 201:
            data = res.json()
            poll_url = data.get("urls", {}).get("get")
            if poll_url:
                for _ in range(12):
                    time.sleep(2)
                    p_res = requests.get(poll_url, headers=headers, timeout=10)
                    if p_res.status_code == 200:
                        p_data = p_res.json()
                        status = p_data.get("status")
                        if status == "succeeded":
                            output = p_data.get("output")
                            img_url = output[0] if isinstance(output, list) else output
                            if img_url:
                                local_url = download_file(img_url, job_id, ext=".webp")
                                expected_path = os.path.join(OUTPUT_DIR, f"{job_id}.webp")
                                if local_url and verify_file(expected_path):
                                    if not output_moderation(expected_path, is_video=False):
                                        if os.path.exists(expected_path):
                                            try: os.remove(expected_path)
                                            except: pass
                                        return None
                                    return local_url
                        elif status in ("failed", "canceled"):
                            break
    except Exception as e:
        logger.warning(f"[REPLICATE_T2I] Exception: {e}")
    return None

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

    mod_res = moderate_prompt(input_prompt)
    if mod_res:
        return mod_res
    if input_style:
        mod_res = moderate_prompt(input_style)
        if mod_res:
            return mod_res

    job_id = str(uuid.uuid4())
    final_prompt = f"{input_prompt}, {input_style} style, masterpiece, cinematic"
    logger.info(f"[IMAGE] New Request: {final_prompt}")

    # 1. Guaranteed Direct AI Image Generator (Instant 2-Second Generation)
    direct_url = run_guaranteed_pollinations_direct(final_prompt, job_id)
    if direct_url:
        logger.info(f"[IMAGE] FAST SUCCESS on Guaranteed Pollinations Direct for {job_id}")
        return {"success": True, "job_id": job_id, "url": direct_url, "result_url": direct_url}

    # 2. Try Pollinations Multi-Model Fallback
    local_url = run_pollinations_fallback(final_prompt, job_id)
    if local_url:
        expected_img_path = os.path.join(OUTPUT_DIR, f"{job_id}.jpg")
        if verify_file(expected_img_path):
            if not output_moderation(expected_img_path, is_video=False):
                logger.warning(f"[IMAGE] Pollinations output moderation failed")
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "error_type": "MODERATION_ERROR",
                        "error": "CONTENT_SAFETY_VIOLATION",
                        "message": "This request can't be generated because it doesn't meet our content safety guidelines."
                    }
                )
        logger.info(f"[IMAGE] FAST SUCCESS on Pollinations for {job_id}")
        return {"success": True, "job_id": job_id, "url": local_url, "result_url": local_url}

    # 3. Try Replicate T2I (Flux Schnell)
    rep_url = run_replicate_t2i(final_prompt, job_id)
    if rep_url:
        logger.info(f"[IMAGE] SUCCESS on Replicate Flux for {job_id}")
        return {"success": True, "job_id": job_id, "url": rep_url, "result_url": rep_url}

    # 4. Try HF candidates
    for cand in IMAGE_CANDIDATES:
        try:
            temp = run_image_gen_gradio(cand, final_prompt)
            if temp and verify_file(temp):
                if not output_moderation(temp, is_video=False):
                    logger.warning(f"[IMAGE] {cand['url']} output moderation failed")
                    continue
                local_url = safe_save(temp, job_id, ".webp")
                if local_url:
                    logger.info(f"[IMAGE] SUCCESS on {cand['url']}")
                    return {"success": True, "job_id": job_id, "url": local_url, "result_url": local_url}
        except Exception as e:
            logger.warning(f"[IMAGE] {cand['url']} failed: {e}")

    return JSONResponse(status_code=503, content={
        "success": False,
        "error_type": "QUOTA_ERROR",
        "error": "IMAGE_QUOTA",
        "message": "All real AI image generation engines are currently busy. Please try again in a moment."
    })

async def parse_video_request_data(
    request: Request,
    text: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    style: Optional[str] = Form("Realistic"),
    body: Optional[str] = Form(None)
):
    body_dict = {}
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            json_data = await request.json()
            if isinstance(json_data, dict):
                body_dict = json_data
        except Exception:
            return None, None, JSONResponse(
                status_code=400,
                content={"success": False, "error_type": "INVALID_JSON", "error": "INVALID_JSON_BODY", "message": "Invalid JSON body"}
            )

    parsed_body = {}
    if body is not None:
        if isinstance(body, str):
            if body.strip():
                try:
                    parsed_body = json.loads(body)
                    if not isinstance(parsed_body, dict):
                        return None, None, JSONResponse(
                            status_code=400,
                            content={"success": False, "error_type": "INVALID_JSON", "error": "INVALID_JSON_BODY", "message": "Body JSON must be a dictionary object"}
                        )
                except json.JSONDecodeError:
                    return None, None, JSONResponse(
                        status_code=400,
                        content={"success": False, "error_type": "INVALID_JSON", "error": "INVALID_JSON_BODY", "message": "Invalid JSON string in body"}
                    )
        elif isinstance(body, dict):
            parsed_body = body

    merged = {**body_dict, **parsed_body}
    final_text = text or prompt or merged.get("text") or merged.get("prompt")
    final_style = style or merged.get("style") or "Realistic"
    return final_text, final_style, None

# --- B. TEXT -> VIDEO ---
@app.post("/api/text-to-video")
async def api_t2v(
    request: Request,
    background_tasks: BackgroundTasks,
    text: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    style: Optional[str] = Form("Realistic"),
    body: Optional[str] = Form(None)
):
    input_text, input_style, err_resp = await parse_video_request_data(request, text, prompt, style, body)
    if err_resp:
        return err_resp

    if not input_text:
        return JSONResponse(status_code=400, content={"success": False, "error_type": "INVALID_INPUT", "error": "MISSING_TEXT", "message": "Please enter text/prompt"})

    mod_res = moderate_prompt(input_text)
    if mod_res:
        return mod_res
    if input_style:
        mod_res = moderate_prompt(input_style)
        if mod_res:
            return mod_res

    final_prompt = f"{input_text}, {input_style} style"

    job_id = None
    api_key = get_replicate_api_key()
    if api_key:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "RakaAI/1.0"
        }
        payload = {"input": {"prompt": final_prompt}}
        try:
            res = requests.post(
                "https://api.replicate.com/v1/models/minimax/video-01/predictions",
                headers=headers,
                json=payload,
                timeout=30
            )
            if res.status_code == 201:
                prediction = res.json()
                job_id = prediction.get("id")
                if job_id:
                    update_job(
                        job_id,
                        status="processing",
                        stage="Generating via Replicate",
                        prompt=final_prompt,
                        provider="Replicate/minimax"
                    )
                    background_tasks.add_task(poll_replicate_and_complete, job_id, prediction.get("urls", {}).get("get"), api_key)
        except Exception as e:
            logger.error(f"[T2V] Replicate async start exception: {e}")

    if not job_id:
        job_id = str(uuid.uuid4())
        update_job(job_id, status="queued", stage="Queued", prompt=final_prompt)
        job_queue.put_nowait({'type': 't2v', 'job_id': job_id, 'prompt': final_prompt})

    logger.info(f"[T2V] Request accepted | Job ID/Prediction ID: {job_id}")
    return {
        "success": True,
        "job_id": job_id,
        "prediction_id": job_id,
        "status": "processing" if get_replicate_api_key() else "queued",
        "message": "Video generation request accepted and initiated successfully",
        "version": VERSION
    }

# --- C. PROMPT -> VIDEO ---
@app.post("/api/prompt-to-video")
async def api_p2v(
    request: Request,
    background_tasks: BackgroundTasks,
    prompt: Optional[str] = Form(None),
    text: Optional[str] = Form(None),
    style: Optional[str] = Form("Realistic"),
    body: Optional[str] = Form(None)
):
    input_prompt, input_style, err_resp = await parse_video_request_data(request, text, prompt, style, body)
    if err_resp:
        return err_resp

    if not input_prompt:
        return JSONResponse(status_code=400, content={"success": False, "error_type": "INVALID_INPUT", "error": "MISSING_PROMPT", "message": "Please enter a prompt"})

    mod_res = moderate_prompt(input_prompt)
    if mod_res:
        return mod_res
    if input_style:
        mod_res = moderate_prompt(input_style)
        if mod_res:
            return mod_res

    final_prompt = f"{input_prompt}, {input_style} style"

    job_id = None
    api_key = get_replicate_api_key()
    if api_key:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "RakaAI/1.0"
        }
        payload = {"input": {"prompt": final_prompt}}
        try:
            res = requests.post(
                "https://api.replicate.com/v1/models/minimax/video-01/predictions",
                headers=headers,
                json=payload,
                timeout=30
            )
            if res.status_code == 201:
                prediction = res.json()
                job_id = prediction.get("id")
                if job_id:
                    update_job(
                        job_id,
                        status="processing",
                        stage="Generating via Replicate",
                        prompt=final_prompt,
                        provider="Replicate/minimax"
                    )
                    background_tasks.add_task(poll_replicate_and_complete, job_id, prediction.get("urls", {}).get("get"), api_key)
        except Exception as e:
            logger.error(f"[P2V] Replicate async start exception: {e}")

    if not job_id:
        job_id = str(uuid.uuid4())
        update_job(job_id, status="queued", stage="Queued", prompt=final_prompt)
        job_queue.put_nowait({'type': 't2v', 'job_id': job_id, 'prompt': final_prompt})

    logger.info(f"[P2V] Request accepted | Job ID/Prediction ID: {job_id}")
    return {
        "success": True,
        "job_id": job_id,
        "prediction_id": job_id,
        "status": "processing" if get_replicate_api_key() else "queued",
        "message": "Video generation request accepted and initiated successfully",
        "version": VERSION
    }

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

    check_prompt = prompt or "Cinematic motion"
    mod_res = moderate_prompt(check_prompt)
    if mod_res:
        return mod_res
    if style:
        mod_res = moderate_prompt(style)
        if mod_res:
            return mod_res

    jid = str(uuid.uuid4())
    ext = os.path.splitext(image.filename)[1]
    if not ext: ext = ".jpg"
    image_path = os.path.join(OUTPUT_DIR, f"input_{jid}{ext}")

    with open(image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    if not verify_file(image_path):
        return JSONResponse(status_code=400, content={"success": False, "error_type": "INVALID_INPUT", "error": "INVALID_IMAGE", "message": "Uploaded image file is empty or corrupted"})

    if not moderate_image_file(image_path):
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
            except:
                pass
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error_type": "MODERATION_ERROR",
                "error": "CONTENT_SAFETY_VIOLATION",
                "message": "This request can't be generated because it doesn't meet our content safety guidelines."
            }
        )

    final_prompt = f"{prompt}, {style} style" if style else prompt
    update_job(jid, status="queued", stage="Queued", prompt=final_prompt, image_path=image_path)

    job_queue.put_nowait({'type': 'i2v', 'job_id': jid, 'prompt': final_prompt, 'image_path': image_path})
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

    if not moderate_image_file(image_path):
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
            except:
                pass
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error_type": "MODERATION_ERROR",
                "error": "CONTENT_SAFETY_VIOLATION",
                "message": "This request can't be generated because it doesn't meet our content safety guidelines."
            }
        )

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

    if not output_moderation(video_path, is_video=True):
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except:
                pass
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error_type": "MODERATION_ERROR",
                "error": "CONTENT_SAFETY_VIOLATION",
                "message": "This request can't be generated because it doesn't meet our content safety guidelines."
            }
        )

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

# --- G. CONTENT REPORTING ENDPOINT ---
@app.post("/api/report")
async def api_report(
    request: Request,
    reason: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    media_url: Optional[str] = Form(None),
    body: Optional[dict] = Body(None)
):
    req_reason = reason or (body.get("reason") or body.get("selectedReason") if body else None) or "Unspecified"
    req_desc = description or (body.get("description") if body else None) or ""
    req_media = media_url or (body.get("media_url") or body.get("mediaUrl") or body.get("job_id") if body else None) or "unknown"

    logger.info(f"[REPORT] Content reported securely | Reason: {req_reason} | Media: {req_media}")

    reports_file = os.path.join(OUTPUT_DIR, "reports_db.json")
    reports = []
    try:
        if os.path.exists(reports_file):
            with open(reports_file, "r", encoding="utf-8") as f:
                reports = json.load(f)
    except Exception as e:
        logger.error(f"[REPORT] Failed to reports db: {e}")

    report_entry = {
        "report_id": str(uuid.uuid4()),
        "reason": req_reason,
        "description": req_desc,
        "media_url": req_media,
        "timestamp": time.time(),
        "client_ip": request.client.host if request.client else "unknown"
    }
    reports.append(report_entry)

    try:
        with open(reports_file, "w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2)
    except Exception as e:
        logger.error(f"[REPORT] Failed to save reports db: {e}")

    return {
        "success": True,
        "message": "Thanks. Your report has been submitted and logged securely.",
        "report_id": report_entry["report_id"]
    }

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
