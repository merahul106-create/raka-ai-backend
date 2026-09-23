import os
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
from gradio_client import Client


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("raka_ai")


# =========================================================
# ENVIRONMENT
# =========================================================

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY", "").strip()


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="Raka AI Production API",
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)


# =========================================================
# MODELS
# =========================================================

class PromptToImageRequest(BaseModel):
    prompt: str
    style: str = "Realistic"


# =========================================================
# HEALTH
# =========================================================

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "Raka AI Production API",
    }


@app.get("/api/")
def root():
    return {
        "status": "ok",
        "service": "Raka AI Production API",
        "version": "1.0.0",
    }


# =========================================================
# IMAGE RESULT PARSER
# =========================================================

def extract_image_path(result):
    """
    Safely extract an image path or URL from Gradio responses.
    """

    if result is None:
        return None

    # -----------------------------------------------------
    # String
    # -----------------------------------------------------

    if isinstance(result, str):

        value = result.strip()

        if not value:
            return None

        if value.startswith(("http://", "https://")):
            return value

        if os.path.isfile(value):
            return value

        return None

    # -----------------------------------------------------
    # pathlib / os.PathLike
    # -----------------------------------------------------

    if isinstance(result, Path) or hasattr(
        result,
        "__fspath__",
    ):

        try:

            path = os.fspath(result)

            if path and os.path.isfile(path):
                return path

        except Exception as e:

            logger.warning(
                f"[IMAGE] Path parsing failed: {e}"
            )

        return None

    # -----------------------------------------------------
    # Dictionary
    # -----------------------------------------------------

    if isinstance(result, dict):

        keys = (
            "path",
            "url",
            "image",
            "value",
            "file",
            "name",
            "filepath",
            "file_path",
            "data",
        )

        for key in keys:

            if key not in result:
                continue

            try:

                found = extract_image_path(
                    result[key]
                )

                if found:
                    return found

            except Exception as e:

                logger.warning(
                    f"[IMAGE] Failed parsing key "
                    f"{key}: {e}"
                )

        return None

    # -----------------------------------------------------
    # List / Tuple
    # -----------------------------------------------------

    if isinstance(
        result,
        (list, tuple),
    ):

        for item in result:

            try:

                found = extract_image_path(
                    item
                )

                if found:
                    return found

            except Exception as e:

                logger.warning(
                    f"[IMAGE] Failed parsing "
                    f"list item: {e}"
                )

        return None

    # -----------------------------------------------------
    # Object attributes
    # -----------------------------------------------------

    for attr in (
        "path",
        "url",
        "filepath",
        "file_path",
        "name",
    ):

        try:

            value = getattr(
                result,
                attr,
                None,
            )

            if value:

                found = extract_image_path(
                    value
                )

                if found:
                    return found

        except Exception:
            pass

    return None


# =========================================================
# GRADIO CLIENT
# =========================================================

def create_gradio_client(space: str):

    if HF_TOKEN:

        logger.info(
            "[HF] Using authenticated Hugging Face client"
        )

        return Client(
            space,
            token=HF_TOKEN,
        )

    logger.warning(
        "[HF] HF_TOKEN is empty. "
        "Trying public Space."
    )

    return Client(space)


# =========================================================
# IMAGE GENERATION
# =========================================================

def run_image_gen(
    cand,
    prompt: str,
) -> Optional[str]:

    provider = cand.get("type")
    space = cand.get("url")

    if not space:

        logger.error(
            "[IMAGE] Provider URL is missing"
        )

        return None

    if not prompt or not prompt.strip():

        logger.error(
            "[IMAGE] Empty prompt"
        )

        return None

    clean_prompt = prompt.strip()

    logger.info(
        f"[IMAGE] Starting provider={provider} "
        f"space={space}"
    )

    try:

        client = create_gradio_client(space)

        # =================================================
        # FLUX SCHNELL / FLUX DEV
        # =================================================

        if provider in (
            "flux_schnell",
            "flux_dev",
        ):

            steps = (
                4
                if provider == "flux_schnell"
                else 28
            )

            logger.info(
                f"[IMAGE] Running {provider} "
                f"steps={steps}"
            )

            result = client.predict(
                clean_prompt,
                0,
                True,
                1024,
                1024,
                steps,
                api_name="/infer",
            )

            logger.info(
                f"[IMAGE] result type={type(result)}"
            )

            logger.info(
                f"[IMAGE] raw result="
                f"{str(result)[:1500]}"
            )

            image_path = extract_image_path(
                result
            )

            if image_path:

                logger.info(
                    f"[IMAGE] SUCCESS: {image_path}"
                )

                return image_path

            logger.error(
                "[IMAGE] Could not extract image"
            )

            return None

        # =================================================
        # SD 3.5
        # =================================================

        if provider == "sd35":

            logger.info(
                "[IMAGE] Running SD3.5"
            )

            result = client.predict(
                clean_prompt,
                "low quality, blurry, distorted",
                0,
                True,
                1024,
                1024,
                4.5,
                28,
                api_name="/infer",
            )

            image_path = extract_image_path(
                result
            )

            if image_path:

                logger.info(
                    f"[IMAGE] SD3.5 SUCCESS: "
                    f"{image_path}"
                )

                return image_path

            return None

        # =================================================
        # SDXL
        # =================================================

        if provider == "sdxl":

            logger.info(
                "[IMAGE] Running SDXL"
            )

            result = client.predict(
                clean_prompt,
                "",
                "",
                "",
                False,
                False,
                False,
                0,
                1024,
                1024,
                5.0,
                5.0,
                25,
                25,
                True,
                api_name="/predict",
            )

            image_path = extract_image_path(
                result
            )

            if image_path:

                logger.info(
                    f"[IMAGE] SDXL SUCCESS: "
                    f"{image_path}"
                )

                return image_path

            return None

        logger.error(
            f"[IMAGE] Unknown provider: {provider}"
        )

        return None

    except Exception as e:

        logger.exception(
            f"[IMAGE] Provider {provider} failed: {e}"
        )

        return None


# =========================================================
# IMAGE PROVIDERS
# =========================================================

IMAGE_PROVIDERS = [

    {
        "type": "flux_schnell",
        "url": "black-forest-labs/FLUX.1-schnell",
    },

    {
        "type": "flux_dev",
        "url": "black-forest-labs/FLUX.1-dev",
    },

]


# =========================================================
# PROMPT TO IMAGE
# =========================================================

@app.post("/api/prompt-to-image")
def prompt_to_image(
    request: PromptToImageRequest,
):

    prompt = request.prompt.strip()

    if not prompt:

        return {
            "success": False,
            "error": "Prompt is required",
        }

    style = (
        request.style.strip()
        if request.style
        else "Realistic"
    )

    final_prompt = (
        f"{prompt}, "
        f"{style} style, "
        "high quality, "
        "highly detailed, "
        "professional image"
    )

    logger.info(
        f"[IMAGE API] Request style={style}"
    )

    for provider in IMAGE_PROVIDERS:

        provider_name = provider["type"]

        logger.info(
            f"[IMAGE API] Trying "
            f"{provider_name}"
        )

        result = run_image_gen(
            provider,
            final_prompt,
        )

        if result:

            return {
                "success": True,
                "provider": provider_name,
                "image_url": result,
            }

    return {
        "success": False,
        "error": "All image providers failed",
    }


# =========================================================
# STARTUP
# =========================================================

@app.on_event("startup")
def startup_event():

    logger.info(
        "=========================================="
    )

    logger.info(
        "Raka AI Production API starting"
    )

    logger.info(
        f"HF_TOKEN configured: "
        f"{bool(HF_TOKEN)}"
    )

    logger.info(
        f"REPLICATE_API_KEY configured: "
        f"{bool(REPLICATE_API_KEY)}"
    )

    logger.info(
        "=========================================="
    )