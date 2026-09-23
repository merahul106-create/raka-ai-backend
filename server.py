def extract_image_path(result):
    """
    Safely extract an image path or URL from Gradio/API responses.

    Supports:
    - local file paths
    - URLs
    - pathlib objects
    - dictionaries
    - lists / tuples
    - nested combinations
    """

    if result is None:
        return None

    # String result
    if isinstance(result, str):
        result = result.strip()

        if not result:
            return None

        # Remote URL
        if result.startswith(("http://", "https://")):
            return result

        # Local file
        if os.path.isfile(result):
            return result

        return None

    # pathlib.Path / os.PathLike
    if hasattr(result, "__fspath__"):
        try:
            path = os.fspath(result)

            if path and os.path.isfile(path):
                return path

        except Exception as e:
            logger.warning(f"[IMAGE] Failed to read filesystem path: {e}")

        return None

    # Dictionary / Gradio FileData
    if isinstance(result, dict):
        for key in (
            "path",
            "url",
            "image",
            "value",
            "file",
            "name",
            "filepath",
            "file_path",
            "data",
        ):
            if key not in result:
                continue

            try:
                found = extract_image_path(result[key])

                if found:
                    return found

            except Exception as e:
                logger.warning(
                    f"[IMAGE] Failed parsing dict key '{key}': {e}"
                )

        return None

    # List / tuple
    if isinstance(result, (list, tuple)):
        for item in result:
            try:
                found = extract_image_path(item)

                if found:
                    return found

            except Exception as e:
                logger.warning(
                    f"[IMAGE] Failed parsing list item: {e}"
                )

        return None

    # Objects containing path/url attributes
    for attr in ("path", "url", "filepath", "file_path", "name"):
        try:
            value = getattr(result, attr, None)

            if value:
                found = extract_image_path(value)

                if found:
                    return found

        except Exception:
            pass

    return None


def run_image_gen(cand, prompt):
    """
    Generate an image using the selected Hugging Face Gradio provider.

    Returns:
        Local file path or remote URL, otherwise None.
    """

    provider = cand.get("type")
    space = cand.get("url")

    if not space:
        logger.error("[IMAGE] Provider URL is missing")
        return None

    if not prompt or not prompt.strip():
        logger.error("[IMAGE] Empty prompt")
        return None

    logger.info(
        f"[IMAGE] Starting provider={provider} space={space}"
    )

    try:
        # ---------------------------------------------------------
        # Create Gradio client
        # ---------------------------------------------------------
        if HF_TOKEN:
            client = Client(
                space,
                token=HF_TOKEN
            )
        else:
            logger.warning(
                "[IMAGE] HF_TOKEN is empty. "
                "Trying provider without authentication."
            )

            client = Client(space)

        # ---------------------------------------------------------
        # FLUX Schnell / FLUX Dev
        # ---------------------------------------------------------
        if provider in ("flux_schnell", "flux_dev"):

            steps = (
                4
                if provider == "flux_schnell"
                else 28
            )

            logger.info(
                f"[IMAGE] Running {provider} "
                f"steps={steps} size=1024x1024"
            )

            res = client.predict(
                prompt.strip(),
                0,
                True,
                1024,
                1024,
                steps,
                api_name="/infer"
            )

            logger.info(
                f"[IMAGE] {provider} raw result type="
                f"{type(res)}"
            )

            logger.info(
                f"[IMAGE] {provider} raw result="
                f"{str(res)[:1000]}"
            )

            image_path = extract_image_path(res)

            if image_path:
                logger.info(
                    f"[IMAGE] {provider} SUCCESS: "
                    f"{image_path}"
                )
                return image_path

            logger.error(
                f"[IMAGE] {provider}: "
                f"Could not extract image path"
            )

            return None

        # ---------------------------------------------------------
        # Stable Diffusion 3.5
        # ---------------------------------------------------------
        if provider == "sd35":

            logger.info(
                "[IMAGE] Running SD3.5 "
                "size=1024x1024"
            )

            res = client.predict(
                prompt.strip(),
                "low quality, blurry, distorted",
                0,
                True,
                1024,
                1024,
                4.5,
                28,
                api_name="/infer"
            )

            logger.info(
                f"[IMAGE] sd35 raw result type="
                f"{type(res)}"
            )

            logger.info(
                f"[IMAGE] sd35 raw result="
                f"{str(res)[:1000]}"
            )

            image_path = extract_image_path(res)

            if image_path:
                logger.info(
                    f"[IMAGE] sd35 SUCCESS: "
                    f"{image_path}"
                )
                return image_path

            logger.error(
                "[IMAGE] sd35: "
                "Could not extract image path"
            )

            return None

        # ---------------------------------------------------------
        # SDXL
        # ---------------------------------------------------------
        if provider == "sdxl":

            logger.info(
                "[IMAGE] Running SDXL "
                "size=1024x1024"
            )

            res = client.predict(
                prompt.strip(),
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
                api_name="/predict"
            )

            logger.info(
                f"[IMAGE] sdxl raw result type="
                f"{type(res)}"
            )

            logger.info(
                f"[IMAGE] sdxl raw result="
                f"{str(res)[:1000]}"
            )

            image_path = extract_image_path(res)

            if image_path:
                logger.info(
                    f"[IMAGE] sdxl SUCCESS: "
                    f"{image_path}"
                )
                return image_path

            logger.error(
                "[IMAGE] sdxl: "
                "Could not extract image path"
            )

            return None

        # ---------------------------------------------------------
        # Unknown provider
        # ---------------------------------------------------------
        logger.error(
            f"[IMAGE] Unknown provider type: {provider}"
        )

        return None

    # -------------------------------------------------------------
    # Gradio / provider error
    # -------------------------------------------------------------
    except Exception as e:

        logger.exception(
            f"[IMAGE] Provider {provider} failed: {e}"
        )

        return None