def extract_image_path(result):
    """
    Safely extract an image path or URL from Gradio/API responses.

    Supports:
    - local file paths
    - URLs
    - pathlib/os.PathLike objects
    - dictionaries
    - lists / tuples
    - nested combinations
    """

    if result is None:
        return None

    # ---------------------------------------------------------
    # String
    # ---------------------------------------------------------
    if isinstance(result, str):
        value = result.strip()

        if not value:
            return None

        # Remote URL
        if value.startswith(("http://", "https://")):
            return value

        # Local file
        if os.path.isfile(value):
            return value

        return None

    # ---------------------------------------------------------
    # pathlib.Path / os.PathLike
    # ---------------------------------------------------------
    if hasattr(result, "__fspath__"):
        try:
            path = os.fspath(result)

            if path and os.path.isfile(path):
                return path

        except Exception as e:
            logger.warning(
                f"[IMAGE] Path parsing failed: {e}"
            )

        return None

    # ---------------------------------------------------------
    # Dictionary / Gradio FileData
    # ---------------------------------------------------------
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
                found = extract_image_path(result[key])

                if found:
                    return found

            except Exception as e:
                logger.warning(
                    f"[IMAGE] Failed parsing key "
                    f"{key}: {e}"
                )

        return None

    # ---------------------------------------------------------
    # List / Tuple
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # Object with path/url attributes
    # ---------------------------------------------------------
    for attr in (
        "path",
        "url",
        "filepath",
        "file_path",
        "name",
    ):

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
    Generate an image using a Hugging Face Gradio provider.

    Returns:
        Local file path or remote URL.
        None when generation fails.
    """

    provider = cand.get("type")
    space = cand.get("url")

    # ---------------------------------------------------------
    # Validate provider
    # ---------------------------------------------------------
    if not space:

        logger.error(
            "[IMAGE] Provider URL is missing"
        )

        return None

    # ---------------------------------------------------------
    # Validate prompt
    # ---------------------------------------------------------
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

        # -----------------------------------------------------
        # Create Gradio Client
        # -----------------------------------------------------
        if HF_TOKEN:

            logger.info(
                "[IMAGE] Using Hugging Face authentication"
            )

            client = Client(
                space,
                token=HF_TOKEN
            )

        else:

            logger.warning(
                "[IMAGE] HF_TOKEN is empty. "
                "Trying public Space without authentication."
            )

            client = Client(space)

        # =====================================================
        # FLUX SCHNELL / FLUX DEV
        # =====================================================
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
                f"steps={steps} "
                f"size=1024x1024"
            )

            res = client.predict(
                clean_prompt,
                0,
                True,
                1024,
                1024,
                steps,
                api_name="/infer",
            )

            logger.info(
                f"[IMAGE] {provider} "
                f"result_type={type(res)}"
            )

            logger.info(
                f"[IMAGE] {provider} "
                f"raw_result={str(res)[:1500]}"
            )

            image_path = extract_image_path(res)

            if image_path:

                logger.info(
                    f"[IMAGE] {provider} SUCCESS: "
                    f"{image_path}"
                )

                return image_path

            logger.error(
                f"[IMAGE] {provider} failed: "
                "image path could not be extracted"
            )

            return None

        # =====================================================
        # STABLE DIFFUSION 3.5
        # =====================================================
        if provider == "sd35":

            logger.info(
                "[IMAGE] Running SD3.5 "
                "size=1024x1024"
            )

            res = client.predict(
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

            logger.info(
                f"[IMAGE] sd35 "
                f"result_type={type(res)}"
            )

            logger.info(
                f"[IMAGE] sd35 "
                f"raw_result={str(res)[:1500]}"
            )

            image_path = extract_image_path(res)

            if image_path:

                logger.info(
                    f"[IMAGE] sd35 SUCCESS: "
                    f"{image_path}"
                )

                return image_path

            logger.error(
                "[IMAGE] sd35 failed: "
                "image path could not be extracted"
            )

            return None

        # =====================================================
        # SDXL
        # =====================================================
        if provider == "sdxl":

            logger.info(
                "[IMAGE] Running SDXL "
                "size=1024x1024"
            )

            res = client.predict(
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

            logger.info(
                f"[IMAGE] sdxl "
                f"result_type={type(res)}"
            )

            logger.info(
                f"[IMAGE] sdxl "
                f"raw_result={str(res)[:1500]}"
            )

            image_path = extract_image_path(res)

            if image_path:

                logger.info(
                    f"[IMAGE] sdxl SUCCESS: "
                    f"{image_path}"
                )

                return image_path

            logger.error(
                "[IMAGE] sdxl failed: "
                "image path could not be extracted"
            )

            return None

        # =====================================================
        # UNKNOWN PROVIDER
        # =====================================================
        logger.error(
            f"[IMAGE] Unknown provider type: {provider}"
        )

        return None

    # =========================================================
    # PROVIDER / GRADIO ERROR
    # =========================================================
    except Exception as e:

        logger.exception(
            f"[IMAGE] Provider {provider} failed: {e}"
        )

        return None