def extract_image_path(result):
    if result is None:
        return None

    if isinstance(result, str):
        if result.startswith(("http://", "https://")):
            return result
        if os.path.isfile(result):
            return result
        return None

    if hasattr(result, "__fspath__"):
        path = os.fspath(result)
        return path if os.path.isfile(path) else None

    if isinstance(result, dict):
        for key in ("path", "url", "image", "value", "file", "name"):
            if key in result:
                found = extract_image_path(result[key])
                if found:
                    return found

    if isinstance(result, (list, tuple)):
        for item in result:
            found = extract_image_path(item)
            if found:
                return found

    return None


def run_image_gen(cand, prompt):
    client = Client(cand["url"], token=HF_TOKEN)

    if cand["type"] in ("flux_schnell", "flux_dev"):
        steps = 4 if cand["type"] == "flux_schnell" else 28

        res = client.predict(
            prompt,
            0,
            True,
            1024,
            1024,
            steps,
            api_name="/infer"
        )

        logger.info(
            f"[IMAGE] {cand['type']} raw result type={type(res)} "
            f"result={str(res)[:500]}"
        )

        return extract_image_path(res)

    if cand["type"] == "sd35":
        res = client.predict(
            prompt,
            "low quality",
            0,
            True,
            1024,
            1024,
            4.5,
            28,
            api_name="/infer"
        )

        return extract_image_path(res)

    if cand["type"] == "sdxl":
        res = client.predict(
            prompt,
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

        return extract_image_path(res)

    return None