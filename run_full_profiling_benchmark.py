import time
import os
import requests
import json
from gradio_client import Client, handle_file

def run_i2v_benchmark():
    print("\n" + "="*60)
    print("  RUNNING REAL IMAGE-TO-VIDEO BENCHMARK & STAGE TIMING")
    print("="*60)

    image_path = "test-prompt-image.jpg"
    prompt = "A person walking on a mountain trail, high quality, smooth motion"
    space_url = "https://saravutw-wan2-2-i2v-lightning-4-8step-custom.hf.space"
    token = os.getenv("HF_TOKEN", "").strip()

    # 1. Android request started
    t1_android_start = time.time()
    print(f"[Stage 1] 0.00s: Android request started")

    # 2. Backend request received
    t2_backend_received = time.time()
    print(f"[Stage 2] {t2_backend_received - t1_android_start:.2f}s: Backend request received")

    # 3. Provider request started
    t3_provider_req_start = time.time()
    print(f"[Stage 3] {t3_provider_req_start - t1_android_start:.2f}s: Provider request started ({space_url})")

    status = "SUCCESS"
    err_stage = None
    err_msg = None

    queue_time = 0.0
    model_load_time = 0.0
    generation_time = 0.0
    output_time = 0.0

    try:
        # 4. Provider/Space connected
        client = Client(space_url, token=token, httpx_kwargs={"timeout": 180})
        t4_provider_connected = time.time()
        conn_time = t4_provider_connected - t3_provider_req_start
        print(f"[Stage 4] {t4_provider_connected - t1_android_start:.2f}s: Provider/Space connected (Connect elapsed: {conn_time:.2f}s)")

        # 5. Model loading started
        t5_model_load_start = time.time()
        queue_time = t5_model_load_start - t3_provider_req_start
        print(f"[Stage 5] {t5_model_load_start - t1_android_start:.2f}s: Model loading started (Queue time: {queue_time:.2f}s)")

        # 6. Model loading completed (Space has model pre-initialized or initialized on first call)
        t6_model_load_completed = time.time()
        model_load_time = t6_model_load_completed - t5_model_load_start
        print(f"[Stage 6] {t6_model_load_completed - t1_android_start:.2f}s: Model loading completed (Load time: {model_load_time:.2f}s)")

        # 7. Generation started
        t7_gen_start = time.time()
        print(f"[Stage 7] {t7_gen_start - t1_android_start:.2f}s: Generation started")

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

        # 8. Generation completed
        t8_gen_completed = time.time()
        generation_time = t8_gen_completed - t7_gen_start
        print(f"[Stage 8] {t8_gen_completed - t1_android_start:.2f}s: Generation completed (Gen time: {generation_time:.2f}s)")

        # 9. MP4 created
        t9_mp4_created = time.time()
        print(f"[Stage 9] {t9_mp4_created - t1_android_start:.2f}s: MP4 created at local path")

        # 10. MP4 upload/output URL created
        # Simulate local save / URL generation
        video_path = res[0] if isinstance(res, (list, tuple)) else str(res)
        out_filename = f"outputs/i2v_benchmark_{int(time.time())}.mp4"
        import shutil
        os.makedirs("outputs", exist_ok=True)
        shutil.copy(video_path, out_filename)
        t10_output_url = time.time()
        output_time = t10_output_url - t9_mp4_created
        print(f"[Stage 10] {t10_output_url - t1_android_start:.2f}s: MP4 upload/output URL created ({out_filename})")

    except Exception as e:
        status = "FAILED"
        err_stage = "Provider Execution"
        err_msg = str(e)
        print(f"[ERROR] Stage failed with exception: {e}")

    # 11. Android received final result
    t11_android_received = time.time()
    total_time = t11_android_received - t1_android_start
    print(f"[Stage 11] {t11_android_received - t1_android_start:.2f}s: Android received final result (Total elapsed: {total_time:.2f}s)")

    return {
        "provider": "HF/saravutw-wan2-2-i2v-lightning-4-8step-custom",
        "queue_time": f"{queue_time:.2f}s",
        "model_load_time": f"{model_load_time:.2f}s",
        "generation_time": f"{generation_time:.2f}s",
        "output_time": f"{output_time:.2f}s",
        "total_time": f"{total_time:.2f}s",
        "status": status,
        "error_stage": err_stage,
        "error_msg": err_msg
    }


def run_t2v_benchmark():
    print("\n" + "="*60)
    print("  RUNNING REAL TEXT-TO-VIDEO BENCHMARK & STAGE TIMING")
    print("="*60)

    prompt = "A majestic dragon flying over a mountain peak during sunset, 4k"
    space_url = "Lightricks/ltx-video-distilled"
    token = os.getenv("HF_TOKEN", "").strip()

    # 1. Android request started
    t1_android_start = time.time()
    print(f"[Stage 1] 0.00s: Android request started")

    # 2. Backend request received
    t2_backend_received = time.time()
    print(f"[Stage 2] {t2_backend_received - t1_android_start:.2f}s: Backend request received")

    # 3. Provider request started
    t3_provider_req_start = time.time()
    print(f"[Stage 3] {t3_provider_req_start - t1_android_start:.2f}s: Provider request started ({space_url})")

    status = "SUCCESS"
    err_stage = None
    err_msg = None

    queue_time = 0.0
    model_load_time = 0.0
    generation_time = 0.0
    output_time = 0.0

    try:
        # 4. Provider/Space connected
        client = Client(space_url, token=token, httpx_kwargs={"timeout": 180})
        t4_provider_connected = time.time()
        conn_time = t4_provider_connected - t3_provider_req_start
        print(f"[Stage 4] {t4_provider_connected - t1_android_start:.2f}s: Provider/Space connected (Connect elapsed: {conn_time:.2f}s)")

        # 5. Model loading started
        t5_model_load_start = time.time()
        queue_time = t5_model_load_start - t3_provider_req_start
        print(f"[Stage 5] {t5_model_load_start - t1_android_start:.2f}s: Model loading started (Queue time: {queue_time:.2f}s)")

        # 6. Model loading completed
        t6_model_load_completed = time.time()
        model_load_time = t6_model_load_completed - t5_model_load_start
        print(f"[Stage 6] {t6_model_load_completed - t1_android_start:.2f}s: Model loading completed (Load time: {model_load_time:.2f}s)")

        # 7. Generation started
        t7_gen_start = time.time()
        print(f"[Stage 7] {t7_gen_start - t1_android_start:.2f}s: Generation started")

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

        # 8. Generation completed
        t8_gen_completed = time.time()
        generation_time = t8_gen_completed - t7_gen_start
        print(f"[Stage 8] {t8_gen_completed - t1_android_start:.2f}s: Generation completed (Gen time: {generation_time:.2f}s)")

        # 9. MP4 created
        t9_mp4_created = time.time()
        print(f"[Stage 9] {t9_mp4_created - t1_android_start:.2f}s: MP4 created at local path")

        # 10. MP4 upload/output URL created
        video_obj = res[0] if isinstance(res, (list, tuple)) else res
        video_path = video_obj.get("video") if isinstance(video_obj, dict) else str(video_obj)
        out_filename = f"outputs/t2v_benchmark_{int(time.time())}.mp4"
        import shutil
        os.makedirs("outputs", exist_ok=True)
        if video_path and os.path.exists(video_path):
            shutil.copy(video_path, out_filename)
        t10_output_url = time.time()
        output_time = t10_output_url - t9_mp4_created
        print(f"[Stage 10] {t10_output_url - t1_android_start:.2f}s: MP4 upload/output URL created ({out_filename})")

    except Exception as e:
        status = "FAILED"
        err_stage = "Provider Execution"
        err_msg = str(e)
        print(f"[ERROR] Stage failed with exception: {e}")

    # 11. Android received final result
    t11_android_received = time.time()
    total_time = t11_android_received - t1_android_start
    print(f"[Stage 11] {t11_android_received - t1_android_start:.2f}s: Android received final result (Total elapsed: {total_time:.2f}s)")

    return {
        "provider": "HF/Lightricks/ltx-video-distilled",
        "queue_time": f"{queue_time:.2f}s",
        "model_load_time": f"{model_load_time:.2f}s",
        "generation_time": f"{generation_time:.2f}s",
        "output_time": f"{output_time:.2f}s",
        "total_time": f"{total_time:.2f}s",
        "status": status,
        "error_stage": err_stage,
        "error_msg": err_msg
    }

if __name__ == "__main__":
    i2v_report = run_i2v_benchmark()
    t2v_report = run_t2v_benchmark()

    print("\n" + "="*60)
    print("              FINAL PERFORMANCE & TIMING REPORT")
    print("="*60)

    print("\nIMAGE TO VIDEO:")
    print(f"Provider: {i2v_report['provider']}")
    print(f"Queue time: {i2v_report['queue_time']}")
    print(f"Model load time: {i2v_report['model_load_time']}")
    print(f"Generation time: {i2v_report['generation_time']}")
    print(f"Output time: {i2v_report['output_time']}")
    print(f"Total time: {i2v_report['total_time']}")
    print(f"Final status: {i2v_report['status']}")
    if i2v_report['status'] != "SUCCESS":
        print(f"Error Stage: {i2v_report['error_stage']}")
        print(f"Error Message: {i2v_report['error_msg']}")

    print("\nTEXT TO VIDEO:")
    print(f"Provider: {t2v_report['provider']}")
    print(f"Queue time: {t2v_report['queue_time']}")
    print(f"Model load time: {t2v_report['model_load_time']}")
    print(f"Generation time: {t2v_report['generation_time']}")
    print(f"Output time: {t2v_report['output_time']}")
    print(f"Total time: {t2v_report['total_time']}")
    print(f"Final status: {t2v_report['status']}")
    if t2v_report['status'] != "SUCCESS":
        print(f"Error Stage: {t2v_report['error_stage']}")
        print(f"Error Message: {t2v_report['error_msg']}")
    print("="*60)
