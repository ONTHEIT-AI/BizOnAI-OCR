# Based on Chandra (https://github.com/datalab-to/chandra) - Apache 2.0
import os
import subprocess
import sys

from bizonai_ocr.settings import settings


def main():
    env = {
        **os.environ,
        "VLLM_ATTENTION_BACKEND": os.environ.get("VLLM_ATTENTION_BACKEND", "TORCH_SDPA"),
    }

    if settings.VLLM_GPUS:
        env["CUDA_VISIBLE_DEVICES"] = settings.VLLM_GPUS

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", settings.MODEL_CHECKPOINT,
        "--port", "8000",
        "--dtype", "bfloat16",
        "--max-model-len", "16384",
        "--max-num-seqs", "32",
        "--gpu-memory-utilization", "0.9",
        "--no-enforce-eager",
        "--trust-remote-code",
        "--served-model-name", settings.VLLM_MODEL_NAME,
    ]

    print(f"Starting vLLM server: {' '.join(cmd)}")
    print(f"  CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print(f"  VLLM_ATTENTION_BACKEND={env.get('VLLM_ATTENTION_BACKEND', 'not set')}")

    try:
        subprocess.run(cmd, check=True, env=env)
    except KeyboardInterrupt:
        print("\nShutting down vLLM server...")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"\nvLLM server exited with error code {e.returncode}")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print("vLLM not found. Install with: pip install bizon-ocr")
        sys.exit(1)


if __name__ == "__main__":
    main()
