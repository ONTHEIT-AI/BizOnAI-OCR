# Based on Chandra (https://github.com/datalab-to/chandra) - Apache 2.0
#!/usr/bin/env python3
"""
BizOnAI-OCR Pipeline - Inference pipeline with automatic vLLM management.

Outputs raw model results. Use bizonai_ocr.postprocess to convert to markdown/html.

Usage:
    # Basic usage (auto-starts vLLM server)
    python -m bizonai_ocr.pipeline ./output --input /path/to/pdfs

    # With external vLLM server
    python -m bizonai_ocr.pipeline ./output --input /path/to/pdfs --server http://localhost:8000/v1

    # For benchmarking
    python -m bizonai_ocr.pipeline ./output --input /path/to/pdfs --benchmark
"""

import argparse
import asyncio
import base64
import logging
import multiprocessing
import os
import signal
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional

import httpx
from PIL import Image
from tqdm.asyncio import tqdm
import threading

from bizonai_ocr.prompts import OCR_LAYOUT_PROMPT
from bizonai_ocr.model.util import detect_repeat_token

# Global lock for pypdfium2 (not thread-safe)
_pdf_lock = threading.Lock()

# Set spawn method only if not already set (avoid conflicts with pypdfium2)
try:
    multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass  # Already set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default settings (matched to original model)
DEFAULT_MODEL = "ONTHEIT/BizOnAI-OCR"
DEFAULT_PORT = 8000
IMAGE_DPI = 192
MIN_PDF_IMAGE_DIM = 1024
MAX_IMAGE_PIXELS = (3072, 2048)  # max_width, max_height for pixel count
MIN_IMAGE_PIXELS = (28, 28)
BBOX_SCALE = 1024
MAX_OUTPUT_TOKENS = 12384


@dataclass
class ProcessResult:
    """Result of processing a single file."""
    input_path: str
    output_path: str
    success: bool
    error: Optional[str] = None


def pdf_to_image(pdf_path: str, page_num: int = 1) -> Optional[Image.Image]:
    """Convert a PDF page to PIL Image using pypdfium2 (matches CLI)."""
    # Use lock because pypdfium2 (PDFium C library) is not thread-safe
    with _pdf_lock:
        try:
            import pypdfium2 as pdfium
            import pypdfium2.raw as pdfium_c

            doc = pdfium.PdfDocument(pdf_path)
            doc.init_forms()

            if page_num > len(doc):
                logger.warning(f"Page {page_num} not found in {pdf_path}, using page 1")
                page_num = 1

            page_obj = doc[page_num - 1]

            # Calculate DPI based on minimum dimension
            min_page_dim = min(page_obj.get_width(), page_obj.get_height())
            scale_dpi = (MIN_PDF_IMAGE_DIM / min_page_dim) * 72
            scale_dpi = max(scale_dpi, IMAGE_DPI)

            # Flatten form fields and annotations (matches CLI)
            rc = pdfium_c.FPDFPage_Flatten(page_obj, pdfium_c.FLAT_NORMALDISPLAY)
            if rc == pdfium_c.FLATTEN_FAIL:
                logger.warning(f"Failed to flatten annotations/form fields on page {page_num}")

            # Re-get page after flatten
            page_obj = doc[page_num - 1]
            pil_image = page_obj.render(scale=scale_dpi / 72).to_pil().convert("RGB")

            doc.close()
            return pil_image

        except ImportError:
            logger.error("pypdfium2 not installed. Install with: pip install pypdfium2")
            return None
        except Exception as e:
            logger.error(f"Error converting PDF {pdf_path}: {e}")
            return None


def scale_to_fit(
    img: Image.Image,
    max_size: tuple = None,
    min_size: tuple = None,
) -> Image.Image:
    """Scale image based on pixel count."""
    import math
    if max_size is None:
        max_size = MAX_IMAGE_PIXELS
    if min_size is None:
        min_size = MIN_IMAGE_PIXELS

    width, height = img.size
    if width == 0 or height == 0:
        return img

    max_width, max_height = max_size
    min_width, min_height = min_size

    current_pixels = width * height
    max_pixels = max_width * max_height
    min_pixels = min_width * min_height

    if current_pixels > max_pixels:
        scale_factor = (max_pixels / current_pixels) ** 0.5
        new_width = math.floor(width * scale_factor)
        new_height = math.floor(height * scale_factor)
    elif current_pixels < min_pixels:
        scale_factor = (min_pixels / current_pixels) ** 0.5
        new_width = math.ceil(width * scale_factor)
        new_height = math.ceil(height * scale_factor)
    else:
        return img

    return img.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = scale_to_fit(image)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def get_prompt(bbox_scale: int = BBOX_SCALE) -> str:
    """Get the OCR prompt with bbox scale."""
    return OCR_LAYOUT_PROMPT.format(bbox_scale=bbox_scale)


class VLLMServer:
    """Manages vLLM server lifecycle with proper stdout/stderr handling."""

    def __init__(self):
        self.process: Optional[asyncio.subprocess.Process] = None
        self.read_task: Optional[asyncio.Task] = None
        self.server_ready = False
        self._log_lines: list[str] = []

    async def start(
        self,
        model: str,
        port: int,
        tensor_parallel_size: int,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 16384,
        max_num_seqs: int = 128,
        data_parallel_size: int = 1,
    ) -> None:
        """Start vLLM server."""
        cmd = [
            "vllm", "serve", model,
            "--port", str(port),
            "--tensor-parallel-size", str(tensor_parallel_size),
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--max-model-len", str(max_model_len),
            "--max-num-seqs", str(max_num_seqs),
            "--max_num_batched_tokens", "65536",
            "--dtype", "bfloat16",
            "--no-enforce-eager",
            "--trust-remote-code",
            "--disable-log-requests",
        ]

        # Add data parallel if > 1
        if data_parallel_size > 1:
            cmd.extend(["--data-parallel-size", str(data_parallel_size)])

        logger.info(f"Starting vLLM server: {' '.join(cmd)}")

        # Set environment variables to match Docker setup
        env = {
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "VLLM_ATTENTION_BACKEND": "TORCH_SDPA",
        }

        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        # Start background task to read stdout/stderr (prevents buffer blocking)
        self.read_task = asyncio.create_task(self._read_output())

    async def _read_output(self) -> None:
        """Read stdout and stderr to prevent buffer blocking."""
        async def read_stream(stream, prefix: str):
            while True:
                line = await stream.readline()
                if not line:
                    break
                try:
                    decoded = line.decode("utf-8").rstrip()
                    self._log_lines.append(f"{prefix}: {decoded}")
                    # Check for ready message
                    if "Application startup complete" in decoded or "Uvicorn running" in decoded:
                        self.server_ready = True
                    # Log errors
                    if "error" in decoded.lower() or "exception" in decoded.lower():
                        logger.warning(f"vLLM: {decoded}")
                except Exception:
                    pass

        if self.process and self.process.stdout and self.process.stderr:
            await asyncio.gather(
                read_stream(self.process.stdout, "stdout"),
                read_stream(self.process.stderr, "stderr"),
                return_exceptions=True,
            )

    async def stop(self) -> None:
        """Stop vLLM server."""
        if self.read_task:
            self.read_task.cancel()
            try:
                await self.read_task
            except asyncio.CancelledError:
                pass

        if self.process:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=10)
            except (ProcessLookupError, asyncio.TimeoutError):
                try:
                    self.process.kill()
                except ProcessLookupError:
                    pass

    def get_logs(self) -> str:
        """Get collected log lines."""
        return "\n".join(self._log_lines[-100:])  # Last 100 lines


async def wait_for_server(url: str, timeout: int = 300) -> bool:
    """Wait for vLLM server to be ready."""
    health_url = url.rstrip("/").replace("/v1", "") + "/health"
    start_time = asyncio.get_event_loop().time()

    async with httpx.AsyncClient() as client:
        while asyncio.get_event_loop().time() - start_time < timeout:
            try:
                resp = await client.get(health_url, timeout=5)
                if resp.status_code == 200:
                    logger.info("vLLM server is ready")
                    return True
            except Exception:
                pass
            await asyncio.sleep(2)
            logger.info("Waiting for vLLM server...")

    logger.error(f"vLLM server did not start within {timeout} seconds")
    return False


async def process_single(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    image: Image.Image,
    api_base: str,
    model_name: str,
    max_retries: int = 6,
    exp_temp_retry: bool = False,
) -> tuple[bool, str, str]:
    """Process a single image through vLLM. Returns raw model output."""
    async with semaphore:
        image_b64 = await asyncio.to_thread(image_to_base64, image)

        for attempt in range(max_retries):
            # First attempt: deterministic (temperature=0, top_p=0.1)
            # Retries: temperature depends on exp_temp_retry flag
            if attempt == 0:
                temperature, top_p = 0.0, 0.1
            elif exp_temp_retry:
                # Exponential: stay low early, ramp up later
                # attempt: 1    2    3    4    5
                # temp:    0.3  0.3  0.4  0.5  0.7
                retry_temps = [0.3, 0.3, 0.4, 0.5, 0.7]
                temperature = retry_temps[min(attempt - 1, len(retry_temps) - 1)]
                top_p = 0.95
            else:
                # Default: fixed values
                temperature, top_p = 0.3, 0.95

            try:
                payload = {
                    "model": model_name,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                            {"type": "text", "text": get_prompt()},
                        ]
                    }],
                    "max_tokens": MAX_OUTPUT_TOKENS,
                    "temperature": temperature,
                    "top_p": top_p,
                }

                response = await client.post(
                    f"{api_base}/chat/completions",
                    json=payload,
                    timeout=300,
                )

                if response.status_code != 200:
                    raise Exception(f"HTTP {response.status_code}: {response.text[:200]}")

                result = response.json()
                raw_output = result["choices"][0]["message"]["content"]

                # Check for repeat tokens (matches CLI behavior)
                has_repeat = detect_repeat_token(raw_output) or (
                    len(raw_output) > 50 and detect_repeat_token(raw_output, cut_from_end=50)
                )

                if has_repeat:
                    if attempt < max_retries - 1:
                        logger.warning(f"Detected repeat token, retrying generation (attempt {attempt + 1})...")
                        continue
                    else:
                        # All retries exhausted due to repeat token - model genuinely can't handle this
                        return False, "", "repeat_token_failure"

                return True, raw_output, ""

            except Exception as e:
                error_msg = str(e) if str(e) else f"{type(e).__name__}"
                if attempt == max_retries - 1:
                    return False, "", error_msg
                logger.warning(f"Error, retrying (attempt {attempt + 1}): {error_msg}")
                await asyncio.sleep(2 * (attempt + 1))  # Linear backoff like CLI

    return False, "", "Unknown error"


async def process_file(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    input_path: Path,
    output_path: Path,
    api_base: str,
    model_name: str,
    page_num: int = 1,
) -> ProcessResult:
    """Process a single PDF or image file."""
    try:
        # Load image
        if input_path.suffix.lower() == ".pdf":
            image = await asyncio.to_thread(pdf_to_image, str(input_path), page_num)
        else:
            image = await asyncio.to_thread(Image.open, str(input_path))
            if image.mode != "RGB":
                image = image.convert("RGB")

        if image is None:
            return ProcessResult(str(input_path), str(output_path), False, "Failed to load image")

        # Process
        success, output, error = await process_single(
            client, semaphore, image, api_base, model_name
        )

        if success:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(output, encoding="utf-8")

        return ProcessResult(str(input_path), str(output_path), success, error)

    except Exception as e:
        return ProcessResult(str(input_path), str(output_path), False, str(e))


async def run_pipeline(
    input_paths: list[Path],
    output_dir: Path,
    api_base: str,
    model_name: str,
    max_concurrent: int = 64,
    benchmark_naming: bool = False,
    skip_existing: bool = False,
    exp_temp_retry: bool = False,
) -> tuple[int, int, int, list[str]]:
    """Run the inference pipeline on all input files."""
    semaphore = asyncio.Semaphore(max_concurrent)

    success_count = 0
    error_count = 0
    skipped_count = 0
    errors = []

    # Step 1: Prepare file list
    file_items = []  # [(input_path, output_path), ...]

    for input_path in input_paths:
        if benchmark_naming:
            out_name = f"{input_path.stem}_pg1_repeat1.md"
        else:
            out_name = f"{input_path.stem}.md"

        try:
            rel_path = input_path.parent.name
            out_file = output_dir / rel_path / out_name
        except Exception:
            out_file = output_dir / out_name

        if skip_existing and out_file.exists():
            skipped_count += 1
            continue

        file_items.append((input_path, out_file))

    # Step 2: Load images in parallel (with lock for thread safety)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm as sync_tqdm

    def load_single(item):
        input_path, out_file = item
        if input_path.suffix.lower() == ".pdf":
            image = pdf_to_image(str(input_path), 1)
        else:
            try:
                image = Image.open(str(input_path))
                if image.mode != "RGB":
                    image = image.convert("RGB")
            except Exception:
                image = None
        return (image, out_file, input_path)

    work_items = []
    # Use limited workers to prevent lock contention issues
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(load_single, item) for item in file_items]
        for future in sync_tqdm(as_completed(futures), total=len(futures), desc="Loading PDFs"):
            image, out_file, input_path = future.result()
            if image is None:
                error_count += 1
                errors.append(f"{input_path}: Failed to load image")
            else:
                work_items.append((image, out_file))

    logger.info(f"Loaded {len(work_items)} images, {skipped_count} skipped, {error_count} failed")

    # Step 2: Process images through vLLM (async, parallel)
    limits = httpx.Limits(max_connections=max_concurrent, max_keepalive_connections=max_concurrent)
    async with httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(300.0)) as client:

        async def process_item(image: Image.Image, output_path: Path) -> ProcessResult:
            try:
                success, output, error = await process_single(
                    client, semaphore, image, api_base, model_name,
                    exp_temp_retry=exp_temp_retry
                )
                if success:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text(output, encoding="utf-8")
                elif benchmark_naming and error == "repeat_token_failure":
                    # Write empty file only for repeat token failure (model genuinely can't handle)
                    # HTTP errors should not write file (need to re-run later)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text("", encoding="utf-8")
                return ProcessResult(str(output_path), str(output_path), success, error)
            except Exception as e:
                # Don't write file on exception - need to re-run later
                return ProcessResult(str(output_path), str(output_path), False, str(e))

        tasks = [process_item(img, out) for img, out in work_items]

        if not tasks:
            logger.info("No files to process (all skipped or empty input)")
        else:
            for coro in tqdm.as_completed(tasks, total=len(tasks), desc="Processing"):
                result = await coro
                if result.success:
                    success_count += 1
                else:
                    error_count += 1
                    if error_count <= 10:
                        errors.append(f"{result.input_path}: {result.error}")

    return success_count, error_count, skipped_count, errors


def collect_input_files(input_paths: list[str]) -> list[Path]:
    """Collect all PDF and image files from input paths."""
    files = []
    extensions = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"}

    for path_str in input_paths:
        path = Path(path_str)

        if path.is_file():
            if path.suffix.lower() in extensions:
                files.append(path)
        elif path.is_dir():
            for ext in extensions:
                files.extend(path.rglob(f"*{ext}"))
        else:
            # Try glob pattern
            import glob
            for match in glob.glob(path_str, recursive=True):
                p = Path(match)
                if p.is_file() and p.suffix.lower() in extensions:
                    files.append(p)

    return sorted(set(files))


async def main():
    parser = argparse.ArgumentParser(
        description="BizOnAI-OCR Pipeline - Inference with automatic vLLM management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (auto-starts vLLM)
    python -m bizonai_ocr.pipeline ./output --input /path/to/pdfs

    # With external server
    python -m bizonai_ocr.pipeline ./output --input /path/to/pdfs --server http://localhost:8000/v1

    # Benchmark mode (olmocr-compatible naming)
    python -m bizonai_ocr.pipeline ./output --input /path/to/pdfs --benchmark

    # Custom model
    python -m bizonai_ocr.pipeline ./output --input /path/to/pdfs --model /path/to/model

Post-processing:
    # Convert raw output to markdown
    python -m bizonai_ocr.postprocess ./output --format markdown --output ./output_md

    # Convert raw output to html
    python -m bizonai_ocr.postprocess ./output --format html --output ./output_html
        """
    )

    parser.add_argument("output", help="Output directory for raw results")
    parser.add_argument("--input", "-i", nargs="+", required=True,
                        help="Input files, directories, or glob patterns")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Model path (default: {DEFAULT_MODEL})")
    parser.add_argument("--server", type=str, default=None,
                        help="External vLLM server URL (e.g., http://localhost:8000/v1). "
                             "If not provided, starts internal server.")
    parser.add_argument("--max-concurrent", type=int, default=1600,
                        help="Max concurrent requests (default: 1600, matches olmocr)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Use olmocr benchmark naming convention ({stem}_pg1_repeat1.md)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip files that already have output")
    parser.add_argument("--exp-temp-retry", action="store_true",
                        help="Use exponential temperature increase on retries (0.3→0.3→0.4→0.5→0.7)")
    parser.add_argument("--max-pixels", type=int, default=None,
                        help="Max total pixels for input images (default: 6291456 = 3072*2048). "
                             "Set to match training max_pixels for consistency (e.g., 451584).")

    # vLLM server options (used when --server is not provided)
    server_group = parser.add_argument_group("vLLM Server Options (when running internal server)")
    server_group.add_argument("--port", type=int, default=DEFAULT_PORT,
                              help=f"Port for internal vLLM server (default: {DEFAULT_PORT})")
    server_group.add_argument("--tensor-parallel-size", "-tp", type=int, default=1,
                              help="Tensor parallel size (default: 1)")
    server_group.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                              help="GPU memory utilization (default: 0.9)")
    server_group.add_argument("--max-model-len", type=int, default=32768,
                              help="Max model length (default: 16384)")
    server_group.add_argument("--max-num-seqs", type=int, default=32,
                              help="Max concurrent sequences for vLLM (default: 32)")
    server_group.add_argument("--data-parallel-size", "-dp", type=int, default=1,
                              help="Data parallel size - replicate model across GPUs (default: 1)")

    args = parser.parse_args()

    # Override max image pixels if specified
    if args.max_pixels is not None:
        global MAX_IMAGE_PIXELS
        import math
        side = int(math.sqrt(args.max_pixels))
        MAX_IMAGE_PIXELS = (side, side)
        logger.info(f"Max image pixels set to {args.max_pixels} (approx {side}x{side})")

    # Collect input files
    input_files = collect_input_files(args.input)
    if not input_files:
        logger.error("No input files found")
        sys.exit(1)

    logger.info(f"Found {len(input_files)} files to process")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine API base
    vllm_server: Optional[VLLMServer] = None
    if args.server:
        api_base = args.server.rstrip("/")
        if not api_base.endswith("/v1"):
            api_base += "/v1"
        logger.info(f"Using external server: {api_base}")
    else:
        # Start internal vLLM server
        api_base = f"http://localhost:{args.port}/v1"
        logger.info(f"Starting internal vLLM server on port {args.port}...")

        vllm_server = VLLMServer()
        await vllm_server.start(
            model=args.model,
            port=args.port,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            data_parallel_size=args.data_parallel_size,
        )

        # Wait for server
        if not await wait_for_server(api_base):
            logger.error("vLLM failed to start. Recent logs:")
            logger.error(vllm_server.get_logs())
            await vllm_server.stop()
            sys.exit(1)

    try:
        # Run pipeline
        success, errors_count, skipped, error_msgs = await run_pipeline(
            input_paths=input_files,
            output_dir=output_dir,
            api_base=api_base,
            model_name=args.model,
            max_concurrent=args.max_concurrent,
            benchmark_naming=args.benchmark,
            skip_existing=args.skip_existing,
            exp_temp_retry=args.exp_temp_retry,
        )

        # Print summary
        print(f"\n{'='*60}")
        print(f"Completed: {success} success, {errors_count} errors, {skipped} skipped")
        print(f"Output directory: {output_dir}")
        if error_msgs:
            print(f"\nSample errors:")
            for e in error_msgs[:5]:
                print(f"  - {e}")
        print(f"{'='*60}")

    finally:
        # Cleanup
        if vllm_server:
            logger.info("Shutting down vLLM server...")
            await vllm_server.stop()


def cli():
    """CLI entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    cli()
