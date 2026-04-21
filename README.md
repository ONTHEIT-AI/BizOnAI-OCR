# BizOnAI-OCR

A **Korean-optimized** OCR model for industrial document processing developed by [ONTHEIT](http://www.ontheit.com/).

BizOnAI-OCR is fine-tuned specifically for Korean documents — **public-sector documents, contracts, financial forms, medical records, and more.** It handles Korean-specific layouts, mixed Korean/English/Chinese text, decorative spacing, vertical tables, and complex forms that trip up general-purpose OCR models.

We also release **KDoc-OCRBench-V2**, a comprehensive Korean document OCR benchmark with 56,197 manually-reviewed test cases — designed to fill the gap in standardized Korean OCR evaluation.

## Benchmark Results

### KDoc-OCRBench-V2 (Korean, expanded)

Evaluated on [**KDoc-OCRBench-V2**](https://huggingface.co/datasets/ONTHEIT/KDoc-OCRBench-V2) — **56,197 manually-reviewed unit tests across 849 Korean public-sector PDFs** (statistics, reports, manuals, notices). V2 is a separate, complementary benchmark with much heavier table coverage (49,115 table cell tests, ~12× v1) and openly-shippable source PDFs (KOGL Type 1). See the [KDoc-OCRBench-V2 section](#kdoc-ocrbench-v2-korean-document-ocr-benchmark) below for details.

| Model | Baseline | Header/Footer | Long Text | Table | **Overall** |
|-------|---------:|--------------:|----------:|------:|------------:|
| **BizOnAI-OCR** | 98.6 | 94.7 | **77.9** | **58.1** | **82.3** |
| PaddleOCR-VL | 100.0 | 95.6 | 66.2 | 48.9 | 77.7 |
| DeepSeek OCR | 99.9 | 95.8 | 64.5 | 46.6 | 76.7 |
| olmOCR v0.2.0 | 99.9 | 95.2 | 65.0 | 44.9 | 76.3 |
| GLM OCR | 99.2 | **97.4** | 20.0 | 30.0 | 61.7 |

> Evaluated with upstream `olmocr.bench.benchmark` (plug-compatible).

### KDoc-OCRBench (Korean)

Evaluated on [**KDoc-OCRBench**](https://huggingface.co/datasets/ONTHEIT/KDoc-OCRBench) — 14,738 unit-test-style assertions across 804 Korean PDFs in 7 industrial document categories (contracts, financial forms, medical records, logistics, education, government, presentations). Tests check text presence/absence, table cell relationships, and header/footer handling. See [kdoc_benchmark/](kdoc_benchmark/) for details.

| Model | Baseline | Header/Footer | Long Text | Table | **Overall** |
|-------|----------|---------------|-----------|-------|-------------|
| **BizOnAI-OCR** | 98.1 | 88.8 | 77.0 | 68.0 | **83.0** |
| olmOCR v0.2.0 | 99.9 | 88.5 | 73.7 | 53.4 | 78.9 |
| PaddleOCR-VL | 100.0 | 90.5 | 65.7 | 57.6 | 78.4 |
| DeepSeek OCR | 99.8 | 88.3 | 66.0 | 53.1 | 76.8 |
| GLM OCR | 99.9 | 96.3 | 41.8 | 31.0 | 67.2 |

> Evaluated with `--table-normalize-spaces --table-normalize-newlines`.

### olmOCR-bench (English)

BizOnAI-OCR is Korean-optimized but also remains competitive on English documents. Evaluated on [**olmOCR-bench**](https://huggingface.co/datasets/allenai/olmOCR-bench) by Allen AI.

| Model | ArXiv | Old Scans Math | Tables | Old Scans | Headers & Footers | Multi Column | Long Tiny Text | Base | Overall |
|-------|------:|---------------:|-------:|----------:|------------------:|-------------:|---------------:|-----:|--------:|
| **BizOnAI-OCR** | 84.0 | 74.2 | 89.7 | 47.3 | 92.4 | 80.2 | 92.5 | 98.7 | 82.4 |
| PaddleOCR-VL | 85.7 | 71.0 | 84.1 | 37.8 | 97.0 | 79.9 | 85.7 | 98.5 | 80.0 |
| olmOCR v0.2.0 | 78.8 | 77.5 | 71.9 | 45.4 | 94.2 | 78.6 | 81.4 | 99.8 | 78.5 |
| DeepSeek OCR | 77.2 | 73.6 | 80.2 | 33.3 | 96.1 | 66.4 | 79.4 | 99.8 | 75.7 |

## Installation

Model weights are hosted on HuggingFace: [ONTHEIT/BizOnAI-OCR](https://huggingface.co/ONTHEIT/BizOnAI-OCR)

```bash
git clone https://github.com/ONTHEIT-AI/BizOnAI-OCR.git
cd BizOnAI-OCR
uv venv && source .venv/bin/activate

# Recommended: install from lock file (exact tested versions)
uv pip install -r requirements.lock

# Or install from pyproject.toml (with benchmark support)
uv pip install -e ".[bench]"
```

> **Tested environment**: Python 3.12, vllm==0.11.0, transformers==4.57.6, torch==2.8.0+cu128 (CUDA 12.8)
>
> If model downloads fail with disk-space errors, set `export HF_HOME=/path/to/large/disk/.cache/huggingface` before running.

## Quick Start

### Option 1: vLLM server (recommended for speed)

```bash
# Start vLLM server (uses GPU 0 by default, configurable via VLLM_GPUS)
bizonai-ocr-vllm

# In another terminal, process documents
bizonai-ocr input.pdf ./output
```

### Option 2: HuggingFace local inference

```bash
# No separate server needed, but slower for multiple files
bizonai-ocr input.pdf ./output --method hf
```

### Option 3: Batch pipeline (for large-scale processing)

The pipeline auto-starts a vLLM server, processes all files, then shuts down.

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 python -m bizonai_ocr.pipeline ./output \
    --model ONTHEIT/BizOnAI-OCR \
    --input /path/to/pdfs \
    --benchmark \
    --max-num-seqs 32 \
    --max-concurrent 64 \
    -tp 1 \
    --port 8111

# Multi-GPU (data parallel - runs model replicas on each GPU)
CUDA_VISIBLE_DEVICES=0,1 python -m bizonai_ocr.pipeline ./output \
    --model ONTHEIT/BizOnAI-OCR \
    --input /path/to/pdfs \
    --benchmark \
    --max-num-seqs 32 \
    --max-concurrent 64 \
    -tp 1 \
    --data-parallel-size 2 \
    --port 8111

# Post-process raw output to markdown
python -m bizonai_ocr.postprocess ./output --format markdown --output ./output_md
```

**Pipeline options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Model path (local or HuggingFace) | `ONTHEIT/BizOnAI-OCR` |
| `--server` | External vLLM server URL (skips auto-start) | auto-start |
| `-tp` | Tensor parallel size (GPUs per model shard) | 1 |
| `--data-parallel-size` | Number of model replicas across GPUs | 1 |
| `--max-num-seqs` | Max concurrent sequences in vLLM | 32 |
| `--max-concurrent` | Max concurrent API requests | 1600 |
| `--benchmark` | Use benchmark naming (`{stem}_pg1_repeat1.md`) | off |
| `--skip-existing` | Skip already processed files | off |

### Configuration

Settings via environment variables or `local.env` file:

```bash
# Model
MODEL_CHECKPOINT=ONTHEIT/BizOnAI-OCR   # HuggingFace repo or local path
MAX_OUTPUT_TOKENS=12384

# vLLM server
VLLM_API_BASE=http://localhost:8000/v1
VLLM_MODEL_NAME=bizonai-ocr
VLLM_GPUS=0                             # Which GPU(s) to use
```

## KDoc-OCRBench-V2 (Korean Document OCR Benchmark)

Existing OCR benchmarks are primarily English-focused, making it difficult to evaluate model performance on Korean documents. **KDoc-OCRBench-V2** addresses this gap with **56,197 manually-reviewed test cases** across 849 Korean public-sector PDFs.

### Overview

- **56,197 verified test cases** across **849 Korean public-sector PDFs** in 4 document categories
- **3 test types**: text presence/absence, table cell + adjacency validation, header/footer detection
- Much heavier table coverage than v1 — **49,115 table cell tests** (~12×)
- GPT-5.5 silver labels + full human review; only `verified` units are kept
- Plug-compatible with the upstream [olmOCR-bench](https://huggingface.co/datasets/allenai/olmOCR-bench) runner
- Source PDFs are openly shippable (Korean public-data terms) and **included** in the dataset

### Document Categories

| Category | Documents | Description |
|------------|----------:|-------------|
| Statistics | 283 | Numerical data, yearbooks, indicators, time-series tables |
| Reports    | 238 | Research / survey / analysis / evaluation reports |
| Manuals    | 216 | Guidelines, manuals, standards, exam criteria, procedures |
| Notices    | 112 | Public announcements, recruitment, press releases, disclosures |

> See the [KDoc-OCRBench-V2 dataset card](https://huggingface.co/datasets/ONTHEIT/KDoc-OCRBench-V2) for full schemas and examples.

### Running the Benchmark

```bash
# 1. Install olmocr with bench deps
git clone https://github.com/allenai/olmocr.git
cd olmocr && pip install -e .[bench] && cd -

# 2. Download benchmark data (PDFs + JSONL tests) from HuggingFace
huggingface-cli download ONTHEIT/KDoc-OCRBench-V2 --repo-type dataset \
    --local-dir kdoc_benchmark_v2/data

# 3. Run your model (produces raw output)
CUDA_VISIBLE_DEVICES=0 python -m bizonai_ocr.pipeline ./kdoc_benchmark_v2/data/my_model \
    --model ONTHEIT/BizOnAI-OCR \
    --input ./kdoc_benchmark_v2/data/pdfs \
    --benchmark -tp 1

# 4. Post-process to markdown
python -m bizonai_ocr.postprocess ./kdoc_benchmark_v2/data/my_model \
    --format markdown --output ./kdoc_benchmark_v2/data/my_model_md

# 5. Evaluate
python -m olmocr.bench.benchmark --dir ./kdoc_benchmark_v2/data --candidate my_model_md
```

See [kdoc_benchmark_v2/README.md](kdoc_benchmark_v2/README.md) for more details.

## KDoc-OCRBench (Korean Document OCR Benchmark)

Existing OCR benchmarks are primarily English-focused, making it difficult to evaluate model performance on Korean documents. **KDoc-OCRBench** addresses this gap with 14,738 test cases specifically designed for Korean industrial documents.

### Overview

- **14,738 test cases** across **804 Korean PDFs** in 7 document categories
- **3 test types**: text presence/absence, table structure validation, header/footer detection
- Unit-test-style evaluation based on [olmOCR-bench](https://huggingface.co/datasets/allenai/olmOCR-bench) methodology
- Korean documents collected and labeled by [ONTHEIT](http://www.ontheit.com/)
- Bootstrap confidence intervals for statistical rigor

### Document Categories

| Category | Documents | Description |
|----------|-----------|-------------|
| CorporateDocs | 124 | Contracts, reports, meeting minutes |
| EducationalDocs | 158 | Textbooks, certificates, transcripts |
| FinancialInsuranceDocs | 64 | Insurance forms, financial statements |
| LogisticsDocs | 72 | Shipping documents, invoices |
| MedicalDocs | 118 | Medical records, prescriptions |
| PresentationSlides | 99 | Presentation slides |
| PublicDocs | 169 | Government forms, public documents |

> See the [KDoc-OCRBench dataset card](https://huggingface.co/datasets/ONTHEIT/KDoc-OCRBench) for detailed examples of each category.

### Running the Benchmark

```bash
# 1. Install benchmark deps
uv pip install -e ".[bench]"

# 2. Download PDFs and test cases from HuggingFace
huggingface-cli download ONTHEIT/KDoc-OCRBench --repo-type dataset --local-dir kdoc_benchmark/data

# 3. Run your model (produces raw output)
CUDA_VISIBLE_DEVICES=0 python -m bizonai_ocr.pipeline ./kdoc_benchmark/data/my_model \
    --model ONTHEIT/BizOnAI-OCR \
    --input ./kdoc_benchmark/data/pdfs \
    --benchmark -tp 1

# 4. Post-process to markdown
python -m bizonai_ocr.postprocess ./kdoc_benchmark/data/my_model \
    --format markdown --output ./kdoc_benchmark/data/my_model_md

# 5. Evaluate
python -m kdoc_benchmark --dir ./kdoc_benchmark/data --candidate my_model_md \
    --table-normalize-spaces --table-normalize-newlines
```

See [kdoc_benchmark/README.md](kdoc_benchmark/README.md) for more details.

## Credits

- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) by Alibaba
- [olmocr](https://github.com/allenai/olmocr) by Allen AI
- [olmOCR-bench](https://huggingface.co/datasets/allenai/olmOCR-bench) by Allen AI
- [Chandra](https://github.com/datalab-to/chandra) by Datalab
- [vLLM](https://github.com/vllm-project/vllm)

## License

This repository is released under the Apache2.0 and OpenRAIL. See the [LICENSE](LICENSE) file for details.
