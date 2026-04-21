# KDoc-OCRBench-V2 (Korean Document OCR Benchmark, v2)

A second, larger Korean document OCR benchmark — **56,197 verified test cases across 849 single-page PDFs** from Korean public-sector documents. Plug-compatible with [olmOCR-Bench](https://huggingface.co/datasets/allenai/olmOCR-bench).

V2 is a complementary release alongside [v1](../kdoc_benchmark/) — v1 covers 804 industrial PDFs (contracts, medical, financial, logistics, education, etc.), V2 covers 849 public-sector PDFs (statistics, reports, manuals, notices) with much heavier table coverage.

**Dataset**: [ONTHEIT/KDoc-OCRBench-V2](https://huggingface.co/datasets/ONTHEIT/KDoc-OCRBench-V2) on HuggingFace (PDFs included)

## Quick Start

```bash
# Install olmocr with bench deps
git clone https://github.com/allenai/olmocr.git
cd olmocr && pip install -e .[bench] && cd -

# Download benchmark data (PDFs + JSONL tests) from HuggingFace
huggingface-cli download ONTHEIT/KDoc-OCRBench-V2 --repo-type dataset \
    --local-dir kdoc_benchmark_v2/data

# Run evaluation on your model's output
python -m olmocr.bench.benchmark --dir ./kdoc_benchmark_v2/data --candidate my_model_md
```

## Data

- `data/pdfs/` — 849 single-page Korean public-sector PDFs (flat layout, Korean filenames)
- `data/text_present.jsonl` — sentence-level presence tests (6,290)
- `data/tables.jsonl` — table cell + adjacency tests (49,115)
- `data/header_footer_tests.jsonl` — header/footer absence tests (792)
- `data/categories.jsonl` — per-PDF category labels

**Total: 56,197 tests across 849 Korean public-sector documents in 4 categories.**

### Categories

| Category   | Documents | Description |
|------------|----------:|-------------|
| Statistics | 283 | Numerical data, yearbooks, indicators, time-series tables |
| Reports    | 238 | Research / survey / analysis / evaluation reports |
| Manuals    | 216 | Guidelines, manuals, standards, exam criteria, procedures |
| Notices    | 112 | Public announcements, recruitment, press releases, disclosures |

## How to Evaluate Your Own Model

### 1. Prepare your model outputs

Run your OCR model on the benchmark PDFs and save the results as **Markdown (.md) files**, named `{pdf_basename}_pg{page}_repeat{n}.md` (use `repeat1` if you only generate once).

### 2. Organize the output directory

```
kdoc_benchmark_v2/data/
├── pdfs/                              # Downloaded from HuggingFace
├── my_model_md/                       # Your model's markdown outputs
├── text_present.jsonl
├── tables.jsonl
├── header_footer_tests.jsonl
└── categories.jsonl
```

### 3. Run evaluation

```bash
python -m olmocr.bench.benchmark --dir ./kdoc_benchmark_v2/data \
    --candidate my_model_md \
    --bootstrap_samples 1000 --confidence_level 0.95
```

Overall = mean of the four per-JSONL pass rates (Baseline / Header-Footer / Long-Text / Table).

## Test Types

- **Text Presence (`present`)** — verifies specific Korean text appears in the output (fuzzy match)
- **Table (`table`)** — validates table cell content and spatial relationships (up/down/left/right, headings)
- **Header/Footer (`absent`)** — verifies headers/footers are NOT in the output's edge regions
- **Baseline** — checks output is non-empty, not repeating, and contains valid characters

For full JSON schemas and matching semantics, see the [dataset card](https://huggingface.co/datasets/ONTHEIT/KDoc-OCRBench-V2).

## Methodology

KDoc-OCRBench-V2 follows the [olmOCR-bench](https://huggingface.co/datasets/allenai/olmOCR-bench) methodology by Allen AI and is directly plug-compatible with the upstream `olmocr.bench` runner. Each test is a unit-test-style assertion (e.g., "this cell exists and has this neighbor"), which makes evaluation robust to formatting differences between models.

## Credits

- Evaluation methodology and runner: [olmOCR-bench](https://huggingface.co/datasets/allenai/olmOCR-bench) and [olmocr](https://github.com/allenai/olmocr) by Allen AI
- Korean public-sector data collection, silver labelling, and manual review: [ONTHEIT](http://www.ontheit.com/)

## License

This repository is released under the Apache2.0 and OpenRAIL. See the [LICENSE](../LICENSE) file for details.
