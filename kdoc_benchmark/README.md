# KDoc-OCRBench (Korean Document OCR Benchmark)

A comprehensive OCR benchmark for Korean documents with 14,738 test cases across 804 PDFs.

Existing OCR benchmarks are primarily English-focused, making it difficult to evaluate model performance on Korean documents. KDoc-OCRBench addresses this gap with test cases specifically designed for Korean industrial documents.

**Dataset**: [ONTHEIT/KDoc-OCRBench](https://huggingface.co/datasets/ONTHEIT/KDoc-OCRBench) on HuggingFace

## Quick Start

```bash
# Install with benchmark dependencies
uv pip install -e ".[bench]"

# Download benchmark data (PDFs + JSONL test files) from HuggingFace
huggingface-cli download ONTHEIT/KDoc-OCRBench --repo-type dataset --local-dir kdoc_benchmark/data

# Run evaluation on your model's output
python -m kdoc_benchmark --dir ./kdoc_benchmark/data --candidate my_model_md \
    --table-normalize-spaces --table-normalize-newlines
```

## Data

- `data/long_tests.jsonl` - Text presence/absence tests (10,137 tests)
- `data/table_tests.jsonl` - Table structure tests (4,147 tests)
- `data/header_footer_tests.jsonl` - Header/footer tests (454 tests)

**Total: 14,738 tests across 804 Korean documents in 7 categories.**

### Categories

| Category | Documents | Description |
|----------|-----------|-------------|
| CorporateDocs | 124 | Contracts, reports, meeting minutes |
| EducationalDocs | 158 | Textbooks, certificates, transcripts |
| FinancialInsuranceDocs | 64 | Insurance forms, financial statements |
| LogisticsDocs | 72 | Shipping documents, invoices |
| MedicalDocs | 118 | Medical records, prescriptions |
| PresentationSlides | 99 | Presentation slides |
| PublicDocs | 169 | Government forms, public documents |

## How to Evaluate Your Own Model

### 1. Prepare your model outputs

Run your OCR model on the benchmark PDFs and save the results as **Markdown (.md) files**.

Each output file must follow this naming convention:

```
{pdf_filename_without_ext}_pg{page}_repeat{n}.md
```

- `page`: page number (1-indexed)
- `repeat`: generation index (use `repeat1` if you only generate once)

**Example:** For a PDF named `CorporateDocs_1002_pg1.pdf`, the output file should be:
```
CorporateDocs_1002_pg1_pg1_repeat1.md
```

> Note: The PDF filenames already contain `_pg1` (since each PDF is a single page), so `pg1` appears twice in the output filename. This is expected.

### 2. Organize the output directory

```
kdoc_benchmark/data/
├── pdfs/                              # Downloaded from HuggingFace
│   ├── CorporateDocs/*.pdf
│   ├── EducationalDocs/*.pdf
│   └── ...
├── my_model_md/                       # Your model's markdown outputs
│   ├── CorporateDocs/
│   │   ├── CorporateDocs_1002_pg1_pg1_repeat1.md
│   │   └── ...
│   ├── EducationalDocs/
│   │   └── ...
│   └── ...
├── long_tests.jsonl
├── table_tests.jsonl
└── header_footer_tests.jsonl
```

### 3. Run evaluation

```bash
python -m kdoc_benchmark --dir ./kdoc_benchmark/data --candidate my_model_md \
    --table-normalize-spaces --table-normalize-newlines
```

**Evaluation options:**

| Flag | Description |
|------|-------------|
| `--candidate` | Evaluate only this candidate folder |
| `--table-normalize-spaces` | Remove whitespace when comparing table cells (recommended for Korean) |
| `--table-normalize-newlines` | Remove `<br/>` tags in table cells |
| `--skip_baseline` | Skip baseline quality checks |
| `--force` | Run even if some PDF outputs are missing |
| `--save-results` / `--no-save-results` | Save detailed JSON results |

### Multiple repeats (optional)

If your model generates multiple outputs per page (e.g., with different temperatures), the benchmark will evaluate each repeat and use majority voting — a test passes if >50% of repeats pass.

## Test Types

- **Text Presence (`present`)**: Verifies specific Korean text appears in the output, with optional fuzzy matching
- **Text Absence (`absent`)**: Verifies certain text (e.g., headers/footers) is NOT in the output
- **Table (`table`)**: Validates table cell content and spatial relationships (up/down/left/right neighbors, column/row headings)
- **Baseline**: Checks output is non-empty, not repeating, and contains valid characters

## Methodology

KDoc-OCRBench follows the evaluation methodology of [olmOCR-bench](https://huggingface.co/datasets/allenai/olmOCR-bench) by Allen AI. Each test is a unit-test-style assertion (e.g., "this text must appear", "this cell must be to the right of that cell") rather than a full-text comparison, which makes evaluation robust to formatting differences between models.

**What we built on top of olmOCR-bench:**
- **Korean document collection**: 804 Korean PDFs across 7 industrial document categories, collected and curated by [ONTHEIT](http://www.ontheit.com/)
- **Korean test labeling**: 14,738 test cases manually labeled for Korean text, tables, and layout — with Korean-specific normalization (decorative spacing, multi-line cells)
- **Evaluation framework**: Adapted from olmocr bench with Korean table normalization options (`--table-normalize-spaces`, `--table-normalize-newlines`)

The evaluation format is compatible with olmOCR-bench — models that produce outputs for olmOCR-bench can be evaluated on KDoc-OCRBench with the same naming convention.

## Credits

- Evaluation code adapted from [olmocr](https://github.com/allenai/olmocr)
- Evaluation methodology inspired by [olmOCR-bench](https://huggingface.co/datasets/allenai/olmOCR-bench)
- Korean document data collected and labeled by [ONTHEIT](http://www.ontheit.com/)

## License

This repository is released under the Apache2.0 and OpenRAIL. See the [LICENSE](../LICENSE) file for details.
