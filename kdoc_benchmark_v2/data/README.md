# KDoc-OCRBench-V2 data

Place the dataset contents here:

```bash
huggingface-cli download ONTHEIT/KDoc-OCRBench-V2 --repo-type dataset --local-dir .
```

This will populate this directory with:

```
.
├── pdfs/                          # 850 single-page Korean public-sector PDFs
├── text_present.jsonl             # 6,292 PRESENT-type tests
├── tables.jsonl                   # 49,127 TABLE-type tests
├── header_footer_tests.jsonl      # 793 ABSENT-type tests
└── categories.jsonl               # 850 per-PDF category labels
```

Access is **gated** — submit the access request form on the HuggingFace page and you'll be approved manually for research / evaluation use.

See [../README.md](../README.md) for full evaluation instructions.
