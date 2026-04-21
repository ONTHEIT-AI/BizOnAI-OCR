# Based on Chandra (https://github.com/datalab-to/chandra) - Apache 2.0
#!/usr/bin/env python3
"""
BizOnAI-OCR Post-processor - Convert raw model output to markdown or HTML.

Usage:
    # Convert to markdown
    python -m bizonai_ocr.postprocess ./raw_output --format markdown --output ./markdown_output

    # Convert to HTML
    python -m bizonai_ocr.postprocess ./raw_output --format html --output ./html_output
"""

import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from bizonai_ocr.output import parse_html, parse_markdown

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def process_file(
    input_path: Path,
    output_path: Path,
    output_format: str,
    include_headers_footers: bool = False,
) -> tuple[bool, str, bool]:
    """Process a single file. Returns (success, error_msg, is_empty)."""
    try:
        raw_content = input_path.read_text(encoding="utf-8")

        # Empty file -> create empty output (for benchmark mode failures)
        if not raw_content.strip():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("", encoding="utf-8")
            return True, "", True

        if output_format == "markdown":
            result = parse_markdown(raw_content, include_headers_footers=include_headers_footers)
        else:  # html
            result = parse_html(raw_content, include_headers_footers=include_headers_footers)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result, encoding="utf-8")

        return True, "", False
    except Exception as e:
        return False, str(e), False


def collect_files(input_path: Path) -> list[Path]:
    """Collect all .md files from input path."""
    if input_path.is_file():
        return [input_path]
    elif input_path.is_dir():
        return sorted(input_path.rglob("*.md"))
    else:
        return []


def main():
    parser = argparse.ArgumentParser(
        description="BizOnAI-OCR Post-processor - Convert raw output to markdown or HTML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert to markdown
    python -m bizonai_ocr.postprocess ./raw_output --format markdown --output ./markdown_output

    # Convert to HTML
    python -m bizonai_ocr.postprocess ./raw_output --format html --output ./html_output

    # Include headers and footers
    python -m bizonai_ocr.postprocess ./raw_output --format markdown --output ./out --include-headers-footers
        """
    )

    parser.add_argument("input", help="Input file or directory containing raw output files")
    parser.add_argument("--format", "-f", choices=["markdown", "html"], required=True,
                        help="Output format")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--include-headers-footers", action="store_true",
                        help="Include headers and footers in output")
    parser.add_argument("--workers", "-w", type=int, default=8,
                        help="Number of parallel workers (default: 8)")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return 1

    # Collect files
    files = collect_files(input_path)
    if not files:
        logger.error("No .md files found")
        return 1

    logger.info(f"Found {len(files)} files to process")

    output_dir = Path(args.output)
    ext = ".md" if args.format == "markdown" else ".html"

    success_count = 0
    empty_count = 0
    error_count = 0
    errors = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}

        for input_file in files:
            # Preserve directory structure
            try:
                rel_path = input_file.relative_to(input_path)
                output_file = output_dir / rel_path.with_suffix(ext)
            except ValueError:
                output_file = output_dir / input_file.with_suffix(ext).name

            future = executor.submit(
                process_file,
                input_file,
                output_file,
                args.format,
                args.include_headers_footers,
            )
            futures[future] = input_file

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            input_file = futures[future]
            success, error, is_empty = future.result()

            if success:
                success_count += 1
                if is_empty:
                    empty_count += 1
            else:
                error_count += 1
                if error_count <= 10:
                    errors.append(f"{input_file}: {error}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Completed: {success_count} success ({empty_count} empty), {error_count} errors")
    print(f"Output directory: {output_dir}")
    if errors:
        print(f"\nSample errors:")
        for e in errors[:5]:
            print(f"  - {e}")
    print(f"{'='*60}")

    return 0 if error_count == 0 else 1


def cli():
    """CLI entry point."""
    exit(main())


if __name__ == "__main__":
    cli()
