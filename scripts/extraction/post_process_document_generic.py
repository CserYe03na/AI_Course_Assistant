#!/usr/bin/env python3
"""Generic post-process for extracted course document JSON."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course-id", required=True, help="Course identifier, for example adl or eods.")
    parser.add_argument(
        "--input",
        dest="input_path",
        help="Input JSON path. Defaults to data/processed/<course_id>/<course_id>_document.json",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        help="Output JSON path. Defaults to data/processed/<course_id>/<course_id>_processed.json",
    )
    return parser.parse_args()


def clean_text(text: Any) -> str | None:
    if text is None:
        return None
    normalized = str(text).replace("\u00a0", " ").replace("\u200b", "")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized or None


def should_keep_block(block: Dict[str, Any]) -> bool:
    block_type = str(block.get("type") or "").lower()
    text = clean_text(block.get("text"))
    caption = clean_text(block.get("caption"))

    if block_type in {"figure", "formula", "table"}:
        return True
    if block_type in {"text", "title", "code"}:
        return bool(text)
    return bool(text or caption)


def normalize_block(block: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(block)
    normalized["text"] = clean_text(block.get("text"))
    normalized["caption"] = clean_text(block.get("caption"))
    return normalized


def normalize_document(payload: Dict[str, Any]) -> Dict[str, Any]:
    output = dict(payload)
    documents: List[Dict[str, Any]] = []

    for document in payload.get("documents", []):
        if not isinstance(document, dict):
            continue
        normalized_document = dict(document)
        pages: List[Dict[str, Any]] = []

        for page in document.get("pages", []):
            if not isinstance(page, dict):
                continue
            normalized_page = dict(page)
            blocks = []
            for block in page.get("blocks", []):
                if not isinstance(block, dict):
                    continue
                normalized_block = normalize_block(block)
                if should_keep_block(normalized_block):
                    blocks.append(normalized_block)
            normalized_page["blocks"] = blocks
            pages.append(normalized_page)

        normalized_document["pages"] = pages
        documents.append(normalized_document)

    output["documents"] = documents
    return output


def main() -> None:
    args = parse_args()
    base_dir = Path("data/processed") / args.course_id
    input_path = Path(args.input_path) if args.input_path else base_dir / f"{args.course_id}_document.json"
    output_path = Path(args.output_path) if args.output_path else base_dir / f"{args.course_id}_processed.json"

    payload = json.loads(input_path.read_text())
    normalized = normalize_document(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(normalized, indent=2, ensure_ascii=False))
    print(f"Wrote generic processed document JSON to {output_path}")


if __name__ == "__main__":
    main()
