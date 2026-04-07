#!/usr/bin/env python3
"""Post-process extracted course document JSON.

Tasks:
1. Clean formula text strings.
2. Remove text blocks that fall inside a figure on the same page.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process extracted document.json.")
    parser.add_argument(
        "--input-json",
        default="data/processed/document.json",
        help="Input document.json path.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output path. Defaults to overwriting input JSON.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def clean_formula_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # Conservatively merge sequences of single-letter tokens such as
    # "L e a k y" -> "Leaky" without relying on a formula-specific dictionary.
    text = re.sub(
        r"(?<![A-Za-z\\])(?:[A-Za-z]\s+){2,}[A-Za-z](?![A-Za-z])",
        lambda match: "".join(match.group(0).split()),
        text,
    )

    # Structural cleanup only: normalize spacing around LaTeX-style syntax
    # without hard-coding specific formula words.
    text = re.sub(r"\s+([,.;:)\]\}])", r"\1", text)
    text = re.sub(r"([(\[\{])\s+", r"\1", text)
    text = re.sub(r"\s*_\s*", "_", text)
    text = re.sub(r"\s*\^\s*", "^", text)
    text = re.sub(r"\s*\\\s+([A-Za-z]+)", r" \\\1", text)
    text = re.sub(r"\{\s+", "{", text)
    text = re.sub(r"\s+\}", "}", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\[\s+", "[", text)
    text = re.sub(r"\s+\]", "]", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def classify_formula_quality(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "truncated"

    brace_diff = abs(stripped.count("{") - stripped.count("}"))
    paren_diff = abs(stripped.count("(") - stripped.count(")"))

    if brace_diff >= 2 or paren_diff >= 2:
        return "truncated"

    if stripped.endswith(("=", "+", "-", "\\", "{", "(")):
        return "truncated"

    if stripped.startswith(("0)", "1)", "2)", "3)", "\\\\", "&")):
        return "truncated"

    if "\\text {" in stripped and "\\\\" in stripped and len(stripped) > 120:
        return "truncated"

    if re.search(r"(?:[A-Za-z]\s+){3,}[A-Za-z](?![A-Za-z])", stripped):
        return "noisy"

    if "\\ " in stripped or re.search(r"\\\\\s+[A-Za-z]", stripped):
        return "noisy"

    if re.search(r"[A-Za-z]+\\max", stripped):
        return "noisy"

    if re.search(r"\\[A-Za-z]+\s+[A-Za-z]+", stripped):
        return "noisy"

    if "\\mathring" in stripped or "\\intercal" in stripped:
        return "noisy"

    if brace_diff == 1 or paren_diff == 1:
        return "noisy"

    return "good"


def bbox_inside(inner: List[float], outer: List[float], tolerance: float = 2.0) -> bool:
    return (
        inner[0] >= outer[0] - tolerance
        and inner[1] >= outer[1] - tolerance
        and inner[2] <= outer[2] + tolerance
        and inner[3] <= outer[3] + tolerance
    )


def bbox_intersects(a: List[float], b: List[float], tolerance: float = 0.0) -> bool:
    return not (
        a[2] < b[0] - tolerance
        or a[0] > b[2] + tolerance
        or a[3] < b[1] - tolerance
        or a[1] > b[3] + tolerance
    )


def bbox_expand(bbox: List[float], margin: float) -> List[float]:
    return [
        bbox[0] - margin,
        bbox[1] - margin,
        bbox[2] + margin,
        bbox[3] + margin,
    ]


def bbox_area(bbox: List[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def is_large_figure(page: Dict[str, Any], block: Dict[str, Any]) -> bool:
    page_area = max(1.0, float(page.get("width", 0.0)) * float(page.get("height", 0.0)))
    figure_area = bbox_area(block.get("bbox") or [0.0, 0.0, 0.0, 0.0])
    return figure_area / page_area >= 0.18


def is_long_body_text(block: Dict[str, Any]) -> bool:
    if block.get("type") != "text":
        return False
    text = (block.get("text") or "").strip()
    return len(text) >= 90 or len(text.split()) >= 12


def is_short_fragment(block: Dict[str, Any]) -> bool:
    if block.get("type") not in {"text", "title"}:
        return False

    text = (block.get("text") or "").strip()
    if not text:
        return True

    return len(text) <= 24 or len(text.split()) <= 4


def should_remove_figure_fragment(
    block: Dict[str, Any],
    figure_blocks: List[Dict[str, Any]],
    has_large_figure_without_body: bool,
) -> bool:
    if block.get("type") not in {"text", "title"}:
        return False

    bbox = block.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    for figure in figure_blocks:
        figure_bbox = figure.get("bbox") or [0.0, 0.0, 0.0, 0.0]
        if bbox_inside(bbox, figure_bbox):
            return True

        if has_large_figure_without_body and is_short_fragment(block):
            expanded_bbox = bbox_expand(figure_bbox, margin=24.0)
            if bbox_intersects(bbox, expanded_bbox):
                return True

    return False


def post_process_page(page: Dict[str, Any]) -> None:
    blocks = page.get("blocks", [])
    figure_blocks = [block for block in blocks if block.get("type") == "figure"]
    has_large_figure_without_body = any(is_large_figure(page, figure) for figure in figure_blocks) and not any(
        is_long_body_text(block) for block in blocks
    )

    processed_blocks: List[Dict[str, Any]] = []
    for block in blocks:
        block_type = block.get("type")

        if block_type == "formula" and block.get("text"):
            block["text"] = clean_formula_text(block["text"])
            block["formula_quality"] = classify_formula_quality(block["text"])
        elif block_type == "formula":
            block["formula_quality"] = "truncated"

        if figure_blocks and should_remove_figure_fragment(
            block,
            figure_blocks=figure_blocks,
            has_large_figure_without_body=has_large_figure_without_body,
        ):
            continue

        processed_blocks.append(block)

    for reading_order, block in enumerate(processed_blocks, start=1):
        block["reading_order"] = reading_order

    page["blocks"] = processed_blocks


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_json)
    output_path = Path(args.output_json) if args.output_json else input_path

    payload = load_json(input_path)
    for document in payload.get("documents", []):
        for page in document.get("pages", []):
            post_process_page(page)

    save_json(output_path, payload)
    print(f"Wrote post-processed document JSON to {output_path}")


if __name__ == "__main__":
    main()
